"""
RVCEdge — main inference engine.
Real-time RVC pipeline: ContentVec → RMVPE → VITS → audio.
"""

import time
import threading
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional, Tuple

from .synthesizer import load_rvc_model
from .hubert import extract_features, preprocess_wav
from .rmvpe import RMVPE


DEVICE_SR = 16000   # feature extraction SR
CHUNK_MS = 200      # 200ms processing chunks
OVERLAP_MS = 30     # crossfade overlap


class RVCEdgeEngine:
    """
    Real-time RVC inference engine.

    Improvements over stock RVC:
    - No fairseq — uses transformers ContentVec
    - WASAPI-ready chunk API (process_chunk)
    - Auto-detects model version (v1/v2) and sample rate
    - torch.autocast for faster GPU inference
    - Configurable pitch shift, index rate, protect
    """

    def __init__(self,
                 device: str = "auto",
                 pitch_shift: float = 0.0,
                 index_rate: float = 0.75,
                 protect: float = 0.33,
                 rmvpe_path: str = "base_models/rmvpe.pt",
                 sid: int = 0):

        self.device_str = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.device = torch.device(self.device_str)
        self.pitch_shift = pitch_shift
        self.index_rate = index_rate
        self.protect = protect
        self.sid = sid
        self.rmvpe_path = rmvpe_path

        self.net_g = None
        self.model_sr = None
        self.model_version = None
        self.model_f0 = True
        self.rmvpe: Optional[RMVPE] = None
        self.index = None   # FAISS index (optional)
        self.big_npy = None # index embeddings

        self.is_ready = False
        self.mutex = threading.Lock()
        self.use_autocast = self.device.type == "cuda"

        self._overlap_samples = 0
        self._last_audio = None
        self._warmup_done = False

        self.inference_time_ms = 0.0

    def load_model(self, pth_path: str, index_path: str = None):
        """Load RVC .pth model and optional .index file."""
        print(f"Loading RVC model: {Path(pth_path).name}")
        self.net_g, self.model_sr, self.model_version, self.model_f0 = load_rvc_model(
            pth_path, device=self.device_str
        )
        print(f"  Version: {self.model_version} | SR: {self.model_sr}Hz | F0: {bool(self.model_f0)}")

        # Load RMVPE
        if self.model_f0:
            rmvpe_p = Path(self.rmvpe_path)
            if not rmvpe_p.exists():
                print(f"  RMVPE not found at {rmvpe_p}. Run download_base.py first.")
            else:
                self.rmvpe = RMVPE(str(rmvpe_p), device=self.device_str, is_half=False)
                print("  RMVPE loaded.")

        # Load FAISS index
        if index_path and Path(index_path).exists():
            try:
                import faiss
                self.index = faiss.read_index(index_path)
                if self.index.ntotal > 0:
                    # Read stored embeddings for retrieval
                    # Applio stores them in the same dir with _npy suffix
                    npy_path = index_path.replace(".index", "_npy.npy")
                    if Path(npy_path).exists():
                        self.big_npy = np.load(npy_path)
                    else:
                        self.big_npy = None
                    print(f"  FAISS index loaded: {self.index.ntotal} vectors")
            except Exception as e:
                print(f"  FAISS index load failed: {e}. Continuing without index.")
                self.index = None

        self._overlap_samples = int(self.model_sr * OVERLAP_MS / 1000)
        self.is_ready = True
        print(f"Model ready. Device: {self.device_str.upper()}")

    @torch.inference_mode()
    def convert_audio(self, audio_16k: np.ndarray) -> np.ndarray:
        """
        Convert a full audio clip (16kHz) to the target voice.
        Returns audio at model SR (40k/48k).
        """
        if not self.is_ready:
            return audio_16k

        t0 = time.perf_counter()

        # 1. ContentVec features
        with (torch.autocast(device_type=self.device.type) if self.use_autocast
              else __import__("contextlib").nullcontext()):
            feats = extract_features(audio_16k, version=self.model_version,
                                     device=self.device_str)  # [1, T, 768]

        # 2. Upsample features to match model SR
        # Feature rate: ~50fps (16kHz / 320 hop)
        # Target rate: model_sr / hop_size (varies)
        n_frames = feats.shape[1]
        feats_np = feats.squeeze(0).float().cpu().numpy()

        # 3. RMVPE pitch extraction + shift
        f0_hz, f0_coarse = None, None
        if self.model_f0 and self.rmvpe is not None:
            f0_hz, f0_coarse = self.rmvpe.infer_from_audio_with_pitch_shift(
                audio_16k, semitones=self.pitch_shift
            )
            # Align f0 length to features
            if len(f0_hz) != n_frames:
                f0_hz = np.interp(
                    np.linspace(0, len(f0_hz) - 1, n_frames),
                    np.arange(len(f0_hz)), f0_hz
                )
                f0_coarse = np.interp(
                    np.linspace(0, len(f0_coarse) - 1, n_frames),
                    np.arange(len(f0_coarse)), f0_coarse
                ).astype(int)

        # 4. FAISS index retrieval (improve speaker similarity)
        if self.index is not None and self.big_npy is not None and self.index_rate > 0:
            try:
                import faiss
                score, ix = self.index.search(feats_np.astype("float32"), k=8)
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                retrieved = np.einsum("nji,nj->ni", self.big_npy[ix], weight)
                feats_np = (1 - self.index_rate) * feats_np + self.index_rate * retrieved
            except Exception:
                pass

        # 5. VITS inference
        feats_t = torch.from_numpy(feats_np).unsqueeze(0).float().to(self.device)
        phone_lengths = torch.LongTensor([n_frames]).to(self.device)
        sid_t = torch.LongTensor([self.sid]).to(self.device)

        if self.model_f0 and f0_hz is not None:
            pitch_t = torch.from_numpy(f0_hz).float().unsqueeze(0).to(self.device)
            pitch_coarse_t = torch.from_numpy(f0_coarse).long().unsqueeze(0).to(self.device)
        else:
            pitch_t = torch.zeros(1, n_frames).to(self.device)
            pitch_coarse_t = torch.zeros(1, n_frames, dtype=torch.long).to(self.device)

        with (torch.autocast(device_type=self.device.type) if self.use_autocast
              else __import__("contextlib").nullcontext()):
            audio_out, _ = self.net_g.infer(
                feats_t, phone_lengths, pitch_coarse_t, pitch_t, sid_t
            )

        audio_out = audio_out.squeeze().float().cpu().numpy()
        self.inference_time_ms = (time.perf_counter() - t0) * 1000
        return audio_out  # at model_sr

    def process_chunk(self, samples_16k: np.ndarray) -> Optional[np.ndarray]:
        """
        Real-time chunk processing.
        Input: float32 numpy at 16kHz
        Output: float32 numpy at 16kHz (resampled from model SR)
        """
        if not self.is_ready:
            return samples_16k

        audio_out_model_sr = self.convert_audio(samples_16k)

        # Resample back to 16kHz for audio output
        if self.model_sr != DEVICE_SR:
            audio_out = librosa.resample(audio_out_model_sr,
                                          orig_sr=self.model_sr,
                                          target_sr=DEVICE_SR)
        else:
            audio_out = audio_out_model_sr

        # Crossfade with previous chunk
        if self._last_audio is not None and len(self._last_audio) > 0:
            ov = min(self._overlap_samples, len(self._last_audio), len(audio_out))
            if ov > 0:
                fade_in = np.linspace(0, 1, ov)
                fade_out = np.linspace(1, 0, ov)
                audio_out[:ov] = audio_out[:ov] * fade_in + self._last_audio[-ov:] * fade_out
        self._last_audio = audio_out.copy()

        return audio_out.astype(np.float32)

    def convert_file(self, input_path: str, output_path: str,
                     chunk_s: float = 5.0) -> str:
        """Convert an audio file. Returns output path."""
        wav, sr = sf.read(input_path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != DEVICE_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=DEVICE_SR)

        chunk_n = int(DEVICE_SR * chunk_s)
        out_chunks = []
        for i in range(0, len(wav), chunk_n):
            chunk = wav[i: i + chunk_n]
            result = self.convert_audio(chunk)
            # Resample to 16k for output
            if self.model_sr != DEVICE_SR:
                result = librosa.resample(result, orig_sr=self.model_sr, target_sr=DEVICE_SR)
            out_chunks.append(result)

        output = np.concatenate(out_chunks)
        sf.write(output_path, output, DEVICE_SR)
        return output_path

    def warmup(self, n: int = 3):
        print(f"Warming up ({n} iterations)...")
        dummy = np.zeros(int(DEVICE_SR * CHUNK_MS / 1000), dtype=np.float32)
        for _ in range(n):
            self.process_chunk(dummy)
        self._last_audio = None
        print("Warmup complete.")

    def list_models(self, models_dir: str = "models") -> list:
        """Return list of .pth files in models directory."""
        p = Path(models_dir)
        return [f for f in p.glob("*.pth")] if p.exists() else []

    def list_indexes(self, models_dir: str = "models") -> list:
        p = Path(models_dir)
        return [f for f in p.glob("*.index")] if p.exists() else []
