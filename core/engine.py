"""
RVCEdge — main inference engine.
Real-time RVC pipeline: ContentVec → pyin F0 → VITS → audio.
"""

import time
import threading
import contextlib
import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional

from .synthesizer import load_rvc_model
from .hubert import extract_features


DEVICE_SR = 16000
CHUNK_MS = 200
OVERLAP_MS = 30
FMIN = 32.70
FMAX = 1975.53


def extract_f0_pyin(audio_16k: np.ndarray, pitch_shift: float = 0.0,
                    hop_length: int = 160) -> tuple:
    """
    Extract F0 with librosa pyin at 100fps (hop=160 at 16kHz).
    Returns (f0_hz, f0_coarse) aligned to feature frame rate.
    """
    f0, voiced_flag, _ = librosa.pyin(
        audio_16k.astype(np.float32),
        fmin=FMIN,
        fmax=FMAX,
        sr=DEVICE_SR,
        hop_length=hop_length,
        frame_length=1024,
        fill_na=0.0,
        center=True,
    )
    f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
    f0[~voiced_flag] = 0.0

    if pitch_shift != 0.0:
        voiced = f0 > 0
        f0[voiced] *= 2 ** (pitch_shift / 12)
        f0[voiced] = np.clip(f0[voiced], FMIN, FMAX)

    f0_coarse = f0.copy()
    voiced_mask = f0_coarse > 0
    f0_coarse[voiced_mask] = (
        np.log2(f0_coarse[voiced_mask] / FMIN) / np.log2(FMAX / FMIN) * 254 + 1
    )
    f0_coarse = np.clip(f0_coarse, 1, 255).astype(np.int64)
    f0_coarse[~voiced_mask] = 0

    return f0, f0_coarse


class RVCEdgeEngine:
    """
    Real-time RVC inference engine.
    - No fairseq: ContentVec via transformers
    - pyin F0 extraction (librosa) — robust, no separate model needed
    - WASAPI-ready process_chunk() API
    - Auto-detects .pth and .safetensors model format
    """

    def __init__(self,
                 device: str = "auto",
                 pitch_shift: float = 0.0,
                 index_rate: float = 0.75,
                 protect: float = 0.33,
                 sid: int = 0):

        self.device_str = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.device = torch.device(self.device_str)
        self.pitch_shift = pitch_shift
        self.index_rate = index_rate
        self.protect = protect
        self.sid = sid

        self.net_g = None
        self.model_sr = None
        self.model_version = None
        self.model_f0 = True
        self.index = None
        self.big_npy = None

        self.is_ready = False
        self.mutex = threading.Lock()
        self.use_autocast = self.device.type == "cuda"

        self._overlap_samples = int(DEVICE_SR * OVERLAP_MS / 1000)
        self._last_audio = None

        self.inference_time_ms = 0.0

    def load_model(self, pth_path: str, index_path: str = None):
        """Load RVC .pth or .safetensors model and optional .index file."""
        print(f"Loading RVC model: {Path(pth_path).name}")
        self.net_g, self.model_sr, self.model_version, self.model_f0 = load_rvc_model(
            pth_path, device=self.device_str
        )
        print(f"  Version: {self.model_version} | SR: {self.model_sr}Hz | F0: {bool(self.model_f0)}")

        if index_path and Path(index_path).exists():
            try:
                import faiss
                self.index = faiss.read_index(index_path)
                npy_path = index_path.replace(".index", "_npy.npy")
                self.big_npy = np.load(npy_path) if Path(npy_path).exists() else None
                print(f"  FAISS index: {self.index.ntotal} vectors")
            except Exception as e:
                print(f"  FAISS index failed ({e}), continuing without.")
                self.index = None

        self.is_ready = True
        print(f"Model ready. Device: {self.device_str.upper()}")

    @torch.inference_mode()
    def convert_audio(self, audio_16k: np.ndarray) -> np.ndarray:
        """
        Convert audio clip (16kHz float32) to target voice.
        Returns audio at model SR (40k/48k).
        """
        if not self.is_ready:
            return audio_16k

        t0 = time.perf_counter()

        # 1. ContentVec features — 50fps, doubled to 100fps for VITS
        with (torch.autocast(device_type=self.device.type) if self.use_autocast
              else contextlib.nullcontext()):
            feats = extract_features(audio_16k, version=self.model_version,
                                     device=self.device_str)  # [1, T, 768]
        feats = feats.repeat_interleave(2, dim=1)  # 50fps → 100fps

        n_frames = feats.shape[1]
        feats_np = feats.squeeze(0).float().cpu().numpy()

        # 2. pyin F0 extraction at 100fps (hop=160 at 16kHz)
        f0_hz, f0_coarse = None, None
        if self.model_f0:
            f0_hz, f0_coarse = extract_f0_pyin(audio_16k, pitch_shift=self.pitch_shift,
                                                 hop_length=160)
            # Align to feature length
            if len(f0_hz) != n_frames:
                f0_hz = np.interp(np.linspace(0, len(f0_hz)-1, n_frames),
                                   np.arange(len(f0_hz)), f0_hz).astype(np.float32)
                f0_coarse = np.round(np.interp(
                    np.linspace(0, len(f0_coarse)-1, n_frames),
                    np.arange(len(f0_coarse)), f0_coarse.astype(np.float32)
                )).astype(np.int64)

        # 3. FAISS index retrieval
        if self.index is not None and self.big_npy is not None and self.index_rate > 0:
            try:
                import faiss
                score, ix = self.index.search(feats_np.astype("float32"), k=8)
                weight = np.square(1 / (score + 1e-6))
                weight /= weight.sum(axis=1, keepdims=True)
                retrieved = np.einsum("nji,nj->ni", self.big_npy[ix], weight)
                feats_np = (1 - self.index_rate) * feats_np + self.index_rate * retrieved
            except Exception:
                pass

        # 4. VITS inference
        feats_t = torch.from_numpy(feats_np).unsqueeze(0).float().to(self.device)
        phone_lengths = torch.LongTensor([n_frames]).to(self.device)
        sid_t = torch.LongTensor([self.sid]).to(self.device)

        if self.model_f0 and f0_hz is not None:
            pitch_t = torch.from_numpy(f0_hz[:n_frames]).float().unsqueeze(0).to(self.device)
            pitch_c_t = torch.from_numpy(f0_coarse[:n_frames]).long().unsqueeze(0).to(self.device)
        else:
            pitch_t = torch.zeros(1, n_frames).to(self.device)
            pitch_c_t = torch.zeros(1, n_frames, dtype=torch.long).to(self.device)

        with (torch.autocast(device_type=self.device.type) if self.use_autocast
              else contextlib.nullcontext()):
            audio_out, _ = self.net_g.infer(feats_t, phone_lengths, pitch_c_t, pitch_t, sid_t)

        audio_out = audio_out.squeeze().float().cpu().numpy()
        self.inference_time_ms = (time.perf_counter() - t0) * 1000
        return audio_out  # at model_sr

    def process_chunk(self, samples_16k: np.ndarray) -> Optional[np.ndarray]:
        """
        Real-time chunk. Input: float32 at 16kHz. Output: float32 at 16kHz.
        """
        if not self.is_ready:
            return samples_16k

        audio_model_sr = self.convert_audio(samples_16k)

        if self.model_sr != DEVICE_SR:
            audio_out = librosa.resample(audio_model_sr,
                                          orig_sr=self.model_sr, target_sr=DEVICE_SR)
        else:
            audio_out = audio_model_sr

        # Clip to safe range
        audio_out = np.clip(audio_out, -1.0, 1.0).astype(np.float32)

        # Crossfade
        if self._last_audio is not None:
            ov = min(self._overlap_samples, len(self._last_audio), len(audio_out))
            if ov > 0:
                fade_in = np.linspace(0, 1, ov, dtype=np.float32)
                fade_out = np.linspace(1, 0, ov, dtype=np.float32)
                audio_out[:ov] = audio_out[:ov] * fade_in + self._last_audio[-ov:] * fade_out
        self._last_audio = audio_out[-self._overlap_samples:].copy()

        return audio_out

    def convert_file(self, input_path: str, output_path: str, chunk_s: float = 3.0) -> str:
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
            if self.model_sr != DEVICE_SR:
                result = librosa.resample(result, orig_sr=self.model_sr, target_sr=DEVICE_SR)
            out_chunks.append(result)

        output = np.concatenate(out_chunks)
        sf.write(output_path, output, DEVICE_SR)
        return output_path

    def warmup(self, n: int = 3):
        print(f"Warming up ({n} iterations)...")
        dummy = np.zeros(int(DEVICE_SR * CHUNK_MS / 1000), dtype=np.float32)
        # Warmup ContentVec and VITS
        for _ in range(n):
            try:
                self.process_chunk(dummy)
            except Exception:
                pass
        self._last_audio = None
        print("Warmup complete.")

    def list_models(self, models_dir: str = "models") -> list:
        p = Path(models_dir)
        if not p.exists():
            return []
        return sorted(list(p.glob("*.pth")) + list(p.glob("*.safetensors")))

    def list_indexes(self, models_dir: str = "models") -> list:
        p = Path(models_dir)
        return sorted(p.glob("*.index")) if p.exists() else []
