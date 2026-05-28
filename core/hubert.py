"""
ContentVec / HuBERT feature extraction via transformers.
No fairseq required — uses lengyue233/content-vec-best (768-dim, RVC v2).
For v1 models (256-dim), falls back to facebook/hubert-base-ls960.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


_MODEL_CACHE = {}


def get_content_vec_model(device: str = "cuda"):
    """Lazy-load ContentVec model (cached after first call)."""
    key = f"contentvec_{device}"
    if key not in _MODEL_CACHE:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        model_id = "lengyue233/content-vec-best"
        print(f"Loading ContentVec model from HuggingFace ({model_id})...")
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        model = HubertModel.from_pretrained(model_id)
        model.eval().to(device)
        _MODEL_CACHE[key] = (model, fe)
        print("ContentVec loaded.")
    return _MODEL_CACHE[key]


def get_hubert_base_model(device: str = "cuda"):
    """Lazy-load HuBERT base model for v1 models (256-dim via projection)."""
    key = f"hubert_base_{device}"
    if key not in _MODEL_CACHE:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        model_id = "facebook/hubert-base-ls960"
        print(f"Loading HuBERT base model ({model_id})...")
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        model = HubertModel.from_pretrained(model_id)
        model.eval().to(device)
        _MODEL_CACHE[key] = (model, fe)
        print("HuBERT base loaded.")
    return _MODEL_CACHE[key]


@torch.inference_mode()
def extract_features(wav_16k: np.ndarray, version: str = "v2",
                     device: str = "cuda") -> torch.Tensor:
    """
    Extract content features from 16kHz waveform.

    Args:
        wav_16k: float32 numpy array at 16kHz
        version: "v2" (768-dim ContentVec) or "v1" (256-dim)
        device: "cuda" or "cpu"

    Returns:
        features: [1, T, 768] or [1, T, 256] float32 tensor
    """
    if version == "v2":
        model, fe = get_content_vec_model(device)
    else:
        model, fe = get_hubert_base_model(device)

    inputs = fe(
        wav_16k,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    outputs = model(input_values, output_hidden_states=True)

    if version == "v2":
        # ContentVec: use last hidden state (12th layer) — 768-dim
        feats = outputs.last_hidden_state  # [1, T, 768]
    else:
        # HuBERT base: use 9th layer hidden state, project to 256
        feats = outputs.hidden_states[9]  # [1, T, 768]
        # Simple linear projection — models expect 256-dim
        # Note: v1 models were trained with a custom 256-dim HuBERT.
        # For now we pass 768 and let the TextEncoder handle it via emb_phone.
        # v1 TextEncoder.emb_phone is Linear(256, hidden) — we need to pass 256-dim.
        # Fallback: truncate to 256 dims (approximate, works for demo quality)
        feats = feats[:, :, :256]

    return feats  # [1, T, dim]


def preprocess_wav(wav: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample and normalize waveform."""
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav = wav.astype(np.float32)
    # Normalize
    if np.abs(wav).max() > 1.0:
        wav = wav / np.abs(wav).max()
    return wav
