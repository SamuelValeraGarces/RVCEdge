"""
ContentVec feature extraction via transformers.
Uses lengyue233/content-vec-best (94M params, speaker-independent 768-dim).
ContentVec was fine-tuned from HuBERT to remove speaker identity — required for RVC v2.
Feature extractor created manually (no preprocessor_config.json needed).
No fairseq required.
"""

import torch
import numpy as np
from transformers import HubertModel, Wav2Vec2FeatureExtractor


_MODEL_CACHE = {}

# Standard wav2vec2/HuBERT feature extractor settings
_FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False,
)


def get_content_vec_model(device: str = "cuda"):
    """Load ContentVec model (cached after first call)."""
    key = f"contentvec_{device}"
    if key not in _MODEL_CACHE:
        model_id = "lengyue233/content-vec-best"
        print(f"Loading ContentVec ({model_id}, ~360MB)...")
        model = HubertModel.from_pretrained(model_id)
        model.eval().to(device)
        _MODEL_CACHE[key] = model
        print("ContentVec loaded.")
    return _MODEL_CACHE[key]


@torch.inference_mode()
def extract_features(wav_16k: np.ndarray, version: str = "v2",
                     device: str = "cuda") -> torch.Tensor:
    """
    Extract speaker-independent content features at ~50fps (16kHz / 320 hop).
    Returns [1, T, 768] — caller should repeat_interleave(2) for RVC 100fps expectation.
    """
    model = get_content_vec_model(device)

    inputs = _FEATURE_EXTRACTOR(
        wav_16k,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    outputs = model(input_values)
    feats = outputs.last_hidden_state  # [1, T, 768]

    if version == "v1":
        feats = feats[:, :, :256]

    return feats


def preprocess_wav(wav: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav = wav.astype(np.float32)
    if np.abs(wav).max() > 1.0:
        wav = wav / np.abs(wav).max()
    return wav
