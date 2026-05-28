"""
HuBERT feature extraction via transformers.
Uses facebook/hubert-base-ls960 (768-dim, proper transformers format).
ContentVec is based on this model — RVC v2 quality is comparable.
No fairseq required. Feature extractor created without extra download.
"""

import torch
import numpy as np


_MODEL_CACHE = {}


def get_content_vec_model(device: str = "cuda"):
    """Load HuBERT base model (cached after first call)."""
    key = f"hubert_{device}"
    if key not in _MODEL_CACHE:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        model_id = "facebook/hubert-base-ls960"
        print(f"Loading HuBERT model ({model_id})...")

        # Create feature extractor programmatically — standard HuBERT settings,
        # no preprocessor_config.json needed
        fe = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )
        model = HubertModel.from_pretrained(model_id)
        model.eval().to(device)
        _MODEL_CACHE[key] = (model, fe)
        print("HuBERT loaded.")
    return _MODEL_CACHE[key]


@torch.inference_mode()
def extract_features(wav_16k: np.ndarray, version: str = "v2",
                     device: str = "cuda") -> torch.Tensor:
    """
    Extract 768-dim content features from 16kHz waveform.

    Args:
        wav_16k: float32 numpy array at 16kHz
        version: "v2" (768-dim) or "v1" (256-dim truncated)
        device: "cuda" or "cpu"

    Returns:
        features: [1, T, 768] tensor
    """
    model, fe = get_content_vec_model(device)

    inputs = fe(
        wav_16k,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs.input_values.to(device)

    outputs = model(input_values, output_hidden_states=True)
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
