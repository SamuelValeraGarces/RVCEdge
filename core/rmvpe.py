"""
RMVPE — Robust Melody-based Vocal Pitch Estimator.
Architecture reverse-engineered from lj1995/VoiceConversionWebUI rmvpe.pt checkpoint.
Pure PyTorch, no fairseq required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


N_MELS = 128
N_CLASS = 360
SAMPLE_RATE = 16000
FMIN = 32.70
FMAX = 1975.53


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    import librosa
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)


# ── Building blocks ────────────────────────────────────────────────────────────

class ConvBlockRes(nn.Module):
    """Residual 2-layer conv block with BN and ReLU."""
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


# ── U-Net components (architecture matches rmvpe.pt checkpoint exactly) ────────

class EncoderLayer(nn.Module):
    """Single encoder stage: n_blocks ConvBlockRes."""
    def __init__(self, in_ch, out_ch, n_blocks, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_ch, out_ch, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_ch, out_ch, momentum))

    def forward(self, x):
        for c in self.conv:
            x = c(x)
        return x


class Encoder(nn.Module):
    """U-Net encoder with BN + n_encoders stages, max-pool between stages."""
    def __init__(self, in_channels, n_encoders, kernel_size, n_filters,
                 n_blocks, momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        ch = in_channels
        for _ in range(n_encoders):
            self.layers.append(EncoderLayer(ch, n_filters, n_blocks, momentum))
            ch = n_filters
            n_filters *= 2
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        x = self.bn(x)
        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips


class IntermediateLayer(nn.Module):
    """Bottleneck stage between encoder and decoder."""
    def __init__(self, in_ch, out_ch, n_blocks, momentum=0.01):
        super().__init__()
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_ch, out_ch, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_ch, out_ch, momentum))

    def forward(self, x):
        for c in self.conv:
            x = c(x)
        return x


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels
        for _ in range(n_inters):
            self.layers.append(IntermediateLayer(ch, out_channels, n_blocks, momentum))
            ch = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder stage.
    conv1: ConvTranspose2d(in_ch → out_ch) — halves channels.
    cat with skip (out_ch) → 2*out_ch.
    conv2: n_blocks ConvBlockRes(2*out_ch → out_ch).
    """
    def __init__(self, in_ch, out_ch, n_blocks, momentum=0.01):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=momentum),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(2 * out_ch, out_ch, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_ch, out_ch, momentum))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = self.conv1(x)          # in_ch → out_ch
        x = torch.cat([x, skip], dim=1)  # out_ch + out_ch = 2*out_ch
        for c in self.conv2:
            x = c(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        ch = in_channels
        for _ in range(n_decoders):
            out_ch = ch // 2
            self.layers.append(DecoderLayer(ch, out_ch, n_blocks, momentum))
            ch = out_ch

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            skip = skips[-(i + 1)]
            x = layer(x, skip)
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4,
                 in_channels=1, en_out_channels=16):
        super().__init__()
        # Encoder: channels double each stage: 1→16→32→64→128→256
        self.encoder = Encoder(
            in_channels, en_de_layers, kernel_size,
            en_out_channels, n_blocks,
        )
        # Bottleneck at 256ch → 512ch (first inter layer expands)
        inter_in = en_out_channels * (2 ** (en_de_layers - 1))   # 256
        inter_out = inter_in * 2                                   # 512
        self.intermediate = Intermediate(
            inter_in, inter_out, inter_layers, n_blocks,
        )
        # Decoder: 512→256→128→64→32→16
        self.decoder = Decoder(inter_out, en_de_layers, n_blocks)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, skips)
        return x


class E2E(nn.Module):
    """Full RMVPE model: mel → F0 salience."""
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5,
                 inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers,
                              in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, 3, padding=1)
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * N_MELS, 256, n_gru),
                nn.Linear(512, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * N_MELS, N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel):
        # mel: [B, T, n_mels] → [B, 1, n_mels, T]
        x = mel.transpose(-1, -2).unsqueeze(1)
        x = self.unet(x)           # [B, en_out_ch, n_mels, T]
        x = self.cnn(x)            # [B, 3, n_mels, T]
        x = x.permute(0, 3, 1, 2).flatten(2)  # [B, T, 3*n_mels]
        x = self.fc(x)             # [B, T, N_CLASS]
        return x


# ── Mel spectrogram ────────────────────────────────────────────────────────────

class MelSpectrogram(nn.Module):
    def __init__(self, n_mel_channels=N_MELS, sampling_rate=SAMPLE_RATE,
                 win_length=1024, hop_length=160, n_fft=None,
                 mel_fmin=30, mel_fmax=8000):
        super().__init__()
        self.hop_length = hop_length
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = mel_filterbank(sampling_rate, n_fft, n_mel_channels,
                                    mel_fmin, mel_fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("hann_window", torch.hann_window(win_length))
        self.win_length = win_length
        self.n_fft = n_fft

    def forward(self, audio):
        audio = F.pad(audio, ((self.n_fft - self.hop_length) // 2,
                               (self.n_fft - self.hop_length) // 2))
        spec = torch.stft(audio, self.n_fft, hop_length=self.hop_length,
                          win_length=self.win_length, window=self.hann_window,
                          center=False, return_complex=True)
        spec = torch.abs(spec)
        mel = torch.matmul(self.mel_basis, spec)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel.transpose(-1, -2)  # [B, T, n_mels]


# ── RMVPE inference wrapper ────────────────────────────────────────────────────

class RMVPE:
    def __init__(self, model_path: str, device: str = "cuda", is_half: bool = False):
        self.device = device
        self.is_half = is_half
        self.mel = MelSpectrogram().to(device)

        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt)
        model.eval()
        if is_half:
            model = model.half()
        self.model = model.to(device)

        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, [0, 1])

    @torch.inference_mode()
    def mel2hidden(self, mel):
        n_frames = mel.shape[1]
        n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad > 0:
            mel = F.pad(mel, (0, 0, 0, n_pad))
        if self.is_half:
            mel = mel.half()
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self._to_local_average_cents(hidden, thred=thred)
        f0 = 10 * 2 ** (cents_pred / 1200)
        f0[f0 == 10] = 0
        return f0

    def _to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=-1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[idx][starts[idx]:ends[idx]])
            todo_cents_mapping.append(self.cents_mapping[starts[idx]:ends[idx]])
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)
        devided = product_sum / (weight_sum + 1e-8)
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided

    @torch.inference_mode()
    def infer_from_audio(self, audio_16k: np.ndarray, thred=0.03) -> np.ndarray:
        audio_t = torch.from_numpy(audio_16k).float().unsqueeze(0).to(self.device)
        mel = self.mel(audio_t)
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()
        return self.decode(hidden, thred=thred)

    @torch.inference_mode()
    def infer_from_audio_with_pitch_shift(self, audio_16k: np.ndarray,
                                           semitones: float = 0.0,
                                           thred: float = 0.03) -> tuple:
        f0 = self.infer_from_audio(audio_16k, thred=thred).astype(np.float32)

        if semitones != 0:
            voiced = f0 > 0
            f0[voiced] *= 2 ** (semitones / 12)

        f0 = np.clip(f0, 0, FMAX)

        f0_coarse = f0.copy()
        voiced_mask = f0_coarse > 0
        f0_coarse[voiced_mask] = (
            np.log2(f0_coarse[voiced_mask] / FMIN) / np.log2(FMAX / FMIN) * 254 + 1
        )
        f0_coarse = np.clip(f0_coarse, 1, 255).astype(int)
        f0_coarse[~voiced_mask] = 0

        return f0, f0_coarse
