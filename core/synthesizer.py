"""
RVC v1/v2 VITS synthesizer — inference path only.
Adapted from RVC-Project (MIT/GPL-3.0).
Supports SynthesizerTrnMs256NSFsid (v1) and SynthesizerTrnMs768NSFsid (v2).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm
from torch.nn import Conv1d, ConvTranspose1d


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    return t_act * s_act


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, x.shape[-1:], self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, window_size=None,
                 heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert t_s == t_t, "relative attention only for self-attention"
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "proximal bias only for self-attention"
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert t_s == t_t, "block attn only for self-attn"
                block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded = F.pad(relative_embeddings, [0, 0, pad_length, pad_length])
        else:
            padded = relative_embeddings
        return padded[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, 1])
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1])
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, [0, length - 1])
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        x_flat = F.pad(x_flat, [length, 0])
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size,
                 p_dropout=0.0, activation=None, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.padding = self._causal_padding if causal else self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        return F.pad(x, [pad_l, 0, 0, 0, 0, 0])

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        return F.pad(x, [pad_l, pad_r, 0, 0, 0, 0])


class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers,
                 kernel_size=1, p_dropout=0.0, window_size=4, **kwargs):  # window_size inferred from checkpoint
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads,
                                   p_dropout=p_dropout, window_size=window_size)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size,
                    p_dropout=p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class TextEncoder(nn.Module):
    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads,
                 n_layers, kernel_size, p_dropout, f0=True, window_size=4):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(256, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers,
                                kernel_size, float(p_dropout), window_size=window_size)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        if hasattr(self, "emb_pitch"):
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        else:
            x = self.emb_phone(phone)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(self._sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask

    @staticmethod
    def _sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


class TextEncoder768(nn.Module):
    """v2 encoder — 768-dim ContentVec input."""
    def __init__(self, out_channels, hidden_channels, filter_channels, n_heads,
                 n_layers, kernel_size, p_dropout, f0=True, window_size=4):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.emb_phone = nn.Linear(768, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)
        self.encoder = Encoder(hidden_channels, filter_channels, n_heads, n_layers,
                                kernel_size, float(p_dropout), window_size=window_size)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        if hasattr(self, "emb_pitch"):
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        else:
            x = self.emb_phone(phone)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(self._sequence_mask(lengths, x.size(2)), 1).to(x.dtype)
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask

    @staticmethod
    def _sequence_mask(length, max_length=None):
        if max_length is None:
            max_length = length.max()
        x = torch.arange(max_length, dtype=length.dtype, device=length.device)
        return x.unsqueeze(0) < length.unsqueeze(1)


class WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers,
                 gin_channels=0, p_dropout=0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.drop = nn.Dropout(p_dropout)
        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()

        if gin_channels != 0:
            self.cond_layer = weight_norm(
                nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            )

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(weight_norm(
                nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size,
                          dilation=dilation, padding=padding)
            ))
            res_skip_ch = 2 * hidden_channels if i < n_layers - 1 else hidden_channels
            self.res_skip_layers.append(weight_norm(nn.Conv1d(hidden_channels, res_skip_ch, 1)))

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            g_l = (g[:, i * 2 * self.hidden_channels:(i + 1) * 2 * self.hidden_channels, :]
                   if g is not None else torch.zeros_like(x_in))
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if hasattr(self, "cond_layer"):
            remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            remove_weight_norm(l)
        for l in self.res_skip_layers:
            remove_weight_norm(l)


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0
        super().__init__()
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers,
                      p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, dim=1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, dim=1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
        x = torch.cat([x0, x1], dim=1)
        return x * x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate,
                 n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(channels, hidden_channels, kernel_size,
                                      dilation_rate, n_layers, gin_channels=gin_channels,
                                      mean_only=True)
            )
            self.flows.append(Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        flows = self.flows if not reverse else reversed(self.flows)
        for flow in flows:
            x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for l in self.flows:
            if hasattr(l, "remove_weight_norm"):
                l.remove_weight_norm()


class Flip(nn.Module):
    def forward(self, x, *args, **kwargs):
        return torch.flip(x, [1])


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1,
                   dilation=dilation[i], padding=get_padding(kernel_size, dilation[i]))
            for i in range(3)
        ])
        self.convs2 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1,
                   dilation=1, padding=get_padding(kernel_size, 1))
            for _ in range(3)
        ])

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x if x_mask is None else x * x_mask


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1,
                   dilation=d, padding=get_padding(kernel_size, d))
            for d in dilation
        ])

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            x = xt + x
        return x if x_mask is None else x * x_mask


class SineGen(nn.Module):
    """Sinusoidal source signal generator for NSF."""
    def __init__(self, samp_rate, harmonic_num=0, sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0, flag_for_pulse=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).float()

    def forward(self, f0, upp):
        # f0: [B, T] at frame rate — upsample to sample rate before computing sines
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)  # [B, T, 1]
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for i in range(self.harmonic_num):
                f0_buf[:, :, i + 1] = f0_buf[:, :, 0] * (i + 2)

            # Upsample F0 to sample rate: [B, T*upp, dim]
            f0_buf_up = F.interpolate(
                f0_buf.transpose(1, 2), scale_factor=upp, mode="nearest"
            ).transpose(1, 2)

            rad_values = (f0_buf_up / self.sampling_rate) % 1
            rand_ini = torch.rand(f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            sines = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * math.pi
            )  # [B, T*upp, dim]

            uv = self._f02uv(f0)  # [B, T, 1]
            uv = F.interpolate(uv.transpose(2, 1), scale_factor=upp, mode="nearest").transpose(2, 1)
            # uv: [B, T*upp, 1] — same length as sines now
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            sines = uv * self.sine_amp * sines + noise_amp * torch.randn_like(sines)
        return sines.transpose(1, 2)  # [B, dim, T*upp]


class SourceModuleHnNSF(nn.Module):
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, add_noise_std=0.003,
                 voiced_threshold=0):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num, sine_amp, add_noise_std,
                                  voiced_threshold)
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x, upp=None):
        sine_waves = self.l_sin_gen(x, upp)
        sine_merge = self.l_tanh(self.l_linear(sine_waves.transpose(1, 2)).transpose(1, 2))
        return sine_merge, None, None


class GeneratorNSF(nn.Module):
    """HiFi-GAN generator with Neural Source Filter conditioning.
    Plain (no weight_norm) — compose_weight_norm handles checkpoint format."""
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes,
                 resblock_dilation_sizes, upsample_rates, upsample_initial_channel,
                 upsample_kernel_sizes, gin_channels, sr):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.f0_upsamp = nn.Upsample(scale_factor=math.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(sr, harmonic_num=0)
        self.noise_convs = nn.ModuleList()
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock_cls = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)
            )
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=stride_f0 * 2, stride=stride_f0, padding=stride_f0 // 2))
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock_cls(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        self.upp = math.prod(upsample_rates)

    def forward(self, x, f0, g=None):
        har_source, _, _ = self.m_source(f0, self.upp)  # [B, 1, T*upp]
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        pass  # no weight_norm applied — compose_weight_norm handles checkpoint format


class SynthesizerTrnMs256NSFsid(nn.Module):
    """RVC v1 — 256-dim HuBERT features."""
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 spk_embed_dim, gin_channels, sr, **kwargs):
        super().__init__()
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        window_size = kwargs.get("window_size", 4)
        flow_n_layers = kwargs.get("flow_n_layers", 4)
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels,
                                  n_heads, n_layers, kernel_size, p_dropout,
                                  window_size=window_size)
        self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes,
                                 resblock_dilation_sizes, upsample_rates,
                                 upsample_initial_channel, upsample_kernel_sizes,
                                 gin_channels=gin_channels, sr=sr)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, flow_n_layers,
                                           gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.randn_like(m_p) * torch.exp(logs_p)) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(
            (z * x_mask)[:, :, :max_len],
            self.f0_to_coarse(nsff0[:, :max_len * (z.shape[2] // nsff0.shape[1])
                                    if max_len else slice(None)]),
            g=g
        )
        return o, x_mask

    @staticmethod
    def f0_to_coarse(f0):
        return f0


class SynthesizerTrnMs768NSFsid(nn.Module):
    """RVC v2 — 768-dim ContentVec features."""
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 spk_embed_dim, gin_channels, sr, **kwargs):
        super().__init__()
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        window_size = kwargs.get("window_size", 4)
        flow_n_layers = kwargs.get("flow_n_layers", 4)
        self.enc_p = TextEncoder768(inter_channels, hidden_channels, filter_channels,
                                     n_heads, n_layers, kernel_size, p_dropout,
                                     window_size=window_size)
        self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes,
                                 resblock_dilation_sizes, upsample_rates,
                                 upsample_initial_channel, upsample_kernel_sizes,
                                 gin_channels=gin_channels, sr=sr)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, flow_n_layers,
                                           gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.randn_like(m_p) * torch.exp(logs_p)) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], nsff0, g=g)
        return o, x_mask


class SynthesizerTrnMs256NSFsid_nono(nn.Module):
    """RVC v1 — no F0 conditioning."""
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 spk_embed_dim, gin_channels, sr=None, **kwargs):
        super().__init__()
        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels,
                                  n_heads, n_layers, kernel_size, p_dropout, f0=False)
        self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes,
                                 resblock_dilation_sizes, upsample_rates,
                                 upsample_initial_channel, upsample_kernel_sizes,
                                 gin_channels=gin_channels, sr=22050)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4,
                                           gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = (m_p + torch.randn_like(m_p) * torch.exp(logs_p)) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], nsff0, g=g)
        return o, x_mask


class SynthesizerTrnMs768NSFsid_nono(nn.Module):
    """RVC v2 — no F0 conditioning."""
    def __init__(self, spec_channels, segment_size, inter_channels, hidden_channels,
                 filter_channels, n_heads, n_layers, kernel_size, p_dropout,
                 resblock, resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_initial_channel, upsample_kernel_sizes,
                 spk_embed_dim, gin_channels, sr=None, **kwargs):
        super().__init__()
        self.enc_p = TextEncoder768(inter_channels, hidden_channels, filter_channels,
                                     n_heads, n_layers, kernel_size, p_dropout, f0=False)
        self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes,
                                 resblock_dilation_sizes, upsample_rates,
                                 upsample_initial_channel, upsample_kernel_sizes,
                                 gin_channels=gin_channels, sr=22050)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4,
                                           gin_channels=gin_channels)
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def infer(self, phone, phone_lengths, pitch, nsff0, sid, max_len=None):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z_p = (m_p + torch.randn_like(m_p) * torch.exp(logs_p)) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], nsff0, g=g)
        return o, x_mask


def compose_weight_norm(state_dict: dict) -> dict:
    """
    Convert weight_g + weight_v pairs into plain weight tensors.
    Required because GeneratorNSF uses plain convs but checkpoints
    may have been saved with weight_norm applied.
    """
    import torch.nn.functional as F_nn
    result = {}
    skip = set()

    # Find all weight_g/weight_v base names
    wn_bases = set()
    for k in state_dict:
        if k.endswith('.weight_g'):
            base = k[:-len('.weight_g')]
            if base + '.weight_v' in state_dict:
                wn_bases.add(base)

    for k, v in state_dict.items():
        if k in skip:
            continue
        base_g = k[:-len('.weight_g')] if k.endswith('.weight_g') else None
        base_v = k[:-len('.weight_v')] if k.endswith('.weight_v') else None

        if base_g is not None and base_g in wn_bases:
            weight_v = state_dict[base_g + '.weight_v']
            weight_g = v
            norm_axes = tuple(range(1, weight_v.ndim))
            result[base_g + '.weight'] = weight_g * F_nn.normalize(weight_v, dim=norm_axes)
            skip.add(k)
            skip.add(base_g + '.weight_v')
        elif base_v is not None and base_v in wn_bases:
            skip.add(k)  # handled by weight_g branch
        else:
            k_clean = k.replace("enc_q.", "")
            result[k_clean] = v

    return result


def infer_config_from_state_dict(state_dict: dict) -> dict:
    """
    Auto-detect RVC model config from state dict (for safetensors / legacy .pth).
    Returns dict with version, f0, sr, config list.
    """
    import re

    # Version: v2 = 768-dim emb_phone, v1 = 256-dim
    phone_w = state_dict.get("enc_p.emb_phone.weight")
    if phone_w is not None:
        version = "v2" if phone_w.shape[1] == 768 else "v1"
        hidden_channels = phone_w.shape[0]
    else:
        version = "v2"
        hidden_channels = 192

    # F0 conditioning
    f0 = 1 if "enc_p.emb_pitch.weight" in state_dict else 0

    # Speaker embedding
    emb_g = state_dict.get("emb_g.weight")
    spk_embed_dim = emb_g.shape[0] if emb_g is not None else 109
    gin_channels = emb_g.shape[1] if emb_g is not None else 256

    # inter_channels from flow pre (half_channels = inter//2)
    flow_pre = state_dict.get("flow.flows.0.pre.weight")
    inter_channels = (flow_pre.shape[1] * 2) if flow_pre is not None else 192

    # Upsample kernel sizes — from ups.X.weight_v or ups.X.weight
    ups_ks = []
    for i in range(8):
        key = f"dec.ups.{i}.weight_v" if f"dec.ups.{i}.weight_v" in state_dict else f"dec.ups.{i}.weight"
        wv = state_dict.get(key)
        if wv is None:
            break
        ups_ks.append(wv.shape[2])

    n_ups = len(ups_ks)
    if n_ups == 5:
        sr_int = 48000
        upsample_rates = [10, 6, 2, 2, 2]
    else:
        sr_int = 40000
        upsample_rates = [10, 10, 2, 2]
        ups_ks = ups_ks[:4]  # ensure 4 entries

    # upsample_initial_channel from conv_pre
    cp_key = "dec.conv_pre.weight_v" if "dec.conv_pre.weight_v" in state_dict else "dec.conv_pre.weight"
    cp = state_dict.get(cp_key)
    upsample_initial_channel = cp.shape[0] if cp is not None else 512

    # Infer window_size from relative attention embeddings
    rel_k = state_dict.get("enc_p.encoder.attn_layers.0.emb_rel_k")
    window_size = (rel_k.shape[1] - 1) // 2 if rel_k is not None else 4

    # Infer flow WN n_layers from cond_layer bias: 2 * hidden_channels * n_layers
    cond_bias = state_dict.get("flow.flows.0.enc.cond_layer.bias")
    if cond_bias is not None and hidden_channels > 0:
        flow_n_layers = cond_bias.shape[0] // (2 * hidden_channels)
    else:
        flow_n_layers = 4

    config = [
        1025, 32, inter_channels, hidden_channels, 768,
        2, 6, 3, 0, "1",
        [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates, upsample_initial_channel, ups_ks,
        spk_embed_dim, gin_channels, sr_int
    ]
    return {
        "version": version, "f0": f0, "sr": sr_int, "config": config,
        "window_size": window_size, "flow_n_layers": flow_n_layers,
    }


def load_rvc_model(path: str, device: str = "cpu"):
    """
    Load a .pth or .safetensors RVC model.
    Auto-detects v1/v2, sample rate, and F0 conditioning.
    Returns (model, sr_int, version, f0).
    """
    path = str(path)
    is_safetensors = path.endswith(".safetensors")

    extra_kwargs = {}

    if is_safetensors:
        from safetensors.torch import load_file
        raw = load_file(path, device="cpu")
        info = infer_config_from_state_dict(raw)
        version = info["version"]
        f0 = info["f0"]
        sr_int = info["sr"]
        cfg = info["config"]
        extra_kwargs = {"window_size": info["window_size"], "flow_n_layers": info["flow_n_layers"]}
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            cfg = checkpoint["config"]
            version = checkpoint.get("version", "v1")
            f0 = checkpoint.get("f0", 1)
            sr_str = checkpoint.get("sr", "40k")
            raw = checkpoint.get("weight", checkpoint)
            sr_map = {"16k": 16000, "32k": 32000, "40k": 40000, "48k": 48000}
            sr_int = sr_map.get(str(sr_str), 40000)
            if isinstance(cfg[-1], str):
                cfg = list(cfg[:-1]) + [sr_int]
            # Also infer window_size and flow_n_layers from actual weights
            info = infer_config_from_state_dict(raw)
            extra_kwargs = {"window_size": info["window_size"], "flow_n_layers": info["flow_n_layers"]}
        else:
            raw = checkpoint if isinstance(checkpoint, dict) else checkpoint
            info = infer_config_from_state_dict(raw)
            version = info["version"]
            f0 = info["f0"]
            sr_int = info["sr"]
            cfg = info["config"]
            extra_kwargs = {"window_size": info["window_size"], "flow_n_layers": info["flow_n_layers"]}

    # Compose weight_norm params into plain weights
    state_dict = compose_weight_norm(raw)

    # Build model
    if version == "v1":
        model_cls = SynthesizerTrnMs256NSFsid if f0 else SynthesizerTrnMs256NSFsid_nono
    else:
        model_cls = SynthesizerTrnMs768NSFsid if f0 else SynthesizerTrnMs768NSFsid_nono

    net_g = model_cls(*cfg, **extra_kwargs)
    missing, unexpected = net_g.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (normal for some formats)")
    net_g.eval().to(device)

    return net_g, sr_int, version, f0
