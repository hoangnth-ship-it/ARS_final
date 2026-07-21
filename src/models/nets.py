"""Network architectures (Sec 4.C/D/E, Sec 8).

- MelCNN         : config-driven CNN on log-mel (blocks/widths/pooling/padding) -> sweep
- EmbeddingMLP   : MLP head on a frozen embedding vector (Wav2Vec2 / any tabular)
- FusionNet      : Whisper + BERT + hand-crafted fusion (concat / cross-attn / gated)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
class AttnPool(nn.Module):
    """Attention pooling over time: [B, C, T] -> [B, C]."""
    def __init__(self, channels: int):
        super().__init__()
        self.w = nn.Linear(channels, 1)

    def forward(self, x):                      # x: [B, C, T]
        h = x.transpose(1, 2)                   # [B, T, C]
        a = torch.softmax(self.w(h), dim=1)     # [B, T, 1]
        return (h * a).sum(dim=1)               # [B, C]


class MelCNN(nn.Module):
    """CNN on [B, 1, n_mels, n_frames].  All size axes come from config (Sec 8.1).

    Downsampling is guarded: a 2x2 pool is applied after a block only while both
    spatial dims stay >= 2, so deep 'valid'-padding stacks never collapse a dim
    to 0.  Conv padding follows the 'padding' axis but is clamped to 'same' once a
    spatial dim gets small, keeping every configured variant runnable.
    """
    def __init__(self, n_mels: int, widths: List[int], pooling: str = "avg",
                 padding: str = "same", dropout: float = 0.3):
        super().__init__()
        self.pad_mode = padding
        self.pooling = pooling
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        cin = 1
        for w in widths:
            self.convs.append(nn.Conv2d(cin, w, 3, padding=0))  # padding applied in forward
            self.bns.append(nn.BatchNorm2d(w))
            cin = w
        self.cin = cin
        if pooling == "attention":
            self.attn = AttnPool(cin)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, 64),
                                  nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):                        # x: [B, n_mels, T]
        import torch.nn.functional as F
        if x.dim() == 3:
            x = x.unsqueeze(1)                   # [B, 1, n_mels, T]
        h = x
        for conv, bn in zip(self.convs, self.bns):
            # 'same' -> pad 1; 'valid' -> pad 0, but clamp to 1 when a dim is small
            small = min(h.shape[2], h.shape[3]) <= 4
            pad = 1 if (self.pad_mode == "same" or small) else 0
            h = F.relu(bn(conv(F.pad(h, (pad, pad, pad, pad)))))
            if min(h.shape[2], h.shape[3]) >= 2:      # guarded downsample
                h = F.max_pool2d(h, 2, ceil_mode=True)
        if self.pooling == "attention":
            z = self.attn(h.mean(dim=2))         # pool mel axis -> [B,C,W]; attn over time
        elif self.pooling == "max":
            z = h.amax(dim=(2, 3))
        else:                                    # avg
            z = h.mean(dim=(2, 3))
        return self.head(z)


class EmbeddingMLP(nn.Module):
    """MLP head over a frozen embedding vector (Sec 4.D)."""
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)


class FusionNet(nn.Module):
    """Whisper + BERT + hand-crafted fusion (Sec 4.E, Sec 10).

    Input is the concatenation [whisper | bert | handcrafted]; `sections` gives the
    dim of each so the module can split and fuse them.
    """
    def __init__(self, sections: List[int], mode: str = "concat",
                 proj: int = 128, dropout: float = 0.3):
        super().__init__()
        self.sections = sections
        self.mode = mode
        self.proj = nn.ModuleList([nn.Linear(d, proj) for d in sections])
        if mode == "cross_attention":
            self.attn = nn.MultiheadAttention(proj, num_heads=4, batch_first=True)
        if mode == "gated":
            self.gate = nn.Linear(proj * len(sections), len(sections))
        fused_dim = proj if mode in ("cross_attention", "gated") else proj * len(sections)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused_dim, 64),
                                  nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        parts, i = [], 0
        for d, p in zip(self.sections, self.proj):
            parts.append(torch.relu(p(x[:, i:i + d])))
            i += d
        stack = torch.stack(parts, dim=1)              # [B, n_mod, proj]
        if self.mode == "cross_attention":
            out, _ = self.attn(stack, stack, stack)    # self/cross attention over modalities
            fused = out.mean(dim=1)
        elif self.mode == "gated":
            flat = stack.reshape(stack.size(0), -1)
            g = torch.softmax(self.gate(flat), dim=1).unsqueeze(-1)  # [B, n_mod, 1]
            fused = (stack * g).sum(dim=1)
        else:                                          # concat
            fused = stack.reshape(stack.size(0), -1)
        return self.head(fused)
