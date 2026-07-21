"""Frozen ImageNet-backbone spectrogram embeddings (Sec 4.C).

Faithful to the brief's "ResNet-18 / MobileNetV2 (ImageNet pretrained, frozen)":
the pretrained CNN is used as a FROZEN feature extractor over the log-mel image
(1->3 channels, resized to 224). Embeddings are cached once per backbone from the
already-cached `melspec` group, so the classifier head then trains fast under the
FULL subject-wise protocol (like the other embedding models).

Groups produced: resnet18_emb (512-d), mobilenetv2_emb (1280-d).
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features import cache
from src.utils.common import get_logger, load_config, pick_device, resolve

LOG = get_logger("spec_cnn_emb")

BACKBONES = {"resnet18_emb": "resnet18", "mobilenetv2_emb": "mobilenet_v2"}


def _build_backbone(kind: str, device: str):
    import torch
    import torchvision.models as M
    if kind == "resnet18":
        net = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()          # -> 512-d
    elif kind == "mobilenet_v2":
        net = M.mobilenet_v2(weights=M.MobileNet_V2_Weights.IMAGENET1K_V1)
        net.classifier = torch.nn.Identity()  # -> 1280-d
    else:
        raise ValueError(kind)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _embed_all(melspec: np.ndarray, kind: str, device: str, batch=32) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    net = _build_backbone(kind, device)
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(melspec), batch):
            xb = torch.tensor(melspec[i:i + batch], dtype=torch.float32, device=device)
            xb = xb.unsqueeze(1)                       # [B,1,M,T]
            # min-max to [0,1] per-sample, then to 3ch, resize 224
            b = xb.flatten(1).amin(1).view(-1, 1, 1, 1)
            t = xb.flatten(1).amax(1).view(-1, 1, 1, 1)
            xb = (xb - b) / (t - b + 1e-6)
            xb = xb.repeat(1, 3, 1, 1)
            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
            xb = (xb - mean) / std
            out.append(net(xb).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def run(cfg=None, only=None) -> None:
    cfg = cfg or load_config()
    spec = cfg["features"]
    fspec = cache.feature_spec(cfg)
    mel = cache.load(cfg, "melspec", cache.cache_key("melspec", fspec))
    if mel is None:
        raise FileNotFoundError("melspec cache missing -- run features first.")
    ids, X = mel
    device = pick_device(cfg["train"]["device"])
    for group, kind in BACKBONES.items():
        if only and group not in only:
            continue
        key = cache.cache_key(group, fspec)
        if cache.exists(cfg, group, key):
            LOG.info("reuse %s [%s]", group, key)
            continue
        LOG.info("extracting %s (%s) on %d spectrograms", group, kind, len(ids))
        emb = _embed_all(X, kind, device)
        cache.save(cfg, group, key, ids, emb, fspec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    run(load_config(args.config), args.only)
