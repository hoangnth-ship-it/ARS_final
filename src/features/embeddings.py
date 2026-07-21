"""Frozen deep embeddings (Sec 3.4): Whisper encoder, Wav2Vec2, BERT.

All models frozen (feature-extraction mode).  Pooling:
  - Whisper encoder: mean over time of chosen hidden layer (attention pooling is
    learned in the fusion head; here we cache a fixed 512-d/768-d vector).
  - Wav2Vec2: mean pooling of last hidden state.
  - BERT: [CLS] token.

Lazily loads models so pure-ML paths never import torch/transformers.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.utils.common import get_logger, pick_device

LOG = get_logger("embeddings")


@lru_cache(maxsize=None)
def _whisper(model_name: str, device: str):
    import torch
    from transformers import WhisperModel, WhisperFeatureExtractor
    fe = WhisperFeatureExtractor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return fe, model


@lru_cache(maxsize=None)
def _wav2vec2(model_name: str, device: str):
    import torch
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return fe, model


@lru_cache(maxsize=None)
def _bert(model_name: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def whisper_embed(y: np.ndarray, sr: int, model_name: str, layer: str = "last",
                  device: str = None) -> np.ndarray:
    import torch
    device = device or pick_device()
    fe, model = _whisper(model_name, device)
    inp = fe(y, sampling_rate=sr, return_tensors="pt")
    feats = inp.input_features.to(device)
    with torch.no_grad():
        out = model.encoder(feats, output_hidden_states=True)
        hs = out.hidden_states
        h = hs[len(hs) // 2] if layer == "mid" else out.last_hidden_state
    return h.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def wav2vec2_embed(y: np.ndarray, sr: int, model_name: str, device: str = None) -> np.ndarray:
    import torch
    device = device or pick_device()
    fe, model = _wav2vec2(model_name, device)
    inp = fe(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model(inp.input_values.to(device))
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)


def bert_embed(text: str, model_name: str, device: str = None) -> np.ndarray:
    import torch
    device = device or pick_device()
    tok, model = _bert(model_name, device)
    enc = tok(text or "", return_tensors="pt", truncation=True, max_length=128,
              padding="max_length")
    with torch.no_grad():
        out = model(**{k: v.to(device) for k, v in enc.items()})
    return out.last_hidden_state[:, 0].squeeze(0).cpu().numpy().astype(np.float32)  # [CLS]
