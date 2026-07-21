"""Deep models on CPU with a pragmatic budget (local full-run helper).

Rationale: Whisper/BERT/Wav2Vec2 are frozen precomputed embeddings, so the SSL
and fusion heads are light -> run them at FULL protocol (LOSO, all seeds).  The
mel-spectrogram CNNs (resnet18/mobilenetv2/neurovoz) cost ~8h each under full
LOSO x 3-seed on CPU, so they run under a reduced-but-still-subject-wise protocol
(StratifiedGroupKFold k=5, 1 seed, fewer epochs).  Regenerate CNN numbers at full
protocol on GPU via colab/run_full_colab.ipynb.

Run:  python run_deep_cpu.py
"""
from __future__ import annotations

import copy

from src.pipelines import run_deep
from src.utils.common import get_logger, load_config

LOG = get_logger("deep_cpu")

# resnet18/mobilenetv2 now use FROZEN ImageNet-backbone embeddings -> full protocol.
# Only the from-scratch mel-CNN (neurovoz2024) needs the reduced CPU budget.
FULL = ["wav2vec2", "fusion_concat", "fusion_xattn", "fusion_gated",
        "resnet18", "mobilenetv2"]
CNN = ["neurovoz2024"]


def main():
    cfg = load_config()

    # frozen-backbone spectrogram embeddings (resnet18_emb / mobilenetv2_emb)
    from src.features import spec_cnn_emb
    spec_cnn_emb.run(cfg)

    LOG.info("=== deep pass A (FULL protocol): %s ===", FULL)
    run_deep.run(copy.deepcopy(cfg), only=FULL)

    LOG.info("=== deep pass B (reduced CPU protocol, SGKF k=5, 1 seed): %s ===", CNN)
    cfg_cnn = copy.deepcopy(cfg)
    cfg_cnn["eval"]["cv"] = "sgkf"
    cfg_cnn["eval"]["k_folds"] = 5
    cfg_cnn["train"]["epochs"] = 15
    cfg_cnn["seeds"] = cfg_cnn["seeds"][:1]
    run_deep.run(cfg_cnn, only=CNN)
    LOG.info("=== DEEP DONE ===")


if __name__ == "__main__":
    main()
