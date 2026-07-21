# Parkinson Detection from Speech — Whisper + BERT + Hand-crafted

Reproducible, leakage-safe benchmark for classifying **Parkinson (PD) vs Healthy
Control (HC)** from English speech, addressing the five review points (C1–C5) in
`MASTER_PROMPT.md`. Every number is produced by code and reproducible with one command.

## What this addresses (teacher's 5 points)

| # | Criticism | Where it's handled |
|---|---|---|
| **C1** | "no comparison of related methods" | 15+ models under one protocol (`src/pipelines/`), stat tests (`src/stats/`), sweep (`src/sweep/`) |
| **C2** | "re-run / re-implement prior methods" | `src/models/paper_baselines.py` (Little/Tsanas/Vasquez/Moro) + NeuroVoz CNN |
| **C3** | "no literature review" | `docs/related_work.md` (table + critical analysis) |
| **C4** | "no model diagrams" | `artifacts/figures/architecture_diagrams.md` (Mermaid) + result plots |
| **C5** | rigor / reproducibility | subject-wise CV, train-only scaler, confound audit, ≥3 seeds, cache-keys |

## Dataset (verified from `Data/`)

- 1091 utterances, 22 speakers: **10 PD** (DL, emma, Faces×6, LW, Tessi) + **12 HC**.
- **Confound:** PD=16 kHz / HC=44.1 kHz; PD spontaneous / HC read; `PD_emma` = 62% of PD.
- Manifest with true (header-read) sample-rate/duration is built by `build_manifest.py`.

## Install

```bash
pip install -r requirements.txt
```
Optional packages (`pyloudnorm`, `webrtcvad`, `opensmile`, `disvoice`, `thop`, `torchaudio`)
degrade gracefully — the pipeline runs without them (RMS/energy/params-only fallbacks).

## One-command pipeline

```bash
python run.py all --fast     # end-to-end smoke run (CPU, minutes; subsampled)
python run.py all            # full run (GPU advised)
# or step by step:
python run.py manifest       # -> artifacts/manifest.csv
python run.py confound       # -> confound audit  [C5]
python run.py features --which all
python run.py baselines      # ML + paper baselines  [C1][C2]
python run.py deep           # CNN / Wav2Vec2 / fusion  [C/D/E]
python run.py stats          # McNemar + Friedman  [C1]
python run.py sweep          # feature/arch size ablation  [Sec 8]
python run.py figures        # all figures  [C4]
python run.py report         # docs/RESULTS.md
pytest -q                    # acceptance tests
```
`make <target>` mirrors these (see `Makefile`).

## Pipeline

```
raw wav ─► preprocess (mono/16k/band-limit 7.5k/LUFS/VAD/fix-len)
        ├─ log-Mel ─► Whisper encoder (frozen) ─► attn-pool 512 ┐
        ├─ waveform ─► Wav2Vec2 (frozen) ─► pool 768            ├─► Fusion ─► head ─► P(PD)
        ├─ transcript ─► BERT (frozen) ─► [CLS] 768             │  (concat/xattn/gated)
        └─ 14 hand-crafted (jitter/shimmer/HNR/MFCC/...)        ┘
```
Full diagrams: `artifacts/figures/architecture_diagrams.md`.

## Models compared (`configs/models.yaml`)

- **A. Classic ML:** SVM-RBF, RandomForest, XGBoost, LogisticRegression, KNN.
- **B. Prior-work re-implementations [C2]:** Little(2009), Tsanas(2012), Vasquez(2018),
  Moro(2019), NeuroVoz(2024).
- **C. CNN spectrogram:** ResNet-like MelCNN, MobileNet-style.
- **D. Self-supervised:** Wav2Vec2 (frozen embedding + MLP).
- **E. Proposed fusion:** Whisper + BERT + hand-crafted, 3 fusion strategies.

Every model: subject-wise CV, train-only `StandardScaler`, ≥3 seeds, per-utterance
predictions saved; metrics reported at utterance **and** subject level.

## Confound audit ([C5])

`src/audit/confound_check.py` trains metadata-only and spectral-fingerprint probes
before/after band-limiting. On this corpus band-limiting alone does **not** fully remove
the recording-pipeline confound (spectral probe ≈ 0.94) — an honest negative result that
means downstream accuracy must be read **relative to these baselines**, not chance.
The cutoff is config-driven (`configs/config.yaml: preprocess.band_limit`) so the loop
can be swept.

## Feature / architecture sweep ([Sec 8])

`configs/sweep.yaml` lists variants; `src/sweep/runner.py` groups them by **input
cache-key** (re-extract only when input size changes, reuse features for arch-only
changes), trains each MelCNN, and profiles Accuracy/AUC/F1/params/FLOPs/latency/memory/
output-dim → `artifacts/results/sweep_results.csv` + Pareto plots.

## Reproducibility

- Fixed seeds (numpy/torch/random); config-driven; logging (no scattered prints).
- Feature cache keyed by input params (`src/features/cache.py`) — fair reuse, no leakage.
- `docs/RESULTS.md` regenerates from CSVs; no hardcoded numbers.

## Layout

```
configs/   config.yaml, models.yaml, sweep.yaml
src/       data/ preprocess/ audit/ features/ models/ eval/ stats/ sweep/ figures/ pipelines/
docs/      related_work.md [C3], RESULTS.md (auto)
tests/     pytest acceptance tests
run.py     one-command driver ; Makefile mirrors it
```

## Notes / limitations

- 22 speakers is small; LOSO is the default CV. Numbers are inflated by the confound
  described above — this project's contribution is the **rigorous protocol + audit**,
  not a headline accuracy.
- Full deep runs benefit from a GPU; `--fast` gives a CPU smoke run.
- Cross-corpus external validation (NeuroVoz/PC-GITA) is wired but needs those datasets.
