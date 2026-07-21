# Literature Review & Prior-Work Comparison (Sec 6, [C3])

This document (a) surveys the speech-based Parkinson (PD) detection literature we
build on, (b) tabulates what each prior method reported vs. what we obtain when we
**re-implement it on THIS corpus** under one leakage-safe protocol, and (c) argues
critically why cross-paper numbers are **not** directly comparable and how that ties
to shortcut learning and dataset shift.

## 6.1 Prior-work comparison table

| Author (year) | Dataset (theirs) | Features | Model | Result THEY reported | Re-implemented here (this corpus) |
|---|---|---|---|---|---|
| Little et al. (2009) | Oxford sustained /a/ | Dysphonia: jitter, shimmer, HNR, RPDE, DFA, PPE | SVM | ~91% acc (sustained vowel) | `little2009` — dysphonia subset + SVM(RBF); see `artifacts/results/model_comparison.csv` |
| Tsanas et al. (2012) | Sustained vowels, telemonitoring | Dysphonia + feature selection (mRMR/LASSO) | RF / SVM | ~99% (subject-level, sustained /a/) | `tsanas2012` — LASSO selection + RandomForest |
| Vasquez-Correa et al. (2018) | PC-GITA (Spanish) | DisVoice: articulation/phonation/prosody | CNN | ~70–80% (continuous speech, multilingual) | `vasquez2018` — DisVoice-style vector → MLP (CNN in original; caveat below) |
| Moro-Velazquez et al. (2019) | PC-GITA | Phoneme-grouped MFCC | GMM / GMM-UBM | ~85–95% (task dependent) | `moro2019` — per-class GMM likelihood-ratio on MFCC sequences |
| NeuroVoz (2024) | NeuroVoz (Spanish) | Mel-spectrogram | ResNet-18 | high (in-corpus) | `neurovoz2024` — mel-spectrogram + CNN (our MelCNN) |
| **Ours (proposed)** | This YouTube PD/HC corpus | Whisper + BERT + hand-crafted | Fusion (concat / cross-attn / gated) | — | `fusion_*` rows |

Numbers marked "reported" are as stated by the authors on **their** datasets/tasks and
are reproduced here from their papers for context only; the right-hand column is the
only apples-to-apples comparison because it uses the same corpus, folds, scaler and
metrics (Sec 4/5).

### Provenance / re-implementation status ([C2])
- All baselines above are **RE-IMPLEMENTATIONS** (original code not run) except where a
  public repo is used. Feature front-ends:
  - dysphonia/phonation via **parselmouth/Praat** (`src/features/acoustic.py`).
  - DisVoice: if the `disvoice` package is importable it is used; otherwise an
    articulation/phonation **fallback vector** built from the same Praat/librosa
    measures keeps the Vasquez baseline runnable (`src/features/extract.py::_disvoice_vector`).
  - mel-spectrograms via **librosa** (`src/features/spectrogram.py`).
- Library versions are pinned in `requirements.txt`; the exact feature spec + cache-key
  for every number is stored next to the results (Sec 8.7).

## 6.2 Why the reported numbers are not directly comparable (critical analysis)

1. **Different corpora / languages / tasks.** Little & Tsanas use sustained vowels
   (an easier, highly controlled phonation task); Vasquez/Moro/NeuroVoz use PC-GITA/
   NeuroVoz (Spanish, clinical protocol). Our corpus is **English, spontaneous (PD) vs
   read (HC), extracted from YouTube**. Accuracy is task-bound; a 99% sustained-vowel
   number says nothing about spontaneous conversational speech.

2. **Protocol differences drive most of the spread.** Many high numbers come from
   utterance-level (not subject-level) splits, which leak speaker identity. We enforce
   **subject-wise LOSO/StratifiedGroupKFold** with a train-only scaler for *every*
   model, so differences here reflect models, not evaluation generosity.

3. **Confounds / shortcut learning (Geirhos et al., 2020).** In this corpus PD=16 kHz
   and HC=44.1 kHz, and the two classes come from different recording pipelines. A model
   can reach near-perfect accuracy by keying on the **recording fingerprint** rather than
   pathology — a textbook "shortcut". Our confound audit (`src/audit/confound_check.py`)
   shows a metadata-only classifier at 100% and a spectral-fingerprint classifier still
   ~0.94 **after** band-limiting: the shortcut is real and only partially removed.
   Consequently, any model result must be read **relative to these confound baselines**,
   not against chance.

4. **Dataset shift (Roberts et al., 2021).** The PD/HC split also coincides with a
   task/style shift (spontaneous vs read) and a speaker-imbalance (PD_emma = 62% of PD
   utterances). These are exactly the "hidden stratification / dataset shift" failure
   modes catalogued for medical ML; we quantify sensitivity to them (drop-emma ablation,
   Sec 5.4) rather than reporting a single optimistic number.

## 6.3 Cross-corpus generalization (planned)

The framework is corpus-agnostic (`build_manifest.py` + config paths). A cross-corpus
test — train here, test on NeuroVoz/PC-GITA (or vice-versa) — is the decisive check for
whether a model learned pathology vs. corpus fingerprint. Hooks are in place
(`manifest` + `features` are dataset-independent); running it requires obtaining those
corpora and is left as the external-validation step.

## 6.4 Reused libraries / code (with versions in `requirements.txt`)

- **parselmouth / Praat** — dysphonia & phonation measures.
- **librosa / soundfile / soxr** — DSP, mel-spectrograms, resampling.
- **openSMILE** (optional) — eGeMAPS/ComParE standard voice-pathology sets.
- **DisVoice** (optional) — Vasquez-Correa articulation/phonation/prosody.
- **HuggingFace transformers** — Whisper (`openai/whisper-base`),
  Wav2Vec2 (`facebook/wav2vec2-base-960h`), BERT (`bert-base-uncased`), all frozen.
- **scikit-learn / xgboost** — classic ML + statistics.

## References
- Little et al., "Suitability of dysphonia measurements for telemonitoring of Parkinson's disease," IEEE TBME, 2009.
- Tsanas et al., "Novel speech signal processing algorithms for high-accuracy classification of Parkinson's disease," IEEE TBME, 2012.
- Vasquez-Correa et al., "Towards an automatic evaluation of the dysarthria level of patients with Parkinson's disease," J. Comm. Disorders / INTERSPEECH, 2018.
- Moro-Velazquez et al., "Phonetic relevance and phonemic grouping of speech in the automatic detection of Parkinson's disease," 2019.
- NeuroVoz corpus, 2024.
- Geirhos et al., "Shortcut learning in deep neural networks," Nature Machine Intelligence, 2020.
- Roberts et al., "Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans," Nature Machine Intelligence, 2021.
