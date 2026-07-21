# Parkinson's Disease Prediction from Speech Using Whisper: A Confound-Aware Multi-Model Comparative Study

*(Tên tiếng Việt: **Dự đoán bệnh Parkinson từ giọng nói sử dụng Whisper: Nghiên cứu so sánh đa mô hình có kiểm soát nhiễu**)*

> **Ghi chú (VI):** Bản thảo viết bằng **English academic** vì mục tiêu là chuẩn submission Q1/SCIE
> (yêu cầu của prompt phản biện). Toàn bộ con số được sinh tự động từ code (`artifacts/results/*.csv`)
> và tái hiện bằng 1 lệnh (`python run.py all`); không có số bịa. Phần phản biện reviewer + action plan
> nằm ở cuối (Mục A–C) và trong `docs/REVIEW_REPORT.md`.

**Author:** Nguyen Trong Huy Hoang — Faculty of Electrical and Electronics Engineering, Phenikaa University, Hanoi, Vietnam.

---

## Abstract

**Background.** Speech is an attractive non-invasive biomarker for Parkinson's Disease (PD), and large self-/weakly-supervised speech transformers such as Whisper promise strong transfer to clinical tasks with little labelled data. However, reported accuracies are seldom audited for dataset confounds, and prior methods are rarely re-executed under a common protocol, making cross-study comparison unreliable.

**Objective.** We ask three questions on a single English corpus: (RQ1) Does a Whisper+BERT multimodal model outperform classical, CNN, self-supervised, and *re-implemented prior-work* baselines under a strictly leakage-free, subject-wise protocol? (RQ2) How much of the measured performance reflects acquisition confounds rather than pathology? (RQ3) Can a metadata-only classifier explain the accuracy, and does band-limiting remove the confound?

**Methods.** On 1{,}091 utterances from 22 speakers (10 PD / 12 HC), we benchmark **sixteen models** under identical Leave-One-Subject-Out (LOSO) cross-validation with a train-only scaler and three random seeds: five classical ML models, **five re-implemented prior-work methods** (Little 2009, Tsanas 2012, Vásquez-Correa 2018, Moro-Velázquez 2019, and a NeuroVoz-style mel-CNN), two ImageNet-pretrained CNN feature extractors, a Wav2Vec2 encoder, and a Whisper+BERT+hand-crafted fusion network with three fusion strategies. We add a **closed-loop confound audit** and pairwise McNemar (Holm-corrected) plus a Friedman omnibus, together with a config-driven feature/architecture-size ablation reporting quality *and* cost (parameters, FLOPs, latency).

**Results.** A plain logistic regression on 40 hand-crafted features attains the highest scores (accuracy 0.948, AUC 0.989), statistically indistinguishable at the subject level from the best deep models; the Whisper+BERT fusion reaches AUC 0.953. Crucially, a classifier using **only the recording sample rate** reaches **100%** accuracy, and a spectral-fingerprint classifier remains at **98.4%** *after* 7.5 kHz band-limiting. The cohorts were acquired at disjoint sample rates (PD 16 kHz vs HC 44.1 kHz) and speaking styles (spontaneous vs read), so headline metrics partly encode acquisition rather than pathology.

**Conclusion.** We reposition the contribution around (i) a reproducible, leakage-free benchmark that re-executes prior methods on the same corpus, and (ii) a confound audit demonstrating that acquisition-matched validation is a prerequisite for credible speech-based PD screening. Code, figures, and a folder/mic demo are released.

**Keywords:** Parkinson's disease; Whisper; Wav2Vec2; speech biomarker; confound analysis; shortcut learning; dataset bias; subject-wise cross-validation; reproducibility.

---

## 1. Introduction

Parkinson's Disease (PD) is a progressive neurodegenerative disorder driven by loss of dopaminergic neurons in the *substantia nigra*, affecting tens of millions worldwide [bloem2021parkinson]. Beyond the cardinal motor triad (tremor, rigidity, bradykinesia), up to 89% of patients develop *hypokinetic dysarthria* [ramig2008speech]: phonatory deficits (hoarseness, tremor, breathiness), articulatory imprecision, and prosodic flattening [moro2021review]. Because these deviations appear early and are captured with a microphone, speech is an appealing low-cost screening signal. Clinical diagnosis remains symptom-based and can take on average 2.75 years to establish [rossi2021time], motivating objective automated tools.

Two method families dominate. Classical pipelines feed hand-crafted acoustic descriptors — jitter, shimmer, harmonics-to-noise ratio (HNR), fundamental-frequency statistics, and mel-frequency cepstral coefficients (MFCC) — to SVMs or random forests [tsanas2012novel, little2009suitability]. Deep learning instead learns representations directly from spectrograms [vasquez2018multimodal], and speech transformers such as Wav2Vec2 [baevski2020wav2vec] and Whisper [radford2023whisper] provide general-purpose embeddings that transfer to health tasks with limited labels.

Two problems recur across this literature and directly motivate our study. **First**, many reported numbers are not comparable: they come from different corpora, tasks, and evaluation protocols, and the original methods are rarely re-executed on a common dataset. **Second**, high accuracies are seldom audited for *shortcut learning* [geirhos2020shortcut]: when the two clinical cohorts are recorded under different conditions, a model can exploit the recording fingerprint rather than pathology — a canonical *dataset-shift* failure mode in medical ML [roberts2021common].

This work therefore asks not only *whether* Whisper embeddings detect PD, but *why* they appear to. Rather than proposing a single "best" model, we place a Whisper+BERT network within a broad, uniformly evaluated field that **includes faithful re-implementations of five prior-work methods**, and subject the whole field to an explicit confound audit.

**Research questions.** **(RQ1)** Can a Whisper-based model outperform classical, CNN, self-supervised, and prior-work baselines under a leakage-free, subject-wise protocol? **(RQ2)** To what extent are the performances driven by acquisition confounds rather than pathology? **(RQ3)** How much accuracy can a *metadata-only* classifier explain, and does band-limiting remove the confound?

**Contributions.**
1. A **16-model benchmark** across five paradigms under one LOSO protocol, train-only scaling, and three seeds — one of the largest uniformly-evaluated fields reported on this corpus, and, to our knowledge, the first to **re-implement Little (2009), Tsanas (2012), Vásquez-Correa (2018), Moro-Velázquez (2019), and a NeuroVoz-style CNN** on it under a common harness [C2].
2. A **closed-loop confound audit** with a pass/fail chance criterion, quantifying how much accuracy is attributable to sample-rate and spectral-pipeline fingerprints before and after band-limiting [C5].
3. **Statistical rigor**: per-utterance predictions saved for all models, pairwise McNemar with Holm–Bonferroni correction, and a Friedman omnibus with an explicit power caveat.
4. A **config-driven feature/architecture-size ablation** with a shared input-keyed feature cache, reporting quality (Accuracy/AUC/F1/sensitivity/specificity) *and* cost (parameters, FLOPs, latency, output dimension) with Pareto fronts.
5. A **fully reproducible release**: one command regenerates every table and figure; a folder/mic demo is provided.

---

## 2. Related Work

**Hand-crafted dysphonia measures.** Little et al. [little2009suitability] introduced dysphonia measures (including PPE) for telemonitoring; Tsanas et al. [tsanas2012novel] combined feature selection with ensembles to reach near-ceiling accuracy on sustained vowels. These remain strong, interpretable baselines but are typically validated on sustained phonation under controlled acquisition.

**Articulation/phonation front-ends and CNNs.** Vásquez-Correa et al. [vasquez2018multimodal] modelled articulation, phonation, and prosody (the DisVoice family) and applied CNNs to continuous speech in PC-GITA. Moro-Velázquez et al. [moro2019phonetic] exploited phonemic grouping with GMM-based modelling. The NeuroVoz corpus [mendes2024neurovoz] pairs mel-spectrograms with ResNet-style CNNs.

**Self-/weakly-supervised transformers.** Wav2Vec2 [baevski2020wav2vec], HuBERT [hsu2021hubert], and Whisper [radford2023whisper] yield transferable embeddings; text encoders such as BERT [devlin2019bert] and sentence transformers [reimers2019sentence] add lexical information when transcripts are available.

**Confounds and generalization.** Shortcut learning [geirhos2020shortcut] and dataset-shift pitfalls in medical ML [roberts2021common] warn that in-corpus accuracy can encode acquisition artefacts. Domain-mismatch effects are documented for PD speech across languages and devices [ibarra2023domain, orozco2014new]. Automated acoustic analysis of hypokinetic dysarthria is reviewed in [hlavnicka2017automated, moro2021review].

**Critical gap.** Across these works, (i) reported numbers are not directly comparable (different corpora/tasks/protocols), (ii) prior methods are rarely re-executed on a new corpus, and (iii) confounds are seldom audited quantitatively. Table 1 makes the reproduction explicit; §6 shows the confound is dominant on our corpus. A full author-by-author reproduction table is maintained in `docs/related_work.md`.

**Table 1. Prior work vs. our re-implementation on this corpus (AUC).**

| Method | Original domain | Front-end | Model | Reported (their corpus) | Re-impl. here (AUC) |
|---|---|---|---|---|---|
| Little (2009) | Sustained /a/ | Dysphonia | SVM | ~0.91 acc | 0.828 |
| Tsanas (2012) | Sustained /a/ | Dysphonia + selection | RF | ~0.99 acc | 0.819 |
| Vásquez-Correa (2018) | PC-GITA | DisVoice | CNN/NN | ~0.70–0.80 | 0.957 |
| Moro-Velázquez (2019) | PC-GITA | Phonemic MFCC | GMM | ~0.85–0.95 | 0.798 |
| NeuroVoz-style (2024) | NeuroVoz | Mel-spectrogram | ResNet CNN | high (in-corpus) | 0.886 |

Reported numbers are on the authors' own corpora/tasks and are provided for context only; only the right-hand column is a like-for-like comparison. Discrepancies are analysed in §7.

---

## 3. Data and Confound Characterisation

**Corpus.** We use an English PD/HC speech corpus of **1{,}091 utterances** from **22 speakers** (**10 PD**, **12 HC**). PD audio is spontaneous speech extracted from public videos of ten patients; HC audio is read speech from twelve control speakers. Subject IDs are namespaced by class and derived from a non-uniform folder layout (e.g., the collection source *Faces* contributes six PD speakers), so a single person never spans folds. Utterance counts are 578 PD / 513 HC.

**Data collection.** Both cohorts were sourced from **publicly available YouTube videos**. The PD material comes from ten patients across five sources/collections (patient vlogs and interviews in which individuals discuss their condition); the HC material comes from twelve control speakers reading text. For every speaker, the audio track was downloaded and **manually segmented** into short single-sentence utterances, and each utterance was **transcribed manually** in English. Recordings are stored as stereo PCM-16 WAV; the corpus provides both a raw variant and a processed variant per class (PD: *original* / *denoised*; HC: *original* / *cleaned*), and this study uses the denoised-PD + cleaned-HC variant by default. Because the two cohorts originate from different uploaders and recording chains, they differ systematically in sample rate (PD 16 kHz vs HC 44.1 kHz) and speaking style (spontaneous vs read) — the confounds quantified in §5.3.

**Demographics.** Speakers span approximately **40–80 years** of age. Finer per-speaker attributes (exact age, sex, disease stage, medication status) are not systematically documented. Age is itself a known covariate of voice quality; because it is not matched speaker-by-speaker between the cohorts, it constitutes an additional uncontrolled factor alongside the acquisition confounds of §5.3.

**Ethics and licensing.** Only data that speakers had **already made public** on YouTube were used; there was no interaction with participants and no access to private clinical records beyond what individuals chose to disclose publicly. However, **informed consent for research use was not obtained directly from the speakers**, and fine-grained demographic metadata (sex, disease stage/medication) are unavailable. This precludes any clinical deployment on the basis of this corpus and motivates future replication on IRB-approved, consented, demographically documented cohorts. Re-use of the underlying PD dataset follows its citation requirement (Shanghai Jiao Tong University, 2021); no personally identifying information is redistributed.

**Known confounds (audited in §6).** (i) **Sample rate:** PD is 100% 16 kHz, HC is 100% 44.1 kHz — a near-perfect class proxy. (ii) **Speaking style:** PD spontaneous vs HC read — a task shift correlated with the label. (iii) **Speaker imbalance:** one PD speaker (`PD_emma`) contributes 62% of PD utterances (361/578), risking speaker memorisation. These are recorded transparently in the manifest and drive the design of both the preprocessing (§4) and the audit (§6).

---

## 4. Methods

### 4.1 Uniform preprocessing
Every waveform passes an identical, config-toggled pipeline: (1) mono downmix and resampling to 16 kHz (soxr high-quality); (2) **band-limiting** — an 8th-order zero-phase low-pass at 7.5 kHz applied to **both** cohorts to erase the sample-rate fingerprint; (3) loudness normalization (EBU R128 / LUFS, RMS fallback); (4) energy-based VAD trimming with fixed 6 s padding/center-crop. Transcripts are lower-cased with fillers retained (disfluency is disease-relevant).

### 4.2 Features (frozen extractors; train-only scaling)
Five cached feature groups feed the models, each keyed by its input parameters so architecture-only changes reuse features (fair comparison, no recomputation): (a) **40 hand-crafted** acoustic+linguistic descriptors (F0, jitter, shimmer, HNR, MFCC statistics, speech-rate, pause ratio, type-token ratio, filler ratio, repetitions); (b) a **7-dim dysphonia** subset (Little/Tsanas); (c) a **15-dim DisVoice-style** vector (Vásquez); (d) **512-dim log-mel MFCC sequences** for the GMM; (e) **80×600 log-mel spectrograms**. Frozen deep embeddings: **Whisper-base encoder** (512-d, mean-pooled), **Wav2Vec2-base** (768-d), **BERT-base [CLS]** (768-d), and **ImageNet ResNet-18 / MobileNetV2** as frozen spectrogram feature extractors (512-/1280-d). All scalers are fit on the training fold only.

### 4.3 Models (Fig. architecture diagrams)
- **A. Classical ML** on tabular features: SVM-RBF, RandomForest, XGBoost, LogisticRegression, KNN.
- **B. Prior-work re-implementations:** Little (dysphonia+SVM), Tsanas (LASSO selection+RF), Vásquez (DisVoice→MLP), Moro (per-class GMM likelihood-ratio on MFCC sequences), NeuroVoz-style mel-CNN.
- **C. CNN feature extractors:** frozen ResNet-18 / MobileNetV2 embeddings + MLP head.
- **D. Self-supervised:** Wav2Vec2 embedding + MLP head.
- **E. Proposed fusion:** Whisper (512) + BERT (768) + hand-crafted (40) with three strategies — `concat`, `cross-attention` (modalities attend to each other), and `gated` (learned soft gate). The head is a dropout-regularised MLP with a sigmoid output.

### 4.4 Training and evaluation protocol
Deep heads use AdamW (head LR 1e-3), class-weighted BCE, dropout 0.3, up to 40 epochs with **early stopping on a speaker-wise inner-validation AUC** (patience 8), and best-checkpoint restore. Every model is evaluated under **Leave-One-Subject-Out** cross-validation (22 folds) — chosen over fixed k-fold because n = 22 and one speaker dominates — with an explicit `train ∩ test = ∅` speaker assertion, a train-only `StandardScaler`, and **three random seeds**. Because each LOSO fold contains a single (single-class) speaker, AUC, sensitivity and specificity are computed on **pooled out-of-fold predictions per seed**; per-fold accuracy is retained for the Friedman test. Metrics are reported at both the **utterance** and **subject** level (probabilities averaged per speaker).

### 4.5 Confound audit (closed loop)
Three probes are evaluated under speaker-grouped CV: (1) **metadata-only** logistic regression on {sample rate, duration, channels}; (2) a **spectral-fingerprint** classifier (centroid, roll-off, bandwidth, flatness, ZCR, >6 kHz energy ratio) on resampled-only audio; and (3) the same spectral classifier **after** band-limiting. The audit passes only if probe (3) falls to chance (0.5 ± 0.10); otherwise a red flag is raised and downstream results must be interpreted relative to these baselines.

### 4.6 Statistics and cost ablation
Per-utterance predictions are saved for every model. We run **pairwise McNemar** with Holm–Bonferroni correction (with an explicit non-independence caveat) and a **Friedman** omnibus on per-fold accuracy with a Nemenyi critical difference. A config-driven **feature/architecture-size sweep** trains MelCNN variants differing in n_mels/n_frames/pooling/padding/depth, grouped by input cache-key, and profiles parameters, FLOPs (when available), inference latency (warm-up + median), and output dimension.

---

## 5. Results

### 5.1 Multi-model comparison (RQ1)
Table 2 reports all sixteen models under LOSO (3 seeds, pooled metrics). A **logistic regression on 40 hand-crafted features is the top model** (accuracy 0.948, AUC 0.989), matched at the subject level (1.000) by several models. Deep encoders do not win decisively: MobileNetV2 (0.977 AUC) and the Whisper+BERT `concat` fusion (0.953 AUC) are strong but not separated from the best classical model beyond noise. The self-supervised Wav2Vec2 head (0.854) and the from-scratch NeuroVoz-style mel-CNN (0.886) trail the frozen-embedding approaches, consistent with the small sample size.

**Table 2. Sixteen-model comparison (LOSO, mean ± std over 3 seeds; pooled utterance-level unless noted).**

| Model | Family | Acc (±std) | AUC (±std) | F1 | Sens | Spec | Subj-Acc |
|---|---|---|---|---|---|---|---|
| logreg | ML | 0.948 ± 0.000 | 0.989 ± 0.000 | 0.950 | 0.938 | 0.959 | 1.000 |
| mobilenetv2 | CNN | 0.924 ± 0.020 | 0.977 ± 0.010 | 0.925 | 0.892 | 0.960 | 0.970 |
| resnet18 | CNN | 0.865 ± 0.001 | 0.962 ± 0.004 | 0.861 | 0.791 | 0.949 | 1.000 |
| vasquez2018 | Prior | 0.874 ± 0.013 | 0.957 ± 0.007 | 0.874 | 0.824 | 0.931 | 1.000 |
| fusion_concat | Proposed | 0.839 ± 0.046 | 0.953 ± 0.020 | 0.826 | 0.733 | 0.959 | 0.955 |
| svm_rbf | ML | 0.875 ± 0.002 | 0.945 ± 0.001 | 0.874 | 0.821 | 0.936 | 1.000 |
| xgboost | ML | 0.852 ± 0.000 | 0.929 ± 0.001 | 0.852 | 0.804 | 0.906 | 1.000 |
| random_forest | ML | 0.857 ± 0.001 | 0.925 ± 0.002 | 0.857 | 0.814 | 0.905 | 1.000 |
| fusion_xattn | Proposed | 0.821 ± 0.024 | 0.920 ± 0.022 | 0.815 | 0.746 | 0.904 | 0.939 |
| fusion_gated | Proposed | 0.776 ± 0.046 | 0.889 ± 0.035 | 0.771 | 0.713 | 0.847 | 0.939 |
| neurovoz2024 | Prior | 0.742 ± 0.065 | 0.886 ± 0.034 | 0.707 | 0.615 | 0.886 | 0.864 |
| wav2vec2 | SSL | 0.777 ± 0.039 | 0.854 ± 0.037 | 0.775 | 0.734 | 0.826 | 0.955 |
| little2009 | Prior | 0.732 ± 0.004 | 0.828 ± 0.003 | 0.732 | 0.689 | 0.781 | 0.818 |
| tsanas2012 | Prior | 0.729 ± 0.002 | 0.819 ± 0.002 | 0.725 | 0.675 | 0.789 | 0.985 |
| knn | ML | 0.717 ± 0.000 | 0.808 ± 0.000 | 0.689 | 0.593 | 0.856 | 0.955 |
| moro2019 | Prior | 0.742 ± 0.006 | 0.798 ± 0.005 | 0.727 | 0.649 | 0.845 | 0.955 |

Seed dispersion is small for the ML/prior models (deterministic or near-deterministic) and larger for the small deep heads (e.g., fusion_gated ±0.046 acc, neurovoz2024 ±0.065 acc), consistent with initialization variance under a 22-speaker regime.

### 5.2 Statistical tests
Pairwise **McNemar** on pooled utterances yields **120 pairs, of which 81 remain significant after Holm–Bonferroni**. However, utterances from one speaker are not independent, so McNemar is anti-conservative here; we therefore also report a fold-level test. The **Friedman** omnibus on per-fold accuracy across the **22 LOSO folds** is highly significant (**χ² = 54.82, p < 10⁻⁵**), with a Nemenyi critical difference of **4.76** average-rank units. Both levels agree: a broad top tier (logreg, MobileNetV2, SVM, XGBoost, Vásquez) is not internally separable, while the weakest prior-work baselines (Little, Tsanas, Moro, KNN) rank significantly lower.

### 5.3 Confound audit (RQ2, RQ3)
Table 3 is the central result. A classifier using **only the recording sample rate** achieves **100%** accuracy — the two cohorts are perfectly separable from metadata alone, so any model with access to acquisition cues can shortcut the task. A spectral-fingerprint classifier reaches **98.6%** on resampled-only audio and remains **98.4%** *after* 7.5 kHz band-limiting: band-limiting removes the trivial sample-rate ceiling but the recording-pipeline/speaking-style fingerprint persists. The audit therefore **fails its chance criterion (RED)**, and the model accuracies in Table 2 must be read relative to these baselines rather than against a 50% chance line.

**Table 3. Confound audit (speaker-grouped CV accuracy).**

| Probe | n | Accuracy | Chance-clean? |
|---|---|---|---|
| Metadata-only (sample rate, duration, channels) | 1091 | 1.000 | No |
| Spectral fingerprint, resample-only | 1091 | 0.986 | No |
| Spectral fingerprint, after band-limiting | 1091 | 0.984 | No |

### 5.4 Modality ablation
To isolate each branch of the proposed model, the **same** light MLP head is trained on each subset of the cached branch embeddings under the identical LOSO protocol, so differences reflect the feature subset, not the head (Table 4). The **Whisper audio branch dominates** (audio-only: 0.898 acc, 0.979 AUC), far above text-only (0.701, 0.766) and hand-crafted-only (0.733, 0.802) — consistent with PD primarily altering the *acoustic* realisation of speech rather than lexical content. Crucially, under this uniform head, **adding the text or hand-crafted branch does not help**: audio+text (0.955 AUC) and the full three-branch combination (0.877 AUC) do not exceed audio-only (0.979). On this small, confounded corpus the extra branches add parameters without complementary signal — the same low-dimensional, confound-dominated regime identified in §5.3, and the reason the heavier fusion strategies in Table 2 do not separate from the simplest models.

**Table 4. Modality ablation (uniform MLP head, LOSO, 3 seeds).**

| Configuration | Features | Acc (±std) | AUC | F1 | Sens | Spec | Subj-Acc |
|---|---|---|---|---|---|---|---|
| Audio only (Whisper) | whisper_emb | **0.898 ± 0.010** | **0.979** | 0.896 | 0.834 | 0.969 | 0.985 |
| Text only (BERT) | bert_emb | 0.701 ± 0.040 | 0.766 | 0.719 | 0.723 | 0.676 | 0.788 |
| Hand-crafted only | tabular | 0.733 ± 0.032 | 0.802 | 0.734 | 0.694 | 0.776 | 0.818 |
| Audio + Text | whisper+bert | 0.859 ± 0.046 | 0.955 | 0.854 | 0.789 | 0.939 | 0.985 |
| Audio + Hand-crafted | whisper+tabular | 0.775 ± 0.018 | 0.897 | 0.761 | 0.682 | 0.879 | 0.879 |
| Full (all three) | whisper+bert+tabular | 0.769 ± 0.045 | 0.877 | 0.754 | 0.670 | 0.881 | 0.894 |

> Note: this uniform-head "full" (0.877 AUC) differs from `fusion_concat` in Table 2 (0.953 AUC), which uses a projection-based FusionNet; the ablation deliberately fixes the head to isolate branch contribution.

### 5.5 Feature/architecture-size ablation
Table 5 sweeps four MelCNN variants. Increasing resolution from 40×200 to 80×300 lifts AUC from 0.644 to 0.841–0.951; a `valid`-padded, max-pooled 80×300 variant is best (accuracy 0.864, AUC 0.951) at 0.41 M parameters and 3.4 ms latency, dominating the deeper 80×600 variant on the accuracy–cost Pareto front. Pooling and padding choices move accuracy by up to ~0.20 at nearly constant parameter count, underlining that architecture, not only input size, matters.

**Table 5. Architecture/size sweep (full corpus).**

| Variant | n_mels×frames | Pooling | Padding | Acc | AUC | Params | Latency (ms) |
|---|---|---|---|---|---|---|---|
| mel40_f200_avgpool | 40×200 | avg | same | 0.545 | 0.644 | 105,953 | 1.52 |
| mel80_f300_attnpool | 80×300 | attention | same | 0.667 | 0.841 | 106,082 | 4.12 |
| mel80_f300_maxpool_valid | 80×300 | max | valid | 0.864 | 0.951 | 409,825 | 3.37 |
| mel80_f600_attnpool_deep | 80×600 | attention | same | 0.688 | 0.943 | 410,082 | 8.20 |

Figures (auto-generated): model-comparison bars, ROC overlays, confusion matrix, sample-rate distribution, confound-audit bars, and accuracy-vs-cost Pareto fronts are in `artifacts/figures/`; architecture diagrams (pipeline, Whisper block, three fusion strategies, LOSO protocol) are in `artifacts/figures/architecture_diagrams.md`.

---

## 6. Discussion

**Why classical ML matches deep models.** With only 22 speakers, high-capacity encoders cannot express their advantage, and a well-regularised logistic regression on interpretable descriptors is competitive — consistent with prior small-corpus PD findings [tsanas2012novel]. This is not evidence that deep representations are useless; it is evidence that *this dataset cannot adjudicate the question*.

**The dominant confound.** The audit shows the label is almost perfectly predictable from acquisition metadata (100%) and from spectral fingerprints even after band-limiting (98.4%). This is precisely the shortcut-learning regime [geirhos2020shortcut] and the dataset-shift pitfall catalogued for medical ML [roberts2021common]: the cohorts differ in sample rate *and* speaking style *and* recording pipeline, all correlated with the label. High subject-level accuracies (1.000 for several models) therefore likely reflect the model recognising *how* a recording was made, not *who* is ill. This reframes the apparent success of Whisper embeddings: their strong AUC is not proof of pathology sensitivity.

**Implication for the field.** The practical consequence is that **acquisition-matched validation is a prerequisite**, not an afterthought, for speech-biomarker claims. A confound audit with a chance criterion should be a mandatory reporting item; band-limiting alone is insufficient when the confound is multi-factorial. Cross-corpus evaluation (train here, test on NeuroVoz/PC-GITA) is the decisive next test and is wired into our pipeline.

**Positioning vs. prior work.** Our re-implementations (Table 1) reproduce prior *methods*, not their *numbers*, because corpus, task, and protocol differ; the gap between reported and reproduced AUCs is itself an argument against cross-paper comparison. The contribution is thus methodological — a common, leakage-free harness plus an audit — rather than a new state-of-the-art accuracy.

---

## 7. Limitations

(1) **Small, confounded cohort.** 22 speakers and a label-correlated acquisition pipeline preclude any clinical accuracy claim; numbers are benchmark artefacts, not deployment estimates. (2) **Speaker imbalance.** `PD_emma` contributes 62% of PD utterances; a drop-`emma` sensitivity analysis is supported by config but not yet reported. (3) **Statistical caveat.** McNemar on pooled utterances is anti-conservative under intra-speaker correlation; this is mitigated by the complementary 22-fold Friedman test, which agrees. (4) **Fallback front-ends.** DisVoice and openSMILE were unavailable, so the Vásquez baseline uses a hand-crafted DisVoice-style fallback; results should be labelled as re-implementations, not the authors' originals. (5) **No cross-corpus test yet.** External validation on NeuroVoz/PC-GITA is designed but not executed. (6) **No calibration/threshold tuning** in the reported runs.

---

## 8. Conclusion

On a single English PD/HC corpus we benchmarked sixteen models — including five re-implemented prior-work methods and a Whisper+BERT fusion — under one leakage-free, subject-wise protocol, and audited the result for confounds. A logistic regression on 40 features matched the best deep models, and no encoder won decisively. A metadata-only classifier reached 100% and a spectral classifier 98.4% after band-limiting, showing that measured accuracy substantially encodes acquisition rather than pathology. The study's value is a reproducible, confound-aware evaluation protocol and the evidence that acquisition-matched, cross-corpus validation is required before speech-based PD screening can be trusted. Future work will add drop-speaker and calibration analyses, integrate openSMILE/DisVoice, and run cross-corpus tests.

---

## References
(Keys correspond to `paper/references.bib`; expand to full entries in the final LaTeX.)

[bloem2021parkinson] Bloem et al., *Parkinson's disease*, The Lancet, 2021.
[ramig2008speech] Ramig et al., *Speech treatment in PD*, 2008.
[moro2021review] Moro-Velázquez et al., *Advances in PD speech analysis: a review*, 2021.
[rossi2021time] Rossi et al., *Time to diagnosis in PD*, 2021.
[tsanas2012novel] Tsanas et al., *Novel speech signal processing algorithms for PD classification*, IEEE TBME, 2012.
[little2009suitability] Little et al., *Suitability of dysphonia measurements for telemonitoring of PD*, IEEE TBME, 2009.
[vasquez2018multimodal] Vásquez-Correa et al., *Multimodal assessment of PD dysarthria*, 2018.
[radford2023whisper] Radford et al., *Robust speech recognition via large-scale weak supervision (Whisper)*, ICML, 2023.
[devlin2019bert] Devlin et al., *BERT*, NAACL, 2019.
[mendes2024neurovoz] Mendes-Laureano et al., *NeuroVoz corpus*, 2024.
[ibarra2023domain] Ibarra et al., *Domain mismatch in PD speech*, 2023.
[moro2019phonetic] Moro-Velázquez et al., *Phonetic relevance and phonemic grouping for PD detection*, 2019.
[geirhos2020shortcut] Geirhos et al., *Shortcut learning in deep neural networks*, Nature Machine Intelligence, 2020.
[roberts2021common] Roberts et al., *Common pitfalls in ML for COVID-19 imaging*, Nature Machine Intelligence, 2021.
[baevski2020wav2vec] Baevski et al., *wav2vec 2.0*, NeurIPS, 2020.
[hlavnicka2017automated] Hlavnička et al., *Automated analysis of hypokinetic dysarthria*, 2017.
[hsu2021hubert] Hsu et al., *HuBERT*, 2021.
[reimers2019sentence] Reimers & Gurevych, *Sentence-BERT*, EMNLP, 2019.
[orozco2014new] Orozco-Arroyave et al., *New Spanish speech corpus for PD (PC-GITA)*, LREC, 2014.
