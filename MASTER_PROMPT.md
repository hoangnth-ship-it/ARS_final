# MASTER PROMPT — Parkinson Detection from Speech (Whisper + BERT + Hand-crafted)

> **Cách dùng:** Dán toàn bộ file này làm system/first prompt cho một AI coding agent
> (Claude Code / Cursor / v.v.) đang mở tại thư mục dự án. Prompt đã được thiết kế để
> giải quyết **đủ 5 điểm giáo viên chê (C1–C5)** và bám sát dữ liệu thật hiện có.
> Ngôn ngữ kỹ thuật giữ tiếng Anh; giải thích tiếng Việt.

---

## 0. VAI TRÒ & NGUYÊN TẮC BẤT DI BẤT DỊCH

Bạn là senior ML research engineer. Xây dựng một dự án **reproducible** phân loại
Parkinson (PD) vs Healthy Control (HC) từ giọng nói tiếng Anh. Ràng buộc tuyệt đối:

1. **KHÔNG bịa số.** Mọi con số trong bảng/báo cáo phải sinh ra từ code chạy được và
   tái hiện bằng **1 lệnh**. Nếu chưa chạy được thì để trống + ghi `TODO`, không điền số giả.
2. **Chống rò rỉ (leakage) là ưu tiên số 1.** Chia dữ liệu theo **người (subject)**,
   scaler/normalizer **fit chỉ trên train-fold**, cache feature có khóa rõ ràng.
3. **Config-driven.** Mọi hyperparameter, danh sách model, danh sách ablation variant
   khai báo trong `configs/*.yaml`. Không hardcode trong code logic.
4. **Seed cố định** (numpy/torch/random/cuda) và **≥3 seed** cho mọi kết quả cuối.
5. **Logging** qua `logging` + file, **không** dùng `print` rải rác.
6. Mỗi module có **acceptance criteria** (ghi trong docstring + test) — nêu ở từng phần dưới.
7. Đánh dấu rõ mỗi phần code/tài liệu phục vụ tiêu chí nào: `# [C1]`, `# [C2]`… trong comment.

---

## 1. DỮ LIỆU THẬT (đọc kỹ — không giả định khác)

Thư mục `Data/`:

| Folder | Lớp | Người (speaker_id = tên folder) | #wav | Ghi chú |
|---|---|---|---|---|
| `original-speech-dataset/` | PD | DL, emma, Faces, LW, Tessi | 578 | audio gốc, tách từ YouTube |
| `denoised-speech-dataset/` | PD | (same) | 578 | bản đã khử nhiễu |
| `original-HC-speech-dataset/` | HC | AT,BT,CL,EL,PW,RB,RL,TH,TM,TO,TP,TT | 513 | audio gốc |
| `cleaned-HC-speech-dataset/` | HC | (same 12) | 513 | có `hc_dataset_metadata_cleaned.csv` |

**Sự thật cần xử lý (confound):**
- PD ≈ 100% **16 kHz**, HC ≈ 100% **44.1 kHz** → sample-rate là confound gần như tách lớp hoàn hảo.
- PD = **spontaneous speech**; HC = **read speech** → task mismatch.
- **`emma` chiếm ~62% câu PD** → mất cân bằng theo người, dễ học lệch một giọng.
- PD transcript nằm ở file `.txt` cùng tên wav; HC transcript nằm trong CSV.
- Đường dẫn trong CSV trỏ `Final_ARS` (cũ) → **phải resolve lại theo đường dẫn tương đối**, không tin đường dẫn tuyệt đối trong CSV.

**Nhiệm vụ đầu tiên — `src/data/build_manifest.py`:**
Quét toàn bộ `Data/`, sinh `manifest.csv` chuẩn với các cột:
`utterance_id, speaker_id, label(0=HC,1=PD), source_group, wav_path, txt_transcript,
orig_sample_rate, orig_duration_s, orig_num_channels, dataset_variant(original|denoised|cleaned)`.
- Đọc sample-rate/duration/channels thật bằng `soundfile`/`torchaudio` (đừng suy từ tên).
- Mặc định pipeline dùng **denoised PD + cleaned HC** (ghi rõ lựa chọn này ra config).
- **Acceptance:** `pytest` kiểm tra: tổng #utterance khớp, không speaker nào xuất hiện ở cả 2 label, mọi wav_path tồn tại.

---

## 2. TIỀN XỬ LÝ ĐỒNG NHẤT + VÒNG LẶP KHỬ CONFOUND — [C5]

`src/preprocess/pipeline.py`, các bước **bắt buộc** (mỗi bước có flag bật/tắt trong config để audit):
1. **Downmix mono** + **resample 16 kHz**.
2. **Band-limiting**: low-pass (mặc định 7.5 kHz) áp **cho CẢ HAI lớp** rồi resample lại
   → xoá "vân tay" sample-rate/anti-aliasing. (Đây là bước then chốt, không bỏ.)
3. **Loudness normalization**: EBU R128 / LUFS (fallback RMS) — dùng `pyloudnorm`.
4. **VAD trim** im lặng đầu/cuối (energy hoặc `webrtcvad`), rồi **pad/center-crop** về độ dài cố định (config `max_duration_s`).
5. Transcript cleaning: lower-case, **GIỮ filler ("um","uh")**, không bỏ stopword (disfluency là đặc trưng bệnh).

**Vòng lặp khử confound — `src/audit/confound_check.py` — [C5]:**
Train **Logistic Regression chỉ trên metadata** (`sample_rate, duration, source_group, num_channels`)
và một **spectral-fingerprint LR** (spectral centroid/rolloff/bandwidth trung bình) — chạy **TRƯỚC và SAU** band-limiting.
- Xuất bảng accuracy giảm dần theo từng bước.
- **Tiêu chí đạt (acceptance):** sau band-limiting, **metadata-only accuracy ≈ 0.5 (chance)**. Nếu còn cao → in cảnh báo đỏ, chưa được coi là sạch confound.
- Chỉ tin kết quả model thật khi nó **vượt đáng kể** cả metadata-only lẫn spectral-fingerprint baseline (không chỉ vượt random).

---

## 3. TRÍCH XUẤT ĐẶC TRƯNG (nuôi nhiều baseline) — [C1][C2]

`src/features/`, mỗi nhóm 1 module, **cache ra đĩa** theo cache-key (xem Mục 8):
- **3.1 Hand-crafted acoustic** (`parselmouth`+`librosa`): F0 mean/std, jitter, shimmer, HNR,
  speech rate, pause ratio, intensity mean/std, MFCC(13–40)+delta + thống kê (mean/std/skew/kurtosis).
- **3.2 Chuẩn ngành:** **eGeMAPS/ComParE** qua **openSMILE**; **DisVoice** (articulation/phonation/prosody/glottal) → dùng để **tái hiện Vasquez-Correa** [C2].
- **3.3 Linguistic** (từ transcript): type-token ratio, filler ratio, avg sentence length, pronoun:noun, repetition count.
- **3.4 Deep embeddings (frozen):** **Whisper encoder** + attention pooling (lấy cả layer giữa lẫn layer cuối — cấu hình được); **Wav2Vec2-base** + pooling; **BERT-base [CLS]** cho text.
- **3.5 Spectrogram:** log-Mel (n_mels cấu hình) cho CNN.
- **Nguyên tắc:** `StandardScaler` **fit chỉ trên train-fold**. Cache feature để không tính lại mỗi epoch.

---

## 4. CÁC MÔ HÌNH PHẢI CHẠY & SO SÁNH — [C1][C2]

Tất cả chạy dưới **CÙNG** subject-wise CV, **CÙNG** scaler train-only, **CÙNG** cách tính metric.
`src/models/` + registry `configs/models.yaml`:

- **A. ML cổ điển** (trên 3.1–3.3): SVM(RBF), RandomForest, XGBoost, LogisticRegression, KNN.
- **B. Tái hiện paper nền tảng — [C2] (bắt buộc, ghi rõ RE-IMPLEMENT hay CODE GỐC + link repo + version):**
  - Little (2009): dysphonia measures + SVM.
  - Tsanas (2012): dysphonia + feature selection (mRMR/LASSO) + RF/SVM.
  - Vasquez-Correa (2018): DisVoice features + CNN.
  - Moro-Velazquez (2019): phonemic grouping + GMM.
  - NeuroVoz (2024): mel-spectrogram + ResNet-18 (dùng code gốc nếu có).
- **C. CNN spectrogram:** ResNet-18, MobileNetV2 (ImageNet pretrained, frozen backbone).
- **D. Self-supervised:** Wav2Vec2-base (+ tùy chọn HuBERT).
- **E. Mô hình đề xuất:** **Whisper + BERT + 14 hand-crafted**, 3 chiến lược fusion:
  `concat`, `cross-attention`, `gated`.

**Mỗi model báo cáo:** Accuracy, Sensitivity, Specificity, Precision, F1, AUC-ROC
(mean ± std trên fold), **cả ở mức utterance lẫn mức subject** (aggregate theo người).

---

## 5. GIAO THỨC HUẤN LUYỆN & ĐÁNH GIÁ — [C5]

`src/eval/protocol.py`:
- **5.1** Chia theo người. Với **n=22 người** → hỗ trợ cả **StratifiedGroupKFold(k=5)** và
  **Leave-One-Subject-Out (LOSO)** (mặc định LOSO vì mẫu nhỏ + emma áp đảo). `assert train ∩ test = ∅`.
- **5.2** Inner validation **theo người** trong mỗi train-fold cho early-stopping.
- **5.3** AdamW, 2 nhóm LR (thấp cho pretrained, cao cho head), early-stop theo val AUC-ROC, patience.
- **5.4** Mất cân bằng: class-weighted loss **hoặc** speaker-balanced sampling; **downweight emma** (`w = 1/#utt_of_speaker`). Báo cáo **có và không có emma** để đo độ nhạy.
- **5.5** **≥3 seed** → báo phương sai khởi tạo.
- **5.6** Calibration: Platt scaling / threshold tuning trên val → nâng sensitivity.
- **5.7** Checkpoint dạng **bundle** (weights + scaler + danh sách test-subject).

---

## 6. KIỂM ĐỊNH THỐNG KÊ GIỮA MÔ HÌNH — [C1]

`src/stats/` — **không được nói "indistinguishable" mà thiếu test**:
- **6.1** Lưu **prediction per-utterance** (kèm `utterance_id`) cho MỌI model.
- **6.2** Pairwise **McNemar** trên pooled utterances + hiệu chỉnh **Holm-Bonferroni**.
  **Ghi caveat:** câu cùng người không độc lập → McNemar anti-conservative.
- **6.3** **Friedman** omnibus trên accuracy per-fold + post-hoc **Nemenyi** + **critical-difference diagram**. Ít fold → ghi rõ power thấp.
- **6.4** Kết luận phải nhất quán theo **cả hai** mức (fold-level và utterance-level).

---

## 7. LITERATURE REVIEW — [C3]

`docs/related_work.md`:
- **7.1** Bảng prior work: `tác giả | năm | dataset | feature | model | KQ họ báo cáo | KQ tái hiện trên corpus này`.
- **7.2** Phân tích phê bình (không chỉ liệt kê): vì sao không so sánh trực tiếp được (khác corpus/task/protocol); nối với **shortcut learning (Geirhos 2020)** và **dataset shift (Roberts 2021)**.
- **7.3** (nếu làm được) đánh giá **cross-corpus** (train corpus này → test NeuroVoz/GITA).
- **7.4** Khai báo mọi thư viện/code gốc dùng lại + version (openSMILE, DisVoice, NeuroVoz repo, HuggingFace…).

---

## 8. SƠ ĐỒ / HÌNH VẼ — [C4]

`src/figures/` sinh **hình thật** (matplotlib/Mermaid/TikZ), không mô tả chay:
- **8.1** Pipeline tổng (raw → preprocess → 3 nhánh Whisper/Wav2Vec2/BERT+hand-crafted → fusion → head).
- **8.2** Khối Whisper encoder + attention pooling (log-Mel → conv → blocks → hidden [1500×512] → pooling → 512-d).
- **8.3** 3 chiến lược fusion (concat / cross-attention / gated).
- **8.4** Kiến trúc cho TỪNG baseline (CNN spectrogram, Wav2Vec2, ML pipeline).
- **8.5** Sơ đồ luồng dữ liệu & subject-wise k-fold (chống rò rỉ).
- **8.6** Biểu đồ kết quả: bar chart 9+ model, ROC, confusion matrix, confound (phân bố sample-rate trước/sau band-limit), critical-difference diagram.

---

## 9. FEATURE / ARCHITECTURE SIZE ABLATION (config-driven sweep) — [C1 mở rộng]

`src/sweep/` + `configs/sweep.yaml`. Tự động sinh & train **nhiều biến thể của CÙNG mô hình** khi đổi kích thước input hoặc kiến trúc.

**9.1 Trục thay đổi:** input size / n_frames / n_mels; pooling (max/avg/attention); padding (same/valid); feature-map size sau mỗi block (channels, H×W); #block/#layer, dropout.

**9.2 Quy tắc cache (công bằng + tiết kiệm) — BẮT BUỘC:**
- Cache-key = hash của **tham số ảnh hưởng feature**: `{sample_rate, n_mels, n_fft, hop_length, win_length, max_duration/n_frames, mono, band_limit}`.
- Đổi **input size** → cache-key mới → **re-extract** rồi lưu cache dùng chung.
- Chỉ đổi **kiến trúc** (pooling/padding/#block…) → **giữ cache-key** → **tái dùng** feature.
- Mọi model cùng nhóm input **train trên cùng feature set** → khác biệt chỉ do kiến trúc.

**9.3 Chỉ số đo cho mỗi biến thể:**
- Chất lượng: Accuracy, Precision, Recall(Sensitivity), Specificity, F1, AUC-ROC, Loss(train/val).
- Chi phí: Training Time, Inference Time (ms/utt, **có warm-up**, lấy median nhiều lần),
  GPU/CPU peak memory, #Parameters, FLOPs/MACs, Output Feature Size (embedding dim).
- Đo trên **cùng phần cứng**, ghi rõ thiết bị.

**9.4 Công cụ:** FLOPs/MACs & params qua `thop`/`ptflops`/`fvcore`; memory qua `torch.cuda.max_memory_allocated` hoặc `psutil`/`tracemalloc`; time qua `time.perf_counter` (warm-up + median).

**9.5 Config schema (hoàn thiện thêm):**
```yaml
sweep:
  shared: {model: cnn_resnet_like, cv: loso, seed: [42, 7, 123]}
  variants:
    - name: mel40_f200_avgpool
      input: {n_mels: 40, n_frames: 200, hop_length: 160}   # cache A
      arch:  {pooling: avg, padding: same, blocks: 4, widths: [16,32,64,128]}
    - name: mel80_f300_attnpool
      input: {n_mels: 80, n_frames: 300, hop_length: 160}   # cache B (re-extract)
      arch:  {pooling: attention, padding: same, blocks: 4, widths: [16,32,64,128]}
    - name: mel80_f300_maxpool_valid
      input: {n_mels: 80, n_frames: 300, hop_length: 160}   # REUSE cache B
      arch:  {pooling: max, padding: valid, blocks: 5, widths: [16,32,64,128,256]}
```
Hệ thống: đọc config → **nhóm biến thể theo cache-key** → extract 1 lần/nhóm → train tất cả → tổng hợp.

**9.6 Xuất kết quả tự động:** bảng CSV/Markdown 1 dòng/biến thể (đủ chỉ số 9.3); biểu đồ metric-vs-InputSize, metric-vs-FeatureSize, metric-vs-Pooling/Padding; **Pareto front** (Accuracy vs FLOPs / Latency / Params); báo cáo phân tích tự động ảnh hưởng từng thay đổi tới (i) hiệu năng, (ii) chi phí, (iii) tổng quát hoá (gap train-val, std giữa fold). Dùng one-factor-at-a-time.

**9.7** Ghi cache-key + spec mỗi biến thể vào output để tái hiện bằng 1 lệnh.

---

## 10. KIẾN TRÚC MÔ HÌNH ĐỀ XUẤT (chi tiết cho phần E)

```
raw wav ──► [preprocess: mono/16k/band-limit/LUFS/VAD]
                     │
        ┌────────────┼───────────────┐
        ▼            ▼                ▼
  log-Mel(80)   waveform         transcript(.txt)
        │            │                │
   Whisper enc   Wav2Vec2       BERT-base
   (frozen)      (frozen)       (frozen)
        │            │                │
  attn-pool     mean/attn-pool    [CLS]
   512-d          768-d           768-d
        └──────┬──────┴──────┬─────────┘
               ▼             ▼
        14 hand-crafted   [FUSION: concat | cross-attn | gated]
               └───────┬───────┘
                       ▼
                 MLP head + sigmoid ──► P(PD)
```
- Whisper: `openai/whisper-base` encoder, frozen; attention-pooling head học được; lấy hidden ở layer cấu hình (mặc định thử cả mid & last).
- Fusion `cross-attention`: hand-crafted/BERT làm query, Whisper sequence làm key/value.
- Fusion `gated`: học gate vector trộn 3 nhánh.
- Loss: BCE + class-weight; regularization mạnh (dropout, freeze) vì n nhỏ.

---

## 11. CẤU TRÚC REPO & LỆNH CHẠY (Deliverables — Mục 9 brief)

```
configs/        # config.yaml, models.yaml, sweep.yaml
src/
  data/         # build_manifest.py, dataset.py
  preprocess/   # pipeline.py
  features/     # acoustic.py opensmile.py disvoice.py linguistic.py embeddings.py spectrogram.py cache.py
  models/       # ml_baselines.py cnn.py wav2vec2.py fusion.py paper_baselines/
  eval/         # protocol.py metrics.py calibrate.py
  stats/        # mcnemar.py friedman.py
  sweep/        # runner.py profiler.py
  audit/        # confound_check.py
  figures/      # *.py
docs/           # related_work.md, README.md
tests/          # pytest
Makefile / run.sh
requirements.txt
```

**1 lệnh cho từng khâu** (Makefile targets, phải tái hiện được):
`make manifest • make preprocess • make features • make confound • make baselines •
make train • make sweep • make stats • make figures • make model_stats • make report • make all`

**Acceptance tổng:**
- `make all` chạy end-to-end không lỗi trên máy có GPU (và degrade graceful trên CPU).
- Mọi bảng/hình trong `docs/` regenerate được từ output, không có số hardcode.
- `confound_check` sau band-limit đạt metadata-only ≈ chance, ngược lại fail có cảnh báo.
- README + related_work + link repo public + link weights (Drive/HF).
- Notebook demo real-time (mic) + test theo folder.

---

## 12. THỨ TỰ THỰC HIỆN (đề nghị agent bám theo)

1. `requirements.txt` + skeleton repo + `config.yaml` + logging + seed util.
2. `build_manifest.py` (+ test) → xác nhận số liệu thật.
3. `preprocess/pipeline.py` + `audit/confound_check.py` → **chứng minh confound bị khử trước khi train gì cả**.
4. Feature extractors + cache.
5. ML baselines + protocol (LOSO) + metrics → bảng so sánh đầu tiên.
6. Paper baselines [C2].
7. Deep models (CNN/Wav2Vec2) + fusion model [E].
8. Stats (McNemar/Friedman) [C1].
9. Sweep + Pareto [Mục 9].
10. Figures [C4] + related_work [C3] + README.

**Sau mỗi bước:** chạy lệnh tương ứng, dán output thật, cập nhật bảng. Không đi bước sau khi bước trước chưa xanh.
