# Reviewer Report (Q1/SCIE-style) on `docs/BAO_CAO.md`

**Field:** speech-based biomarkers for neurodegenerative disease / medical machine learning.
**Role:** strict pre-submission reviewer. This report follows the 15-point rubric supplied by the author,
critiques the current manuscript, and marks which fixes are **already applied** vs **pending**.

---

## 1. Fit with a Q1/SCIE journal
- **Novelty:** The novelty is *not* a new accuracy record (correctly acknowledged) but a **confound-aware, leakage-free harness that re-executes five prior methods** on one corpus plus a **closed-loop audit**. This is a legitimate, publishable methodological contribution for a methods/repro-focused venue (e.g., *IEEE JBHI*, *Computer Speech & Language*, *PLOS Digital Health*), **but is at risk of being read as "negative result + small dataset"** at a high-impact clinical venue.
- **Verdict:** Adequate fit for a methods/reproducibility or biomedical-informatics journal; **weak fit** for a top clinical journal without external validation.

## 2. Title, Abstract, Keywords
- Title is descriptive and honest but long; acceptable. Abstract contains background, gap, objective, method, key numbers, and contribution — **complete**. Keywords are searchable.
- **Minor:** the abstract could state n = 22 speakers *earlier* to preempt "too small" reactions, and name the target of the audit (sample rate) in the first two sentences. **Applied** (abstract already leads with the metadata-only 100% result).

## 3. Introduction
- Motivation, gap, RQs, and explicit contributions are present and well-sequenced. The two-problem framing (non-comparability + confounds) is strong.
- **Major (pending):** the introduction claims the "largest uniformly-evaluated field on this corpus" — this must be softened unless the prior paper's field is cited head-to-head. **Action:** hedge to "one of the largest" and cite the earlier study.

## 4. Related Work
- Grouped, critical, and ends with an explicit gap + reproduction table — good. Ties to shortcut learning / dataset shift are appropriate.
- **Minor:** add 1–2 *recent* (2023–2024) speech-foundation-model PD papers to demonstrate currency; currently the newest transformer cites are Whisper/HuBERT.

## 5. Methodology
- Preprocessing, features, models, protocol, audit, and stats are described in reproducible detail; LOSO + train-only scaler + pooled-AUC handling is correct and clearly justified.
- **Major (pending) validity threats a Q1 reviewer will raise:**
  (a) **DisVoice/openSMILE fallback** weakens the Vásquez reproduction — must be labelled clearly (done in Limitations) and ideally replaced with the real library.
  (b) **No calibration** and **no drop-`emma`** run — both are supported by config but unreported.
  (c) **Single band-limit cutoff** — the audit would be stronger as a *sweep* over cutoffs showing the accuracy floor.
- **Applied:** leakage assertion, seed control, per-utterance logging, cost profiling.

## 6. Results
- Tables are clear and every number is traceable to a CSV. The confound table is the centrepiece and is well-placed.
- **Major:** report **mean ± std across seeds** in Table 2 (currently point estimates); Q1 reviewers expect dispersion. **Action:** add ± std columns (available in `model_comparison.csv`).
- **Minor:** state the operating threshold (0.5) and whether it was tuned (it was not).

## 7. Discussion
- Goes beyond restating results: explains *why* classical ML matches deep models and *why* the confound dominates, with field-level implications. Strong.
- **Minor:** add one paragraph contrasting subject-level (1.000) vs utterance-level accuracy to explain the gap explicitly.

## 8. Conclusion
- Summarises contributions, avoids overclaiming, and lists concrete future work. Well-calibrated to the evidence.

## 9. Language & Academic Style
- Generally concise and academic. A few long sentences in §1 and §6 should be split. No major grammar issues. **Action:** light copy-edit pass.

## 10. Structure & Coherence
- Standard IMRaD+ structure with a dedicated Data/Confound section — coherent. Figures/diagrams are referenced but should be **embedded** in the camera-ready.

## 11. References & Citation Quality
- 19 references, used to build arguments (not padding). Foundational (Little, Tsanas, Whisper, Geirhos, Roberts) present.
- **Major:** add 2–4 **2023–2024** speech-foundation-model PD studies; current recency is thin. Ensure every empirical claim in §5 cites a table/figure (mostly done).

## 12. Risk of Desk Rejection — **MODERATE**
- **Why not Low:** small, heavily confounded single corpus; the headline is a negative/repositioning result; no external validation; some baselines use fallbacks.
- **Why not High:** the work is transparent, reproducible, methodologically rigorous, and the confound audit is a genuine, timely contribution; the framing already pre-empts the obvious objections.
- **Decisive levers to reach Low:** (i) add cross-corpus validation, (ii) report seed dispersion + drop-`emma`, (iii) replace fallback front-ends.

## 13. Action Plan Before Submission

| Problem | Severity | Concrete fix | Section |
|---|---|---|---|
| No cross-corpus (external) validation | **Critical** | Train here → test NeuroVoz/PC-GITA; report AUC drop | §5, §6, new §5.5 |
| Point estimates without seed dispersion | **Major** | Add mean±std (already in CSV) to Table 2 | §5.1 |
| DisVoice/openSMILE fallback | **Major** | Install libraries; rerun Vásquez/eGeMAPS; relabel | §2, §4.2, Table 1 |
| Drop-`emma` sensitivity not reported | **Major** | Run `eval.drop_emma=true`; add a paragraph | §5, §7 |
| Single band-limit cutoff in audit | **Major** | Sweep cutoffs (8/6/4 kHz); plot accuracy floor | §5.3 |
| No calibration/threshold analysis | Minor | Add Platt/threshold tuning on val; report shift | §4.4, §5 |
| "Largest field" overclaim | Minor | Hedge; cite the earlier in-house study head-to-head | §1 |
| Thin 2023–2024 citations | Minor | Add 2–4 recent foundation-model PD papers | §2, refs |
| Long sentences in §1/§6 | Minor | Copy-edit pass | §1, §6 |
| Figures referenced, not embedded | Minor | Embed PNGs + diagrams in camera-ready | all |

## 14. Reviewer-style Comments
**Summary.** The manuscript benchmarks sixteen models — including five re-implemented prior methods and a Whisper+BERT fusion — for PD detection on one English corpus under a leakage-free, subject-wise protocol, and audits the result for acquisition confounds. It finds classical ML competitive with deep encoders and shows the label is almost perfectly predictable from recording metadata, repositioning the contribution as a reproducible, confound-aware evaluation.

**Major comments.** (1) External validation is essential to convert a repositioning result into a robust claim. (2) Report seed dispersion and the drop-`emma` sensitivity. (3) Strengthen the audit into a cutoff sweep. (4) Replace fallback feature front-ends or clearly bound their impact.

**Minor comments.** Add recent citations; split long sentences; embed figures; state the fixed decision threshold; contrast subject- vs utterance-level accuracy.

**Recommendation:** **Major revision.** The core is sound, transparent, and timely; acceptance hinges on external validation and dispersion reporting.

## 15. Priority Revision Checklist (top 10)
1. Add **cross-corpus** train→test (NeuroVoz/PC-GITA); report the AUC drop. *(pending — pipeline wired)*
2. Add **mean ± std over 3 seeds** to Table 2. *(quick — data already in CSV)*
3. Run and report **drop-`emma`** sensitivity. *(quick — config flag exists)*
4. **Sweep band-limit cutoffs** in the audit; plot the accuracy floor. *(medium)*
5. Install and rerun **DisVoice / openSMILE**; relabel the Vásquez baseline. *(medium)*
6. Add **calibration/threshold** tuning and report the sensitivity gain. *(medium)*
7. Add **2–4 recent (2023–2024)** foundation-model PD citations. *(quick)*
8. **Embed** figures and architecture diagrams into the manuscript. *(quick)*
9. **Copy-edit** long sentences; standardise tense to past for methods/results. *(quick)*
10. **Hedge** the "largest field" claim; cite the earlier in-house study head-to-head. *(quick)*

---

### Status note
Items 2, 3, 7, 8, 9, 10 are low-effort and can be closed before submission using existing artifacts; items 1, 4, 5, 6 require additional runs (external datasets / library installs / extra experiments) and are honestly disclosed in §7 (Limitations) of the current draft.
