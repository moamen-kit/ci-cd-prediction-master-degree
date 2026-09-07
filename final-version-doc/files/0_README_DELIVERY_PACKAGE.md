# 📦 MSc Thesis Defense Package

**Student:** Moamen Mohamed Aly Hussein (ID: 202401681)
**Thesis:** Predicting CI/CD Pipeline Build Failures Using Machine Learning Techniques
**Defense Date:** Friday, September 11, 2026

> **⚠️ This package supersedes the June version.** The evaluation protocol was
> corrected and every reported figure regenerated. The headline failure-class F1
> is now **0.4216**, not 0.5924. See `6_Corrections_Since_Submission.md` for
> what changed and why. Any document quoting 0.5924, $383,000 or "+27pp" as a
> current result is out of date.

---

## 🗂️ Files in This Package

| # | File | Purpose | Action Required |
|---|------|---------|----------------|
| 0 | `0_README_DELIVERY_PACKAGE.md` | This index | — |
| 1 | `1_MSc_Thesis_FINAL.docx` | **The thesis document** — 9 chapters, appendices, embedded figures | Replace `[Insert supervisor full name]`, then submit |
| 2 | `2_Defense_Presentation.pptx` | **26-slide defense deck** | Replace `[Insert Supervisor Name]` on title slide; open once to check text fits |
| 3 | `3_QA_Defense_Prep_Guide.md` | **Q&A preparation** — updated, several answers changed | Read & practice |
| 4 | `4_Thesis_Source_Markdown.md` | Markdown source — **the authoritative text** | The `.docx` is derived from this |
| 5 | `5_Manual_Insertion_Guide.md` | Instructions for adding remaining figures | Optional |
| 6 | `6_Corrections_Since_Submission.md` | **What changed since June, and why** | Read before the defense |
| 7 | `7_Reference_1_Verification_Note.md` | Unresolved citation for reference [1] | **Check against your PDF** |

**Also in the repository, outside this folder:**

| File | Purpose |
|---|---|
| `defence/DEFENCE_BRIEF.md` | Speaking aid — opens with a CI/CD-from-zero explanation |
| `defence/Dataset_Guide.xlsx` | Committee-readable dataset description — **show this, not the raw CSV** |
| `cicd-failure-prediction/README.md` | Source package overview |
| `cicd-failure-prediction/reproduce.sh` | One command, ~3 min, regenerates every reported number |
| `cicd-failure-prediction/results/README.md` | Which results file is authoritative for what |

---

## 📊 Headline Results

Commit-grouped five-fold cross-validation, decision threshold selected on a
validation fold and never on the test data:

| Configuration | Failure F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.4311 ± 0.0633 | 0.4092 | 0.8276 |
| Random Forest | 0.4240 ± 0.0812 | 0.3909 | 0.8008 |
| XGBoost *(selected on PR-AUC)* | 0.4216 ± 0.0638 | 0.4803 | 0.8240 |
| **Categorical only** | **0.4808 ± 0.0767** | **0.5467** | **0.8649** |

**Two things to know before presenting:**

1. The three classifiers are **not separable** — they differ by less than 0.01
   while the standard deviation across folds exceeds 0.06. Do not call any of
   them "the winner" on F1.
2. **Categorical-only beats them all.** A model told nothing but the repository,
   workflow, branch and trigger outperforms the full hybrid on every metric.
   This is the central finding: the system is a project-level risk estimator.

Business estimate: **$243,670** per year net of false alarms, with a break-even
false-alarm cost of **$20.41**.

---

## 📖 What's in the Thesis (`1_MSc_Thesis_FINAL.docx`)

**Chapters:**
1. Introduction (objectives + scope)
2. Problem Definition (stakeholders + as-is)
3. Existing Solution Approaches (prior work + comparison)
4. Proposed Solution (Hybrid Pipeline architecture)
5. System Analysis and Design (FRs/NFRs + use cases)
6. Implementation (module-level details)
7. Testing and Evaluation — **includes new Section 7.3.1** (evaluation protocol
   and the correction of an earlier design) and **new Section 7.4.5**
   (attribution of the previously reported result)
8. Discussion (achievements + limitations)
9. Conclusion (future work)

**Appendices A–E**, of which B (reproduction instructions) and C (complete
metrics tables) were rewritten. **24 IEEE-format references**, four of them
added to support the corrected methodology.

---

## 🎤 What's in the Presentation (`2_Defense_Presentation.pptx`)

**26 slides.** Four are new; ten carried figures that were corrected.

| # | Slide | Note |
|---|-------|------|
| 1 | Title | Replace supervisor name |
| 2 | Agenda | |
| 3 | **What Is CI/CD? And What Is GitHub?** | **NEW — do not skip this** |
| 4 | **Why This Problem Is Hard** | **NEW** — the 89/11 split |
| 5 | The Problem | |
| 6 | Literature Gap | |
| 7 | Research Objectives | |
| 8 | The Dataset | |
| 9 | Hybrid Pipeline Architecture | with diagram |
| 10 | Feature Engineering | |
| 11 | Evaluation Regime | rewritten for grouped CV |
| 12 | Results — Default Threshold | corrected table |
| 13 | Ablation Study | corrected chart |
| 14 | **The Discriminating Experiment** | **NEW** — categorical-only wins |
| 15 | Feature Importance | corrected chart |
| 16 | Threshold Optimization | corrected: +9.3pp, not +27 |
| 17 | Before/After Threshold | corrected chart |
| 18 | **Where The Original Number Went** | **NEW** — the attribution |
| 19 | Final Results | 0.422 |
| 20 | Business Impact | $243,670 + break-even |
| 21 | Objectives Achieved | Objective 3 now *partially* achieved |
| 22 | Key Contributions | rewritten |
| 23 | Honest Limitations | two limitations replaced |
| 24 | Future Work | |
| 25 | In Summary | rewritten |
| 26 | Thank You / Q&A | |

**Slide 3 is the most important slide in the deck.** The previous committee did
not know what CI/CD or GitHub are, and every number afterwards landed on
nothing. Practice it out loud.

---

## ✅ Before You Submit

- [ ] Replace `[Insert supervisor full name]` in the thesis
- [ ] Replace `[Insert Supervisor Name]` on the title slide
- [ ] Open the deck in PowerPoint and check no text overflows its box
- [ ] **Verify reference [1] (Patel) against the source PDF** — evidence
      suggests the venue is TIJER vol. 4 no. 11 (Nov 2017), not IJERT vol. 8
      no. 11 (2019) as cited. Neither could be confirmed from a primary source.
      See `7_Reference_1_Verification_Note.md` for exactly what to check.
- [ ] Read `6_Corrections_Since_Submission.md`
- [ ] Practice the CI/CD explanation (slide 3) three times out loud
