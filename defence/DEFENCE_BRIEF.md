# Defence Brief — CI/CD Build-Failure Prediction

**Not thesis prose. A speaking aid.** Written for the 2026-09-11 committee session.

Last committee session, three things went wrong: the code questions were not
prepared for, the dataset could not be read on screen, and — most importantly —
**the committee did not know what CI/CD or GitHub are.** This brief is ordered
to fix those in that order of importance.

---

## Part 0 — The single most important decision

**Assume the committee does not know what CI/CD is. Explain it before anything
else, without being asked, and without making it sound like an apology.**

If you open with "I built a hybrid ML pipeline for CI/CD failure prediction",
you have already lost the room. Every number after that lands on nothing.

Do not say "as you know" or "obviously". Do not ask "are you familiar with
GitHub?" — a committee will say yes to avoid losing face, and then follow
nothing. Just explain it as the natural first slide.

---

## Part 1 — Explaining the domain from zero (2 minutes, memorise this)

### The analogy to use

> Imagine that every time a student uploads a draft chapter, the university
> automatically runs a formatting check, a plagiarism check, and a reference
> check, and then emails back either "accepted" or "rejected". The student does
> not do this by hand. A machine does it, every single time, automatically.
>
> **GitHub** is the shared filing cabinet where all the drafts are kept. It
> records who changed what, and when.
>
> **A commit** is one saved change — one revision of the document, with a note
> from the author saying what they changed.
>
> **CI/CD** is that automatic checking machine. In software, it rebuilds the
> whole program and runs thousands of automated tests, every time any developer
> saves a change. If everything passes, the change is safe. If anything fails,
> a human has to stop and fix it.
>
> The problem is that this checking is slow and expensive. It can take minutes
> to hours, it costs real money in cloud computing, and when it fails, the
> developer has usually moved on to something else and has to come back.
>
> **My research asks: at the moment the change is saved, before the checking
> machine starts, can we predict whether it is going to fail?**

### The three sentences to follow it with

1. "I collected 9,772 real checks from 18 well-known open-source projects —
   React, PyTorch, Rust, Python itself — using GitHub's public API."
2. "Only 11 percent of them fail. That is what makes this hard: if I simply
   guess 'it will pass' every single time, I am right 89 percent of the time,
   and I have learned nothing. So accuracy is a useless measure here, and I
   report a measure called F1 instead, which only rewards catching the failures."
3. "The hard constraint I set myself is that I only use information that exists
   *before* the machine runs. That is what makes the prediction useful — a
   prediction made afterwards saves nobody anything."

### If someone asks "what is GitHub?" mid-talk

"It is the standard place where the world's software is stored and where
changes to it are recorded. Roughly 100 million developers use it. Think of it
as a very careful version-controlled archive with a full history of who changed
what."

### Vocabulary to avoid entirely

| Do not say | Say instead |
|---|---|
| repository | project / codebase |
| commit | a saved change |
| workflow run / build | one automatic check |
| pipeline | the automatic checking process |
| pre-execution features | information available before the check starts |
| class imbalance | only 11 percent of cases are failures |
| the positive class | the thing I am trying to detect: a failure |

---

## Part 2 — Dataset one-pager

Hand them `Dataset_Guide.xlsx`. Do not show the raw CSV on screen. It is
9,772 rows by 20 columns and it is unreadable by design — a spreadsheet of raw
API output is not a presentation artifact.

| | |
|---|---|
| Source | GitHub Actions REST API — `/actions/runs`, joined to `/commits/{sha}` |
| Collection script | `src/collect_github_data.py` |
| Projects | 18 (React, PyTorch, TensorFlow, Rust, CPython, Node, VS Code, Elasticsearch, …) |
| Cap | 600 most recent runs per project |
| Filter | kept only runs that finished, with outcome `success` or `failure` |
| Rows | **9,772 workflow runs** |
| Distinct commits | **2,835** — so 3.45 runs per commit on average, max 179 |
| Balance | 8,700 success (89.03%) / 1,072 failure (10.97%) |
| Time span | 2025-11-25 → 2026-05-29 (six months of run start times) |
| Columns | 20 raw; 16 engineered features actually used |

### The four things to volunteer before being asked

1. **`run_duration_sec` and `run_attempt` are excluded.** They are known only
   *after* the check finishes. A failing build's duration is a *consequence* of
   the failure, not a cause of it. Including them would inflate the result and
   make the system useless, because it could only predict the past.
2. **`files_changed` is capped at 300 by the GitHub API.** 110 rows (51 commits)
   report exactly 300 for what may be far more. Disclosed, not hidden.
3. **`author_association` is 100 percent empty.** A collection bug: the field
   was read from an endpoint that does not return it. The column is dropped.
4. **The rows are runs, but the features describe commits.** This is the single
   most important structural fact about the dataset, and it is what caused the
   methodological error described in Part 5.

---

## Part 3 — Code one-pager

Six modules matter. Be able to say what each one *owns* in one sentence.

| Module | What it owns |
|---|---|
| `collect_github_data.py` | Talks to the GitHub API. Produces the one raw CSV. Never run again. |
| `data_preparation.py` | Cleaning, feature engineering, text cleaning, and **all the train/test splitting**. |
| `hybrid_pipeline.py` | The four-branch feature transformer and the three classifier factories. |
| `cross_validation.py` | The commit-grouped cross-validation protocol and threshold selection. |
| `train_evaluate.py` | Metrics, the ablation study, and the business cost model. |
| `run_corrected_evaluation.py` | The orchestrator. Produces every number and figure in Chapter 7. |

### The four-branch transformer — the architecture question

One `ColumnTransformer` fuses four kinds of input into a single matrix:

| Branch | Input | Transformation | Why |
|---|---|---|---|
| Numerical | 5 size features | `StandardScaler` | Puts different units on a comparable scale |
| Categorical | repository, workflow, branch, event | `OneHotEncoder` | Converts names into 0/1 columns |
| Binary | 6 yes/no flags | passthrough | Already 0/1, nothing to do |
| Text | cleaned commit message | `TF-IDF` | Converts words into numbers by how distinctive they are |

### Why `LabelEncoderForBinary` exists

"My labels are the words `success` and `failure`, not the numbers 0 and 1.
Scikit-learn's two models accept words. XGBoost requires numbers. Rather than
convert the labels globally — which would have made it easy to lose track of
which number meant failure — I wrapped XGBoost in a thin adapter that converts
on the way in and converts back on the way out. All three models then have an
identical interface, so the evaluation code cannot accidentally treat one
differently from another."

---

## Part 4 — What came before, and what is different here

### The three prior lines of work

| Line | Representative | What they did | Limitation for this problem |
|---|---|---|---|
| Classical defect prediction | Hassan & Holt 2005 [13]; Kim et al. [15]; Zimmermann et al. [14] | Predict which *files* contain bugs, from change history and complexity | Works at file level, not at the level of "will this check fail" |
| Build-outcome prediction | **Hassan & Zhang 2006 [24]**; Patel 2019 [1] | Predict whether a build passes certification. Patel uses runtime telemetry — CPU, memory, durations, retries | **Uses information that only exists after the run.** Cannot save the cost of the run |
| NLP on commit text | TravisTorrent [2] and follow-ups | Use commit messages as an extra signal, via TF-IDF or embeddings | Gains reported are modest and vary by dataset |

### The four things this work does differently

1. **A strict pre-execution boundary.** Every feature is available at the moment
   the commit is saved. No durations, no retry counts, no runtime telemetry.
   This is the constraint that makes a prediction economically useful, and it is
   also what makes the numbers lower than the prior art — which is the honest
   trade and should be stated as such.
2. **Fresh GitHub Actions data.** The prior work leans on TravisTorrent, which
   covers Travis CI, a platform now largely displaced by GitHub Actions. This
   dataset was collected specifically for this work.
3. **A stronger evaluation protocol than the literature typically reports.**
   Commits are never split across train and test; the decision threshold is
   chosen on a validation set and never on the test set; results are
   cross-validated rather than taken from one lucky split.
4. **A published measurement of what weak protocols are worth.** This is the
   contribution the author is most confident in — see Part 5.

### If asked "is your result better than the prior work?"

**Do not claim it is.** Say this:

> "No, and I would be suspicious of anyone who claimed it. My headline F1 is
> 0.42, and the papers I compare against report 0.40 to 0.60. But those numbers
> are not comparable, for two reasons. First, most of them use information
> available only after the build ran, which I deliberately refuse to use.
> Second — and this is the part I measured — most published papers do not state
> whether they grouped their splits or where they chose their threshold. I
> measured what those two choices are worth on my own data: together, 0.171 of
> F1. That is larger than the entire spread between the published results. So
> comparing headline numbers across papers with undescribed protocols is not a
> comparison of models. My contribution is a number you can actually interpret."

---

## Part 5 — The results, in the order to present them

### Result 1 — the honest headline

Commit-grouped 5-fold cross-validation, threshold chosen on validation:

| Model | Failure F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.4311 ± 0.0633 | 0.4092 | 0.8276 |
| Random Forest | 0.4240 ± 0.0812 | 0.3909 | 0.8008 |
| XGBoost | 0.4216 ± 0.0638 | **0.4803** | 0.8240 |

**Say the second sentence out loud, do not let them find it:** "These three are
statistically indistinguishable. The gap between them is smaller than the
variation between folds. I select XGBoost on PR-AUC, not on F1, because
claiming a winner on F1 would not be supportable."

### Result 2 — the finding (lead with this if time is short)

| Configuration | Failure F1 | PR-AUC |
|---|---|---|
| Text only | 0.1962 | 0.2080 |
| Hybrid (everything) | 0.4216 | 0.4803 |
| Structured only | 0.4225 | 0.4825 |
| **Categorical only** | **0.4808** | **0.5467** |

> "A model that is told *only* which project, which job, which branch, and what
> triggered it — and is told nothing at all about the size of the change or what
> the developer wrote — beats my full model on every measure.
>
> This refutes the hypothesis I started with. The honest conclusion is that my
> system is largely a **project-level risk estimator**, not a commit-level one.
> It predicts that a build in a historically unreliable project is likely to
> fail. That is useful, but it is a weaker claim than predicting that a
> *particular change* will break the build, and I want to be the one who says so
> rather than have it discovered."

Why: the failure rate spans the **whole range from 0.0% to 38.3%** across the
18 projects — an **elevenfold** spread across the seventeen that fail at all. From 0.0% for
Elasticsearch (600 runs, zero failures) to 38.3% for Prisma.

### Result 3 — the methodological contribution

| Step | Protocol | F1 | Change |
|---|---|---|---|
| A | Row-level split, threshold picked on test | 0.5924 | — |
| B | + commits kept together across the split | 0.5326 | **−0.060** |
| C | + threshold picked on validation instead | 0.4065 | **−0.126** |
| D | + 5-fold cross-validation | 0.4216 | +0.015 |

> "An earlier version of this work reported 0.5924. I now report 0.4216 for the
> same model on the same data. The model did not get worse; the ruler got
> honest. I can attribute the difference exactly, because each correction can be
> applied on its own.
>
> The result I did not expect: choosing the threshold on the test set was worth
> more than twice the data leakage. The leakage is the error everyone looks for.
> The threshold error is the one that actually cost more."

Supporting citation: Kapoor & Narayanan [21] found leakage of this kind in 294
papers across 17 fields.

### Result 4 — business impact, stated defensively

$243,670/year estimated net saving. **Immediately follow with the honest
caveat**, because it is your strongest move:

> "That number depends on an assumption I did not measure — that a false alarm
> costs $2.50. So rather than defend the estimate, I report the break-even: the
> false-alarm cost at which each configuration stops paying for itself. For
> Logistic Regression it is $10.27, which is only about eight minutes of a
> developer's attention. For my selected model it is $20.41. The model with the
> highest estimated saving is the least robust one, and that is a better basis
> for choosing than the headline figure."

---

## Part 6 — Anticipated questions, with honest answers

**Q: How did you split the data, and why is that valid?**
> "Grouped on the commit. Every row is one automated check, but every feature I
> use describes the commit that triggered it, and one commit can trigger up to
> 179 checks. So if I split rows randomly, near-identical copies of the same
> commit land on both sides, and the model gets credit for recognising something
> it has already seen. I verify programmatically that no commit appears on both
> sides — it is in `results/split_integrity.json`."

**Q: Is your chronological evaluation a real temporal holdout?**
> "Yes, but a short-horizon one, and I want to be precise about the limitation.
> I cut each project at its own point in time, so for all 18 projects the test
> data comes strictly after the training data. But because I capped collection
> at 600 runs per project, a busy project like Rust only covers a few days.
> So I have demonstrated stability over hours to days, not over months. A global
> time cut would have been worse — it would have given a nine-hour test window
> covering only 11 of the 18 projects."

**Q: Where did threshold 0.06 come from?** *(the trap question — the old answer was wrong)*
> "In the earlier version, it came from the test set, which was a mistake: I
> chose the value that maximised the score, and then reported that score. That
> is circular. It now comes from a separate validation set inside the training
> data, and it averages 0.076. I measured what the old procedure was worth:
> 0.126 of F1."

**Q: Is the model just learning which repository is flaky?**
> "Substantially, yes — and I ran the experiment that proves it rather than
> waiting to be asked. A categorical-only model outperforms my full model. The
> failure rate runs from 0.0% to 38.3% across projects — elevenfold across the
> seventeen that fail at all — and Elasticsearch has 600 runs
> and zero failures, so project identity alone is a very strong prior. I report
> this as a finding rather than a failure, because it tells you what the useful
> deployment is: project-level and workflow-level triage, not per-commit advice."

**Q: Why 18 repositories, and why those?**
> "They are large, active, public projects across different languages and
> ecosystems — JavaScript, Python, Rust, C, Ruby, Java — chosen so the result is
> not an artifact of one language or one team's habits. They are public so the
> dataset is redistributable and the work is reproducible."

**Q: Why is your F1 only 0.42? That seems low.**
> "Three reasons, in order of size. First, I refuse post-execution information,
> which is the single biggest constraint. Second, I measure honestly — 0.171 of
> the figure I originally reported was protocol, not model. Third, the task is
> genuinely hard: commit metadata carries limited signal about whether a test
> suite will fail, and my ablation shows most of the available signal is
> project-level. I would rather defend 0.42 that is real than 0.59 that is not."

**Q: Why did you not use deep learning / BERT / an LLM?**
> "The ablation answers this. Text in isolation scores 0.196, and adding text to
> the structured features makes the model slightly *worse*, not better. A more
> sophisticated text encoder would be optimising the branch that carries the
> least signal. The bottleneck is what the features can know, not how they are
> encoded."

**Q: Your two temporal features — do they work?**
> "Honestly, they are close to noise as I defined them. `is_off_hours_commit`
> and `is_weekend_commit` are computed in UTC, across globally distributed
> projects. 18:00 UTC is mid-afternoon in California and midnight in Tokyo. To
> do this properly I would need each author's timezone, which the API does not
> reliably give. I report them because they are in the feature set, not because
> I think they are informative."

**Q: Is there any leakage left?**
> "One mild case that I disclose. The median thresholds, the category
> vocabularies, and the 693-word stoplist are computed over the whole dataset
> before splitting. That is technically fitting on data the model later tests
> on. It is mild because none of those quantities looks at the outcome column —
> they are target-independent — but the correct place for them is inside the
> pipeline, where they would see the training fold only. It is on the
> future-work list."

**Q: Can we run your code?**
> "Yes. One command, `./reproduce.sh`, from the raw data to every number and
> figure in Chapter 7, in about three minutes. It re-verifies the split
> integrity as it goes and prints the headline table. `results/README.md` says
> which file is authoritative for which number."

---

## Part 7 — Posture

**Do:**
- Explain CI/CD first, unprompted.
- State the honest number and then explain why it is honest.
- Raise your own limitations before the committee does. Every limitation in
  Part 6 is one you found and measured — that is a strength, presented correctly.
- Say "I do not know" when you do not know, then say what you would do to find out.

**Do not:**
- Claim to beat the prior art.
- Apologise for 0.42. It is not a bad result; it is a measured one.
- Say "the model failed" about the hybrid hypothesis. Say "the experiment
  refuted my hypothesis, and I report that."
- Improvise a number. If you cannot remember one, say "it is in Chapter 7, I do
  not want to misquote it."

**The sentence that wins the room, if the leakage comes up:**
> "I found this myself, I measured exactly what it was worth, I corrected it,
> and I published the correction inside the thesis. That is what the method is
> supposed to do."
