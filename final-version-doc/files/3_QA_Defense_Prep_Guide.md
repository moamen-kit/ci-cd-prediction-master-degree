# Q&A Defense Preparation Guide

## MSc Thesis: Predicting CI/CD Pipeline Build Failures Using Machine Learning Techniques

**Student:** Moamen Mohamed Aly Hussein (ID: 202401681)
**Defense Date:** Friday, September 11, 2026

> **⚠️ هذه نسخة محدثة.** الأرقام في النسخة القديمة (F1 = 0.59، $383k، +27pp) لم تعد صحيحة.
> تم تصحيح منهجية التقييم، والأرقام الجديدة أقل ولكنها قابلة للدفاع عنها.
> **لا تحفظ الإجابات القديمة** — بعضها يناقض ما هو مكتوب الآن في الرسالة.

---

# 📌 الـ Defense Strategy

## Golden Rules

1. **اشرح CI/CD و GitHub في الأول** — اللجنة الفاتت ما كانتش فاهمة. ده أهم سطر في الملف ده.
2. **اعرف أرقامك الجديدة** — F1 = 0.4216 ± 0.064 · PR-AUC = 0.480 · categorical_only = 0.4808 · $243,670
3. **الأرقام نزلت، والسبب ده نقطة قوة مش ضعف** — أنا لقيت الغلط بنفسي وقِسته وصححته
4. **كن صريح في النواقص** — كل limitation في الملف ده أنا قِسته، مش بس ذكرته
5. **لو ما تعرفش، قول "I don't know, but I would investigate by..."**
6. **اربط بالتطبيق العملي** — إنت DevOps engineer، وده ميزة حقيقية

---

# 🎯 Section 0: الأسئلة التأسيسية (اشرحها قبل ما تتسأل)

## Q0-A: يعني إيه CI/CD؟ ويعني إيه GitHub؟

**قول ده في أول العرض من غير ما حد يسأل:**

> Imagine that every time a student uploads a draft chapter, the university automatically runs a formatting check, a plagiarism check, and a reference check, and emails back "accepted" or "rejected". No human does this — a machine does, every single time.
>
> **GitHub** is the shared filing cabinet where all the drafts are kept, recording who changed what and when. **A commit** is one saved change. **CI/CD** is that automatic checking machine: in software, it rebuilds the entire program and runs thousands of automated tests every time anyone saves a change.
>
> The problem is that this checking is slow and expensive — minutes to hours, real compute cost — and when it fails, a developer has to stop and come back to fix it.
>
> **My research asks: at the moment the change is saved, before the checking machine starts, can we predict that it is going to fail?**

## Q0-B: ليه المشكلة دي صعبة؟

> Only 11 percent of these checks fail. So if I simply guess "it will pass" every single time, I am right 89 percent of the time and I have learned nothing. Accuracy is therefore a useless measure for this problem, and I report F1 on the failure class, which rewards only the catching of actual failures.

## Q0-C: أرقامك اتغيرت عن المرة اللي فاتت. ليه؟

**ده أهم سؤال ممكن يتسأل. الإجابة دي بتقلب الموقف لصالحك:**

> Yes, and I want to explain exactly why, because the change is the result I am most confident in.
>
> I previously reported a failure-class F1 of 0.5924. I now report 0.4216 for the same model on the same data. The model did not get worse — the measuring instrument got honest.
>
> Two defects were found in the evaluation protocol. First, my data has 9,772 workflow runs but only 2,835 distinct commits, so the same commit appears many times. My original split scattered near-identical copies of the same commit across both training and test, and the model was rewarded for recognising things it had already seen. Second, I selected the decision threshold by maximising F1 on the test set, and then reported that same F1 as the result — which is circular.
>
> I corrected both, and because each correction can be applied independently, I can attribute the difference exactly: 0.060 from the commit leakage, 0.126 from the threshold selection. The finding I did not expect is that the threshold error — the less conspicuous one — cost more than twice what the leakage did.
>
> I found this myself, measured it, corrected it, and published the correction inside the thesis as Section 7.4.5.

---

# 🎯 Section 1: أسئلة عن المنهجية (Methodology)

## Q1: ليه اخترت Binary Classification بدل Multi-class؟

> The choice reflects the operational use case: at commit time the system needs to answer one question — "will this build fail?" — to inform a single decision, whether to allocate full pipeline resources or apply an intervention. A multi-class formulation predicting *which* stage fails is academically interesting but does not change that decision, since any failure justifies the same intervention. Additionally, the GitHub Actions API exposes only the final conclusion, not the failure stage, so multi-class would have required a different data source or log parsing, both outside scope.

## Q2: لو الـ Ablation أثبت إن الـ Structured أحسن، ليه سميته Hybrid؟

**⚠️ الإجابة دي اتغيرت تماماً. النتيجة الجديدة أقوى:**

> The finding is now stronger than that, and it goes against my own hypothesis. Under the corrected protocol I ran a fourth configuration that the original ablation never tested: categorical-only, where the model is told nothing but the repository, the workflow name, the branch, and the trigger.
>
> That configuration achieves a failure-class F1 of 0.4808 against 0.4216 for the full hybrid, and a PR-AUC of 0.5467 against 0.4803. It wins on every metric.
>
> So the honest conclusion is that the hybrid hypothesis is refuted, and more comprehensively than I first reported. The text branch contributes nothing measurable, and the numerical and binary features contribute nothing beyond the categorical ones. I keep the hybrid architecture in the thesis because it is what I built and what the ablation is measured against, but I describe the system accurately: it is substantially a project-level risk estimator, not a commit-level one.

## Q3: ليه ما استخدمتش Deep Learning (BERT, Transformers)؟

> The ablation answers this. Text in isolation scores 0.196, and adding text to the structured features produces no measurable lift at all. A more sophisticated text encoder would be optimising the branch that carries the least signal. The bottleneck is what the features can know, not how they are encoded.
>
> Beyond that: a transformer would require GPU compute, contradicting the non-functional requirement that the system run on commodity hardware, and it introduces training stochasticity that would complicate the reproducibility this project achieves.

## Q4: ليه ما عملتش Cross-Validation؟

**⚠️ الإجابة القديمة كانت بتدافع عن عدم عمل CV. دلوقتي إحنا بنعملها:**

> I do. The primary results are five-fold commit-grouped cross-validation.
>
> I moved to it because a single split was demonstrably unsafe on this dataset. Repository identity is the model's strongest feature, and no grouped splitter balances the repository mix across folds, so the composition of any one fold moves the result substantially. Single-fold estimates of F1 varied across a range of about 20 percentage points depending on which fold was drawn. Cross-validation is therefore not a refinement here — it is a precondition for reporting a number at all.
>
> Every one of the 9,772 runs receives exactly one prediction, from a model that never saw any run of that run's commit.

## Q5: إزاي تأكدت إن ما فيش Data Leakage؟

**ده بقى أقوى سؤال ليك:**

> I did not just check for it — I found some, measured it, and corrected it.
>
> Three kinds are addressed. **Post-execution leakage:** run duration, retry count, and status are known only after the run finishes, so they are excluded from every feature set by construction. A failing build's duration is a consequence of the failure, not a cause.
>
> **Identity leakage in the text:** author logins and project names were leaking into the TF-IDF features, so I built a 693-token stoplist from the author and repository vocabularies to remove them.
>
> **Duplicate-commit leakage:** this is the one I found later and it was real. 88.3 percent of my original test rows shared a commit with the training set. Every split is now grouped on the commit hash, and I verify programmatically that no commit appears in more than one partition — it is recorded in `results/split_integrity.json`, which anyone can open.
>
> One mild case remains and I disclose it: the median thresholds, category vocabularies and stoplist are computed over all the data before splitting. They are target-independent, so the effect is mild, but they belong inside the pipeline and that is on the future-work list.

---

# 🎯 Section 2: أسئلة عن النتائج (Results)

## Q6: الـ F1 = 0.42 ده كويس ولا وحش؟

> It is a measured number rather than a flattering one, and I would rather defend 0.42 that is real than 0.59 that is not.
>
> Three things depress it relative to published work, in order of size. First, I refuse post-execution telemetry, which is the single biggest constraint and the entire point of the contribution — a prediction made after the run cannot save the cost of the run. Second, I measure honestly: 0.171 of the figure I originally reported was protocol rather than model. Third, the task is genuinely hard — my own ablation shows most of the available signal is project-level, and commit metadata carries limited information about whether a test suite will fail.

## Q7: ليه الـ Recall أعلى من الـ Precision؟

> It depends on which model. Logistic Regression has recall 0.569 against precision 0.353 — a high-vigilance posture. XGBoost inverts it, at precision 0.520 and recall 0.369.
>
> Which is preferable is an economic question, not a statistical one, and I answer it in the business analysis. Because a missed failure costs $18.81 and a false alarm $2.50, the cost model structurally rewards recall. But that ratio is assumed, not measured, so I report the break-even false-alarm cost instead: Logistic Regression stops paying for itself above $10.27, XGBoost above $20.41.

## Q8: إيه الفرق بين الـ splits اللي عندك؟

> There are three, and only two are used for results.
>
> **Commit-grouped cross-validation** is the primary. It guarantees that every run of a given commit lands on one side of the split, which is required because my rows are runs but my features describe commits.
>
> **Per-repository chronological** is the secondary deployment check. Each project is cut at its own point in time, so for all 18 projects the model trains on that project's past and is tested on its future.
>
> **The stratified random split** is retained but never used for results. It exists solely to demonstrate the defect it contains: 913 commits shared between its training and test sets. Keeping it lets the thesis quantify what that defect was worth.

## Q9: ليه الـ XGBoost اتحسن كده بعد الـ threshold tuning؟

**⚠️ الرقم القديم (+27pp) كان غلط:**

> The improvement is real but much smaller than I first reported. Under honest selection it is +9.3 percentage points, from 0.304 at the default threshold to 0.397 at the selected one.
>
> The +27 points I reported in June came from choosing the threshold on the test set and then reporting the score on that same test set. A quantity selected to maximise a value cannot also be an unbiased estimate of it. The threshold is now chosen on a validation partition carved from the training data, and I measured the residual optimism that the old procedure would still have bought: about 0.009 of F1 for XGBoost.
>
> The methodological lesson survives, and is arguably strengthened: threshold selection is powerful, and precisely because it is powerful it must be done on data you do not then report on.

## Q10: ليه ما عملتش Hyperparameter Tuning؟

> A limited amount was done, but I deliberately did not invest heavily, and the ablation explains why. The gap between feature sets — 0.196 for text-only against 0.481 for categorical-only — is far larger than anything hyperparameter search would recover. The variation across cross-validation folds is ±0.06 to ±0.08, so any tuning gain smaller than that would not be distinguishable from noise on this sample size. Tuning against fold noise is how models get overfitted to a validation set.

---

# 🎯 Section 3: أسئلة عن الـ Hybrid Claim

## Q11: لو الـ Text features ضعيفة، ليه ما شيلتهاش؟

> Because removing them would have hidden the finding rather than reporting it. The ablation is the experiment; the hybrid is what it is measured against. Reporting that a component I built contributes nothing is the result, and deleting the component would delete the evidence.
>
> For a production deployment I would recommend the categorical-only configuration: it performs better, it is far cheaper to compute, and it needs no text pipeline or stoplist at all.

## Q12: ليه الكلمات اللي طلعت مش زي المتوقع (fix, bug, revert)؟

> The failure-discriminative vocabulary is timezone, utc, thresholds, borrow — not the emotional vocabulary one expects. The reason is that developers do not know at commit time that their build will fail. If they knew, they would fix it first. So the signal is not "the developer was worried", it is "this commit touched a fragile area of the code".
>
> That said, the ablation shows these tokens add nothing once project identity is known, so this is an interesting observation about the data rather than a driver of the model.

## Q13: مش ممكن الـ Repository feature هي اللي عاملة كل الشغل؟

**⚠️ الإجابة هنا "أيوة، وأنا قِستها":**

> Substantially yes — and I ran the experiment that establishes it rather than waiting to be asked.
>
> A categorical-only model, given nothing but repository, workflow, branch and trigger, outperforms my full model on every metric. The mechanism is visible in the data: the failure rate ranges from 0.0 percent for elastic/elasticsearch, which contributes 600 runs and not one failure, to 38.3 percent for prisma/prisma — an elevenfold spread across the seventeen repositories that fail at all, from 3.5 percent for ruby/ruby to 38.3 percent for prisma/prisma. A one-hot encoding of repository identity therefore encodes a strong prior before any property of the individual commit is consulted.
>
> I report this as a finding rather than a failure, because it tells you what the useful deployment is: project-level and workflow-level triage, not per-commit advice to a developer. It also bounds the claim honestly — the system cannot distinguish two commits to the same repository on the same branch.

---

# 🎯 Section 4: أسئلة عن الـ Dataset

## Q14: ليه 18 repo بالظبط؟

> They are large, active, public projects spanning six languages — JavaScript, Python, Rust, C, Ruby, Java — chosen so the result is not an artifact of one language or one team's conventions. They are public so the dataset is redistributable and the work reproducible. The constraint was API rate limits: collection takes roughly two hours, and 18 projects at 600 runs each gave close to 10,000 runs, which was the target.

## Q15: ليه ما اخترتش corporate dataset؟

> Access. Corporate CI/CD data is rarely shareable, and a thesis whose dataset cannot be published is not reproducible. I chose reproducibility over representativeness and I disclose the trade: corporate environments differ in commit cadence, branch policy and tooling, and transfer is unverified.

## Q16: الـ Class Imbalance 89:11 هيكون مختلف في corporate setting؟

> Almost certainly, and in both directions. A team with strict pre-merge gates would see fewer failures; a team with flaky infrastructure would see more. This matters operationally because my decision thresholds are calibrated to an 11 percent prior and would need recalibration. That is a strength of the threshold work rather than a weakness: the calibration step is explicit, documented, and takes seconds to redo.

---

# 🎯 Section 5: أسئلة عن الـ Business Impact

## Q17: الـ $243,670 ده رقم حقيقي؟

**⚠️ الرقم القديم كان $383,000 وكان مبني على حسابات غلط:**

> It is an order-of-magnitude estimate under stated assumptions, and I would not defend it to the dollar.
>
> The previous figure of $382,802 was wrong for two structural reasons, not just imprecise. It assumed a 30 percent failure rate against the 11 percent actually observed, and it charged nothing at all for false alarms. That second omission meant the estimated saving was a function of recall alone, which is why the earlier work reported the odd result that its tuned model saved *less* than its untuned one.
>
> The rebuilt model uses the observed 11 percent and prices false alarms at $2.50. The estimate is $243,670 per year net.
>
> But the honest answer is that I do not lead with that number. I lead with the break-even: the false-alarm cost at which each configuration stops paying for itself. For my selected model that is $20.41. That is a more defensible basis for a decision than a point estimate resting on a ratio I assumed rather than measured.

## Q18: ليه الـ False Alarm cost أقل بكتير من الـ Missed Failure cost؟

> Because they are different events. A false alarm costs an operator about two minutes to dismiss. A missed failure costs the wasted compute plus roughly fifteen minutes of a developer's context switching, which is the expensive part.
>
> The ratio is 7.5 to 1, and I want to be explicit that this ratio drives the entire ranking of configurations — it is what makes the cost model reward recall. Since I assumed it rather than measured it, Table 7.6 reports the break-even cost for each configuration so a reader can substitute their own assumption. The configuration with the highest estimated saving turns out to be the least robust to that assumption being wrong.

---

# 🎯 Section 6: أسئلة عن الـ Technology Choices

## Q19: ليه TF-IDF مش Word2Vec أو GloVe؟

> TF-IDF is deterministic, interpretable, and needs no pretrained artifact, which supports the reproducibility requirement. And the ablation retrospectively justifies not investing further: the text branch contributes nothing measurable, so a richer text representation would have been effort spent on the weakest signal in the system.

## Q20: ليه XGBoost مش LightGBM أو CatBoost؟

> XGBoost is the most widely benchmarked of the three, which makes the result easier to compare against published work, and it handles sparse matrices natively — my fused feature matrix is about 3,090 columns and mostly sparse.
>
> I would add that under the corrected evaluation the three classifiers I did compare are statistically indistinguishable on F1, differing by less than 0.01 while the cross-fold standard deviation exceeds 0.06. That strongly suggests the choice of gradient-boosting library is not where the remaining performance is.

## Q21: ليه scikit-learn مش PyTorch أو TensorFlow؟

> The task is tabular with a small text branch, which is the regime where gradient-boosted trees remain state of the art. A deep learning framework would add GPU dependencies and training stochasticity for no expected gain — and the ablation confirms the text branch, the only part a neural encoder would improve, contributes nothing.

---

# 🎯 Section 7: الأسئلة الصعبة (Tricky Questions)

## Q22: عملياً إزاي حد يدمج النظام ده؟

> As a webhook on the commit-creation event. The model is a single joblib file under three megabytes, loaded into a Python process, queried with sub-millisecond latency. It returns a failure probability, and downstream tooling decides what to do with it — route to a smaller pre-flight suite, defer to off-peak capacity, or flag for review.
>
> Given the ablation, I would be honest with an adopter about what they are buying: reliable project-level and workflow-level risk triage, not per-commit advice to an individual developer.

## Q23: إيه أكبر مفاجأة في المشروع؟

> Two, and both changed what the thesis says.
>
> The first is that a model told nothing but which project, workflow and branch beats the model I spent the project building. That refuted my hypothesis.
>
> The second is the one I consider most transferable: when I decomposed my own inflated result, choosing the decision threshold on the test set cost 0.126 of F1, while the data leakage everyone looks for cost 0.060. The less conspicuous error was worth more than twice the conspicuous one. Both defects leave every individual step of the analysis looking correct, and neither is visible in the reported metrics.

## Q24: لو رجعتلك وقت زيادة، إيه اللي كنت هتعمله مختلف؟

> Three things, in order.
>
> First, I would have grouped the split on the commit from the beginning. The defect existed because I did not ask early enough what a single row actually represents.
>
> Second, I would move the preprocessing — the medians, the vocabularies, the stoplist — inside the pipeline so it fits on training folds only. It is a mild issue but it is the one methodological weakness I know about and have not fixed.
>
> Third, I would collect differently. Capping at 600 runs per repository is what limits my temporal evaluation to a short horizon. Sampling a fixed time window per project instead of a fixed count would have let me test whether the model survives months of drift rather than hours.

## Q25: مين أكبر منافس، وليه إنت أحسن؟

**⚠️ ما تقولش إنك أحسن. الإجابة دي أقوى:**

> I would not claim to be better, and I would be suspicious of anyone who did on these numbers.
>
> The closest prior work is Patel 2019 and, further back, Hassan and Zhang 2006. Published figures in this space sit around 0.40 to 0.60 F1, and mine is 0.42. But those numbers are not comparable, for two reasons. Most of that work consumes post-execution telemetry, which I deliberately refuse. And most published papers do not state whether they grouped their splits or where they selected their threshold — and I measured, on my own data, that those two choices together are worth 0.171 of F1, which is larger than the entire spread between the published results.
>
> So my contribution is not a better number. It is a number that can actually be interpreted, accompanied by the protocol and the verification artifacts to check it.

---

# 🎯 Section 8: أسئلة عن الـ Future Work

## Q26: لو هتكمل PhD، إيه أول حاجة؟

> Given what the ablation found, the priority is not a better model — it is better features. The system is currently a project-level risk estimator because commit metadata carries little commit-level signal. To get genuine per-commit discrimination I would need features about *what the change touches*: which files, their historical failure rates, test-to-code coupling, dependency changes. That is the direction with the most headroom.
>
> Second would be a corporate validation study, and third a live deployment measuring whether developers actually act on the predictions.

---

# 📋 Section 9: Quick Reference Card (للحفظ)

## الأرقام الأساسية

| الرقم | القيمة |
|---|---|
| Rows / commits | 9,772 runs · 2,835 commits · 3.45 runs per commit |
| Repositories | 18 |
| Class balance | 89.03% success / 10.97% failure |
| **XGBoost failure F1** | **0.4216 ± 0.0638** |
| XGBoost PR-AUC / ROC-AUC | 0.480 / 0.824 |
| **categorical_only F1** | **0.4808 ± 0.0767** (الأفضل) |
| text_only F1 | 0.1962 |
| Selected threshold | 0.076 (XGB) · 0.538 (RF) · 0.646 (LR) |
| Chronological F1 | 0.399 |
| **Annual net saving** | **$243,670** |
| Break-even false-alarm cost | $20.41 (XGB) · $10.27 (LR) |
| Attribution | −0.060 leakage · −0.126 threshold · +0.015 CV |
| Repository failure spread | 0.0% → 38.3% (١١ ضعف بين المشاريع اللي بتفشل) |
| elasticsearch — docs/checksum share of its runs | 591 / 600 (98%) |

## النقاط الذهبية (Golden Points)

1. اشرح CI/CD و GitHub في الأول — من غير ما حد يسأل
2. الموديلات التلاتة **مش مختلفين إحصائياً** — ما تقولش "الفايز"
3. `categorical_only` بيكسب الكل — ودي أهم نتيجة عندك
4. الأرقام نزلت لأن القياس بقى أمين — وأنا اللي لقيت الغلط
5. الـ threshold error كلّف ضعف الـ leakage — دي النتيجة اللي محدش قاسها قبل كده

## Phrases للاستخدام

- "I found this myself, measured it, corrected it, and published the correction."
- "I would rather defend 0.42 that is real than 0.59 that is not."
- "That is a finding, not a failure."
- "I do not know, but I would investigate it by..."
- "Let me be precise about the limitation..."

## Closing Statement (محفوظة للنهاية)

> This project set out to predict CI/CD build failures before they happen, using only what is knowable at commit time. It achieves a failure-class F1 of 0.42 under an evaluation protocol I can fully defend.
>
> Along the way it produced two findings I did not expect. The first is that project identity alone outperforms the model I built, which means the system is a project-level risk estimator and I say so. The second is that when I decomposed my own earlier, higher result, selecting the decision threshold on the test set had cost more than twice what the data leakage did.
>
> Both findings are less flattering than what I set out to prove. Reporting them, with the measurements and the code to check them, is what I consider the contribution.

---

# 🎯 Bonus: نصايح عملية

## قبل الـ Defense
- افتح الـ deck وشوف مفيش نص خارج البوكسات
- افتح `Dataset_Guide.xlsx` وجرب تشرح منه — **ما تفتحش الـ CSV الخام قدامهم**
- اتمرن على Q0-A بصوت عالي ٣ مرات — دي أهم دقيقتين في العرض كله
- راجع رقم واحد بس لو نسيت كل حاجة: **0.4216**

## أثناء الـ Defense
- اتكلم بالراحة، وابدأ من الصفر في الشرح
- لو سألوك رقم مش فاكره: "It is in Chapter 7, I would rather check than misquote it"
- لما تذكر limitation، اذكر إنك قِسته

## لو وقعت في سؤال
- "That is a good question and I do not have a measured answer. What I would do is..."
- ما تخترعش رقم. أبداً.
