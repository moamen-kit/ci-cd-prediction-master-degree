# Verification note — reference [1] (Patel)

**Status: UNRESOLVED, and now materially weaker.** The author holds three
reference PDFs in `dataset/`, and **none of them is the Patel paper.** All three
have been identified from their own title pages and added to the thesis as
references [25], [26] and [27]. See the section at the end of this note.

Chapter 3 positions this project's contribution relative to reference [1], so
the citation needs to be right. It could not be confirmed from here.

## What the thesis currently cites

> [1] A. Patel, "Research the Use of Machine Learning Models to Predict and
> Prevent Failures in CI/CD Pipelines and Infrastructure," *International
> Journal of Engineering Research & Technology*, vol. 8, no. 11, 2019.

## What the evidence suggests instead

A paper with **exactly this title** does exist — it is indexed on ResearchGate
(publication id 384695412). But an independent search returns different
publication details:

> TIJER — International Research Journal, ISSN 2349-9249,
> **Volume 4, Issue 11, pages a8–a16, November 2017.**

The two are mutually exclusive. Note also that **TIJER and IJERT are different
journals with confusingly similar names**, which is the most likely origin of
the discrepancy.

One consistency check supports the TIJER reading: TIJER's own volume numbering
puts July 2019 at Volume 6, so Volume 4 corresponds to roughly 2017. The cited
"vol. 8, no. 11, 2019" does not fit TIJER's sequence, though it is plausible for
IJERT's own numbering.

## Why it was not resolved here

Both primary sources are unreachable from the build environment: `researchgate.net`
and `tijer.org` are blocked by the network egress proxy. The TIJER attribution
rests on a single search-engine summary and is **not corroborated by a primary
source**, so the citation was deliberately left unchanged rather than replaced
with a value that might also be wrong.

## What to check on the source PDF (two minutes)

Open the PDF and read the header or footer of the first page. Papers in both
journals print the journal name, ISSN, volume, issue and month there. Confirm:

1. Journal name — **TIJER** or **IJERT**?
2. **ISSN** — TIJER is 2349-9249
3. Volume and issue
4. Month and year
5. The author's full name and initial

Then correct the entry in `4_Thesis_Source_Markdown.md` (the References section
at the end) and regenerate the derived documents:

```bash
cd <repo root>
cicd-failure-prediction/.venv/bin/python tools/md2docx.py \
  final-version-doc/files/4_Thesis_Source_Markdown.md \
  final-version-doc/files/1_MSc_Thesis_FINAL.docx \
  final-version-doc/files/1_MSc_Thesis_FINAL.docx
cicd-failure-prediction/.venv/bin/python tools/md2pdf.py
```

## If the paper turns out to be weak or unverifiable

The argument of Chapter 3 does not depend on it alone. Reference **[24] Hassan
and Zhang, ASE 2006** — verified against IEEE Xplore — is cited as the origin of
build-outcome prediction, and it carries the same point: that the value of a
prediction lies in its availability before the expensive process completes. If
[1] proves unciteable, Chapter 3 can lean on [24] with only local edits.

If asked in the defence and unsure, the safe answer is factual: *"That citation
is to a paper I have read; I would want to double-check the exact volume and
issue before quoting them, because two journals with similar names publish in
this area."* Do not guess a volume number aloud.


---

# The three reference PDFs the author actually holds

Identified by reading each file's own title page — these details are taken from
the documents, not from a search engine.

## `dataset/1703.04142.pdf` — reference [25]

> L. Madeyski and M. Kawalerowicz, "Continuous Defect Prediction: The Idea and a
> Related Dataset," in *Proceedings of the 14th International Conference on
> Mining Software Repositories*, Buenos Aires, Argentina, 2017, pp. 515–518.
> doi: 10.1109/MSR.2017.46.

Peer-reviewed, MSR 2017. The DOI and page range are printed on the paper's first
page. A file-level dataset of 11 million records across 1,265 projects, built on
TravisTorrent. Cited in Section 3.1 for establishing the feasibility of joining
continuous-integration outcomes to repository-mined process metrics at scale.

## `dataset/frai-9-1776546.pdf` — reference [26]

> R. Dhawan and M. Dhawan, "AI-Augmented Reliability in CI/CD: A Framework for
> Predictive, Adaptive, and Self-Correcting Pipelines," *Frontiers in Artificial
> Intelligence*, vol. 9, art. 1776546, 2026. doi: 10.3389/frai.2026.1776546.

Peer-reviewed, open access (CC BY), published 01 April 2026. Type: Hypothesis
and Theory. Cited in Section 3.1 as complementary framing — a predictor of the
kind this thesis builds is the input such a framework consumes.

## `dataset/osh-1.pdf` — reference [27]

> R. Sharma, E. Petrova, and J. O. Connolly, "Machine Learning-Based Failure
> Prediction in Continuous Integration and Deployment Workflows," unpublished
> manuscript, Nov. 12, 2025. [Online]. Available:
> https://www.researchgate.net/publication/401540054 (accessed Sep. 7, 2026).

**Resolved as far as it can be.** The author supplied the ResearchGate URL, and
the file he supplied is byte-identical to the copy in `dataset/` (matching MD5).
The paper is self-archived: no journal, conference, volume, ISSN or DOI appears
anywhere in its thirteen pages, and none is discoverable. The citation above is
the correct IEEE form for a self-archived manuscript — authors, title,
"unpublished manuscript", date, and the online location with an access date.

**One item for the author to check on the ResearchGate page**, which is
reachable from a normal browser but blocked from this build environment: a
search index lists the record with a **fourth author, "Uthman Usman"**, which
does not appear anywhere in the PDF — the title page names three. Open the page
and compare. If the record does list four authors, add the fourth to reference
[27]; if it lists three, the entry is already correct. Do not add a name on the
strength of a search summary alone.

**This is the most useful of the three, and it needs one caution.**

Useful, because it is the most directly comparable recent work: XGBoost, Random
Forest and SVM on 116,000 workflows from 25,000 projects, reporting failure-class
F1 = 0.88 at 89.7 per cent accuracy. Its SHAP analysis names **build duration**
as the single most influential predictor. Build duration is not known until the
build has finished, so their result is an upper bound on what is achievable
*with* post-execution telemetry — which is exactly the boundary this thesis
declines to cross. It converts the gap between 0.88 and 0.4216 from an
embarrassment into the measured cost of the constraint. Section 3.1 now makes
that argument explicitly.

**The caution (unchanged by the URL):** a ResearchGate link establishes where the
document lives, not that it was peer-reviewed. No publication venue, volume,
ISSN or DOI for the paper itself appears anywhere in the document. It carries a full reference list and a
conventional academic structure, but it was produced in Microsoft Word and its
PDF metadata lists the author as "ola". It is cited above as an unpublished
manuscript, which is the honest form. **If a committee member asks whether it is
peer-reviewed, the answer is that no venue is printed on it.** If a published
version exists, find it and update the entry; if not, the citation stands as a
manuscript and the argument it supports does not depend on its peer-review
status, only on what it reports about its own features.

## Consequence for reference [1]

Chapter 3 no longer rests on [1] alone. It now cites [24] Hassan and Zhang
(verified, ASE 2006) for the origin of build-outcome prediction and [27] Sharma
et al. for the modern comparison. Reference [1] remains in place and remains
unverified. The options are to correct its venue against the source PDF, or — if
that PDF cannot be produced — to remove it, since the argument it was carrying is
now carried by [24] and [27].
