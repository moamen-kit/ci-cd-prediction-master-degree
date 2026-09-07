# Verification note — reference [1] (Patel)

**Status: UNRESOLVED. Requires the author to check the source PDF.**

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
