#!/usr/bin/env bash
#
# Reproduce every metric and figure reported in the thesis, from data/raw/.
#
# Takes about three minutes. Writes to data/processed/, results/ and figures/.
# It does not write to data/raw/ and does not require network access.
#
# Usage:  ./reproduce.sh
#
set -euo pipefail

cd "$(dirname "$0")"

PY=".venv/bin/python"

if [[ ! -x "$PY" ]]; then
    echo "error: $PY not found." >&2
    echo "Create the environment first:" >&2
    echo "    python3.11 -m venv .venv" >&2
    echo "    .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi

if [[ ! -f data/raw/github_actions_real.csv ]]; then
    echo "error: data/raw/github_actions_real.csv is missing." >&2
    echo "It is committed to the repository; re-collecting it requires a" >&2
    echo "GitHub token and roughly two hours of API time." >&2
    exit 1
fi

echo "=============================================================="
echo " Versions"
echo "=============================================================="
"$PY" - <<'PYVER'
import sklearn, xgboost, pandas, numpy
print(f"  python       {__import__('sys').version.split()[0]}")
print(f"  pandas       {pandas.__version__}")
print(f"  numpy        {numpy.__version__}")
print(f"  scikit-learn {sklearn.__version__}   (results produced with 1.8.0)")
print(f"  xgboost      {xgboost.__version__}   (results produced with 3.2.0)")
PYVER

echo
echo "=============================================================="
echo " Step 1/2  Data preparation, splits, and split-integrity checks"
echo "=============================================================="
PYTHONPATH=. "$PY" -m src.run_phase2_5

echo
echo "=============================================================="
echo " Step 2/2  Cross-validated evaluation, business model, figures"
echo "=============================================================="
PYTHONPATH=. "$PY" -m src.run_corrected_evaluation

echo
echo "=============================================================="
echo " Verifying split integrity"
echo "=============================================================="
PYTHONPATH=. "$PY" - <<'PYCHK'
import json, sys

report = json.load(open("results/split_integrity.json"))
failures = []

for name in ("grouped", "chronological"):
    r = report[name]
    if not r["commit_overlap_clean"]:
        failures.append(f"{name}: commits shared between partitions: {r['commit_overlap']}")
    print(f"  {name:<16} commit overlap {r['commit_overlap']}  -> clean")

chrono = report["chronological"]
if chrono["temporal_invariant_holds"] is not True:
    failures.append("chronological: per-repository temporal invariant violated")
n_repos = len(chrono.get("per_repository", {}))
print(f"  chronological    min(test) >= max(train) for {n_repos}/18 repositories")

leaky = report["stratified_leaky"]["commit_overlap"]["train_vs_test"]
print(f"  stratified       {leaky} shared commits (retained to demonstrate the defect)")

if failures:
    print("\nFAILED:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("\n  All integrity checks passed.")
PYCHK

echo
echo "=============================================================="
echo " Headline results"
echo "=============================================================="
PYTHONPATH=. "$PY" - <<'PYSUM'
import json
cv = json.load(open("results/grouped_cv.json"))
print("  Commit-grouped 5-fold cross-validation, failure class:\n")
print(f"  {'configuration':<24}{'F1 (mean +/- sd)':>20}{'PR-AUC':>10}{'ROC-AUC':>10}")
print("  " + "-" * 64)
for section in ("models", "ablation"):
    for name, v in cv[section].items():
        if section == "ablation" and name == "hybrid_full":
            continue
        f1 = v["f1_failure"]
        o = v["pooled_out_of_fold"]
        print(f"  {name:<24}{f1['mean']:>10.4f} +/-{f1['std']:>6.4f}"
              f"{o['pr_auc']:>10.4f}{o['roc_auc']:>10.4f}")

ladder = json.load(open("results/metric_attribution_ladder.json"))
print("\n  Attribution of the previously reported F1 = 0.5924:\n")
for s in ladder["steps"]:
    d = "" if s["delta_f1_vs_previous"] is None else f"{s['delta_f1_vs_previous']:+.4f}"
    print(f"  {s['f1_failure']:.4f} {d:>8}   {s['step']}")

b = json.load(open("results/business_impact.json"))
print(f"\n  Estimated annual net saving: ${b['annual_net_saved_usd']:,.0f}")
PYSUM

echo
echo "Done. See results/README.md for which file is authoritative for what."
