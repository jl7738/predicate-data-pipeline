#!/usr/bin/env bash
# Reads what the probe / pilot wrote and prints a go / no-go summary. Run on the VM or in Cloud Shell, from the repo root.
#   ./scripts/check_run.sh              # probe (data_probe/) and pilot (data/reports_pilot/ or data/reports/)
set -uo pipefail
cd "$(dirname "$0")/.."

echo "================ PROBE ================"
CSV=data_probe/probe_summary.csv
if [ -f "$CSV" ]; then
python3 - "$CSV" <<'PY'
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
n = len(rows)
if not n:
    print("no rows"); raise SystemExit
w = collections.Counter(x.strip() for r in rows for x in r["warnings"].split("|") if x.strip())
has_pred = sum(1 for r in rows if r["primary"] not in ("", "-", "None"))
no_text = sum(1 for r in rows if int(r["chars"] or 0) < 400)
print(f"documents probed            : {n}")
print(f"with a primary predicate    : {has_pred}/{n}")
print(f"almost no text (<400 chars) : {no_text}/{n}")
print("warnings:", dict(w) if w else "none")
verdict = "OK to continue" if no_text == 0 and has_pred >= 0.6 * n else "LOOK AT data_probe/probe_report.md BEFORE the pilot"
print("probe verdict               :", verdict)
print("(statement-only and very old documents are expected to have no predicate; check the rest by hand against the PDFs)")
PY
else
  echo "no $CSV yet: run ./scripts/run_probe.sh"
fi

echo; echo "================ PILOT AUDIT ================"
A=data/reports_pilot/audit.md; [ -f "$A" ] || A=data/reports/audit.md
if [ -f "$A" ]; then
  echo "file: $A"; head -1 "$A"
  echo "--- FAIL / WARN checks ---"
  grep -E '^\| (FAIL|WARN) \|' "$A" || echo "(none)"
  echo "--- funnel (all records) ---"
  grep -E '^\| (openFDA 510|substantially|summary available|PDF downloaded|text extracted|parsed without|usable)' "$A" | head -10
  echo "--- OCR cost ---"
  grep -i -E 'seconds per|s/doc|OCR' "$A" | head -6
  echo; echo "Rule: any FAIL = stop and fix. A WARN is fine only if you can explain it (e.g. small pilot slice)."
else
  echo "no audit yet: run ./scripts/run_pilot.sh"
fi

echo; echo "================ HAND CHECK ================"
echo "python -m predx.cli --config config/pilot.yaml label-sample   # then fill labeling/label_sample.csv (done=y)"
echo "python -m predx.cli --config config/pilot.yaml label-eval     # prints precision / recall / primary-predicate accuracy"
echo "Bar I would use: primary accuracy and recall both >= ~0.90, and include the OCR'd documents in the sample."
