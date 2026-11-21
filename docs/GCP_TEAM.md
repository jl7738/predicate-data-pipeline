# Shared Google Cloud project: setup, probe, pilot, and how to judge them

Everything runs inside ONE project so the team shares the VM, the results bucket and the budget.
Cloud Shell is per person (not shareable); the VM and the bucket are what you share.

## A. One-time setup (project owner)
1. Console -> project picker -> **New project** (e.g. `coms4705-predx`). Note the *project ID*.
2. Console -> **Billing** -> link a billing account. Create a **budget** (e.g. $50) with alerts at 50% / 90%.
   Alerts do not stop spending by themselves: stop the VM when idle.
3. Open Cloud Shell (`>_`), upload `predicate-data.zip` (three-dot menu -> Upload), then:
   ```bash
   unzip -q predicate-data.zip && cd predicate-data
   PROJECT=coms4705-predx BUCKET=coms4705-predx-data \
   TEAM="teammate1@example.com,teammate2@example.com,teammate3@example.com" bash scripts/gcp_team_setup.sh
   ```
   (enables APIs, grants each teammate `roles/editor` (override with `ROLE=`), creates the shared bucket.)
4. If the console refuses (university-managed account), use a personal Google account for the project or ask course staff for credits.

## B. Probe on real PDFs (about 10 min). Cloud Shell is enough
```bash
cd ~/predicate-data && bash scripts/vm_setup.sh          # apt + venv + tests + demo
echo 'CONTACT_EMAIL=you@example.com' >> .env
echo 'GCS_BUCKET=coms4705-predx-data' >> .env
nano config/probe_ks.txt                                  # add 15-20 varied K-numbers (eras, product codes, one statement-only)
bash scripts/run_probe.sh
bash scripts/check_run.sh                                 # go / no-go summary
less data_probe/probe_report.md                           # per-document detail
```
Judge it:
* every K-number downloaded; the log shows which URL folder worked (`pdf`, `pdf99`, `pdf4`, `pdf24`...). Many 404s = a fetch problem.
* primary predicate present and **correct on the ones you open by hand** (open 3-5 PDFs next to the report).
* `NO_IFU` / `NO_DESC` rare on 2005+; `MOSTLY_OCR` only on old scans. Statement-only and very old files with no K-numbers are *expected* to have no predicate.

## C. Pilot (shared VM; 1-3 hours)
```bash
PROJECT=coms4705-predx BUCKET=coms4705-predx-data MACHINE=e2-standard-4 DISK_GB=60 bash scripts/gcp_create_vm.sh
gcloud compute scp --recurse --zone=us-central1-a ../predicate-data predx-extract:~/
gcloud compute ssh predx-extract --zone=us-central1-a       # teammates: console -> Compute Engine -> SSH
cd ~/predicate-data && CONTACT_EMAIL=you@example.com GCS_BUCKET=coms4705-predx-data bash scripts/vm_setup.sh
tmux new -s predx                                           # job survives a closed browser; re-attach: tmux attach -t predx
PANEL=RA N=400 bash scripts/run_pilot.sh
bash scripts/check_run.sh
```
Judge it (`data/reports_pilot/audit.md`, section 0 first):
* any **FAIL** = stop and fix. A WARN is acceptable only if you can explain it.
* download success ~ >= 0.95, text extraction ~ >= 0.95, indications+description found ~ >= 0.85 on non-statement docs.
* usable rate by year: high from ~2005, lower but not near zero for the 1990s.
* `raw_gold_k` / residual leak rows = 0 after masking.
* measured OCR seconds/doc: this prices the full run (the audit section 10 projects it).
* then the **hand check**: `label-sample`, fill the CSV for ~100 documents (include OCR'd ones), `label-eval`.
  Bar I would use: primary-predicate accuracy and recall >= ~0.90 (a judgement call, not a threshold from the proposal).

## D. Decide
* Go to the full run (`scripts/run_full.sh`, resumable) if the probe is clean, the pilot has no FAIL, and the hand check clears the bar.
* Rule changes re-run cheaply: text is cached, so only `parse` / `build` / `audit` repeat.
* If OCR cost is too high, restrict to recent years (`--set corpus.min_decision_date=2015-01-01`, see README section 6).

## E. Team hygiene
* One person runs a given stage at a time; a second person re-running is safe (everything resumes) but wastes quota.
* `gcloud compute instances stop predx-extract --zone=us-central1-a` when idle; delete the VM when extraction is finished.
* Results live in `gs://<bucket>/predx/data` (`scripts/sync_gcs.sh up|down`).
