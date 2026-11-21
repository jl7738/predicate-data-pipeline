# Runbook: from the zip to a cleaned dataset (full run)

"Cleaning" here means everything between the raw FDA data and the files the training code reads:
download the PDFs, read their text (OCR when scanned), cut out *indications* and *description*, find the cited predicates,
drop unusable documents, **mask anything that would leak the answer**, split by date, and audit the result.
It is one command (`run_full.sh`); the work for you is deciding the settings, running it safely, and judging the output.

Everything is resumable. If anything stops, run the same command again and finished work is skipped.

---

## 0. Before you start (do once)

| need | how to check |
|---|---|
| Pilot passed (audit core rates PASS, `check_run.sh` clean) | `docs/GCP_TEAM.md` part C |
| Hand check done: >= 15-25 documents labelled, primary accuracy and recall >= ~0.90 | Colab notebook "Hand check", or `label-sample` / `label-eval` |
| Google Cloud project, billing, budget alert, bucket, teammates added | `scripts/gcp_team_setup.sh`, `docs/GCP_TEAM.md` part A |
| (optional but recommended) Hugging Face token with EmbeddingGemma licence accepted | huggingface.co/google/embeddinggemma-300m, then a token under Settings -> Access Tokens. Without it the audit's token counts are labelled APPROXIMATE |

## 1. Decide the settings (5 minutes, as a team)

These change the dataset, so decide *before* the long run. Defaults are in `config/default.yaml`.

| decision | default | when to change | how |
|---|---|---|---|
| Which years | all (1976 on); splits fixed: train 2015-2023, dev 2024, test 2025-2026 | If time is short, keep only newer documents (digital PDFs, far less OCR). Roughly 2 pts of test ceiling is lost by starting in 2006 | `SETS="corpus.min_decision_date=2006-01-01"` |
| Usable = both fields | `usable_policy: both` (needs indications AND description) | `ifu_only` keeps more documents, with no description text | `SETS="corpus.usable_policy=ifu_only"` |
| Old summaries with no "Device Description" heading | `desc_fallback: summary_body` (uses the summary page, labelled `desc_source`) | `none` to be strict | `SETS="parse.desc_fallback=none"` |
| Leakage masking | `standard` (K-numbers and cited device names replaced) | `strict` also drops sentences that talk about the predicate; run both and compare in the ablation | `SETS="leakage.mask_level=strict"`, or `python -m predx.cli build --mask-level strict` after the run |
| "Reference Predicate Device" | counted as an additional predicate | Decide from your hand check whether it should be gold (FDA treats reference devices as not predicates) | rule is in `predx/predicates.py` (`_LABELS`); edit, then re-run parse |
| Keep PDFs? | yes (~70 GB at full scale) | `false` deletes each PDF once its text is cached | `SETS="fetch.keep_pdfs=false"` |

Write the choices down: they go in your milestone's data section, and `dataset_card.json` records them.

## 2. Create the shared VM (in Cloud Shell, once)

```bash
cd ~/predicate-data      # unzip -q predicate-data.zip first if needed
PROJECT=YOUR_PROJECT_ID BUCKET=YOUR_BUCKET bash scripts/gcp_create_vm.sh
```
Default is a Spot VM (cheap, may be stopped by Google; the run resumes). OCR is CPU-bound and scales with cores, so for the full run
re-create it larger if you can: `MACHINE=e2-standard-16 DISK_GB=300` (Cloud Shell: add the variables before the command).
Estimate from the pilot: about 4 CPU-seconds per document averaged over digital and scanned, so roughly 120-130 CPU-hours for ~100k documents
(about 16 h on 8 vCPUs, about 8 h on 16). That is an extrapolation, so measure on the first hour of the run and rescale.

## 3. Put the code on the VM

```bash
# Cloud Shell
gcloud compute scp --recurse --zone=us-central1-a ~/predicate-data predx-extract:~/
# Console -> Compute Engine -> VM instances -> SSH (any teammate)
cd ~/predicate-data
CONTACT_EMAIL=you@example.com GCS_BUCKET=YOUR_BUCKET HF_TOKEN=hf_xxx bash scripts/vm_setup.sh
```
`vm_setup.sh` installs OCR tools and Python packages, runs the tests and a small demo. Do not continue if the tests fail.

## 4. Smoke test on the VM (10 minutes, catches setup mistakes cheaply)

```bash
bash scripts/run_probe.sh        # real PDFs: should fetch ~all and match the known predicates
bash scripts/check_run.sh
```

## 5. Start the full run

```bash
tmux new -s predx                                   # keeps running if your browser closes
AUTORESUME=1 SHUTDOWN_WHEN_DONE=1 ./scripts/run_full.sh
```
- `AUTORESUME=1`: if Google stops the Spot VM, start it again (`gcloud compute instances start predx-extract --zone=us-central1-a`) and it restarts the job by itself.
- `SHUTDOWN_WHEN_DONE=1`: powers the VM off after the audit so compute billing stops (the disk still costs a few dollars a month).
- Detach with `Ctrl-b d`; come back with `tmux attach -t predx`.

Settings from step 1 go in `SETS` (space separated, `a.b=c`), and are remembered if the VM is restarted:
```bash
SETS="corpus.min_decision_date=2006-01-01 leakage.mask_level=strict" AUTORESUME=1 SHUTDOWN_WHEN_DONE=1 ./scripts/run_full.sh
```
(`CONFIG=config/other.yaml` selects another config file the same way.) The first log line prints the config and settings in use: check it.
Use the **same** `SETS` every time you re-run, or the later stages will rebuild with different rules.

What it does, in order (`run-all --stages subjects,closure,full,build,audit`):

| stage | what happens | expected time |
|---|---|---|
| meta | downloads the openFDA metadata (~176k records) | ~1 min |
| subjects | fetch + read + parse all subject devices decided 2015 onward | hours (download is capped at 4 PDFs/s) |
| closure | fetch + read the predicates those subjects cite | hours |
| full | the rest of the corpus (everything usable that could be a candidate) | the longest part, mostly OCR |
| build | filters, masking, splits, pairs -> `data/dataset/` | minutes |
| audit | numbers and checks -> `data/reports/` | minutes |

Results are copied to your bucket every 10 minutes (`gs://BUCKET/predx/data`).

## 6. Watch it (any time, any teammate)

```bash
tmux attach -t predx                      # live log
python -m predx.cli status                # how far each stage is
tail -5 logs/full_*.log
df -h .                                   # disk
```
Healthy signs: documents/second steady (digital fast, OCR slow, ~0.4-0.5/s per 2 cores), `errors` staying at or near 0, disk growing about 0.7 MB per document.
Stop and look if: many `not_found` or `error` in fetch (FDA throttling or a URL-folder problem), `errors` climbing in the text stage, or disk above ~85%.

If the VM stops: restart it (command above). If the disk is lost: `./scripts/sync_gcs.sh down` restores the results from the bucket (PDFs are re-downloadable).

## 7. When it finishes (`data/reports/audit.md`)

```bash
cat data/reports/audit.md | head -60      # section 0 first
python -m predx.cli validate              # dataset files are well-formed (must report no problems)
ls -lh data/dataset/
```

Read in this order:
1. **Section 0**: any FAIL means stop and fix. A WARN is acceptable only if you can explain it (for example a small test slice).
2. **Section 1 (funnel)**: how many of the ~176k records survive each step and why the rest dropped (statement-only, no PDF, no indications, ...).
3. **Section 2**: download success and OCR share by year. This is where extraction trouble shows up.
4. **Section 4**: the recall ceiling and *why* predicates are missing (older than the corpus, no PDF, statement-only). Report this number; no model can beat it.
5. **Sections 5-6**: candidates per query and whether the test split is large enough for your confidence interval.
6. **Leakage rows**: `no K-numbers left in query text` must be 0.

Then spot-check by eye: 10-15 documents from `data/labeling/` with `label-sample` and `label-eval` on the real run, including some OCR'd ones.
(Pilot hand check validates the rules; this validates the run.)

## 8. If something is wrong: fix cheaply, do not redo the download

| problem | what to change | what re-runs (cached text is reused) |
|---|---|---|
| Wrong predicates or missing sections | edit `predx/sections.py` or `predx/predicates.py` | `python -m predx.cli parse && python -m predx.cli build && python -m predx.cli audit` |
| Leak or too much text removed | `leakage.mask_level` | `python -m predx.cli build --mask-level strict && python -m predx.cli audit` |
| Different years or usable rule | `corpus.*` settings | `build` and `audit` only |
| OCR itself wrong (K-numbers misread) | `text.ocr.*` settings | `rm -rf data/text` for the affected documents, then `text` again (slow) |

Only the OCR case repeats slow work. Always re-run `check_run.sh` / the audit after any change and save the old report for comparison.

## 9. Hand the dataset to training

Copy `data/dataset/` (and `dataset_card.json`) to the bucket: `./scripts/sync_gcs.sh up`.
The training code reads: `corpus.jsonl.gz`, `queries_{train,dev,test}.jsonl.gz`, `train_pairs.jsonl.gz`, and `dataset_card.json`
(similarity, prompts, max lengths, and the mask level it was built with). For the milestone, the audit tables are your data section:
funnel (§1), token lengths (§3), recall ceiling (§4), candidates and gold per query (§5), sufficiency and CI (§6).

Stop or delete the VM when done: `gcloud compute instances stop predx-extract --zone=us-central1-a` (or delete it once the data is in the bucket).

## 10. One-page checklist

- [ ] pilot passes; hand check >= ~0.90
- [ ] settings from step 1 decided and written down
- [ ] project, budget alert, bucket, teammates
- [ ] VM created (size chosen), code uploaded, `vm_setup.sh` tests pass
- [ ] probe on the VM matches
- [ ] `tmux`, then `AUTORESUME=1 SHUTDOWN_WHEN_DONE=1 ./scripts/run_full.sh`
- [ ] check progress after the first hour and rescale the time estimate
- [ ] audit section 0 has no FAIL; `validate` is clean
- [ ] spot-check 10-15 documents; fix rules and re-parse if needed
- [ ] results synced to the bucket; VM stopped
