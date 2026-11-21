# Shared Google Cloud runbook: one project, one VM, one bucket, everyone authorised, rerun against the same data

Goal: run the full extraction on Google Cloud, keep **all results in a cloud bucket** (optionally also in BigQuery),
let the whole team use it, and be able to **rerun any stage later against the same stored data**.

Honesty note: the repo's scripts were tested here with stubs and on a laptop-sized synthetic run. The Google Cloud steps themselves
(console menus, IAM, quotas, billing) could not be run from the sandbox that built this repo, so menu names may differ a little.
Where something is a guess it says so.

---

## 0. The picture

```
 your browser / Cloud Shell ──(gcloud)──►  PROJECT  coms4705-predx
                                              │
        teammates (IAM roles) ────────────────┤
                                              ├─ VM  predx-extract   (Spot, 8-16 vCPU, 300 GB disk)   ← the ONLY writer
                                              │     ~/predicate-data/data/…   (working copy: PDFs, text cache, dataset)
                                              │            │  rsync every 10 min + at the end  (scripts/sync_gcs.sh up)
                                              ▼            ▼
                                        BUCKET  gs://BUCKET/predx/
                                              ├─ data/        working state, everything except raw PDFs (resume + reruns)
                                              └─ releases/<name>/{dataset,reports}/MANIFEST.txt   frozen versions (training reads these)
                                              (optional)  BigQuery dataset `predx`   ← SQL over a release
```

Rules that keep it sane:
1. **One writer.** Only the VM writes to `gs://BUCKET/predx/data` (rsync is last-writer-wins and does not merge).
2. **The bucket is the source of truth, the VM disk is a cache.** If the VM or disk is lost, `sync_gcs.sh down` restores everything but the PDFs.
3. **Releases are frozen.** Training and the writeup use `releases/<name>`, never the moving `data/` folder.

---

## 1. Project owner: create the project (once, ~15 min)

**1.1 Project.** https://console.cloud.google.com → project picker (top left) → **New project** → name `coms4705-predx` → **Create**.
Note the **Project ID** (shown on the dashboard; may differ from the name). Make sure it is selected in the picker.

**1.2 Billing.** ☰ → **Billing** → link a billing account to this project (a free-trial account works).
Then **Budgets & alerts → Create budget**: amount e.g. $60 (your shared budget is ~$100 including GPU fine-tuning), alerts at 50 / 90 / 100%,
add the teammates' emails as recipients. Alerts only email you; they do not stop spending. You stop spending by stopping the VM (§6.4).

**1.3 Open Cloud Shell.** The `>_` icon top right. Set the project and check it:
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config list project
```

**1.4 Upload the repo.** Cloud Shell toolbar ⋮ → **Upload** → `predicate-data.zip`, then:
```bash
unzip -q predicate-data.zip && cd predicate-data
```

**1.5 CPU quota check (do this now, it is the most common surprise).** New and trial projects often have a low vCPU limit.
```bash
gcloud compute regions describe us-central1 --flatten="quotas[]" \
  --format="table(quotas.metric,quotas.limit,quotas.usage)" | grep -E "CPUS|SSD_TOTAL|DISKS_TOTAL"
```
If the limit is below what you want (16 for an `e2-standard-16`), either pick `e2-standard-8` or request an increase
(☰ → IAM & Admin → **Quotas** → filter "CPUs", region `us-central1` → **Edit quotas**). Increases can take minutes to a day, so ask early.
(Spot VMs may count against a separate "Preemptible CPUs" quota: if `instances create` later says QUOTA_EXCEEDED, look at that one.)

---

## 2. Authorise the team (project owner, ~10 min)

### 2.1 Pick an access level

| level | roles | who | notes |
|---|---|---|---|
| **Simple (recommended for a 4-person course project)** | `roles/editor` on the project | everyone | can create/use VMs and buckets, SSH, read/write storage. Cannot manage IAM or billing |
| **Tighter** | `roles/compute.instanceAdmin.v1` + `roles/iam.serviceAccountUser` + `roles/storage.objectAdmin` + `roles/compute.osLogin` (or `roles/compute.osAdminLogin` for sudo) | the person who runs the pipeline | |
| **Read-only** | `roles/storage.objectViewer` on the **bucket** only | a teammate who just trains on releases | no VM access needed |

Owners add people; teammates do not need billing roles.

### 2.2 Grant access (Console)
☰ → **IAM & Admin → IAM → Grant access** → paste the teammates' Google emails → pick the role(s) → **Save**.
Each teammate must use the **same Google account** they will log in with (a Columbia account may be restricted by the university; if so use personal accounts or ask course staff).

### 2.3 Grant access (command line, same thing, repeatable)
```bash
PROJECT=YOUR_PROJECT_ID
for m in teammate1@example.com teammate2@example.com teammate3@example.com; do
  gcloud projects add-iam-policy-binding $PROJECT --member="user:$m" --role="roles/editor" --condition=None
done
```
or let the repo script do it together with API enabling and bucket creation:
```bash
PROJECT=YOUR_PROJECT_ID BUCKET=coms4705-predx-data \
TEAM="teammate1@example.com,teammate2@example.com,teammate3@example.com" bash scripts/gcp_team_setup.sh
```

### 2.4 Check it worked
```bash
gcloud projects get-iam-policy $PROJECT --flatten="bindings[].members" \
  --format="table(bindings.role,bindings.members)" | grep -E "editor|owner"
```
Ask a teammate to open the console, pick the project in the picker, and open **Compute Engine → VM instances** (after §4). If they cannot see the project, they are on a different Google account.

### 2.5 The VM's own permission (usually nothing to do)
`gcp_create_vm.sh` runs the VM with the default Compute service account and `--scopes=cloud-platform`, so the VM can write to the bucket
without anyone logging in on it. If your organisation removed the default Editor grant from that service account, `sync_gcs.sh up` will log `WARN: sync failed`.
Fix:
```bash
SA=$(gcloud iam service-accounts list --filter="displayName:Compute Engine default" --format="value(email)")
gcloud storage buckets add-iam-policy-binding gs://BUCKET --member="serviceAccount:$SA" --role="roles/storage.objectAdmin"
```

### 2.6 Secrets
`.env` on the VM holds `CONTACT_EMAIL`, `GCS_BUCKET` and optionally `HF_TOKEN`. It is `chmod 600`, never commit it, never paste the token in chat.
Do not make the bucket public (keep uniform bucket-level access on; the scripts create it that way).

---

## 3. The bucket (owner, once)

Created by `gcp_team_setup.sh` or `gcp_create_vm.sh` (`--uniform-bucket-level-access`, same region as the VM). Then, once:
```bash
BUCKET=coms4705-predx-data
gcloud storage buckets update gs://$BUCKET --versioning        # keeps old versions of overwritten files: an undo button for reruns
```
Per-bucket access (instead of project-wide Editor), e.g. a read-only teammate:
```bash
gcloud storage buckets add-iam-policy-binding gs://$BUCKET --member="user:teammate@example.com" --role="roles/storage.objectViewer"
```
Layout the scripts use:

| path | written by | content |
|---|---|---|
| `gs://BUCKET/predx/data/` | `run_full.sh` (every 10 min + at the end), `sync_gcs.sh up` | everything under `data/` except `pdf/`: metadata, text cache, `parsed.jsonl.gz`, status logs, `dataset/`, `reports/`, `labeling/` |
| `gs://BUCKET/predx/data/pdf/` | `sync_gcs.sh up-pdfs` only | raw PDFs (~70 GB; re-downloadable, so normally NOT stored) |
| `gs://BUCKET/predx/releases/<name>/` | `snapshot_dataset.sh` | frozen `dataset/`, `reports/`, `MANIFEST.txt` |

---

## 4. Create the VM and let everyone use it (owner, ~10 min)

**4.1 Create.** In Cloud Shell, repo folder:
```bash
PROJECT=YOUR_PROJECT_ID BUCKET=coms4705-predx-data MACHINE=e2-standard-16 DISK_GB=300 bash scripts/gcp_create_vm.sh
```
(`e2-standard-8` is the default; 16 vCPUs roughly halve the OCR wall time. Spot VMs are 60-90% cheaper but Google can stop them: the job resumes, see §5.4.)

**4.2 Copy the code up.**
```bash
gcloud compute scp --recurse --zone=us-central1-a ~/predicate-data predx-extract:~/
```

**4.3 Log in.** Console → ☰ → **Compute Engine → VM instances** → **SSH** next to `predx-extract` (a browser terminal opens), or `gcloud compute ssh predx-extract --zone=us-central1-a`.

**4.4 Install.** On the VM:
```bash
cd ~/predicate-data
CONTACT_EMAIL=you@example.com GCS_BUCKET=coms4705-predx-data HF_TOKEN=hf_xxx bash scripts/vm_setup.sh
```
This installs OCR tools, makes the venv, runs the tests and a demo. Stop if the tests fail. (`HF_TOKEN` needs the EmbeddingGemma licence accepted at huggingface.co/google/embeddinggemma-300m; without it the audit's token counts are labelled approximate.)

**4.5 How several people share the one session.** Each person who SSHes in gets their *own* Linux user, with their own home folder and their own `tmux`.
So the repo, `.env` and the running job belong to the user who installed it. Everyone else attaches as that user:
```bash
sudo -iu OWNER_LINUX_USER            # OWNER_LINUX_USER = whoever ran vm_setup.sh (see `ls /home` on the VM)
tmux attach -t predx
```
This needs sudo on the VM: users added through the console get it by default; if you use OS Login, give those teammates `roles/compute.osAdminLogin`. (I could not test the multi-user behaviour here; if `sudo -iu` is refused, run everything from the one owner account and share by screen-sharing the SSH window.)

---

## 5. Run the full pipeline (on the VM)

**5.1 Gate.** Pilot audit core rates pass, hand check ≥ ~0.90 (`docs/GCP_TEAM.md` part C), settings decided (`docs/RUNBOOK_FULL.md` §1).

**5.2 Smoke test the VM** (10 min):
```bash
cd ~/predicate-data && bash scripts/run_probe.sh && bash scripts/check_run.sh
```

**5.3 Start.**
```bash
tmux new -s predx
SETS="corpus.min_decision_date=2006-01-01" AUTORESUME=1 SHUTDOWN_WHEN_DONE=1 ./scripts/run_full.sh
# detach: Ctrl-b then d.   later:  tmux attach -t predx
```
Leave out `SETS=...` to include every year. `SETS` takes any `a.b=c` config overrides (space separated) and is remembered across restarts.
The first log line prints the settings in use; write them in your notes.

**5.4 If Google stops the Spot VM.** It stays stopped (Spot VMs do not restart themselves). From Cloud Shell:
```bash
gcloud compute instances start predx-extract --zone=us-central1-a
```
With `AUTORESUME=1` the VM's startup script restarts the job inside tmux with the same settings; nothing finished is repeated (writes are atomic, logs append-only).
At most ~10 minutes of un-synced results are at risk, and those are re-derivable.

**5.5 Watch it.** `tmux attach -t predx`, `python -m predx.cli status`, `tail -5 logs/full_*.log`, `df -h .`. After the first hour compute docs/second and rescale the time estimate (pilot: ~0.4-0.5 docs/s on 2 cores).

**5.6 Done.** The run syncs a last time and, with `SHUTDOWN_WHEN_DONE=1`, powers the VM off. Check from Cloud Shell:
```bash
gcloud storage ls gs://coms4705-predx-data/predx/data/dataset/
gcloud storage cat gs://coms4705-predx-data/predx/data/reports/audit.md | head -60
```

---

## 6. Store the results: bucket (always) and BigQuery (optional)

### 6.1 Freeze a release (on the VM, after you have read the audit)
```bash
./scripts/sync_gcs.sh up                                    # make sure the working state is in the bucket
./scripts/snapshot_dataset.sh 2026-11-23_standard_from2006  # new name every time; existing releases are refused, never overwritten
```
This copies `data/dataset/` and `data/reports/` to `gs://BUCKET/predx/releases/<name>/` and writes a `MANIFEST.txt` (mask level, counts, the corpus and parse settings used).
Name releases so the settings are readable: `<date>_<mask level>_<year cutoff>`.

### 6.2 Optional database: BigQuery
For SQL over the data (counts by year, join queries to devices, quick checks), after a release exists:
```bash
./scripts/load_bigquery.sh 2026-11-23_standard_from2006          # creates dataset `predx`, tables corpus, queries_*, train_pairs
bq query --use_legacy_sql=false 'SELECT product_code, COUNT(*) n FROM predx.corpus GROUP BY 1 ORDER BY n DESC LIMIT 10'
```
Needs the BigQuery API (`gcloud services enable bigquery.googleapis.com`). I tested only that the script issues the right `bq` commands, not against a real BigQuery:
schema auto-detection can fail on fields that are always empty (for example `reference` when no query has one); if a load errors on a column, load that table with
`--schema` or as a single JSON string column. BigQuery is a convenience for browsing; **training should read the files in the release folder**, not the tables.

### 6.3 What the bucket costs
Everything except raw PDFs is small (hundreds of MB). Storage is a few cents per GB-month. Storing the PDFs (~70 GB) is the only meaningful cost and not needed.

### 6.4 Stop the meter
```bash
gcloud compute instances stop predx-extract --zone=us-central1-a      # compute billing stops; disk (~$) keeps billing
gcloud compute instances delete predx-extract --zone=us-central1-a    # only when everything is in the bucket
```

---

## 7. Rerun things against the same data

Everything below reuses what is already stored; only the stage you change is recomputed.

| I want to… | where | commands | recomputed |
|---|---|---|---|
| **Resume** after a stop or error | VM | `./scripts/run_full.sh` (same `SETS`) | only unfinished work |
| **Fix a parsing rule** (`sections.py`, `predicates.py`) | VM | `python -m predx.cli parse && python -m predx.cli build && python -m predx.cli audit` then `./scripts/sync_gcs.sh up` | parse, build, audit (minutes; no download, no OCR) |
| **Try a different mask level** (`standard` ↔ `strict`) or usable rule | VM | `python -m predx.cli build --mask-level strict && python -m predx.cli audit` (or set the config key and `STAGES=build,audit SETS="…" ./scripts/run_full.sh`) | build, audit |
| **Change the year window or splits** | VM | `SETS="corpus.min_decision_date=…" STAGES=build,audit ./scripts/run_full.sh` | build, audit |
| **Change the OCR settings** | VM | delete the affected text cache (`data/text/…`) and run `python -m predx.cli text` | OCR (slow) |
| **Start from scratch on a new VM** (disk lost, new person, new project) | new VM or Cloud Shell | `./scripts/vm_setup.sh` then `./scripts/sync_gcs.sh down` then any stage above | nothing is re-downloaded except PDFs if needed |
| **Compare two settings** | VM | build one, `snapshot_dataset.sh A`; change, build, `snapshot_dataset.sh B`; compare the two `reports/audit.md` and `MANIFEST.txt` | |
| **Train on a fixed version** | Colab / training VM | §8 | none |

After any change: read audit section 0 again, then snapshot a **new** release name. Keep the old one so earlier results stay reproducible.
If you rerun on a *second* machine, treat it as read-only for the bucket's `data/` folder (rule 1) or use a different `GCS_BUCKET` path for its experiments.

---

## 8. Use the data from Colab or a training VM

**Colab:**
```python
from google.colab import auth; auth.authenticate_user()          # sign in with the Google account that has access
!gcloud storage cp -r gs://coms4705-predx-data/predx/releases/2026-11-23_standard_from2006/dataset /content/dataset
!cat /content/dataset/dataset_card.json | head -c 600
```
**Training VM / GPU:** same `gcloud storage cp -r gs://…/releases/<name>/dataset .` (the VM's service account already has access).
The training code reads `corpus.jsonl.gz`, `queries_{train,dev,test}.jsonl.gz`, `train_pairs.jsonl.gz` and `dataset_card.json`
(the card records similarity, prompts, max lengths and the mask level). Always cite the release name in results.

---

## 9. Costs and safety, in one list
- Budget alert set before the VM exists (§1.2). Check **Billing → Reports** after the first day.
- Spot VM + `SHUTDOWN_WHEN_DONE=1` + `AUTORESUME=1`; stop the VM whenever nobody is using it.
- Disk keeps billing when the VM is stopped: delete the VM after the final sync.
- Do not store PDFs in the bucket unless you need them.
- Do not give anyone Owner or billing roles; do not share `.env`.
- Delete the project (☰ → IAM & Admin → Settings → **Shut down**) when the course is over to stop all charges.

## 10. Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| `instances create` fails with `QUOTA_EXCEEDED` | CPU quota too low | smaller `MACHINE`, or request more quota (§1.5) |
| `gcloud` says permission denied for a teammate | wrong Google account or missing role | check §2.4; have them run `gcloud auth list` |
| `WARN: sync failed` in the log | VM service account cannot write to the bucket | §2.5 |
| teammate cannot see the repo or the running job on the VM | they are a different Linux user | §4.5 (`sudo -iu OWNER`) |
| job gone after the VM was stopped | Spot preemption without `AUTORESUME` | start the VM, `tmux new -s predx`, rerun the same `run_full.sh` command |
| many `not_found` in fetch | FDA has no PDF for those records (expected for a few %), or a folder-pattern problem | the audit §2 shows fetch success by year; a sudden zero for one era means a pattern problem: send me the log lines |
| `snapshot_dataset.sh` says the release exists | names are never overwritten | choose a new name |
| bucket name taken | names are global | add a suffix (`coms4705-predx-data-rjm`) |
| out of disk | PDFs | `SETS="fetch.keep_pdfs=false"` for a rerun, or a bigger `DISK_GB` |

## 11. Checklist
- [ ] Project created, billing linked, budget alert with all emails (§1)
- [ ] CPU quota enough for the chosen machine (§1.5)
- [ ] Teammates added with a role; each confirmed they can see the project (§2)
- [ ] Bucket exists, versioning on (§3)
- [ ] VM created, code copied, `vm_setup.sh` tests pass, probe matches (§4, §5.2)
- [ ] `tmux`; full run started with `AUTORESUME=1 SHUTDOWN_WHEN_DONE=1`, settings noted (§5.3)
- [ ] Rate checked after the first hour; time estimate updated
- [ ] Audit section 0 has no FAIL; `validate` clean; 10-15 documents spot-checked
- [ ] `sync_gcs.sh up`, then `snapshot_dataset.sh <name>` (§6.1); optional BigQuery (§6.2)
- [ ] VM stopped (deleted once the release is in the bucket)
