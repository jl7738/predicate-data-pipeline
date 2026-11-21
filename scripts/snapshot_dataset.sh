#!/usr/bin/env bash
# Freeze the current dataset + reports into a named, immutable-by-convention folder in the bucket, so training runs and
# reruns can point at an exact version.   Run on the VM (or any machine with data/ restored via sync_gcs.sh down).
#   ./scripts/snapshot_dataset.sh 2026-11-23_standard_from2006
# Result: gs://$GCS_BUCKET/predx/releases/<name>/{dataset,reports}/   (+ MANIFEST.txt with counts, mask level and git-free config)
. "$(dirname "$0")/_common.sh"
: "${GCS_BUCKET:?set GCS_BUCKET in .env or the environment}"
NAME="${1:?usage: $0 <release-name>   e.g. 2026-11-23_standard_from2006}"
DEST="gs://${GCS_BUCKET}/predx/releases/${NAME}"
[ -f data/dataset/dataset_card.json ] || { echo "no data/dataset/dataset_card.json: run the build first"; exit 1; }
if gcloud storage ls "${DEST}/" >/dev/null 2>&1; then echo "release '${NAME}' already exists in the bucket: pick a new name (releases are never overwritten)"; exit 1; fi

python - <<'PY' > "$REPO_DIR/logs/MANIFEST.txt"
import json, datetime as dt
c = json.load(open("data/dataset/dataset_card.json"))
print("snapshot_utc :", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
print("mask_level   :", c.get("mask_level"))
print("n_meta       :", c.get("n_meta"), "  n_corpus:", c.get("n_corpus"))
print("counts       :", json.dumps(c.get("counts")))
print("corpus cfg   :", json.dumps((c.get("config") or {}).get("corpus")))
print("parse cfg    :", json.dumps((c.get("config") or {}).get("parse")))
PY
log "uploading release to ${DEST}"
gcloud storage cp -r data/dataset "${DEST}/"
[ -d data/reports ] && gcloud storage cp -r data/reports "${DEST}/"
gcloud storage cp "$REPO_DIR/logs/MANIFEST.txt" "${DEST}/MANIFEST.txt"
log "done: ${DEST}"
cat "$REPO_DIR/logs/MANIFEST.txt"
