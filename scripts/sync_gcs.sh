#!/usr/bin/env bash
# Sync results between the VM and the bucket.
#   ./scripts/sync_gcs.sh up        # VM -> bucket (everything except raw PDFs, which are re-downloadable)
#   ./scripts/sync_gcs.sh up-pdfs   # also upload the PDFs
#   ./scripts/sync_gcs.sh down      # bucket -> VM (restore after losing the disk, or to move to a new VM)
. "$(dirname "$0")/_common.sh"
: "${GCS_BUCKET:?set GCS_BUCKET in .env or the environment}"
DEST="gs://${GCS_BUCKET}/predx/data"
case "${1:-up}" in
  up)      gcloud storage rsync -r --exclude='^pdf/.*' data "$DEST" ;;
  up-pdfs) gcloud storage rsync -r data "$DEST" ;;
  down)    mkdir -p data && gcloud storage rsync -r "$DEST" data ;;
  *) echo "usage: $0 up|up-pdfs|down"; exit 2 ;;
esac
