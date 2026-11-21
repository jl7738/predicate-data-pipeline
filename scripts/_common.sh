# Sourced by the other scripts. Resolves the repo root, activates the venv, loads optional env files.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Optional local settings (CONTACT_EMAIL, HF_TOKEN, GCS_BUCKET ...). Never commit this file.
[ -f "$REPO_DIR/.env" ] && { set -a; . "$REPO_DIR/.env"; set +a; }

if [ -f "$REPO_DIR/.venv/bin/activate" ]; then . "$REPO_DIR/.venv/bin/activate"; fi
mkdir -p "$REPO_DIR/logs"
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log()   { echo "[$(stamp)] $*"; }

# One-way sync of data/ to the bucket when GCS_BUCKET is set. Safe to call repeatedly (rsync, no deletes).
sync_up() {
  if [ -n "${GCS_BUCKET:-}" ] && command -v gcloud >/dev/null 2>&1; then
    log "sync -> gs://${GCS_BUCKET}/predx/"
    gcloud storage rsync -r --exclude='^pdf/.*' "$REPO_DIR/data" "gs://${GCS_BUCKET}/predx/data" >/dev/null 2>&1 \
      || log "WARN: sync failed (continuing)"
  fi
}
