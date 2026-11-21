#!/usr/bin/env bash
# Run ON the VM, once, from the repo root.  Installs system packages + a venv, runs the offline tests and the demo.
#   CONTACT_EMAIL=you@columbia.edu GCS_BUCKET=my-bucket HF_TOKEN=hf_xxx ./scripts/vm_setup.sh
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo ">> system packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip poppler-utils tesseract-ocr tmux unzip
tesseract --version | head -1
pdftotext -v 2>&1 | head -1

echo ">> python venv"
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip wheel >/dev/null
pip install -r requirements.txt

# Local settings used by every script (kept out of git).
touch .env
upsert() { grep -q "^$1=" .env && sed -i "s|^$1=.*|$1=$2|" .env || echo "$1=$2" >> .env; }
[ -n "${CONTACT_EMAIL:-}" ] && upsert CONTACT_EMAIL "$CONTACT_EMAIL"
[ -n "${GCS_BUCKET:-}" ]    && upsert GCS_BUCKET "$GCS_BUCKET"
[ -n "${HF_TOKEN:-}" ]      && upsert HF_TOKEN "$HF_TOKEN"
chmod 600 .env
grep -q '^CONTACT_EMAIL=' .env || echo "NOTE: set CONTACT_EMAIL in .env (it goes in the User-Agent sent to FDA)."

# Lets vm_startup.sh find the repo after a Spot preemption.
printf 'REPO_DIR=%s\nRUN_USER=%s\n' "$REPO_DIR" "$(id -un)" | sudo tee /etc/predx.env >/dev/null

echo ">> unit tests (offline, ~30 s)"
python -m pytest tests -q
echo ">> synthetic demo (offline, ~5 s)"
python -m predx.cli --config config/demo.yaml demo --no-tokenizers
echo
echo "Setup OK. Read data_demo/reports/audit.md to see what the audit looks like, then FIRST: ./scripts/run_probe.sh (real PDFs), then ./scripts/run_pilot.sh"
