#!/usr/bin/env bash
# Run this from your laptop or Cloud Shell (NOT on the VM).  Creates a Spot VM + a GCS bucket for the results.
#
#   PROJECT=my-gcp-project BUCKET=my-unique-bucket-name ./scripts/gcp_create_vm.sh
#
# Defaults are sized for the FULL run (about 35k subjects + their predicates; OCR is the CPU-heavy part).
# For the pilot alone you can use MACHINE=e2-standard-4 and DISK_GB=60.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT=your-gcp-project-id}"
ZONE="${ZONE:-us-central1-a}"
VM="${VM:-predx-extract}"
MACHINE="${MACHINE:-e2-standard-8}"      # 8 vCPU / 32 GB; OCR scales with vCPUs. e2-standard-16 halves the OCR wall time
DISK_GB="${DISK_GB:-300}"                # PDFs dominate. The audit's projection section tells you the real number
BUCKET="${BUCKET:-}"                     # optional but recommended: Spot VMs can be stopped, the bucket keeps results
REGION="${ZONE%-*}"

gcloud config set project "$PROJECT" >/dev/null

if [ -n "$BUCKET" ]; then
  if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
    echo "creating bucket gs://$BUCKET in $REGION"
    gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access
  fi
fi

# Spot: ~60-90% cheaper, may be stopped at any time. The pipeline is resumable (atomic writes + append-only
# status logs), and --instance-termination-action=STOP keeps the disk so work resumes after you restart the VM.
gcloud compute instances create "$VM" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script="$(dirname "$0")/vm_startup.sh"

cat <<MSG

VM '$VM' created.  Next:
  1. Copy the repo up:   gcloud compute scp --recurse --zone=$ZONE ./predicate-data $VM:~/   (or unzip it on the VM)
  2. SSH in:             gcloud compute ssh $VM --zone=$ZONE
  3. On the VM:          cd ~/predicate-data && ./scripts/vm_setup.sh
  4. Then:               ./scripts/run_pilot.sh        (see README.md)
  Results bucket:        ${BUCKET:-<none: results stay on the VM disk>}

Stop billing for compute when idle:  gcloud compute instances stop $VM --zone=$ZONE   (disk still bills, a few \$/month)
Delete everything when done:         gcloud compute instances delete $VM --zone=$ZONE
MSG
