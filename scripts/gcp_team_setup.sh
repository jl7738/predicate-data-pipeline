#!/usr/bin/env bash
# One-time TEAM setup of a shared GCP project. Run from Cloud Shell by the project owner (not on the VM).
#
#   PROJECT=coms4705-predx BUCKET=coms4705-predx-data \
#   TEAM="julianna@columbia.edu,saumil@columbia.edu,keyvan@columbia.edu" ./scripts/gcp_team_setup.sh
#
# What it does: selects the project, enables the APIs, gives each teammate a role, creates the shared results bucket.
# What it cannot do: link billing and create the budget alert (do those in the console; the printout says where).
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT=your-gcp-project-id (create it in the console first)}"
BUCKET="${BUCKET:?set BUCKET=a-globally-unique-bucket-name}"
TEAM="${TEAM:-}"                                    # comma-separated Google account emails
ROLE="${ROLE:-roles/editor}"                        # simplest for a course project; tighten later if needed
REGION="${REGION:-us-central1}"

gcloud config set project "$PROJECT" >/dev/null
echo ">> enabling APIs (Compute Engine, Cloud Storage)"
gcloud services enable compute.googleapis.com storage.googleapis.com

if [ -n "$TEAM" ]; then
  IFS=',' read -ra MEMBERS <<< "$TEAM"
  for m in "${MEMBERS[@]}"; do
    m="$(echo "$m" | xargs)"; [ -n "$m" ] || continue
    echo ">> granting $ROLE to $m"
    gcloud projects add-iam-policy-binding "$PROJECT" --member="user:$m" --role="$ROLE" --condition=None >/dev/null
  done
fi

if ! gcloud storage buckets describe "gs://$BUCKET" >/dev/null 2>&1; then
  echo ">> creating bucket gs://$BUCKET ($REGION)"
  gcloud storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access
fi

cat <<MSG

Project '$PROJECT' is ready.  Still to do in the console (cannot be scripted safely):
  1. Billing -> link a billing account to the project.
  2. Billing -> Budgets & alerts -> create a budget (e.g. \$50) with email alerts at 50% and 90%.
  3. Teammates open https://console.cloud.google.com, pick the project, and open Cloud Shell (>_).

Next:  PROJECT=$PROJECT BUCKET=$BUCKET ./scripts/gcp_create_vm.sh     (pilot-sized: MACHINE=e2-standard-4 DISK_GB=60)
MSG
