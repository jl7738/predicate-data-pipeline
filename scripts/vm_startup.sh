#!/usr/bin/env bash
# GCE startup script (runs as root on EVERY boot, including the restart after a Spot preemption).
# If a previous run left the .autoresume marker, restart run_full.sh inside tmux as the normal user.
# The marker is created by  AUTORESUME=1 ./scripts/run_full.sh  and removed when the run finishes.
# NOTE: a stopped Spot VM does not restart itself: run  gcloud compute instances start <vm>  (this script then resumes the job).
[ -f /etc/predx.env ] || exit 0           # VM not set up yet
. /etc/predx.env
[ -f "$REPO_DIR/.autoresume" ] || exit 0
command -v tmux >/dev/null || exit 0
sleep 20                                  # let the network come up
su - "$RUN_USER" -c "tmux has-session -t predx 2>/dev/null || tmux new-session -d -s predx 'cd $REPO_DIR && AUTORESUME=1 ./scripts/run_full.sh; bash'"
