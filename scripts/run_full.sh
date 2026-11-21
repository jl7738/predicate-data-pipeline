#!/usr/bin/env bash
# The full extraction.  Resumable: if the Spot VM is stopped, start it again and re-run (or let vm_startup.sh do it
# when .autoresume exists).  Stages: subjects -> closure -> full corpus -> build -> validate -> audit.
#   ./scripts/run_full.sh                       # run in the foreground
#   AUTORESUME=1 SHUTDOWN_WHEN_DONE=1 ./scripts/run_full.sh   # unattended: resume after preemption, power off at the end
#   SETS="corpus.min_decision_date=2006-01-01 leakage.mask_level=strict" ./scripts/run_full.sh   # config overrides (space separated a.b=c)
#   CONFIG=config/pilot.yaml ./scripts/run_full.sh              # use another config file
. "$(dirname "$0")/_common.sh"
[ -n "${CONTACT_EMAIL:-}" ] || log "WARN: CONTACT_EMAIL is not set (see .env)"

# Restore the shutdown preference of the original launch when we were restarted by vm_startup.sh.
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-$(sed -n 's/^SHUTDOWN_WHEN_DONE=//p' "$REPO_DIR/.autoresume" 2>/dev/null | head -1 || true)}"
SHUTDOWN_WHEN_DONE="${SHUTDOWN_WHEN_DONE:-0}"
# Same for CONFIG / SETS: a restart by vm_startup.sh must run with the settings of the original launch.
CONFIG="${CONFIG:-$(sed -n 's/^CONFIG=//p' "$REPO_DIR/.autoresume" 2>/dev/null | head -1 || true)}"
SETS="${SETS:-$(sed -n 's/^SETS=//p' "$REPO_DIR/.autoresume" 2>/dev/null | head -1 || true)}"
if [ "${AUTORESUME:-0}" = "1" ]; then
  printf 'SHUTDOWN_WHEN_DONE=%s\nCONFIG=%s\nSETS=%s\n' "$SHUTDOWN_WHEN_DONE" "$CONFIG" "$SETS" > "$REPO_DIR/.autoresume"
fi
ARGS=()
[ -n "$CONFIG" ] && ARGS+=(--config "$CONFIG")
for kv in $SETS; do ARGS+=(--set "$kv"); done
STAGES="${STAGES:-subjects,closure,full,build,audit}"
LOG="logs/full_$(date -u +%Y%m%dT%H%M%SZ).log"
log "run-all stages=$STAGES config=${CONFIG:-default} sets=${SETS:-none} (log: $LOG)"

# background syncer: push results to GCS every 10 min so a preempted VM loses little
( while sleep 600; do sync_up; done ) >> logs/sync.log 2>&1 & SYNC_PID=$!
trap 'pkill -P $SYNC_PID 2>/dev/null || true; kill $SYNC_PID 2>/dev/null || true' EXIT

if python -m predx.cli ${ARGS[@]+"${ARGS[@]}"} run-all --stages "$STAGES" ${NO_BASELINES:+--no-baselines} 2>&1 | tee "$LOG"; then
  touch "$REPO_DIR/.full_done"; rm -f "$REPO_DIR/.autoresume"
  sync_up
  log "done. Read data/reports/audit.md"
  if [ "$SHUTDOWN_WHEN_DONE" = "1" ]; then log "shutting down in 1 min (disk is kept; compute billing stops)"; sudo shutdown -h +1; fi
else
  log "run-all exited with an error; re-run to resume (nothing is repeated). Log: $LOG"
  exit 1
fi
