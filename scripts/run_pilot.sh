#!/usr/bin/env bash
# The extraction pilot (reviewer: "prioritize the extraction pilot early").  One specialty, ~400 subjects + the
# predicates they cite.  Download is a few minutes at 4 req/s; OCR of scanned PDFs dominates (the audit prints measured seconds per document).  Safe to re-run: finished work is skipped.
#   PANEL=RA N=400 ./scripts/run_pilot.sh
. "$(dirname "$0")/_common.sh"
PANEL="${PANEL:-RA}"; N="${N:-400}"
[ -n "${CONTACT_EMAIL:-}" ] || log "WARN: CONTACT_EMAIL is not set (see .env)"
LOG="logs/pilot_$(date -u +%Y%m%dT%H%M%SZ).log"
log "pilot panel=$PANEL n=$N  (log: $LOG)"
python -m predx.cli --config config/pilot.yaml pilot --panel "$PANEL" --n "$N" 2>&1 | tee "$LOG"
# keep the pilot report: the full run overwrites data/reports
rm -rf data/reports_pilot && cp -r data/reports data/reports_pilot
sync_up
log "done. Read data/reports_pilot/audit.md, then:  python -m predx.cli --config config/pilot.yaml label-sample"
