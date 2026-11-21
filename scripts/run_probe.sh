#!/usr/bin/env bash
# FIRST contact with real FDA PDFs (run on the VM, needs network). ~5 minutes.
#   1. downloads the K-numbers in config/probe_ks.txt (edit it: add 10-20 varied ones, see the comments in the file)
#   2. runs the real pipeline on them and writes data_probe/probe_report.md
#   3. runs the synthetic OCR degradation curve (informational)
# Extra K-numbers can be passed as arguments:  ./scripts/run_probe.sh K211140 K160123
. "$(dirname "$0")/_common.sh"
[ -n "${CONTACT_EMAIL:-}" ] || log "WARN: CONTACT_EMAIL is not set (see .env)"
python -m predx.cli probe --ks-file config/probe_ks.txt "$@" 2>&1 | tee "logs/probe_$(date -u +%Y%m%dT%H%M%SZ).log"
python -m predx.cli ocr-stress --n 8 --levels 0,1,2 2>&1 | tail -12
log "read data_probe/probe_report.md (warnings first), then fix rules or go on with ./scripts/run_pilot.sh"
