#!/usr/bin/env bash
# OPTIONAL: load a dataset release into BigQuery so teammates can query it with SQL.
#   ./scripts/load_bigquery.sh <release-name> [bq_dataset]        # default bq dataset: predx
# Loads corpus, queries_train/dev/test, train_pairs from gs://$GCS_BUCKET/predx/releases/<name>/dataset/ (gzipped JSON lines;
# nested fields such as `gold` become repeated records).  Tables are replaced on each run.
. "$(dirname "$0")/_common.sh"
: "${GCS_BUCKET:?set GCS_BUCKET}"
NAME="${1:?usage: $0 <release-name> [bq_dataset]}"
DS="${2:-predx}"
SRC="gs://${GCS_BUCKET}/predx/releases/${NAME}/dataset"
bq --location=US mk --dataset --description "predx release ${NAME}" "${DS}" 2>/dev/null || true
for t in corpus queries_train queries_dev queries_test train_pairs; do
  log "loading ${t}"
  bq --location=US load --replace --source_format=NEWLINE_DELIMITED_JSON --autodetect "${DS}.${t}" "${SRC}/${t}.jsonl.gz"
done
log "done. Example:  bq query --use_legacy_sql=false 'SELECT split, COUNT(*) FROM (SELECT \"train\" split FROM ${DS}.queries_train UNION ALL SELECT \"dev\" FROM ${DS}.queries_dev UNION ALL SELECT \"test\" FROM ${DS}.queries_test) GROUP BY split'"
