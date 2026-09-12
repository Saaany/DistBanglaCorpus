#!/usr/bin/env bash
set -euo pipefail
: "${BUCKET_URI:?Set BUCKET_URI, e.g. gs://my-bucket}"
: "${CLUSTER:=distbangla-cluster}"
: "${REGION:=us-central1}"

COMMON_PROPS="spark.executor.memory=10g,spark.driver.memory=4g,spark.executor.cores=2,spark.sql.shuffle.partitions=800"

submit () {
  local script="$1"
  echo "=== Running $script ==="
  gcloud dataproc jobs submit pyspark "$BUCKET_URI/scripts/$script" \
    --cluster="$CLUSTER" \
    --region="$REGION" \
    --properties="$COMMON_PROPS" \
    -- \
    --bucket "$BUCKET_URI"
}

submit parse_shards.py
submit domain_inference.py
submit analytics.py
submit bias_audit_gender_geo.py

echo "Pipeline complete. Outputs: $BUCKET_URI/output/"
