#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_ID:?Set PROJECT_ID}"
: "${CLUSTER:=distbangla-cluster}"
: "${REGION:=us-central1}"

gcloud config set project "$PROJECT_ID"

gcloud dataproc clusters create "$CLUSTER" \
  --region="$REGION" \
  --master-machine-type=n1-standard-4 \
  --master-boot-disk-size=200GB \
  --worker-machine-type=n1-standard-4 \
  --worker-boot-disk-size=200GB \
  --worker-boot-disk-type=pd-ssd \
  --num-workers=4 \
  --image-version=2.1-debian11 \
  --max-idle=1h
