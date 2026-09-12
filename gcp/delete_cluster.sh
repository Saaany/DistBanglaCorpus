#!/usr/bin/env bash
set -euo pipefail
: "${CLUSTER:=distbangla-cluster}"
: "${REGION:=us-central1}"
gcloud dataproc clusters delete "$CLUSTER" --region="$REGION" --quiet
