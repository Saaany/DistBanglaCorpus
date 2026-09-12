#!/usr/bin/env bash
set -euo pipefail
: "${BUCKET_URI:?Set BUCKET_URI, e.g. gs://my-bucket}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

gcloud storage cp "$ROOT"/scripts/*.py "$BUCKET_URI/scripts/"
gcloud storage cp "$ROOT"/resources/domain/bangla_stopwords.txt "$BUCKET_URI/domain_resources/bangla_stopwords.txt"
gcloud storage cp --recursive "$ROOT"/resources/domain/lexicons "$BUCKET_URI/domain_resources/"
gcloud storage cp "$ROOT"/resources/bias/*.txt "$BUCKET_URI/bias_resources/"
gcloud storage cp "$ROOT"/config/config.example.json "$BUCKET_URI/domain_resources/config.json"

echo "Uploaded pipeline resources to $BUCKET_URI"
