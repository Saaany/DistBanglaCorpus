#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${TMPDIR:-/tmp}/distbangla-smoke"
rm -rf "$WORK"
mkdir -p "$WORK/corpus/raw" "$WORK/domain_resources/lexicons" "$WORK/bias_resources"
cp "$ROOT"/examples/sample/corpus/raw/*.txt "$WORK/corpus/raw/"
cp "$ROOT"/resources/domain/bangla_stopwords.txt "$WORK/domain_resources/"
cp "$ROOT"/resources/domain/lexicons/*.txt "$WORK/domain_resources/lexicons/"
cp "$ROOT"/resources/bias/*.txt "$WORK/bias_resources/"
cp "$ROOT"/examples/sample/config.json "$WORK/domain_resources/config.json"
BUCKET_URI="file://$WORK"

spark-submit --master local[2] "$ROOT/scripts/parse_shards.py" --bucket "$BUCKET_URI"
spark-submit --master local[2] "$ROOT/scripts/domain_inference.py" --bucket "$BUCKET_URI"
spark-submit --master local[2] "$ROOT/scripts/analytics.py" --bucket "$BUCKET_URI"
spark-submit --master local[2] "$ROOT/scripts/bias_audit_gender_geo.py" --bucket "$BUCKET_URI"

echo "Smoke-test outputs: $WORK/output"
