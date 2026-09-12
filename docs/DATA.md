# Data access and layout

## Bangla2B+

The reference experiment uses Bangla2B+, introduced with BanglaBERT. The raw corpus is **not included** in this repository. Obtain it from the original provider and comply with its access and redistribution terms.

The pipeline expects the raw shards at:

```text
<BUCKET>/corpus/raw/*.txt
```

where `<BUCKET>` may be a GCS URI such as `gs://my-bucket`.

Each raw file is expected to contain UTF-8 Bangla text with blank lines separating documents.

## Cloud-side resource layout

`gcp/upload_pipeline.sh` creates the following layout:

```text
<BUCKET>/
├── scripts/
├── domain_resources/
│   ├── config.json
│   ├── bangla_stopwords.txt
│   └── lexicons/
├── bias_resources/
├── corpus/
│   ├── raw/
│   ├── parsed/
│   └── labelled/
├── models/
└── output/
```

## What is safe to publish

The repository includes source code, lexicons, configuration, output schemas, and aggregate reference results. It intentionally excludes raw corpus text and the `top_dup_clusters` output because that artifact contains leading text shingles copied from source documents.
