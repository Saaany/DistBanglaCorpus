# DistBanglaCorpus

**A distributed big-data analytics framework for large-scale linguistic, domain, duplication, topic, gender, and geographic analysis of Bangla web corpora.**

DistBanglaCorpus is the reproducible software pipeline accompanying the paper:

> *DistBanglaCorpus: A Distributed Big Data Analytics Framework for Large-Scale Linguistic, Domain, and Bias Analysis of Bangla Web Corpus*

The reference experiment analyzes the 27.5 GB Bangla2B+ corpus using Apache Spark on Google Cloud Dataproc. The raw Bangla2B+ text is **not redistributed** by this repository; users must obtain it from its original provider and place the shards in their own storage.

## What the pipeline does

1. **Parse raw shards** into structured Parquet records with document statistics and SHA-256 hashes.
2. **Infer document domains** using Bangla lexicons and structural rules.
3. **Run corpus analytics**: domain statistics, TTR, token frequencies, exact/near-duplicate analysis, LDA topic modeling, and a runtime benchmark.
4. **Audit representation bias** using Bangla-specific gender markers and Bangladesh division mentions.

## Repository layout

```text
DistBanglaCorpus/
├── scripts/
│   ├── parse_shards.py
│   ├── domain_inference.py
│   ├── analytics.py
│   └── bias_audit_gender_geo.py
├── resources/
│   ├── domain/
│   │   ├── bangla_stopwords.txt
│   │   └── lexicons/
│   └── bias/
├── config/
│   └── config.example.json
├── gcp/
│   ├── create_cluster.sh
│   ├── upload_pipeline.sh
│   ├── run_pipeline.sh
│   └── delete_cluster.sh
├── docs/
│   ├── DATA.md
│   ├── REPRODUCIBILITY.md
│   └── OUTPUT_SCHEMA.md
├── results/reference/
├── examples/sample/
└── .github/workflows/ci.yml
```

## Reproduce the paper experiment on GCP

### 1. Prerequisites

Install and authenticate the Google Cloud CLI, then choose a GCP project with billing enabled.

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Create a bucket (example):

```bash
gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1
```

### 2. Obtain Bangla2B+

DistBanglaCorpus does not redistribute Bangla2B+. After obtaining the corpus from its original source, upload the 16 raw text shards so that they are available as:

```text
gs://YOUR_BUCKET/corpus/raw/*.txt
```

See [`docs/DATA.md`](docs/DATA.md).

### 3. Configure the pipeline

```bash
cp config/config.example.json config/config.json
```

The default example matches the paper's final experiment: 10% vocabulary sample, 20% LDA sample, 20 LDA topics, 50k vocabulary, leading-8-token shingle deduplication, and the reported Spark memory/partition settings.

### 4. Upload code and resources

```bash
export BUCKET_URI=gs://YOUR_BUCKET
bash gcp/upload_pipeline.sh
```

### 5. Create the Dataproc cluster

```bash
export PROJECT_ID=YOUR_PROJECT_ID
export CLUSTER=distbangla-cluster
export REGION=us-central1
bash gcp/create_cluster.sh
```

The reference configuration is one master plus four `n1-standard-4` workers, 200 GB boot disks, Dataproc image `2.1-debian11`, executor memory 10 GB, driver memory 4 GB, two executor cores, and 800 shuffle partitions.

### 6. Run the full pipeline

```bash
export BUCKET_URI=gs://YOUR_BUCKET
export CLUSTER=distbangla-cluster
export REGION=us-central1
bash gcp/run_pipeline.sh
```

Stages run sequentially:

```text
parse_shards.py
    ↓
domain_inference.py
    ↓
analytics.py
    ↓
bias_audit_gender_geo.py
```

### 7. Download aggregate outputs

```bash
gcloud storage cp --recursive gs://YOUR_BUCKET/output ./outputs
```

Reference aggregate outputs from the paper run are provided in [`results/reference/`](results/reference/). Raw corpus text and text-bearing duplicate examples are intentionally excluded.

## Reference results

The paper run produced 9,951,200 parsed documents and approximately 1.77 billion whitespace-delimited words. Among the main reported findings are a 24.62% leading-shingle near-duplicate rate (among documents with at least eight Bangla tokens), 20 LDA topics with final log perplexity 8.5311, 87% male-leaning evaluated occupations under the corpus-level marker definition, and 53.3% of division mentions referring to Dhaka.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the exact environment and caveats.

## Data and licensing

The software and lexicon resources in this repository can be released under the repository license. **Bangla2B+ itself is not included** and remains subject to the original dataset's terms. Do not upload raw shards or derived text excerpts unless the source license explicitly permits redistribution.

## Citation

If you use DistBanglaCorpus, please cite the accompanying paper. A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

## Authors

- Al-Amin Sany — BUET
- Bijoy Ahmed Saiem — BUET
- Abdullah Adnan — BUET
