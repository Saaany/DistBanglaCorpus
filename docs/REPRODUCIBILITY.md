# Reproducibility guide

This document records the configuration corresponding to the final DistBanglaCorpus experiment.

## Reference data

- Raw corpus size: 27.5 GB
- Source websites: 110
- Parsed documents: 9,951,200
- Whitespace-delimited words: approximately 1.77 billion
- Raw format: 16 plaintext shards, blank-line document boundaries

## Dataproc environment

- Region: `us-central1`
- Dataproc image: `2.1-debian11`
- Master: `n1-standard-4`, 4 vCPUs, 15 GB RAM, 200 GB disk
- Workers: 4 × `n1-standard-4`, each 4 vCPUs, 15 GB RAM, 200 GB disk
- Aggregate: 20 vCPUs, approximately 75 GB RAM, 1 TB provisioned boot disk
- Executor memory: 10 GB
- Driver memory: 4 GB
- Executor cores: 2
- Spark SQL shuffle partitions: 800 for analytics

## Analytical parameters

### Vocabulary

- Sample fraction: 0.10
- Seed: 42
- TTR computed per inferred domain
- Top-100 Bangla tokens retained per domain

### Near-duplicate analysis

- Exact deduplication: SHA-256
- Near-duplicate method: identical leading Bangla token shingles
- Shingle size: 8 tokens
- Minimum document length for this stage: 8 Bangla tokens
- Documents eligible for shingle analysis in reference run: 8,322,960
- Near-duplicate clusters: 1,417,757
- Redundant documents after retaining one representative per cluster: 2,049,107
- Reported rate: 0.2462 relative to the 8,322,960 eligible documents

**Important denominator note:** the 24.62% rate is calculated over documents with at least eight Bangla tokens, not over all 9,951,200 parsed documents. When reporting the result, state this denominator explicitly.

`domain_dup_breakdown` counts documents *participating in duplicate clusters* by inferred domain. It is therefore not the same quantity as `near_duplicate_documents`, which subtracts one representative per cluster.

### LDA

- Sample fraction: 0.20
- Seed: 42
- Topics: 20
- Vocabulary size: 50,000
- Minimum document frequency: 5
- Iterations: 20
- Optimizer: online
- Final reference log perplexity: 8.5311

### Bias audit

- Gender context window: ±10 tokens
- Gender measure: smoothed log-odds from explicit Bangla lexical markers
- Geography: regex/lexicon matching of Bangladesh division names and variants

## Reference runtimes

| Stage | Runtime |
|---|---:|
| Document parsing | 33 min |
| Domain inference | 1 hr |
| Analytics | 8 hr 11 min |
| Bias audit | 15 hr 52 min |
| Total | ~25.6 hr |

Runtime depends on Dataproc image revision, GCS throughput, cluster contention, and corpus location.