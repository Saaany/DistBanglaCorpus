# Output schema

The pipeline writes outputs under `<BUCKET>/output/`.

| Path | Format | Key fields / meaning |
|---|---|---|
| `domain_stats/` | CSV | `predicted_domain`, `document_count`, token/sentence/character statistics, Banglish rate |
| `ttr/` | CSV | `predicted_domain`, `total_tokens`, `unique_tokens`, `ttr` |
| `token_frequency/` | Parquet | top Bangla tokens per domain with counts/ranks |
| `dedup_stats/` | JSON text | exact and leading-shingle duplicate summary |
| `domain_dup_breakdown/` | CSV | number of documents participating in duplicate clusters by inferred domain |
| `top_dup_clusters/` | CSV | largest leading-shingle clusters; may contain source text and is not included in the public reference results |
| `topics_json/` | JSON text | LDA topic metadata and terms |
| `topics_csv/` | CSV | topic id and representative terms |
| `bias/gender_bias/` | JSONL | occupation, domain, average bias score, document count, direction |
| `bias/geographic_bias/` | JSONL | division, domain, mention count |
| `bias/summary_gender_geo/` | JSON text | combined gender/geographic summary |
| `benchmark/` | JSON text | Spark runtime/configuration summary |

## Parsed corpus schema

The parser emits Parquet rows containing:

- `doc_id`: deterministic content hash identifier (SHA-256 in the public pipeline)
- `text`
- `sentence_count`
- `token_count`
- `char_count`
- `bangla_token_count`
- `latin_token_count`
- `banglish_rate`
- `sha256`
- `predicted_domain`
- `id`: Spark-generated row identifier

The raw `text` field is not part of the public release.
