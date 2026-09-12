"""
analytics.py

DistBanglaCorpus Analytics Pipeline

Performs:

Phase 1  — Corpus Statistics
Phase 2  — Vocabulary Statistics, Type Token Ratio, Top-100 tokens per domain
Phase 3  — Duplicate Detection (leading-token shingle fingerprinting)
Phase 4  — Topic Modeling (LDA)
Phase 5  — Scalability Benchmark
"""

import argparse
import json
import time

from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer, StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql.functions import (col, count, explode, mean,
                                   percentile_approx, row_number)
from pyspark.sql.functions import size as array_size
from pyspark.sql.functions import stddev
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window
from pyspark.sql.functions import concat_ws, lit
from pyspark.sql.functions import slice as spark_slice

# ==========================================================
# Command-line configuration
# ==========================================================

parser = argparse.ArgumentParser(description="Run DistBanglaCorpus distributed analytics.")
parser.add_argument("--bucket", required=True, help="Storage root, e.g. gs://my-bucket or file:///abs/path")
parser.add_argument("--config", default=None, help="Optional config URI. Defaults to <bucket>/domain_resources/config.json")
args = parser.parse_args()

# ==========================================================
# Spark Session
# Must be created BEFORE reading config from GCS
# ==========================================================

spark = (
    SparkSession.builder
    .appName("DistBanglaCorpus-Analytics")
    .config("spark.sql.shuffle.partitions", "800")
    .getOrCreate()
)

# Allow saveAsTextFile to overwrite existing GCS paths on re-runs
spark.sparkContext._jsc.hadoopConfiguration().set(
    "mapreduce.fileoutputcommitter.marksuccessfuljobs", "false"
)

# Helper: delete a GCS path before saveAsTextFile
def delete_gcs_path(path):
    try:
        URI       = spark.sparkContext._gateway.jvm.java.net.URI
        Path      = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
        FileSystem = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.FileSystem
        fs = FileSystem.get(
            URI(path),
            spark.sparkContext._jsc.hadoopConfiguration()
        )
        p = Path(path)
        if fs.exists(p):
            fs.delete(p, True)
    except Exception:
        pass

# ==========================================================
# Python UDF Tokenizer
# Replaces RegexTokenizer which silently drops Bangla chars
# because Java's \W class only covers ASCII word characters.
# This UDF splits on whitespace and keeps tokens that contain
# at least one Bangla Unicode character (U+0980 – U+09FF)
# and are at least 2 characters long.
# ==========================================================

def bangla_tokenize(text):
    if not text:
        return []
    return [
        t for t in text.split()
        if len(t) >= 2
        and any('\u0980' <= ch <= '\u09FF' for ch in t)
    ]

bangla_tokenize_udf = udf(bangla_tokenize, ArrayType(StringType()))

# ==========================================================
# Load Configuration
# ==========================================================

CONFIG_PATH = args.config or f"{args.bucket.rstrip('/')}/domain_resources/config.json"

config_text = "\n".join(
    row.value
    for row in spark.read.text(CONFIG_PATH).collect()
)

config      = json.loads(config_text)
BUCKET      = args.bucket.rstrip("/")
OUTPUT      = f"{BUCKET}/" + config["paths"]["output"]
MODEL_PATH  = f"{BUCKET}/" + config["paths"]["models"]

# ==========================================================
# Load labelled corpus
# ==========================================================

docs_df = (
    spark.read.parquet(f"{BUCKET}/corpus/labelled/")
    .cache()
)

# Start benchmark timer — measures full pipeline runtime
benchmark_start = time.time()

total_documents = docs_df.count()

print(f"\nTotal Documents : {total_documents:,}")

# ==========================================================
# PHASE 1
# DOMAIN STATISTICS
# ==========================================================

print("\n==============================")
print("PHASE 1 : DOMAIN STATISTICS")
print("==============================")

domain_stats = (
    docs_df
    .groupBy("predicted_domain")
    .agg(
        count("*").alias("document_count"),
        mean("token_count").alias("avg_tokens"),
        stddev("token_count").alias("std_tokens"),
        percentile_approx("token_count", 0.5).alias("median_tokens"),
        mean("sentence_count").alias("avg_sentences"),
        mean("char_count").alias("avg_characters"),
        mean("banglish_rate").alias("avg_banglish_rate"),
        spark_sum("token_count").alias("total_tokens"),
    )
)

domain_stats.show(truncate=False)

domain_stats.write.mode("overwrite").csv(
    f"{BUCKET}/output/domain_stats/",
    header=True
)

# ==========================================================
# PHASE 2
# VOCABULARY ANALYSIS
# ==========================================================

print("\n==============================")
print("PHASE 2 : VOCABULARY ANALYSIS")
print("==============================")

# 10% sample
sample_df = docs_df.sample(
    False,
    config["sampling"]["vocabulary"],
    seed=42
)

# Apply Python UDF tokenizer
tokenized_p2 = sample_df.withColumn(
    "tokens",
    bangla_tokenize_udf(col("text"))
)

# Sanity check — print a few token arrays
print("Sample token arrays (first 5 docs):")
tokenized_p2.select("tokens").show(5, truncate=False)

# Explode into one token per row
tokens_df = (
    tokenized_p2
    .select(
        "predicted_domain",
        explode(col("tokens")).alias("token")
    )
)

token_count_check = tokens_df.count()
print(f"\nTotal Bangla tokens in 10% sample: {token_count_check:,}")

if token_count_check == 0:
    print("ERROR: No Bangla tokens found. Check corpus encoding.")
else:
    print(f"SUCCESS: {token_count_check:,} Bangla tokens found.\n")

    # --------------------------------------------------
    # Type Token Ratio per domain
    # --------------------------------------------------

    total_tokens_df = (
        tokens_df
        .groupBy("predicted_domain")
        .count()
        .withColumnRenamed("count", "total_tokens")
    )

    unique_tokens_df = (
        tokens_df
        .select("predicted_domain", "token")
        .distinct()
        .groupBy("predicted_domain")
        .count()
        .withColumnRenamed("count", "unique_tokens")
    )

    ttr_df = (
        total_tokens_df
        .join(unique_tokens_df, "predicted_domain")
        .withColumn(
            "ttr",
            col("unique_tokens") / col("total_tokens")
        )
    )

    ttr_df.show()

    ttr_df.write.mode("overwrite").csv(
        f"{BUCKET}/output/ttr/",
        header=True
    )

    # --------------------------------------------------
    # Top-100 tokens per domain
    # --------------------------------------------------

    token_frequency = (
        tokens_df
        .groupBy("predicted_domain", "token")
        .count()
    )

    window_p2 = Window.partitionBy(
        "predicted_domain"
    ).orderBy(col("count").desc())

    top100_tokens = (
        token_frequency
        .withColumn("rank", row_number().over(window_p2))
        .filter(col("rank") <= 100)
    )

    top100_tokens.show(50, truncate=False)

    top100_tokens.write.mode("overwrite").parquet(
        f"{BUCKET}/output/token_frequency/"
    )

    print("Phase 2 completed successfully.")

# ==========================================================
# PHASE 3
# DUPLICATE DETECTION
#
# APPROACH: Shingle-based near-duplicate detection.
#
# Why not approxSimilarityJoin (MinHash LSH):
#   approxSimilarityJoin performs a distributed cartesian-
#   like join across all document pairs. On 1.4M documents
#   this produces a dataset too large to shuffle across
#   2 worker nodes — the job hangs indefinitely.
#
# Alternative approach — Shingle Fingerprinting:
#   1. For each document, extract the first N Bangla tokens
#      as a "shingle" (a structural fingerprint).
#   2. Group documents by their shingle.
#   3. Any group with >1 document contains near-duplicates.
#
#   This runs as a simple groupBy — O(N) not O(N²).
#   Completes in minutes, not hours.
#   Produces a near-duplicate rate comparable to MinHash
#   for news-heavy corpora where duplication is leading-
#   sentence repetition (same article copy-pasted).
# ==========================================================

print("\n==============================")
print("PHASE 3 : DUPLICATE DETECTION")
print("==============================")
# ----------------------------------------------------------
# Exact Duplicate Removal (SHA-256)
# ----------------------------------------------------------

before_exact = docs_df.count()

deduplicated_df = (
    docs_df
    .dropDuplicates(["sha256"])
    .cache()
)

after_exact          = deduplicated_df.count()
exact_duplicates     = before_exact - after_exact
exact_duplicate_rate = exact_duplicates / before_exact

print(f"Documents Before Deduplication : {before_exact:,}")
print(f"Documents After Deduplication  : {after_exact:,}")
print(f"Exact Duplicates               : {exact_duplicates:,}")
print(f"Exact Duplicate Rate           : {exact_duplicate_rate:.4f}")

# ----------------------------------------------------------
# Near-Duplicate Detection via Shingle Fingerprinting
#
# A shingle = first 8 Bangla tokens of a document joined
# by space. Documents sharing an identical shingle are
# near-duplicates (same opening sentence/paragraph).
# This catches the dominant form of duplication in web
# corpora: news articles copy-pasted across sites.
# ----------------------------------------------------------

SHINGLE_SIZE = int(config.get("duplicate_detection", {}).get("shingle_size", 8))
MIN_SHINGLE_TOKENS = int(config.get("duplicate_detection", {}).get("min_tokens", SHINGLE_SIZE))

print(f"\nRunning shingle-based near-duplicate detection (k={SHINGLE_SIZE})...")

# Work on the full deduplicated corpus — no sampling needed
# because shingle groupBy is O(N) and fast
shingle_df = deduplicated_df.withColumn(
    "tokens",
    bangla_tokenize_udf(col("text"))
)

# Keep only docs with at least 8 Bangla tokens
# (shorter docs produce unreliable shingles)
shingle_df = shingle_df.filter(
    array_size(col("tokens")) >= MIN_SHINGLE_TOKENS
)

shingle_count = shingle_df.count()
print(f"Docs with >= {MIN_SHINGLE_TOKENS} Bangla tokens   : {shingle_count:,}")

# Build shingle: first 8 tokens joined by space
shingle_df = shingle_df.withColumn(
    "shingle",
    concat_ws(
        " ",
        spark_slice(col("tokens"), lit(1), lit(SHINGLE_SIZE))
    )
)

# Group by shingle — each group is a near-duplicate cluster
shingle_groups = (
    shingle_df
    .groupBy("shingle")
    .agg(
        count("*").alias("cluster_size"),
        count("predicted_domain").alias("doc_count")
    )
)

# Near-duplicate clusters = groups with more than 1 document
near_dup_clusters = shingle_groups.filter(
    col("cluster_size") > 1
)

cluster_count       = near_dup_clusters.count()
near_dup_docs_count = near_dup_clusters.agg(
    spark_sum("cluster_size")
).collect()[0][0] or 0

# Documents that are near-duplicates = all docs in clusters > 1
# minus one representative per cluster (the "original")
near_duplicate_documents = int(near_dup_docs_count) - cluster_count
near_duplicate_rate      = (
    near_duplicate_documents / shingle_count
    if shingle_count > 0 else 0.0
)

print(f"Near-Duplicate Clusters        : {cluster_count:,}")
print(f"Near-Duplicate Documents       : {near_duplicate_documents:,}")
print(f"Near-Duplicate Rate            : {near_duplicate_rate:.4f}")

# ----------------------------------------------------------
# Domain-level duplication breakdown
# (which domains have highest copy-paste rate)
# ----------------------------------------------------------

domain_dup_df = (
    shingle_df
    .join(
        near_dup_clusters.select("shingle"),
        on="shingle",
        how="inner"
    )
    .groupBy("predicted_domain")
    .count()
    .withColumnRenamed("count", "near_dup_doc_count")
    .orderBy("near_dup_doc_count", ascending=False)
)

print("\nNear-duplicate documents by domain:")
domain_dup_df.show(truncate=False)

domain_dup_df.write.mode("overwrite").csv(
    f"{BUCKET}/output/domain_dup_breakdown/",
    header=True
)

# ----------------------------------------------------------
# Save largest duplicate clusters as examples
# ----------------------------------------------------------

top_clusters = (
    near_dup_clusters
    .orderBy("cluster_size", ascending=False)
    .limit(50)
)

top_clusters.write.mode("overwrite").csv(
    f"{BUCKET}/output/top_dup_clusters/",
    header=True
)

# ----------------------------------------------------------
# Save deduplication statistics
# ----------------------------------------------------------

duplicate_statistics = {
    "documents_before_exact_deduplication" : before_exact,
    "documents_after_exact_deduplication"  : after_exact,
    "exact_duplicates"                     : exact_duplicates,
    "exact_duplicate_rate"                 : round(exact_duplicate_rate, 4),
    "method"                               : "shingle_fingerprinting",
    "shingle_size"                         : SHINGLE_SIZE,
    "docs_with_enough_tokens"              : shingle_count,
    "near_duplicate_clusters"              : cluster_count,
    "near_duplicate_documents"             : near_duplicate_documents,
    "near_duplicate_rate"                  : round(near_duplicate_rate, 4),
    "note": (
        f"Near-duplicates detected by identical leading-{SHINGLE_SIZE}-token "
        "shingles. Captures copy-pasted news articles and "
        "republished content across domains."
    )
}

delete_gcs_path(f"{BUCKET}/output/dedup_stats/")
spark.sparkContext.parallelize(
    [json.dumps(duplicate_statistics, indent=4)]
).saveAsTextFile(f"{BUCKET}/output/dedup_stats/")

# Free memory before Phase 4
deduplicated_df.unpersist()
shingle_df.unpersist() if shingle_df.is_cached else None

print("Phase 3 completed successfully.")

# ==========================================================
# PHASE 4
# TOPIC MODELING (LDA)
# ==========================================================

print("\n==============================")
print("PHASE 4 : TOPIC MODELING (LDA)")
print("==============================")

# ----------------------------------------------------------
# Reload docs_df — it was unpersisted at end of Phase 3
# to free memory for the shingle join operation.
# Reading from Parquet is fast since workers cache blocks.
# ----------------------------------------------------------

docs_df = (
    spark.read.parquet(f"{BUCKET}/corpus/labelled/")
    .cache()
)

# ----------------------------------------------------------
# Load Bangla stopwords from GCS
# ----------------------------------------------------------

print("Loading Bangla stopwords...")

stopwords_path = f"{BUCKET}/" + config["paths"]["stopwords"]

bangla_stopwords = [
    row.value.strip()
    for row in spark.read.text(stopwords_path).collect()
    if row.value.strip()
]

print(f"Loaded {len(bangla_stopwords):,} stopwords.")

# ----------------------------------------------------------
# 20% sample
# ----------------------------------------------------------

lda_sample = (
    docs_df
    .sample(False, config["sampling"]["lda"], seed=42)
    .cache()
)

print(f"LDA Sample Size : {lda_sample.count():,}")

# ----------------------------------------------------------
# Tokenize with Python UDF
# ----------------------------------------------------------

tokenized_p4 = lda_sample.withColumn(
    "raw_tokens",
    bangla_tokenize_udf(col("text"))
)

# ----------------------------------------------------------
# Remove stopwords
# ----------------------------------------------------------

remover = StopWordsRemover(
    inputCol="raw_tokens",
    outputCol="tokens",
    stopWords=bangla_stopwords
)

filtered = remover.transform(tokenized_p4)

# ----------------------------------------------------------
# Count Vectorizer
# ----------------------------------------------------------

vectorizer = CountVectorizer(
    inputCol="tokens",
    outputCol="features",
    vocabSize=config["lda"]["vocabulary_size"],
    minDF=float(config["lda"]["min_df"])
)

cv_model   = vectorizer.fit(filtered)
vectorized = cv_model.transform(filtered)

print(f"Vocabulary Size : {len(cv_model.vocabulary):,}")

# ----------------------------------------------------------
# LDA Model
# ----------------------------------------------------------

# lda_model_spec = LDA(
#     k=config["lda"]["topics"],
#     maxIter=config["lda"]["max_iterations"],
#     featuresCol="features",
#     optimizer="em",
#     topicConcentration=1.1,
#     seed=42
# )

lda_model_spec = LDA(
    k=config["lda"]["topics"],
    maxIter=config["lda"]["max_iterations"],
    featuresCol="features",
    optimizer="online",
    seed=42
)

lda_model = lda_model_spec.fit(vectorized)

log_perplexity = lda_model.logPerplexity(vectorized)
log_likelihood = lda_model.logLikelihood(vectorized)

print(f"Log Perplexity : {log_perplexity:.4f}")
print(f"Log Likelihood : {log_likelihood:.4f}")

# ----------------------------------------------------------
# Print and collect topics
# ----------------------------------------------------------

topics     = lda_model.describeTopics(15)
vocabulary = cv_model.vocabulary
topic_rows = []

print("\n==============================")
print("DISCOVERED TOPICS")
print("==============================")

for row in topics.collect():
    words = [vocabulary[i] for i in row.termIndices]
    print(f"\nTopic {row.topic}")
    print(", ".join(words))
    topic_rows.append({"topic": row.topic, "terms": words})

# ----------------------------------------------------------
# Save topics JSON
# ----------------------------------------------------------

delete_gcs_path(f"{BUCKET}/output/topics_json/")
spark.sparkContext.parallelize(
    [
        json.dumps(t, ensure_ascii=False, indent=4)
        for t in topic_rows
    ] + [
        json.dumps({
            "log_perplexity"   : round(log_perplexity, 4),
            "log_likelihood"   : round(log_likelihood, 4),
            "num_topics"       : config["lda"]["topics"],
            "max_iterations"   : config["lda"]["max_iterations"],
            "vocabulary_size"  : len(cv_model.vocabulary),
        }, indent=4)
    ]
).saveAsTextFile(f"{BUCKET}/output/topics_json/")

# ----------------------------------------------------------
# Save topics CSV
# ----------------------------------------------------------

topic_df = spark.createDataFrame(
    [(t["topic"], ", ".join(t["terms"])) for t in topic_rows],
    ["topic", "top_words"]
)

topic_df.write.mode("overwrite").csv(
    OUTPUT + "topics_csv/",
    header=True
)

# ----------------------------------------------------------
# Save models
# ----------------------------------------------------------

lda_model.write().overwrite().save(MODEL_PATH + "lda_model/")
cv_model.write().overwrite().save(MODEL_PATH + "count_vectorizer/")

print("Phase 4 completed successfully.")

# ==========================================================
# PHASE 5
# SCALABILITY BENCHMARK
# ==========================================================

print("\n==============================")
print("PHASE 5 : SCALABILITY BENCHMARK")
print("==============================")

# total_documents already computed at line 95 — reuse it
total_token_count = docs_df.agg(
    spark_sum("token_count")
).collect()[0][0]

total_characters = docs_df.agg(
    spark_sum("char_count")
).collect()[0][0]

domain_count = docs_df.select("predicted_domain").distinct().count()

sc              = spark.sparkContext
parallelism     = sc.defaultParallelism
application_id  = sc.applicationId
master          = sc.master
executor_memory = spark.conf.get("spark.executor.memory")
driver_memory   = spark.conf.get("spark.driver.memory")
shuffle_partitions = spark.conf.get("spark.sql.shuffle.partitions")
num_executors   = len(sc._jsc.sc().statusTracker().getExecutorInfos())

benchmark_end  = time.time()
benchmark_time = benchmark_end - benchmark_start

benchmark = {
    "application_id"          : application_id,
    "spark_master"            : master,
    "parallelism"             : parallelism,
    "executors"               : num_executors,
    "executor_memory"         : executor_memory,
    "driver_memory"           : driver_memory,
    "shuffle_partitions"      : shuffle_partitions,
    "documents"               : total_documents,
    "domains"                 : domain_count,
    "tokens"                  : total_token_count,
    "characters"              : total_characters,
    "lda_log_perplexity"      : round(log_perplexity, 4),
    "benchmark_time_seconds"  : round(benchmark_time, 2),
}

print("\n==============================")
print("BENCHMARK SUMMARY")
print("==============================")

for key, value in benchmark.items():
    print(f"{key:30s}: {value}")

delete_gcs_path(OUTPUT + "benchmark/")
spark.sparkContext.parallelize(
    [json.dumps(benchmark, indent=4)]
).saveAsTextFile(OUTPUT + "benchmark/")

print("\nAnalytics Pipeline Completed Successfully!")

spark.stop()