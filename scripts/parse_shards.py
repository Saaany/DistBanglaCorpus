"""
parse_shards.py
Runs on Dataproc master node.
Reads raw .txt shards from GCS, converts to Parquet.

Uses Hadoop TextInputFormat via newAPIHadoopFile to stream
UTF-8 text line-by-line while preserving Bangla Unicode.
"""
import argparse
import hashlib
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import (FloatType, IntegerType, StringType, StructField,
                               StructType)

parser = argparse.ArgumentParser(description="Parse Bangla2B+ raw shards into structured Parquet records.")
parser.add_argument("--bucket", required=True, help="Storage root, e.g. gs://my-bucket or file:///abs/path")
args = parser.parse_args()

spark = SparkSession.builder \
    .appName("DistBanglaCorpus-Parse") \
    .getOrCreate()

sc = spark.sparkContext
BUCKET = args.bucket.rstrip("/")

# ── Step 1: Read shards line-by-line forcing UTF-8 ──────
# Instead of binaryFiles (which loads whole files into memory), we use 
# the underlying Hadoop text input format configuration. This streams the 
# files line-by-line natively while strictly forcing UTF-8 encoding.

conf = {
    "mapreduce.input.fileinputformat.inputdir": f"{BUCKET}/corpus/raw/*.txt",
    "mapreduce.input.keyvaluelineinputformat.key.regex": "(.*)",
    "io.serializations": "org.apache.hadoop.io.serializer.JavaSerialization"
}

raw_rdd = (
    sc.newAPIHadoopFile(
        f"{BUCKET}/corpus/raw/*.txt",
        "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
        "org.apache.hadoop.io.LongWritable",
        "org.apache.hadoop.io.Text",
        conf=conf
    )
    # The value (v) is a Java Text object, which Spark automatically 
    # converts to a properly encoded UTF-8 Python string.
    .map(lambda kv: kv[1]) 
)

# Quick sanity check — verify Bangla chars present in raw data
sample_lines = raw_rdd.take(200)
bangla_lines = [
    l for l in sample_lines
    if any('\u0980' <= ch <= '\u09FF' for ch in l)
]
print(f"\nSanity check: {len(bangla_lines)}/200 sample lines contain Bangla chars")
if len(bangla_lines) == 0:
    raise RuntimeError(
        "No Bangla characters found in raw shards. "
        "Check that shard files are valid UTF-8 encoded Bangla text."
    )

# ── Step 2: Parse into documents ──────────────────────────────
def parse_partition(lines):
    """
    Group consecutive non-empty lines into documents.
    Empty line = document boundary.
    """
    current_sentences = []
    file_doc_idx = 0

    for line in lines:
        if line.strip() == "":
            if current_sentences:
                text = " ".join(current_sentences)
                sha256 = hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest()

                # Count Bangla vs Latin tokens
                tokens = text.split()
                bangla_tokens = sum(
                    1 for t in tokens
                    if any('\u0980' <= c <= '\u09FF' for c in t)
                )
                latin_tokens = sum(
                    1 for t in tokens
                    if re.match(r'^[a-zA-Z]{2,}$', t)
                )
                banglish_rate = (
                    latin_tokens / len(tokens)
                    if tokens else 0.0
                )

                yield {
                    "doc_id": sha256,
                    "text": text,
                    "sentence_count": len(current_sentences),
                    "token_count": len(tokens),
                    "char_count": len(text),
                    "bangla_token_count": bangla_tokens,
                    "latin_token_count": latin_tokens,
                    "banglish_rate": round(banglish_rate, 4),
                    "sha256": sha256,
                    "predicted_domain": "unknown"
                }
                file_doc_idx += 1
                current_sentences = []
        else:
            current_sentences.append(line.strip())

    # Last document (no trailing newline)
    if current_sentences:
        text = " ".join(current_sentences)
        sha256 = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        tokens = text.split()
        yield {
            "doc_id": sha256,
            "text": text,
            "sentence_count": len(current_sentences),
            "token_count": len(tokens),
            "char_count": len(text),
            "bangla_token_count": sum(
                1 for t in tokens
                if any('\u0980' <= c <= '\u09FF' for c in t)
            ),
            "latin_token_count": sum(
                1 for t in tokens
                if re.match(r'^[a-zA-Z]{2,}$', t)
            ),
            "banglish_rate": round(
                (sum(1 for t in tokens if re.match(r'^[a-zA-Z]{2,}$', t)) / len(tokens))
                if tokens else 0.0, 4
            ),
            "sha256": sha256,
            "predicted_domain": "unknown"
        }

docs_rdd = raw_rdd.mapPartitions(parse_partition)

# ── Step 3: Convert to DataFrame ──────────────────────────────
schema = StructType([
    StructField("doc_id",              StringType()),
    StructField("text",                StringType()),
    StructField("sentence_count",      IntegerType()),
    StructField("token_count",         IntegerType()),
    StructField("char_count",          IntegerType()),
    StructField("bangla_token_count",  IntegerType()),
    StructField("latin_token_count",   IntegerType()),
    StructField("banglish_rate",       FloatType()),
    StructField("sha256",              StringType()),
    StructField("predicted_domain",    StringType()),
])

docs_df = spark.createDataFrame(docs_rdd, schema=schema)
docs_df = docs_df.withColumn(
    "id", monotonically_increasing_id()
)

# ── Step 4: Save as Parquet (faster than JSONL for Spark) ─────
docs_df.write.mode("overwrite").parquet(
    f"{BUCKET}/corpus/parsed/"
)

total = docs_df.count()
print(f"\n✓ Total documents parsed: {total:,}")
docs_df.printSchema()
docs_df.show(5, truncate=80)

spark.stop()