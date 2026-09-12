"""
domain_inference.py

Assigns domain labels to each parsed document using:
1. Domain lexicons loaded from external files
2. Structural signals
3. Confidence-based rejection

DistBanglaCorpus
"""

import argparse
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

parser = argparse.ArgumentParser(description="Infer document domains using Bangla lexicons and structural rules.")
parser.add_argument("--bucket", required=True, help="Storage root, e.g. gs://my-bucket or file:///abs/path")
args = parser.parse_args()

spark = SparkSession.builder \
    .appName("DistBanglaCorpus-DomainInference") \
    .getOrCreate()

BUCKET = args.bucket.rstrip("/")

# --------------------------------------------------
# Load domain lexicons from external files
# --------------------------------------------------

LEXICON_DIR = f"{BUCKET}/domain_resources/lexicons"

DOMAIN_LIST = [
    "news",
    "encyclopedia",
    "education",
    "blogs",
    "social",
    "literature"
]

def load_domain_seeds_spark(seed_dir):
    domain_seeds = {}
    for domain in DOMAIN_LIST:
        path = f"{seed_dir}/{domain}.txt"
        # Reads the file as a Spark DataFrame and collects rows to a local driver list
        rdd = spark.sparkContext.textFile(path)
        seeds = [line.strip() for line in rdd.collect() if line.strip()]
        domain_seeds[domain] = seeds
    return domain_seeds


DOMAIN_SEEDS = load_domain_seeds_spark(LEXICON_DIR)

print("\n=== DOMAIN LEXICON STATISTICS ===")

for domain, seeds in DOMAIN_SEEDS.items():
    print(f"{domain}: {len(seeds)} seeds")


# --------------------------------------------------
# Load Parsed Corpus
# --------------------------------------------------
docs_df = spark.read.parquet(f"{BUCKET}/corpus/parsed/")

def infer_domain(text):
    if not text:
        return "unknown"

    scores = {d: 0 for d in DOMAIN_SEEDS}

    # --------------------------------------------------
    # CHANGE: Lexicon Matching
    # Multiple occurrences contribute to score
    # --------------------------------------------------

    for domain, seeds in DOMAIN_SEEDS.items():

        scores[domain] += sum(
            text.count(
                seed
            )
            for seed in seeds
        )

    # --------------------------------------------------
    # NEWS SIGNALS
    # --------------------------------------------------

    if re.search(
        r'(নিজস্ব প্রতিবেদক|স্টাফ রিপোর্টার|'
        r'বিশেষ প্রতিনিধি)',
        text
    ):
        scores["news"] += 4

    # CHANGE: stronger metadata detection

    if re.search(
        r'(প্রকাশিত|সর্বশেষ আপডেট|'
        r'সংবাদ সম্মেলন|প্রেস বিজ্ঞপ্তি)',
        text
    ):
        scores["news"] += 6

    # CHANGE: news-style date patterns

    if re.search(
        r'\d{1,2}\s+'
        r'(জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|'
        r'মে|জুন|জুলাই|আগস্ট|সেপ্টেম্বর|'
        r'অক্টোবর|নভেম্বর|ডিসেম্বর)',
        text
    ):
        scores["news"] += 2

    # --------------------------------------------------
    # ENCYCLOPEDIA SIGNALS
    # --------------------------------------------------

    if re.search(
        r'(তথ্যসূত্র|বহিঃসংযোগ|'
        r'বিষয়শ্রেণী|আরও দেখুন)',
        text
    ):
        scores["encyclopedia"] += 6

    if re.search(
        r'(জাতীয়তা|পেশা|জন্ম|মৃত্যু)',
        text
    ):
        scores["encyclopedia"] += 3

    # --------------------------------------------------
    # EDUCATION SIGNALS
    # --------------------------------------------------

    if re.search(
        r'(সেমিস্টার|জিপিএ|সিজিপিএ|'
        r'গবেষণা|থিসিস|'
        r'অ্যাসাইনমেন্ট|মিডটার্ম|'
        r'ফাইনাল)',
        text
    ):
        scores["education"] += 4

    # --------------------------------------------------
    # BLOG SIGNALS
    # --------------------------------------------------

    first_person_count = len(
        re.findall(
            r'\b(আমি|আমার|আমাকে|আমাদের)\b',
            text
        )
    )

    if first_person_count >= 5:
        scores["blogs"] += 4

    # --------------------------------------------------
    # SOCIAL SIGNALS
    # --------------------------------------------------

    if re.search(
        r'[\U0001F300-\U0001FAFF]',
        text
    ):
        scores["social"] += 3

    if re.search(
        r'[@#][A-Za-z0-9_]+',
        text
    ):
        scores["social"] += 4

    if re.search(
        r'[!?]{3,}',
        text
    ):
        scores["social"] += 2

    # --------------------------------------------------
    # Banglish Rate
    # --------------------------------------------------
    word_list = text.split()

    if word_list:

        latin_count = sum(
            1
            for word in word_list
            if re.match(
                r'^[a-zA-Z]{2,}$',
                word
            )
        )

        banglish_ratio = (
            latin_count /
            len(word_list)
        )

        if banglish_ratio > 0.15:
            scores["social"] += 2


    # --------------------------------------------------
    # Short Documents
    # --------------------------------------------------
    if len(word_list) < 30:
        scores["social"] += 1

    # --------------------------------------------------
    # Literature Signals
    # --------------------------------------------------
    dialogue_count = len(
        re.findall(
            r'[“"].+?[”"]',
            text
        )
    )

    if dialogue_count >= 5:
        scores["literature"] += 4

    # best_score = max(scores.values())
    # if best_score == 0:
    #     return "unknown"
    # --------------------------------------------------
    # Confidence-Based Rejection
    # -------------------------------------------------
    best_domain = max(
        scores,
        key=scores.get
    )

    best_score = scores[
        best_domain
    ]

    total_score = sum(
        scores.values()
    )

    if best_score == 0:
        return "unknown"

    confidence = (
        best_score /
        total_score
    )

    # CHANGE:
    # Reject uncertain predictions

    if confidence < 0.35:
        return "unknown"

    return best_domain

    # return max(scores, key=scores.get)

# --------------------------------------------------
# Spark UDF
# --------------------------------------------------
domain_udf = udf(
    infer_domain,
    StringType()
)

# --------------------------------------------------
# Predict Domains
# --------------------------------------------------
docs_df = docs_df.withColumn(
    "predicted_domain",
    domain_udf(
        col("text")
    )
)

# --------------------------------------------------
# Save Results
# --------------------------------------------------
docs_df.write \
    .mode("overwrite") \
    .parquet(
        f"{BUCKET}/corpus/labelled/"
    )

# --------------------------------------------------
# Domain Distribution
# --------------------------------------------------
print("\n=== DOMAIN DISTRIBUTION ===")

docs_df.groupBy(
    "predicted_domain"
).count() \
 .orderBy(
    "count",
    ascending=False
 ) \
 .show(
    truncate=False
 )

# Print distribution
# print("\n=== DOMAIN DISTRIBUTION ===")
# docs_df.groupBy("predicted_domain") \
#        .count() \
#        .orderBy("count", ascending=False) \
#        .show()

spark.stop()