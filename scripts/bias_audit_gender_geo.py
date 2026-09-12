"""
bias_audit_gender_geo.py

DistBanglaCorpus — Gender and Geographic Bias Audit

Performs two bias analyses:

1. Gender-Occupation Co-occurrence Bias
   Measures whether occupations co-occur more with
   male markers (ভাই, বাবা, পুরুষ...) than female
   markers (আপা, মা, নারী...) within a 10-token window.
   Bangla pronouns are gender-neutral (সে = he/she both)
   so we use explicit gendered nouns as anchors instead.
   Outputs log-odds ratio per occupation per domain.

2. Geographic Representation Bias
   Counts mentions of each of Bangladesh's 8 divisions
   across domains to test whether the corpus
   over-represents Dhaka-centric content.

No transformer/GPU dependencies are required; the analysis uses Spark SQL, Python UDFs, and curated Bangla lexicons.
"""

import argparse
import json
import math
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, IntegerType

# ==========================================================
# Spark Session
# ==========================================================

parser = argparse.ArgumentParser(description="Run corpus-level gender and geographic bias audits.")
parser.add_argument("--bucket", required=True, help="Storage root, e.g. gs://my-bucket or file:///abs/path")
args = parser.parse_args()

spark = (
    SparkSession.builder
    .appName("DistBanglaCorpus-BiasAudit-GenderGeo")
    .config("spark.sql.shuffle.partitions", "800")
    .getOrCreate()
)

BUCKET = args.bucket.rstrip("/")
OUTPUT = f"{BUCKET}/output/bias/"

BIAS_RESOURCE_DIR = f"{BUCKET}/bias_resources"

# ==========================================================
# Helper: delete GCS path before saveAsTextFile
# ==========================================================

def delete_gcs_path(path):
    try:
        URI        = spark.sparkContext._gateway.jvm.java.net.URI
        Path       = spark.sparkContext._gateway.jvm.org.apache.hadoop.fs.Path
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
# Load Labelled Corpus
# ==========================================================

print("Loading labelled corpus...")

docs_df = (
    spark.read.parquet(f"{BUCKET}/corpus/labelled/")
    .select("doc_id", "text", "predicted_domain")
    .cache()
)

total = docs_df.count()
print(f"Total documents: {total:,}")

# ==========================================================
# BIAS 1: GENDER-OCCUPATION CO-OCCURRENCE
# ==========================================================

print("\n==============================")
print("BIAS 1 : GENDER-OCCUPATION CO-OCCURRENCE")
print("==============================")

# ----------------------------------------------------------
# Bangla does not have gendered pronouns (সে = he AND she).
# We use explicit gendered nouns/titles as anchors.
# ----------------------------------------------------------

# ==========================================================
# Gender and Occupation Marker Loading
# ==========================================================

def load_marker_file(filename):
    """
    Load a marker file from GCS with explicit UTF-8 decoding.
    Uses binaryFiles() instead of textFile() to avoid the same
    UTF-8 corruption issue that affected parse_shards.py.
    """
    path = f"{BIAS_RESOURCE_DIR}/{filename}"

    content = (
        spark.sparkContext
        .binaryFiles(path)
        .map(lambda kv: kv[1].decode("utf-8", errors="replace"))
        .collect()
    )
    lines = content[0].splitlines() if content else []
    markers = [line.strip() for line in lines if line.strip()]
    return markers




def load_male_markers():
    """
    Load strong / reliable male gender markers.

    IMPORTANT:
    Ambiguous occupational terms such as 'অধ্যাপক',
    'শিক্ষক', 'লেখক', etc. should NOT be placed here
    if they are commonly used for both genders.
    """
    return load_marker_file("male_markers.txt")


def load_female_markers():
    """
    Load strong / reliable female gender markers.

    Explicitly feminine occupational forms such as
    'অধ্যাপিকা', 'শিক্ষিকা', 'লেখিকা', etc. may remain
    in this file because they strongly indicate female gender.
    """
    return load_marker_file("female_markers.txt")


def load_occupations():
    """
    Load all occupation markers.

    This file should contain occupations for all genders,
    including both gender-neutral and explicitly gendered
    occupational forms.

    Examples:
        অধ্যাপক
        অধ্যাপিকা
        সাংবাদিক
        বিচারক
        ডাক্তার
        নারী সাংবাদিক
        মহিলা বিচারক
    """
    return load_marker_file("occupation_markers.txt")

male_markers_list = load_male_markers()
female_markers_list = load_female_markers()
occupations_list = load_occupations()

MALE_MARKERS   = set(male_markers_list)
FEMALE_MARKERS = set(female_markers_list)
OCCUPATIONS    = list(dict.fromkeys(occupations_list))  # ordered, deduplicated

# ==========================================================
# Text Normalization and Multi-word Marker Matching
# ==========================================================

def normalize_text(text):
    """
    Basic Bangla text normalization.

    Handles:
    - repeated whitespace
    - punctuation attached to words
    """

    if not text:
        return ""

    # Normalize all whitespace
    text = re.sub(r"\s+", " ", text)

    # Separate common punctuation
    text = re.sub(
        r"([,।!?;:()\[\]{}\"“”‘’])",
        r" \1 ",
        text
    )

    return text.strip()


def tokenize(text):
    """
    Tokenize normalized Bangla text.
    """
    return normalize_text(text).split()


def marker_to_tuple(marker):
    """
    Convert a marker into a tuple of tokens.

    Example:
        'নারী সাংবাদিক'
        ->
        ('নারী', 'সাংবাদিক')
    """
    return tuple(marker.split())


# ----------------------------------------------------------
# Precompute sorted marker tuples ONCE at module load time.
# find_marker_matches used to rebuild + sort this list on
# every call (twice per document, per occupation) — with a
# 9.9M-doc corpus that's a lot of repeated O(M log M) work
# for a list that never changes. Compute it once here.
# ----------------------------------------------------------

def build_marker_tuples(markers):
    marker_tuples = [
        marker_to_tuple(marker)
        for marker in markers
        if marker.strip()
    ]
    marker_tuples.sort(key=len, reverse=True)
    return marker_tuples

MALE_MARKER_TUPLES   = build_marker_tuples(MALE_MARKERS)
FEMALE_MARKER_TUPLES = build_marker_tuples(FEMALE_MARKERS)


def find_marker_matches(tokens, marker_tuples, search_start=0, search_end=None):
    """
    Find all occurrences of pre-tupled markers within
    tokens[search_start:search_end] (defaults to the whole
    token list). Uses longest-match-first matching.

    marker_tuples must already be tuple-converted and sorted
    longest-first — pass MALE_MARKER_TUPLES / FEMALE_MARKER_TUPLES,
    built once at module load via build_marker_tuples().

    Restricting to [search_start, search_end) instead of always
    scanning the full document matters at scale: callers only
    care about markers within a small window around an
    occupation mention, so there's no reason to walk the rest
    of a possibly very long document to find matches that will
    just be discarded afterward.

    Example:
        'নারী সাংবাদিক'

    is matched as one phrase rather than separately
    matching 'নারী' and 'সাংবাদিক'.
    """

    if search_end is None:
        search_end = len(tokens)

    matches = []

    i = search_start

    while i < search_end:

        matched = False

        for marker_tuple in marker_tuples:

            marker_length = len(marker_tuple)

            if (
                i + marker_length <= search_end
                and
                tuple(tokens[i:i + marker_length])
                == marker_tuple
            ):

                matches.append({
                    "start": i,
                    "end": i + marker_length,
                    "marker": " ".join(marker_tuple),
                    "length": marker_length
                })

                i += marker_length

                matched = True

                break

        if not matched:
            i += 1

    return matches

# ==========================================================
# Contextual Gender Detection for Occupations
# ==========================================================

# Explicit gender prefixes used with occupations.
#
# Examples:
#   নারী সাংবাদিক
#   মহিলা বিচারক
#   পুরুষ সাংবাদিক
#   পুরুষ বিচারক
#
# These are handled dynamically rather than requiring every
# possible combination to be manually stored in the marker files.

FEMALE_OCCUPATION_PREFIXES = {
    "নারী",
    "মহিলা",
}

MALE_OCCUPATION_PREFIXES = {
    "পুরুষ",
}


def is_gendered_occupation_prefix(
    tokens,
    occupation_start,
    occupation_end,
    occupation_tuple
):
    """
    Detect explicit gender prefixes for an occupation, in
    either of two forms:

    A) Baked into the occupation phrase itself — e.g. the
       occupation_markers.txt entry is the compound phrase
       'নারী সাংবাদিক', so occupation_tuple[0] is 'নারী'
       and there is no separate token before it to check.

    B) A separate token immediately before the occupation —
       e.g. occupation is 'সাংবাদিক' and the surrounding
       text happens to read '... নারী সাংবাদিক ...'.

    Checking only (B), as the previous version did, misses
    (A) entirely: if the occupation match already consumes
    the gender prefix as its own first token, occupation_start
    points *after* the prefix and the one-token-back check
    never sees it.

    Returns:
        ("female", 1.0)
        ("male", 1.0)
        (None, 0.0)
    """

    # Case A: prefix is the first token of the occupation
    # phrase itself.
    if occupation_tuple:
        first_token = occupation_tuple[0]
        if first_token in FEMALE_OCCUPATION_PREFIXES:
            return "female", 1.0
        if first_token in MALE_OCCUPATION_PREFIXES:
            return "male", 1.0

    # Case B: prefix is a separate token immediately before
    # the occupation match.
    if occupation_start <= 0:
        return None, 0.0

    prefix = tokens[occupation_start - 1]

    if prefix in FEMALE_OCCUPATION_PREFIXES:
        return "female", 1.0

    if prefix in MALE_OCCUPATION_PREFIXES:
        return "male", 1.0

    return None, 0.0



#===========================================
WINDOW_SIZE = 10

# ==========================================================
# Compute Gender Bias
# ==========================================================

def compute_gender_bias(text, occupation):
    """
    Compute weighted gender bias for an occupation.

    Gender evidence comes from:

    1. Strong male markers
       Example:
           বাবা
           ভাই
           স্বামী

    2. Strong female markers
       Example:
           মা
           বোন
           স্ত্রী

    3. Explicitly gendered occupation phrases
       Example:
           নারী সাংবাদিক
           মহিলা বিচারক
           পুরুষ সাংবাদিক

    4. Explicitly feminine occupational forms
       Example:
           অধ্যাপিকা
           শিক্ষিকা
           লেখিকা

    Ambiguous occupational terms are NOT treated as gender
    evidence.

    Example:

        'তিনি একজন অধ্যাপক।'

    -> অধ্যাপক = occupation
    -> no gender evidence

    But:

        'তিনি একজন নারী অধ্যাপক।'

    -> অধ্যাপক = occupation
    -> নারী = strong female evidence

    Returns:
        positive value -> male-leaning
        negative value -> female-leaning
        None           -> no reliable gender evidence
    """

    if not text or not occupation:
        return None

    tokens = tokenize(text)

    # ------------------------------------------------------
    # Find occurrences of the target occupation
    # ------------------------------------------------------

    occupation_tuple = marker_to_tuple(occupation)

    occupation_positions = []

    occupation_length = len(occupation_tuple)

    for i in range(
        len(tokens) - occupation_length + 1
    ):

        if (
            tuple(
                tokens[
                    i:i + occupation_length
                ]
            )
            == occupation_tuple
        ):

            occupation_positions.append(
                (
                    i,
                    i + occupation_length
                )
            )

    if not occupation_positions:
        return None

    # ------------------------------------------------------
    # Initialize scores
    # ------------------------------------------------------

    male_score = 0.0
    female_score = 0.0

    # ------------------------------------------------------
    # Analyze each occurrence of the occupation
    # ------------------------------------------------------

    for (
        occupation_start,
        occupation_end
    ) in occupation_positions:

        # --------------------------------------------------
        # Window around occupation
        # --------------------------------------------------

        window_start = max(
            0,
            occupation_start - WINDOW_SIZE
        )

        window_end = min(
            len(tokens),
            occupation_end + WINDOW_SIZE
        )

        # --------------------------------------------------
        # Find gender marker occurrences, restricted to this
        # occurrence's window. Markers use precomputed
        # MALE_MARKER_TUPLES / FEMALE_MARKER_TUPLES (sorted
        # once at module load) rather than rebuilding and
        # re-sorting them on every call, and only the window
        # is scanned rather than the whole document.
        # --------------------------------------------------

        male_matches = find_marker_matches(
            tokens, MALE_MARKER_TUPLES, window_start, window_end
        )

        female_matches = find_marker_matches(
            tokens, FEMALE_MARKER_TUPLES, window_start, window_end
        )

        # ==================================================
        # 1. Explicit gendered occupation prefix — checks
        #    both a prefix baked into the occupation phrase
        #    itself (e.g. 'নারী সাংবাদিক') and a separate
        #    token immediately before it.
        # ==================================================

        gender, weight = is_gendered_occupation_prefix(
            tokens,
            occupation_start,
            occupation_end,
            occupation_tuple
        )

        if gender == "male":
            male_score += weight

        elif gender == "female":
            female_score += weight

        # ==================================================
        # 2. Strong male markers in context
        # ==================================================

        for match in male_matches:

            marker_start = match["start"]
            marker_end = match["end"]

            # Exact same span as the occupation itself: the
            # occupation IS an explicitly gendered form (e.g.
            # অধ্যাপিকা listed in both occupation_markers.txt
            # and female_markers.txt) — that's strong direct
            # evidence, not noise, so count it.
            if (
                marker_start == occupation_start
                and marker_end == occupation_end
            ):
                male_score += 1.0
                continue

            # Partial (non-identical) overlap with the
            # occupation span is a coincidental collision —
            # skip it rather than double-counting.
            if (
                marker_start < occupation_end
                and
                marker_end > occupation_start
            ):
                continue

            male_score += 1.0

        # ==================================================
        # 3. Strong female markers in context
        # ==================================================

        for match in female_matches:

            marker_start = match["start"]
            marker_end = match["end"]

            if (
                marker_start == occupation_start
                and marker_end == occupation_end
            ):
                female_score += 1.0
                continue

            if (
                marker_start < occupation_end
                and
                marker_end > occupation_start
            ):
                continue

            female_score += 1.0

    # ------------------------------------------------------
    # No reliable gender evidence
    # ------------------------------------------------------

    if (
        male_score == 0
        and
        female_score == 0
    ):
        return None

    # ------------------------------------------------------
    # Smoothed log-odds ratio
    # ------------------------------------------------------

    score = math.log(
        (male_score + 0.5)
        /
        (female_score + 0.5)
    )

    return float(score)



print(f"Window size: (+/-){WINDOW_SIZE} tokens")
print(f"Male markers: {len(MALE_MARKERS)} unique "
      f"({len(male_markers_list)} total entries)")

print(f"Female markers: {len(FEMALE_MARKERS)} unique "
      f"({len(female_markers_list)} total entries)")

print(f"Occupations: {len(OCCUPATIONS)} unique "
      f"({len(occupations_list)} total entries)")

bias_rows = []

for occupation in OCCUPATIONS:

    bias_udf = udf(
        lambda text, occ=occupation: compute_gender_bias(text, occ),
        FloatType()
    )

    occ_df = (
        docs_df
        .filter(col("text").contains(occupation))
        .withColumn("bias_score", bias_udf(col("text")))
        .filter(col("bias_score").isNotNull())
    )

    # Overall bias score for this occupation
    overall = occ_df.agg(
        mean("bias_score").alias("avg_bias"),
        count("bias_score").alias("doc_count")
    ).collect()[0]

    avg_bias  = overall["avg_bias"]
    doc_count = overall["doc_count"]

    if avg_bias is not None:
        direction = "male-leaning" if avg_bias > 0 else "female-leaning"
        print(
            f"  {occupation:20s}: {avg_bias:+.4f}  "
            f"({direction})  n={doc_count:,}"
        )
    else:
        direction = "insufficient data"
        print(f"  {occupation:20s}: no data")

    # Per-domain breakdown
    domain_bias = (
        occ_df
        .groupBy("predicted_domain")
        .agg(
            mean("bias_score").alias("avg_bias"),
            count("bias_score").alias("doc_count")
        )
        .collect()
    )

    for row in domain_bias:
        bias_rows.append({
            "occupation"       : occupation,
            "domain"           : row["predicted_domain"],
            "avg_bias_score"   : round(float(row["avg_bias"]), 4)
                                 if row["avg_bias"] is not None else None,
            "doc_count"        : int(row["doc_count"]),
            "direction"        : (
                "male-leaning"
                if row["avg_bias"] is not None and row["avg_bias"] > 0
                else "female-leaning"
            ),
        })

    bias_rows.append({
        "occupation"     : occupation,
        "domain"         : "ALL",
        "avg_bias_score" : round(float(avg_bias), 4)
                           if avg_bias is not None else None,
        "doc_count"      : int(doc_count),
        "direction"      : direction,
    })

# Serialize bias rows for JSON output

bias_rows_serializable = [
    {
        "occupation"     : r["occupation"],
        "domain"         : r["domain"],
        "avg_bias_score" : r["avg_bias_score"],
        "doc_count"      : str(r["doc_count"]),
        "direction"      : r["direction"],
    }
    for r in bias_rows
]

delete_gcs_path(f"{OUTPUT}gender_bias/")
spark.sparkContext.parallelize(
    [json.dumps(r, ensure_ascii=False) for r in bias_rows_serializable]
).saveAsTextFile(f"{OUTPUT}gender_bias/")

print(f"\nGender bias results saved to {OUTPUT}gender_bias/")

# ==========================================================
# BIAS 2: GEOGRAPHIC REPRESENTATION
# ==========================================================

print("\n==============================")
print("BIAS 2 : GEOGRAPHIC REPRESENTATION")
print("==============================")

# Bangladesh's 8 administrative divisions
# Include common alternate spellings
DIVISIONS = {
    "Dhaka":       ["ঢাকা", "ঢাকাইয়া", "ডাকা", "জাহাঙ্গীরনগর"],
    "Chittagong":  ["চট্টগ্রাম", "চিটাগং", "চাটগাইয়া", "চাটগাঁইয়া", "চাটগাঁ", "চাটগা", "চিটাগাং", "চাটগাও"],
    "Sylhet":      ["সিলেট", "ছিলট", "সিলট", "সিলেটি", "শ্রীহট্ট", "জালালাবাদ"],
    "Rajshahi":    ["রাজশাহী", "রাজশাহি", "রামপুর বোয়ালিয়া"],
    "Khulna":      ["খুলনা", "খুলনাইয়া", "জাহানাবাদ"],
    "Barisal":     ["বরিশাল", "বইশাল", "বইরশাল", "বরিশাইল্যা", "চন্দ্রদ্বীপ", "বাকেরগঞ্জ"],
    "Rangpur":     ["রংপুর", "রাংপুর", "রংপুরী", "রংপুরিয়া"],
    "Mymensingh":  ["ময়মনসিংহ", "ময়মনসিং", "মৈমনসিং", "মোমেনশাহী", "নাসিরাবাদ"],
}

geo_results = []

print("\nDivision mention counts:")

def count_matches(text, pattern):
    """
    Counts non-overlapping regex matches of `pattern` in
    `text`. Portable replacement for pyspark.sql.functions
    .regexp_count, which requires PySpark 3.5+ (this cluster
    is on an older Spark runtime).
    """
    if not text:
        return 0
    return len(re.findall(pattern, text))

for division_en, spellings in DIVISIONS.items():

    # Use regexp with alternation for multiple spellings
    pattern = "|".join(spellings)

    count_udf = udf(
        lambda text, p=pattern: count_matches(text, p),
        IntegerType()
    )

    mention_df = docs_df.withColumn(
        "mention_count",
        count_udf(col("text"))
    )

    # Total mentions across corpus
    total_mentions = mention_df.agg(
        spark_sum("mention_count")
    ).collect()[0][0] or 0

    print(f"  {division_en:15s}: {int(total_mentions):>10,} mentions")

    # Per-domain breakdown
    domain_mentions = (
        mention_df
        .groupBy("predicted_domain")
        .agg(spark_sum("mention_count").alias("mentions"))
        .collect()
    )

    for row in domain_mentions:
        geo_results.append({
            "division"         : division_en,
            "domain"           : row["predicted_domain"],
            "total_mentions"   : int(row["mentions"] or 0),
        })

    geo_results.append({
        "division"       : division_en,
        "domain"         : "ALL",
        "total_mentions" : int(total_mentions),
    })

# Save geographic bias results
delete_gcs_path(f"{OUTPUT}geographic_bias/")
spark.sparkContext.parallelize(
    [json.dumps(r, ensure_ascii=False) for r in geo_results]
).saveAsTextFile(f"{OUTPUT}geographic_bias/")

print(f"\nGeographic bias results saved to {OUTPUT}geographic_bias/")

# ==========================================================
# SUMMARY REPORT (gender + geographic only)
# ==========================================================

print("\n==============================")
print("GENDER / GEOGRAPHIC BIAS SUMMARY")
print("==============================")

print("\nTop 5 most male-leaning occupations (corpus-wide):")
all_bias = [
    r for r in bias_rows
    if r["domain"] == "ALL"
    and r["avg_bias_score"] is not None
]
all_bias.sort(
    key=lambda x: x["avg_bias_score"] or 0,
    reverse=True
)
for r in all_bias[:5]:
    print(
        f"  {r['occupation']:20s}: {r['avg_bias_score']:+.4f}  "
        f"n={r['doc_count']:,}"
    )

print("\nTop 5 most female-leaning occupations (corpus-wide):")
for r in all_bias[-5:]:
    print(
        f"  {r['occupation']:20s}: {r['avg_bias_score']:+.4f}  "
        f"n={r['doc_count']:,}"
    )

print("\nGeographic representation (corpus-wide):")
geo_all = [r for r in geo_results if r["domain"] == "ALL"]
geo_all.sort(key=lambda x: x["total_mentions"], reverse=True)
total_geo = sum(r["total_mentions"] for r in geo_all)
for r in geo_all:
    pct = r["total_mentions"] / total_geo * 100 if total_geo > 0 else 0
    print(
        f"  {r['division']:15s}: {r['total_mentions']:>10,}  "
        f"({pct:.1f}%)"
    )

# Save gender + geographic summary
# (sentiment_bias is appended separately by
#  bias_audit_sentiment.py — see that script's summary output)
summary = {
    "gender_bias"     : bias_rows_serializable,
    "geographic_bias" : geo_results,
}

delete_gcs_path(f"{OUTPUT}summary_gender_geo/")
spark.sparkContext.parallelize(
    [json.dumps(summary, ensure_ascii=False, indent=2)]
).saveAsTextFile(f"{OUTPUT}summary_gender_geo/")

print(f"\nGender/geographic summary saved to {OUTPUT}summary_gender_geo/")
print("\nGender + Geographic Bias Audit Completed Successfully!")

spark.stop()