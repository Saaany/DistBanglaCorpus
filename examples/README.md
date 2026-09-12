# Local smoke test

The sample corpus is synthetic and is included only to validate that the pipeline wiring works. It is **not** intended to reproduce paper metrics.

Requirements: Java, Python 3.10+, and PySpark (see `requirements.txt`).

Run from the repository root:

```bash
bash examples/run_local_smoke.sh
```

The script creates a temporary local storage root, copies the sample shards and lexicons into the same directory structure used on GCS, and runs the four Spark stages with `local[2]`.
