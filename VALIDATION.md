# Verification record

Local verification: September 30, 2026, Windows, Python 3.12, dependencies pinned in `requirements.txt`.

`python -m pytest -q`: **12 passed** in 24.70 seconds. The run exercised:

- Parquet feature construction across shards, case-level history aggregation, and missing join indicators.
- Duplicate base, static, and history-key rejection.
- Invalid numeric values and missing required columns.
- Chronological windows with unused gap weeks.
- Both models, separate sigmoid calibration, and validation-based selection.
- TreeSHAP additivity against raw LightGBM scores.
- HTML/chart generation and SQLite-backed MLflow artifact/metric logging.
- Labeled and unlabeled batch scoring.
- Rejection of mismatched feature configurations and modified model artifacts.
- Constant/all-missing reference distributions and PSI alerts.
- Undefined ranking metrics for one-class cohorts.
- Deterministic case sampling.
- Unchanged model selection when only final test labels change.

One third-party SQLAlchemy deprecation warning arose inside MLflow. Logging completed successfully; the warning should be revisited when upgrading those dependencies.

The test records are generated temporarily and contain no real borrower data. These checks demonstrate software behavior, not predictive performance. Tests also do not establish security against a malicious model artifact, economic value, fairness, or suitability for lending decisions.

## Outstanding verification

An authenticated download and run on the actual Home Credit release is still needed. That run must confirm the selected source schema, cohort sizes, temporal coverage, target interpretation, memory use, and measured model performance. No full-dataset runtime or real-data accuracy is claimed here.

GitHub Actions is configured to run the same tests on Ubuntu/Python 3.12. Its live status is available from the repository's Actions tab; the local test result above should not be interpreted as a claim that a remote workflow has passed.
