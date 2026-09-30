# Walkthrough

## 1. Verify the software

Create the environment and run `python -m pytest -q`. The tests create temporary records with the expected schema. Their purpose is to check joins, splitting, artifact integrity, and prediction behavior. They do not measure lending performance.

## 2. Build a small first cohort

After downloading the official files:

```powershell
python build_features.py --data data/raw --output runs/features_small --sample-fraction 0.1 --memory 2GB
```

Open `runs/features_small/manifest.json`. Check the source files, application count, and missing-join rates. Review `queries.sql` to see how the features were constructed. Missing related records remain missing; an explicit presence indicator tells the model that the table did not match that case.

The builder rejects duplicate base IDs, duplicate static IDs, duplicate history keys, missing required columns, invalid dates, and non-finite numeric inputs. Every history table is aggregated to case level before joining, so multiple previous applications do not multiply the output rows.

## 3. Inspect the weeks and labels

Run this Python snippet in a shell or notebook:

```python
import pandas as pd
df = pd.read_parquet('runs/features_small/features.parquet')
print(df.groupby('WEEK_NUM').target.agg(['size', 'sum', 'mean']).to_string())
```

Choose three increasing end weeks and a gap. Each retained window must contain at least 20 applications and both outcome classes; that minimum is a software guard, not a claim of statistical adequacy. Use larger cohorts for meaningful estimates.

## 4. Train without reusing the test set

```powershell
python train.py --features runs/features_small --output runs/small_baseline --train-end 50 --calibration-end 65 --validation-end 80 --gap-weeks 4
```

These example boundaries must fit the distribution you inspected. Windows are:

- Training: week <= 50.
- Calibration: 54 < week <= 65.
- Validation: 69 < week <= 80.
- Test: week > 84.

Rows in the gaps are unused. Imputation and scaling are fitted on training rows only. Each model is calibrated on the next window. Validation log loss picks the champion. The test window is evaluated after selection. Do not repeatedly change the project based on test performance and continue calling it an untouched holdout.

The unknown target observation horizon prevents a claim that all labels were available at those historical cutoffs. Document that limitation alongside every result.

## 5. Review the evidence

Open `runs/small_baseline/report.html`. Compare ranking, probability quality, and weekly consistency. Inspect small or one-class weeks before interpreting a bad metric. A missing ranking metric means the cohort cannot support it, not that the model scored zero.

Read `tree_contributions.csv` for the LightGBM challenger. Contributions are calculated on up to 1,000 test cases and checked against the raw model output. They are global mean absolute contributions, not per-applicant explanations.

## 6. Score another cohort

Build its features with the same `features.json` and use `score.py`. The feature-configuration hash must match training. Only load model bundles you trust: joblib uses Python serialization and is not a safe format for unknown files. A checksum detects accidental modification; it is not authentication against a malicious model distributor.

The monitor compares raw features with training distributions and score distributions with validation scores. Bins are frozen from those references. A separate missing bucket catches fields disappearing entirely. The default PSI review threshold of 0.2 is a configurable function argument and a heuristic, not a universal acceptance criterion.

## 7. Record the real run

Use a new output directory for every run. Record dataset release, input checksums, feature changes, split boundaries, package versions, cohort sizes, and observed metrics. Publish only aggregate results allowed by the data terms. Do not check raw data, model bundles, application-level predictions, or credentials into GitHub.

For MLflow, pass a local tracking URI to `train.py`. It logs the run configuration, champion test metrics, and generated artifacts. The first command-line run should be small enough to inspect before expanding to all applications.
