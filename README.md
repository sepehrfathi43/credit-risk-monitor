# Credit Risk Monitor

**Does a credit model still work when the applicant population changes?**

This project uses Home Credit's lending data to compare an interpretable baseline with a tree model, then follow their performance across later application weeks. The aim is to make the entire decision process inspectable: where features came from, how the holdout was chosen, whether probabilities are calibrated, and what changes would warrant a review.

**Current status:** the data pipeline, training, batch scoring, and reporting are implemented. Automated tests use small artificial records to check software behavior. A run on the actual competition files is pending authenticated Kaggle access. There are no claimed real-data performance results yet.

## Data

The source is [Home Credit — Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data), a real lending dataset with application records and related credit histories. Access requires a Kaggle account and compliance with the competition's terms. Raw data is not included in this repository.

The starting feature set deliberately stays small: credit amount, active credit count, debt fields, previous-application counts, and source-table availability. `features.json` defines every input. Additional numeric history fields can be added there without changing the SQL builder. This is a baseline feature set, not a competition-winning solution.

## What the code does

| Stage | Implementation | Why it matters |
|---|---|---|
| Data preparation | DuckDB joins and aggregates Parquet shards; checks keys, schemas, and numeric values | A duplicate join can quietly invalidate an otherwise good model |
| Provenance | SHA-256 input checksums, saved feature configuration, SQL log | A result should be traceable to the files and transformations that produced it |
| Modeling | Logistic regression and LightGBM with fixed initial settings | Establishes a useful baseline before tuning |
| Evaluation | Separate chronological training, calibration, validation, and test windows | Keeps model selection out of the final test set |
| Calibration | Sigmoid calibration fitted only on the calibration window | Risk estimates need useful probabilities as well as ranking |
| Monitoring | Frozen reference bins, feature/score PSI, missingness, weekly labeled metrics | Helps distinguish population changes from deteriorating predictions |
| Interpretation | Native TreeSHAP contributions for the LightGBM challenger | Shows which inputs influence raw tree scores |
| Delivery | Saved model bundle, batch scoring, HTML report, optional MLflow tracking | Makes the analysis repeatable outside a notebook |

DuckDB can spill to disk during feature engineering. Training and scoring currently load the resulting feature table into memory. This is a single-machine implementation; it does not claim a distributed production deployment.

## Run it

Use Python 3.12. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. Obtain the official files using [DATA_ACCESS.md](DATA_ACCESS.md), then follow [WALKTHROUGH.md](WALKTHROUGH.md).

```powershell
python build_features.py --data data/raw --output runs/features --memory 2GB
```

Inspect the week distribution before choosing split boundaries. The following is an **example configuration**, not an empirically validated maturity gap:

```powershell
python train.py --features runs/features --output runs/baseline --train-end 50 --calibration-end 65 --validation-end 80 --gap-weeks 4
```

The run writes `results.json`, `report.html`, `calibration.png`, `tree_contributions.csv`, test predictions, and a model bundle. To track the same run locally, add `--tracking-uri sqlite:///mlflow.db`. Do not publish individual application scores or raw data without permission under the data terms.

To score the competition's unlabeled test files:

```powershell
python build_features.py --data data/raw --split test --output runs/test_features
python score.py --model runs/baseline --features runs/test_features --output runs/test_scores
```

Scoring writes probabilities and a monitoring report. Without labels, performance metrics are left empty. PSI flags are prompts for investigation, not automatic rejection or retraining rules.

## How to read the results

The champion is selected by **validation log loss**, before test outcomes are scored. Both candidates receive test ROC AUC, average precision, log loss, Brier score, calibration bins, and weekly results. AUC measures ranking; calibration measures agreement between predicted and observed risk. Neither establishes that a lending policy will be profitable or fair.

There are several limitations worth keeping visible:

- The released target definition does not provide enough information here to prove labels had matured at each simulated training date. Temporal splits are retrospective; the configured gap does not solve that on its own.
- `case_id` identifies an application case. It is not a verified persistent borrower ID, so repeat-borrower leakage cannot be ruled out.
- Data-provider availability can change over time. Missingness and source-presence indicators may capture those changes rather than borrower risk.
- TreeSHAP explains the uncalibrated tree score in log-odds units. It is not a causal explanation or a validated adverse-action reason.
- Threshold optimization, recovery modeling, fairness validation, and live lending decisions are outside the current scope.

See [MODEL_CARD.md](MODEL_CARD.md) for the intended use and [VALIDATION.md](VALIDATION.md) for the verification record.

## Next research step

Run the official data, record the cohort counts and measured results, and inspect the worst-performing weeks. Expand features only after the baseline and leakage checks are understood. That is the evidence needed before describing this as a completed credit-risk study.
