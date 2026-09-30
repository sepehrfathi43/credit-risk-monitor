# Model card

## Status and use

Research and software demonstration using the Home Credit Credit Risk Model Stability dataset. No model trained on the actual competition data is currently distributed. No live lending system, borrower decision, or regulatory compliance claim is made.

The intended task is to estimate the competition's binary target probability and assess stability across application weeks. The exact target definition must be read from the official release. This project does not relabel it as a proven 12-month probability of default.

## Models and features

Logistic regression uses training-only median imputation, missing indicators, and scaling. All-empty columns are retained. LightGBM handles missing values natively. Both models use the same numeric feature list from the build manifest. Calendar fields, targets, and case identifiers are excluded from predictors.

The initial configuration includes static lending amounts/counts and aggregated previous-application counts. Source presence is also included and needs special scrutiny because provider changes can act as time proxies. Hyperparameters are fixed before the first evaluation.

## Validation design

Chronological training, calibration, validation, and test windows are separated by configurable unused weeks. A sigmoid calibrator fits on the calibration window. Validation log loss chooses the champion. Test metrics are reported for both candidates after selection. The tree challenger also receives an additive-contribution check.

The reported metrics are ROC AUC, average precision, log loss, Brier score, equal-width-bin calibration error, calibration bins, and weekly performance. There are no confidence intervals yet. Small weekly cohorts can be noisy. One-class weeks return null ranking metrics.

## Known limits

The precise outcome observation horizon and historical label availability have not been established. Gap weeks alone cannot establish a leakage-free prospective simulation. Repeat borrowers cannot be reliably identified from case IDs. External-provider coverage can shift. There is no reject-inference adjustment, causal identification, or evidence of generalization to a different lender or country.

No protected-group fairness assessment, affordability assessment, policy threshold, cost model, or recovery model has been validated. Risk scores alone do not determine an appropriate lending decision. SHAP contributions are not legally validated adverse-action explanations.

## Monitoring

Frozen-bin PSI and missingness compare incoming features with training data. Score drift uses validation scores as a reference. Alerts request investigation; they do not automatically retrain a model or modify credit policy. Labeled performance is calculated only when labels are supplied, and the operator must verify they are mature and comparable.

## Compute and reproducibility

Feature engineering uses DuckDB with a configurable memory limit and spill directory. Training and scoring use pandas and require the feature matrix to fit in memory. Artifacts record hashes, versions, cohort ranges, metrics, and model identity. Generated joblib files must be treated as executable, trusted artifacts.

## Results

Pending the first authenticated download and real-data run. Test-fixture results must never be substituted into this section.
