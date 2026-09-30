"""Serializable estimators with calibration fitted on a separate time window."""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier


def candidate(name):
    if name == 'logistic':
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
                             StandardScaler(), LogisticRegression(max_iter=2000, random_state=43))
    if name == 'lightgbm':
        return LGBMClassifier(n_estimators=150, num_leaves=15, learning_rate=0.04,
                              min_child_samples=50, reg_lambda=5, random_state=43,
                              n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)
    raise ValueError(name)


def logits(model, x):
    p = np.clip(model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def calibrate(model, x, y):
    return LogisticRegression(C=1e6, max_iter=1000).fit(logits(model, x), y)


def predict(bundle, x):
    return bundle['calibrator'].predict_proba(logits(bundle['model'], x))[:, 1]
