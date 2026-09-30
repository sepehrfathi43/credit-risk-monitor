"""Shared artifact and numeric-data contracts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

META = {'case_id', 'target', 'WEEK_NUM', 'MONTH', 'date_decision'}


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_cases(frame, labels=False):
    required = {'case_id', 'WEEK_NUM'} | ({'target'} if labels else set())
    if not required.issubset(frame):
        raise ValueError(f'Missing required columns: {sorted(required - set(frame))}')
    if frame.empty or frame.case_id.isna().any() or frame.case_id.duplicated().any():
        raise ValueError('Empty data, null case IDs, or duplicate case IDs')
    weeks = pd.to_numeric(frame.WEEK_NUM, errors='raise')
    if weeks.isna().any() or not np.isfinite(weeks).all() or (weeks % 1 != 0).any():
        raise ValueError('WEEK_NUM must contain finite integers')
    if labels and (frame.target.isna().any() or not frame.target.isin([0, 1]).all()):
        raise ValueError('target must contain only 0 and 1')


def numeric_frame(frame, features):
    missing = set(features) - set(frame)
    if missing:
        raise ValueError(f'Missing model features: {sorted(missing)}')
    result = frame[features].apply(pd.to_numeric, errors='raise').astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError('Infinite feature values are not supported')
    return result
