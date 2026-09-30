"""Train -> calibrate -> select -> test, with explicit chronological boundaries."""
import argparse
import importlib.metadata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from common import digest, load_json, save_json, validate_cases, numeric_frame
from evaluate import metrics
from modeling import candidate, calibrate, predict
from monitoring import profile, compare
from report import render


def time_windows(frame, train_end, calibration_end, validation_end, gap):
    if not train_end < calibration_end < validation_end or gap < 0:
        raise ValueError('Boundaries must increase; gap must be nonnegative')
    w = frame.WEEK_NUM
    masks = {'train': w <= train_end,
             'calibration': (w > train_end + gap) & (w <= calibration_end),
             'validation': (w > calibration_end + gap) & (w <= validation_end),
             'test': w > validation_end + gap}
    windows = {k: frame.loc[v].copy() for k, v in masks.items()}
    for name, part in windows.items():
        if len(part) < 20 or part.target.nunique() != 2:
            raise ValueError(f'{name} requires at least 20 cases and both labels')
    return windows


def train(feature_dir, output, train_end, calibration_end, validation_end, gap, tracking_uri=None):
    feature_dir, output = Path(feature_dir), Path(output)
    if output.exists():
        raise FileExistsError('Choose a new run directory')
    manifest = load_json(feature_dir / 'manifest.json')
    path = feature_dir / 'features.parquet'
    if manifest['status'] != 'complete' or digest(path) != manifest['feature_sha256']:
        raise ValueError('Feature manifest or checksum is invalid')
    frame = pd.read_parquet(path)
    validate_cases(frame, labels=True)
    features = manifest['features']
    numeric_frame(frame, features)
    windows = time_windows(frame, train_end, calibration_end, validation_end, gap)
    x = {k: numeric_frame(v, features) for k, v in windows.items()}
    result = {'data_kind': manifest['data_kind'], 'feature_sha256': digest(path),
              'features': features, 'gap_weeks': gap, 'models': {}, 'windows': {},
              'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'duckdb']}}
    for name, part in windows.items():
        result['windows'][name] = {'n': len(part), 'min_week': int(part.WEEK_NUM.min()), 'max_week': int(part.WEEK_NUM.max())}
    bundles = {}
    for name in ['logistic', 'lightgbm']:
        model = candidate(name).fit(x['train'], windows['train'].target)
        bundle = {'model': model, 'calibrator': calibrate(model, x['calibration'], windows['calibration'].target),
                  'features': features, 'data_kind': manifest['data_kind'], 'versions': result['versions'],
                  'feature_config_sha256': manifest['config_sha256']}
        bundles[name] = bundle
        result['models'][name] = {'validation': metrics(windows['validation'].target, predict(bundle, x['validation']))}
    champion = min(bundles, key=lambda name: result['models'][name]['validation']['log_loss'])
    result['champion'] = champion
    output.mkdir(parents=True)
    for name, bundle in bundles.items():
        probabilities = predict(bundle, x['test'])
        result['models'][name]['test'] = metrics(windows['test'].target, probabilities)
    bundle = bundles[champion]
    bundle['reference'] = profile(x['train'])
    bundle['score_reference'] = profile(pd.DataFrame({'probability': predict(bundle, x['validation'])}))
    bundle['name'] = champion
    joblib.dump(bundle, output / 'model.joblib')
    result['model_sha256'] = digest(output / 'model.joblib')
    probs = predict(bundle, x['test'])
    heldout = windows['test'][['case_id', 'WEEK_NUM', 'target']].copy()
    heldout['probability'] = probs
    heldout.to_parquet(output / 'test_predictions.parquet', index=False)
    result['weekly_test'] = {str(int(week)): metrics(part.target, part.probability) for week, part in heldout.groupby('WEEK_NUM')}
    result['drift_test'] = compare(bundle['reference'], x['test'])
    # TreeSHAP describes the tree challenger even if logistic wins.
    tree = bundles['lightgbm']['model']
    sample = x['test'].iloc[:1000]
    contributions = tree.booster_.predict(sample, pred_contrib=True)
    if not np.allclose(contributions.sum(axis=1), tree.booster_.predict(sample, raw_score=True), atol=1e-6):
        raise AssertionError('TreeSHAP additivity failed')
    pd.DataFrame({'feature': features, 'mean_abs_raw_logodds_contribution': np.abs(contributions[:, :-1]).mean(axis=0)}).sort_values(
        'mean_abs_raw_logodds_contribution', ascending=False).to_csv(output / 'tree_contributions.csv', index=False)
    save_json(output / 'results.json', result)
    render(output, result)
    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment('credit-risk-monitor')
        with mlflow.start_run():
            mlflow.log_params({'champion': champion, 'gap': gap, 'data_kind': result['data_kind'], 'feature_sha256': result['feature_sha256']})
            mlflow.log_metrics({k: v for k, v in result['models'][champion]['test'].items() if isinstance(v, (float, int))})
            mlflow.log_artifacts(str(output))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--train-end', type=int, required=True)
    p.add_argument('--calibration-end', type=int, required=True)
    p.add_argument('--validation-end', type=int, required=True)
    p.add_argument('--gap-weeks', type=int, required=True)
    p.add_argument('--tracking-uri')
    a = p.parse_args()
    train(a.features, a.output, a.train_end, a.calibration_end, a.validation_end, a.gap_weeks, a.tracking_uri)
