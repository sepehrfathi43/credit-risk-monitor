"""Batch scoring and monitoring. Load only model bundles you trust."""
import argparse
import importlib.metadata
from pathlib import Path
import joblib
import pandas as pd
from common import load_json, save_json, digest, validate_cases, numeric_frame
from modeling import predict
from monitoring import compare
from evaluate import metrics


def score(model_dir, feature_dir, output):
    model_dir, feature_dir, output = Path(model_dir), Path(feature_dir), Path(output)
    if output.exists():
        raise FileExistsError(output)
    results = load_json(model_dir / 'results.json')
    if digest(model_dir / 'model.joblib') != results['model_sha256']:
        raise ValueError('Model checksum mismatch')
    bundle = joblib.load(model_dir / 'model.joblib')
    for package, expected in bundle['versions'].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'Runtime version differs from training: {package}; recreate the recorded environment')
    manifest = load_json(feature_dir / 'manifest.json')
    if manifest['status'] != 'complete' or manifest['config_sha256'] != bundle['feature_config_sha256']:
        raise ValueError('Feature configuration differs from training')
    if manifest['data_kind'] != bundle['data_kind']:
        raise ValueError('Cannot mix test fixtures with real-data artifacts')
    if digest(feature_dir / 'features.parquet') != manifest['feature_sha256']:
        raise ValueError('Feature checksum mismatch')
    frame = pd.read_parquet(feature_dir / 'features.parquet')
    validate_cases(frame, labels='target' in frame)
    x = numeric_frame(frame, bundle['features'])
    probabilities = predict(bundle, x)
    prediction = frame[['case_id', 'WEEK_NUM']].copy()
    prediction['probability'] = probabilities
    report = {'data_kind': bundle['data_kind'], 'n': len(frame), 'feature_drift': compare(bundle['reference'], x),
              'score_drift': compare(bundle['score_reference'], pd.DataFrame({'probability': probabilities})),
              'performance': metrics(frame.target, probabilities) if 'target' in frame else None,
              'note': 'Drift flags request investigation, not automatic retraining. Label maturity must be verified.'}
    output.mkdir(parents=True)
    prediction.to_parquet(output / 'predictions.parquet', index=False)
    save_json(output / 'monitoring.json', report)
    return prediction, report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    score(a.model, a.features, a.output)
