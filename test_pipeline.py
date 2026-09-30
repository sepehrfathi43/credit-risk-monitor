"""Small synthetic records test software behavior, never model quality."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from build_features import build
from common import load_json, save_json, validate_cases, numeric_frame
from train import train, time_windows
from score import score
from monitoring import profile, compare
from evaluate import metrics


@pytest.fixture
def raw(tmp_path):
    rng = np.random.default_rng(43)
    n = 1600
    folder = tmp_path / 'raw' / 'parquet_files' / 'train'
    folder.mkdir(parents=True)
    base = pd.DataFrame({'case_id': np.arange(n), 'WEEK_NUM': np.repeat(np.arange(20), 80),
                         'date_decision': pd.date_range('2020-01-01', periods=n, freq='h'),
                         'target': np.tile([0, 0, 0, 1], n // 4)})
    base.to_parquet(folder / 'train_base.parquet', index=False)
    static = pd.DataFrame({'case_id': np.arange(n), 'credamount_770A': rng.normal(10000, 2000, n),
                           'numactivecreds_622L': rng.integers(0, 5, n), 'totaldebt_9A': rng.normal(500, 50, n),
                           'currdebt_22A': np.full(n, np.nan)})
    static.iloc[:800].to_parquet(folder / 'train_static_0_0.parquet', index=False)
    static.iloc[800:].to_parquet(folder / 'train_static_0_1.parquet', index=False)
    pd.DataFrame({'case_id': np.repeat(np.arange(0, n, 2), 2),
                  'num_group1': np.tile([0, 1], n // 2)}).to_parquet(folder / 'train_applprev_1_0.parquet', index=False)
    return tmp_path / 'raw'


def make_features(raw, tmp_path):
    build(raw, Path(__file__).with_name('features.json'), tmp_path / 'features', fixture=True)
    return tmp_path / 'features'


def test_end_to_end(raw, tmp_path):
    features = make_features(raw, tmp_path)
    frame = pd.read_parquet(features / 'features.parquet')
    assert len(frame) == 1600 and frame.case_id.is_unique
    assert (frame.loc[frame.case_id % 2 == 0, 'previous__records'] == 2).all()
    assert (frame.loc[frame.case_id % 2 == 1, 'previous__present'] == 0).all()
    result = train(features, tmp_path / 'model', 7, 11, 15, 1, 'sqlite:///' + (tmp_path / 'mlflow.db').as_posix())
    assert result['data_kind'] == 'test-fixture'
    assert result['champion'] == min(result['models'], key=lambda k: result['models'][k]['validation']['log_loss'])
    assert result['windows']['test']['min_week'] == 17
    predictions, report = score(tmp_path / 'model', features, tmp_path / 'scored')
    assert predictions.probability.between(0, 1).all()
    assert report['performance']['n'] == 1600
    assert 'TEST FIXTURE' in (tmp_path / 'model' / 'report.html').read_text(encoding='utf-8')
    assert (tmp_path / 'model' / 'tree_contributions.csv').exists()
    # No outcomes required for future-batch scoring.
    future = raw / 'parquet_files' / 'test'
    future.mkdir()
    for source in (raw / 'parquet_files' / 'train').glob('*.parquet'):
        data = pd.read_parquet(source).drop(columns=['target'], errors='ignore')
        data.to_parquet(future / source.name.replace('train_', 'test_'), index=False)
    build(raw, Path(__file__).with_name('features.json'), tmp_path / 'future', split='test', fixture=True)
    _, unlabeled = score(tmp_path / 'model', tmp_path / 'future', tmp_path / 'future_scores')
    assert unlabeled['performance'] is None
    manifest_path = tmp_path / 'future' / 'manifest.json'
    manifest = load_json(manifest_path)
    manifest['config_sha256'] = 'changed'
    save_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='configuration'):
        score(tmp_path / 'model', tmp_path / 'future', tmp_path / 'mismatched')
    # A tampered artifact is refused before unpickling.
    with (tmp_path / 'model' / 'model.joblib').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        score(tmp_path / 'model', features, tmp_path / 'bad_score')


@pytest.mark.parametrize('filename', ['train_base.parquet', 'train_static_0_0.parquet', 'train_applprev_1_0.parquet'])
def test_duplicate_keys_rejected(raw, tmp_path, filename):
    path = raw / 'parquet_files' / 'train' / filename
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[:1]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        make_features(raw, tmp_path)
    assert load_json(tmp_path / 'features' / 'manifest.json')['status'] == 'failed'


def test_bad_numeric_rejected(raw, tmp_path):
    path = raw / 'parquet_files' / 'train' / 'train_static_0_0.parquet'
    frame = pd.read_parquet(path)
    frame.loc[0, 'credamount_770A'] = np.inf
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='Invalid numeric'):
        make_features(raw, tmp_path)


def test_missing_column_rejected(raw, tmp_path):
    path = raw / 'parquet_files' / 'train' / 'train_static_0_0.parquet'
    pd.read_parquet(path).drop(columns=['totaldebt_9A']).to_parquet(path, index=False)
    with pytest.raises(ValueError, match='missing'):
        make_features(raw, tmp_path)


def test_temporal_windows_exclude_gaps():
    frame = pd.DataFrame({'WEEK_NUM': np.repeat(np.arange(20), 30), 'target': np.tile([0, 1], 300)})
    windows = time_windows(frame, 7, 11, 15, 1)
    for a, b in zip(list(windows.values())[:-1], list(windows.values())[1:]):
        assert a.WEEK_NUM.max() + 1 < b.WEEK_NUM.min()
        assert set(a.index).isdisjoint(b.index)
    with pytest.raises(ValueError):
        time_windows(frame, 11, 7, 15, 1)


def test_drift_handles_constant_and_missing():
    reference = pd.DataFrame({'x': [1.] * 100, 'empty': [np.nan] * 100})
    frozen = profile(reference)
    same = compare(frozen, reference)
    assert same['x']['psi'] == 0 and same['empty']['psi'] == 0
    changed = compare(frozen, pd.DataFrame({'x': [10.] * 100, 'empty': [1.] * 100}))
    assert changed['empty']['review']
    assert changed['x']['review']
    assert np.isfinite(changed['x']['psi'])


def test_single_class_metrics():
    report = metrics([0, 0], [0.1, 0.2])
    assert report['roc_auc'] is None and report['average_precision'] is None
    assert report['brier'] > 0


def test_contract_rejects_duplicate_ids_and_infinity():
    with pytest.raises(ValueError):
        validate_cases(pd.DataFrame({'case_id': [1, 1], 'WEEK_NUM': [0, 1]}))
    with pytest.raises(ValueError):
        numeric_frame(pd.DataFrame({'x': [np.inf]}), ['x'])


def test_sampling_is_reproducible(raw, tmp_path):
    config = Path(__file__).with_name('features.json')
    a = build(raw, config, tmp_path / 'a', sample_fraction=0.2, fixture=True)
    b = build(raw, config, tmp_path / 'b', sample_fraction=0.2, fixture=True)
    assert set(pd.read_parquet(a).case_id) == set(pd.read_parquet(b).case_id)
    assert 0 < len(pd.read_parquet(a)) < 1600


def test_test_labels_do_not_affect_selection(raw, tmp_path):
    features = make_features(raw, tmp_path)
    first = train(features, tmp_path / 'first', 7, 11, 15, 1)
    path = features / 'features.parquet'
    frame = pd.read_parquet(path)
    frame.loc[frame.WEEK_NUM > 16, 'target'] = 1 - frame.loc[frame.WEEK_NUM > 16, 'target']
    frame.to_parquet(path, index=False)
    from common import digest
    manifest = load_json(features / 'manifest.json')
    manifest['feature_sha256'] = digest(path)
    save_json(features / 'manifest.json', manifest)
    second = train(features, tmp_path / 'second', 7, 11, 15, 1)
    assert first['champion'] == second['champion']
    for name in first['models']:
        assert first['models'][name]['validation'] == second['models'][name]['validation']
