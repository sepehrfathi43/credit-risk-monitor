"""Frozen reference bins, including a separate missing-value bucket."""
import numpy as np


def proportions(values, cuts):
    a = np.asarray(values, dtype=float)
    present = a[~np.isnan(a)]
    counts = np.bincount(np.searchsorted(cuts, present, side='right'), minlength=len(cuts) + 1)
    counts = np.append(counts, np.isnan(a).sum()).astype(float) + 0.5
    return counts / counts.sum()


def profile(frame):
    result = {}
    for column in frame:
        present = frame[column].dropna().to_numpy()
        cuts = np.unique(np.append(np.quantile(present, np.linspace(0, 1, 11)),
                                   np.nextafter(present.max(), np.inf))).tolist() if len(present) else []
        result[column] = {'cuts': cuts, 'proportions': proportions(frame[column], cuts).tolist(),
                          'missing_rate': float(frame[column].isna().mean())}
    return result


def compare(reference, frame, threshold=0.2):
    results = {}
    for column, spec in reference.items():
        expected = np.asarray(spec['proportions'])
        actual = proportions(frame[column], spec['cuts'])
        psi = float(np.sum((actual - expected) * np.log(actual / expected)))
        results[column] = {'psi': psi, 'missing_rate': float(frame[column].isna().mean()),
                           'reference_missing_rate': spec['missing_rate'], 'review': psi >= threshold}
    return results
