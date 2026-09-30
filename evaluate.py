"""Evaluation uses probabilities, never an arbitrary approval threshold."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if not len(y) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Empty cohort or invalid probabilities')
    both = len(np.unique(y)) == 2
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        if mask.any():
            bins.append({'n': int(mask.sum()), 'predicted': float(p[mask].mean()),
                         'observed': float(y[mask].mean())})
    return {'n': len(y), 'events': int(y.sum()), 'event_rate': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, p)) if both else None,
            'average_precision': float(average_precision_score(y, p)) if both else None,
            'log_loss': float(log_loss(y, p, labels=[0, 1])),
            'brier': float(brier_score_loss(y, p)),
            'ece': sum(b['n'] * abs(b['observed'] - b['predicted']) for b in bins) / len(y),
            'calibration_bins': bins, 'ranking_note': None if both else 'One-class cohort; ranking undefined'}
