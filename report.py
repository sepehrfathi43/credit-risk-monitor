"""Portable HTML report; all numbers come from a saved run."""
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(output, result):
    output = Path(output)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    for name, details in result['models'].items():
        bins = details['test']['calibration_bins']
        ax.plot([b['predicted'] for b in bins], [b['observed'] for b in bins], 'o-', label=name)
    ax.set(xlabel='Mean predicted probability', ylabel='Observed event rate', title='Held-out test calibration')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'calibration.png', dpi=150)
    plt.close(fig)
    title = 'TEST FIXTURE — NOT REAL CREDIT RESULTS' if result['data_kind'] == 'test-fixture' else 'Home Credit evaluation'
    body = '<!doctype html><meta charset="utf-8"><title>Credit risk evaluation</title>'
    body += '<h1>' + title + '</h1><p>Champion selected by validation log loss: ' + html.escape(result['champion']) + '</p>'
    body += '<p>Retrospective temporal evaluation. Label maturity and borrower overlap remain unresolved.</p>'
    body += '<img src="calibration.png" alt="Held-out calibration chart"><pre>' + html.escape(json.dumps(result, indent=2)) + '</pre>'
    (output / 'report.html').write_text(body, encoding='utf-8')
