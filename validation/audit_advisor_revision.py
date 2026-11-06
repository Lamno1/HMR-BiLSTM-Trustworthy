"""CPU audit of advisor revisions and saved predictions; no training.

Run: python -m validation.audit_advisor_revision
Output: results/tables/advisor_revision_audit.json
"""
from pathlib import Path
import json

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456)


def recompute(path, stored, grouped=False):
    with np.load(path) as d:
        y, pred = d['y'], d['logits'].argmax(axis=1)
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    assert np.isclose(f[:4].mean(), stored['f1_macro'], atol=1e-12, rtol=0), path
    assert np.isclose((y == pred).mean(), stored['accuracy'], atol=1e-12, rtol=0), path
    for i, c in enumerate('NSVF'):
        expected = stored['f1_per_class'][c] if grouped else stored['per_class'][c]['f1']
        assert np.isclose(f[i], expected, atol=1e-12, rtol=0), (path, c)
        if not grouped:
            for key, value in [('precision', p[i]), ('recall', r[i]), ('support', n[i])]:
                assert np.isclose(value, stored['per_class'][c][key], atol=1e-12, rtol=0), (path, c, key)
    if not grouped:
        assert np.array_equal(confusion_matrix(y, pred, labels=range(5)), stored['confusion_NSVFQ']), path
    return float(f[:4].mean())


def main():
    from validation.run_controlled_mit_subject_cv import build_protocol
    saved = json.loads((ROOT/'results/controlled_mit_subject_cv/protocol_manifest.json').read_text())
    rebuilt, arrays = build_protocol(saved['k'], saved['fold_seed'], saved['inner_val_fraction'])
    assert rebuilt['group_records'] == saved['group_records']
    assert rebuilt['folds'] == saved['folds'], 'Rebuilt split differs from locked manifest'
    del arrays
    result = {'protocol_rebuilt_from_pool_matches': True, 'prediction_runs_checked': {}}
    scores = {}
    for folder in ('matched_recipe', 'matched_recipe_mlii'):
        count = 0
        scores[folder] = {}
        for path in sorted((ROOT/'results'/folder).glob('*_seed*.json')):
            data = json.loads(path.read_text())
            pred = path.with_name(path.stem + '_test_logits.npz')
            if not pred.exists():
                raise FileNotFoundError(pred)
            value = recompute(pred, data['test'])
            scores[folder][path.stem] = value
            count += 1
        result['prediction_runs_checked'][folder] = count
    count = 0
    for folder, pattern in [('controlled_mit_subject_cv', 'fold*_seed*_flat.json'),
                            *[(f'sgcv_arch/{a}', 'fold*_seed*.json') for a in
                              ('normc', 'normc_nodrop', 'bilstm', 'bilstm_nocw', 'bilstm_ce')]]:
        for path in sorted((ROOT/'results'/folder).glob(pattern)):
            data = json.loads(path.read_text())
            recompute(path.with_name(path.stem + '_preds.npz'), data['test_metrics'], grouped=True)
            count += 1
    assert count == 60, count
    result['prediction_runs_checked']['SGCV'] = count
    result['mlii'] = {}
    for a in ('full', 'normc', 'bilstm'):
        old = np.array([scores['matched_recipe'][f'{a}_seed{s}'] for s in SEEDS])
        new = np.array([scores['matched_recipe_mlii'][f'{a}_seed{s}'] for s in SEEDS])
        result['mlii'][a] = {'original': old.tolist(), 'MLII': new.tolist(),
                             'original_mean': float(old.mean()), 'MLII_mean': float(new.mean()),
                             'original_sd': float(old.std(ddof=1)), 'MLII_sd': float(new.std(ddof=1)),
                             'paired_MLII_minus_original': (new-old).tolist()}
    full = np.array(result['mlii']['full']['MLII'])
    for a in ('normc', 'bilstm'):
        diff = full - np.array(result['mlii'][a]['MLII'])
        result['mlii'][f'full_minus_{a}'] = {'paired': diff.tolist(), 'mean': float(diff.mean())}
    from validation.matched_diagnostics import cm, macro_f1
    records = np.load(ROOT/'data/processed/splits/inter_test_records.npy')
    unique = sorted(set(records.tolist()))
    weights = np.random.default_rng(0).multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=2000)
    sampled = {}
    for a in ('full', 'normc', 'bilstm'):
        values = []
        for s in SEEDS:
            with np.load(ROOT/f'results/matched_recipe_mlii/{a}_seed{s}_test_logits.npz') as d:
                y, pred = d['y'], d['logits'].argmax(axis=1)
            counts = np.stack([cm(y[records == r], pred[records == r]) for r in unique])
            values.append(macro_f1(np.einsum('nr,rij->nij', weights, counts)))
        sampled[a] = np.mean(values, axis=0)
    for a in ('normc', 'bilstm'):
        result['mlii'][f'full_minus_{a}']['record_bootstrap_ci95'] = np.percentile(
            sampled['full']-sampled[a], [2.5, 97.5]).tolist()
    result['mlii_bootstrap'] = {'resamples': 2000, 'seed': 0, 'unit': 'DS2 record',
                               'seed_aggregation': 'mean of per-seed macro F1; paired resampling'}
    result['limits'] = ['Recomputes saved predictions; does not retrain models or rerun inference.',
                        'Literature Table 2 requires checking against original publications.',
                        'Historical MLII runs do not store split hashes/data_dir in result JSON.']
    out = ROOT/'results/tables/advisor_revision_audit.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
