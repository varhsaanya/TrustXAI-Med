"""Stage 2: trust scores for every patient, leave-one-site-out.
Usage: python -m trustxai.run_trust --data_dir <path> --model rf [--max_per_site 100]"""
import argparse
import pandas as pd
from .data import load_uci_sites, make_synthetic, CORE_FEATURES
from .models import fit_with_calibration
from .trust import compute_trust, combine
from . import validate as V

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='data')
ap.add_argument('--model', default='rf', choices=['rf', 'xgb', 'logreg'])
ap.add_argument('--max_per_site', type=int, default=0, help='0 = all patients')
ap.add_argument('--n_noise', type=int, default=10)
ap.add_argument('--lime_samples', type=int, default=1000)
ap.add_argument('--synthetic', action='store_true', help='smoke test only')
a = ap.parse_args()

df = make_synthetic() if a.synthetic else load_uci_sites(a.data_dir)
parts = []
for site in df['site'].unique():
    print(f'\n== Held-out site: {site}', flush=True)
    tr, te = df[df['site'] != site], df[df['site'] == site]
    if a.max_per_site:
        te = te.sample(min(a.max_per_site, len(te)), random_state=0)
    base, cal = fit_with_calibration(a.model, tr, tr['target'], CORE_FEATURES)
    t = compute_trust(base, cal, tr, te, CORE_FEATURES, n_noise=a.n_noise, lime_samples=a.lime_samples)
    t['site'], t['target'] = site, te['target']
    t['pred'] = (t['p_calibrated'] >= .5).astype(int)
    t['correct'] = (t['pred'] == t['target']).astype(int)
    parts.append(t)

res = pd.concat(parts)
res['trust'] = combine(res)
out = f'trust_scores_{a.model}.csv'
res.to_csv(out)
print(f'\nSaved {out}')
pd.set_option('display.width', 200)
print('\n=== Per-site summary ===\n', V.per_site(res).to_string())
print('\n=== Error-detection AUROC (0.5 = useless; trust must beat confidence alone) ===')
print(V.error_detection_auroc(res).to_string())
print('\n=== Accuracy if we only act on the most-trusted predictions ===')
print(V.risk_coverage(res).to_string(index=False))
print('\n=== Same, ranking by confidence alone (the baseline to beat) ===')
print(V.risk_coverage(res, 'confidence').to_string(index=False))
