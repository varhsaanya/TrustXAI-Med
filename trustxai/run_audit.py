"""Stage 3: python -m trustxai.run_audit --data_dir <path> --model rf [--max_per_site 100]
Computes everything once (slow, LIME) -> audit_<model>.csv. Re-analyse anytime with --analyze_only."""
import argparse
import pandas as pd
from .data import load_uci_sites, make_synthetic, CORE_FEATURES
from .models import fit_with_calibration
from .audit import compute_audit
from .analyze import analyze

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='data')
ap.add_argument('--model', default='rf', choices=['rf', 'xgb', 'logreg'])
ap.add_argument('--max_per_site', type=int, default=0, help='0 = all patients')
ap.add_argument('--n_noise', type=int, default=10)
ap.add_argument('--n_random', type=int, default=10)
ap.add_argument('--lime_samples', type=int, default=1000)
ap.add_argument('--analyze_only', action='store_true')
ap.add_argument('--synthetic', action='store_true', help='smoke test only')
a = ap.parse_args()
out = f'audit_{a.model}.csv'

if a.analyze_only:
    res = pd.read_csv(out, index_col=0)
else:
    df = make_synthetic() if a.synthetic else load_uci_sites(a.data_dir)
    parts = []
    for site in df['site'].unique():
        print(f'\n== Held-out site: {site}', flush=True)
        tr, te = df[df['site'] != site], df[df['site'] == site]
        if a.max_per_site:
            te = te.sample(min(a.max_per_site, len(te)), random_state=0)
        base, cal = fit_with_calibration(a.model, tr, tr['target'], CORE_FEATURES)
        t = compute_audit(base, cal, tr, te, CORE_FEATURES, n_noise=a.n_noise,
                          n_random=a.n_random, lime_samples=a.lime_samples)
        t['site'], t['target'] = site, te['target']
        parts.append(t)
    res = pd.concat(parts)
    res.to_csv(out)
    print(f'\nSaved {out}')
analyze(res)
