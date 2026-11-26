"""Stage 1: baselines. Usage: python -m trustxai.run_baselines --data_dir data  (or --synthetic)"""
import argparse
import pandas as pd
from .data import load_uci_sites, make_synthetic, missingness_report, CORE_FEATURES
from .evaluate import repeated_cv, leave_one_site_out

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='data')
ap.add_argument('--synthetic', action='store_true', help='smoke test only')
a = ap.parse_args()

df = make_synthetic() if a.synthetic else load_uci_sites(a.data_dir)
print('Rows per site:\n', df['site'].value_counts().to_string())
print('\nMissingness by site:\n', missingness_report(df).to_string())

full = CORE_FEATURES + ['slope', 'ca', 'thal']
clev = df[df['site'] == 'cleveland']
print('\n=== Cleveland only, repeated 5x3 CV (all 13 features) ===')
for m in ['logreg', 'rf', 'xgb']:
    print(m, repeated_cv(clev, full, m))

print('\n=== Pooled 4 sites, repeated CV (core features) ===')
for m in ['logreg', 'rf', 'xgb']:
    print(m, repeated_cv(df, CORE_FEATURES, m))

print('\n=== Leave-one-site-out (core features) ===')
for m in ['logreg', 'rf', 'xgb']:
    print('\n', m); print(leave_one_site_out(df, CORE_FEATURES, m).to_string(index=False))
