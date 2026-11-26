"""python -m trustxai.run_margin_faith --data_dir <path> --model rf   (needs audit_<model>.csv)"""
import argparse
import pandas as pd
from .data import load_uci_sites, make_synthetic, CORE_FEATURES
from .models import build_model
from .margin_faith import compute_margin_faith, analyze_margin

ap = argparse.ArgumentParser()
ap.add_argument('--data_dir', default='data')
ap.add_argument('--model', default='rf', choices=['rf', 'xgb', 'logreg'])
ap.add_argument('--n_random', type=int, default=50)
ap.add_argument('--synthetic', action='store_true', help='smoke test only')
a = ap.parse_args()
audit = pd.read_csv(f'audit_{a.model}.csv', index_col=0)
df = make_synthetic() if a.synthetic else load_uci_sites(a.data_dir)
parts = []
for site in df['site'].unique():
    tr = df[df['site'] != site]
    te = df[(df['site'] == site) & (df.index.isin(audit.index))]
    base = build_model(a.model, CORE_FEATURES).fit(tr[CORE_FEATURES], tr['target'])
    parts.append(compute_margin_faith(base, tr, te, CORE_FEATURES, n_random=a.n_random))
    print(f'  done {site}', flush=True)
res = analyze_margin(pd.concat(parts), audit)
res.to_csv(f'margin_faith_{a.model}.csv')
print(f'\nSaved margin_faith_{a.model}.csv')
