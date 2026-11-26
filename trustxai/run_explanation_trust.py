"""python -m trustxai.run_explanation_trust --model rf   (needs audit_<model>.csv from run_audit)"""
import argparse
import pandas as pd
from .explain_trust import run

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='rf', choices=['rf', 'xgb', 'logreg'])
a = ap.parse_args()
df = pd.read_csv(f'audit_{a.model}.csv', index_col=0)
res = run(df, outdir=f'figures_{a.model}')
res.to_csv(f'explanation_trust_{a.model}.csv')
print(f'\nSaved explanation_trust_{a.model}.csv')
