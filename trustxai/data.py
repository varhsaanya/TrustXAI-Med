"""Data loading for the 4 UCI Heart Disease sites (+ a synthetic stand-in for smoke tests)."""
import numpy as np
import pandas as pd
from pathlib import Path

COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
SITES = ['cleveland', 'hungarian', 'switzerland', 'va']
NUMERIC = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
# ca / thal / slope are almost entirely missing outside Cleveland -> use for cross-site work
CORE_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak']


def load_uci_sites(data_dir):
    """Read processed.<site>.data for all 4 sites into one frame with a `site` column."""
    frames = []
    for site in SITES:
        path = Path(data_dir) / f'processed.{site}.data'
        df = pd.read_csv(path, names=COLUMNS, na_values=['?', '-9', -9])
        df['site'] = site
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.apply(pd.to_numeric, errors='coerce').assign(site=df['site'])
    # physiologically impossible zeros are missing values (common in Switzerland / VA)
    for col in ['chol', 'trestbps']:
        df.loc[df[col] == 0, col] = np.nan
    df['target'] = (df['num'] > 0).astype(int)
    return df.drop(columns=['num'])


def missingness_report(df):
    cols = [c for c in df.columns if c not in ('site', 'target')]
    return df[cols].isna().groupby(df['site']).mean().round(2)


def make_synthetic(n_per_site=(300, 290, 120, 190), seed=0):
    """SMOKE-TEST ONLY: random data with the real schema. Never report results from it."""
    rng = np.random.default_rng(seed)
    frames = []
    for site, n in zip(SITES, n_per_site):
        t = rng.integers(0, 2, n)
        d = pd.DataFrame({
            'age': rng.normal(54 + 4 * t, 8, n).round(),
            'sex': rng.integers(0, 2, n),
            'cp': np.where(t == 1, rng.choice([1, 2, 3, 4], n, p=[.1, .1, .1, .7]),
                           rng.choice([1, 2, 3, 4], n, p=[.2, .3, .3, .2])),
            'trestbps': rng.normal(130 + 5 * t, 17, n).round(),
            'chol': rng.normal(245, 50, n).round(),
            'fbs': rng.integers(0, 2, n),
            'restecg': rng.integers(0, 3, n),
            'thalach': rng.normal(160 - 18 * t, 20, n).round(),
            'exang': (rng.random(n) < (.15 + .4 * t)).astype(int),
            'oldpeak': np.clip(rng.normal(.4 + 1.2 * t, 1, n), 0, 6).round(1),
            'slope': rng.choice([1, 2, 3], n),
            'ca': rng.choice([0, 1, 2, 3], n, p=[.6, .2, .12, .08]) if site == 'cleveland' else np.nan,
            'thal': rng.choice([3, 6, 7], n) if site == 'cleveland' else np.nan,
            'target': t,
        })
        d['site'] = site
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
