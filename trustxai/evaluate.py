"""Metrics with bootstrap CIs, plus within-site CV and leave-one-site-out evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from .models import build_model


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
               for b in range(bins) if (idx == b).any())


def metrics(y, p, n_boot=500, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= .5).astype(int)
    out = {'accuracy': accuracy_score(y, pred), 'f1': f1_score(y, pred),
           'auroc': roc_auc_score(y, p), 'brier': brier_score_loss(y, p),
           'ece': expected_calibration_error(y, p)}
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            aucs.append(roc_auc_score(y[i], p[i]))
    out['auroc_ci'] = (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5))
    return out


def repeated_cv(df, features, model_name, n_splits=5, n_repeats=3, seed=42):
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for tr, te in cv.split(df, df['target']):
        m = build_model(model_name, features, seed).fit(df.iloc[tr][features], df.iloc[tr]['target'])
        p = m.predict_proba(df.iloc[te][features])[:, 1]
        r = metrics(df.iloc[te]['target'], p, n_boot=0 + 1)
        r.pop('auroc_ci'); rows.append(r)
    r = pd.DataFrame(rows)
    return r.mean().round(3).to_dict() | {'auroc_sd': round(r['auroc'].std(), 3)}


def leave_one_site_out(df, features, model_name, seed=42):
    rows = []
    for site in df['site'].unique():
        tr, te = df[df['site'] != site], df[df['site'] == site]
        m = build_model(model_name, features, seed).fit(tr[features], tr['target'])
        p = m.predict_proba(te[features])[:, 1]
        r = metrics(te['target'], p, n_boot=300)
        ci = r.pop('auroc_ci')
        rows.append({'held_out_site': site, 'n': len(te), **{k: round(v, 3) for k, v in r.items()},
                     'auroc_95ci': f'{ci[0]:.2f}-{ci[1]:.2f}'})
    return pd.DataFrame(rows)
