"""Stage 5: margin-robust faithfulness (SHAP-based, fast: no LIME needed).

For each patient: remove SHAP's top-k features vs n_random random k-subsets (training-mean baseline)
and record the PERCENTILE of the SHAP drop among the random drops (0.5 = no better than random,
1.0 = beats every random subset). A within-patient percentile is not distorted by how large or small
each patient's probability is, which addresses the confidence confound. k = 3, 5, 8 also tests
whether results depend on the top-5 set that agreement itself uses.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .audit import _remove
from .trust import _shap_matrix
from .explain_trust import partial_spearman


def compute_margin_faith(base, X_train, X_test, features, ks=(3, 5, 8), n_random=50, seed=0):
    rng = np.random.default_rng(seed)
    Xt_tr, Xt_te = base[:-1].transform(X_train[features]), base[:-1].transform(X_test[features])
    cols, n_f = list(Xt_tr.columns), Xt_tr.shape[1]
    clf = base.named_steps['clf']
    predict = lambda a: clf.predict_proba(pd.DataFrame(a, columns=cols))[:, 1]
    p_raw = predict(Xt_te.values)
    pred = (p_raw >= .5).astype(int)
    sv = _shap_matrix(base, Xt_tr, Xt_te)
    baseline, Xv = Xt_tr.mean().values, Xt_te.values
    ppred = lambda p: np.where(pred == 1, p, 1 - p)
    p0 = ppred(p_raw)
    order = np.argsort(-np.abs(sv), axis=1)
    out = {}
    for k in ks:
        d_top = p0 - ppred(predict(_remove(Xv, [order[i, :k] for i in range(len(Xv))], baseline)))
        d_rand = np.stack([p0 - ppred(predict(_remove(Xv, [rng.choice(n_f, k, replace=False) for _ in range(len(Xv))], baseline)))
                           for _ in range(n_random)])
        out[f'faith_pct_k{k}'] = (d_rand < d_top).mean(0) + 0.5 * (d_rand == d_top).mean(0)
        out[f'faith_relgap_k{k}'] = (d_top - d_rand.mean(0)) / np.maximum(p0, 1e-6)
    return pd.DataFrame(out, index=X_test.index).assign(p_raw_check=p_raw)


def analyze_margin(mf, audit):
    pd.set_option('display.width', 220)
    df = mf.join(audit[['site', 'confidence', 'agreement', 'pred', 'target', 'p_raw'] +
                       [c for c in audit.columns if c.startswith('stab_')]], how='inner')
    print(f'\nSanity check: max |p_raw difference| vs audit file = {np.abs(df["p_raw"] - df["p_raw_check"]).max():.2e} (should be ~0)')
    levels = sorted(c for c in df.columns if c.startswith('stab_'))
    ks = sorted(c for c in df.columns if c.startswith('faith_pct_'))
    print('\n##### 1. DOES FAITHFULNESS DROP WHEN THE MODEL IS UNSURE? (within-patient percentile; 0.5 = random)')
    q = pd.qcut(df['confidence'], 4, duplicates='drop')
    t = df.groupby(q, observed=True).agg(n=('faith_pct_k5', 'size'), mean_pct=('faith_pct_k5', 'mean'),
                                         share_beats_90pct_of_random=('faith_pct_k5', lambda x: (x >= .9).mean()),
                                         share_beats_random_at_all=('faith_pct_k5', lambda x: (x > .5).mean()),
                                         mean_rel_gap=('faith_relgap_k5', 'mean')).round(3)
    t.index = [f'conf Q{i + 1}' + (' (lowest)' if i == 0 else ' (highest)' if i == len(t) - 1 else '') for i in range(len(t))]
    print(t.to_string())
    for k in ks:
        r, p = spearmanr(df['confidence'], df[k])
        print(f'  Spearman(confidence, {k}) = {r:+.3f}  p={p:.2g}')
    print('\n##### 2. CONTROLS WITH THE MARGIN-ROBUST MEASURE (partial rho | confidence + site)')
    site_d = pd.get_dummies(df['site'], drop_first=True).values
    rows = []
    for name in ['agreement'] + levels:
        r = {'score': name}
        for k in ks:
            rho, p = partial_spearman(df[name], df[k], [df['confidence']], site_d)
            r[k.replace('faith_pct_', 'k=')] = f'{rho:+.3f} (p={p:.2g})'
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nPer site: Spearman(agreement, faith_pct_k5)')
    for s, g in df.groupby('site'):
        print(f'  {s:<12} n={len(g):<4} rho={spearmanr(g["agreement"], g["faith_pct_k5"])[0]:+.3f}')
    return df
