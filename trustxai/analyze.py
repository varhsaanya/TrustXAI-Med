"""Statistics for the audit CSV: error-detection with bootstrap CIs, noise sweep, faithfulness."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

W = {'confidence': 0.4, 'agreement': 0.3, 'stability': 0.3}


def _auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


def boot_auc(y, s, n=1000, seed=0):
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a = _auc(y[i], s[i])
        if not np.isnan(a): v.append(a)
    return _auc(y, s), np.percentile(v, 2.5), np.percentile(v, 97.5)


def boot_diff(y, s1, s2, n=1000, seed=0):
    """AUROC(s1) - AUROC(s2) with paired bootstrap 95% CI."""
    y, s1, s2 = map(np.asarray, (y, s1, s2))
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a, b = _auc(y[i], s1[i]), _auc(y[i], s2[i])
        if not (np.isnan(a) or np.isnan(b)): v.append(a - b)
    return _auc(y, s1) - _auc(y, s2), np.percentile(v, 2.5), np.percentile(v, 97.5)


def _fmt(t): return f'{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]'


def analyze(df):
    df = df.copy()
    df['correct'] = (df['pred'] == df['target']).astype(int)
    levels = sorted(c for c in df.columns if c.startswith('stab_'))
    base_lvl = levels[0]
    df['stability'] = df[base_lvl]
    df['trust'] = sum(w * df[k] for k, w in W.items()) / sum(W.values())
    pd.set_option('display.width', 220)

    print(f'\n##### 1. ERROR DETECTION (can the score separate right from wrong predictions?) n={len(df)}')
    print('AUROC [95% bootstrap CI]; 0.5 = no information')
    for name in ['confidence', 'agreement', 'stability', 'trust']:
        print(f'  pooled {name:<11}', _fmt(boot_auc(df['correct'], df[name])))
    print('  trust minus confidence (pooled):', _fmt(boot_diff(df['correct'], df['trust'], df['confidence'])),
          ' <- CI containing 0 means no demonstrated gain')
    rows = []
    for s, g in df.groupby('site'):
        r = {'site': s, 'n': len(g), 'acc': round(g['correct'].mean(), 3), 'disease_rate': round(g['target'].mean(), 3)}
        for name in ['confidence', 'agreement', 'stability', 'trust']:
            r[name] = round(_auc(g['correct'], g[name]), 3)
        d = boot_diff(g['correct'], g['trust'], g['confidence'], n=500)
        r['trust-conf [CI]'] = _fmt(d)
        rows.append(r)
    print('\nPer site:\n' + pd.DataFrame(rows).to_string(index=False))

    print(f'\n##### 2. STABILITY NOISE SWEEP (is stability stuck near 1.0?)')
    r = []
    for c in levels:
        r.append({'noise_level': c.replace('stab_', ''), 'mean': round(df[c].mean(), 3), 'sd': round(df[c].std(), 3),
                  'min': round(df[c].min(), 3), 'share>=0.99': round((df[c] >= .99).mean(), 3),
                  'AUROC(errors)': round(_auc(df['correct'], df[c]), 3),
                  'rho_vs_faith_gap': round(spearmanr(df[c], df['faith_shap_gap'])[0], 3)})
    print(pd.DataFrame(r).to_string(index=False))

    print('\n##### 3. FAITHFULNESS (does removing the explained features change the prediction more than random?)')
    f = {}
    for m in ['shap', 'lime']:
        g = df[f'faith_{m}_gap']
        f[m.upper()] = {'mean_drop_top5': round(df[f'faith_{m}_drop'].mean(), 3),
                        'mean_drop_random': round(df['faith_random_drop'].mean(), 3),
                        'mean_gap': round(g.mean(), 3), 'share_gap>0': round((g > 0).mean(), 3)}
    print(pd.DataFrame(f).T.to_string())
    print('\nDoes explanation quality track faithfulness? (Spearman rho with SHAP faithfulness gap)')
    for name in ['agreement'] + levels:
        rho, p = spearmanr(df[name], df['faith_shap_gap'])
        print(f'  {name:<14} rho={rho:+.3f}  p={p:.3g}')
    q = pd.qcut(df['agreement'], 4, duplicates='drop')
    print('\nMean SHAP faithfulness gap by SHAP-LIME agreement quartile:')
    print(df.groupby(q, observed=True)['faith_shap_gap'].agg(['mean', 'count']).round(3).to_string())

    print('\n##### 4. TWO-AXIS VIEW per site (prediction trust vs explanation trust)')
    best = max(levels, key=lambda c: df[c].std())
    df['explanation_trust'] = 0.5 * df['agreement'] + 0.5 * df[best]
    print(f'(explanation trust = mean of agreement and {best}, the noise level with most spread)')
    print(df.groupby('site')[['confidence', 'explanation_trust', 'faith_shap_gap']].mean().round(3).to_string())
    print('\nCorrelation between the two axes (low = they measure different things):',
          round(spearmanr(df['confidence'], df['explanation_trust'])[0], 3))
    return df
