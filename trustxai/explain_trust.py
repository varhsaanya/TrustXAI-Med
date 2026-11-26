"""Stage 4: controls, learned explanation-trust score (leave-one-site-out), quadrant table, figures.
Reads audit_<model>.csv produced by run_audit; no recomputation of LIME needed."""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, t as tdist
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def partial_spearman(x, y, ranked_covars=(), dummies=None):
    n = len(x)
    cols = [np.ones(n)] + [rankdata(c) for c in ranked_covars]
    if dummies is not None:
        cols += list(np.asarray(dummies, float).T)
    Z = np.column_stack(cols)
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    ex, ey = res(rankdata(x)), res(rankdata(y))
    r = np.corrcoef(ex, ey)[0, 1]
    dof = n - Z.shape[1] - 1
    p = 2 * tdist.sf(abs(r * np.sqrt(dof / max(1 - r ** 2, 1e-12))), dof)
    return r, p


def prepare(df):
    df = df.copy()
    df['correct'] = (df['pred'] == df['target']).astype(int)
    df['faith_mean_gap'] = 0.5 * (df['faith_shap_gap'] + df['faith_lime_gap'])
    return df


def controls(df, levels):
    print('\n##### A. CONTROLS: does explanation quality really track faithfulness?')
    print('Spearman rho between a score and a faithfulness gap. "| conf" = controlling for confidence; "| conf+site" also removes site effects.')
    site_d = pd.get_dummies(df['site'], drop_first=True).values
    rows = []
    for name in ['agreement'] + levels:
        for gap in ['shap', 'lime']:
            g = df[f'faith_{gap}_gap']
            raw = spearmanr(df[name], g)
            pc = partial_spearman(df[name], g, [df['confidence']])
            pcs = partial_spearman(df[name], g, [df['confidence']], site_d)
            rows.append({'score': name, 'faithfulness_of': gap.upper(),
                         'rho': round(raw[0], 3), 'p': f'{raw[1]:.2g}',
                         'rho|conf': round(pc[0], 3), 'p|conf': f'{pc[1]:.2g}',
                         'rho|conf+site': round(pcs[0], 3), 'p|conf+site': f'{pcs[1]:.2g}'})
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nPer site (agreement vs faithfulness gap): does the effect hold at every hospital?')
    r = []
    for s, g in df.groupby('site'):
        r.append({'site': s, 'n': len(g),
                  'rho_vs_SHAP_gap': round(spearmanr(g['agreement'], g['faith_shap_gap'])[0], 3),
                  'rho_vs_LIME_gap': round(spearmanr(g['agreement'], g['faith_lime_gap'])[0], 3),
                  'share_faithful>0': round((g['faith_mean_gap'] > 0).mean(), 3)})
    print(pd.DataFrame(r).to_string(index=False))


def learned_explanation_trust(df, levels):
    print('\n##### B. LEARNED EXPLANATION-TRUST SCORE (trained on 3 hospitals, tested on the 4th)')
    print('Target: mean of SHAP and LIME faithfulness gaps. Metric: Spearman rho between score and gap on the held-out hospital.')
    feats = {'agreement only': ['agreement'],
             f'stability only ({levels[-1]})': [levels[-1]],
             'learned: agreement + stability': ['agree_top5', 'agree_rank'] + levels,
             'learned: + confidence': ['agree_top5', 'agree_rank', 'confidence'] + levels}
    df['equal_weight'] = 0.5 * df['agreement'] + 0.5 * df[levels[-1]]
    feats['equal weight (agreement, stability)'] = None
    sites = list(df['site'].unique())
    out, preds = [], {}
    for name, cols in feats.items():
        p = pd.Series(index=df.index, dtype=float)
        for s in sites:
            te = df['site'] == s
            if cols is None:
                p[te] = df.loc[te, 'equal_weight']
            elif len(cols) == 1:
                p[te] = df.loc[te, cols[0]]
            else:
                m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(df.loc[~te, cols], df.loc[~te, 'faith_mean_gap'])
                p[te] = m.predict(df.loc[te, cols])
        preds[name] = p
        r = {'score': name}
        for s in sites:
            te = df['site'] == s
            r[s] = round(spearmanr(p[te], df.loc[te, 'faith_mean_gap'])[0], 3)
        r['mean_over_sites'] = round(np.mean([r[s] for s in sites]), 3)
        top = p >= p.median()
        r['gap_top_half'] = round(df.loc[top, 'faith_mean_gap'].mean(), 3)
        r['gap_bottom_half'] = round(df.loc[~top, 'faith_mean_gap'].mean(), 3)
        out.append(r)
    print(pd.DataFrame(out).to_string(index=False))
    cols = feats['learned: agreement + stability']
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(df[cols], df['faith_mean_gap'])
    print('\nStandardised weights of the learned score (all hospitals):')
    print(pd.Series(m[-1].coef_, index=cols).round(4).to_string())
    df['expl_trust'] = preds['learned: agreement + stability']
    return df, out


def quadrants(df):
    print('\n##### C. QUADRANT TABLE (median splits): prediction trust x explanation trust')
    hc = df['confidence'] >= df['confidence'].median()
    he = df['expl_trust'] >= df['expl_trust'].median()
    lab = np.select([hc & he, hc & ~he, ~hc & he], ['HIGH conf / HIGH expl', 'HIGH conf / low expl', 'low conf / HIGH expl'],
                    'low conf / low expl')
    df['quadrant'] = lab
    t = df.groupby('quadrant').agg(n=('correct', 'size'), accuracy=('correct', 'mean'),
                                   mean_faith_gap=('faith_mean_gap', 'mean'),
                                   share_faithful=('faith_mean_gap', lambda x: (x > 0).mean()),
                                   mean_conf=('confidence', 'mean'), mean_expl=('expl_trust', 'mean')).round(3)
    print(t.to_string())
    print('\nCorrelation between the two axes (confidence vs explanation trust): '
          f"{spearmanr(df['confidence'], df['expl_trust'])[0]:.3f}")
    ok = df['correct']
    if ok.nunique() == 2:
        print(f"Error-detection AUROC: confidence {roc_auc_score(ok, df['confidence']):.3f}, "
              f"explanation trust {roc_auc_score(ok, df['expl_trust']):.3f}")
    return t


def figures(df, out, results_B, outdir):
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        return
    os.makedirs(outdir, exist_ok=True)
    q = pd.qcut(df['agreement'], 4, duplicates='drop')
    g = df.groupby(q, observed=True)['faith_mean_gap'].agg(['mean', 'sem'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(g)), g['mean'], yerr=1.96 * g['sem'], color='#028090', capsize=4)
    ax.set_xticks(range(len(g))); ax.set_xticklabels([f'Q{i + 1}' for i in range(len(g))])
    ax.set_xlabel('SHAP-LIME agreement quartile (Q1 lowest)'); ax.set_ylabel('Faithfulness gap')
    ax.set_title('More agreement -> more faithful explanations'); plt.tight_layout()
    plt.savefig(f'{outdir}/fig1_agreement_vs_faithfulness.png', dpi=200); plt.close()

    t = df.groupby('quadrant').agg(accuracy=('correct', 'mean'), gap=('faith_mean_gap', 'mean'))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, c, ttl in zip(axs, ['accuracy', 'gap'], ['Prediction accuracy', 'Explanation faithfulness gap']):
        ax.barh(t.index, t[c], color=['#00A896', '#02C39A', '#028090', '#E4572E'][:len(t)])
        ax.set_title(ttl)
    plt.tight_layout(); plt.savefig(f'{outdir}/fig2_quadrants.png', dpi=200); plt.close()

    r = pd.DataFrame(results_B).set_index('score')['mean_over_sites']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(r.index, r.values, color='#028090'); ax.set_xlabel('Mean Spearman rho with faithfulness (held-out hospital)')
    ax.set_title('Which score best tracks explanation faithfulness?'); plt.tight_layout()
    plt.savefig(f'{outdir}/fig3_learned_vs_baselines.png', dpi=200); plt.close()
    print(f'\nFigures saved in ./{outdir}/')


def run(df, outdir='figures'):
    pd.set_option('display.width', 220)
    df = prepare(df)
    levels = sorted(c for c in df.columns if c.startswith('stab_'))
    controls(df, levels)
    df, results_B = learned_explanation_trust(df, levels)
    quadrants(df)
    figures(df, None, results_B, outdir)
    return df
