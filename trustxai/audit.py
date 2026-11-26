"""Stage 3 compute: confidence, agreement, stability at several noise levels, and FAITHFULNESS.

Faithfulness (comprehensiveness gap): replace an explanation's top-k features with their training
mean and measure how much the predicted-class probability drops, minus the drop from removing
k RANDOM features. Positive gap = the explanation really pointed at features the model uses.
Caveat: removal happens in the model's (transformed) feature space, so engineered features can be
momentarily inconsistent with the raw ones; this is the standard mean-baseline approximation.
"""
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from .data import NUMERIC
from .trust import _shap_matrix, _topk, _jaccard, _rank_corr, confidence_score, TOP_K


def _remove(Xvals, sets, baseline):
    Xm = Xvals.copy()
    for i, idx in enumerate(sets):
        Xm[i, list(idx)] = baseline[list(idx)]
    return Xm


def compute_audit(base, calibrated, X_train, X_test, features,
                  noise_levels=(0.05, 0.25, 0.5), n_noise=10, n_random=10,
                  lime_samples=1000, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    Xtr, Xte = X_train[features], X_test[features]
    Xt_tr, Xt_te = base[:-1].transform(Xtr), base[:-1].transform(Xte)
    cols, n_f = list(Xt_tr.columns), Xt_tr.shape[1]
    clf = base.named_steps['clf']
    predict = lambda arr: clf.predict_proba(pd.DataFrame(arr, columns=cols))[:, 1]

    p_raw = predict(Xt_te.values)
    p_cal = calibrated.predict_proba(Xte)[:, 1]
    pred = (p_raw >= .5).astype(int)
    sv = _shap_matrix(base, Xt_tr, Xt_te)

    # --- stability at several noise levels (noise = frac * each feature's training std) ---
    num = [c for c in features if c in NUMERIC]
    sd = Xtr[num].std().values
    stab = {}
    for lvl in noise_levels:
        copies = []
        for _ in range(n_noise):
            Xn = Xte.copy()
            Xn[num] = Xn[num] + rng.normal(0, 1, Xn[num].shape) * (lvl * sd)
            copies.append(_shap_matrix(base, Xt_tr, base[:-1].transform(Xn)))
        stab[lvl] = copies

    # --- LIME weights for every patient ---
    lime = LimeTabularExplainer(Xt_tr.values, feature_names=cols, mode='classification',
                                discretize_continuous=True, random_state=seed)
    lw = np.zeros((len(Xte), n_f))
    for i in range(len(Xte)):
        exp = lime.explain_instance(Xt_te.values[i], lambda a: clf.predict_proba(pd.DataFrame(a, columns=cols)),
                                    num_features=n_f, num_samples=lime_samples, labels=(1,))
        for j, w in exp.as_map()[1]:
            lw[i, j] = w
        if verbose and (i + 1) % 25 == 0:
            print(f'    LIME {i + 1}/{len(Xte)}', flush=True)

    # --- faithfulness ---
    baseline = Xt_tr.mean().values
    Xv = Xt_te.values
    ppred = lambda p: np.where(pred == 1, p, 1 - p)
    p0 = ppred(p_raw)
    drop_shap = p0 - ppred(predict(_remove(Xv, [np.argsort(-np.abs(sv[i]))[:TOP_K] for i in range(len(Xv))], baseline)))
    drop_lime = p0 - ppred(predict(_remove(Xv, [np.argsort(-np.abs(lw[i]))[:TOP_K] for i in range(len(Xv))], baseline)))
    drop_rand = np.mean([p0 - ppred(predict(_remove(Xv, [rng.choice(n_f, TOP_K, replace=False) for _ in range(len(Xv))], baseline)))
                         for _ in range(n_random)], axis=0)

    rows = []
    for i in range(len(Xte)):
        a_top, a_rank = _jaccard(_topk(sv[i]), _topk(lw[i])), _rank_corr(sv[i], lw[i])
        r = {'confidence': confidence_score(p_cal[i]), 'agree_top5': a_top, 'agree_rank': a_rank,
             'agreement': 0.5 * (a_top + a_rank),
             'faith_shap_drop': drop_shap[i], 'faith_lime_drop': drop_lime[i], 'faith_random_drop': drop_rand[i],
             'faith_shap_gap': drop_shap[i] - drop_rand[i], 'faith_lime_gap': drop_lime[i] - drop_rand[i]}
        for lvl in noise_levels:
            t = np.mean([_jaccard(_topk(sv[i]), _topk(n[i])) for n in stab[lvl]])
            k = np.mean([_rank_corr(sv[i], n[i]) for n in stab[lvl]])
            r[f'stab_{lvl}'] = 0.5 * (t + k)
        rows.append(r)
    out = pd.DataFrame(rows, index=X_test.index)
    out['p_raw'], out['p_calibrated'], out['pred'] = p_raw, p_cal, pred
    return out
