"""Per-patient trust components: calibrated confidence, SHAP-LIME agreement, explanation stability."""
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from .data import NUMERIC

TOP_K = 5


def _shap_matrix(base, Xt_bg, Xt_q):
    """|SHAP| for the disease class, shape (n_patients, n_features)."""
    clf = base.named_steps['clf']
    if hasattr(clf, 'coef_'):
        ex = shap.LinearExplainer(clf, Xt_bg)
        sv = ex.shap_values(Xt_q)
    else:
        ex = shap.TreeExplainer(clf)
        sv = ex.shap_values(Xt_q)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


def _topk(v, k=TOP_K):
    return set(np.argsort(-np.abs(v))[:k])


def _jaccard(a, b):
    return len(a & b) / len(a | b)


def _rank_corr(a, b):
    r = spearmanr(np.abs(a), np.abs(b))[0]
    return 0.0 if np.isnan(r) else max(r, 0.0)


def confidence_score(p):
    """0 at p=0.5 (coin flip), 1 at p=0 or p=1 (certain)."""
    return 2 * np.abs(np.asarray(p) - 0.5)


def compute_trust(base, calibrated, X_train, X_test, features, n_noise=10,
                  noise_frac=0.05, lime_samples=1000, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    Xtr, Xte = X_train[features], X_test[features]
    Xt_tr = base[:-1].transform(Xtr)
    Xt_te = base[:-1].transform(Xte)
    cols = list(Xt_tr.columns)
    clf = base.named_steps['clf']
    predict = lambda arr: clf.predict_proba(pd.DataFrame(arr, columns=cols))

    p_cal = calibrated.predict_proba(Xte)[:, 1]      # calibrated -> used for confidence
    p_raw = base.predict_proba(Xte)[:, 1]            # the model the explanations describe
    sv = _shap_matrix(base, Xt_tr, Xt_te)

    # noisy copies: Gaussian noise on continuous inputs only, scaled to each feature's spread
    num = [c for c in features if c in NUMERIC]
    sd = Xtr[num].std()
    noisy_sv = []
    for _ in range(n_noise):
        Xn = Xte.copy()
        Xn[num] = Xn[num] + rng.normal(0, 1, Xn[num].shape) * (noise_frac * sd.values)
        noisy_sv.append(_shap_matrix(base, Xt_tr, base[:-1].transform(Xn)))

    lime = LimeTabularExplainer(Xt_tr.values, feature_names=cols, mode='classification',
                                discretize_continuous=True, random_state=seed)
    rows = []
    for i in range(len(Xte)):
        exp = lime.explain_instance(Xt_te.values[i], predict, num_features=len(cols),
                                    num_samples=lime_samples, labels=(1,))
        lw = np.zeros(len(cols))
        for j, w in exp.as_map()[1]:
            lw[j] = w
        agree_top = _jaccard(_topk(sv[i]), _topk(lw))
        agree_rank = _rank_corr(sv[i], lw)
        stab_top = np.mean([_jaccard(_topk(sv[i]), _topk(n[i])) for n in noisy_sv])
        stab_rank = np.mean([_rank_corr(sv[i], n[i]) for n in noisy_sv])
        rows.append({'confidence': confidence_score(p_cal[i]),
                     'agreement': 0.5 * (agree_top + agree_rank),
                     'stability': 0.5 * (stab_top + stab_rank),
                     'agree_top5': agree_top, 'agree_rank': agree_rank,
                     'stab_top5': stab_top, 'stab_rank': stab_rank})
        if verbose and (i + 1) % 25 == 0:
            print(f'    {i + 1}/{len(Xte)} patients', flush=True)
    out = pd.DataFrame(rows, index=X_test.index)
    out['p_calibrated'] = p_cal
    out['p_raw'] = p_raw
    return out


# v1 weights (no plausibility yet). To be justified by the validation stage, not assumed.
WEIGHTS = {'confidence': 0.4, 'agreement': 0.3, 'stability': 0.3}


def combine(df, weights=WEIGHTS):
    return sum(w * df[k] for k, w in weights.items()) / sum(weights.values())
