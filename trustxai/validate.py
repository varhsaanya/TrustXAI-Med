"""Does the trust score earn its keep? Error detection, risk-coverage, and per-site behaviour."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def error_detection_auroc(df, cols=('confidence', 'agreement', 'stability', 'trust')):
    """AUROC for ranking CORRECT predictions above WRONG ones. 0.5 = useless."""
    ok = df['correct']
    res = {}
    for c in cols:
        res[c] = roc_auc_score(ok, df[c]) if ok.nunique() == 2 else np.nan
    return pd.Series(res).round(3)


def risk_coverage(df, score='trust', levels=(1.0, .9, .8, .7, .6, .5)):
    d = df.sort_values(score, ascending=False)
    return pd.DataFrame([{'coverage': c, 'n': int(len(d) * c),
                          'accuracy': round(d.head(int(len(d) * c))['correct'].mean(), 3)}
                         for c in levels])


def per_site(df):
    g = df.groupby('site')
    t = g.agg(n=('trust', 'size'), accuracy=('correct', 'mean'), trust=('trust', 'mean'),
              confidence=('confidence', 'mean'), agreement=('agreement', 'mean'),
              stability=('stability', 'mean'), disease_rate=('target', 'mean'))
    return t.round(3)
