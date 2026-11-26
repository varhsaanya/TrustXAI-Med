"""Model zoo. Each model is a full Pipeline (impute -> engineer -> [scale] -> classifier)."""
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from .features import make_preprocessor


def build_model(name, features, seed=42):
    if name == 'logreg':
        return Pipeline(make_preprocessor(features, scale=True).steps +
                        [('clf', LogisticRegression(max_iter=3000, random_state=seed))])
    if name == 'rf':
        return Pipeline(make_preprocessor(features).steps +
                        [('clf', RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                                        class_weight='balanced', random_state=seed))])
    if name == 'xgb':
        return Pipeline(make_preprocessor(features).steps +
                        [('clf', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                               subsample=0.8, colsample_bytree=0.8,
                                               eval_metric='logloss', random_state=seed))])
    raise ValueError(name)


def fit_with_calibration(name, X, y, features, seed=42):
    """Returns (base pipeline for SHAP/LIME, calibrated copy for trustworthy probabilities)."""
    base = build_model(name, features, seed).fit(X[features], y)
    calibrated = CalibratedClassifierCV(clone(build_model(name, features, seed)),
                                        method='sigmoid', cv=5).fit(X[features], y)
    return base, calibrated
