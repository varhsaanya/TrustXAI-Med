"""Leakage-safe preprocessing: imputation is fit INSIDE the pipeline on training data only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import NUMERIC, CATEGORICAL


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the 4 engineered features (row-wise and stateless, so no leakage possible)."""
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]  # marks the transformer as fitted (needed for pipeline slicing)
        return self

    def transform(self, X):
        X = X.copy()
        if 'age' in X:
            X['age_middle'] = ((X['age'] > 40) & (X['age'] <= 55)).astype(int)
            X['age_senior'] = (X['age'] > 55).astype(int)
        if {'trestbps', 'thalach'} <= set(X.columns):
            X['bp_hr_product'] = X['trestbps'] * X['thalach']
        if {'oldpeak', 'thalach'} <= set(X.columns):
            X['oldpeak_norm'] = X['oldpeak'] / (X['thalach'] + 1e-6)
        flags = 0
        if 'exang' in X: flags = flags + X['exang']
        if 'oldpeak' in X: flags = flags + (X['oldpeak'] > 1).astype(int)
        if 'slope' in X: flags = flags + (X['slope'] > 1).astype(int)
        X['risk_flags'] = flags
        return X


def make_preprocessor(features, scale=False):
    num = [c for c in features if c in NUMERIC]
    cat = [c for c in features if c in CATEGORICAL]
    imp = ColumnTransformer(
        [('num', SimpleImputer(strategy='median'), num),
         ('cat', SimpleImputer(strategy='most_frequent'), cat)],
        verbose_feature_names_out=False).set_output(transform='pandas')
    steps = [('impute', imp), ('fe', FeatureEngineer())]
    if scale:
        steps.append(('scale', StandardScaler().set_output(transform='pandas')))
    return Pipeline(steps)
