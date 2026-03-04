import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew


# ===============================
# AutoPowerTransformer
# ===============================

class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []
        self.pt = PowerTransformer(method='yeo-johnson')

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self

        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()

        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols])

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not self.skewed_cols:
            return X

        X_copy = X.copy()
        X_copy[self.skewed_cols] = self.pt.transform(X_copy[self.skewed_cols])
        return X_copy


# ===============================
# FeatureEngineer
# ===============================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, windows=[5, 10]):
        self.windows = windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        price = X.iloc[:, 0]
        X_out = pd.DataFrame(index=X.index)

        for w in self.windows:
            X_out[f'EMA_{w}'] = price.ewm(span=w, min_periods=w).mean()
            X_out[f'ROC_{w}'] = price.pct_change(w)

        return X_out


# ===============================
# FeatureSelector
# ===============================

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, missing_threshold=0.3, corr_threshold=0.03, cardinality_threshold=0.9):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.cardinality_threshold = cardinality_threshold
        self.features_to_keep = []

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()
        X_filtered = X[cols_low_missing]

        cat_cols = X_filtered.select_dtypes(exclude='number').columns
        cols_to_drop = []

        for col in cat_cols:
            uniqueness_ratio = X_filtered[col].nunique() / len(X_filtered)
            if uniqueness_ratio > self.cardinality_threshold:
                cols_to_drop.append(col)

        remaining_cats = [c for c in cat_cols if c not in cols_to_drop]

        numeric_X = X_filtered.select_dtypes(include='number')
        if y is not None and not numeric_X.empty:
            temp_df = numeric_X.copy()
            temp_df['target'] = y
            correlations = temp_df.corr()['target'].abs().drop('target')
            numeric_to_keep = correlations[correlations >= self.corr_threshold].index.tolist()
        else:
            numeric_to_keep = numeric_X.columns.tolist()

        self.features_to_keep = numeric_to_keep + remaining_cats
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.features_to_keep]