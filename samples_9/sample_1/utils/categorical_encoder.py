from sklearn.base import BaseEstimator, TransformerMixin

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # No fitting necessary for this transformer

    def transform(self, X):
        X = X.copy()
        X['SEX'] = X['SEX'].map({'M': 1, 'F': 0})
        return X