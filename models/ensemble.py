# models/ensemble.py

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np


class StackingEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models=None, meta_model=None, n_folds=5, use_proba=True, random_state=42):
        self.base_models = base_models if base_models is not None else []
        self.meta_model = meta_model if meta_model is not None else LogisticRegression()
        self.n_folds = n_folds
        self.use_proba = use_proba
        self.random_state = random_state

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_features_ = np.zeros((X.shape[0], len(self.base_models)))
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for i, model in enumerate(self.base_models):
            for train_idx, val_idx in kfold.split(X, y):
                cloned_model = clone(model)
                cloned_model.fit(X[train_idx], y[train_idx])
                self.base_models_[i].append(cloned_model)
                if self.use_proba:
                    preds = cloned_model.predict_proba(X[val_idx])[:, 1]
                else:
                    preds = cloned_model.predict(X[val_idx])
                self.meta_features_[val_idx, i] = preds

        self.meta_model.fit(self.meta_features_, y)
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack([
            np.mean([model.predict_proba(X)[:, 1] if self.use_proba else model.predict(X)
                    for model in base_models], axis=0)
            for base_models in self.base_models_
        ])
        meta_probs = self.meta_model.predict_proba(meta_features)
        return meta_probs

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y, self.predict_proba(X)[:, 1])