# models/lightgbm.py

import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

class LightGBMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 learning_rate=0.05, 
                 num_leaves=31, 
                 n_estimators=100, 
                 max_depth=-1,
                 random_state=42):
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        self.model = lgb.LGBMClassifier(
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='binary_logloss',
            early_stopping_rounds=20,
            verbose=False
        )
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)