# models/catboost.py

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

class CatBoostModel(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 iterations=500,
                 learning_rate=0.03,
                 depth=6,
                 l2_leaf_reg=3.0,
                 loss_function='Logloss',
                 eval_metric='AUC',
                 random_seed=42,
                 verbose=0):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.random_seed = random_seed
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_seed
        )

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function=self.loss_function,
            eval_metric=self.eval_metric,
            random_seed=self.random_seed,
            verbose=self.verbose
        )

        self.model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        y_proba = self.predict_proba(X)[:, 1]
        return roc_auc_score(y, y_proba)