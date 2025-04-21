# training/train.py

import numpy as np
from sklearn.model_selection import train_test_split
from models.ensemble import StackingEnsembleClassifier
from models.lightgbm import LGBMClassifier
from models.xgboost import XGBClassifier
from models.randomforest import RFClassifier
from models.catboost import CatBoostClassifier
from models.mlp import MLPClassifier
from models.rnn import RNNClassifier
from models.transformer import TransformerClassifier
from utils.visualization import plot_learning_curve
from sklearn.metrics import classification_report, roc_auc_score


def train_model(X, y, config):
    """
    モデルを訓練する関数
    Args:
        X (numpy.ndarray): 特徴量データ
        y (numpy.ndarray): 目的変数
        config (dict): 設定情報（ハイパーパラメータ等）

    Returns:
        StackingEnsembleClassifier: 訓練済みアンサンブルモデル
    """
    # モデル設定
    base_models = [
        LGBMClassifier(config['lgbm_params']),
        XGBClassifier(config['xgb_params']),
        RFClassifier(config['rf_params']),
        CatBoostClassifier(config['catboost_params']),
        MLPClassifier(config['mlp_params']),
        RNNClassifier(config['rnn_params']),
        TransformerClassifier(config['transformer_params']),
    ]

    # アンサンブルモデル
    ensemble_model = StackingEnsembleClassifier(
        base_models=base_models,
        meta_model=config['meta_model'],
        n_folds=config['n_folds'],
        use_proba=config['use_proba']
    )

    # データ分割（学習/検証）
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 学習
    ensemble_model.fit(X_train, y_train)

    # 検証
    y_val_pred = ensemble_model.predict(X_val)
    print("Classification Report:")
    print(classification_report(y_val, y_val_pred))
    print("ROC AUC Score:", roc_auc_score(y_val, ensemble_model.predict_proba(X_val)[:, 1]))

    # 学習曲線のプロット
    plot_learning_curve(ensemble_model, X_train, y_train)

    return ensemble_model


if __name__ == "__main__":
    # データのロード（仮定のロード方法）
    # X, y = load_data()  # 適宜データを読み込む処理を追加

    # ハイパーパラメータ設定
    config = {
        'lgbm_params': {'num_leaves': 31, 'learning_rate': 0.05},
        'xgb_params': {'max_depth': 6, 'learning_rate': 0.1},
        'rf_params': {'n_estimators': 100, 'max_depth': 5},
        'catboost_params': {'iterations': 1000, 'learning_rate': 0.05},
        'mlp_params': {'hidden_layer_sizes': (100, 50), 'activation': 'relu'},
        'rnn_params': {'input_size': 1, 'hidden_size': 64, 'num_layers': 2},
        'transformer_params': {'d_model': 64, 'nhead': 4, 'num_layers': 2},
        'meta_model': 'logistic',  # 'logistic' or another meta-model
        'n_folds': 5,
        'use_proba': True
    }

    # モデルの学習
    # X, y は事前にデータを準備してください
    # trained_model = train_model(X, y, config)