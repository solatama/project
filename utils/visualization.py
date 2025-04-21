# main.py

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    STOP_LOSS,
    TAKE_PROFIT,
    TRAIL_STOP,
    COMMISSION,
    SLIPPAGE,
    FEATURES
)

from data.preprocessing import load_data
from models.ensemble import EnsembleModel
from training.train import train_models
from evaluation.evaluate import auto_model_evaluation
from utils.visualization import plot_learning_curve, plot_prediction_result

def main():
    start_time = time.time()

    # データの読み込み
    df = load_data()

    # モデルの学習
    models = train_models(df, FEATURES)

    # アンサンブルモデルの作成と学習
    ensemble_model = EnsembleModel(models)
    ensemble_model.fit(df[FEATURES], df['long_target'])

    # バックテストの実行
    pnl = ensemble_model.backtest(df, FEATURES)

    # シャープレシオの計算
    sharpe_ratio = ensemble_model.calculate_sharpe_ratio(pnl)
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    # モデルの評価
    best_params_dict = {
        'lightgbm': {},
        'xgboost': {},
        'catboost': {},
        'randomforest': {},
        'mlp': {'hidden_dim': 64, 'num_layers': 2},
        'transformer': {'hidden_dim': 64, 'num_heads': 4, 'num_layers': 2, 'dropout': 0.1},
        'rnn': {'hidden_dim': 64, 'num_layers': 1}
    }

    models_to_run = ['lightgbm', 'xgboost', 'catboost', 'randomforest', 'mlp', 'transformer', 'rnn']
    model_type_map = {
        'lightgbm': 'tree-based',
        'xgboost': 'tree-based',
        'catboost': 'tree-based',
        'randomforest': 'tree-based',
        'mlp': 'neural-network',
        'transformer': 'neural-network',
        'rnn': 'neural-network'
    }

    auto_model_evaluation(best_params_dict, models_to_run, model_type_map, df, FEATURES)

    end_time = time.time()
    print(f"✅ 全体の実行時間: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()