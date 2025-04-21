# config.py

# 損切り・利確・トレーリングストップのパラメータ（%）
STOP_LOSS = 0.05
TAKE_PROFIT = 0.10
TRAIL_STOP = 0.03

# 売買手数料・スリッページ
COMMISSION = 0.001  # 0.1%
SLIPPAGE = 0.001     # 0.1%

# 学習対象のカラム名
TARGET_COLUMN = 'long_target'

# 使用する特徴量（例：事前に作成したテクニカル指標やラグ特徴量など）
FEATURES = [
    'close_lag1', 'close_lag2', 'close_lag3',
    'sma_5', 'sma_10', 'sma_20',
    'rsi_14', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower',
    'volume_lag1', 'volatility_5'
]

# データの読み込み元（将来的にパスを外出しにすることも可能）
DATA_PATH = "data/your_stock_data.csv"

# モデル保存先のディレクトリ（必要に応じて使用）
MODEL_DIR = "models/saved_models"

# モデル評価の分割設定
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42