# data/preprocessing.py

import pandas as pd
import numpy as np
import yfinance as yf
import talib

def load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """株価データの取得"""
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標の追加と特徴量エンジニアリング"""

    open_col = df['open']
    high_col = df['high']
    low_col = df['low']
    close_col = df['close']
    volume = df['volume']

    orig_columns = df.columns

    hilo = (df['high'] + df['low']) / 2
    # 価格(hilo または close)を引いた後、価格(close)で割ることで標準化してるものあり

    # ATR
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=7)
    df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=7)

    # ボリンジャーバンド
    df['BBANDS_upperband'], df['BBANDS_middleband'], df['BBANDS_lowerband'] = talib.BBANDS(df['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    df['BBANDS_upperband'] = (df['BBANDS_upperband'] - hilo) / df['close']
    df['BBANDS_middleband'] = (df['BBANDS_middleband'] - hilo) / df['close']
    df['BBANDS_lowerband'] = (df['BBANDS_lowerband'] - hilo) / df['close']

    # 移動平均
    df['DEMA'] = (talib.DEMA(close_col, timeperiod=30) - hilo) / close_col
    df['EMA'] = (talib.EMA(close_col, timeperiod=30) - hilo) / close_col
    df['EMA_short'] = (talib.EMA(close_col, timeperiod=5) - hilo) / close_col
    df['EMA_middle'] = (talib.EMA(close_col, timeperiod=20) - hilo) / close_col
    df['EMA_long'] = (talib.EMA(close_col, timeperiod=40) - hilo) / close_col
    df['HT_TRENDLINE'] = (talib.HT_TRENDLINE(close_col) - hilo) / close_col
    df['KAMA'] = (talib.KAMA(close_col, timeperiod=30) - hilo) / close_col
    df['MA'] = (talib.MA(close_col, timeperiod=30, matype=0) - hilo) / close_col
    df['MIDPOINT'] = (talib.MIDPOINT(close_col, timeperiod=14) - hilo) / close_col
    df['SMA'] = (talib.SMA(close_col, timeperiod=30) - hilo) / close_col
    df['T3'] = (talib.T3(close_col, timeperiod=5, vfactor=0) - hilo) / close_col
    df['HMA'] = talib.WMA(close_col, timeperiod=30)
    df['TEMA'] = (talib.TEMA(close_col, timeperiod=30) - hilo) / close_col
    df['TRIMA'] = (talib.TRIMA(close_col, timeperiod=30) - hilo) / close_col
    df['WMA'] = (talib.WMA(close_col, timeperiod=30) - hilo) / close_col

    # MACD
    df['MACD_macd'], df['MACD_macdsignal'], df['MACD_macdhist'] = talib.MACD(close_col, fastperiod=12, slowperiod=26, signalperiod=9) # Use close_col instead of close
    df['MACD_macd'] /= close_col # Use close_col instead of close
    df['MACD_macdsignal'] /= close_col # Use close_col instead of close
    df['MACD_macdhist'] /= close_col # Use close_col instead of close
    df['MACD_EXT'], df['MACD_SIGNAL_EXT'], df['MACD_HIST_EXT'] = talib.MACDEXT(close_col, fastperiod=12, slowperiod=26, signalperiod=9, fastmatype=0, slowmatype=0, signalmatype=0) # Use close_col instead of close

    # 線形回帰系
    df['LINEARREG'] = (talib.LINEARREG(close_col, timeperiod=14) - close_col) / close_col # Use close_col instead of close
    df['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close_col, timeperiod=14) / close_col # Use close_col instead of close
    df['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close_col, timeperiod=14) # Use close_col instead of close
    df['LINEARREG_INTERCEPT'] = (talib.LINEARREG_INTERCEPT(close_col, timeperiod=14) - close_col) / close_col # Use close_col instead of close

    # AD系
    df['AD'] = talib.AD(high_col, low_col, close_col, volume) / close_col # Use high_col, low_col, close_col instead of high, low, close
    df['ADX'] = talib.ADX(high_col, low_col, close_col, timeperiod=14) # Use high_col, low_col, close_col instead of high, low, close
    df['ADXR'] = talib.ADXR(high_col, low_col, close_col, timeperiod=14) # Use high_col, low_col, close_col instead of high, low, close
    df['ADOSC'] = talib.ADOSC(high_col, low_col, close_col, volume, fastperiod=3, slowperiod=10) / close_col # Use high_col, low_col, close_col instead of high, low, close
    df['OBV'] = talib.OBV(close_col, volume) / close_col # Use close_col instead of close

    # オシレーター系
    df['APO'] = talib.APO(close_col, fastperiod=12, slowperiod=26, matype=0) / close_col  # Changed close to close_col
    df['BOP'] = talib.BOP(open_col, high_col, low_col, close_col)  # Changed open, high, low, close to their respective _col variables
    df['CCI'] = talib.CCI(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['DX'] = talib.DX(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['MFI'] = talib.MFI(high_col, low_col, close_col, volume, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['MINUS_DI'] = talib.MINUS_DI(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['PLUS_DI'] = talib.PLUS_DI(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['MOM'] = talib.MOM(close_col, timeperiod=10) / close_col  # Changed close to close_col
    df['RSI'] = talib.RSI(close_col, timeperiod=14)  # Changed close to close_col
    df['TRIX'] = talib.TRIX(close_col, timeperiod=30)  # Changed close to close_col
    df['ULTOSC'] = talib.ULTOSC(high_col, low_col, close_col, timeperiod1=7, timeperiod2=14, timeperiod3=28)  # Changed high, low, close to their respective _col variables
    df['WILLR'] = talib.WILLR(high_col, low_col, close_col, timeperiod=14)  # Changed high, low, close to their respective _col variables
    df['SAR'] = talib.SAR(high_col, low_col, acceleration=0.02, maximum=0.2)  # Changed high, low to their respective _col variables

    # ストキャスティクス
    df['STOCH_slowk'], df['STOCH_slowd'] = talib.STOCH(high_col, low_col, close_col, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0) # Changed high, low, close to high_col, low_col, close_col
    df['STOCHF_fastk'], df['STOCHF_fastd'] = talib.STOCHF(high_col, low_col, close_col, fastk_period=5, fastd_period=3, fastd_matype=0) # Changed high, low, close to high_col, low_col, close_col
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(close_col, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0) # Changed close to close_col

    # ボラティリティ系
    df['MINUS_DM'] = talib.MINUS_DM(high_col, low_col, timeperiod=14) / close_col # Changed high, low, close to high_col, low_col, close_col
    df['PLUS_DM'] = talib.PLUS_DM(high_col, low_col, timeperiod=14) / close_col # Changed high, low, close to high_col, low_col, close_col
    df['STDDEV'] = talib.STDDEV(close_col, timeperiod=5, nbdev=1) # Changed close to close_col
    df['TRANGE'] = talib.TRANGE(high_col, low_col, close_col) # Changed high, low, close to high_col, low_col, close_col
    df['VAR'] = talib.VAR(close_col, timeperiod=5, nbdev=1) # Changed close to close_col
    df['ATR'] = talib.ATR(high_col, low_col, close_col, timeperiod=14) # Changed high, low, close to high_col, low_col, close_col
    df['NATR'] = talib.NATR(high_col, low_col, close_col, timeperiod=14) # Changed high, low, close to high_col, low_col, close_col
    df['VOLATILITY_index'] = df['ATR'] / df['STDDEV']

    # ヒルベルト変換
    df['HT_DCPERIOD'] = talib.HT_DCPERIOD(close_col) # Changed close to close_col
    df['HT_DCPHASE'] = talib.HT_DCPHASE(close_col) # Changed close to close_col
    df['HT_PHASOR_inphase'], df['HT_PHASOR_quadrature'] = talib.HT_PHASOR(close_col) # Changed close to close_col
    df['HT_PHASOR_inphase'] /= close_col # Changed close to close_col
    df['HT_PHASOR_quadrature'] /= close_col # Changed close to close_col
    df['HT_SINE_sine'], df['HT_SINE_leadsine'] = talib.HT_SINE(close_col) # Changed close to close_col
    df['HT_SINE_sine'] /= close_col # Changed close to close_col
    df['HT_SINE_leadsine'] /= close_col # Changed close to close_col
    df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close_col) # Changed close to close_col

    # その他
    df['ROC'] = talib.ROC(close_col, timeperiod=10) / close_col # Changed close to close_col
    df['STDDEV'] = talib.STDDEV(close_col, timeperiod=5, nbdev=1) / close_col # Changed close to close_col
    df['TRANGE'] = talib.TRANGE(high_col, low_col, close_col) / close_col # Changed high, low, close to high_col, low_col, close_col
    df['AROON_aroondown'], df['AROON_aroonup'] = talib.AROON(high_col, low_col, timeperiod=14) # Changed high, low to high_col, low_col
    df['AROONOSC'] = talib.AROONOSC(high_col, low_col, timeperiod=14) # Changed high, low to high_col, low_col
    df['BETA'] = talib.BETA(high_col, low_col, timeperiod=5) # Changed high, low to high_col, low_col
    df['CORREL'] = talib.CORREL(high_col, low_col, timeperiod=30) # Changed high, low to high_col, low_col
    df['Price_ratio'] = df['close'] / df['close'].shift(1)  # Changed df[close_col] to df['close']
    df['HIGH_ratio'] = df['high'] / df['high'].shift(1)  # Changed df[high_col] to df['high']
    df['LOW_ratio'] = df['low'] / df['low'].shift(1)  # Changed df[low_col] to df['low']
    # Lag特徴量
    df['CLOSE_lag_1'] = df['close'].shift(1)  # 1日遅れの終値 # Changed df[close_col] to df['close']
    df['CLOSE_lag_5'] = df['close'].shift(5)  # 5日遅れの終値 # Changed df[close_col] to df['close']
    df['MOVIENG_avg_5'] = df['close'].rolling(window=5).mean()  # 5日移動平均 # Changed df[close_col] to df['close']
    # 周期性の特徴量
    df['DAY_of_week'] = df.index.dayofweek  # 曜日（0=月曜, ..., 6=日曜）
    df['DAY_of_month'] = df.index.day       # 月内の日付（1〜31）
    df['MONTH'] = df.index.month            # 月（1〜12）
    df['WEEK_of_year'] = df.index.isocalendar().week.astype(int)  # 年内の週番号
    df['QUARTER'] = df.index.quarter        # 四半期（1〜4）
    df['IS_weekend'] = (df['DAY_of_week'] >= 5).astype(int)  # 週末かどうか
    df['SIN_day'] = np.sin(2 * np.pi * df['DAY_of_week'] / 7)  # 日周期
    df['COS_day'] = np.cos(2 * np.pi * df['DAY_of_week'] / 7)  # 日周期
    df['SIN_month'] = np.sin(2 * np.pi * df['MONTH'] / 12)  # 月周期
    df['COS_month'] = np.cos(2 * np.pi * df['MONTH'] / 12)  # 月周期

    # ローソク足パターン（ベスト50）
    # ハンマー（Hammer）
    def is_hammer(df):
        body = abs(df['close'] - df['open'])
        range_ = df['high'] - df['low']
        return (body / range_) < 0.3  # ボディが範囲の30％未満
    df['is_hammer'] = is_hammer(df).astype(int)

    # 逆ハンマー（Inverted Hammer）
    def is_inverted_hammer(df):
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        return (upper_shadow > body) & (lower_shadow < body)
    df['is_inverted_hammer'] = is_inverted_hammer(df).astype(int)

    # ピンバー（Pin Bar）
    def is_pin_bar(df):
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        return (upper_shadow > body) & (lower_shadow > body)
    df['is_pin_bar'] = is_pin_bar(df).astype(int)

    # ドージ（Doji）
    def is_doji(df):
        body = abs(df['close'] - df['open'])
        range_ = df['high'] - df['low']
        return body / range_ < 0.1  # ボディが範囲の10％未満
    df['is_doji'] = is_doji(df).astype(int)

    # エングルフィング（Engulfing）˙
    def is_engulfing(df):
        return (df['close'].shift(1) < df['open'].shift(1)) & (df['open'] > df['close']) & (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
    df['is_engulfing'] = is_engulfing(df).astype(int)

    # モーニングスター（Morning Star）
    def is_morning_star(df):
        prev_candle = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] < df['open'])
        next_candle = (df['close'] > df['open'])
        return prev_candle & next_candle
    df['is_morning_star'] = is_morning_star(df).astype(int)

    # イヴニングスター（Evening Star）
    def is_evening_star(df):
        prev_candle = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] > df['open'])
        next_candle = (df['close'] < df['open'])
        return prev_candle & next_candle
    df['is_evening_star'] = is_evening_star(df).astype(int)

    # 三川（Three River）
    def is_three_river(df):
        return (df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2))
    df['is_three_river'] = is_three_river(df).astype(int)

    # ダーククラウドカバー（Dark closeoud Cover）
    def is_dark_cloud_cover(df):  # Changed function name to is_dark_cloud_cover
        return (df['close'] < df['open']) & (df['open'].shift(1) > df['close'].shift(1)) & (df['close'] < df['open'].shift(1))
    df['is_dark_cloud_cover'] = is_dark_cloud_cover(df).astype(int)

    # 反転ハラミ（Harami）
    def is_harami(df):
        return (df['open'] < df['close'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'] > df['close']) & (df['close'] < df['open'])
    df['is_harami'] = is_harami(df).astype(int)

    # 反転ハラミ（Bearish Harami）
    def is_bearish_harami(df):
        return (df['open'] > df['close'].shift(1)) & (df['close'] < df['open'].shift(1)) & (df['open'] < df['close']) & (df['close'] > df['open'])
    df['is_bearish_harami'] = is_bearish_harami(df).astype(int)

    # インサイドバー（Inside Bar）
    def is_inside_bar(df):
        return (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['is_inside_bar'] = is_inside_bar(df).astype(int)

    # アウトサイドバー（Outside Bar）
    def is_outside_bar(df):
        return (df['high'] > df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    df['is_outside_bar'] = is_outside_bar(df).astype(int)

    # グランビルの法則1（Grandville's Rule 1）
    def is_grandville_rule_1(df):
        return (df['close'] > df['open'].shift(1)) & (df['close'] > df['high'].shift(1))
    df['is_grandville_rule_1'] = is_grandville_rule_1(df).astype(int)

    # グランビルの法則2（Grandville's Rule 2）
    def is_grandville_rule_2(df):
        return (df['close'] < df['open'].shift(1)) & (df['close'] < df['low'].shift(1))
    df['is_grandville_rule_2'] = is_grandville_rule_2(df).astype(int)

    # フォーリングナイト（Falling Knight）
    def is_falling_knight(df):
        return (df['open'] < df['close'].shift(1)) & (df['high'] < df['high'].shift(1)) & (df['low'] < df['low'].shift(1))
    df['is_falling_knight'] = is_falling_knight(df).astype(int)

    # ライジングサン（Rising Sun）
    def is_rising_sun(df):
        return (df['close'] > df['open'].shift(1)) & (df['high'] > df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['is_rising_sun'] = is_rising_sun(df).astype(int)

    # コマ（Kicker）
    def is_kicker(df):
        return (df['close'] > df['open'].shift(1)) & (df['close'] < df['open']) & (df['open'] > df['close'].shift(1))
    df['is_kicker'] = is_kicker(df).astype(int)

    # エンベロープ（Envelope）
    def is_envelope(df):
        return (df['high'] > df['open']) & (df['low'] < df['open'])
    df['is_envelope'] = is_envelope(df).astype(int)

    # ダブルトップ（Double Top）
    def is_double_top(df):
        return (df['high'].shift(1) > df['high']) & (df['high'].shift(-1) > df['high'])
    df['is_double_top'] = is_double_top(df).astype(int)

    # ダブルボトム（Double Bottom）
    def is_double_bottom(df):
        return (df['low'].shift(1) < df['low']) & (df['low'].shift(-1) < df['low'])
    df['is_double_bottom'] = is_double_bottom(df).astype(int)

    # トリプルトップ（Triple Top）
    def is_triple_top(df):
        return (df['high'].shift(1) > df['high']) & (df['high'].shift(2) > df['high'])
    df['is_triple_top'] = is_triple_top(df).astype(int)

    # トリプルボトム（Triple Bottom）
    def is_triple_bottom(df):
        return (df['low'].shift(1) < df['low']) & (df['low'].shift(2) < df['low'])
    df['is_triple_bottom'] = is_triple_bottom(df).astype(int)

    # ショートライン（Short Line）
    def is_short_line(df):
        return (df['close'] - df['open']).abs() < (df['high'] - df['low']) * 0.2
    df['is_short_line'] = is_short_line(df).astype(int)

    # ロングライン（Long Line）
    def is_long_line(df):
        return (df['close'] - df['open']).abs() > (df['high'] - df['low']) * 0.7
    df['is_long_line'] = is_long_line(df).astype(int)

    # バルス（Bulls）
    def is_bulls(df):
        return (df['close'] > df['open']) & (df['close'] > df['close'].shift(1))
    df['is_bulls'] = is_bulls(df).astype(int)

    # ベアス（Bears）
    def is_bears(df):
        return (df['close'] < df['open']) & (df['close'] < df['close'].shift(1))
    df['is_bears'] = is_bears(df).astype(int)

    # スター（Star）
    def is_star(df):
        return (df['close'] > df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
    df['is_star'] = is_star(df).astype(int)

    # ドージ（Doji）
    def is_doji(df):
        return (abs(df['open'] - df['close']) / (df['high'] - df['low'])) < 0.1
    df['is_doji2'] = is_doji(df).astype(int)

    # ピンバー（Pin Bar）
    def is_pin_bar(df):
        body = abs(df['close'] - df['open'])
        range_ = df['high'] - df['low']
        return ((body / range_) < 0.3).astype(int)
    df['is_pin_bar2'] = is_pin_bar(df).astype(int)

    # トンボ（Dragonfly Doji）
    def is_dragonfly_doji(df):
        return (abs(df['close'] - df['open']) < 0.1) & (df['low'] == df['open'])
    df['is_dragonfly_doji'] = is_dragonfly_doji(df).astype(int)

    # 逆トンボ（Gravestone Doji）
    def is_gravestone_doji(df):
        return (abs(df['close'] - df['open']) < 0.1) & (df['high'] == df['open'])
    df['is_gravestone_doji'] = is_gravestone_doji(df).astype(int)

    # アタックバー（Attack Bar）
    def is_attack_bar(df):
        return (df['close'] > df['open']) & (df['close'] > df['high'].shift(1))
    df['is_attack_bar'] = is_attack_bar(df).astype(int)

    # ピンバー逆（Inverted Pin Bar）
    def is_inverted_pin_bar(df):
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['close', 'open']].max(axis=1)
        lower_shadow = df[['close', 'open']].min(axis=1) - df['low']
        return (upper_shadow > body) & (lower_shadow < body)
    df['is_inverted_pin_bar'] = is_inverted_pin_bar(df).astype(int)

    # 上昇三法（Three White Soldiers）
    def is_three_white_soldiers(df):
        return (df['close'] > df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2))
    df['is_three_white_soldiers'] = is_three_white_soldiers(df).astype(int)

    # 下降三法（Three Black Crows）
    def is_three_black_crows(df):
        return (df['close'] < df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) < df['open'].shift(2))
    df['is_three_black_crows'] = is_three_black_crows(df).astype(int)

    # ヘッドアンドショルダーズ（Head and Shoulders）
    def is_head_and_shoulders(df):
        return (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])
    df['is_head_and_shoulders'] = is_head_and_shoulders(df).astype(int)

    # カップアンドハンドル（Cup and Handle）
    def is_cup_and_handle(df):
        return (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])
    df['is_cup_and_handle'] = is_cup_and_handle(df).astype(int)

    # 上昇の旗（Rising Flag）
    def is_rising_flag(df):
        return (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1))
    df['is_rising_flag'] = is_rising_flag(df).astype(int)

    # 下降の旗（Falling Flag）
    def is_falling_flag(df):
        return (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1))
    df['is_falling_flag'] = is_falling_flag(df).astype(int)

    # ラダーパターン（Ladder Pattern）
    def is_ladder_pattern(df):
        return (df['high'] > df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['is_ladder_pattern'] = is_ladder_pattern(df).astype(int)

    # サポートライン（Support Line）
    def is_support_line(df):
        return df['low'] == df['low'].shift(1)
    df['is_support_line'] = is_support_line(df).astype(int)

    # レジスタンスライン（Resistance Line）
    def is_resistance_line(df):
        return df['high'] == df['high'].shift(1)
    df['is_resistance_line'] = is_resistance_line(df).astype(int)

    # フラッグパターン（Flag Pattern）
    def is_flag_pattern(df):
        return (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2))
    df['is_flag_pattern'] = is_flag_pattern(df).astype(int)

    # ペナントパターン（Pennant Pattern）
    def is_pennant_pattern(df):
        return (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'].shift(2) > df['open'].shift(2))
    df['is_pennant_pattern'] = is_pennant_pattern(df).astype(int)

    # 価格変化のラグ特徴量
    for lag in range(1, 4):
        df[f'return_lag_{lag}'] = df['Close'].pct_change(lag)

    # ターゲット変数（例: 翌日のリターンがプラスなら1, マイナスなら0）
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    # 欠損値の除去
    df.dropna(inplace=True)

    return df