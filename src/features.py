# features.py
import pandas as pd
import numpy as np

def rolling_features(prices, returns, window_short=5, window_long=21, vol_window=21):
    # returns: DataFrame of log returns
    feats = {}
    feats['ret'] = returns
    feats['ma_short'] = returns.rolling(window_short).mean()
    feats['ma_long']  = returns.rolling(window_long).mean()
    feats['vol']      = returns.rolling(vol_window).std()
    # Flatten features per asset by suffixing asset names
    df_list = []
    for name in returns.columns:
        df = pd.concat([
            returns[name].rename(f'{name}_ret'),
            feats['ma_short'][name].rename(f'{name}_ma5'),
            feats['ma_long'][name].rename(f'{name}_ma21'),
            feats['vol'][name].rename(f'{name}_vol21')
        ], axis=1)
        df_list.append(df)
    combined = pd.concat(df_list, axis=1)
    combined = combined.dropna()
    return combined
