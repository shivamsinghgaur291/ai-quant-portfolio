# data_loader.py
import yfinance as yf
import pandas as pd
import numpy as np

DEFAULT_TICKERS = {
    'SPX': '^GSPC',
    'FTSE': '^FTSE',
    'NIKKEI': '^N225',
    'EEM': 'EEM',
    'GOLD': 'GC=F',     # or 'XAUUSD=X' depending on source
    'UST10Y': '^TNX'    # proxy (may be different); you can replace with FRED later
}

def download_data(tickers=DEFAULT_TICKERS, start='2010-01-01', end='2020-12-31', save_csv=False):
    df_close = pd.DataFrame()
    for name, t in tickers.items():
        df = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
        if 'Adj Close' in df.columns:
            s = df['Adj Close']
        else:
            s = df['Close']
        s.name = str(name)   # ✅ set the series name directly
        df_close = pd.concat([df_close, s], axis=1)
    df_close = df_close.dropna(how='all')
    if save_csv:
        df_close.to_csv('data/prices.csv')
    return df_close


def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()
