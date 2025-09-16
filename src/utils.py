import numpy as np, pandas as pd

def metrics_from_nav(nav_series):
    # nav_series indexed by date, daily
    daily_ret = nav_series.pct_change().dropna()
    ann_ret = (nav_series.iloc[-1] / nav_series.iloc[0]) ** (252.0/len(daily_ret)) - 1
    ann_vol = daily_ret.std() * (252**0.5)
    sharpe = ann_ret / ann_vol if ann_vol>0 else np.nan
    # max drawdown
    cum = nav_series.cummax()
    drawdown = (nav_series - cum) / cum
    mdd = drawdown.min()
    return dict(annual_return=ann_ret, annual_vol=ann_vol, sharpe=sharpe, max_drawdown=mdd)
