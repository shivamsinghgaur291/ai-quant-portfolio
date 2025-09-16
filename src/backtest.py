# backtest.py
import numpy as np
import pandas as pd

def simulate_daily(prices, model, feature_df, scaler, lookback=60, tx_cost=0.001):
    # prices: DataFrame of adjusted closes
    # feature_df: DataFrame of features (aligned)
    # model: trained torch model (expect numpy input)
    T = len(feature_df)
    assets = prices.columns
    nav = [1.0]
    weights = np.zeros((T, len(assets)))
    prev_w = np.zeros(len(assets))
    for t in range(lookback, T-1):
        # prepare model input of last lookback rows
        X_window = feature_df.iloc[t-lookback+1:t+1].values[np.newaxis,:,:]  # (1, seq_len, nfeat)
        # model predict
        import torch
        model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_window).float()
            pred = model(x_t).cpu().numpy()[0]  # predicted next-day returns
        # estimate cov from recent returns window
        ret_window = np.log(prices.pct_change().dropna()).iloc[t-59:t+1]  # 60 days
        cov = ret_window.cov().values
        # get weights
        from .optimizer_layer import max_sharpe_weights
        try:
            w = max_sharpe_weights(pd.Series(pred, index=assets), cov_matrix=cov)
        except Exception as e:
            # fallback equal weight
            w = np.repeat(1/len(assets), len(assets))
        # apply transaction cost (approx)
        turnover = np.sum(np.abs(w - prev_w))
        nav.append(nav[-1] * (1 + (prices.iloc[t+1].values / prices.iloc[t].values - 1).dot(w) - turnover * tx_cost))
        weights[t+1] = w
        prev_w = w
    nav = pd.Series(nav, index=feature_df.index[lookback:]) 
    return nav, weights
