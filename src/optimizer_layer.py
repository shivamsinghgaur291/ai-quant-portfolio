# optimizer_layer.py
import numpy as np
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

def max_sharpe_weights(pred_returns, cov_matrix, long_only=True, weight_bounds=(0,1)):
    # pred_returns: 1D array of expected returns (for assets)
    mu = pd.Series(pred_returns)
    # pypfopt expects price series to estimate cov, but we'll pass cov directly:
    # Use EfficientFrontier with objective to max_sharpe.
    from pypfopt import objective_functions
    ef = EfficientFrontier(mu, cov_matrix, weight_bounds=weight_bounds)
    ef.max_sharpe()
    w = ef.clean_weights()
    return np.array([w.get(k,0.0) for k in mu.index])
