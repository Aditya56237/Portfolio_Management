# src/risk_analysis.py
import numpy as np
import pandas as pd

def historical_var(portfolio_returns: pd.Series, alpha=0.05):
    return float(np.percentile(portfolio_returns, 100*alpha))

def parametric_var(mean, std, alpha=0.05):
    from scipy.stats import norm
    return float(mean + std*norm.ppf(alpha))

def monte_carlo_portfolio(mu, Sigma, weights, n_steps=252, n_paths=10000, seed=42):
    rng = np.random.default_rng(seed)
    w = np.asarray(weights)
    # simulate daily portfolio returns ~ N(w^T mu, w^T Sigma w)
    p_mean = float(w @ mu)
    p_var  = float(w @ Sigma @ w)
    draws = rng.normal(p_mean, np.sqrt(p_var), size=(n_paths, n_steps))
    # compound to horizon
    path_pnl = draws.sum(axis=1)
    return path_pnl  # vector of simulated horizon returns

def sharpe_ratio(mean, std, rf=0.0):
    return (mean - rf)/std if std>0 else np.nan
