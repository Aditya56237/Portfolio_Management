# src/portfolio.py
import numpy as np
import pandas as pd

def mean_cov(returns: pd.DataFrame):
    mu = returns.mean().values
    Sigma = returns.cov().values
    return mu, Sigma

def minimum_variance_portfolio(returns: pd.DataFrame):
    """Closed-form MVP (weights sum to 1, no other constraints)."""
    _, Sigma = mean_cov(returns)
    inv = np.linalg.inv(Sigma)
    ones = np.ones((inv.shape[0], 1))
    w = (inv @ ones) / (ones.T @ inv @ ones)
    return w.flatten()

def efficient_frontier(returns: pd.DataFrame, n_points=50):
    """Sample the unconstrained efficient frontier.
    Returns DataFrame with columns: target_return, vol, sharpe, weights (as arrays)
    """
    mu, Sigma = mean_cov(returns)
    inv = np.linalg.inv(Sigma)
    ones = np.ones_like(mu)

    A = ones @ inv @ ones
    B = ones @ inv @ mu
    C = mu   @ inv @ mu
    D = A*C - B**2

    mu_min, mu_max = mu.min(), mu.max()
    targets = np.linspace(mu_min, mu_max, n_points)

    rows = []
    for m in targets:
        lam = (C - B*m)/D
        gam = (A*m - B)/D
        w = lam*(inv@ones) + gam*(inv@mu)
        ret = w @ mu
        vol = np.sqrt(w @ Sigma @ w)
        sharpe = ret/vol if vol>0 else np.nan
        rows.append({'target_return': float(ret), 'vol': float(vol), 'sharpe': float(sharpe), 'weights': w})
    return pd.DataFrame(rows)

def tangency_portfolio(returns: pd.DataFrame, rf: float = 0.0):
    """Max Sharpe portfolio (unconstrained)."""
    mu, Sigma = mean_cov(returns)
    excess = mu - rf
    inv = np.linalg.inv(Sigma)
    w = inv @ excess
    w = w / np.sum(w)  # scale to sum to 1 (same direction)
    return w

def portfolio_stats(weights, returns: pd.DataFrame, rf: float = 0.0):
    mu, Sigma = mean_cov(returns)
    w = np.asarray(weights)
    mean = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    sharpe = (mean - rf)/vol if vol>0 else np.nan
    return dict(mean=mean, vol=vol, sharpe=sharpe)
