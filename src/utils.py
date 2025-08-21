# src/utils.py
import pandas as pd
import numpy as np

def load_price_csvs(file_map):
    """Load multiple price CSVs (Date, Price columns). Returns aligned price DataFrame."""
    dfs = []
    for name, path in file_map.items():
        s = pd.read_csv(path, parse_dates=[0])
        s.columns = ['Date', name]
        s = s.set_index('Date').sort_index()
        dfs.append(s)
    prices = pd.concat(dfs, axis=1).dropna()
    return prices

def prices_to_returns(prices, method='log'):
    if method == 'log':
        rets = np.log(prices/prices.shift(1)).dropna()
    else:
        rets = prices.pct_change().dropna()
    return rets

def synthetic_returns(n_assets=4, n_periods=1000, seed=42):
    rng = np.random.default_rng(seed)
    means = rng.normal(0.0005, 0.0002, size=n_assets)     # ~12-20% annualized drift proxy
    A = rng.normal(size=(n_assets, n_assets))
    cov = (A @ A.T) / 1e4                                 # positive semidefinite small vol
    rets = rng.multivariate_normal(means, cov, size=n_periods)
    cols = [f'A{i+1}' for i in range(n_assets)]
    return pd.DataFrame(rets, columns=cols)
