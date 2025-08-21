# run_demo.py
import numpy as np
import pandas as pd

from src.utils import synthetic_returns
from src.portfolio import minimum_variance_portfolio, efficient_frontier, tangency_portfolio, portfolio_stats
from src.risk_analysis import historical_var, parametric_var, monte_carlo_portfolio, sharpe_ratio
from src.models import capm

def main():
    # generate synthetic asset returns (~daily)
    rets = synthetic_returns(n_assets=4, n_periods=1200, seed=7)
    mu = rets.mean().values
    Sigma = rets.cov().values

    print('Assets:', rets.columns.tolist())
    print('Sample mean (daily):', np.round(mu, 6))
    print('Sample vol (daily): ', np.round(np.sqrt(np.diag(Sigma)), 6))

    # MVP
    w_mvp = minimum_variance_portfolio(rets)
    stats_mvp = portfolio_stats(w_mvp, rets)
    print('\n--- Minimum Variance Portfolio ---')
    print('weights:', np.round(w_mvp, 4))
    print('mean(daily)=%.6f vol=%.6f sharpe=%.3f' % (stats_mvp['mean'], stats_mvp['vol'], stats_mvp['sharpe']))

    # Tangency (assume rf=0 for daily scale)
    w_tan = tangency_portfolio(rets, rf=0.0)
    stats_tan = portfolio_stats(w_tan, rets, rf=0.0)
    print('\n--- Tangency (Max Sharpe) Portfolio ---')
    print('weights:', np.round(w_tan, 4))
    print('mean(daily)=%.6f vol=%.6f sharpe=%.3f' % (stats_tan['mean'], stats_tan['vol'], stats_tan['sharpe']))

    # Efficient Frontier sample
    ef = efficient_frontier(rets, n_points=10)
    print('\n--- Efficient Frontier (first 3 rows) ---')
    print(ef[['target_return','vol','sharpe']].head(3))

    # Risk analysis for tangency portfolio
    port_daily = rets.values @ w_tan
    var95_hist = historical_var(pd.Series(port_daily), alpha=0.05)
    var99_hist = historical_var(pd.Series(port_daily), alpha=0.01)
    var95_para = parametric_var(port_daily.mean(), port_daily.std(ddof=1), alpha=0.05)
    print('\n--- VaR (daily) for Tangency Portfolio ---')
    print('Historical VaR 95%%:', round(var95_hist, 6), '  99%%:', round(var99_hist,6))
    print('Parametric VaR 95%%:', round(var95_para, 6))

    # Monte Carlo 1-year (252 days) horizon returns
    sim_pnl = monte_carlo_portfolio(mu, Sigma, w_tan, n_steps=252, n_paths=20000, seed=11)
    print('\n--- Monte Carlo (1y horizon) ---')
    print('mean=%.5f std=%.5f VaR95=%.5f' % (sim_pnl.mean(), sim_pnl.std(ddof=1), np.percentile(sim_pnl, 5)))

    # CAPM on Asset 1 vs synthetic market = equal-weight index
    rf = 0.0
    market = rets.mean(axis=1)  # simple proxy
    summary, _ = capm(rets['A1']-rf, market-rf)
    print('\n--- CAPM (A1 vs synthetic market) ---')
    print(summary)

if __name__ == '__main__':
    main()
