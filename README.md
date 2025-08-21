# Portfolio Management — Advanced Portfolio Construction

This repository implements core portfolio management components aligned with your project:

- Diversification & Covariance Analysis
- Minimum-Variance Portfolio (closed-form)
- Efficient Frontier (unconstrained) & Tangency Portfolio with a risk-free asset
- Risk Analysis: Historical & Parametric VaR, Monte Carlo simulation (vectorized)
- Asset Pricing: CAPM & Fama–French Three-Factor regressions

## Quickstart

```bash
pip install -r requirements.txt
python run_demo.py
```

`run_demo.py` generates synthetic asset returns (so you can run immediately) and
demonstrates:
- MVP weights
- Efficient frontier sampling
- Tangency portfolio weights (max Sharpe)
- Monte Carlo simulation of portfolio PnL
- VaR at 95%/99%
- CAPM regression (on one asset vs synthetic market)

To use **real data**, place CSVs under `data/` and load with the helpers in `src/utils.py`.

## Project Structure
```
src/
  portfolio.py       # optimization & analytics
  risk_analysis.py   # VaR, Monte Carlo, Sharpe
  models.py          # CAPM & Fama-French
  utils.py           # load returns, synthetic data
run_demo.py
data/                # put your CSVs here
```

## Notes
- All optimization is **unconstrained fully-invested** (weights sum to 1; shorting allowed). 
  For no-short or weight bounds, swap in a QP solver (cvxpy) if needed.
- Fama–French requires a factors CSV with columns: `MKT`, `SMB`, `HML`, `RF`.
