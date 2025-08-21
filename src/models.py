# src/models.py
import pandas as pd
import numpy as np
import statsmodels.api as sm

def capm(asset_excess: pd.Series, market_excess: pd.Series):
    X = sm.add_constant(market_excess.values)
    y = asset_excess.values
    model = sm.OLS(y, X).fit()
    summary = {
        'alpha': float(model.params[0]),
        'beta' : float(model.params[1]),
        'alpha_t': float(model.tvalues[0]),
        'beta_t' : float(model.tvalues[1]),
        'r2' : float(model.rsquared)
    }
    return summary, model

def fama_french(asset_excess: pd.Series, factors: pd.DataFrame):
    # expects columns: MKT, SMB, HML
    X = sm.add_constant(factors[['MKT','SMB','HML']].values)
    y = asset_excess.values
    model = sm.OLS(y, X).fit()
    coef = dict(zip(['alpha','MKT','SMB','HML'], model.params))
    return {k: float(v) for k,v in coef.items()}, model
