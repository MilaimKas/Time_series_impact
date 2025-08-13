import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from TimeSeries_impact.utilities import seasonal_default

class Decomposer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = df.columns
        self.components = {}

    def decompose(self, seasonal=None, period=7, trend=None):
        """
        decompose the given time series into trend, seasonal and noise components

        Args:
            seasonal (_type_, optional): _description_. Defaults to None.
            period (int, optional): _description_. Defaults to 7.
            trend (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if seasonal is None:
            seasonal = seasonal_default(len(self.df))

        trend = trend or max(15, period * 2 + 1)
        seasonal = seasonal if seasonal % 2 == 1 else seasonal + 1  # Ensure odd
        trend = trend if trend % 2 == 1 else trend + 1

        trends = pd.DataFrame(index=self.df.index)
        seasonals = pd.DataFrame(index=self.df.index)
        residuals = pd.DataFrame(index=self.df.index)

        for col in self.columns:
            stl = STL(self.df[col], seasonal=seasonal, trend=trend, period=period).fit()
            # check for significant detected period
            if not self.check_seasonal_significance(self.df[col], period=period):
                print(f"Warning: period {period} was not detected with 80% confidence for {col}")
            trends[col] = stl.trend
            seasonals[col] = stl.seasonal
            residuals[col] = stl.resid

        self.components["trend"] = trends
        self.components["seas"] = seasonals
        self.components["resid"] = residuals

        return self.components

    def check_seasonal_significance(self, serie: pd.Series, period: int, alpha: float = 0.2):
        n = len(serie)
        acorr, confint, qstat, pval = acf(serie, qstat=True, alpha=alpha)
        significant_lags = np.where(acorr > np.abs(confint[:,0]))[0]
        if period not in significant_lags:
            flags = False
        else:
            flags = True
        return flags
