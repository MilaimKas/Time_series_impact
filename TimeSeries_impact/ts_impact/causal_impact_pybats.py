
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pybats.define_models import define_dglm
from pybats.analysis import analysis
from pybats.point_forecast import median
from pybats.latent_factor import Y_lf, seas_weekly_lf, multi_latent_factor

import yaml

from TimeSeries_impact.ts_impact.causal_impact_base import CausalImpactBase

class PyBatsMixin:
    def __init__(self):
        pass

    def fit(self, model_kwargs=None, k=None, latent_factor_df=None):
        
        # read config yaml
        with open("TimeSeries_impact/ts_impact/model_config.yaml", 'r') as file:
            model_default = yaml.safe_load(file)

        if k is None:
            k = len(self.post_data)
        self.k = k

        self.model_kwargs = model_kwargs or model_default["pybats"]

        # check familiy - data type compatibilty
        self.check_family_compatibility(self.data.iloc[:,0].values, self.model_kwargs["family"])

        self.latent_factor_names = None
        self.latent_factors = None
        self.multi_lf = None

        # calculate latent factors
        if latent_factor_df is not None:
            if not isinstance(latent_factor_df, pd.DataFrame) or not latent_factor_df.index.equals(self.data.index) :
                raise ValueError("The latent_factor argument must be a DataFrame with the same indices as the data, each column being one latent factor.")
            print("Including Y and weekly seasonal latent factors")
            if len(latent_factor_df.columns) > 1:
                X = latent_factor_df.iloc[:, 1:].values
            else:
                X=None
            self.latent_factors = analysis(Y=latent_factor_df.iloc[:, 0].values,
                        X=X,
                        k=self.k,
                        forecast_start=self.post_period[0],
                        forecast_end=self.post_period[1],
                        dates=self.dates, 
                        ret=['new_latent_factors'], 
                        new_latent_factors= [Y_lf, seas_weekly_lf],
                        **self.model_kwargs)
            self.latent_factor_names = ['Target_lf', 'weekly_seasonal_lf']
            self.multi_lf = multi_latent_factor(self.latent_factors[:2])

        self.model, self.model_result, self.model_coef = analysis(
            Y=self.pre_data.iloc[:, 0].values,
            X=self.data.iloc[:, 1:].values,
            k=self.k,
            forecast_start=self.post_period[0],
            forecast_end=self.post_period[1],
            dates=self.dates, 
            ret=['model', 'forecast', 'model_coef'],
            latent_factor = self.multi_lf,
            **self.model_kwargs
        )

    def predict(self):

        # get point estimate of pre period -> not possible with PyBats
        self.pred_pre = np.zeros(self.npre)  #self.model_result[:, :self.npre, 0].mean(axis=0)
        
        # get point predictions
        self.pred_mean = np.mean(self.model_result, axis=0)[0, :]
        self.pred_median = np.median(self.model_result, axis=0)[0, :]
        
        # get credible intervals
        self.pred_ci_95 = np.percentile(self.model_result[:, 0, :], [2.5, 97.5], axis=0).T
        self.pred_ci_90 = np.percentile(self.model_result[:, 0, :], [5, 95], axis=0).T
        self.pred_ci_80 = np.percentile(self.model_result[:, 0, :], [10, 90], axis=0).T

        # model's performance
        self.model_performance = {}
    
    def plot_components(self, plot_kwargs={}):
        pass

    def get_model_coeff(self):
        """
        PyBats returns a dictionary with the following keys
        "m" → Posterior Mean of the Regression Coefficients
            A vector of shape (p,) where p is the number of predictors.
            interpret m as the "expected" regression coefficient at that time step.

        "C" → Posterior Covariance Matrix of the Coefficients
            A matrix of shape (p, p) for each time step.
            Represents the uncertainty (variance and correlation) in the β estimates.
            The diagonal elements are the variances of each coefficient.

        "n" → Posterior Degrees of Freedom
            Scalar. Relates to the uncertainty in the residual variance (σ²).
            Comes from the inverse gamma prior on the variance.

        "s" → Posterior Scale Parameter for the Variance
            Scalar. Posterior estimate of the residual sum of squares, used with n to define the posterior over σ².
        """

        # take only point estimate for the coefficients. coeff is an array of shape (n_timesteps, p)
        coeff = self.model_coef["m"]

        # make a dictionary with keys with the model's parameter names
        
        # get indices for the parameters
        trend_idx = self.model.itrend
        seas_idx = self.model.iseas
        reg_idx = self.model.iregn

        return {"trend": coeff[:, trend_idx],
                "seasonal": coeff[:, seas_idx],
                "regression": coeff[:, reg_idx]}

    def check_family_compatibility(self, y, family):
        """
        Check if the target variable `y` is compatible with the specified family.
        
        Parameters:
            y (array-like): target variable (1D array, pandas Series, etc.)
            family (str): one of ["normal", "bernoulli", "poisson"]
            
        Raises:
            ValueError if data is incompatible with the specified family.
        """
        if isinstance(y, pd.Series):
            y = y.values
        y = np.asarray(y)

        if family == "normal":
            # No restriction: normal can be continuous with any real support
            if not np.issubdtype(y.dtype, np.number):
                raise ValueError("Normal family requires numeric values.")
        
        elif family == "bernoulli":
            # Must be binary (0/1)
            unique_vals = np.unique(y[~np.isnan(y)])  # ignore NaNs
            if not np.all(np.isin(unique_vals, [0, 1])):
                raise ValueError("Bernoulli family requires only 0 and 1 values.")
        
        elif family == "poisson":
            # Must be non-negative integers
            if not np.all(np.isfinite(y)):
                raise ValueError("Poisson family requires finite numeric values.")
            if not np.all((y >= 0) & (y == np.floor(y))):
                raise ValueError("Poisson family requires non-negative integer values.")
        
        else:
            raise ValueError(f"Unsupported family: '{family}'")

        return True  # passed check

class CausalImpactBayes(CausalImpactBase, PyBatsMixin):
    def __init__(self, data, pre_period, post_period, standardize_controls=False):
        CausalImpactBase.__init__(self, data, pre_period, post_period, standardize_controls=standardize_controls)
        PyBatsMixin.__init__(self)



