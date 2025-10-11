import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybats.define_models import define_dglm
from pybats.analysis import analysis
from pybats.point_forecast import median
import yaml

from TimeSeries_impact.ts_impact.causal_impact_base import CausalImpactBase

class PyBatsMixin:
    def __init__(self):
        pass

    def fit(self, model_kwargs=None, k=None):
        
        # read config yaml
        with open("TimeSeries_impact/ts_impact/model_config.yaml", 'r') as file:
            model_default = yaml.safe_load(file)

        self.model_kwargs = model_kwargs or model_default["pybats"]

        if k is None:
            k = len(self.post_data)
        self.k = k

        self.model, self.model_result, self.model_coef = analysis(
            Y=self.pre_data.iloc[:, 0].values,
            X=self.data.iloc[:, 1:].values,
            k=self.k,
            forecast_start=self.post_period[0],
            forecast_end=self.post_period[1],
            dates=self.dates, ret=['model', 'forecast', 'model_coef'],
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

class CausalImpactBayes(CausalImpactBase, PyBatsMixin):
    def __init__(self, data, pre_period, post_period, standardize_controls=False):
        CausalImpactBase.__init__(self, data, pre_period, post_period, standardize_controls=standardize_controls)
        PyBatsMixin.__init__(self)



