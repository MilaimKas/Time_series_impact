import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybats.define_models import define_dglm
from pybats.analysis import analysis
from pybats.point_forecast import median

from TimeSeries_impact.ts_impact.causal_impact_base import CausalImpactBase

class PyBatsMixin:
    def __init__(self):
        pass

    def fit(self, model_kwargs=None, k=None):

        self.model_kwargs = model_kwargs or {
            "seasPeriods": [7], "family": "normal", "prior_length": 12,
            "ntrend": 2, "deltrend": 0.99, "nsamps": 5000
        }

        if k is None:
            k = len(self.post_data)
        self.k = k

        self.model, self.model_result = analysis(
            Y=self.pre_data.iloc[:, 0].values,
            X=self.data.iloc[:, 1:].values,
            k=self.k,
            forecast_start=self.post_period[0],
            forecast_end=self.post_period[1],
            dates=self.dates,
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

class CausalImpactBayes(CausalImpactBase, PyBatsMixin):
    def __init__(self, data, pre_period, post_period):
        CausalImpactBase.__init__(self, data, pre_period, post_period)
        PyBatsMixin.__init__(self)



