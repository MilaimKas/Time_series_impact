# causal_impact_ml.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.structural import UnobservedComponents
import warnings
from scipy.sparse import SparseEfficiencyWarning

import yaml

from TimeSeries_impact.ts_impact.causal_impact_base import CausalImpactBase


class MLEMixin:
    def __init__(self):
        pass

    def fit(self, model_kwargs=None):

        # read config yaml
        with open("TimeSeries_impact/ts_impact/model_config.yaml", 'r') as file:
            model_default = yaml.safe_load(file)        
        self.model_kwargs = model_kwargs or model_default["MLE"]

        if self.model_kwargs.get("standartized_controls", False):
            self.data = self._standardize_controls(self.data, self.pre_period)
        
        self.mle_kwargs = {k: v for k, v in self.model_kwargs.items() if k != "standartized_controls"}

        target = self.pre_data.iloc[:, 0]
        exog = self.pre_data.iloc[:, 1:]

        self.model = UnobservedComponents(endog=target, exog=exog, **self.mle_kwargs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning) # silence sparse matrix warnings
            self.model_results = self.model.fit(disp=False)

        if not self.model_results.mle_retvals.get("converged", True):
            print(f"Warning: model did not converge. Data length = {len(self.pre_data)}")

    def predict(self):
        
        if not self.model_results:
            raise ValueError("Model not yet fitted. Run fit() first")
        
        # get point prediction for pre period
        self.pred_pre = self.model_results.get_prediction().predicted_mean.values

        # get point prediction for post period
        self.k = self.post_data.iloc[:, 1:]
        self.predicted_post = self.model_results.get_forecast(steps=len(self.k), exog=self.post_data.iloc[:, 1:])
        self.pred_mean = self.predicted_post.predicted_mean.to_numpy()

        # get confidence intervals
        self.pred_ci_95 = self.predicted_post.conf_int(alpha=0.05).to_numpy()
        self.pred_ci_90 = self.predicted_post.conf_int(alpha=0.10).to_numpy()
        self.pred_ci_80 = self.predicted_post.conf_int(alpha=0.20).to_numpy()

        # model's performance
        self.model_performance = {
            "aic": self.model_results.aic,
            "bic": self.model_results.bic
        }

    def plot_components(self, plot_kwargs={}):

        if not self.model_results:
            raise ValueError("Model not yet fitted. Run fit()")
        
        fig = self.model_results.plot_components(**plot_kwargs)   
        plt.close()
        return fig

    def _standardize_controls(self, data, pre_period):

        df_pre =  data.loc[pre_period[0]:pre_period[1]]
        
        self.control_means = df_pre.iloc[:, 1:].mean()
        self.control_stds = df_pre.iloc[:, 1:].std(ddof=0).replace(0, 1)

        controls_scaled = (data.iloc[:, 1:] - self.control_means) / self.control_stds
        
        data_scaled = pd.concat([data.iloc[:, 0], controls_scaled], axis=1)
        
        return data_scaled

    def _unstandardize_controls(self):
        return self.controls_scaled * self.control_stds + self.control.means
    
    def get_model_coeff(self):
        if not self.model_results:
            raise ValueError("Model not yet fitted. Run fit() first")
        
        coeff = self.model_results.params
        if self.model_kwargs.get("standartized_controls", False):
            # unstandardize control coefficients
            ctr_idx = 0
            for i in range(len(coeff)):
                if "control" in coeff.index[i]:
                    coeff[i] = coeff[i] / self.control_stds[ctr_idx]
                    ctr_idx += 1
        
        return coeff


class CausalImpactMLE(CausalImpactBase, MLEMixin):
    def __init__(self, data, pre_period, post_period):
        CausalImpactBase.__init__(self, data, pre_period, post_period)
        MLEMixin.__init__(self)