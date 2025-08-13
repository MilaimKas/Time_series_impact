# causal_impact_bayes.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybats.define_models import define_dglm
from pybats.analysis import analysis
from pybats.point_forecast import median

class CausalImpactBayes:
    def __init__(self, data: pd.DataFrame, pre_period, post_period, model_kwargs=None):

        self.model_kwargs = model_kwargs or {"seasPeriods": 7, "family":"normal", "prior_length":12,
                                            "ntrend":2,"deltrend":.99, "nsamps":5000, "ntrend":2}

        self.pre_period = pre_period
        self.post_period = post_period

        self.pre_data = data.loc[pre_period[0]:pre_period[1]]
        self.post_data = data.loc[post_period[0]:post_period[1]]
        self.intervention_date = post_period[0]

        self.model = None
        self.results = None
        self.results_samples = None
        self.predicted_post = None
        self.predicted_pre = None

        if len(self.pre_data.iloc[:, 0].dropna()) != len(self.pre_data.iloc[:, 1:].dropna()):
            raise ValueError("Different Nan detected in target and controls. Fill missing values first")

    def fit(self):
        self.model, self.results_samples = analysis(Y = self.pre_data[:,1], X=self.pre_data[:,1:], k = len(self.post_data),
                        forecast_start = self.post_period[0], forecast_end = self.post_period[1],
                        dates=self.pre_data.index, **self.model_kwargs)

    def predict(self):
        self.pred_mean = self.samples.mean(axis=1)
        self.pred_median = self.samples.median(axis=1)
        self.pred_ci_95 = np.percentile(self.samples, [2.5, 97.5], axis=1).T
        self.pred_ci_90 = np.percentile(self.samples, [5, 95], axis=1).T
        self.pred_ci_80 = np.percentile(self.samples, [10, 90], axis=1).T

        self.effect = self.post_data - self.pred_mean
        self.cum_effect = np.cumsum(self.effect)
        self.rel_effect = 100 * self.effect / (self.pred_mean + 1e-10)

    def run(self):
        self.fit()
        self.predict()

    def get_inference(self):
        actual = self.post_data

        def rel_ci(ci):
            return pd.DataFrame({
                'lower target': 100 * (actual - ci[:, 1]) / (self.pred_mean + 1e-10),
                'upper target': 100 * (actual - ci[:, 0]) / (self.pred_mean + 1e-10),
            }, index=self.post_data.index)

        def abs_ci(ci):
            return pd.DataFrame({
                'lower target': actual - ci[:, 1],
                'upper target': actual - ci[:, 0],
            }, index=self.post_data.index)

        ci_95 = abs_ci(self.pred_ci_95)
        ci_90 = abs_ci(self.pred_ci_90)
        ci_80 = abs_ci(self.pred_ci_80)

        return {
            "pred_mean": pd.Series(self.pred_mean, index=self.post_data.index),
            "pred_ci_95": pd.DataFrame(self.pred_ci_95, columns=["lower", "upper"], index=self.post_data.index),
            "pred_ci_90": pd.DataFrame(self.pred_ci_90, columns=["lower", "upper"], index=self.post_data.index),
            "pred_ci_80": pd.DataFrame(self.pred_ci_80, columns=["lower", "upper"], index=self.post_data.index),

            "abs_effect": pd.Series(self.effect, index=self.post_data.index),
            "ci_95_abs_effect": ci_95,
            "ci_90_abs_effect": ci_90,
            "ci_80_abs_effect": ci_80,

            "cum_effect": pd.Series(self.cum_effect, index=self.post_data.index),
            "ci_95_cum_abs_effect": ci_95.cumsum(),
            "ci_90_cum_abs_effect": ci_90.cumsum(),
            "ci_80_cum_abs_effect": ci_80.cumsum(),

            "abs_rel_effect": pd.Series(self.rel_effect, index=self.post_data.index),
            "ci_95_abs_rel_effect": rel_ci(self.pred_ci_95),
            "ci_90_abs_rel_effect": rel_ci(self.pred_ci_90),
            "ci_80_abs_rel_effect": rel_ci(self.pred_ci_80),

            "loglike": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "predicted_pre": None,
            "plot_MLE_components": None
        }

    def plot(self):

        actual = self.data[:,1]
        dates = self.data.index

        full_effect = np.concatenate([np.zeros(self.npre), effect])
        full_cum = np.concatenate([np.zeros(self.npre), cum_effect])

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(dates, actual, label="Actual", color="black")
        axes[0].plot(dates, self.pred_mean, label="Predicted", color="blue")
        axes[0].fill_between(dates, pred_ci_95[:,0], pred_ci_95[:,1], color="blue", alpha=0.2)
        axes[0].axvline(self.intervention_date)
        axes[0].set_ylabel("Observed vs Predicted")
        axes[0].legend()

        axes[1].plot(dates, full_effect, label="Effect", color="purple")
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].axvline(self.intervention_date)
        axes[1].set_ylabel("Pointwise Effect")
        axes[1].legend()

        axes[2].plot(dates, full_cum, label="Cumulative Effect", color="green")
        axes[2].axhline(0, color="black", linestyle="--")
        axes[2].axvline(self.intervention_date)
        axes[2].set_ylabel("Cumulative Effect")
        axes[2].set_xlabel("Date")
        axes[2].legend()

        plt.tight_layout()
        plt.close()
        return fig

if __name__ == "__main__":

    from TimeSeries_impact import synthetic_ts

    data = synthetic_ts.make_time_series(N=200)["data"]

    pre_period = [data.index[0], data.index[-30]]
    post_period = [data.index[-30], data.index[-1]]

    pyb = CausalImpactBayes(data, pre_period=pre_period, post_period=post_period)
    pyb.fit()

