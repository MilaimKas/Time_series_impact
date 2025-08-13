# causal_impact_ml.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents

class CausalImpactMLE:
    def __init__(self, data: pd.DataFrame, pre_period, post_period, model_kwargs=None):

        self.model_kwargs = model_kwargs or {"level": "local linear trend", "seasonal": 7, "standartized_controls":True}

        if self.model_kwargs.get("standartized_controls", False):
            self.data = self._standardize_controls(data, pre_period)
        else:
            self.data = data.copy()

        self.pre_period = pre_period
        self.post_period = post_period

        self.pre_data = data.loc[pre_period[0]:pre_period[1]]
        self.post_data = data.loc[post_period[0]:post_period[1]]
        self.intervention_date = post_period[0]

        self.model = None
        self.results = None
        self.predicted_post = None
        self.predicted_pre = None

        if len(self.pre_data.iloc[:, 0].dropna()) != len(self.pre_data.iloc[:, 1:].dropna()):
            raise ValueError("Different Nan detected in target and controls. Fill missing values first")

    def fit(self):
        target = self.pre_data.iloc[:, 0]
        exog = self.pre_data.iloc[:, 1:]

        self.model = UnobservedComponents(endog=target, exog=exog, **self.model_kwargs)
        self.results = self.model.fit(disp=False)

        if not self.results.mle_retvals.get("converged", True):
            print(f"⚠️ Warning: model did not converge. Data length = {len(self.pre_data)}")

        # Store model fit for pre-period as in-sample (pseudo forecast) prediction.
        self.predicted_pre = self.results.get_prediction().summary_frame(alpha=0.05)

    def predict(self):
        exog_post = self.post_data.iloc[:, 1:]
        self.predicted_post = self.results.get_forecast(steps=len(exog_post), exog=exog_post)
        self.pred_mean = self.predicted_post.predicted_mean

        self.pred_ci_95 = self.predicted_post.conf_int(alpha=0.05)
        self.pred_ci_90 = self.predicted_post.conf_int(alpha=0.10)
        self.pred_ci_80 = self.predicted_post.conf_int(alpha=0.20)

        actual = self.post_data.iloc[:, 0]
        self.effect = actual - self.pred_mean
        self.cum_effect = self.effect.cumsum()

        self.rel_effect = 100 * self.effect / (self.pred_mean + 1e-10)

    def summary(self):

        actual = self.post_data.iloc[:, 0]
        mean_actual = actual.mean()
        mean_pred = self.pred_mean.mean()

        rel_effect = 100 * (mean_actual - mean_pred) / (mean_pred + 1e-10)
        abs_effect = mean_actual - mean_pred

        ci_lower = self.pred_ci_95.iloc[:, 0].mean()
        ci_upper = self.pred_ci_95.iloc[:, 1].mean()
        rel_ci_lower = 100 * (mean_actual - ci_lower) / (mean_pred + 1e-10)
        rel_ci_upper = 100 * (mean_actual - ci_upper) / (mean_pred + 1e-10)

        print("Causal Impact Summary:")
        print(f"  Absolute effect: {abs_effect:.3f}")
        print(f"  Relative effect: {rel_effect:.2f}%")
        print(f"  95% CI: [{rel_ci_lower:.2f}%, {rel_ci_upper:.2f}%]")

    def run(self):
        self.fit()
        self.predict()

    def get_inference(self):
        actual = self.post_data.iloc[:, 0]
        ci_95 = pd.DataFrame({'lower target': actual - self.pred_ci_95.iloc[:, 1],'upper target': actual - self.pred_ci_95.iloc[:, 0]})
        ci_90 = pd.DataFrame({'lower target': actual - self.pred_ci_90.iloc[:, 1],'upper target': actual - self.pred_ci_90.iloc[:, 0]})
        ci_80 = pd.DataFrame({'lower target': actual - self.pred_ci_80.iloc[:, 1],'upper target': actual - self.pred_ci_80.iloc[:, 0]})

        try:
            aic = self.results.aic
            bic = self.results.bic
        except Exception:
            aic = np.nan
            bic = np.nan
        
        return {
            # actuals
            "pred_mean": self.pred_mean,
            "pred_ci_95": self.pred_ci_95,
            "pred_ci_90": self.pred_ci_90,
            "pred_ci_80": self.pred_ci_80,
            # absolute effect
            "abs_effect":self.effect,
            "ci_95_abs_effect": ci_95,
            "ci_90_abs_effect": ci_90,
            "ci_80_abs_effect": ci_80,            
            # cummulative effect
            "cum_effect": self.cum_effect,
            "ci_95_cum_abs_effect": ci_95.cumsum(),
            "ci_90_cum_abs_effect": ci_90.cumsum(),
            "ci_80_cum_abs_effect": ci_80.cumsum(),  
            # absolute relative effect
            "abs_rel_effect": self.rel_effect,
            "ci_95_abs_rel_effect": ci_95.div(self.pred_mean+ 1e-10, axis=0),
            "ci_90_abs_rel_effect": ci_90.div(self.pred_mean+ 1e-10, axis=0),
            "ci_80_abs_rel_effect": ci_80.div(self.pred_mean+ 1e-10, axis=0),  
            # model's performance
            "loglike": self.results.llf,
            "aic": aic,
            "bic": bic,
            "predicted_pre": self.predicted_pre,
            "plot_MLE_components": self.plot_components()
        }

    def plot(self, drop_ini=10):
        actual_pre = self.pre_data.iloc[:, 0]
        actual_post = self.post_data.iloc[:, 0]
        full_actual = pd.concat([actual_pre, actual_post])[drop_ini:]
        dates = full_actual.index

        pred_pre = self.predicted_pre["mean"]
        ci_pre = self.predicted_pre[["mean_ci_lower", "mean_ci_upper"]]

        pred_post = self.pred_mean
        ci_post = self.pred_ci_95

        full_pred = pd.concat([pred_pre, pred_post])[drop_ini:]
        full_ci_lower = pd.concat([ci_pre.iloc[:, 0], ci_post.iloc[:, 0]])[drop_ini:]
        full_ci_upper = pd.concat([ci_pre.iloc[:, 1], ci_post.iloc[:, 1]])[drop_ini:]

        effect = full_actual - full_pred
        cum_effect = pd.concat([pd.Series(0, index=pred_pre.index), self.cum_effect])[drop_ini:]
        cum_ci_low = pd.concat([pd.Series(0, index=pred_pre.index), (actual_post - ci_post.iloc[:, 0])]).cumsum()[drop_ini:]
        cum_ci_up = pd.concat([pd.Series(0, index=pred_pre.index), (actual_post - ci_post.iloc[:, 1])]).cumsum()[drop_ini:]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(dates, full_actual, label="Actual", color="black")
        axes[0].plot(dates, full_pred, label="Predicted", color="blue")
        axes[0].fill_between(dates, full_ci_lower, full_ci_upper, color="blue", alpha=0.2)
        axes[0].axvline(self.intervention_date)
        axes[0].set_ylabel("Observed vs Predicted")
        axes[0].legend()

        axes[1].plot(dates, effect, label="Effect", color="purple")
        axes[1].fill_between(dates, full_actual - full_ci_lower, full_actual - full_ci_upper, color="purple", alpha=0.2)
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].axvline(self.intervention_date)
        axes[1].set_ylabel("Pointwise Effect")
        axes[1].legend()

        axes[2].plot(dates, cum_effect, label="Cumulative Effect", color="purple")
        axes[2].fill_between(dates, cum_ci_low, cum_ci_up, color="purple", alpha=0.2)
        axes[2].axhline(0, color="black", linestyle="--")
        axes[2].axvline(self.intervention_date)
        axes[2].set_ylabel("Cumulative Effect")
        axes[2].set_xlabel("Date")
        axes[2].legend()

        plt.tight_layout()
        plt.close()
        return fig

    def plot_components(self, plot_kwargs={}):
        if not self.results:
            raise ValueError("Model not yet fitted. Run fit first")
        
        fig = self.results.plot_components(**plot_kwargs)   
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
