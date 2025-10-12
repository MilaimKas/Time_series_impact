
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CausalImpactBase:

    def __init__(self, data, pre_period, post_period, standardize_controls):

        if standardize_controls:
            self.data = self._standardize_controls(data, pre_period)
        else:
            self.data = data
        self.standardize_controls = standardize_controls

        self.pre_period = pre_period
        self.post_period = post_period
        self.dates = data.index

        self.pre_data = data.loc[pre_period[0]:pre_period[1]]
        self.post_data = data.loc[post_period[0]:post_period[1]]
        self.intervention_date = post_period[0]

        self.target = self.data.iloc[:, 0]
        self.npre = len(self.pre_data)

        self.model = None
        self.model_results = None
        self.model_performance = None
        self.df_results = None
        self.k = None

        if len(self.pre_data.iloc[:, 0].dropna()) != len(self.pre_data.iloc[:, 1:].dropna()):
            raise ValueError("Different Nan detected in target and controls. Fill missing values first")

    def run(self, model_kwargs={}, predict_kwargs={}, extra_kwargs={}):

        self.fit(model_kwargs=model_kwargs, **extra_kwargs)
        self.predict(**predict_kwargs)
        self.get_inference()

        # add performance metrics

        # Drop first observation from pre_data target and predictions
        pred = self.pred_pre[:]
        actual = self.pre_data.iloc[:, 0].values
        delta = pred-actual

        self.model_performance.update({
            "mae": np.mean(np.abs(delta)),
            "mse": np.mean((delta)**2),
            "rmse": np.sqrt(np.mean((delta)**2))})

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

    def get_inference(self):

        if self.model is None:
            raise ValueError("Model has not been fitted yet, call run()")
        
        actual = self.post_data.iloc[:, 0].to_numpy()

        def rel_ci(ci):
            return ((actual.reshape(-1, 1) - ci) / (self.pred_mean.reshape(-1, 1) + 1e-10))[:, ::-1]

        def abs_ci(ci):
            return (actual.reshape(-1, 1) - ci)[:, ::-1]

        # point estimate
        self.effect = self.post_data.iloc[:, 0].values - self.pred_mean
        self.cum_effect = np.cumsum(self.effect)
        self.rel_effect = self.effect / (self.pred_mean + 1e-10)

        # confidence/credible intervals
        abs_ci_95 = abs_ci(self.pred_ci_95)
        abs_ci_90 = abs_ci(self.pred_ci_90)
        abs_ci_80 = abs_ci(self.pred_ci_80)
        rel_ci_95 = rel_ci(self.pred_ci_95)
        rel_ci_90 = rel_ci(self.pred_ci_90)
        rel_ci_80 = rel_ci(self.pred_ci_80)
        
        self.df_results = {
            "pred_mean": self.pred_mean,
            "pred_ci_95": self.pred_ci_95,
            "pred_ci_90": self.pred_ci_90,
            "pred_ci_80": self.pred_ci_80,

            "abs_effect": self.effect,
            "ci_95_abs_effect": abs_ci_95,
            "ci_90_abs_effect": abs_ci_90,
            "ci_80_abs_effect": abs_ci_80,

            "cum_effect": self.cum_effect,
            "ci_95_cum_abs_effect": abs_ci_95.cumsum(axis=0),
            "ci_90_cum_abs_effect": abs_ci_90.cumsum(axis=0),
            "ci_80_cum_abs_effect": abs_ci_80.cumsum(axis=0),

            "abs_rel_effect": self.rel_effect,
            "ci_95_abs_rel_effect": rel_ci_95,
            "ci_90_abs_rel_effect": rel_ci_90,
            "ci_80_abs_rel_effect": rel_ci_80
        }

        return self.df_results

    def plot(self,counterfactual=None,
             xrange=None,
             yrange=None):

        dates = self.dates

        # build full series for plotting
        full_effect = np.concatenate([np.zeros(self.npre), self.effect])
        full_cum = np.concatenate([np.zeros(self.npre), self.cum_effect])
        full_pred_mean = np.concatenate([np.full(self.npre, None, dtype=float), self.pred_mean])
        full_pred_ci_95_low = np.concatenate([np.full(self.npre, None, dtype=float), self.pred_ci_95[:, 0]])
        full_pred_ci_95_up = np.concatenate([np.full(self.npre, None, dtype=float), self.pred_ci_95[:, 1]])
        full_eff_ci_95_low = np.concatenate([np.full(self.npre, None, dtype=float), self.df_results["ci_95_abs_effect"][:, 0]])
        full_eff_ci_95_up = np.concatenate([np.full(self.npre, None, dtype=float), self.df_results["ci_95_abs_effect"][:, 1]])
        full_cumeff_ci_95_low = np.concatenate([np.full(self.npre, None, dtype=float), self.df_results["ci_95_cum_abs_effect"][:, 0]])
        full_cumeff_ci_95_up = np.concatenate([np.full(self.npre, None, dtype=float), self.df_results["ci_95_cum_abs_effect"][:, 1]])

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(dates, self.target, label="Actuals", color="black")
        axes[0].plot(dates, full_pred_mean, label="Predicted", color="blue")
        axes[0].fill_between(dates, full_pred_ci_95_low, full_pred_ci_95_up, color="blue", alpha=0.2)
        axes[0].axvline(self.intervention_date)
        axes[0].set_ylabel("Observed vs Predicted")
        # if counterfactual is provided, plot it
        if counterfactual is not None:
            axes[0].plot(dates, counterfactual, label="Counterfactual", color="black", linestyle="--")
        axes[0].legend()

        axes[1].plot(dates, full_effect, label="Effect", color="purple")
        axes[1].fill_between(dates, full_eff_ci_95_low, full_eff_ci_95_up, color="blue", alpha=0.2)
        axes[1].axhline(0, color="black", linestyle="--")
        axes[1].axvline(self.intervention_date)
        axes[1].set_ylabel("Pointwise Effect")
        axes[1].legend()

        axes[2].plot(dates, full_cum, label="Cumulative Effect", color="green")
        axes[2].fill_between(dates, full_cumeff_ci_95_low, full_cumeff_ci_95_up, color="blue", alpha=0.2)
        axes[2].axhline(0, color="black", linestyle="--")
        axes[2].axvline(self.intervention_date)
        axes[2].set_ylabel("Cumulative Effect")
        axes[2].set_xlabel("Date")
        axes[2].legend()

        # set x and y ranges if provided
        if  xrange is not None:
            axes[0].set_xlim(xrange)
            axes[1].set_xlim(xrange)
            axes[2].set_xlim(xrange)
        if yrange is not None:
            axes[0].set_ylim(yrange)
            axes[1].set_ylim(yrange)
            axes[2].set_ylim(yrange)

        plt.tight_layout()
        plt.close()
        return fig
    
    def get_model_performance(self):
        return self.model_performance
    
    def get_model_params(self):
        return self.get_model_coeff()

    def _standardize_controls(self, data, pre_period):

        df_pre =  data.loc[pre_period[0]:pre_period[1]]
        
        self.control_means = df_pre.iloc[:, 1:].mean()
        self.control_stds = df_pre.iloc[:, 1:].std(ddof=0).replace(0, 1)

        controls_scaled = (data.iloc[:, 1:] - self.control_means) / self.control_stds
        
        data_scaled = pd.concat([data.iloc[:, 0], controls_scaled], axis=1)
        
        return data_scaled

    def _unstandardize_controls(self):
        return self.controls_scaled * self.control_stds + self.control.means