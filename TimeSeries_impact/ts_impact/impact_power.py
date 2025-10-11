# sim_impact.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TimeSeries_impact.ts_impact.causal_impact_mle import CausalImpactMLE
from TimeSeries_impact.utilities import add_effect, seasonal_default
from TimeSeries_impact.ts_analysis.ts_decomposition import Decomposer
import yaml


class SimImpact:
    def __init__(self, df, backend="MLE"):

        self.target = df.iloc[:, 0]
        self.controls = df.iloc[:, 1:]
        self.target.dropna(inplace=True)
        self.controls.dropna(inplace=True)
        self.raw_data = pd.concat([self.target, self.controls], axis=1)
        self.index = self.raw_data.index

        if len(self.target) != len(self.controls):
            raise ValueError("Length of the target and control time series differ after removing nan values. Did you forget to fill in missing values ?")

        self.res_sim = []
        self.res_power = {}
        self.test_size = None

        # read config yaml
        with open("TimeSeries_impact/ts_impact/model_config.yaml", 'r') as file:
            model_config = yaml.safe_load(file)

        if backend not in model_config.keys():
            raise ValueError(f"Backend {backend} not recognized, available models are {list(model_config.keys())}")
        self.model_para_default = model_config[backend]

    def _on_trend(self, data, split_decomposition, pre_period, post_period, 
                  tsize, up, decompose_kwargs, add_effect_args):

        if split_decomposition:
            pre = data.loc[pre_period[0]:pre_period[1]]
            post = data.loc[post_period[0]:post_period[1]]
            pre_dec = Decomposer(pre)
            post_dec = Decomposer(post)
            pre_dec.decompose(**(decompose_kwargs or {}))
            post_dec.decompose(**(decompose_kwargs or {}))
            pre_trend = pre_dec.components["trend"]
            post_trend = post_dec.components["trend"]
            data = pd.concat([pre_trend, post_trend])

        else:
            full_dec = Decomposer(data)
            full_dec.decompose(**(decompose_kwargs or {}))
            data = full_dec.components["trend"]

        # store true counterfactual
        data_target_true = data.iloc[:,0]

        # add effect on the post period trend
        obs = add_effect(data.iloc[:,0], up, tsize, **add_effect_args)
        data = pd.concat([obs, data.iloc[:,1:]], axis=1)

        return obs, data, data_target_true
    
    def make_sim(self, relup_list, test_size,
                 model_kwargs=None,
                 add_effect_args=None,
                 on_trend=False,
                 decompose_kwargs=None,
                 split_decomposition=True):

        model_kwargs = model_kwargs or self.model_para_default 
        add_effect_args = add_effect_args or {"scale_std": 1., "log_len": test_size//10}
        decompose_kwargs = decompose_kwargs or {"period": 7, "seasonal": None, "trend": None}

        if decompose_kwargs["seasonal"] is None:
            decompose_kwargs["seasonal"] = seasonal_default(len(self.target))

        self.test_size = test_size
        uplift_list = np.array([np.mean(self.target[-test_size:]) * ru for ru in relup_list])
        self.res_sim = []

        for up, rel in zip(uplift_list, relup_list):

            pre_period = [self.index[0], self.index[-1 - test_size]]
            post_period = [self.index[-test_size], self.index[-1]]

            # simulate on the trend after decomposition
            if on_trend:
                
                data = self.raw_data.copy()
                obs, data, data_target_true = self._on_trend(data, split_decomposition, pre_period, post_period,\
                            test_size, up, decompose_kwargs, add_effect_args)
                model_kwargs["nseasons"] = None # remove  seasonality in the model
            
            else:

                # store true counterfactual
                data_target_true = self.target.copy()
                # add effect
                obs = add_effect(self.target, up, test_size, **add_effect_args)
                data = pd.concat([obs, self.controls], axis=1)
            
            # check if index are datetime
            try:
                data.index = pd.to_datetime(self.index)
            except:
                print("Index could not be converted to date, continue with original index")
                data.index = self.index

            ci = CausalImpactMLE(data, pre_period, post_period)
            ci.run(model_kwargs=model_kwargs)

            inference = ci.get_inference()

            # store infos and results
            data["true_target"] = data_target_true
            res_tmp = {"chart":ci.plot(counterfactual=data_target_true), "true_relup":rel, "test_size":test_size, "data":data}
            res_tmp.update(inference)
            self.res_sim.append(res_tmp)
    
    def plot_sim_rel(self):

        if not self.res_sim:
            raise ValueError("The simulation has not been performed yet, please call the make_sim() function first")

        true_relup = np.array([sim["true_relup"] for sim in self.res_sim])*100
        sim_data = self.res_sim

        # make average relarive effect
        relup_obs = np.array([np.mean(v["abs_rel_effect"]) for v in sim_data])*100
        lower_95 = np.array([np.mean(v["ci_95_abs_rel_effect"][:,0]) for v in sim_data])*100
        upper_95 = np.array([np.mean(v["ci_95_abs_rel_effect"][:,1]) for v in sim_data])*100
        lower_90 = np.array([np.mean(v["ci_90_abs_rel_effect"][:,0]) for v in sim_data])*100
        upper_90 = np.array([np.mean(v["ci_90_abs_rel_effect"][:,1]) for v in sim_data])*100
        lower_80 = np.array([np.mean(v["ci_80_abs_rel_effect"][:,0]) for v in sim_data])*100
        upper_80 = np.array([np.mean(v["ci_80_abs_rel_effect"][:,1]) for v in sim_data])*100

        fig = plt.figure()
        plt.plot(true_relup, relup_obs, 'o-', label="Estimated rel. effect")
        plt.plot(true_relup, true_relup, 'o--', label="True rel. effect")
        plt.fill_between(true_relup, lower_95, upper_95, color="blue", alpha=0.2, label="95% CI")
        plt.fill_between(true_relup, lower_90, upper_90, color="green", alpha=0.2, label="90% CI")
        plt.fill_between(true_relup, lower_80, upper_80, color="orange", alpha=0.2, label="80% CI")

        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Relative uplift (true)")
        plt.ylabel("Estimated relative effect")
        plt.xticks(ticks=true_relup, labels=[f"{int(r)}%" for r in true_relup])
        plt.legend()
        plt.close()

        return fig

    def plot_sim_cum(self, kpi="KPI"):

        if not self.res_sim:
            raise ValueError("The simulation has not been performed yet, please call the make_sim() function first")

        sim_data = self.res_sim
        relup = np.array([sim["true_relup"] for sim in sim_data])
        cum_effect = [v["cum_effect"] for v in sim_data]
        cum_upper = [v["ci_95_cum_abs_effect"][:, 1] for v in sim_data]
        cum_lower = [v["ci_95_cum_abs_effect"][:, 0] for v in sim_data]

        fig = plt.figure()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        test_size = sim_data[0]["test_size"]
        x = np.arange(test_size)

        for cu, up, lo, label, col in zip(cum_effect, cum_upper, cum_lower, relup, colors):
            plt.plot(x, up[-test_size:], label=f"{label*100:.0f}%", color=col)
            plt.plot(x, lo[-test_size:], label="_nolabel_", linestyle="--", color=col)

        plt.axhline(0, color="black")
        plt.title("Cumulative Effect Bounds (Solid = Upper, Dashed = Lower)")
        plt.ylabel(f"Cumulative effect on total {kpi}")
        plt.xlabel("Days after intervention")
        plt.legend(title="Uplift")
        plt.close()

        return fig

    def power_analyse(self, relup_list=None, test_sizes=None,
                      model_kwargs=None, add_effect_args=None,
                      on_trend=False, decompose_kwargs=None, split_decomposition=False,
                      n_rollouts=10, min_pre=30):

        model_kwargs = model_kwargs or self.model_para_default

        relup_list = relup_list or np.linspace(0.01, 0.2, 10)
        test_sizes = test_sizes or np.arange(20, 40, 5)
        add_effect_args = add_effect_args or {}
        decompose_kwargs = decompose_kwargs or {}

        data_length = len(self.target)
        power_matrix_p05 = np.zeros((len(relup_list), len(test_sizes)))
        power_matrix_p20 = np.zeros((len(relup_list), len(test_sizes)))

        for i, rel in enumerate(relup_list):
            for j, tsize in enumerate(test_sizes):
                power_count_p05 = 0
                power_count_p20 = 0

                # Estimate pre-period start points evenly across valid range
                max_pre = data_length - tsize - 1
                starts = np.linspace(min_pre, max_pre, n_rollouts, dtype=int)

                # loop over intervention start
                for start in starts:

                    idx_start, idx_end = start, start + tsize
                    if idx_end > data_length:
                        continue

                    sub_target = self.target.iloc[:idx_end].copy()
                    sub_controls = self.controls.iloc[:idx_end,:].copy()

                    up = rel * np.mean(sub_target[-tsize:])

                    pre_period = [self.index[0], self.index[-1 - tsize]]
                    post_period = [self.index[-tsize], self.index[-1]]

                    if on_trend:

                        data = self.raw_data.copy()
                        obs, data, data_target_true = self._on_trend(data, split_decomposition, pre_period, post_period,\
                                    tsize, up, decompose_kwargs, add_effect_args)
                        model_kwargs["nseasons"] = None # remove  seasonality in the model

                    else:

                        # store true counterfactual
                        data_target_true = self.target.copy()
                        # add effect
                        obs = add_effect(self.target, up, tsize, **add_effect_args)
                        data = pd.concat([obs, self.controls], axis=1)

                    ci = CausalImpactMLE(data, pre_period, post_period)
                    ci.run(model_kwargs=model_kwargs)
                    inf = ci.get_inference()
                    
                    ci_bounds = inf["ci_95_abs_effect"]
                    ci_l = ci_bounds[-1, 0]
                    ci_u = ci_bounds[-1, 1]
                    if (ci_l > 0 or ci_u < 0):
                        power_count_p05 += 1
                    ci_bounds = inf["ci_80_abs_effect"]
                    ci_l = ci_bounds[-1, 0]
                    ci_u = ci_bounds[-1, 1]
                    if (ci_l > 0 or ci_u < 0):
                        power_count_p20 += 1
                    est = ci.get_inference()["abs_effect"][-1]

                power_matrix_p05[i, j] = power_count_p05 / max(1, len(starts))
                power_matrix_p20[i, j] = power_count_p20 / max(1, len(starts))

        self.res_power["power_matrix_p05"] = power_matrix_p05
        self.res_power["power_matrix_p20"] = power_matrix_p20
        self.res_power["relup_list"] = relup_list
        self.res_power["test_sizes"] = test_sizes

        return self.res_power

    def plot_power(self, power_level=[60,80,95], alpha=5, imshow_kwargs={}):

        if not self.res_power:
            raise ValueError("First call the power_analyse method.")
        
        if alpha == 5:
            power_mat = self.res_power["power_matrix_p05"]
        elif alpha == 20:
            power_mat = self.res_power["power_matrix_p20"]
        else:
            raise ValueError(f"Only alpha 5 and 20 are allowed, got {alpha}")

        fig, ax = plt.subplots()
        im = ax.imshow(power_mat*100, cmap="Greens", aspect="auto", origin="lower", interpolation="spline16", **imshow_kwargs)
        interp_data = im.get_array()

        CS = ax.contour(interp_data, levels=power_level, colors="black")

        ax.set_xticks(np.arange(len(self.res_power["test_sizes"])))
        ax.set_xticklabels(self.res_power["test_sizes"])
        ax.set_yticks(np.arange(len(self.res_power["relup_list"])))
        ax.set_yticklabels([f"{int(r*100)}%" for r in self.res_power["relup_list"]])
        ax.set_xlabel("Test Period Length")
        ax.set_ylabel("Relative Uplift")
        ax.set_title("Power (Detection Probability) for p-value=5%")

        # add countour labels
        ax.clabel(CS, power_level, inline=1, fontsize=10)

        plt.colorbar(im, ax=ax, label="Power")
        plt.close()

        return fig
