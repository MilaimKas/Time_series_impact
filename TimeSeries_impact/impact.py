import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from TimeSeries_impact.dep_causalimpact.analysis import CausalImpact as dep_CausalImpact
import TimeSeries_impact.utilities as utilities

from causalimpact import CausalImpact

from tqdm import trange


class SimImpact:

    """
        Class for performing some simulation (aka power analysis) on time series.
        This calls the deprecated version of causal impact because it uses maximum likelihood estimation instead of pure posterior Bayesian. 
        Hence the calculation is faster, which is needed for the 
    """

    def __init__(self, df):
        """

        Args:
            target (pd.Series): pandas serie/dataframe for the target with date index
            controls (pd.Series): pandas serie/dataframe for the controls with date index
        """

        self.target = df.iloc[:,0]
        self.controls = df.iloc[:,1:]

        # remove nan
        self.target.dropna(inplace=True)
        self.controls.dropna(inplace=True)

        # initialize list of results
        self.res_sim = {}
        self.res_power = {}

        self.test_size = None
        self.relup = None

    def make_sim(self, relup_list, test_size, 
                        model_args={"nseasons":7}, 
                        add_effect_args={"scale_std":0.05, "log_len":0},
                        on_trend=False, decompose_kwars={"period":7, "seasonal":None, "trend":None}):
        """
        This function performs a simulation following these steps:
            - choose a test size (post perdiod size)
            - take the target value for the post period and add an effect with some noise
            - perform a causal impact analysis for each effect size    

        Args:
            relup_list (np.array): array with the relative uplift in fraction.
            test_size (int): size of the post period in number of time unit (e.g. days). 
            model_args (dict, optional): arguments for the causal impact model. Defaults to {"nseasons":7}.
            add_effect_args (dict, optional): arguments for the simulated effect. Defaults to {"scale_std":0.01, "log_len":2}.

        Returns:
            plt.figure: main plot with the results
        """

        # default value for the seasonal smoothing
        if decompose_kwars["seasonal"] is None:
            decompose_kwars["seasonal"] = utilities.seasonal_default(len(self.target))

        # store given test size in class value
        self.test_size = test_size

        # calculate absolute uplift from the target mean value in the post period
        uplift_list = np.array([np.mean(self.target[-test_size:])*ru for ru in relup_list])

        # initialize list of results
        if self.res_sim:
            print("Warning: result of simulation overwritten")
        self.res_sim = {}

        # loop over uplifts
        for up, rel in zip(uplift_list, relup_list):
            
            # add uplift with randomness to target data
            obs = utilities.add_effect(self.target, up, test_size, **add_effect_args)
            data = pd.concat([obs, self.controls], axis=1) # merge obs and controls
            data.index = pd.to_datetime(data.index)
            
            # decompose TS
            if on_trend:
                data, _, _ = utilities.decompose_components(data, **decompose_kwars)
                model_args["nseasons"] = None # remove seasonal components from model

            # Causal impact analysis
            pre_period = [data.index[0], data.index[-1-test_size]]
            post_period = [data.index[-test_size], data.index[-1]]
            ci = dep_CausalImpact(data, pre_period, post_period, model_args=model_args)
            ci.run()
            ci.summary()

            # store formated results for given uplift
            self.res_sim.update({"{:.4f}".format(rel):[utilities.store_ci(ci), ci]})
        
        self.relup = relup_list

    
    def plot_sim_rel(self, res_sim=None):
        """
        Plot with 95% and 80% CI

        Args:
            res_sim (dictionary, optional): dictionary with formated causal impact result. Defaults to None.

        Returns:
            plt.figure:
        """

        if res_sim is None:
            res_sim = self.res_sim
        
        # extract results from res_sim class
        relup = self.relup
        rel = np.array([r[0].rel for r in res_sim.values()])
        rel_ci_l = np.array([r[0].rel_l for r in res_sim.values()])
        rel_ci_u = np.array([r[0].rel_u for r in res_sim.values()])
        rel_ci_l_80 = [r[0].rel_l_80 for r in res_sim.values()]
        rel_ci_u_80 = [r[0].rel_u_80 for r in res_sim.values()]

        fig = plt.figure()

        plt.plot(relup, rel, 'o-', label="predicted total average rel. effect")
        plt.plot(relup, relup*100, 'o-', label="real total average rel. effect")
        plt.fill_between(relup, y1=rel_ci_l, y2=rel_ci_u, color="blue", alpha=0.2)
        plt.fill_between(relup, y1=rel_ci_l_80, y2=rel_ci_u_80, color="orange", alpha=0.2)

        #plt.axvline(0.6, color="black", linestyle="dashed")
        plt.axhline(0, color="black")
        plt.xticks(ticks=relup, labels=["{:.0f} %".format(re) for re in relup*100], rotation=70)

        plt.xlabel("Relative uplift (real)")
        plt.ylabel("Relative uplift total")
        plt.legend()
        plt.close()

        return fig
    
    def plot_sim_cum(self, test_size=None, res_sim=None, kpi="regs"):
        """
        Plot the cummulative impact for all effect size

        Args:
            test_size (int): _description_
            res_sim (_type_, optional): _description_. Defaults to None.
            kpi (str): plot xlabel

        Returns:
            plt.figure: y axis with cummulative effect, x axis with time unit
        """

        if test_size is None:
            test_size = self.test_size
        
        if test_size is None:
            raise ValueError("The simulation has not been performed yet, please call the make_sim() function first")

        if res_sim is None:
            res_sim = self.res_sim

        # extract results from res_sim class
        relup = self.relup
        cum_effect = [r[0].cum for r in res_sim.values()]
        cum_effect_l = [r[0].cum_l for r in res_sim.values()]
        cum_effect_u = [r[0].cum_u for r in res_sim.values()]

        fig = plt.figure()

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        x = np.arange(test_size)

        for c, u, l, up, col in zip(cum_effect, cum_effect_u, cum_effect_l, relup, colors):
            plt.plot(x, u[-test_size:], label="{:.0f} %".format(up*100), color=col)
            plt.plot(x, l[-test_size:], label="_nolabel_", color=col, linestyle="dashed")

        plt.axhline(0, color="black")

        plt.title("Upper (dashed) and lower (solid) bound of the confidence interval for cumulative effect")
        plt.ylabel("Cumulative effect on total {}".format(kpi))
        plt.xlabel("days after intervention")
        plt.legend(title="uplift")

        # dont display
        plt.close()

        return fig


    def power_analyse(self, relup_list=None, 
                    post_per_length_max=30, post_per_length_min=4, min_training_len=None, N_training=5, 
                    model_args={"nseasons":7}, 
                    add_effect_args={"scale_std":0.05, "log_len":0},
                    on_trend=False, **decompose_kwargs):
        """
        This function performs more involved simulation to "mimic" a standard power analysis within the time serie framework.
        Through the following steps:
             - same step as make_sim()
             - for different intervention dates
             - for different test_size 
             - take the mean and/or max

        Args:
            relup_list (array_like, optional): list of relative uplift in fraction. Defaults to 0:0.2.
            post_per_length_max (int, optional): max test size in time unit. Defaults to 30.
            min_training_len (int, optional): min training size in time unit. Defaults to 0.2*len(data).
            N_training (int, optional): number of different intervention dates. Defaults to 5.
            model_args (dict, optional): argument for the causal impact model. Defaults to {"nseasons":7}.
            add_effect_args (dict, optional): argument for the simulated effect. Defaults to {"scale_std":0.01, "log_len":2}.

        Raises:
            ValueError: wrong min - max for training size

        Returns:
            plt.figure: heatmap with contours
        """

        if min_training_len is None:
            min_training_len = int(0.2*len(self.target))
        elif min_training_len < 7:
            print("WARNING: minimum training length is smaller than 7. This is very small ...")

        if relup_list is None:
            # arr of relative (negative) uplifts in fraction
            relup_list = -np.linspace(0.01, 0.2, 20)

        # arr of intervention length
        intervention_length = np.arange(post_per_length_min, post_per_length_max, 2)

        # information for monte carlo simulation
        max_training_len = len(self.target)-max(intervention_length)
        if min_training_len >= max_training_len:
            raise ValueError(f"Min training size of {min_training_len} is larger than max intervention length of {max_training_len}")
        # create space for variation in intervention date
        training_size_list = np.linspace(min_training_len, max_training_len, dtype=int, num=N_training)

        # initialize variables
        signi = np.zeros((len(relup_list), len(intervention_length))) # boolean signi not-signi
        pval = np.zeros_like(signi) # pvalue
        roll_avg = np.zeros_like(signi) # rolling average
        ci_width = np.zeros_like(signi) # width of the confidence interval
        ci_obj = []
        
        # create custom result class (a dictionray could be used instead)
        class res_power:
            def __init__(self, pval, ci_width, roll_avg, signi, relup_list, intervention_length, ci_obj):
                self.pval, self.ci_width, self.roll_avg, self.signi, self.relup_list, self.intervention_length, self.ci_obj \
                        = pval, ci_width, roll_avg, signi, relup_list, intervention_length, ci_obj

        # start simulations
        print("Starting simulation")

        # loop over relative uplift

        for i in trange(len(relup_list)):

            relup = relup_list[i]
            
            ci_obj_tmp = []
            # loop over post period length (intervention_length)
            j = 0
            for post_period_len in intervention_length:

                # temporary lists
                pval_tmp = np.zeros(len(training_size_list))
                roll_avg_tmp = np.zeros_like(pval_tmp)
                ci_width_tmp = np.zeros_like(pval_tmp)

                # loop over pre-period length (intervention date)
                k = 0
                for pre_period_len in training_size_list:

                    controls = self.controls[:pre_period_len+1+post_period_len]
                    target = self.target[:pre_period_len+1+post_period_len]

                    # calculate absolute uplift from post period 
                    up = relup*np.mean(target[pre_period_len+1:])

                    # add uplift with randomness to target
                    obs = utilities.add_effect(target, up, post_period_len+1, **add_effect_args)
                    data = pd.concat([obs, controls], axis=1) # merge controls and target
                    data.index = pd.to_datetime(data.index)

                    # decompose TS
                    if on_trend:
                        data, _, _ = utilities.decompose_components(data, **decompose_kwargs)
                        model_args["nseasons"] = None # remove seasonal components from model

                    pre_period = [data.index[0], data.index[pre_period_len]]
                    post_period = [data.index[pre_period_len+1], data.index[-1]]

                    ci = dep_CausalImpact(data, pre_period, post_period, model_args=model_args)
                    ci.run()
                    ci.summary()
                    res = utilities.store_ci(ci)

                    # interval from model -> abs CI width
                    ci_width_tmp[k] = abs(ci.inferences.point_effect_lower.values[-1] - \
                                        ci.inferences.point_effect_upper.values[-1])

                    # rolling average
                    roll_avg_tmp[k] = utilities.rolling_average(ci.inferences.point_effect.values[-post_period_len:])[-1]

                    # pval
                    pval_tmp[k] = float(ci.summary_df.loc["P-value", "Average"][:-1])

                    k += 1
                
                # take mean out of all simulations
                ci_width[i, j] = np.mean(ci_width_tmp)
                roll_avg[i,j] = np.mean(roll_avg_tmp)
                signi[i,j] = (roll_avg[i,j]+ci_width[i,j]/2 <= 0)
                pval[i,j] = np.max(pval_tmp) # does not make a lot of sense to take the mean of p_values ?

                ci_obj_tmp.append(ci)

                j += 1
            
            ci_obj.append(ci_obj_tmp)
        
        # store result in res_power class
        if self.res_power:
            print("Warning: result of power analysis overwritten")
        self.res_power = res_power(pval, ci_width, roll_avg, signi, relup_list, intervention_length, ci_obj)

        # return heatmap 
        return self.plot_power()
    

    def plot_power(self, pval=None, relup_list=None, intervention_length=None, alpha=[5]):
        """
        Heat map using the result of the power analysis

        Args:
            pval (np.array, optional): Matrix with pvalues. Defaults to None, take the class value.
            relup_list (array_like, optional): list of relative uplift. Defaults to None, take class value.
            intervention_length (array_like, optional): list of test sizes. Defaults to None, take class values
            alpha (list, optional): list of desired contour for the p-values. Defaults to [5].

        Returns:
            plt.figure: heatmap
        """
        
        # get result from simualtion

        if pval is None:
            pval = self.res_power.pval
        if relup_list is None:
            relup_list = self.res_power.relup_list
        if intervention_length is None:
            intervention_length = self.res_power.intervention_length
            
        # plot results
        fig = plt.figure()
        im = plt.imshow(pval, interpolation="spline16")
        interp_data = im.get_array()

        plt.colorbar(label="p-value")
        CS = plt.contour(interp_data, levels=alpha, colors="white")
        plt.yticks(ticks=np.arange(len(relup_list)), labels=["{:.0f} %".format(up*100) for up in relup_list], rotation=0)
        plt.xticks(ticks=np.arange(len(intervention_length))[::2], labels=intervention_length[::2])
        plt.ylabel("Relative uplift")
        plt.xlabel("Days after intervention")

        # add countour labels
        plt.clabel(CS, alpha, inline=1, fontsize=10)

        # dont dipsplay
        plt.close()

        return fig
