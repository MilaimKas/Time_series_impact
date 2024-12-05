import numpy as np
import pandas as pd

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.seasonal import MSTL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.structural import UnobservedComponents

from scipy.optimize import minimize
from scipy.optimize import differential_evolution

from TimeSeries_impact.utilities import *
from TimeSeries_impact import plot_functions

import matplotlib.pyplot as plt
import seaborn as sns

from dtaidistance import dtw



high_contrast_colors = [
    'tab:blue',    # Blue
    'tab:orange',  # Orange
    'tab:green',   # Green
    'tab:red',     # Red
    'tab:purple',  # Purple
    'tab:brown',   # Brown
    'tab:pink',    # Pink
    'tab:gray',    # Gray
    'tab:olive',   # Olive
    'tab:cyan'     # Cyan
]


def fill_data(df, inplace=True, plot=False, 
                    level=True, trend=False, seasonal=7,
                    stochastic_level=True, stochastic_trend=False, stochastic_seasonal=True):
    """
        fill in missing data on a target variable when giving regressor with all data


    Args:
        df (pd.DataFrame): first column is the target, other are regressors. 
                            The target columns mus contain nan values that need to be filled
        inplace (bool, optional): change value inplace. Defaults to True.
        plot (bool, optional): plot the original and new data. Defaults to False.
        level (bool, optional): level component. Defaults to True.
        trend (bool, optional): trend component. Defaults to False.
        seasonal (int, optional): seasonal period. Defaults to 7.
        stochastic_level (bool, optional): stochastic level component. Defaults to True.
        stochastic_trend (bool, optional): stochastic trend component. Defaults to False.
        stochastic_seasonal (bool, optional): stochastic seasonal component. Defaults to True.

    Returns:
        pd.Series: target data with filled values
    """

    # Split data into target and regressors
    target = df.iloc[:,0]
    regressors = df.iloc[:,1:]

    # Identify the index of the first missing value
    first_missing_idx = target.index[target.isna()][0]

    # Split the data into training data (prior to the gap) and forecasting data
    train_end_idx = target.index.get_loc(first_missing_idx) - 1
    target_train = target.iloc[:train_end_idx+1]
    regressors_train = regressors.iloc[:train_end_idx+1]
    regressors_forecast = regressors.iloc[train_end_idx+1:]

    # Define and fit the UC model
    model = UnobservedComponents(target_train, exog=regressors_train, 
                    level=level, trend=trend, seasonal=seasonal,
                    stochastic_level=stochastic_level, stochastic_trend=stochastic_trend, 
                    stochastic_seasonal=stochastic_seasonal)
    result = model.fit(disp=False)

    # Forecast the missing values using the trained model
    forecast = result.predict(start=target.index[train_end_idx+1], end=target.index[-1], exog=regressors_forecast)

    # Forecast or interpolate the missing values
    target_filled = target.copy()
    target_filled.loc[target.isna()] = forecast[target.isna()]

    # plot the results
    plt.plot(target_filled, label="filled data")
    plt.plot(target, label="original data")
    plt.xlabel("timestamp")
    plt.ylabel("kpi")
    plt.legend()
    plt.xticks(rotation=70)

    if plot:
        plt.show()
    else:
        plt.close()

    # return or store inplace results 
    if inplace:
        df.iloc[:,0] = target_filled
    else:
        return target_filled


class TSA:

    def __init__(self, df, intervention_date=None):

        self.data = df.rename(columns={df.columns[0]: df.columns[0] + " -> target" }).sort_index()
        self.date = self.data.index
        self.intervention_date = intervention_date

        self.target = df.iloc[:,0]
        self.controls = df.iloc[:,1:]

        self.columns = self.data.columns
        self.control_columns = self.controls.columns

        self.component = {}

    def plot(self, max_xticks=20, **decompose_kwargs):
        """
        Plot scaled values with trend components as time series

        Returns:
            _type_: _description_
        """

        if not self.component:
            self.decompose(**decompose_kwargs)  

        # scaled data and trend components
        fig = plt.figure()

        # plot target
        min, max = self.target.min(), self.target.max()
        plt.plot(self.date, scale_minmax(self.target, min=min, max=max), label="target", alpha=0.4, color=high_contrast_colors[0])
        plt.plot(self.date, scale_minmax(self.component["trend"].iloc[:,0], min=min, max=max), 
                            label="target trend", color=high_contrast_colors[0], linestyle="solid", linewidth=5)
        
        #plot controls
        for c, col in zip(self.control_columns, high_contrast_colors[1:]):
            min, max = self.controls[c].min(), self.controls[c].max()
            plt.plot(self.date, scale_minmax(self.controls[c], min=min, max=max), label=c, alpha=0.4, color=col)
            plt.plot(self.date, scale_minmax(self.component["trend"][c], min=min, max=max), 
                            label=c+" trend", color=col, linestyle="dashed", linewidth=3)

        # xticks labels
        steps = len(self.date)//max_xticks
        plt.xticks(ticks=self.date[::steps], labels=self.date.astype(str)[::steps], rotation=70)

        if self.intervention_date is not None:
            plt.axvline(pd.to_datetime(self.intervention_date), color="black", linestyle="dashed")

        plt.ylabel("Scaled values")
        plt.xlabel("date")

        plt.legend()
        plt.close()

        return fig

    def decompose(self, **decompose_kwargs):
        """
                perform seasonal analysis and decomposition


        Args:
            period (int, optional): seasonal period. Defaults to 7 (weekly periodicity).
            trend (int, optional): trend window smoothing. Defaults to None.
            seasonal (int, optional): seasonal window smoothing. Defaults to None -> len(df)//5
        """

        # default parameters
        if "seasonal" not in decompose_kwargs.keys():
            seasonal = seasonal_default(len(self.data))
        else:
            seasonal = decompose_kwargs["seasonal"]
        decompose_kwargs["seasonal"] = seasonal
        if "period" not in decompose_kwargs.keys():
            decompose_kwargs["period"] = 7

        # autocorrelation
        # ------------------------------------------------------------------------------

        fig_acf, ax_acf = plt.subplots()
        for c, col in zip(self.columns, high_contrast_colors):

            acorr, confint, qstat, pval = acf(self.data[c], qstat=True, alpha=0.05)
            plot_acf(self.data[c], ax=ax_acf, color=col, label=c)

            # Extract the confidence interval around 0 (null hypothesis)
            confint_zero_centered = np.array([[-1.96/np.sqrt(len(self.data)), \
                        1.96/np.sqrt(len(self.data))] for _ in range(len(acorr))])
            # Identify lags where the ACF values exceed the upper confidence limit
            significant_lags = np.where(acorr > confint_zero_centered[:, 1])[0]

            if "period" in decompose_kwargs.keys():
                if decompose_kwargs["period"] not in significant_lags:
                    print(f"WARNING: for {c}, no seasonality of period {decompose_kwargs['period']} was detected at 5% significance")

        # add legend
        ax_acf.legend(self.columns)
        # store figure as class attribute
        self.correlation_plot = fig_acf
        
        # do not show plots inline
        plt.close()


        # decomposition
        # ------------------------------------------------------------------------------
        
        trend, seas, resid = decompose_components(self.data, **decompose_kwargs)
        # store as class attribute
        self.component.update({"trend":trend, "seas":seas, "resid":resid})

    
    def analyze(self, period=7):
        """
        analyse components of target and controls and asses if similar
        """

        # residuals -> normal similarity metric (different score based on normal assumption) 
        res_res = pd.DataFrame()
        for c in self.columns[1:]:
            target_scaled = scale_minmax(self.component["resid"].iloc[:,0])
            control_scaled = scale_minmax(self.component["resid"][c])
            metrics = normal_similarity_metrics(target_scaled, control_scaled)
            res_res[c] = metrics.transpose()
        
        # seasonal -> DTW coefficient
        res_seas = pd.DataFrame()
        target_seas_res = self.component["seas"].iloc[:,0]
        for c in self.columns[1:]:
            control_seas_resid = self.component["seas"][c]
            res_seas[c] = [dtw.distance(target_seas_res, control_seas_resid)/len(target_seas_res)]

        return res_res, res_seas
    
    def optimize_decomposition(self, period=7, seas_weight=1):
        """
        This function finds the set of trend window and seasonal window that 
        maximizes the similarity between the residuals and seasonal components of the 
        target and the sum of control time series.

        Args:
            period (int, optional): period for the decomposition. Defaults to 7 (daily).

        Returns:
            _type_: _description_
        """

        # Initial parameter values
        initial_params = [50, 20]  # starting with windows as initial guess for both trend and seasonality
        
        # Parameter bounds: [min, max] for seasonal and trend windows
        bounds = [(period+1, len(self.data)), (period+1, len(self.data))]

        # Optimize using differential algorithm (allows to tackle integers)
        result = differential_evolution(
            self.similarity_metric, # objectif function
            bounds,
            strategy="best1bin",  # integer-based steps
            disp=True,  # Set to True to display convergence messages
            args=(period, seas_weight)
            )
    
        # Round final parameters to integers and make sure parameters are odds
        optimal_seasonal = int(round(result.x[0]))
        if optimal_seasonal % 2 == 0:
            optimal_seasonal += 1
        optimal_trend = int(round(result.x[1]))
        if optimal_trend % 2 == 0:
            optimal_trend += 1

        return optimal_seasonal, optimal_trend   # optimal parameters

    def similarity_metric(self, params, period=7, seas_weight=1, resid_metric="Bhattacharyya Distance"):
        """
        Function that calculates a similarity score as weighted sum of the residuals and seasonal components

        Args:
            params (_type_): _description_
            period (int, optional): _description_. Defaults to 7.

        Returns:
            _type_: _description_
        """
        
        seasonal, trend = params
        
        # Ensure parameters are integers and odd
        seasonal = int(round(params[0]))
        if seasonal % 2 == 0:
            seasonal += 1
        trend = int(round(params[1]))
        if trend % 2 == 0:
            trend += 1
        
        # Perform STL decomposition
        stl_target = STL(self.target, seasonal=seasonal, trend=trend, period=period).fit()
        stl_control = STL(self.controls.sum(axis=1), seasonal=seasonal, trend=trend, period=period).fit()
        
        # Extract seasonal components
        target_seasonal = stl_target.seasonal
        control_seasonal = stl_control.seasonal
        
        # Extract residualy components
        target_resid = stl_target.resid
        control_resid = stl_control.resid
        
        # scale components
        target_seasonal = scale_std(target_seasonal)
        control_seasonal = scale_std(control_seasonal)
        target_seasonal = scale_std(target_resid)
        control_seasonal = scale_std(control_resid)
        
        # Calculate similarity metrics for seasonal and residual component
        seasonal_similarity = dtw.distance(target_seasonal, control_seasonal)/len(target_seasonal) # DTW distance
        residual_similarity = normal_similarity_metrics(target_resid, control_resid)
        
        if resid_metric in residual_similarity.keys():
            residual_similarity = residual_similarity[resid_metric]
        else:
            raise ValueError(f"Similarity metric not recognize, available metrics: {residual_similarity.keys()}")
        
        # Give more weight to seasonal components
        total_similarity = abs(seasonal_similarity + residual_similarity)
        
        return total_similarity
    
    def plot_component(self):
        """
        Plot the trend, seasonal and residual component of all time series
        """

        if not self.component:
            raise ValueError("Not components found, perform decomposition first")

        # residuals
        for col, color in zip(self.component["resid"].columns, high_contrast_colors):
            sns.histplot(scale_minmax(self.component["resid"][col]), label=col, color=color, stat="density")
        plt.ylabel("residuals density")
        plt.title("Residual component")
        plt.legend()
        plt.show()

        # trend
        for col, color in zip(self.component["trend"].columns, high_contrast_colors):
            if "target" in col:
                sns.lineplot(scale_minmax(self.component["trend"][col]), label=col, color=color, linestyle="solid")
            else:
                sns.lineplot(scale_minmax(self.component["trend"][col]), label=col, color=color, linestyle="dashed")
        plt.ylabel("Normalized metric value")
        plt.xlabel("timestamp")
        plt.legend()
        plt.title("Trend component")
        plt.show()

        # seasonalit
        for col, color in zip(self.component["seas"].columns, high_contrast_colors):
            sns.lineplot(scale_minmax(self.component["seas"][col]), label=col, color=color)
        plt.ylabel("Metric value")
        plt.title("Seasonal component")
        plt.ylabel("timestamp")
        plt.legend()
        plt.show()
    
    def plot_autocorrelation(self):
        """
        Plot autocorrelation and partial autocorrelation fro all data

        Returns:
             show autocorrelation plot 
        """

        if not self.component:
            self.decompose()  

        # autocorrelation
        fig, ax = plt.subplots()
        handles = []
        for col in self.columns:
            acorr, confint, qstat, pval = acf(self.data[col], qstat=True, alpha=0.1)
            plot_acf(self.data[col], ax=ax, label=col)
            lines = ax.get_lines()
            handles.append(lines[-1])
        ax.legend(handles, self.columns)
        plt.show()

        # partial autocorrelation
        fig, ax = plt.subplots()
        handles = []
        for col in self.columns:
            acorr, confint, qstat, pval = acf(self.data[col], qstat=True, alpha=0.1)
            plot_pacf(self.data[col], ax=ax, label=col)
            lines = ax.get_lines()
            handles.append(lines[-1])
        ax.legend(handles, self.data.columns)
        plt.show()
        
    def plot_scaled_view(self):
        return plot_functions.scaled_view(self.data, None)


    def correlation(self, df=None, on_trend=False, **decompose_kwargs):
        """
        Calculates standard correlation coefficients

        Args:
            df (pd.DataFrame, optional): df containing data. Defaults to None.
            on_trend (bool, optional): True if using trend component. Defaults to False.

        Returns:
            _type_: _description_
        """
        
        if df is None:
            if on_trend:
                if not self.component:
                    self.decompose(**decompose_kwargs)
                df = self.component["trend"]
            else:
                df = self.data.copy()

        # Calculate Pearson correlation coefficient
        pearson_corr = df.corr(method='pearson')

        # Calculate Spearman correlation coefficient
        spearman_corr = df.corr(method='spearman')

        # Calculate Kendall correlation coefficient
        kendall_corr = df.corr(method='kendall')

        return pearson_corr, spearman_corr, kendall_corr


    

