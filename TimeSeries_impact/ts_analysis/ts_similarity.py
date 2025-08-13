# ts_similarity.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dtaidistance import dtw
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from TimeSeries_impact.utilities import scale_minmax, scale_std, normal_similarity_metrics
from statsmodels.tsa.seasonal import STL
from scipy.optimize import differential_evolution

class SimilarityAnalyzer:
    def __init__(self, target: pd.Series, controls: pd.DataFrame):
        self.target = target
        self.controls = controls
        self.columns = controls.columns

    def analyze_components(self, components: dict, period=7):

        # residuals
        res_res = pd.DataFrame()
        target_resid = scale_minmax(components["resid"].iloc[:, 0])

        for c in self.columns:
            control_resid = scale_minmax(components["resid"][c])
            dict_simi = normal_similarity_metrics(target_resid, control_resid)
            res_res[c] = dict_simi.values()
        res_res.index = dict_simi.keys()

        # seasonal
        res_seas = pd.DataFrame()
        target_seas = components["seas"].iloc[:, 0]
        for c in self.columns:
            control_seas = components["seas"][c]
            dtw_dist = dtw.distance(target_seas, control_seas) / len(target_seas)
            res_seas[c] = [dtw_dist]
        res_seas.index = ["dtw"]
        
        return res_res, res_seas

    def optimize_decomposition(self, target=None, controls=None, period=7, seas_weight=1):
        
        if target is None:
            target = self.target
        if controls is None:
            controls = self.controls

        bounds = [(period + 1, len(target)), (period + 1, len(target))]
        result = differential_evolution(
            self.similarity_metric,
            bounds,
            strategy="best1bin",
            args=(target, controls, period, seas_weight),
            disp=True
        )

        seasonal, trend = map(int, map(round, result.x))
        if seasonal % 2 == 0: seasonal += 1
        if trend % 2 == 0: trend += 1

        return seasonal, trend

    def similarity_metric(self, params, target=None, controls=None, period=7, seas_weight=1, resid_metric="Bhattacharyya Distance"):

        if target is None:
            target = self.target
        if controls is None:
            controls = self.controls

        seasonal, trend = map(int, map(round, params))
        if seasonal % 2 == 0: seasonal += 1
        if trend % 2 == 0: trend += 1

        stl_target = STL(target, seasonal=seasonal, trend=trend, period=period).fit()
        stl_control = STL(controls.sum(axis=1), seasonal=seasonal, trend=trend, period=period).fit()

        target_seas = scale_std(stl_target.seasonal)
        control_seas = scale_std(stl_control.seasonal)
        target_resid = scale_std(stl_target.resid)
        control_resid = scale_std(stl_control.resid)

        seasonal_sim = dtw.distance(target_seas, control_seas) / len(target_seas)
        residual_sim = normal_similarity_metrics(target_resid, control_resid).get(resid_metric, np.nan)

        return abs(seasonal_sim + seas_weight * residual_sim)

    def plot_similarity_surface(self, target=None, controls=None, period=7, seas_weight=1, resid_metric="Bhattacharyya Distance",
                                seasonal_range=None, trend_range=None):
        
        if target is None:
            target = self.target
        if controls is None:
            controls = self.controls

        if seasonal_range is None:
            seasonal_range = range(period + 1, min(len(target), 60), 5)
        if trend_range is None:
            trend_range = range(period + 1, min(len(target), 60), 5)

        sim_matrix = np.zeros((len(seasonal_range), len(trend_range)))

        for i, s in enumerate(seasonal_range):
            for j, t in enumerate(trend_range):
                sim_matrix[i, j] = self.similarity_metric([s, t], target, controls, period, seas_weight, resid_metric)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(sim_matrix, xticklabels=list(trend_range), yticklabels=list(seasonal_range), cmap="viridis", ax=ax)
        ax.set_xlabel("Trend Window")
        ax.set_ylabel("Seasonal Window")
        ax.set_title("Similarity Score Surface")
        plt.tight_layout()
        plt.show()

    def compute_correlation(self, df: pd.DataFrame):
        return (
            df.corr(method="pearson"),
            df.corr(method="spearman"),
            df.corr(method="kendall")
        )
    
    def compute_similarity_metrics(self, scaled=True):
        metrics = []

        if scaled:
            target = scale_minmax(self.target)
            controls = self.controls.apply(scale_minmax)
        else:
            target = self.target
            controls = self.controls

        metrics.append([
            np.sqrt(np.sum((target.values - controls[c].values) ** 2))
            for c in controls.columns
        ])
        metrics.append([
            np.mean(np.abs((target.values - controls[c].values) / (target.values + 1e-10)))
            for c in controls.columns
        ])
        metrics.append([
            np.corrcoef(target.values, controls[c].values)[0, 1]
            for c in controls.columns
        ])
        metrics.append([
            dtw.distance(target.values, controls[c].values) / len(target)
            for c in controls.columns
        ])

        return pd.DataFrame(metrics, columns=self.columns, index=["euclidean", "MAPE", "pearson", "dtw"])

