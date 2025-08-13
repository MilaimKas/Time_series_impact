# tsa.py
import pandas as pd
from TimeSeries_impact.ts_analysis.ts_decomposition import Decomposer
from TimeSeries_impact.ts_analysis.ts_similarity import SimilarityAnalyzer
from TimeSeries_impact.ts_analysis.ts_plotting import Plotter
from TimeSeries_impact.ts_analysis.ts_fill_missing import fill_missing_values

class TSA:

    def __init__(self, df: pd.DataFrame, intervention_date=None):

        self.data = df.rename(columns={df.columns[0]: df.columns[0] + " -> target"}).sort_index()
        self.date = self.data.index
        self.intervention_date = intervention_date

        self.target = df.iloc[:, 0]
        self.controls = df.iloc[:, 1:]

        self.columns = self.data.columns
        self.control_columns = self.controls.columns

        self.decomposer = Decomposer(self.data)
        self.similarity_analyzer = SimilarityAnalyzer(self.target, self.controls)
        self.plotter = Plotter(self)

    def decompose(self, **kwargs):
        self.decomposer.decompose(**kwargs)

    def get_components_similarity(self, period=7):
        return self.similarity_analyzer.analyze_components(self.decomposer.components, period)
    
    def get_data_similarity(self, ):
        return self.similarity_analyzer.compute_similarity_metrics()

    def optimize_decomposition(self, period=7, seas_weight=1):
        return self.similarity_analyzer.optimize_decomposition(self.target, self.controls, period, seas_weight)

    def fill_data(self, inplace=True, plot=False, **kwargs):
        res = fill_missing_values(self.data, inplace=inplace, plot=plot, **kwargs)
        # reset if data has been modified inplace 
        if inplace:
            self.reset()
        return res

    def plot(self, max_xticks=20, scaled=True, shifted=False, with_trend=True):
        return self.plotter.plot(max_xticks, scaled=scaled, shifted=shifted, with_trend=with_trend)

    def plot_components(self):
        self.plotter.plot_components(self.decomposer.components)

    def plot_autocorrelation(self):
        self.plotter.plot_autocorrelation(self.data)

    def get_correlation(self, df=None, on_trend=False, **kwargs):
        if df is None:
            df = self.decomposer.components["trend"] if on_trend else self.data
        return self.similarity_analyzer.compute_correlation(df)
    
    def reset(self):
        self.target = self.data.iloc[:, 0]
        self.controls = self.data.iloc[:, 1:]
        self.decomposer = Decomposer(self.data)
        self.decompose()
        self.similarity_analyzer = SimilarityAnalyzer(self.target, self.controls)

