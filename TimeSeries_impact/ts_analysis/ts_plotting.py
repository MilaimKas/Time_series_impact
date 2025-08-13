# ts_plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from TimeSeries_impact.utilities import scale_minmax
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools

HIGH_CONTRAST_COLORS = [
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
]

class Plotter:
    def __init__(self, tsa_instance, color_map=None):

        # check if decompose has been called
        if not tsa_instance.decomposer:
            raise ValueError("tsa instance does not have any components, did you forget to call decompose first ?")

        self.tsa = tsa_instance
        self.data = tsa_instance.data
        self.date = tsa_instance.date
        self.columns = tsa_instance.columns
        self.target = tsa_instance.target
        self.controls = tsa_instance.controls
        self.control_columns = tsa_instance.control_columns
        self.intervention_date = tsa_instance.intervention_date
        self.colors = color_map or dict(zip(self.columns, HIGH_CONTRAST_COLORS))

    def plot(self, max_xticks=20, scaled=True, shifted=True, with_trend=True, sns_kwargs={}):
        fig = plt.figure()

        for col in self.columns:
            series = self.data[col]
            trend = self.tsa.decomposer.components["trend"][col]
            color = self.colors[col]

            if scaled:
                min_val = min(series.min(), trend.min())
                max_val = max(series.max(), trend.max())
                series = scale_minmax(series, min_val, max_val)
                trend = scale_minmax(trend, min_val, max_val)

            if shifted:
                shift_amount = self.columns.get_loc(col)
                series += shift_amount
                trend += shift_amount

            plt.plot(self.date, series, label=col, alpha=0.4, color=color, **sns_kwargs)
            if with_trend:
                plt.plot(self.date, trend, label=f"{col} trend", linestyle="--", linewidth=2, color=color, **sns_kwargs)

        if self.intervention_date:
            plt.axvline(pd.to_datetime(self.intervention_date), color="black", linestyle="--")
        
        # y label
        y_label = ""
        if scaled:
            y_label += "scaled "
        else:
            y_label += "raw "
        if shifted:
            y_label += "shifted "
        y_label += "values"

        step = len(self.date) // max_xticks
        plt.xticks(self.date[::step], self.date.astype(str)[::step], rotation=70)
        plt.ylabel(y_label)
        plt.xlabel("Date")
        plt.legend(loc='lower right')
        plt.close()

        return fig

    def plot_components(self, components):
        for comp_name in ["resid", "trend", "seas"]:
            plt.figure()
            for col in components[comp_name].columns:
                color = self.colors.get(col, "gray")
                data = scale_minmax(components[comp_name][col])
                if comp_name == "resid":
                    sns.histplot(data, label=col, color=color, stat="density")
                else:
                    sns.lineplot(x=self.date, y=data, label=col, color=color)
                    plt.ylabel("Scaled values")
            plt.title(f"{comp_name.capitalize()} component")
            plt.legend()
            plt.xticks(rotation=70)
            plt.show()

    def plot_autocorrelation(self, df):
        for kind, func in zip(["Auto-Correlation function", "Partial Auto-Correlation Function"], [plot_acf, plot_pacf]):
            fig, ax = plt.subplots()
            for col in df.columns:
                color = self.colors.get(col, "gray")
                func(df[col], ax=ax, label=f"{col}", color=color)
            plt.title(kind)
            plt.legend()
            plt.show()
