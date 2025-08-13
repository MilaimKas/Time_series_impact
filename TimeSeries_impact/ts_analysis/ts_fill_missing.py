# ts_fill.py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.structural import UnobservedComponents
from TimeSeries_impact.utilities import scale_minmax


def fill_missing_values(
    df: pd.DataFrame,
    inplace: bool = True,
    plot: bool = False,
    model_uc_kwargs: dict = None
) -> pd.Series:
    """
    Fill in missing values in the first column of `df` using Unobserved Components model.

    Args:
        df (pd.DataFrame): First column is target with missing values; others are regressors.
        inplace (bool): If True, modifies df in-place. Otherwise returns filled series.
        plot (bool): Whether to display plot of original vs filled series.
        model_uc_kwargs (dict): Parameters for UnobservedComponents.

    Returns:
        pd.Series: The filled target series (only if inplace=False).
    """
    if model_uc_kwargs is None:
        model_uc_kwargs = dict(
            level=True,
            trend=True,
            seasonal=7,
            stochastic_level=True,
            stochastic_trend=True,
            stochastic_seasonal=True
        )

    target = df.iloc[:, 0]
    regressors = df.iloc[:, 1:]

    if target.isna().sum() == 0:
        return target.copy()

    first_missing_idx = target.index[target.isna()][0]
    train_end_idx = target.index.get_loc(first_missing_idx) - 1

    target_train = target.iloc[:train_end_idx + 1]
    regressors_train = regressors.iloc[:train_end_idx + 1]
    regressors_forecast = regressors.iloc[train_end_idx + 1:]

    model = UnobservedComponents(target_train, exog=regressors_train, **model_uc_kwargs)
    result = model.fit(disp=False)

    forecast = result.predict(
        start=target.index[train_end_idx + 1],
        end=target.index[-1],
        exog=regressors_forecast
    )

    filled = target.copy()
    filled.loc[target.isna()] = forecast.loc[target.isna()]

    if plot:
        plt.plot(scale_minmax(target), label="original", linewidth=2)
        plt.plot(scale_minmax(filled), label="filled", linestyle="dashed", linewidth=1)
        plt.xticks(rotation=70)
        plt.legend()
        plt.title("Missing value imputation")
        plt.show()

    if inplace:
        df.iloc[:, 0] = filled
    
    return filled
