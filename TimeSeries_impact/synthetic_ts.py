# synthetic_generator.py
import numpy as np
import pandas as pd


def make_time_series(
    N,
    sig=[2, 1, 5],
    trend_line_coeff=[10, 8, 4],
    trend_line_interc=[100, 70, 20],
    trend_list=None,
    amp=[[5], [2], [3]],
    freq=[7],
    nbr_rand_event=0,
    rand_event_str=0.2,
):
    """
    Construct synthetic time series with linear trends, seasonality, noise, and optional random events.

    Returns a dict with target series, controls, seasonality components, and baseline trends.
    """
    if trend_list is None:
        trend_list = [None] * len(sig)

    x = np.arange(N)
    n_ts = len(sig)
    n_controls = n_ts - 1

    # Validation
    assert len(trend_line_coeff) == n_ts
    assert len(trend_line_interc) == n_ts
    assert len(trend_list) == n_ts
    assert len(amp) == n_ts
    assert all(len(amp[i]) == len(freq) for i in range(n_ts))

    # Generate trends
    trends = []
    for i in range(n_ts):
        if trend_list[i] is not None:
            trend = np.array(trend_list[i])
            assert len(trend) == N
        else:
            trend = trend_line_interc[i] + x * (trend_line_coeff[i] / x.max())
        trends.append(trend)

    # Add random events
    rand_event_pos = []
    if nbr_rand_event > 0:
        starts = np.random.randint(0, N - 10, size=nbr_rand_event)
        lengths = np.random.randint(5, 10, size=nbr_rand_event)
        for st, ln in zip(starts, lengths):
            mag = rand_event_str * np.random.random()
            sign = np.random.choice([-1, 1])
            for idx in np.random.choice(n_ts, size=2, replace=False):
                amp_mod = np.mean(trends[idx]) * mag * sign
                trends[idx][st:st + ln] += amp_mod
            rand_event_pos.append((st, st + ln))

    # Add seasonality and noise
    data, seasonal_components = [], []
    for i in range(n_ts):
        series = trends[i].copy()
        season_total = np.zeros(N)
        for a, f in zip(amp[i], freq):
            season = a * np.cos(2 * np.pi * x / f)
            series += season
            season_total += season
        series += np.random.normal(scale=sig[i], size=N)
        data.append(series)
        seasonal_components.append(season_total)

    df = pd.DataFrame(index=pd.date_range(start="2023-01-01", periods=N, freq="D"))
    df["target"] = data[0]
    for i, control in enumerate(data[1:], start=1):
        df[f"control{i}"] = control

    return {
        "data": df,
        "season": seasonal_components,
        "base": trends,
        "rand_event": rand_event_pos,
    }
