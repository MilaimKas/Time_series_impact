
import numpy as np
import pandas as pd
import TimeSeries_impact.utilities as utilities
from statsmodels.tsa.seasonal import seasonal_decompose, STL
import matplotlib.pypplot as plt
from causalimpact import CausalImpact


def plot_cum_effect(self, ci, test_size, kpi="registrations"):
    """
    Use custom res class to plot cumulative impact for kpi

    Args:
        ci (ci object): causal impact object after run.
        test_size (int): number of elapased days since intervention.
        kpi (str, optional): name of the kpi. Defaults to "registrations".

    Returns:
        pyplot fig object
    """

    # get ci res class
    ci_res = utilities.store_ci(ci)

    fig = plt.figure()

    x = ci_res.cum.index[-test_size:]

    plt.plot(x, ci_res.cum[-test_size:], '-o', label="{} difference".format(kpi))
    plt.fill_between(x, ci_res.cum_l[-test_size:], ci_res.cum_u[-test_size:], alpha=0.3, label="95% CI")
    plt.fill_between(x, ci_res.cum_l_80[-test_size:], ci_res.cum_u_80[-test_size:], alpha=0.3, label="80% CI")
    plt.axhline(0, color="black", linestyle="dashed")

    # max 15 dates to show on x axis
    step = max(len(x)//15, 1)
    plt.xticks(ticks=x[::step], labels=x[::step].strftime('%Y-%m-%d'), rotation=70)
    plt.ylabel("Cumulative effect on {}".format(kpi))
    plt.xlabel("Dates")
    plt.legend()
    
    return fig


def plot_rel_effect(self, ci, test_size, kpi="registrations"):
    """
    Plot rolling average impact estimation

    Args:
        ci (ci object): causal impact object after run.
        test_size (int): number of elapsed days since intervention date.
        kpi (str, optional): name of the kpi. Defaults to "registrations".

    Returns:
        pyplot fig object
    """

    # get ci res class
    ci_res = utilities.store_ci(ci)

    fig = plt.figure()

    # get time stamps 
    x = ci_res.cum.index[-test_size:]
    
    # cumulative to avg CI
    avg_ci_width = abs(ci_res.cum_l[-test_size:] - ci_res.cum_u[-test_size:])/range(1, len(ci_res.cum[-test_size:])+1)
    avg_ci_width_80 = abs(ci_res.cum_l_80[-test_size:] - ci_res.cum_u_80[-test_size:])/range(1, len(ci_res.cum[-test_size:])+1)

    avg = ci_res.cum[-test_size:]/range(1, len(ci_res.cum[-test_size:])+1)
    ci_u = avg+avg_ci_width/2
    ci_l = avg-avg_ci_width/2
    ci_u_80 = avg+avg_ci_width_80/2
    ci_l_80 = avg-avg_ci_width_80/2

    plt.plot(x, avg, '-o', label="Rolling average difference")
    plt.fill_between(x, ci_l, ci_u, alpha=0.3, label="95% CI")
    plt.fill_between(x, ci_l_80, ci_u_80, alpha=0.3, label="80% CI")
    plt.axhline(0, color="black", linestyle="dashed")

    # max 15 dates to show
    step = max(len(x)//15, 1)
    plt.xticks(ticks=x[::step], labels=x[::step].strftime('%Y-%m-%d'), rotation=70)    
    plt.xlabel("Dates")
    plt.ylabel("Rolling average on {}".format(kpi))
    plt.legend()

    return fig

def plot_post_period(ci, kpi="registrations", past_days=60):
    """
    Plot the observations and predicted values around the intervention date

    Args:
        ci (ci object): causal impact object after run.
        kpi (str, optional): name of the kpi. Defaults to "registrations".
        past_days (int, optional): number of days prior to intervention date to depict. Defaults to 60.

    Returns:
        pyplot fig object
    """

    # get ci res class
    ci_res = utilities.store_ci(ci)

    fig = plt.figure()

    from_date = ci.params["post_period"][0] - timedelta(days=past_days)

    point_pred = ci.inferences['point_pred'][ci.inferences['point_pred'].index>from_date]
    resp = ci.inferences['response'][ci.inferences['response'].index>from_date]
    x = point_pred.index

    # visualize post period - past_days
    plt.plot(point_pred, '-o', color="blue", linestyle="dashed", label="predicted")
    plt.plot(resp, '-o', color="gray", label="observation")

    # intervention date
    plt.axvline(ci.params["post_period"][0], color="black", label="intervention")
    #plt.annotate('intervention',xy=(ci.params["post_period"][0],max(point_pred)))

    # max 15 dates to show
    step = max(len(x)//15, 1)
    plt.xticks(ticks=x[::step], labels=x[::step].strftime('%Y-%m-%d'), rotation=70)

    plt.xlabel("Dates")
    plt.ylabel(kpi)
    plt.legend()
    
    return fig

