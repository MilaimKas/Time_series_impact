import numpy as np


def rolling_average(a):
    ret = np.cumsum(a)
    return ret / range(1, len(a)+1)
    
def scale_minmax(arr):
    arr_res = (arr-arr.min())/(arr.max()-arr.min())
    return arr_res

def scale_std(arr):
    arr_res = (arr-arr.mean())/arr.std()
    return arr_res

def store_ci(ci):
    """
    Extract results from causalimpact object and store them in the res class

    Args:
        ci (_object_): casualimpact object

    Returns:
        class: res with relative and absolute effect values
    """

    pval = float(ci.summary_df.loc["P-value"]["Average"][-2])

    rel = float(ci.summary_df.loc["Relative Effect"]["Average"][:-1])
    rel_u = float(ci.summary_df.loc["95% CI"]["Average"][2][0][:-1])
    rel_l = float(ci.summary_df.loc["95% CI"]["Average"][2][1][:-1])

    abs = float(ci.summary_df.loc["Absolute Effect"]["Cumulative"])
    abs_u = float(ci.summary_df.loc["95% CI"]["Cumulative"][1][0])
    abs_l = float(ci.summary_df.loc["95% CI"]["Cumulative"][1][1])

    pred = ci.inferences["point_pred"]
    cum = ci.inferences['cum_effect']

    # estimate 80% CI from 95% using z-ratio
    abs_l_80 = abs_l + 0.35 * np.abs(abs_u-abs_l)
    abs_u_80 = abs_u - 0.35 * np.abs(abs_u-abs_l)
    rel_l_80 = rel_l + 0.35 * np.abs(rel_u-rel_l)
    rel_u_80 = rel_u - 0.35 * np.abs(rel_u-rel_l)

    if ci.inferences['cum_effect_lower'][-1] > ci.inferences['cum_effect_upper'][-1]:
        cum_u = ci.inferences['cum_effect_lower']
        cum_l = ci.inferences['cum_effect_upper']
    else:
        cum_l = ci.inferences['cum_effect_lower']
        cum_u = ci.inferences['cum_effect_upper']

    # estimate 80% CI from 95% using z-ratio
    cum_l_80 = cum_l + 0.35 * np.abs(cum_u-cum_l)
    cum_u_80 = cum_u - 0.35 * np.abs(cum_u-cum_l)

    # results class
    class res:
        def __init__(self, pval, rel, rel_u, rel_l, abs, abs_u, abs_l, pred, cum, cum_l, cum_u, rel_l_80, rel_u_80):
            self.pval, self.rel, self.rel_u, self.rel_l, self.abs, self.abs_u, \
                self.abs_l, self.pred, self.cum, self.cum_l, self.cum_u, self.rel_l_80, self.rel_u_80 =\
            pval, rel, rel_u, rel_l, abs, abs_u, abs_l, pred, cum, cum_l, cum_u, rel_l_80, rel_u_80

    return res(pval, rel, rel_u, rel_l, abs, abs_u, abs_l, pred, cum, cum_l, cum_u, rel_l_80, rel_u_80)

def add_effect(arr, up, test_size, scale_std=0.01, log_len=2):
    """
    Add a synthetic effect up on a time series arr 's last test_size values.

    Args:
        arr (array_like): time series array (must have len > test_size).
        up (float): mean value for the effect size.
        test_size (int): size of the post intervention perdiod.
        scale_std (float, optional): scale for the std of the nornal distribution. Defaults to 0.01.
        log_len (int, optional): length of the log smoother. Defaults to 5.

    Returns:
        np.array: original array with effect
    """

    # random uplift
    N = np.random.normal(loc = up, size=test_size, scale=np.mean(arr)*scale_std)
    # log smoother
    if log_len >= test_size:
        log_len = test_size
    if (log_len is not None) & (log_len != 0): 
        log_smooth = np.log(0.1*(np.arange(log_len)+1))
        log_smooth = scale_minmax(log_smooth)
        N[:log_len] *= log_smooth

    # effect
    effect = np.zeros(len(arr)) 
    effect[-test_size:] = N
    obs = arr + effect
    
    return obs 