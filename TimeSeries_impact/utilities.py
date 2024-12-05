import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose, STL

from scipy.stats import ks_2samp 


def rolling_average(a):
    ret = np.cumsum(a)
    return ret / range(1, len(a)+1)
    
def scale_minmax(arr, min=None, max=None):
    if min is None:
        min = arr.min()
    if max is None:
        max = arr.max()
    arr_res = (arr-arr.min())/(arr.max()-arr.min())
    return arr_res

def scale_std(arr):
    arr_res = (arr-arr.mean())/arr.std()
    return arr_res

# Bhattacharyya Distance
def bhattacharyya_distance(mu1, sigma1, mu2, sigma2):
    term1 = 0.25 * np.log(0.25 * ((sigma1**2 / sigma2**2) + (sigma2**2 / sigma1**2) + 2))
    term2 = 0.25 * (((mu1 - mu2)**2) / (sigma1**2 + sigma2**2))
    return term1 + term2

# Kullback-Leibler Divergence
def kullback_leibler_divergence(mu1, sigma1, mu2, sigma2):
    return np.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

# Hellinger Distance
def hellinger_distance(mu1, sigma1, mu2, sigma2):
    term1 = (2 * sigma1 * sigma2) / (sigma1**2 + sigma2**2)
    term2 = np.exp(-0.25 * ((mu1 - mu2)**2) / (sigma1**2 + sigma2**2))
    return np.sqrt(1 - np.sqrt(term1) * term2)

# Wasserstein Distance
def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    return np.sqrt((mu1 - mu2)**2 + (sigma1 - sigma2)**2)

def kolmogorov_test(dist1, dist2):
    return ks_2samp(dist1, dist2)[0]

def normal_similarity_metrics(dist1, dist2):
    
    mu1  = np.mean(dist1) 
    mu2  = np.mean(dist2)
    sigma1 = np.std(dist1)
    sigma2 = np.std(dist2)

    res = pd.DataFrame()

    res["Bhattacharyya Distance"] = [bhattacharyya_distance(mu1, sigma1, mu2, sigma2)]
    res["Kullback-Leibler Divergence"] = [kullback_leibler_divergence(mu1, sigma1, mu2, sigma2)]
    res["Hellinger Distance"] = [hellinger_distance(mu1, sigma1, mu2, sigma2)]
    res["Wasserstein Distance"] = [wasserstein_distance(mu1, sigma1, mu2, sigma2)]
    res["Kolmogorov test"] = [kolmogorov_test(dist1, dist2)]

    return res


def seasonal_default(length):
    """
    Calculate default seasonal window smoothing

    Args:
        length (int): length of the data set

    Returns:
        _type_: _description_
    """
    smooth_default = 1+length//5 if (length//5)%2 == 0 else (length//5)
    return max(smooth_default, 3)

def decompose_components(df, **kwargs):
    """
    perform components decmposition on all columns of a dataframe

    Args:
        df (pd.DataFrame): 
    """
    trend_data = {}
    seas_data = {}
    resid_data = {}

    for column in df.columns:
        stl = STL(df[column], **kwargs).fit()
        trend_data[column] = stl.trend
        seas_data[column] = stl.seasonal
        resid_data[column] = stl.resid

    trend_df = pd.DataFrame(trend_data)
    seas_df = pd.DataFrame(seas_data)
    resid_df = pd.DataFrame(resid_data)

    return trend_df, seas_df, resid_df

def store_ci(ci):
    """
    Extract results from causalimpact object and store them in the res class

    Args:
        ci (_object_): casualimpact object

    Returns:
        class: res with relative and absolute effect values
    """

    pval = float(ci.summary_df.loc["P-value"]["Average"][:-1])

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

def add_effect(arr, up, test_size, scale_std=0.05, log_len=5):
    """
    Adds a random uplift effect to the end of an array.
    
    Parameters:
        arr (numpy.ndarray): Input array to which the effect is added.
        up (float): Mean of the normal distribution for the uplift effect.
        test_size (int): Number of elements at the end of the array to apply the effect.
        scale_std (float): Scaling factor for the standard deviation of the normal distribution.
        log_len (int): Number of initial elements of the uplift effect to apply logarithmic smoothing.
        
    Returns:
        numpy.ndarray: Modified array with the uplift effect added.
    """
    
    # Validate parameters
    if test_size <= 0:
        raise ValueError("test_size must be a positive integer.")
    if log_len < 0:
        raise ValueError("log_len must be a non-negative integer.")
    elif (log_len < 3) & (log_len > 0):
        print("WARNING: it makes little sense to put a log smoothing curve length smaller than 3 timestamps\
                    it will put the effect on the first day of intervention to 0")
    
    # Generate random uplift
    N = np.random.normal(loc=up, scale=np.mean(arr) * scale_std, size=test_size)
    # Apply logarithmic smoothing
    if log_len > 0:
        log_len = min(log_len, test_size)
        log_smooth = np.log(0.1 * (np.arange(log_len) + 1))
        log_smooth = scale_minmax(log_smooth)
        N[:log_len] *= log_smooth
        
    # Create the effect array
    effect = np.zeros(len(arr))
    effect[-test_size:] = N
    
    # Add effect to the original array
    obs = arr + effect
    
    return obs 


def _store_ci(ci):
    """
    using tfcausalimpact
    """

    pval = ci.p_value

    rel = float(ci.summary_data.loc["rel_effect"]["average"])
    rel_l = float(ci.summary_data.loc["rel_effect_lower"]["average"])
    rel_u = float(ci.summary_data.loc["rel_effect_upper"]["average"])

    abs = float(ci.summary_data.loc["abs_effect"]["cumulative"])
    abs_l = float(ci.summary_data.loc["abs_effect_lower"]["cumulative"])
    abs_u = float(ci.summary_data.loc["abs_effect_upper"]["cumulative"])

    pred = ci.inferences.complete_preds_means
    cum = ci.inferences.post_cum_effects_means

    if ci.inferences.post_cum_effects_lower.iloc[-1] > ci.inferences.post_cum_effects_upper.iloc[-1]:
        cum_u = ci.inferences.post_cum_effects_lower
        cum_l = ci.inferences.post_cum_effects_upper
    else:
        cum_l = ci.inferences.post_cum_effects_lower
        cum_u = ci.inferences.post_cum_effects_upper

    # estimate 80% CI from 95% using z-ratio
    abs_l_80 = abs_l + 0.35 * np.abs(abs_u-abs_l)
    abs_u_80 = abs_u - 0.35 * np.abs(abs_u-abs_l)
    rel_l_80 = rel_l + 0.35 * np.abs(rel_u-rel_l)
    rel_u_80 = rel_u - 0.35 * np.abs(rel_u-rel_l)

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


def make_time_serie(N, sig=[2,1,5], trend_line_coeff=[10,-3, 8], trend_line_interc=[100, 70, 20], trend_list=[None, None, None], 
                    amp=[[5], [2], [3]], freq=[7], 
                    nbr_rand_event=0, rand_event_str=0.2):
    """
    Construct a "observed" (actuals) and "control" time serie using the following components:
    - linear trend and level (intercept) or given trend
    - up to 2 seanonality (model by cosinus functions) define by frequence and amplitudes
    - noise term
    - random events

    Args:
        N (int): number of date points.
        sig (list, optional): standard deviation for the noise terms for obs and control. Defaults to [2,1].
        trend_line_coeff (list, optional): slope of the trend for obs and control. Defaults to [10,-3] (control has a negative trend).
        trend_line_interc (list, optional): intercept for the trend component for obs and control. Defaults to [100, 70].
        amp (list, optional): amplitude of the sesonality for obs and control. Defaults to [[5,5], [2,3]] ([[obs_seas1, obs_seas2],[con_seas1, con_seas2]]).
        freq (list, optional): frequence of the seasonality component for obs and control. Defaults to [7,30].
        nbr_rand_event (int, optional): number of random event not taking account by the components. Defaults to 0.

    Returns:
        dict: dictionary with the data
    """

    x = np.arange(N)

    # cehck inputs
    n_ts = len(sig)
    n_controls = n_ts-1
    trend_list = np.array(trend_list)

    if len(trend_line_coeff) != n_ts:
        raise ValueError("trend line coefficients must be given for each time series")
    if len(trend_line_interc) != n_ts:
        raise ValueError("trend line intercept must be given for each time series")
    if len(trend_list) != n_ts:
        raise ValueError("Given trend component must be given for each time series")
    if len(amp) != n_ts:
        raise ValueError("amplitude(s) must be given fro each time series")
    if len(amp[0]) != len(freq):
        raise ValueError("length of amplitude for each time series must be the same as length freq")
    
    # base line: either linear or given in trend_list
    # -------------------------------------------------
    
    # for target
    if trend_list[0] is None:
        base_line_target = trend_line_interc[0] + x*(trend_line_coeff[0]/x.max())
    elif len(trend_list[0]) != N:
        print("WARNING: the given time serie length ({}) deviates from the length of the given trend ({}). \
                The final length will be taken from the given trend array.".format(len(trend_list[0]), N))
        N = len(trend_list[0])
        base_line_target = trend_list[0].copy()

    # check controls
    base_line_controls = []
    for i in range(n_controls):

        # check if control trend are compatible
        if trend_list[1+i] is not None:
            if (len(trend_list[0]) != len(trend_list[1+i])):
                raise ValueError("Given trends must be the same length")
            #else:
            #    corr = pearsonr(trend_list[1], trend_list[0])[0]
            #    print("Correlation between trend is {:.2f}". format(corr))
            #    if abs(corr) < 0.3:
            #        print("WARNING: the trend for the time serie is only weakly correlated to the trend for the control variable")
            base_line_controls.append(trend_list[1+i])

        else:
            base_line_controls.append(trend_line_interc[1+i] + x*(trend_line_coeff[1+i]/x.max()))



    # add random event with certain duration
    # -------------------------------------------------

    if nbr_rand_event != 0:
        length_rand = np.random.randint(5, 10, size=nbr_rand_event) # duration
        start_rand = np.random.randint(N, size=nbr_rand_event-1) #start date
        rand_event_pos = []

        controls_random_events_idx = np.random.randint(0, len(base_line_controls), size=nbr_rand_event)
        for st, d, control_idx in zip(start_rand, length_rand, controls_random_events_idx):
            str_sign = np.random.choice([-1,1])
            str_event = np.mean(base_line_target)*rand_event_str*np.random.random()
            base_line_target[st: st+d] = base_line_target[st: st+d]+str_event*str_sign
            # distribute random events in controls
            str_event = np.mean(base_line_controls[control_idx])*rand_event_str*np.random.random()
            base_line_controls[control_idx][st: st+d] += str_event*str_sign
            rand_event_pos.append([st,st+d])
    else:
        rand_event_pos = None

    # add seasonality and noise
    # -------------------------------------------------w

    season_component_list = []
    season_component = np.zeros(len(x))
    observation_target = base_line_target.copy()
    # observation (target)
    for a, f in zip(amp[0], freq):
        season = a*np.cos(x*2*np.pi/(f))
        observation_target += season
        season_component += season
    season_component_list.append(season_component)
    observation_target += np.random.normal(loc=0, scale=sig[0], size=N)

    base_list = [base_line_target]

    # control
    observation_controls = []
    season_component = np.zeros(len(x))
    for c_i in range(n_controls):
        control_i = base_line_controls[c_i].copy()
        for a, f in zip(amp[1+c_i], freq):
            season = a*np.cos(x*2*np.pi/(f))
            control_i += season
            season_component += season
        season_component_list.append(season_component)
        control_i += np.random.normal(loc=0, scale=sig[1+c_i], size=N)
        observation_controls.append(control_i)
        base_list.append(base_line_controls[c_i])

    return {"obs":observation_target, "control":observation_controls, "rand_event":rand_event_pos, 
                "season":season_component_list , "base":base_list}