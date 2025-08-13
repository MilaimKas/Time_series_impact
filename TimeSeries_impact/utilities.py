# utilities.py
import numpy as np
import pandas as pd
from scipy.stats import entropy, norm


def scale_minmax(series: pd.Series, min: float = None, max: float = None) -> pd.Series:
    min_val = min if min is not None else series.min()
    max_val = max if max is not None else series.max()
    if max_val - min_val == 0:
        return series * 0  # Avoid division by zero
    return (series - min_val) / (max_val - min_val)


def scale_std(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0:
        return series * 0
    return (series - series.mean()) / std

def seasonal_default(n: int) -> int:
    """Return default seasonal window for STL as roughly n // 5 rounded to nearest odd number."""
    val = max(7, int(round(n / 5)))
    return val if val % 2 == 1 else val + 1


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Bhattacharyya distance between two distributions."""
    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)
    return -np.log(np.sum(np.sqrt(p * q)))


def normal_similarity_metrics(x: pd.Series, y: pd.Series) -> dict:
    """
    Assumes x and y are scaled residuals. Returns multiple similarity metrics assuming normality.
    """
    hist_x, bin_edges = np.histogram(x, bins=30, density=True)
    hist_y, _ = np.histogram(y, bins=bin_edges, density=True)

    # Add small value to avoid log(0) in KL divergence
    hist_x += 1e-10
    hist_y += 1e-10

    kl_div = entropy(hist_x, hist_y)
    js_div = 0.5 * entropy(hist_x, 0.5 * (hist_x + hist_y)) + 0.5 * entropy(hist_y, 0.5 * (hist_x + hist_y))
    bc = bhattacharyya_distance(hist_x, hist_y)
    mse = np.mean((x - y) ** 2)
    cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)

    return {
        "KL Divergence": kl_div,
        "Jensen-Shannon Divergence": js_div,
        "Bhattacharyya Distance": bc,
        "MSE": mse,
        "Cosine Similarity": cos_sim,
    }

def add_effect(arr, up, test_size, scale_std=2., log_len=5):
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
    N = np.random.normal(loc=up, scale=np.std(arr) * scale_std, size=test_size)
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