import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from scipy.ndimage import uniform_filter
import xarray as xr
from typing import Union

from typing import Union
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter


def crps_ensemble(observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray]) -> np.array:
    """
    Compute the CRPS for a set of forecast ensembles and the corresponding observations.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray]
        Array of shape (w, h) containing the observed values.
    output : Union[np.array, xr.DataArray]
        Array of shape (n, w, h) containing the values from an ensemble
        forecast of n members with m * n observations.

    Returns
    -------
    np.array
        The CRPS values.
    """
    # checks
    if observed.shape != output.shape[1:]:
        raise ValueError(
            "The shape of output does not match the shape of observed %s != %s"
            % (output.shape[1:], observed.shape)
        )
    
    output = np.vstack([output[i, :].flatten() for i in range(output.shape[0])]).T
    observed = observed.flatten()

    
    # Create a mask to remove NaN values from the observed and output arrays
    na_mask = np.logical_and(np.all(np.isfinite(output), axis = 1), np.isfinite(observed))
    
    # Remove NaN values
    output = output[na_mask, :].copy()

    # Sort each ensemble member in-place
    opt = output.copy()
    opt.sort(axis=1)
    obs = observed.copy()

    opt_below = opt < obs[..., None]
    crps = np.zeros_like(obs)

    for i in range(opt.shape[-1]):
        below = opt_below[..., i]
        weight = ((i + 1) ** 2 - i ** 2) / opt.shape[-1] ** 2
        crps[below] += weight * (obs[below] - opt[..., i][below])
    
    for i in range(opt.shape[-1] - 1, -1, -1):
        above = ~opt_below[..., i]
        k = opt.shape[-1] - 1 - i
        weight = ((k + 1) ** 2 - k ** 2) / opt.shape[-1] ** 2
        crps[above] += weight * (opt[..., i][above] - obs[above])

    return crps



def ROC_AUC(observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray]) -> np.array:
    """    
    Compute the ROC-AUC score for a set of forecast ensembles and the corresponding observations.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray]
        Array of shape (w, h) containing the observed values.
    output : Union[np.array, xr.DataArray]
        Array of shape (n, w, h) containing the values from an ensemble
        forecast of n members with m * n observations.

    Returns
    -------
    np.array
        The ROC-AUC values.
    """

    # checks
    if observed.shape != output.shape[1:]:
        raise ValueError(
            "The shape of output does not match the shape of observed %s != %s"
            % (output.shape[1:], observed.shape)
        )
    
    output = np.vstack([output[i, :].flatten() for i in range(output.shape[0])]).T
    observed = observed.flatten()

    # Create a mask to remove NaN values from the observed and output arrays
    na_mask = np.logical_and(np.all(np.isfinite(output), axis = 1), np.isfinite(observed))

    # Remove NaN values
    output = output[na_mask, :].copy()

    # Compute the ROC-AUC score
    fpr, tpr, _ = roc_curve(observed, output.mean(axis=1))
    roc_auc = auc(fpr, tpr)

    return roc_auc



def spectral_entropy(observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray], fs: int = 1) -> np.array:
    """
    Compute the spectral entropy for a set of forecast ensembles and the corresponding observations.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray]
        Array of shape (w, h) containing the observed values.
    output : Union[np.array, xr.DataArray]
        Array of shape (n, w, h) containing the values from an ensemble
        forecast of n members with m * n observations.
    fs : int
        The sampling frequency of the data.

    Returns
    -------
    np.array
        The spectral entropy values.
    """
    # checks
    if observed.shape != output.shape[1:]:
        raise ValueError(
            "The shape of output does not match the shape of observed %s != %s"
            % (output.shape[1:], observed.shape)
        )
    
    output = np.vstack([output[i, :].flatten() for i in range(output.shape[0])]).T
    observed = observed.flatten()

    # Create a mask to remove NaN values from the observed and output arrays
    na_mask = np.logical_and(np.all(np.isfinite(output), axis = 1), np.isfinite(observed))

    # Remove NaN values
    output = output[na_mask, :].copy()

    # Compute the spectral entropy
    f, Pxx = welch(output, fs = fs, nperseg = 256, axis = 0)
    Pxx = Pxx / Pxx.sum(axis = 0)
    spectral_entropy = -np.sum(Pxx * np.log2(Pxx), axis = 0)

    return spectral_entropy


def sharpness(observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray]) -> np.array:
    """
    Compute the sharpness for a set of forecast ensembles and the corresponding observations.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray]
        Array of shape (w, h) containing the observed values.
    output : Union[np.array, xr.DataArray]
        Array of shape (n, w, h) containing the values from an ensemble
        forecast of n members with m * n observations.

    Returns
    -------
    np.array
        The sharpness values.
    """
    # checks
    if observed.shape != output.shape[1:]:
        raise ValueError(
            "The shape of output does not match the shape of observed %s != %s"
            % (output.shape[1:], observed.shape)
        )
    
    output = np.vstack([output[i, :].flatten() for i in range(output.shape[0])]).T
    observed = observed.flatten()

    # Create a mask to remove NaN values from the observed and output arrays
    na_mask = np.logical_and(np.all(np.isfinite(output), axis = 1), np.isfinite(observed))

    # Remove NaN values
    output = output[na_mask, :].copy()

    # Compute the sharpness
    sharpness = output.std(axis = 1)

    return sharpness


def reliability(observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray]) -> np.array:
    """
    Compute the reliability for a set of forecast ensembles and the corresponding observations.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray]
        Array of shape (w, h) containing the observed values.
    output : Union[np.array, xr.DataArray]
        Array of shape (n, w, h) containing the values from an ensemble
        forecast of n members with m * n observations.

    Returns
    -------
    np.array
        The reliability values.
    """
    # checks
    if observed.shape != output.shape[1:]:
        raise ValueError(
            "The shape of output does not match the shape of observed %s != %s"
            % (output.shape[1:], observed.shape)
        )
    
    output = np.vstack([output[i, :].flatten() for i in range(output.shape[0])]).T
    observed = observed.flatten()

    # Create a mask to remove NaN values from the observed and output arrays
    na_mask = np.logical_and(np.all(np.isfinite(output), axis = 1), np.isfinite(observed))

    # Remove NaN values
    output = output[na_mask, :].copy()

    # Compute the reliability
    observed = np.tile(observed, (output.shape[0], 1))
    observed = observed.flatten()
    output = output.flatten()
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(output, bins)
    bin_observed = np.zeros((len(bins), len(bins)))
    for i in range(len(bins)):
        for j in range(len(bins)):
            bin_observed[i, j] = np.mean(observed[bin_indices == i] == j)
    reliability = np.abs(bin_observed - bins)

    return reliability


def resolution(observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray]) -> np.array:
    """
    Compute the resolution for a set of forecast ensembles and the corresponding observations.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray]
        Array of shape (w, h) containing the observed values.
    output : Union[np.array, xr.DataArray]
        Array of shape (n, w, h) containing the values from an ensemble
        forecast of n members with m * n observations.

    Returns
    -------
    np.array
        The resolution values.
    """
    # checks
    if observed.shape != output.shape[1:]:
        raise ValueError(
            "The shape of output does not match the shape of observed %s != %s"
            % (output.shape[1:], observed.shape)
        )
    
    output = np.vstack([output[i, :].flatten() for i in range(output.shape[0])]).T
    observed = observed.flatten()

    # Create a mask to remove NaN values from the observed and output arrays
    na_mask = np.logical_and(np.all(np.isfinite(output), axis = 1), np.isfinite(observed))

    # Remove NaN values
    output = output[na_mask, :].copy()

    # Compute the resolution
    observed = np.tile(observed, (output.shape[0], 1))
    observed = observed.flatten()
    output = output.flatten()
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(output, bins)
    bin_observed = np.zeros((len(bins), len(bins)))
    for i in range(len(bins)):
        for j in range(len(bins)):
            bin_observed[i, j] = np.mean(observed[bin_indices == i] == j)
    resolution = np.abs(bin_observed - bin_observed.mean(axis = 0))

    return resolution



def brier_score(observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray]) -> np.array:
    """
    Compute the Brier score for a set of forecast ensembles and the corresponding observations.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray]
        Array of shape (w, h) containing the observed values.
    output : Union[np.array, xr.DataArray]
        Array of shape (n, w, h) containing the values from an ensemble
        forecast of n members with m * n observations.

    Returns
    -------
    np.array
        The Brier score values.
    """

    # checks
    if observed.shape != output.shape[1:]:
        raise ValueError(
            "The shape of output does not match the shape of observed %s != %s"
            % (output.shape[1:], observed.shape)
        )
    
    output = np.vstack([output[i, :].flatten() for i in range(output.shape[0])]).T
    observed = observed.flatten()

    # Create a mask to remove NaN values from the observed and output arrays
    na_mask = np.logical_and(np.all(np.isfinite(output), axis = 1), np.isfinite(observed))

    # Remove NaN values
    output = output[na_mask, :].copy()

    # Compute the Brier score
    brier_score = np.mean((output.mean(axis = 1) - observed) ** 2)

    return brier_score




def fss_init(threshold, scale):
    """Initialize a fractions skill score (FSS) verification object."""
    fss = dict(threshold=threshold, scale=scale, sum_fct_sq=0.0, sum_fct_obs=0.0, sum_obs_sq=0.0)
    return fss

def calculate_bp(data, threshold):
    """
    Calculate the Binary Predictor (BP) for the given data and threshold.
    
    Parameters:
    data (np.ndarray): Array of data (observed or forecasted).
    threshold (float): Threshold value for binarizing the data.
    
    Returns:
    np.ndarray: Binary predictor array.
    """
    return (data >= threshold).astype(np.single)

def calculate_np(bp, scale):
    """
    Calculate the Neighborhood Predictor (NP) for the given Binary Predictor (BP) and neighborhood size.
    
    Parameters:
    bp (np.ndarray): Binary predictor array.
    scale (int): Size of the neighborhood for calculating fractions.
    
    Returns:
    np.ndarray: Neighborhood predictor array.
    """
    if scale > 1:
        n_bp = uniform_filter(bp, size=scale, mode='constant', cval=0.0)
    else:
        n_bp = bp
    return n_bp

def calculate_fbs(np_f, np_o):
    """
    Calculate the Fractions Brier Score (FBS).
    
    Parameters:
    np_f (np.ndarray): Neighborhood predictor for the forecasted data.
    np_o (np.ndarray): Neighborhood predictor for the observed data.
    
    Returns:
    float: Fractions Brier Score.
    """
    return np.mean((np_f - np_o) ** 2)

def calculate_wfbs(np_f, np_o):
    """
    Calculate the Weighted Fractions Brier Score (WFBS).
    
    Parameters:
    np_f (np.ndarray): Neighborhood predictor for the forecasted data.
    np_o (np.ndarray): Neighborhood predictor for the observed data.
    
    Returns:
    float: Weighted Fractions Brier Score.
    """
    return np.mean(np_f ** 2) + np.mean(np_o ** 2)

def fss_update(fss, forecast, observed):
    """
    Update the FSS object with new forecast and observed data.
    
    Parameters:
    fss (dict): FSS object.
    forecast (np.ndarray): Forecasted data.
    observed (np.ndarray): Observed data.
    """
    threshold = fss['threshold']
    scale = fss['scale']

    bp_f = calculate_bp(forecast, threshold)
    bp_o = calculate_bp(observed, threshold)
    
    np_f = calculate_np(bp_f, scale)
    np_o = calculate_np(bp_o, scale)

    fss['sum_fct_sq'] += np.sum(np_f ** 2)
    fss['sum_obs_sq'] += np.sum(np_o ** 2)
    fss['sum_fct_obs'] += np.sum(np_f * np_o)

def fss_compute(fss):
    """
    Calculate the Fractions Skill Score (FSS)
    
    Parameters:
    fss (dict): FSS object.
    
    Returns:
    float: Fractions Skill Score.
    """
    sum_fct_sq = fss['sum_fct_sq']
    sum_obs_sq = fss['sum_obs_sq']
    sum_fct_obs = fss['sum_fct_obs']

    fbs = sum_fct_sq + sum_obs_sq - 2 * sum_fct_obs
    wfbs = sum_fct_sq + sum_obs_sq

    if wfbs == 0:
        return np.nan

    fss_value = 1 - fbs / wfbs
    return fss_value

