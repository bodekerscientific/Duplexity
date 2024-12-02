"""
Probabilistic
===============================

.. automodule:: duplexity.probabilistic
    :members:
    :undoc-members:
    :show-inheritance:

.. autosummary::
    :toctree: ../generated/


    
"""




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



def CRPS(observed: Union[np.array, xr.DataArray], ensemble_output: Union[np.array, xr.DataArray]) -> float:
    """
    Compute the continuous ranked probability score(CRPS) for a set of forecast ensembles and the corresponding observations.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray]
        Array of shape (w, h) containing the observed values.
    ensemble_output : Union[np.array, xr.DataArray]
        Array of shape (n, w, h) containing the values from an ensemble
        model output data of n members with m * n observations.

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


