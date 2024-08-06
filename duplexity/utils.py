import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Tuple, Union, Optional


def _to_numpy(data:Union[
                     np.array, 
                     xr.DataArray, 
                     pd.DataFrame, 
                     List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]
                     ]) -> np.array:
    """
    Convert input data to numpy array.
    
    Parameters
    ----------
    data : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Input data to be converted.
        
    Returns
    -------
    np.array
        Converted numpy array.
    """
    if isinstance(data, xr.DataArray):
        return data.to_numpy()
    elif isinstance(data, pd.DataFrame):
        return data.values
    elif isinstance(data, list):
        return np.array([d.to_numpy() if isinstance(d, xr.DataArray) else d.values for d in data])
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Supported data types are: np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]")
    

    


def _check_shapes(observed, output):
    """
    Check if the observed and output data have the same shape.

    Raises
    ------
    ValueError
        If the observed and output data do not have the same shape.
    """
    if observed.shape != output.shape:
        raise ValueError("Observed and output data must have the same shape.")
    

def _check_binary_data(data: np.array) -> bool:
    """
    Check if the given numpy array contains only binary data (0s and 1s).

    Parameters:
    data (np.array): Input numpy array.

    Returns:
    bool: True if the array contains only 0s and 1s, False otherwise.
    """
    # Check if all elements are either 0 or 1
    if data.ndim != 2:
        raise ValueError("Input data must contain only binary values (0s and 1s).")

def _check_2d_data(data: np.array) -> bool:
    """
    Check if the given numpy array is 2D.

    Parameters:
    data (np.array): Input numpy array.

    Returns:
    bool: True if the array is 2D, False otherwise.
    """
    if len(data.shape) != 2:
        raise ValueError("Input data must be 2D.")

def _binary_classification(data: np.array, threshold: float) -> np.array:
    """
    Perform binary classification on input data based on a threshold.

    Parameters
    ----------
    data : np.array
        Input data to be classified.
    threshold : float
        Threshold value for classification.

    Returns
    -------
    np.array
        Binary classified data.
    """
    binary_data = np.array(data >= threshold)
    return binary_data.astype(int)
    
