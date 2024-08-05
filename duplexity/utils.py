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
        raise ValueError("Unsupported data type")
    


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
    binary_data = data >= threshold
    return binary_data.astype(int)
    
