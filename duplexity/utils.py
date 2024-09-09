import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Tuple, Union, Optional, Callable, Dict


def _to_numpy(data: Union[
                     np.ndarray, 
                     xr.DataArray, 
                     xr.Dataset,  
                     pd.DataFrame, 
                     List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]
                     ], var: str = None) -> np.ndarray:  # Corrected return type
    """
    Convert input data to a numpy array.
    
    Parameters
    ----------
    data : Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Input data to be converted.
    var : str, optional
        The variable name to extract from an xr.Dataset. Only used if data is an xr.Dataset.
        
    Returns
    -------
    np.ndarray
        Converted numpy array.
    """
    if isinstance(data, xr.DataArray):
        return data.to_numpy()
    elif isinstance(data, xr.Dataset):
        if var:
            if var in data:
                return data[var].to_numpy()
            else:
                raise ValueError(f"Variable '{var}' not found in the dataset.")
        else:
            raise ValueError("No variable specified for xr.Dataset. Please specify a variable to convert.")
    elif isinstance(data, pd.DataFrame):
        return data.values
    elif isinstance(data, list):
        return np.array([_to_numpy(d, var=var) if isinstance(d, (xr.DataArray, xr.Dataset)) else d.values for d in data])
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Supported data types are: np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]")



    


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
    # Check if all elements are either 0 or 1
    if not np.all((data == 0) | (data == 1)):
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
    if np.issubdtype(data.dtype, np.bool_):
        binary_data = data
    else:
        binary_data = np.array(data >= threshold)
    return binary_data.astype(int)
    
    
