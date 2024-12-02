import numpy as np
from typing import Union, Callable, Tuple, List, Dict
from pixelwise import *
import numpy as np
import xarray as xr
import pandas as pd
from typing import Union, Callable, Dict, List, Tuple
from duplexity.pixelwise import *
from duplexity.utils import _to_numpy, _check_shapes,_binary_classification
from duplexity.get_metric import _get_pixelwise_function



data_type = Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]
    # Define available metrics




def ensemble_metric(observed: data_type,
                    ensemble_output: data_type,
                    metric: str,
                    mean: bool = False,
                    **kwargs
                    ) -> Union[np.ndarray, dict]:

                                 
    """
    Calculate the specified metric for each ensemble member against the observed data.


    """
    # Convert inputs to numpy arrays if they are not already
    if isinstance(observed, (xr.DataArray, xr.Dataset, pd.DataFrame)):
        observed = observed.to_numpy()
    
    if isinstance(ensemble_output, (xr.DataArray, xr.Dataset, pd.DataFrame)):
        ensemble_output = ensemble_output.to_numpy()


    # Ensure the ensemble data has 3 dimensions
    if ensemble_output.ndim != 3:
        raise ValueError(f"The input ensemble_data should have 3 dimensions (n_members, h, w), but got {ensemble_output.ndim} dimensions.")
    
    _check_shapes(ensemble_output[0], observed)  # Ensure each member has the same shape as observed

    metric = metric.lower()  # Convert metric name to lowercase to handle case insensitivity
    # Check if metric_func is provided, default to None (for all metrics)
    if metric is None:
        raise ValueError("At least one metric function must be provided.")
    elif metric in ['cm', 'precision', 'recall', 'f1', 'accuracy', 'csi', 'far', 'pod', 'gss', 'hss', 'pss', 'sedi']:
        # Get the corresponding metric function using _get_pixelwise_function
        metric_function = _get_pixelwise_function(metric)
        # Apply categorical metrics with threshold
        result = np.array([metric_function(observed, output_member, threshold=threshold, **kwargs) 
                         for output_member in ensemble_output])
    elif metric in ['mae', 'mse', 'rmse', 'bias', 'drmse', 'corr']:
        # Get the corresponding metric function using _get_pixelwise_function
        metric_function = _get_pixelwise_function(metric)
        # Apply continuous metrics
        result = np.array([metric_function(observed, output_member, **kwargs) 
                         for output_member in ensemble_output])   

    
    # If 'mean' is True, calculate the mean of the ensemble output
    if mean:
        result = np.mean(result)
    else:
        result = result

    return result



















