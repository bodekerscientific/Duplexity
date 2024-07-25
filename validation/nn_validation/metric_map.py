import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from typing import Union
from deterministic import ContinuousScore, CategoricalScore



def initialize_metrics(shape: tuple) -> np.array:
    """
    Initialize metric arrays with the given shape.

    Parameters:
    shape (tuple): Shape of the metric arrays.

    Returns:
    np.array: Initialized metric arrays.
    """
    return np.zeros(shape)

def update_metrics(metric_array: np.array, observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray], metric_function: callable) -> np.array:
    """
    Update metric arrays for each grid point
    
    Parameters:
    metric_array (np.array): Array containing metric values.
    observed (Union[np.array, xr.DataArray]): Array containing the observed values.
    output (Union[np.array, xr.DataArray]): Array containing the output values.
    metric_function (callable): Function to calculate the metric.

    Returns:
    np.array: Updated metric array.
    """
    # Check shapes of metric arrays before updating
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            result = metric_function(observed[i, j], output[i, j])
            metric_array[i, j] += result if not np.isnan(result) else 0
    return metric_array


def calculate_metrics_map(observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray]) -> np.array:
    """
    Calculate specified metrics for each grid point.
    
    Parameters:
    observed (Union[np.array, xr.DataArray]): Array containing the observed values.
    output (Union[np.array, xr.DataArray]): Array containing the output values.
    
    Returns:
    dict: A dictionary with metric names as keys and metric arrays as values.
    """

    observed = observed.as_numpy() if isinstance(observed, xr.DataArray) else observed
    output = output.as_numpy() if isinstance(output, xr.DataArray) else output

    continuous_score = ContinuousScore()

    shape = observed.shape
    metrics_result = {}

    # Initialize metric arrays
    metrics_result["MAE"] = initialize_metrics(shape)
    metrics_result["MSE"] = initialize_metrics(shape)
    metrics_result["RMSE"] = initialize_metrics(shape)
    metrics_result["Bias"] = initialize_metrics(shape)

    # Update metrics for each grid point
    metrics_result["MAE"] = update_metrics(metrics_result["MAE"], observed, output, continuous_score.mean_absolute_error)
    metrics_result["MSE"] = update_metrics(metrics_result["MSE"], observed, output, continuous_score.mean_squared_error)
    metrics_result["RMSE"] = update_metrics(metrics_result["RMSE"], observed, output, continuous_score.root_mean_squared_error)
    metrics_result["Bias"] = update_metrics(metrics_result["Bias"], observed, output, continuous_score.bias)


    return metrics_result

