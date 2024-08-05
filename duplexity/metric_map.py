import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from duplexity.deterministic_score import mean_absolute_error, mean_squared_error, bias, pearson_correlation, root_mean_squared_error
from typing import Union, Callable, Dict, List, Tuple
from duplexity.utils import _to_numpy, _check_shapes


def initialize_metrics(shape: tuple) -> np.array:
    """
    Initialize metric arrays with the given shape.

    Parameters:
    shape (tuple): Shape of the metric arrays.

    Returns:
    np.array: Initialized metric arrays.
    """
    return np.zeros(shape)

def update_metrics(metric_array: np.array, observed:np.array, output:np.array, metric_function: Callable) -> np.array:
    """
    Update metric arrays for each grid point
    
    Parameters:
    metric_array (np.array): Array containing metric values.
    observed (Union[np.array, xr.DataArray]): Array containing the observed values.
    output (Union[np.array, xr.DataArray]): Array containing the output values.
    metric_function (Callable): Function to calculate the metric.

    Returns:
    np.array: Updated metric array.
    """
    # Check shapes of metric arrays before updating
    _check_shapes(observed, output)

    if output.ndim == 3:
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                if np.all(output == output[0]) or np.isnan(output).any():
                    continue
                result = metric_function(observed[:,i, j], output[:,i, j])
                metric_array[i, j] += result if not np.isnan(result) else 0


    elif output.ndim == 2:
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                result = metric_function(observed[i, j], output[i, j])
                metric_array[i, j] += result if not np.isnan(result) else 0
    return metric_array

def grid_point_calculate(observed: Union[
                     np.array, 
                     xr.DataArray, 
                     pd.DataFrame, 
                     List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]
                 ], 
                   output: Union[
                     np.array, 
                     xr.DataArray, 
                     pd.DataFrame, 
                     List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]
                 ],
                 metrics: Union[str, Tuple[str], List[str]] = None
                 ) -> dict:

    """
    Calculate specified metrics for each grid point.
    
    Parameters:
    observed (Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]): 
        Array containing the observed values.
    output (Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]):
        Array containing the output values.
    
    
    Returns:
    dict: A dictionary with metric names as keys and metric arrays as values.
    """

    observed = _to_numpy(observed)
    output = _to_numpy(output)

    if observed.ndim == 2:
        shape = observed.shape
    elif observed.ndim == 3:
        shape = observed.shape[1:]
    else:
        raise ValueError("Unsupported number of dimensions for observed data")
    
    available_metrics = {
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": root_mean_squared_error,
        "Bias": bias
    }

    if metrics is None:
        selected_metrics = available_metrics.keys()
    else:
        if isinstance(metrics, str):
            selected_metrics = [metrics]
        else:
            selected_metrics = metrics

    # Initialize metric arrays
    metrics_result = {metric: initialize_metrics(shape) for metric in selected_metrics}



    for metric in selected_metrics:
        if metric in available_metrics:
            metrics_result[metric] = update_metrics(metrics_result[metric], observed, output, available_metrics[metric])
        else:
            raise ValueError(f"Metric '{metric}' is not recognized. Available metrics are: {list(available_metrics.keys())}")



    return metrics_result

 
def plot_metrics_map(metrics: dict, metric_name: str, title: str, save_path: str = None, vminvmax: tuple = None, camp: str = 'viridis', land_mask: np.array = None):
    """
    Plot the given metric map.
    
    Parameters:
    metrics (dict): 
        A dictionary with metric names as keys and metric arrays as values.
    metric_name (str): 
        Name of the metric to plot.
    title (str): 
        Title of the plot.
    save_path (str): 
        Path to save the plot.
    vminvmax (tuple): 
        Tuple containing the minimum and maximum values for the colormap.
    camp (str): 
        Colormap to use.
    land_mask (np.array):
        Array containing land mask.
    """
    metric = metrics[metric_name] * land_mask if land_mask is not None else metrics[metric_name]

    plt.imshow(metric, cmap='viridis', vmin=vminvmax[0], vmax=vminvmax[1])
    plt.colorbar()
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.show()



#def plot_metrics_map_plotly(metrics: Dict[str, np.array], metric_name: str, title: str, save_path: str = None):
#    """
#    Plot the given metric map using plotly.
#    
#    Parameters:
#    metrics (Dict[str, np.array]): A dictionary with metric names as keys and metric arrays as values.
#    metric_name (str): Name of the metric to plot.
#    title (str): Title of the plot.
#    save_path (str): Path to save the plot.
#    """
#    metric = metrics[metric_name]
#    fig = go.Figure(data=go.Heatmap(z=metric, colorscale='Viridis'))
#    fig.update_layout(title=title)
#    if save_path:
#        fig.write_image(save_path)
#    fig.show()
#


#def plot_metrics_map_seaborn(metrics: Dict[str, np.array], metric_name: str, title: str, save_path: str = None):
#    """
#    Plot the given metric map using seaborn.
#    
#    Parameters:
#    metrics (Dict[str, np.array]): A dictionary with metric names as keys and metric arrays as values.
#    metric_name (str): Name of the metric to plot.
#    title (str): Title of the plot.
#    save_path (str): Path to save the plot.
#    """
#    metric = metrics[metric_name]
#    plt.figure(figsize=(10, 8))
#    sns.heatmap(metric, cmap='viridis', cbar=True)
#    plt.title(title)
#    if save_path:
#        plt.savefig(save_path)
#    plt.show()
#