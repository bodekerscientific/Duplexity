import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
from duplexity.deterministic_score import DeterministicScore
from typing import Union, Callable, Dict, List
#import seaborn as sns


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
    assert output.shape == observed.shape

    if output.ndim == 3:
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                for k in range(output.shape[2]):
                    result = metric_function(observed[i, j, k], output[i, j, k])
                    metric_array[i, j, k] += result if not np.isnan(result) else 0
        return np.mean(metric_array, axis=0)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            result = metric_function(observed[i, j], output[i, j])
            metric_array[i, j] += result if not np.isnan(result) else 0
    return metric_array

def _to_numpy(data: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> np.array:
    """
    Convert input data to numpy array.
    
    Parameters:
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
        return np.array([d.to_numpy() if isinstance(d, xr.DataArray) else (d.values if isinstance(d, pd.DataFrame) else d) for d in data])
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise ValueError("Unsupported data type")


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
                 ]
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
    det_score = DeterministicScore(observed=observed, output=output)



    shape = observed.shape
    metrics_result = {}

    # Initialize metric arrays
    metrics_result["MAE"] = initialize_metrics(shape)
    metrics_result["MSE"] = initialize_metrics(shape)
    metrics_result["RMSE"] = initialize_metrics(shape)
    metrics_result["Bias"] = initialize_metrics(shape)

    # Update metrics for each grid point
    metrics_result["MAE"] = update_metrics(metrics_result["MAE"], observed, output, det_score.mae)
    metrics_result["MSE"] = update_metrics(metrics_result["MSE"], observed, output, det_score.mse)
    metrics_result["RMSE"] = update_metrics(metrics_result["RMSE"], observed, output, det_score.rmse)
    metrics_result["Bias"] = update_metrics(metrics_result["Bias"], observed, output, det_score.bias)


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