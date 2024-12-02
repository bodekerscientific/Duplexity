"""
Plotting Utilities
==================

This module contains utility functions for plotting metrics and results.

.. autosummary::
    :toctree: ../generated/

    plot_fss
    plot_rapsd


"""



import os
import numpy as np
import xarray as xr
import pandas as pd
from duplexity.pixelwise import mean_absolute_error, mean_squared_error, bias, pearson_correlation, root_mean_squared_error
from typing import Union, Callable, Dict, List, Tuple
from duplexity.utils import _to_numpy, _check_shapes
import matplotlib.pyplot as plt
from typing import List, Dict, Union



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


"""
FSS Plotting Utilities
----------------------

This module contains utility functions for plotting Fractions Skill Score (FSS) results.

.. autosummary::

    plot_fss
"""


def plot_fss(fss_results_all: Dict[Union[float, int], List[float]],
                               scales: Union[List[int], range],
                               title: str =  None,
                               xlabel: str = None,
                               ylabel: str = None,
                               figsize: tuple = (8, 5),
                               grid: bool = True,
                               save_plot: bool = False,
                               save_path: str = None) -> None:
    """
    Plot multiple FSS results for different thresholds across multiple datasets in one plot,
    with each line clearly labeled.

    Parameters:
    ----------
    fss_results_all : dict
        A dictionary where the keys are thresholds and the values are lists of FSS scores.
        Each list of FSS scores corresponds to the mean FSS across datasets for each scale.
    scales : Union[List[int], range]
        The list or range of scales used in the FSS calculation.
    title : str, optional
        Title of the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    figsize : tuple, optional
        Size of the plot figure.
    grid : bool, optional
        Whether to show grid lines on the plot.
    save_plot : bool, optional
        Whether to save the plot as an image file.
    save_path : str, optional
        Path to save the plot. Required if save_plot is True.

    Returns:
    -------
    None

    Raises:
    ------
    ValueError
        If save_plot is True but save_path is not provided.  

    Examples:
    --------
    >>> import numpy as np
    >>> from duplexity.plot import plot_fss
    >>> # Generate random FSS results
    >>> scales = [1, 2, 5, 10]
    >>> fss_results = {0.5: [0.6, 0.7, 0.8, 0.85], 0.75: [0.5, 0.6, 0.7, 0.8]}
    >>> # Plot the FSS results
    >>> plot_fss(fss_results, scales, title="FSS Results", xlabel="Scale", ylabel="FSS")
  
    """
    plt.figure(figsize=figsize)
    
    # Loop through each threshold and corresponding FSS results
    for key, value in fss_results_all.items():
        label = f'{key}'
        plt.plot(scales, value, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    plt.ylim(0, 1)  # FSS values range between 0 and 1
    plt.legend()

    if save_plot:
        if save_path is None:
            raise ValueError("Please provide a path to save the plot.")
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()



"""
RAPSD Plotting Utilities
---------------------------

This module contains utility functions for plotting radially averaged power spectral density (RAPSD) results.

.. autosummary::

    plot_rapsd
    
"""

# Plot the RAPSD on a log-log scale
def plot_rapsd(freqs, radial_profile, 
               x_units=None, y_units=None, 
               spatial_scales =None, 
               color="k", lw=1.0, 
               label=None, ax=None, save_path=None, title=None):
    """
    Plot the radially averaged power spectral density (RAPSD) on a log-log scale.
    
    Parameters
    ----------
    freqs (np.array): 
        1D array containing the  frequencies computed with the function
        :py:func:`duplexity.spatial.rapsd`.
    radial_profile (np.array): 
        1D array containing the radially averaged power spectral density.
    x_units: str, optional
        Units of the X variable, distance (e.g."pixel', "km").
    y_units : str, optional
        Units of the Y variable, amplitude (e.g. "dBR").
    spatial_scales (np.array):
        Array containing the spatial scales (e.g. wavelengths). If provided, the x-axis will be labeled with the spatial scales.
    color: str, optional
        Line color.
    lw: str, optional
        Line width.
    label: str, optional
        Label (for legend).
    ax: Axes, optional
        Plot axes.
    save_path: str, optional
        Path to save the plot.
    title: str, optional
        Title of the plot.
    
    Returns:
    ------- 
    ax: Axes
        Plot axes


    Examples:
    --------
    >>> import numpy as np
    >>> from duplexity.plot import plot_rapsd
    >>> # Generate a random dataset
    >>> freqs = np.linspace(0.1, 100, 100)
    >>> profile = np.random.rand(100)
    >>> # Plot the RAPSD
    >>> plot_rapsd(freqs, profile, x_units="km", y_units="dBR", title="RAPSD Example")
    """

    if len(freqs) != len(radial_profile):
        raise ValueError(f"The input arrays must have the same dimension. freqs: {len(freqs)}, profile: {len(radial_profile)}" )

    # Create a new axis if none is provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot spectrum in log-log scale
    ax.loglog(
        freqs[np.where(freqs > 0.0)],
        radial_profile[np.where(freqs > 0.0)],
        color=color, 
        linewidth=lw, 
        label=label)


    # X-axis label
    if spatial_scales is not None:
        ax.set_xticks(1 / spatial_scales)
        ax.set_xticklabels(spatial_scales)
        ax.set_xlabel(f"Wavelength [{x_units}]" if x_units else "Wavelength")
    else:
        ax.set_xlabel(f"Frequency [{x_units}]" if x_units else "Frequency")


    # Set the y-axis label with the provided units
    if y_units is not None and x_units is not None:
        units = fr"$\left[\frac{{{y_units}^2}}{{{x_units}}}\right]$"
        ax.set_ylabel(f"Power Density {units}")
    elif y_units is not None:
        units = fr"$\left[{y_units}^2\right]$"
        ax.set_ylabel(f"Power Density {units}")
    else:
        ax.set_ylabel("Power Density")

    # Set the title of the plot
    if title is not None:
        ax.set_title(title)

    # Save the plot if a path is provided
    if save_path is not None:
        pltsave = os.path.join(save_path, f"RAPSD_{title}.png")
        plt.savefig(pltsave, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
