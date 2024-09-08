import numpy as np
import xarray as xr
import pandas as pd
from scipy.ndimage import affine_transform
from scipy.fftpack import fftshift, fft2
from typing import Union, List
from utils import _to_numpy, _check_2d_data, _check_shapes, _binary_classification
from pixelwise import mean_squared_error
from scipy.ndimage import uniform_filter





################################
##           FSS              ##
################################

def fss_initialize(threshold: float, scale: int) -> dict:
    """
    Initialize a fractions skill score (FSS) object.
    
    Parameters:
        threshold (float): The intensity threshold value for binarizing the data.
        scale (int): Size of the neighborhood for calculating fractions, in pixels.

    Returns:
        dict: FSS verification object.

    """
    fss = dict(threshold=threshold, 
               scale=scale, 
               sum_output_sq=0.0, 
               sum_output_observed=0.0, 
               sum_observed_sq=0.0)
    return fss



def fss_update(fss: dict, observed: np.array, output: np.array) -> None:
    """
    Update the FSS object with new forecast and observed data.
    
    Parameters:
    fss (dict): 
        FSS verification object.
    output (np.array): 
        Model output data array in shape (height, width).
    observed (np.array): 
        Observed data array in shape (height, width).
    """
    _check_shapes(output, observed)
    _check_2d_data(output)
    _check_2d_data (observed)


    threshold = fss['threshold']
    scale = fss['scale']

    binary_output = (_binary_classification(output, threshold)).astype(float)
    binary_observed =(_binary_classification(observed, threshold)).astype(float)
    
    if scale > 1:
        smoothed_forecast = uniform_filter(binary_output, size=scale, mode="constant", cval=0.0)
        smoothed_observation = uniform_filter(binary_observed, size=scale, mode="constant", cval=0.0)
    else:
        smoothed_forecast = binary_output
        smoothed_observation = binary_observed

    fss["sum_output_sq"] += np.nansum(smoothed_forecast ** 2)
    fss["sum_output_observed"] += np.nansum(smoothed_forecast * smoothed_observation)
    fss["sum_observed_sq"] += np.nansum(smoothed_observation ** 2)



def fss_compute(fss: dict) -> float:
    """
    Calculate the Fractions Skill Score (FSS).
    
    Parameters:
    fss (dict): FSS verification object.
    
    Returns:
    float: Fractions Skill Score.
    """
    # Calculate the FSS value
    sum_output_sq = fss['sum_output_sq']
    sum_observed_sq = fss['sum_observed_sq']
    sum_output_observed = fss['sum_output_observed']
    
    numerator = sum_output_sq + sum_observed_sq - 2 * sum_output_observed
    denominator = sum_output_sq + sum_observed_sq

    if denominator == 0:
        return np.nan

    fss_value = 1 - numerator / denominator
    return fss_value


def fss(
        observed:Union[np.array, xr.DataArray, pd.DataFrame, xr.Dataset],
        output:Union[np.array, xr.DataArray, pd.DataFrame, xr.Dataset],
        threshold: Union[float, int] , scale: Union[int, List[int], range] = 1,
        var: str = None
        ) -> Union[float, List[float]]:
    """
    Calculate the Fractions Skill Score (FSS) between observed and model output values based on specified thresholds and scales.

    The Fractions Skill Score (FSS) is a metric used to assess the spatial accuracy of high-resolution forecasts, 
    particularly in the context of precipitation or other spatial fields. Unlike traditional categorical metrics, 
    the FSS considers the spatial distribution of the forecast and observed fields by computing the fraction of 
    grid points exceeding a specified threshold within a defined neighborhood size (scale). The FSS ranges from 0 
    to 1, where 1 indicates perfect agreement between the forecast and observed fractions, and 0 indicates no skill.


    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, xr.Dataset]
        The observed data array, which can be a NumPy array, xarray DataArray, xarray Dataset, or Pandas DataFrame.
        The data should be two-dimensional with shape (h, w), where `h` is the height and `w` is the width.
        
    output : Union[np.array, xr.DataArray, pd.DataFrame, xr.Dataset]
        The model output data array, which can be a NumPy array, xarray DataArray, xarray Dataset, or Pandas DataFrame.
        The data should be two-dimensional with shape (h, w), where `h` is the height and `w` is the width.

    threshold : Union[float, int]
        A single threshold value used to convert continuous output values into binary classifications.
        Grid points with values greater than or equal to this threshold will be classified as 1 (event occurred), 
        and values below the threshold will be classified as 0 (event did not occur).

    scale : Union[int, List[int], range], optional
        A single scale value representing the neighborhood size for which the fractions are computed, expressed in grid 
        points or distance units. A larger scale increases the area over which fractions are calculated, smoothing out 
        smaller-scale variations.

    var : str, optional
        If the input data is an xarray Dataset, `var` specifies the variable name to extract for the calculation. 
        If not provided, the first variable in the Dataset will be used.

    Raises
    ------
    ValueError
        If the `observed` and `output` arrays do not have the same shape.

    TypeError
        If the input data type is not supported.


    Returns
    -------
    float or List[float]
        The Fractions Skill Score (FSS), which ranges from 0 to 1. An FSS of 1 indicates perfect agreement between the observed and forecast fractions, 
        while an FSS of 0 indicates no skill.

    Notes
    -----
    - The FSS is particularly useful for evaluating the spatial accuracy of forecasts in scenarios where exact 
      placement of features (like precipitation) may vary, but the overall spatial pattern is important.
    - The threshold and scale parameters can be tuned based on the specific application and the nature of the 
      forecast and observed fields.

    Example
    -------
    >>> observed_data = np.random.rand(100, 100)
    >>> output_data = np.random.rand(100, 100)
    >>> calculate_fss_score(observed_data, output_data, threshold=0.5, scale=10)
    0.85  # Example output, depends on the random values

    In this example, the `calculate_fss_score` function calculates the FSS by comparing the observed values 
    with the model output values, using a threshold of 0.5 and a scale of 10 grid points.
    """
    # Convert the input data to NumPy arrays
    output = _to_numpy(output, var)
    observed = _to_numpy(observed, var)
    # Check the shapes of the observed and output data
    _check_shapes(output, observed)
    
    if isinstance(scale, (int, float)):
        # Single scale, return a single FSS score
        fss_obj = fss_initialize(threshold, scale)
        # Update the FSS object with the observed and output data
        fss_update(fss_obj, observed, output)
        # Compute the FSS value
        fss_results = fss_compute(fss_obj)
    
    elif isinstance(scale, (list, range)):
        # Multiple scales, return a list of FSS scores
        fss_results = []
        for scale_item in scale:
            fss_obj = fss_initialize(threshold, scale_item)
            fss_update(fss_obj, observed, output)
            fss_results.append(fss_compute(fss_obj))
    
    return fss_results




## neighborhood probability
def neighborhood_probability(data: np.ndarray, threshold: float, neighborhood_size: int) -> np.ndarray:
    """
    Calculate the neighborhood probability of exceeding a threshold.

    Parameters
    ----------
    data : np.ndarray
        The observed data array.
    threshold : float
        The threshold value to define the event.
    neighborhood_size : int
        The size of the neighborhood (in grid points) around each point to consider.

    Returns
    -------
    np.ndarray
        The neighborhood probability field.
    """
    binary_field = (data >= threshold).astype(float)
    neighborhood_sum = np.convolve(binary_field, np.ones((neighborhood_size, neighborhood_size)), mode='same')
    return neighborhood_sum / (neighborhood_size ** 2)


################################
##           SPS              ##
################################

def spatial_probability_score(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                threshold:float,
                                neighborhood_size: int) -> float:
    """
    Calculate the Spatial Probability Score (SPS) between observed and model output values based on a specified threshold.
    It assesses how well a forecast matches observed spatial patterns by considering neighborhood-based variability.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) containing observed binary or continuous values, where h is the height, and w is the width.

    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) containing model output binary or continuous values, where h is the height, and w is the width.

    threshold : float
        The threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    neighborhood_size : int
        The size of the neighborhood (in grid points) around each point to consider.

    Returns
    -------
    float
        The Spatial Probability Score (SPS), which ranges from 0 to 1. A higher SPS indicates better agreement between the observed and forecast spatial patterns.

    Notes
    -----
    The Spatial Probability Score (SPS) is a metric used to evaluate the spatial accuracy of high-resolution forecasts, particularly in the context of precipitation.
    It quantifies the similarity between the spatial patterns of observed and forecast fields by considering neighborhood-based variability.
    A higher SPS value indicates better agreement between the forecast and observed spatial patterns.

    Example
    -------
    >>> observed_data = np.random.rand(100, 100)
    >>> output_data = np.random.rand(100, 100)
    >>> spatial_probability_score(observed_data, output_data, threshold=0.5, neighborhood_size=10)
    0.75  # Example output, depends on the random values

    In this example, the `spatial_probability_score` function calculates the SPS by comparing the observed values with the model output values,
    using a threshold of 0.5 and a neighborhood size of 10 grid points.
    """
    output = _to_numpy(output)

    observed = _to_numpy(observed)

    _check_shapes(output, observed)

    binary_output = (_binary_classification(output, threshold)).astype(float)
    binary_observed = (_binary_classification(observed, threshold)).astype(float)

    obs_prob = neighborhood_probability(binary_observed, threshold, neighborhood_size)
    out_prob = neighborhood_probability(binary_output, threshold, neighborhood_size)

    return np.mean((obs_prob - out_prob) ** 2)







def mean_spatial_error(observed: np.ndarray, output: np.ndarray, neighborhood_size: int) -> float:
    """
    Calculate the Mean Spatial Error (MSE) between observed and forecast data.

    Parameters
    ----------
    observed : np.ndarray
        The observed data array.
    output : np.ndarray
        The forecast data array.
    neighborhood_size : int
        The size of the neighborhood (in grid points) around each point to consider.

    Returns
    -------
    float
        The Mean Spatial Error (MSE).
    """
    spatial_error = np.abs(observed - output)
    return np.mean(np.convolve(spatial_error, np.ones((neighborhood_size, neighborhood_size)), mode='same'))







def upscaled_probability_score(observed: np.ndarray, output: np.ndarray, threshold: float, scales: List[int]) -> dict:
    """
    Calculate the Upscaled Probability Score (UPS) across multiple spatial scales.

    Parameters
    ----------
    observed : np.ndarray
        The observed data array.
    output : np.ndarray
        The forecast data array.
    threshold : float
        The threshold value to define the event.
    scales : List[int]
        List of neighborhood sizes (scales) to evaluate.

    Returns
    -------
    dict
        A dictionary of UPS values for each scale.
    """
    ups_scores = {}
    for scale in scales:
        ups_scores[f'Scale_{scale}'] = spatial_probability_score(observed, output, threshold, scale)
    return ups_scores
