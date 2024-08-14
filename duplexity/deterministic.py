"""
Deterministic Score
===================

Forecast evaluation and skill scores for deterministic continuous forecasts.


Continuous Metrics
------------------
.. autosummary::
    :toctree: ../generated/

    mean_absolute_error
    mean_squared_error
    root_mean_squared_error
    bias
    debiased_root_mean_squared_error
    pearson_correlation
    fss_score

Categorical Metrics
-------------------
.. autosummary::
    :toctree: ../generated/

    confusion_matrix
    precision
    recall
    f1_score
    accuracy
    critical_success_index
    equitable_threat_score
    false_alarm_ratio
    probability_of_detection
    gilbert_skill_score
    heidke_skill_score
    peirce_skill_score
    symmetric_extremal_dependence_index

"""





import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Tuple, Union, Optional
from scipy.ndimage import uniform_filter
import scipy.signal
from skimage.draw import disk
from duplexity.utils import _to_numpy, _check_shapes, _check_2d_data, _binary_classification



###########################################
##             Continuous Score          ##
###########################################

all_continuous_metrics = [
    "MAE",  # Mean Absolute Error
    "MSE",  # Mean Squared Error
    "RMSE", # Root Mean Squared Error
    "Bias", # Bias
    "DRMSE", # Debiased Root Mean Squared Error
    "Pearson Correlation" # Pearson Correlation
]

def mean_absolute_error(observed: Union[
                     np.ndarray, 
                     xr.DataArray, 
                     xr.Dataset,  
                     pd.DataFrame, 
                     List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]
                     ],
                        output: Union[
                     np.ndarray, 
                     xr.DataArray, 
                     xr.Dataset,  
                     pd.DataFrame, 
                     List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]
                     ]) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    Returns
    -------
    float
        The Mean Absolute Error (MAE), which measures the average magnitude of the absolute errors 
        between observed and predicted values. MAE provides a linear score that does not consider the direction of errors.

    Notes
    -----
    The MAE is a widely used metric in regression analysis and is particularly useful for evaluating model performance 
    where all errors are weighted equally.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the MAE for each pair of elements in the lists and then return the average of these individual MAEs.

    Example
    -------
    >>> observed_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> output_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> mean_absolute_error(observed_data, output_data)
    0.337  # Example output, depends on the random values

    In this example, `mean_absolute_error` calculates the MAE for each pair of `xr.DataArray` objects in `observed_data` 
    and `output_data` and then averages these values to produce the final MAE.
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)
    return np.mean(np.abs(observed - output))

def mean_squared_error(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Mean Squared Error (MSE) between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    Returns
    -------
    float
        The Mean Squared Error (MSE), which measures the average squared difference between observed and predicted values. 
        MSE is a quadratic scoring rule that penalizes larger errors more than smaller ones.

    Notes
    -----
    The MSE is a common metric in regression analysis, used to measure the accuracy of a model. 
    Unlike MAE, it gives more weight to larger errors due to the squaring of differences.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the MSE for each pair of elements in the lists and then return the average of these individual MSEs.

    Example
    -------
    >>> observed_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> output_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> mean_squared_error(observed_data, output_data)
    0.112  # Example output, depends on the random values

    In this example, `mean_squared_error` calculates the MSE for each pair of `xr.DataArray` objects in `observed_data` 
    and `output_data` and then averages these values to produce the final MSE.
    """

    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    return np.mean((observed - output) ** 2)

def root_mean_squared_error(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                            output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    Returns
    -------
    float
        The Root Mean Squared Error (RMSE), which is the square root of the average squared differences 
        between observed and predicted values. RMSE is sensitive to large errors and is often used to assess the accuracy of a model.

    Notes
    -----
    RMSE is a commonly used metric in regression analysis that provides an overall measure of the error magnitude. 
    It is particularly useful when comparing different models or algorithms, as it combines the advantages of both 
    the mean absolute error (MAE) and the mean squared error (MSE).

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the RMSE for each pair of elements in the lists and then return the average of these individual RMSEs.

    Example
    -------
    >>> observed_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> output_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> root_mean_squared_error(observed_data, output_data)
    0.355  # Example output, depends on the random values

    In this example, `root_mean_squared_error` calculates the RMSE for each pair of `xr.DataArray` objects in `observed_data` 
    and `output_data` and then averages these values to produce the final RMSE.
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    return np.sqrt(np.mean((observed - output) ** 2))

def bias(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the bias between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    Returns
    -------
    float
        The bias, which is the average difference between the observed and model output values. 
        Positive bias indicates overestimation by the model, while negative bias indicates underestimation.

    Notes
    -----
    Bias is a simple but important metric that indicates the overall tendency of a model to overestimate or 
    underestimate the observed values. It is often used in conjunction with other metrics like RMSE or MAE 
    to provide a fuller picture of model performance.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the bias for each pair of elements in the lists and then return the average of these individual biases.

    Example
    -------
    >>> observed_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> output_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> bias(observed_data, output_data)
    -0.027  # Example output, depends on the random values

    In this example, `bias` calculates the bias for each pair of `xr.DataArray` objects in `observed_data` 
    and `output_data` and then averages these values to produce the final bias.
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    return np.mean(output - observed)

def debiased_root_mean_squared_error(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                     output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Debiased Root Mean Squared Error (DRMSE) between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    Returns
    -------
    float
        The Debiased Root Mean Squared Error (DRMSE), which is the square root of the mean squared error 
        calculated after removing the bias between observed and predicted values.

    Notes
    -----
    The Debiased Root Mean Squared Error (DRMSE) adjusts for any systematic bias in the predictions 
    by first removing the bias from the predictions and then calculating the root mean squared error. 
    This metric provides a clearer indication of the model's performance by focusing on the variability 
    in the errors after accounting for bias.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the DRMSE for each pair of elements in the lists and then return the average of these individual DRMSEs.

    Example
    -------
    >>> observed_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> output_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> debiased_root_mean_squared_error(observed_data, output_data)
    0.278  # Example output, depends on the random values

    In this example, `debiased_root_mean_squared_error` calculates the DRMSE for each pair of `xr.DataArray` objects in `observed_data` 
    and `output_data` and then averages these values to produce the final DRMSE.
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    bias_value = np.mean(output - observed)
    debiased_predictions = output - bias_value
    return np.sqrt(np.mean((observed - debiased_predictions) ** 2))

def pearson_correlation(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Pearson correlation coefficient between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    Returns
    -------
    float
        The Pearson correlation coefficient, a measure of the linear relationship between the observed and model output values. 
        The coefficient ranges from -1 to 1, where 1 indicates a perfect positive linear relationship, -1 indicates a perfect negative 
        linear relationship, and 0 indicates no linear relationship.

    Notes
    -----
    The Pearson correlation coefficient is a widely used statistical measure to assess the strength and direction 
    of the linear relationship between two variables. A high absolute value of the coefficient indicates a strong linear relationship.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the Pearson correlation for each pair of elements in the lists and then return the average of these individual coefficients.

    Example
    -------
    >>> observed_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> output_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> pearson_correlation(observed_data, output_data)
    0.756  # Example output, depends on the random values

    In this example, `pearson_correlation` calculates the Pearson correlation coefficient for each pair of `xr.DataArray` objects in `observed_data` 
    and `output_data` and then averages these values to produce the final coefficient.
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    return np.corrcoef(observed.flatten(), output.flatten())[0, 1]

def calculate_continuous_metrics(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                 output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                 metrics: Union[str, Tuple[str], List[str]] = None) -> dict:
    """
    Calculate specified continuous metrics between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    metrics : Union[str, Tuple[str], List[str]], optional
        A string, tuple, or list of strings specifying the metrics to calculate. 
        If not provided, all available metrics will be calculated. Available metrics are:
        - "MAE" (Mean Absolute Error)
        - "MSE" (Mean Squared Error)
        - "RMSE" (Root Mean Squared Error)
        - "Bias"
        - "DRMSE" (Debiased Root Mean Squared Error)
        - "Pearson Correlation"

    Returns
    -------
    dict
        A dictionary where the keys are the names of the metrics and the values are the corresponding calculated values.

    Notes
    -----
    This function allows for flexible calculation of multiple continuous metrics between observed and model output data.
    Users can specify one or more metrics, or calculate all available metrics by leaving the `metrics` parameter as `None`.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the specified metrics for each pair of elements in the lists and then return the average of these individual metrics.

    Example
    -------
    >>> observed_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> output_data = [xr.DataArray(np.random.rand(3, 3)), xr.DataArray(np.random.rand(3, 3))]
    >>> calculate_continuous_metrics(observed_data, output_data, metrics=["MAE", "RMSE", "Bias"])
    {'MAE': 0.243, 'RMSE': 0.371, 'Bias': -0.015}  # Example output, depends on the random values

    In this example, `calculate_continuous_metrics` calculates the Mean Absolute Error, Root Mean Squared Error, 
    and Bias for each pair of `xr.DataArray` objects in `observed_data` and `output_data`, and returns the results as a dictionary.
    """
    available_metrics = {
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": root_mean_squared_error,
        "Bias": bias,
        "DRMSE": debiased_root_mean_squared_error,
        "Pearson Correlation": pearson_correlation
    }

    if metrics is None:
        results = {name: func(observed, output) for name, func in available_metrics.items()}
    else:
        if isinstance(metrics, str):
            metrics = [metrics]
        results = {}
        for metric in metrics:
            if metric in available_metrics:
                results[metric] = available_metrics[metric](observed, output)
            else:
                raise ValueError(f"Metric '{metric}' is not recognized. Available metrics are: {list(available_metrics.keys())}")
    return results

###########################################
##             Categorical Score         ##        
###########################################

all_categorical_metrics = [
    "Confusion Matrix",
    "Precision",
    "Recall",
    "F1 Score",
    "Accuracy",
    "CSI",
    "ETS",
    "FAR",
    "POD",
    "GSS",
    "HSS",
    "PSS",
    "SEDI"
]

def confusion_matrix(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                     output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                     threshold:float) -> np.array:
    """
    Calculate the confusion matrix between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    np.array
        A confusion matrix in the form of a 2x2 NumPy array, where:
        - The first row corresponds to the actual negative cases (True Negative, False Positive).
        - The second row corresponds to the actual positive cases (False Negative, True Positive).

    Notes
    -----
    The confusion matrix is a widely used tool for evaluating the performance of a classification model. 
    It provides insights into the types of errors the model makes and can be used to derive other metrics like precision, recall, and F1-score.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the confusion matrix for each pair of elements in the lists and then return the average confusion matrix.

    Example
    -------
    >>> observed_data = np.array([0, 1, 0, 1, 0, 1])
    >>> output_data = np.array([0.2, 0.8, 0.1, 0.6, 0.4, 0.9])
    >>> confusion_matrix(observed_data, output_data, threshold=0.5)
    array([[2, 1],
           [0, 3]])

    In this example, the `confusion_matrix` function calculates the confusion matrix by comparing the observed values 
    with the model output values, using a threshold of 0.5 to classify the output data.
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    observed_binary = _binary_classification(observed, threshold)
    output_binary = _binary_classification(output, threshold)
    
    TP = np.logical_and(output_binary == 1, observed_binary == 1)
    FN = np.logical_and(output_binary == 0, observed_binary == 1)
    FP = np.logical_and(output_binary == 1, observed_binary == 0)
    TN = np.logical_and(output_binary == 0, observed_binary == 0)
    
    return np.array([[TP.sum(), FN.sum()], [FP.sum(), TN.sum()]])

def precision(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
              output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
              threshold:float) -> float:
    """
    Calculate the precision between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The precision, which is the ratio of true positives to the sum of true positives and false positives.
        Precision = TP / (TP + FP)

    Notes
    -----
    Precision is a key metric in binary classification that measures the accuracy of the positive predictions made by the model. 
    It is particularly useful in situations where the cost of false positives is high.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the precision for each pair of elements in the lists and then return the average precision.

    Example
    -------
    >>> observed_data = np.array([0, 1, 0, 1, 0, 1])
    >>> output_data = np.array([0.2, 0.8, 0.1, 0.6, 0.4, 0.9])
    >>> precision(observed_data, output_data, threshold=0.5)
    1.0

    In this example, the `precision` function calculates the precision by comparing the observed values 
    with the model output values, using a threshold of 0.5 to classify the output data.
    """
    cm = confusion_matrix(observed, output, threshold)
    return cm[0, 0] / (cm[0, 0] + cm[1, 0])

def recall(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
           output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
           threshold:float) -> float:
    """
    Calculate the recall between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The recall, which is the ratio of true positives to the sum of true positives and false negatives.
        Recall = TP / (TP + FN)

    Notes
    -----
    Recall, also known as sensitivity or true positive rate, is a key metric in binary classification that measures 
    the model's ability to correctly identify all positive instances. It is particularly important in scenarios where 
    minimizing false negatives is crucial.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the recall for each pair of elements in the lists and then return the average recall.

    Example
    -------
    >>> observed_data = np.array([0, 1, 0, 1, 0, 1])
    >>> output_data = np.array([0.2, 0.8, 0.1, 0.6, 0.4, 0.9])
    >>> recall(observed_data, output_data, threshold=0.5)
    1.0

    In this example, the `recall` function calculates the recall by comparing the observed values 
    with the model output values, using a threshold of 0.5 to classify the output data.
    """

    cm = confusion_matrix(observed, output, threshold)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

def f1_score(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
             output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
             threshold:float) -> float:
    """
    Calculate the F1 score between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The F1 score, which is the harmonic mean of precision and recall.
        F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

    Notes
    -----
    The F1 score is a widely used metric in binary classification that balances precision and recall, making it 
    particularly useful in scenarios where the distribution of classes is imbalanced or where both false positives 
    and false negatives need to be considered.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the F1 score for each pair of elements in the lists and then return the average F1 score.

    Example
    -------
    >>> observed_data = np.array([0, 1, 0, 1, 0, 1])
    >>> output_data = np.array([0.2, 0.8, 0.1, 0.6, 0.4, 0.9])
    >>> f1_score(observed_data, output_data, threshold=0.5)
    1.0

    In this example, the `f1_score` function calculates the F1 score by first computing the precision and recall using the 
    specified threshold, and then calculating the harmonic mean of these two metrics.
    """

    precision_value = precision(observed, output, threshold)
    recall_value = recall(observed, output, threshold)
    return 2 * (precision_value * recall_value) / (precision_value + recall_value)

def accuracy(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
             output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
             threshold:float) -> float:
    """
    Calculate the accuracy between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The accuracy, which is the ratio of the number of correct predictions to the total number of predictions.
        Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Notes
    -----
    Accuracy is a widely used metric in binary classification that measures the overall correctness of the model's predictions.
    It is most useful when the classes are balanced; however, it can be misleading when dealing with imbalanced datasets.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the accuracy for each pair of elements in the lists and then return the average accuracy.

    """
    cm = confusion_matrix(observed, output, threshold)
    return (cm[0, 0] + cm[1, 1]) / cm.sum()

def critical_success_index(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           threshold:float) -> float:
    """
    Calculate the Critical Success Index (CSI) between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The Critical Success Index (CSI), which is the ratio of true positives to the sum of true positives, false negatives, and false positives.
        CSI = TP / (TP + FN + FP)

    Notes
    -----
    The Critical Success Index (CSI), also known as the Threat Score, is a metric used in binary classification to measure 
    the accuracy of positive predictions. Unlike accuracy, CSI accounts for both false positives and false negatives, 
    making it particularly useful in assessing model performance in imbalanced datasets or for rare events.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the CSI for each pair of elements in the lists and then return the average CSI.
    """

    cm = confusion_matrix(observed, output, threshold)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[1, 0])

def equitable_threat_score(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           threshold:float) -> float:
    """
    Calculate the Equitable Threat Score (ETS) between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The Equitable Threat Score (ETS), which adjusts the Critical Success Index (CSI) by accounting for hits due to random chance.
        ETS = (TP - CH) / (TP + FN + FP - CH)
        where CH (Chance Hits) = (TP + FN) * (TP + FP) / (TP + FN + FP + TN)

    Notes
    -----
    The Equitable Threat Score (ETS) is a metric used in binary classification to measure the skill of a model in predicting positive events, 
    adjusted for the number of hits that could occur by random chance. ETS is particularly useful in scenarios involving rare events or imbalanced datasets, 
    as it provides a more accurate assessment of model performance than the Critical Success Index (CSI) alone.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the ETS for each pair of elements in the lists and then return the average ETS.
    """
    cm = confusion_matrix(observed, output, threshold)
    hits_random = (cm[0, 0] + cm[1, 0]) * (cm[0, 0] + cm[0, 1]) / cm.sum()
    return (cm[0, 0] - hits_random) / (cm[0, 0] + cm[0, 1] + cm[1, 0] - hits_random)

def false_alarm_ratio(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                      output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                      threshold:float) -> float:
    """
    Calculate the False Alarm Ratio (FAR) between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The False Alarm Ratio (FAR), which is the ratio of false positives to the sum of false positives and true positives.
        FAR = FP / (FP + TP)

    Notes
    -----
    The False Alarm Ratio (FAR) is a metric used in binary classification to measure the proportion of positive predictions that are incorrect. 
    It is particularly important in scenarios where false positives are costly or problematic. FAR ranges from 0 to 1, with 0 indicating no false alarms 
    and 1 indicating that all positive predictions are false.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the FAR for each pair of elements in the lists and then return the average FAR.
    """

    cm = confusion_matrix(observed, output, threshold)
    return cm[1, 0] / (cm[0, 0] + cm[1, 0])


def probability_of_detection(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                             output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                             threshold:float) -> float:
    """
    Calculate the Probability of Detection (POD) between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The Probability of Detection (POD), which is the ratio of true positives to the sum of true positives and false negatives.
        POD = TP / (TP + FN)

    Notes
    -----
    The Probability of Detection (POD), also known as sensitivity or the true positive rate, is a key metric in binary classification 
    that measures the ability of the model to correctly identify positive cases. A POD of 1 indicates perfect detection of all positive cases, 
    while a POD of 0 indicates that no positive cases were detected.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the POD for each pair of elements in the lists and then return the average POD.

    """
    
    cm = confusion_matrix(observed, output, threshold)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

def gilbert_skill_score(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        threshold:float) -> float:
    """
    Calculate the Gilbert Skill Score (GSS) between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The Gilbert Skill Score (GSS), also known as the Equitable Threat Score (ETS), which adjusts the Critical Success Index (CSI) 
        by accounting for hits that could occur due to random chance.
        GSS = (TP - CH) / (TP + FN + FP - CH)
        where CH (Chance Hits) = (TP + FN) * (TP + FP) / (TP + FN + FP + TN)

    Notes
    -----
    The Gilbert Skill Score (GSS) is a metric used in binary classification to assess the skill of a model by considering 
    both correct predictions and the impact of random chance. It is particularly useful in cases involving rare events 
    or imbalanced datasets, where traditional metrics like accuracy may be misleading.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the GSS for each pair of elements in the lists and then return the average GSS.
    """
    cm = confusion_matrix(observed, output, threshold)
    hits_random = (cm[0, 0] + cm[1, 0]) * (cm[0, 0] + cm[0, 1]) / cm.sum()
    return (cm[0, 0] - hits_random) / (cm[0, 0] + cm[0, 1] + cm[1, 0] - hits_random)

def heidke_skill_score(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       threshold:float) -> float:
    """
    Calculate the Heidke Skill Score (HSS) between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The Heidke Skill Score (HSS), which measures the skill of a binary classification model compared to random chance.
        HSS = 2 * (TP * TN - FP * FN) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))

    Notes
    -----
    The Heidke Skill Score (HSS) is a metric used to assess the accuracy of a model's predictions relative to random chance. 
    Unlike some other metrics, HSS considers all elements of the confusion matrix (TP, TN, FP, FN) and is particularly useful 
    when the goal is to compare model performance against a baseline of random prediction. HSS ranges from -1 to 1, 
    where 1 indicates perfect skill, 0 indicates no skill, and negative values indicate worse-than-random performance.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the HSS for each pair of elements in the lists and then return the average HSS.
    """    
    cm = confusion_matrix(observed, output, threshold)
    hits = cm[1, 1]
    false_alarms = cm[0, 1]
    misses = cm[1, 0]
    correct_negatives = cm[0, 0]
    total = hits + false_alarms + misses + correct_negatives
    accuracy_random = ((hits + false_alarms) * (hits + misses) + (correct_negatives + misses) * (correct_negatives + false_alarms)) / (total * total)
    accuracy_observed = (hits + correct_negatives) / total
    return (accuracy_observed - accuracy_random) / (1 - accuracy_random) if (1 - accuracy_random) != 0 else 0

def peirce_skill_score(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       threshold:float) -> float:
    """
    Calculate the Peirce Skill Score (PSS) between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The Peirce Skill Score (PSS), also known as the True Skill Statistic (TSS), which is calculated as:
        PSS = TP / (TP + FN) - FP / (FP + TN)

    Notes
    -----
    The Peirce Skill Score (PSS) is a metric used to measure the ability of a binary classifier to distinguish between positive and negative cases. 
    PSS is particularly useful in evaluating model performance on imbalanced datasets, as it is unaffected by the proportion of positive and negative cases.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `np.array`, or `pd.DataFrame`, 
    the function will calculate the PSS for each pair of elements in the lists and then return the average PSS.
    """
    cm = confusion_matrix(observed, output, threshold)
    POD = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    POFD = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    return POD - POFD

def sedi(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         threshold:float) -> float:
    """
    Calculate the Symmetric Extremal Dependence Index (SEDI) between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    float
        The Symmetric Extremal Dependence Index (SEDI), calculated as:
        SEDI = (log(FP / (FP + TN)) - log(TP / (TP + FN))) / (log(FP / (FP + TN)) + log(TP / (TP + FN)))

    Notes
    -----
    The Symmetric Extremal Dependence Index (SEDI) is a metric used to evaluate the performance of a binary classifier, 
    particularly in the context of rare events. It accounts for the balance between false positives and false negatives, 
    providing a more nuanced assessment of model performance in extreme situations.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `xr.Dataset`, or `pd.DataFrame`, 
    the function will calculate the SEDI for each pair of elements in the lists and then return the average SEDI.
    """
    cm = confusion_matrix(observed, output, threshold)
    H = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    F = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    if H in [0, 1] or F in [0, 1]:
        return float('nan')  # Avoid division by zero and log(0)
    return (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))

def calculate_categorical_metrics(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                  output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                  metrics: Union[str, Tuple[str], List[str]] = None,
                                  threshold: float = 0.5) -> dict:
    """
    Calculate specified categorical metrics between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.

    metrics : Union[str, Tuple[str], List[str]], optional
        A string, tuple, or list of strings specifying the categorical metrics to calculate. 
        If not provided, all available metrics will be calculated. Available metrics are:
        - "Confusion Matrix"
        - "Precision"
        - "Recall"
        - "F1 Score"
        - "Accuracy"
        - "CSI" (Critical Success Index)
        - "ETS" (Equitable Threat Score)
        - "FAR" (False Alarm Ratio)
        - "POD" (Probability of Detection)
        - "GSS" (Gilbert Skill Score)
        - "HSS" (Heidke Skill Score)
        - "PSS" (Peirce Skill Score)
        - "SEDI" (Symmetric Extremal Dependence Index)

    threshold : float, optional
        A threshold value used to convert continuous output values into binary classifications (0 or 1). 
        Default is 0.5. Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    Returns
    -------
    dict
        A dictionary where the keys are the names of the metrics and the values are the corresponding calculated values.

    Notes
    -----
    This function allows for flexible calculation of multiple categorical metrics between observed and model output data. 
    Users can specify one or more metrics, or calculate all available metrics by leaving the `metrics` parameter as `None`.

    If the inputs `observed` and `output` are provided as lists of `xr.DataArray`, `xr.Dataset`, or `pd.DataFrame`, 
    the function will calculate the specified metrics for each pair of elements in the lists and then return the average of these individual metrics.
    
    
    Example
    -------
    >>> observed_data = np.array([0, 1, 0, 1, 0, 1])
    >>> output_data = np.array([0.2, 0.8, 0.1, 0.6, 0.4, 0.9])
    >>> calculate_categorical_metrics(observed_data, output_data, metrics=["Precision", "Recall", "F1 Score"], threshold=0.5)
    {'Precision': 1.0, 'Recall': 1.0, 'F1 Score': 1.0}

    In this example, `calculate_categorical_metrics` calculates the Precision, Recall, and F1 Score 
    by comparing the observed values with the model output values, using a threshold of 0.5 to classify the output data.

    """
    available_metrics = {
        "Confusion Matrix": confusion_matrix,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "Accuracy": accuracy,
        "CSI": critical_success_index,
        "ETS": equitable_threat_score,
        "FAR": false_alarm_ratio,
        "POD": probability_of_detection,
        "GSS": gilbert_skill_score,
        "HSS": heidke_skill_score,
        "PSS": peirce_skill_score,
        "SEDI": sedi
    }

    if metrics is None:
        results = {name: func(observed, output, threshold) for name, func in available_metrics.items()}
    else:
        if isinstance(metrics, str):
            metrics = [metrics]
        results = {}
        for metric in metrics:
            if metric in available_metrics:
                results[metric] = available_metrics[metric](observed, output, threshold)
            else:
                raise ValueError(f"Metric '{metric}' is not recognized. Available metrics are: {list(available_metrics.keys())}")
    return results





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
    fss = dict(threshold=threshold, scale=scale, sum_output_sq=0.0, sum_output_observed=0.0, sum_observed_sq=0.0)
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
    sum_output_sq = fss['sum_output_sq']
    sum_observed_sq = fss['sum_observed_sq']
    sum_output_observed = fss['sum_output_observed']

    numerator = sum_output_sq + sum_observed_sq - 2 * sum_output_observed
    denominator = sum_output_sq + sum_observed_sq

    if denominator == 0:
        return np.nan

    fss_value = 1 - numerator / denominator
    return fss_value


def fss_score(
              observed:Union[np.array, xr.DataArray, pd.DataFrame],
              output:Union[np.array, xr.DataArray, pd.DataFrame],
              threshold: Union[float, int, List[Union[float, int]]] , scale:Union[float, int,  List[Union[float, int]]]
              ) -> float:
    """
    Calculate the Fractions Skill Score (FSS) between observed and model output values based on specified thresholds and scales.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame]
        Array of shape (h, w) containing observed binary or continuous values, where h is the height, and w is the width.
        
    output : Union[np.array, xr.DataArray, pd.DataFrame]
        Array of shape (h, w) containing model output binary or continuous values, where h is the height, and w is the width.

    threshold : Union[float, int, List[Union[float, int]]]
        A single threshold value or a list of threshold values used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    scale : Union[float, int, List[Union[float, int]]]
        A single scale value or a list of scale values representing the neighborhood size for which the fractions are computed.
        The scale is typically expressed in grid points or distance units.

    Returns
    -------
    float
        The Fractions Skill Score (FSS), which ranges from 0 to 1. An FSS of 1 indicates perfect agreement between the observed and forecast fractions, 
        while an FSS of 0 indicates no skill.

    Notes
    -----
    The Fractions Skill Score (FSS) is a metric used to assess the spatial accuracy of high-resolution forecasts, particularly in the context of precipitation. 
    Unlike traditional categorical metrics, FSS considers the spatial distribution of the forecast and observed fields, making it well-suited for evaluating 
    forecasts with spatial uncertainty.

    If a list of thresholds or scales is provided, the function will calculate the FSS for each combination of threshold and scale and then return the average FSS.

    Example
    -------
    >>> observed_data = np.random.rand(100, 100)
    >>> output_data = np.random.rand(100, 100)
    >>> calculate_fss_score(observed_data, output_data, threshold=0.5, scale=10)
    0.85  # Example output, depends on the random values

    In this example, the `calculate_fss_score` function calculates the FSS by comparing the observed values 
    with the model output values, using a threshold of 0.5 and a scale of 10 grid points.
    """
    output = _to_numpy(output)
    observed = _to_numpy(observed)
    _check_shapes(output, observed)



    if isinstance(threshold, (float, int)):
        threshold = [threshold]

    if isinstance(scale, (int, float)):
        scale = [scale]


    fss_scores = []
    for scale_item in scale:
        for thre_item in threshold:
            fss = fss_initialize(thre_item, scale_item)
            fss_update(fss, output, observed)
            fss_scores.append(fss_compute(fss))

    return fss_scores



#############################################################
##          Precipitation Smoothing Distance               ##
#############################################################

def circular_kernel(radius: int) -> np.array:
    """
    Creates a circular kernel with the given radius.
    
    Parameters:
    radius (int): The radius of the circular kernel.
    
    Returns:
    np.array: A 2D array representing the circular kernel.
    """
    # Calculate the size of the kernel
    r_max = int(np.floor(radius))
    size = 2 * r_max + 1
    shape = (size, size)
    kernel = np.zeros((size, size), dtype =np.float32)
    
    # Calculate the center of the kernel
    center = r_max
    
    # Get the row and column coordinates for the disk
    row_coords, col_coords = disk((center, center), radius + 0.0001, shape=shape)

    # Set the pixels inside the disk to 1
    kernel[row_coords, col_coords] = 1

    # Normalize the kernel so that the sum of its elements equals 1
    kernel /= np.sum(kernel)

    return kernel



def calculate_pss(output_smooth: np.array, observed_smooth: np.array, radius: int, Q: float) -> float:
    """
    Calculates the Precipitation Smooth Score (PSS).
    
    Parameters:
    output_smooth (np.array): Smoothed forecast data.
    observed_smooth (np.array): Smoothed observed data.
    radius (float): Radius for smoothing.
    Q (float): Quality factor.
    
    Returns:
    float: Precipitation Symmetry Score (PSS).
    """
    kernel = circular_kernel(radius)
    output_smooth = scipy.signal.fftconvolve(output_smooth, kernel, mode='full')
    observed_smooth = scipy.signal.fftconvolve(observed_smooth, kernel, mode='full')
    PSS = 1.0 - 1.0 / (2.0 * float(output_smooth.size) * Q) * np.abs(output_smooth - observed_smooth).sum()
    return PSS

def calculate_psd(output: np.ndarray, observed: np.ndarray) -> float:
    """
    """
    output = output.copy()
    observed = observed.copy()

    _check_shapes(output, observed)
    
    if output.ndim != 2 or observed.ndim != 2:
        raise ValueError("output and observed data are not two-dimensional.")
    
    if output.size == 0 or observed.size == 0:
        raise ValueError("output and observed data are empty.")
    
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(observed)):
        raise ValueError("output and observed data contain non-numeric values.")
    
    if isinstance(output, np.ma.MaskedArray) or isinstance(observed, np.ma.MaskedArray):
        raise ValueError("output and observed data are masked arrays which is not allowed.")
    
    if np.any(output < 0) or np.any(observed < 0):
        raise ValueError("output and observed data contain negative values which is not allowed.")
    
    output = output.astype(float)
    observed = observed.astype(float)
    
    output_avg = np.average(output)
    observed_avg = np.average(observed)
    
    if output_avg == 0 or observed_avg == 0:
        return np.nan  # Return NaN for empty fields    
    
    output_norm = output / output_avg
    observed_norm = observed / observed_avg
    
    if np.array_equal(output_norm, observed_norm):
        return 0
    
    output_diff = output_norm - np.minimum(output_norm, observed_norm)
    observed_diff = observed_norm - np.minimum(output_norm, observed_norm)
    
    Q = output_diff.sum() / output_norm.sum()
    
    initial_radius = 1
    PSS_initial = calculate_pss(output_diff, observed_diff, initial_radius, Q)
    
    if PSS_initial > 0.5:
        return 1
    
    diagonal = np.sqrt(output.shape[0]**2 + output.shape[1]**2)
    dr = np.ceil(diagonal * 0.05)
    
    radius2 = initial_radius + dr
    PSS2 = calculate_pss(output_diff, observed_diff, radius2, Q)
    
    while PSS2 < 0.5:
        initial_radius = radius2
        PSS_initial = PSS2
        radius2 = initial_radius + dr
        PSS2 = calculate_pss(output_diff, observed_diff, radius2, Q)
    
    while radius2 - initial_radius > 1:
        new_radius = int((initial_radius + radius2) / 2)
        PSS_new = calculate_pss(output_diff, observed_diff, new_radius, Q)
        
        if PSS_new > 0.5:
            radius2 = new_radius
            PSS2 = PSS_new
        else:
            initial_radius = new_radius
            PSS_initial = PSS_new
    
    PSD = 0.808 * Q * float(radius2)
    
    return PSD


def validate_with_psd(observed: np.ndarray, output: np.ndarray) -> np.ndarray:
    """
    Validate the forecasted data with the Precipitation Symmetry Distance (PSD).
    
    Parameters:
    observed (np.ndarray): Observed data array.
    forecasted (np.ndarray): Forecasted data array.
    
    Returns:
    np.ndarray: Array of PSD values.
    """
    psd_results = []
    for i in tqdm(range(output.shape[0]), desc="Calculating PSD"):
        observed_slice = observed[i]
        output_slice = output[i]
        psd = calculate_psd(observed_slice, output_slice)
        psd_results.append(psd)
    return np.array(psd_results)






##################################################################################
##            Radially Averaged Power Spectral Density (RAPSD)                  ##
##################################################################################


def compute_centred_coord_array(H: int, W: int) -> Tuple[np.array, np.array]:
    """Compute a 2D coordinate array, where the origin is at the center.
    Parameters
    ----------
    H : int
      The height of the array.
    W : int
      The width of the array.
    Returns
    -------
    out : ndarray
      The coordinate array.
    Examples
    --------
    >>> compute_centred_coord_array(2, 2)
    (array([[-2],\n
        [-1],\n
        [ 0],\n
        [ 1],\n
        [ 2]]), array([[-2, -1,  0,  1,  2]]))
    """

    if H % 2 == 1:
        H_slice = np.s_[-int(H / 2): int(H / 2) + 1]
    else:
        H_slice = np.s_[-int(H / 2): int(H / 2)]

    if W % 2 == 1:
        W_slice = np.s_[-int(W / 2): int(W / 2) + 1]
    else:
        W_slice = np.s_[-int(W / 2): int(W / 2)]

    y_coords, x_coords = np.ogrid[H_slice, W_slice]

    return y_coords, x_coords


def rapsd(
    data, fft_method=None, return_freq=False, d=1.0, normalize=False, **fft_kwargs
):
    """Compute radially averaged power spectral density (RAPSD) from the given
    2D input field.
    Parameters
    ----------
    field: array_like
        A 2d array of shape (m, n) containing the input field.
    fft_method: object
        A module or object implementing the same methods as numpy.fft and
        scipy.fftpack. If set to None, field is assumed to represent the
        shifted discrete Fourier transform of the input field, where the
        origin is at the center of the array
        (see numpy.fft.fftshift or scipy.fftpack.fftshift).
    return_freq: bool
        Whether to also return the Fourier frequencies.
    d: scalar
        Sample spacing (inverse of the sampling rate). Defaults to 1.
        Applicable if return_freq is 'True'.
    normalize: bool
        If True, normalize the power spectrum so that it sums to one.
    Returns
    -------
    out: ndarray
      One-dimensional array containing the RAPSD. The length of the array is
      int(l/2) (if l is even) or int(l/2)+1 (if l is odd), where l=max(m,n).
    freq: ndarray
      One-dimensional array containing the Fourier frequencies.
    References
    ----------
    :cite:`RC2011`
    """

    if len(data.shape) == 2:
        h, w = data.shape
    elif len(data.shape) == 3:
        h,w = data.shape[1:]
    else:
        raise ValueError(
            f"{len(data.shape)} dimensions are found, but the number "
            "of dimensions should be 2"
        )

    if np.sum(np.isnan(data)) > 0:
        raise ValueError("input field should not contain nans")


    y_coords,  x_coords = compute_centred_coord_array(h, w)
    radial_grid = np.sqrt( x_coords *  x_coords + y_coords * y_coords).round()
    max_dim = max(data.shape[0], data.shape[1]) 

    if max_dim % 2 == 1:
        radial_range = np.arange(0, int(max_dim / 2) + 1)
    else:
        radial_range = np.arange(0, int(max_dim / 2))

    if fft_method is not None:
        psd = fft_method.fftshift(fft_method.fft2(data, **fft_kwargs))
        psd = np.abs(psd) ** 2 / psd.size
    else:
        psd = data

    result = []
    for r in radial_range:
        mask = radial_grid == r
        psd_vals = psd[mask]
        result.append(np.mean(psd_vals))

    result = np.array(result)

    if normalize:
        result /= np.sum(result)

    if return_freq:
        frequencies = np.fft.fftfreq(max_dim, d=d)
        frequencies = frequencies[radial_range]
        return result, frequencies
    else:
        return result
