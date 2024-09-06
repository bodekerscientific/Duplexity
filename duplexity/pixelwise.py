"""
PixelWise Metrics
===================

The `pixelwise` module provides functions for evaluating the performance of models on pixel-wise data.
These functions calculate a variety of metrics that compare observed and model output values at each pixel.

The module is divided into two main categories of metrics: continuous metrics and categorical metrics.

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
    false_alarm_ratio
    probability_of_detection
    gilbert_skill_score
    heidke_skill_score
    peirce_skill_score
    symmetric_extremal_dependence_index

Pixelwise Metric Calculation
--------------------
.. autosummary::
    :toctree: ../generated/

    calculate_pixelwise_metric

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
    "Corr" # Pearson Correlation
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
                     ],
                     var: str = None) -> float:
    """
    Calculate the Mean Absolute Error (MAE) between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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
    observed = _to_numpy(observed, var)

    output = _to_numpy(output, var)

    _check_shapes(observed, output)
    return np.mean(np.abs(observed - output))

def mean_squared_error(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       var: str = None) -> float:
    """
    Calculate the Mean Squared Error (MSE) between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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

    observed = _to_numpy(observed, var)

    output = _to_numpy(output, var)

    _check_shapes(observed, output)

    return np.mean((observed - output) ** 2)

def root_mean_squared_error(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                            output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],var: str = None) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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
    observed = _to_numpy(observed, var)

    output = _to_numpy(output, var)

    _check_shapes(observed, output)

    return np.sqrt(np.mean((observed - output) ** 2))

def bias(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],var: str = None) -> float:
    """
    Calculate the bias between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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
    observed = _to_numpy(observed, var)

    output = _to_numpy(output, var)

    _check_shapes(observed, output)

    return np.mean(output - observed)

def debiased_root_mean_squared_error(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                     output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],var: str = None) -> float:
    """
    Calculate the Debiased Root Mean Squared Error (DRMSE) between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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
    observed = _to_numpy(observed, var)

    output = _to_numpy(output, var)

    _check_shapes(observed, output)

    bias_value = np.mean(output - observed)
    debiased_predictions = output - bias_value
    return np.sqrt(np.mean((observed - debiased_predictions) ** 2))

def pearson_correlation(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],var: str = None) -> float:
    """
    Calculate the Pearson correlation coefficient between observed and model output values.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output values, where n is the number of samples, h is the height, and w is the width.

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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
    observed = _to_numpy(observed, var)

    output = _to_numpy(output, var)

    _check_shapes(observed, output)

    return np.corrcoef(observed.flatten(), output.flatten())[0, 1]


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
                     threshold:float,
                     var:str = None) -> np.array:
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

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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
    observed = _to_numpy(observed, var)

    output = _to_numpy(output, var)

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
              threshold:float,
              var: str = None) -> float:
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

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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
    cm = confusion_matrix(observed, output, threshold, var)
    return cm[0, 0] / (cm[0, 0] + cm[1, 0])

def recall(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
           output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
           threshold:float,
           var: str = None) -> float:
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

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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

    cm = confusion_matrix(observed, output, threshold, var)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

def f1_score(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
             output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
             threshold:float,
             var: str = None) -> float:
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

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.

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
    
    precision_value = precision(observed, output, threshold, var)
    recall_value = recall(observed, output, threshold, var)
    return 2 * (precision_value * recall_value) / (precision_value + recall_value)

def accuracy(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
             output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
             threshold:float = 0.5,
             var:str = None) -> float:
    """
    Calculate the accuracy between observed and model output values based on a specified threshold.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float (default: 0.5)
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.

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
    cm = confusion_matrix(observed, output, threshold, var)
    return (cm[0, 0] + cm[1, 1]) / cm.sum()

def critical_success_index(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           threshold:float,
                           var:str = None) -> float:
    """
    Calculate the Critical Success Index (CSI) between observed and model output values based on a specified threshold. 
    It measures the ability of a forecast to predict the occurrence of events (e.g., rain) while accounting for both false alarms and missed events.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.
    
    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.


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

    cm = confusion_matrix(observed, output, threshold, var)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[1, 0])



def false_alarm_ratio(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                      output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                      threshold:float,
                      var:str = None) -> float:
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

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.

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

    cm = confusion_matrix(observed, output, threshold, var)
    return cm[1, 0] / (cm[0, 0] + cm[1, 0])


def probability_of_detection(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                             output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                             threshold:float,
                             var:str = None) -> float:
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

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.

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
    
    cm = confusion_matrix(observed, output, threshold, var)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

def gilbert_skill_score(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        threshold:float,
                        var:str = None) -> float:
    """
    Calculate the Gilbert Skill Score (GSS), also known as the Equitable Threat Score, between observed and model output values based on a specified threshold.
    It adjusts the Critical Success Index by accounting for hits that could occur by random chance.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.
    
    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.
     
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
    cm = confusion_matrix(observed, output, threshold, var)
    hits_random = (cm[0, 0] + cm[1, 0]) * (cm[0, 0] + cm[0, 1]) / cm.sum()
    return (cm[0, 0] - hits_random) / (cm[0, 0] + cm[0, 1] + cm[1, 0] - hits_random)

def heidke_skill_score(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       threshold:float,
                       var:str = None) -> float:
    """
    Calculate the Heidke Skill Score (HSS) between observed and model output values based on a specified threshold.
    It compares the accuracy of a forecast relative to random chance. It accounts for all components of the contingency table (hits, misses, false alarms, and correct negatives).

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.

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
    cm = confusion_matrix(observed, output, threshold, var)
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
                       threshold:float = 0.5,
                       var:str =None) -> float:
    """
    Calculate the Peirce Skill Score (PSS),also known as the True Skill Statistic (TSS), between observed and model output values based on a specified threshold.
    It measures the ability of the outputs to distinguish between events and non-events, without being influenced by the base rate of the event.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values, where n is the number of samples, h is the height, and w is the width.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values, where n is the number of samples, h is the height, and w is the width.
    
    threshold : float
        A threshold value used to convert continuous output values into binary classifications (0 or 1).
        Values greater than or equal to the threshold will be classified as 1, and values below the threshold will be classified as 0.

    var : str (default: None)
        The name of the variable to be used in the calculation. If `var` is None, the function will use the first variable in the dataset.
        Only applicable when the inputs are provided as xarray Datasets.

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
    cm = confusion_matrix(observed, output, threshold, var)
    POD = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    POFD = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    return POD - POFD

def sedi(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         threshold:float =0.5,
         var:str =None) -> float:
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
    cm = confusion_matrix(observed, output, threshold, var)
    H = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    F = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    if H in [0, 1] or F in [0, 1]:
        return float('nan')  # Avoid division by zero and log(0)
    return (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))

#########################################
#     Calculate Pixelwise Metrics       #
#########################################



# Main Function
def calculate_pixelwise_metrics(observed: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame],
                                output: Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame],
                                metrics: Union[str, Tuple[str], List[str]] = None,
                                metric_type: str = None,
                                **kwargs) -> dict:
    """
    Calculate specified metrics (categorical or continuous) between observed and model output values.

    This function allows for the flexible calculation of both categorical and continuous metrics. Users can specify 
    the type of metrics they want to calculate using the `metric_type` parameter, and can provide a list of metrics 
    to be calculated. The metric names are case-insensitive, so both lowercase and uppercase names are accepted.

    Parameters
    ----------
    observed : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame]
        Array of shape (h, w) or (n, h, w) containing observed binary or continuous values.
        
    output : Union[np.ndarray,  xr.DataArray, xr.Dataset, pd.DataFrame]
        Array of shape (h, w) or (n, h, w) containing model output binary or continuous values.

    metrics : Union[str, Tuple[str], List[str]], optional, default=None
        A string, tuple, or list of strings specifying the categorical metrics to calculate.
        If not provided, all available metrics will be calculated. Available metrics are:

        Continuous metrics:
            - "MAE" (Mean Absolute Error)
            - "MSE" (Mean Squared Error)
            - "RMSE" (Root Mean Squared Error)
            - "Bias" (Frequency Bias)
            - "DRMSE" (Debiased Root Mean Squared Error)
            - "Corr" (Pearson Correlation)

        Categorical metrics:
            - "CM" (the Confusion Matrix)
            - "Precision" (Positive Predictive Value)
            - "Recall" (True Positive Rate)
            - "F1" (the harmonic mean of precision and recall)
            - "Accuracy" (the ratio of correct predictions to total predictions)
            - "CSI" (Critical Success Index)
            - "FAR" (False Alarm Ratio)
            - "POD" (Probability of Detection)
            - "GSS" (Gilbert Skill Score, also known as Equitable Threat Score)
            - "HSS" (Heidke Skill Score)
            - "PSS" (Peirce Skill Score)
            - "SEDI" (Symmetric Extremal Dependence Index)

    metric_type : str, optional, default=None
        A string specifying the type of metrics to calculate. Available options are:
        - "categorical" (for binary classification metrics)
        - "continuous" (for regression metrics)
        It is not necessary to specify the metric_type if the metrics parameter is provided.

            threshold : float, optional
        A threshold value used to convert continuous output values into binary classifications. 
        Default is 0.5.

    var : str, optional
        If the input data is an xarray Dataset, `var` specifies the variable name to extract for the calculation. 
        If not provided, the first variable in the Dataset will be used.

    Returns
    -------
    dict
        A dictionary where the keys are the names of the metrics and the values are the corresponding calculated values.

    """

    # Define available metrics
    cate_metrics = {
        "cm": confusion_matrix,
        "precision": precision,
        "recall": recall,
        "F1": f1_score,
        "accuracy": accuracy,
        "csi": critical_success_index,
        "far": false_alarm_ratio,
        "pod": probability_of_detection,
        "gss": gilbert_skill_score,
        "hss": heidke_skill_score,
        "pss": peirce_skill_score,
        "sedi": sedi
    }

    cont_metrics = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error,
        "bias": bias,
        "drmse": debiased_root_mean_squared_error,
        "corr": pearson_correlation
    }

    # Extract optional parameters from kwargs
    threshold = kwargs.get("threshold", 0.5)  # Default threshold is 0.5
    var = kwargs.get("var", None)
    observed = _to_numpy(observed,var)
    output = _to_numpy(output,var)

    # Convert metric names to lowercase to handle case insensitivity
    if isinstance(metrics, str):
        metrics = [metrics.lower()]
    elif isinstance(metrics, (tuple, list)):
        metrics = [m.lower() for m in metrics]

    

    # Select available metrics based on the specified type
    available_metrics = _get_available_metrics(metric_type, cate_metrics, cont_metrics)

    # If metrics is None, calculate all available metrics
    if metrics is None and metric_type is None:
        metrics = list(available_metrics.keys())
    

    # Separate and calculate categorical and continuous metrics
    cate_results = _calculate_categorical_metrics(observed, output, metrics, cate_metrics, threshold)
    cont_results = _calculate_continuous_metrics(observed, output, metrics, cont_metrics)

    return {**cate_results, **cont_results}
