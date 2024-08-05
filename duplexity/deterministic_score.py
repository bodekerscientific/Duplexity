import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Tuple, Union, Optional
from scipy.ndimage import uniform_filter
import scipy.signal
from skimage.draw import disk

from duplexity.utils import _check_shapes, _to_numpy, _binary_classification

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

def mean_absolute_error(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Mean Absolute Error (MAE).

    The MAE measures the average magnitude of the absolute errors between observed and predicted values, 
    providing a linear score that does not consider the direction of errors.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Mean Absolute Error (MAE).
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)
    return np.mean(np.abs(observed - output))

def mean_squared_error(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Mean Squared Error (MSE).

    The MSE measures the average of the squares of the errors, which is the average squared difference 
    between the estimated values and the actual value. It is a measure of the quality of an estimator.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Mean Squared Error (MSE).
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    return np.mean((observed - output) ** 2)

def root_mean_squared_error(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                            output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE).

    The RMSE is the square root of the average of squared differences between prediction and actual observation. 
    It is a measure of the differences between values predicted by a model and the values observed.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Root Mean Squared Error (RMSE).
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    return np.sqrt(np.mean((observed - output) ** 2))

def bias(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
         output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Bias.

    Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. 
    High bias can cause an algorithm to miss the relevant relations between features and target outputs.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Bias value.
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    return np.mean(output - observed)

def debiased_root_mean_squared_error(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                     output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Debiased Root Mean Squared Error (DRMSE).

    DRMSE adjusts the RMSE by removing the bias from the predictions before computing the error. 
    This provides a better indication of the accuracy of predictions by compensating for systematic errors.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Debiased Root Mean Squared Error (DRMSE).
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    bias_value = np.mean(output - observed)
    debiased_predictions = output - bias_value
    return np.sqrt(np.mean((observed - debiased_predictions) ** 2))

def pearson_correlation(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Pearson correlation coefficient.

    The Pearson correlation coefficient measures the linear correlation between two sets of data. 
    It ranges from -1 to 1, where 1 is total positive linear correlation, 0 is no linear correlation, 
    and -1 is total negative linear correlation.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)

    return np.corrcoef(observed.flatten(), output.flatten())[0, 1]

def calculate_continuous_metrics(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                 output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                 metrics: Union[str, Tuple[str], List[str]] = None) -> dict:
    """
    Calculate all defined continuous metrics and return them as a dictionary.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.
    metrics : Union[str, Tuple[str], List[str]], optional
        Specific metric(s) to calculate, by default None. If None, all metrics are calculated.

    Returns
    -------
    dict
        Dictionary with calculated metric names and their values.
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

def confusion_matrix(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                     output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                     threshold:float) -> np.array:
    """
    Calculate the confusion matrix.

    The confusion matrix is used to evaluate the accuracy of a classification model by comparing the 
    predicted and actual classifications.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    np.array
        Confusion matrix.
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

def precision(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
              output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
              threshold:float) -> float:
    """
    Calculate the precision score.

    Precision is the ratio of correctly predicted positive observations to the total predicted positives. 
    It is also called Positive Predictive Value.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Precision score.
    """
    cm = confusion_matrix(observed, output, threshold)
    return cm[0, 0] / (cm[0, 0] + cm[1, 0])

def recall(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
           output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
           threshold:float) -> float:
    """
    Calculate the recall score.

    Recall is the ratio of correctly predicted positive observations to all observations in the actual class. 
    It is also called Sensitivity or True Positive Rate.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Recall score.
    """
    cm = confusion_matrix(observed, output, threshold)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

def f1_score(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
             output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
             threshold:float) -> float:
    """
    Calculate the F1 score.

    The F1 score is the weighted average of Precision and Recall. 
    It is useful when the class distribution is imbalanced.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        F1 score.
    """
    precision_value = precision(observed, output, threshold)
    recall_value = recall(observed, output, threshold)
    return 2 * (precision_value * recall_value) / (precision_value + recall_value)

def accuracy(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
             output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
             threshold:float) -> float:
    """
    Calculate the accuracy score.

    Accuracy is the ratio of correctly predicted observations to the total observations. 
    It is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Accuracy score.
    """
    cm = confusion_matrix(observed, output, threshold)
    return (cm[0, 0] + cm[1, 1]) / cm.sum()

def critical_success_index(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           threshold:float) -> float:
    """
    Calculate the Critical Success Index (CSI).

    CSI, also known as the Threat Score, is the ratio of correctly predicted events to the sum of the correctly 
    predicted events, false alarms, and missed events.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Critical Success Index (CSI).
    """
    cm = confusion_matrix(observed, output, threshold)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[1, 0])

def equitable_threat_score(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                           threshold:float) -> float:
    """
    Calculate the Equitable Threat Score (ETS).

    ETS is a measure of the skill of a binary classifier, considering the possibility of random hits. 
    It ranges from -1/3 to 1, with 1 being a perfect score.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Equitable Threat Score (ETS).
    """
    cm = confusion_matrix(observed, output, threshold)
    hits_random = (cm[0, 0] + cm[1, 0]) * (cm[0, 0] + cm[0, 1]) / cm.sum()
    return (cm[0, 0] - hits_random) / (cm[0, 0] + cm[0, 1] + cm[1, 0] - hits_random)

def false_alarm_ratio(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                      output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                      threshold:float) -> float:
    """
    Calculate the False Alarm Ratio (FAR).

    FAR is the ratio of the number of false alarms to the total number of events that were forecast.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        False Alarm Ratio (FAR).
    """
    cm = confusion_matrix(observed, output, threshold)
    return cm[1, 0] / (cm[0, 0] + cm[1, 0])

def probability_of_detection(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                             output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                             threshold:float) -> float:
    """
    Calculate the Probability of Detection (POD).

    POD, also known as Hit Rate or Sensitivity, is the ratio of correctly predicted positive observations 
    to all observations in the actual class.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Probability of Detection (POD).
    """
    cm = confusion_matrix(observed, output, threshold)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

def gilbert_skill_score(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                        threshold:float) -> float:
    """
    Calculate the Gilbert Skill Score (GSS).

    GSS measures the skill of a forecast relative to random chance, considering both hits and false alarms. 
    It ranges from -1/3 to 1, where 1 indicates a perfect score.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Gilbert Skill Score (GSS).
    """
    cm = confusion_matrix(observed, output, threshold)
    hits_random = (cm[0, 0] + cm[1, 0]) * (cm[0, 0] + cm[0, 1]) / cm.sum()
    return (cm[0, 0] - hits_random) / (cm[0, 0] + cm[0, 1] + cm[1, 0] - hits_random)

def heidke_skill_score(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       threshold:float) -> float:
    """
    Calculate the Heidke Skill Score (HSS).

    HSS measures the skill of a forecast relative to random chance, adjusting for both hits and correct negatives. 
    It ranges from -∞ to 1, with 1 being a perfect score.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Heidke Skill Score (HSS).
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

def peirce_skill_score(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                       threshold:float) -> float:
    """
    Calculate the Peirce Skill Score (PSS).

    PSS measures the ability of a forecast to discriminate between events and non-events, adjusting for 
    both hits and false alarms. It ranges from -1 to 1, with 1 indicating a perfect score.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Peirce Skill Score (PSS).
    """
    cm = confusion_matrix(observed, output, threshold)
    POD = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    POFD = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    return POD - POFD

def sedi(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
         output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
         threshold:float) -> float:
    """
    Calculate the Symmetric Extremal Dependence Index (SEDI).

    SEDI measures the skill of a forecast in discriminating between events and non-events, particularly at 
    the extremes of the distribution. It ranges from -∞ to 1, with 1 being a perfect score.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.

    Returns
    -------
    float
        Symmetric Extremal Dependence Index (SEDI).
    """
    cm = confusion_matrix(observed, output, threshold)
    H = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    F = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    if H in [0, 1] or F in [0, 1]:
        return float('nan')  # Avoid division by zero and log(0)
    return (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))

def calculate_categorical_metrics(observed: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                  output: Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                                  metrics: Union[str, Tuple[str], List[str]] = None,
                                  threshold: float = 0.5) -> dict:
    """
    Calculate all defined categorical metrics and return them as a dictionary.

    Parameters
    ----------
    observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Observed values.
    output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
        Model output values.
    metrics : Union[str, Tuple[str], List[str]], optional
        Specific metric(s) to calculate, by default None. If None, all metrics are calculated.

    Returns
    -------
    dict
        Dictionary with calculated metric names and their values.
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

def fss_init(threshold: float, scale: int) -> dict:
    """
    Initialize a fractions skill score (FSS) verification object.
    
    Parameters:
        threshold (float): Threshold value for binarizing the data.
        scale (int): Size of the neighborhood for calculating fractions.

    """
    fss = dict(threshold=threshold, scale=scale, sum_output_sq=0.0, sum_output_observed=0.0, sum_observed_sq=0.0)
    return fss



def fss_update(fss: dict, output: np.ndarray, observed: np.ndarray) -> None:
    """
    Update the FSS object with new forecast and observed data.
    
    Parameters:
    fss (dict): FSS verification object.
    output (np.array): Forecasted data array.
    observed (np.array): Observed data array.
    """
    if len(output.shape) != 2 or len(observed.shape) != 2 or output.shape != observed.shape:
        raise ValueError("Forecast and observation must be two-dimensional having the same dimensions")

    threshold = fss['threshold']
    scale = fss['scale']

    binary_output = _binary_classification(output, threshold)
    binary_observed = _binary_classification(observed, threshold)
    
    if fss["scale"] > 1:
        smoothed_forecast = uniform_filter(binary_output, size=scale, mode="constant", cval=0.0)
        smoothed_observation = uniform_filter(binary_observed, size=scale, mode="constant", cval=0.0)
    else:
        smoothed_forecast = binary_output
        smoothed_observation = binary_observed


    fss["sum_observation_power"] += np.nansum(smoothed_observation ** 2)
    fss["sum_forecast_observation_correlation"] += np.nansum(smoothed_forecast * smoothed_observation)
    fss["sum_forecast_power"] += np.nansum(smoothed_forecast ** 2)


def fss_compute(fss: dict) -> float:
    """
    Calculate the Fractions Skill Score (FSS).
    
    Parameters:
    fss (dict): FSS verification object.
    
    Returns:
    float: Fractions Skill Score.
    """

    sum_output_sq = fss['sum_output_sq']
    sum_observed_sq = fss['sum_obs_sq']
    sum_output_observed = fss['sum_output_observed']

    numerator = sum_output_sq + sum_observed_sq - 2 * sum_output_observed
    denominator = sum_output_sq + sum_observed_sq

    if denominator == 0:
        return np.nan

    fss_values = 1 - numerator / denominator
    return fss_values

def calculate_fss_score(output:Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                   observed:Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]],
                   thresholds: Union[float,List[float]], scales:Union[int, List[int]]) -> float:
    """
    Calculate the Fractions Skill Score (FSS) for the given forecast and observed data.
    
    Parameters:
    forecast (np.ndarray): Forecasted data.
    observed (np.ndarray): Observed data.
    threshold (float): Threshold value for binarizing the data.
    scales (Union[int, List[int]]): Size of the neighborhood for calculating fractions.
    
    Returns:
    float: Fractions Skill Score (FSS) value.
    """
    output = _to_numpy(output)
    observed = _to_numpy(observed)
    _check_shapes(output, observed)

    if isinstance(thresholds, float):
        thresholds = [thresholds]

    if isinstance(scales, int):
        scales = [scales]


    fss_scores = []
    for scale in scales:
        for threshold in thresholds:
            fss = fss_init(threshold, scale)
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


def validate_with_psd(observed: np.ndarray, forecasted: np.ndarray) -> np.ndarray:
    """
    Validate the forecasted data with the Precipitation Symmetry Distance (PSD).
    
    Parameters:
    observed (np.ndarray): Observed data array.
    forecasted (np.ndarray): Forecasted data array.
    
    Returns:
    np.ndarray: Array of PSD values.
    """
    psd_results = []
    for i in tqdm(range(forecasted.shape[0]), desc="Calculating PSD"):
        observed_slice = observed[i]
        forecasted_slice = forecasted[i]
        psd = calculate_psd(observed_slice, forecasted_slice)
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
