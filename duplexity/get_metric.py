
from typing import Callable
from pixelwise import *


def _get_pixelwise_function(metric_name: str) -> Callable:
    """
    Get the callable metric function from the given metric name or function.
    
    Parameters
    ----------
    metric_name : str

        The metric to retrieve, either as a function or string name.

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
            
    
    Returns
    -------
    Callable
        The metric function.
    
    Raises
    ------
    ValueError
        If the metric is not recognized.
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

    # Convert metric_name to lowercase for case-insensitive matching
    metric_name = metric_name.lower()

    all_metrics = {**cate_metrics, **cont_metrics}

    # Check if metric_name exists in the combined metrics
    if metric_name in all_metrics:
        return all_metrics[metric_name]
    else:
        raise ValueError(f"Invalid metric name: {metric_name}. Available metrics are: {list(all_metrics.keys())}")








