import numpy as np
from typing import Union, Callable, Tuple, List, Dict
from pixelwise import *
import numpy as np
import xarray as xr
import pandas as pd
from typing import Union, Callable, Dict, List, Tuple
from duplexity.pixelwise import *
from duplexity.utils import _to_numpy, _check_shapes,_binary_classification, _get_available_metrics, _calculate_categorical_metrics, _calculate_continuous_metrics, _get_metric_function
import itertools



data_type = Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]
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




def ensemble_metric(observed: data_type,
                    ensemble_output: data_type,
                    metric_func: Union[str, Tuple[str], List[str]] = None,
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



    # Extract optional parameters from kwargs
    threshold = kwargs.get("threshold", 0.5)  # Default threshold is 0.5
    var = kwargs.get("var", None)
    # Convert observed and output to numpy arrays
    observed = _to_numpy(observed, var)
    ensemble_output = _to_numpy(ensemble_output, var)


    # Convert metric names to lowercase to handle case insensitivity
    if isinstance(metrics, str):
        metrics = [metrics.lower()]
    elif isinstance(metrics, (tuple, list)):
        metrics = [m.lower() for m in metrics]


    # If mean is True, calculate metrics for the mean of ensemble members
    if mean:
        ensemble_output = np.mean(ensemble_output, axis=0)
        cate_results = _calculate_categorical_metrics(observed, ensemble_output, metrics, cate_metrics, threshold)
        cont_results = _calculate_continuous_metrics(observed, ensemble_output, metrics, cont_metrics)
        return {**cate_results, **cont_results}
    
    # Otherwise, calculate the metric for each ensemble member
    metrics_result = []
    for member_data in ensemble_output:
        member_result = {}
        member_result.update(_calculate_categorical_metrics(observed, member_data, metrics, cate_metrics, threshold))
        member_result.update(_calculate_continuous_metrics(observed, member_data, metrics, cont_metrics))
        metrics_result.append(member_result)

    return metrics_result




def ensemble_pairwise_skill(ensemble_output: np.ndarray, metric_func: Union[Callable, str]) -> float:
    """
    Compute the mean skill between all possible pairs of ensemble members.

    Parameters
    ----------
    ensemble_output : Union[np.ndarray, List[np.ndarray]]
        Ensemble forecast data. Should be of shape (n_members, h, w), where `n_members` is the number of ensemble members,
        and `h`, `w` are the height and width of the grid.
    skill_func : Callable
        The function that computes the skill between two ensemble members.

    Returns
    -------
    float
        The mean skill computed between all possible pairs of the ensemble members.
    """

    # Ensure the ensemble data has 3 dimensions
    if ensemble_output.ndim != 3:
        raise ValueError(f"The input ensemble_output should have 3 dimensions (n_members, h, w), but got {ensemble_output.ndim} dimensions.")

    # Number of ensemble members
    n_members = ensemble_output.shape[0]

    # List to store skill scores for each pair
    skill_scores = []
    available_metrics = {**cate_metrics, **cont_metrics}
    metric_func = _get_metric_function(metric_func, available_metrics)

    # Generate all possible pairs of ensemble members
    for (i, j) in itertools.combinations(range(n_members), 2):
        # Compute the skill between two ensemble members using the skill function
        skill = metric_func(ensemble_output[i], ensemble_output[j])
        skill_scores.append(skill)

    # Compute the mean skill
    mean_skill = np.mean(skill_scores)

    return mean_skill


























