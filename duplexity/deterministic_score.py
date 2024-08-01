
import numpy as np
import xarray as xr
import pandas as pd
from typing import List, Tuple, Union, Optional



class DeterministicScore:
    """
    A class used to validate neural network model outputs against observed data.

    Methods
    -------
    mean_absolute_error() -> float
        Calculate Mean Absolute Error (MAE).
    mean_squared_error() -> float
        Calculate Mean Squared Error (MSE).
    root_mean_squared_error() -> float
        Calculate Root Mean Squared Error (RMSE).
    bias() -> float
        Calculate the Bias between observed and output values.
    debiased_root_mean_squared_error() -> float
        Calculate the debiased root mean squared error (DRMSE).
    pearson_correlation() -> float
        Calculate the Pearson correlation coefficient between observed and output values.
        Calculate the Symmetric Extremal Dependence Index (SEDI).
    calculate_metrics() -> dict
        Calculate all defined metrics and return them as a dictionary.
    """
    def __init__(self, 
                 observed: Union[
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
                 is_categorical: bool = False, 
                 threshold: float = 0.5):
        """
        Initialize the DeterministicScore class with observed and output data.

        Parameters
        ----------
        observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
            Array containing the observed values.
        output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
            Array containing the output values.
        is_categorical : bool
            Flag to indicate if the data is categorical.
        threshold : float  
            Threshold value to binarize the continuous outputs.

        Raises
        ------
        ValueError
            If the shape of output does not match the shape of observed.
        """
        self.observed = self._to_numpy(observed)
        self.output = self._to_numpy(output)
        self.threshold = threshold
        self.is_categorical = is_categorical
                
        self._check_shapes()

        self.output_binary = self.output >= self.threshold
        self.observed_binary = self.observed >= self.threshold


    def _to_numpy(self, data) -> np.array:
        """
        Convert input data to numpy array.
        
        Parameters
        ----------
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
            return np.array([d.to_numpy() if isinstance(d, xr.DataArray) else d.values for d in data])
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unsupported data type")
        
    def _check_shapes(self):
        if self.observed.shape != self.output.shape:
            raise ValueError("Observed and output data must have the same shape.")


    def mae(self) -> float:
        """
        Calculate Mean Absolute Error (MAE).

        Returns
        -------
        float
            Mean Absolute Error (MAE)
        """
        return np.mean(np.abs(self.observed - self.output))

    def mse(self) -> float:
        """
        Calculate Mean Squared Error (MSE).

        Returns
        -------
        float
            Mean Squared Error (MSE)
        """
        return np.mean((self.observed - self.output) ** 2)

    def rmse(self) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).

        Returns
        -------
        float
            Root Mean Squared Error (RMSE)
        """
        return np.sqrt(np.mean((self.observed - self.output) ** 2))

    def bias(self) -> float:
        """
        Calculate the Bias between observed and output values.

        Returns
        -------
        float
            The Bias value.
        """
        return np.mean(self.output - self.observed)
    
    def drmes(self) -> float:
        """
        Calculate the debiased root mean squared error (DRMSE).

        Returns
        -------
        float
            Debiased Root Mean Squared Error (DRMSE)
        """
        bias = np.mean(self.output - self.observed)
        debiased_predictions = self.output - bias
        return np.sqrt(np.mean((self.observed - debiased_predictions) ** 2))

    def pearson_corr(self) -> float:
        """
        Calculate the Pearson correlation coefficient between observed and output values.

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        return np.corrcoef(self.observed.flatten(), self.output.flatten())[0, 1]


    def calculate_metrics(self, metrics:Union[str,tuple] = None) -> dict:
        """
        Calculate all defined metrics and return them as a dictionary.

        Returns
        -------
        dict
            A dictionary with all metrics results.
        """
        if metrics is None:
            results = {
                "MSE": self.mse(),
                "RMSE": self.rmse(),
                "MAE": self.mae(),
                "Bias": self.bias(),
                "Correlation": self.pearson_corr()
            }
        elif isinstance(metrics, str):
            results = {
                metrics: getattr(self, metrics)()
            }

            return results



class CategoricalScore:

    """
    Class to calculate scores for categorical data.

    Methods
    -------
    confusion_matrix() -> np.array
        Calculate the confusion matrix.
    precision() -> float
        Calculate the precision score.
    recall() -> float
        Calculate the recall score.
    f1_score() -> float
        Calculate the F1 score.
    accuracy() -> float
        Calculate the accuracy score.
    critical_success_index() -> float
        Calculate the Critical Success Index (CSI).
    equitable_threat_score() -> float
        Calculate the Equitable Threat Score (ETS).
    false_alarm_ratio() -> float
        Calculate the False Alarm Ratio (FAR).
    probability_of_detection() -> float
        Calculate the Probability of Detection (POD).
    gilbert_skill_score() -> float
        Calculate the Gilbert Skill Score (GSS).
    heidke_skill_score() -> float
        Calculate the Heidke Skill Score (HSS).
    peirce_skill_score() -> float
        Calculate the Peirce Skill Score (PSS).
    sedi() -> float
        Calculate the Symmetric Extremal Dependence Index (SEDI).
    calculate_metrics() -> dict
        Calculate all defined metrics and return them as a dictionary.
    """

    def __init__(self, 
                 observed: Union[
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
                 threshold: float = 0.5):
        """
        Initialize the ContinuousScore class.

        Parameters
        ----------
        observed : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
            Observed data.
        output : Union[np.array, xr.DataArray, pd.DataFrame, List[Union[xr.DataArray, xr.Dataset, pd.DataFrame]]]
            Output data.
        threshold : float, optional
            Threshold value to convert continuous data to binary, by default 0.5.
        """
                
        self.observed = self._to_numpy(observed)
        self.output = self._to_numpy(output)
        self.threshold = threshold                
        self._check_shapes()

        self.output_binary = self.output >= self.threshold
        self.observed_binary = self.observed >= self.threshold


    def _to_numpy(self, data) -> np.array:
        """
        Convert input data to numpy array.
        
        Parameters
        ----------
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
            return np.array([d.to_numpy() if isinstance(d, xr.DataArray) else d.values for d in data])
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Unsupported data type")
        
    def _check_shapes(self):
        if self.observed.shape != self.output.shape:
            raise ValueError("Observed and output data must have the same shape.")
    
    def confusion_matrix(self):
        
        """
        Calculate the confusion matrix.
        
        Returns
        -------
            np.array
        """
        TP = np.logical_and(self.output_binary == 1, self.observed_binary == 1) # True Positives, hits
        FN = np.logical_and(self.output_binary == 0, self.observed_binary == 1) # False Negatives
        FP = np.logical_and(self.output_binary == 1, self.observed_binary == 0) # False Positives
        TN = np.logical_and(self.output_binary == 0, self.observed_binary == 0) # True Negatives, Correct Rejects
        return np.array([[TP.sum(), FN.sum()], [FP.sum(), TN.sum()]])
    
    def precision(self):
        """
        Calculate the precision score.
        
        Returns
        -------
            float
        """
        cm = self.confusion_matrix()
        return cm[0, 0] / (cm[0, 0] + cm[1, 0])

    
    def recall(self):
        """
        Calculate the recall score.
        
        Returns
        -------
            float
        """
        cm = self.confusion_matrix()
        return cm[0, 0] / (cm[0, 0] + cm[0, 1])
    
    def f1_score(self):
        """
        Calculate the F1 score.
        
        Returns
        -------
            float
        """
        precision = self.precision()
        recall = self.recall()
        return 2 * (precision * recall) / (precision + recall)
    
    def accuracy(self) -> float:
        """
        Calculate the accuracy score.

        Returns
        -------
        float
            Accuracy score.
        """
        cm = self.confusion_matrix()
        return (cm[0, 0] + cm[1, 1]) / cm.sum()


    def csi(self) -> float:
        """
        Calculate the Critical Success Index (CSI).

        Returns
        -------
        float
            Critical Success Index (CSI).
        """
        cm = self.confusion_matrix()
        return cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[1, 0])

    def ets(self) -> float:
        """
        Calculate the Equitable Threat Score (ETS).

        Returns
        -------
        float
            Equitable Threat Score (ETS).
        """
        cm = self.confusion_matrix()
        hits_random = (cm[0, 0] + cm[1, 0]) * (cm[0, 0] + cm[0, 1]) / cm.sum()
        return (cm[0, 0] - hits_random) / (cm[0, 0] + cm[0, 1] + cm[1, 0] - hits_random)



    def far(self) -> float: 
        """
        Calculate the False Alarm Ratio (FAR).

        Returns
        -------
        float
            False Alarm Ratio (FAR).
        """
        cm = self.confusion_matrix()
        return cm[1, 0] / (cm[0, 0] + cm[1, 0])

    def pod(self) -> float:
        """
        Calculate the Probability of Detection (POD).

        Returns
        -------
        float
            Probability of Detection (POD).
        """
        cm = self.confusion_matrix()
        return cm[0, 0] / (cm[0, 0] + cm[0, 1])

    def gss(self) -> float:
        """
        Calculate the Gilbert Skill Score (GSS).

        Returns
        -------
        float
            Gilbert Skill Score (GSS).
        """
        cm = self.confusion_matrix()
        hits_random = (cm[0, 0] + cm[1, 0]) * (cm[0, 0] + cm[0, 1]) / cm.sum()
        return (cm[0, 0] - hits_random) / (cm[0, 0] + cm[0, 1] + cm[1, 0] - hits_random)
    

    def hss(self) -> float:
        """
        Calculate the Heidke Skill Score (HSS).

        Returns
        -------
        float
            Heidke Skill Score (HSS).

        """    
        cm = self.confusion_matrix()
        hits = cm[1, 1]
        false_alarms = cm[0, 1]
        misses = cm[1, 0]
        correct_negatives = cm[0, 0]
        total = hits + false_alarms + misses + correct_negatives
        accuracy_random = ((hits + false_alarms) * (hits + misses) + (correct_negatives + misses) * (correct_negatives + false_alarms)) / (total * total)
        accuracy_observed = (hits + correct_negatives) / total
        return (accuracy_observed - accuracy_random) / (1 - accuracy_random) if (1 - accuracy_random) != 0 else 0

    




    def pss(self) -> float:
        """
        Calculate the Peirce Skill Score (PSS).

        Returns
        -------
        float
            Peirce Skill Score (PSS).
        """
        cm = self.confusion_matrix()
        POD = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        POFD = cm[1, 0] / (cm[1, 0] + cm[1, 1])
        return POD - POFD

        


    def sedi(self) -> float:
        """
        Calculate the Symmetric Extremal Dependence Index (SEDI).

        Returns
        -------
        float
            Symmetric Extremal Dependence Index (SEDI).
        """
        cm = self.confusion_matrix()
        H = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        F = cm[1, 0] / (cm[1, 0] + cm[1, 1])
        if H in [0, 1] or F in [0, 1]:
            return float('nan')  # Avoid division by zero and log(0)
        return (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))



    def calculate_metrics(self) -> dict:
        """
        Calculate all defined metrics and return them as a dictionary.

        Returns
        -------
        dict
            A dictionary with all metrics results.
        """
        results = {
            "Accuracy": self.accuracy(),
            "CSI": self.csi(),
            "ETS": self.ets(),
            "FAR": self.far(),
            "POD": self.pod(),
            "GSS": self.gss(),
            "HSS": self.hss(),
            "PSS": self.pss(),
            "SEDI": self.sedi()
        }
        return results

