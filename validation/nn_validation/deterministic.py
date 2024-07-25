import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from scipy.ndimage import uniform_filter
import xarray as xr
from typing import Union


class DeterminsticScore:
    """
    A class used to validate neural network model outputs against observed data.

    Attributes
    ----------
    observed : Union[np.array, xr.DataArray]
        Array containing the observed values.
    output : Union[np.array, xr.DataArray]
        Array containing the output values.
    threshold : float
        Threshold value to binarize the continuous outputs.
    


    Methods
    -------
    mean_absolute_error()
        Calculate Mean Absolute Error (MAE).
    mean_squared_error()
        Calculate Mean Squared Error (MSE).
    root_mean_squared_error()   
        Calculate Root Mean Squared Error (RMSE).
    bias()
        Calculate the Bias between observed and output values.
    debiased_root_mean_squared_error()
        Calculate the debiased root mean squared error (DRMSE).
    pearson_correlation()
        Calculate the Pearson correlation coefficient between observed and output values.
    calculate_metrics()
        Calculate all defined metrics and return them as a dictionary.
    

    """
    def __init__(self, observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray]):
        """
        Initialize the class with observed and output data.

        Parameters
        ----------
        observed : Union[np.array, xr.DataArray]
            Array containing the observed values.
        output : Union[np.array, xr.DataArray]
            Array containing the output values.
        """
        self.observed = observed.as_numpy() if isinstance(observed, xr.DataArray) else observed
        self.output = output.as_numpy() if isinstance(output, xr.DataArray) else output
        
        # checks
        if observed.shape != output.shape:
            raise ValueError(
                "the shape of output does not match the shape of observed %s!=%s"
                % (output.shape, observed.shape)
            )

    def mean_absolute_error(self) -> float:
        """
        Calculate Mean Absolute Error (MAE).

        Returns
        -------
        float
            Mean Absolute Error (MAE)
        """
        return np.mean(np.abs(self.observed - self.output))

    def mean_squared_error(self) -> float:
        """
        Calculate Mean Squared Error (MSE).

        Returns
        -------
        float
            Mean Squared Error (MSE)
        """
        return np.mean((self.observed - self.output) ** 2)

    def root_mean_squared_error(self) -> float:
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
    
    def debiased_root_mean_squared_error(self) -> float:
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

    def pearson_correlation(self) -> float:
        """
        Calculate the Pearson correlation coefficient between observed and output values.

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        return np.corrcoef(self.observed.flatten(), self.output.flatten())[0, 1]
    
    def calculate_metrics(self) -> dict:
        """
        Calculate all defined metrics and return them as a dictionary.

        Returns
        -------
        dict
            A dictionary with all metrics results.
        """
        results = {
            "MSE": self.mean_squared_error(),
            "RMSE": self.root_mean_squared_error(),
            "MAE": self.mean_absolute_error(),
            "Bias": self.bias(),
            "Correlation": self.pearson_correlation()
        }
        return results



    def __init__(self, observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray], threshold: float):
        """
        Initialize the NNModelValidation class with observed and output data.

        Parameters
        ----------
        observed : Union[np.array, xr.DataArray]
            Array containing the observed values.
        output : Union[np.array, xr.DataArray]
            Array containing the output values.
        """
        self.observed = observed
        self.output = output
        self.threshold = threshold
        
        # checks
        if observed.shape != output.shape:
            raise ValueError(
                "the shape of output does not match the shape of observed %s!=%s"
                % (output.shape, observed.shape)
            )

        self.output_binary = self.output >= self.threshold
        self.observed_binary = self.observed >= self.threshold

    def calculate(self):
        TP = np.logical_and(self.output_binary == 1, self.observed_binary == 1) #True Positives,hits
        FN = np.logical_and(self.output_binary == 0, self.observed_binary == 1) #False Negatives
        FP = np.logical_and(self.output_binary == 1, self.observed_binary == 0) #False Positives
        TN = np.logical_and(self.output_binary == 0, self.observed_binary == 0) #True Negatives, Correct Rejects
        return TP, FN, FP, TN

    def accuracy(self) -> float:
        """
        Calculate the accuracy score.

        Returns
        -------
        float
            Accuracy score.
        """
        TP, FN, FP, TN = self.calculate()
        return (TP.sum() + TN.sum()) / (TP.sum() + FN.sum() + FP.sum() + TN.sum())

    def critical_success_index(self) -> float:
        """
        Calculate the Critical Success Index (CSI).

        Returns
        -------
        float
            Critical Success Index (CSI).
        """
        TP, FN, FP, TN = self.calculate()
        return TP.sum() / (TP.sum() + FN.sum() + FP.sum())

    def equitable_threat_score(self) -> float:
        """
        Calculate the Equitable Threat Score (ETS).

        Returns
        -------
        float
            Equitable Threat Score (ETS).
        """
        TP, FN, FP, TN = self.calculate()
        return (TP.sum() - (TP.sum() + FN.sum()) * (TP.sum() + FP.sum()) / (TP.sum() + FN.sum() + FP.sum() + TN.sum())) / (TP.sum() + FN.sum() + FP.sum() - (TP.sum() + FN.sum()) * (TP.sum() + FP.sum()) / (TP.sum() + FN.sum() + FP.sum() + TN.sum()))

    def false_alarm_ratio(self) -> float: 
        """
        Calculate the False Alarm Ratio (FAR).

        Returns
        -------
        float
            False Alarm Ratio (FAR).
        """
        TP, FN, FP, TN = self.calculate()
        return FP.sum() / (TP.sum() + FP.sum())

    def probability_of_detection(self) -> float:
        """
        Calculate the Probability of Detection (POD).

        Returns
        -------
        float
            Probability of Detection (POD).
        """
        TP, FN, FP, TN = self.calculate()
        return TP.sum() / (TP.sum() + FN.sum())

    def gilbert_skill_score(self) -> float:
        """
        Calculate the Gilbert Skill Score (GSS).

        Returns
        -------
        float
            Gilbert Skill Score (GSS).
        """
        TP, FN, FP, TN = self.calculate()
        return (TP.sum() - (TP.sum() + FP.sum()) * (TP.sum() + FN.sum()) / (TP.sum() + FN.sum() + FP.sum() + TN.sum())) / (TP.sum() + FN.sum() + FP.sum())

    def heidke_skill_score(self) -> float:
        """
        Calculate the Heidke Skill Score (HSS).

        Returns
        -------
        float
            Heidke Skill Score (HSS).
        """
        TP, FN, FP, TN = self.calculate()
        return (2 * (TP.sum() * TN.sum() - FN.sum() * FP.sum())) / ((TP.sum() + FN.sum()) * (FN.sum() + TN.sum()) + (TP.sum() + FP.sum()) * (FP.sum() + TN.sum()))

    def peirce_skill_score(self) -> float:
        """
        Calculate the Peirce Skill Score (PSS).

        Returns
        -------
        float
            Peirce Skill Score (PSS).
        """
        TP, FN, FP, TN = self.calculate()
        return (TP.sum() / (TP.sum() + FN.sum())) - (FP.sum() / (FP.sum() + TN.sum()))

    def sedi(self) -> float:
        """
        Calculate the Symmetric Extremal Dependence Index (SEDI).

        Returns
        -------
        float
            Symmetric Extremal Dependence Index (SEDI).
        """
        TP, FN, FP, TN = self.calculate()
        H = TP.sum() / (TP.sum() + FN.sum())
        F = FP.sum() / (FP.sum() + TN.sum())
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
            "CSI": self.critical_success_index(),
            "ETS": self.equitable_threat_score(),
            "FAR": self.false_alarm_ratio(),
            "POD": self.probability_of_detection(),
            "GSS": self.gilbert_skill_score(),
            "HSS": self.heidke_skill_score(),
            "PSS": self.peirce_skill_score(),
            "SEDI": self.sedi()
        }
        return results
