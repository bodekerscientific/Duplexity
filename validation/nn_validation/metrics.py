"""
Neural Network Model Validation Script

This script contains the NNModelValidation class, which is used to validate neural network model outputs
against observed data. It calculates various performance metrics.

Requirements:
- numpy
- scipy
- matplotlib
- scikit-learn
- tqdm
- properscoring

Author(s): Lexi Xu
GitHub Repository: 
Conda Environment: See environment.yaml file for the conda environment setup
"""




import unittest
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from scipy.ndimage import uniform_filter
#import properscoring as ps
from typing import Union
import xarray as xr



class NNModelValidation:
    """
    A class used to validate neural network model outputs against observed data.

    Attributes
    ----------
    observed : Union[np.array, xr.DataArray]
        Array containing the observed values.
    output : Union[np.array, xr.DataArray]
        Array containing the output values.
    """

    def __init__(self, observed: Union[np.array, xr.DataArray], output: Union[np.array, xr.DataArray]):
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

    def pearson_correlation(self) -> float:
        """
        Calculate the Pearson correlation coefficient between observed and output data.

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        observed = self.observed.flatten()
        output = self.output.flatten()

        # Remove NaN values from both arrays
        mask = ~np.isnan(observed) & ~np.isnan(output)
        observed = observed[mask]
        output = output[mask]

        # Calculate Pearson correlation coefficient using NumPy's function
        correlation_matrix = np.corrcoef(observed, output)

        return correlation_matrix[0, 1]
    

    

    def fss(self, threshold: float, neighborhood_size: int) -> float:
        """
        Calculate the Fractions Skill Score (FSS) for a given threshold and neighborhood size.

        Parameters
        ----------
        threshold : float
            The threshold value for binary conversion.
        neighborhood_size : int
            The size of the neighborhood for fraction calculation.

        Returns
        -------
        float
            Fractions Skill Score (FSS)
        """
        def fss_init(thr, scale):
            """ Initialize the FSS object with the given threshold and neighborhood size. """
            fss = dict(thr=thr, scale=scale, sum_fct_sq=0.0, sum_fct_obs=0.0, sum_obs_sq=0.0)
            return fss

        def fss_accum(fss):
            binary_output = (self.output >= fss["thr"]).astype(np.single)
            binary_observed = (self.observed >= fss["thr"]).astype(np.single)

            if fss["scale"] > 1:
                smoothed_observed = uniform_filter(binary_observed, size=fss["scale"], mode="constant", cval=0.0)
            else:
                smoothed_observed = binary_observed

            if fss["scale"] > 1:
                smoothed_output = uniform_filter(binary_output, size=fss["scale"], mode="constant", cval=0.0)
            else:
                smoothed_output = binary_output

                fss["sum_obs_sq"] += np.nansum(smoothed_observed ** 2)
                fss["sum_fct_obs"] += np.nansum(smoothed_output * smoothed_observed)
                fss["sum_fct_sq"] += np.nansum(smoothed_output ** 2)

        def fss_compute(fss):
            numer = fss["sum_fct_sq"] - 2.0 * fss["sum_fct_obs"] + fss["sum_obs_sq"]
            denom = fss["sum_fct_sq"] + fss["sum_obs_sq"]
            return 1.0 - numer / denom

        fss = fss_init(threshold, neighborhood_size)
        fss_accum(fss)
        return fss_compute(fss)
    
    def calculate_fss_metrics(self, precip_values, neighborhood_size):
        """
        Calculate FSS metrics for multiple threshold values.

        Parameters
        ----------
        precip_values : list
            List of precipitation threshold values.
        neighborhood_size : int
            The size of the neighborhood for fraction calculation.

        Returns
        -------
        dict
            Dictionary containing FSS scores for each threshold.
        """
        fss_results = {}
        for threshold in tqdm(precip_values, desc="Calculating FSS"):
            fss_score = self.fss(threshold, neighborhood_size)
            fss_results[threshold] = fss_score
        return fss_results

    

    def extreme_dependency_score(self, threshold: float) -> float:
        """
        Calculate the Extreme Dependency Score (EDS).

        Parameters
        ----------
        threshold : float
            The threshold value for binary conversion.

        Returns
        -------
        float
            Extreme Dependency Score (EDS)
        """
        observed_extreme = (self.observed >= threshold).astype(int)
        forecasted_extreme = (self.output >= threshold).astype(int)
        
        hit = np.sum((forecasted_extreme == 1) & (observed_extreme == 1))
        false_alarm = np.sum((forecasted_extreme == 1) & (observed_extreme == 0))
        miss = np.sum((forecasted_extreme == 0) & (observed_extreme == 1))
        correct_rejection = np.sum((forecasted_extreme == 0) & (observed_extreme == 0))
        
        total = hit + false_alarm + miss + correct_rejection
        hit_rate = hit / (hit + miss)
        false_alarm_rate = false_alarm / (false_alarm + correct_rejection)
        
        eds = (hit_rate - false_alarm_rate) / (hit_rate + false_alarm_rate)
        
        return eds






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

        

