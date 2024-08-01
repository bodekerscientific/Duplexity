import unittest
import numpy as np
import xarray as xr
import pandas as pd
import sys
# Add the path to the sys.path
path = '/home/lexi/WeatherValidation/WeatherValidation/'
if path not in sys.path:
    sys.path.append(path)
from duplexity.deterministic_score import DeterministicScore, CategoricalScore


class TestDeterministicScore(unittest.TestCase):

    def setUp(self):
        # Create synthetic observed and output data
        self.observed_data = np.random.rand(192, 144)  # Random data to simulate observations
        self.output_data = self.observed_data + np.random.normal(0, 0.1, self.observed_data.shape)  # Simulate model output with some noise

        self.observed_np = self.observed_data
        self.output_np = self.output_data

        self.observed_xr = xr.DataArray(self.observed_data)
        self.output_xr = xr.DataArray(self.output_data)

        self.observed_pd = pd.DataFrame(self.observed_data)
        self.output_pd = pd.DataFrame(self.output_data)

        # Set a threshold for binary classification
        self.threshold = 0.5

        # Initialize the DeterministicScore class for continuous and categorical data
        self.det_score_continuous_np = DeterministicScore(self.observed_np, self.output_np)
        self.det_score_categorical_np = CategoricalScore(self.observed_np, self.output_np, threshold=self.threshold)

        self.det_score_continuous_xr = DeterministicScore(self.observed_xr, self.output_xr)
        self.det_score_categorical_xr = CategoricalScore(self.observed_xr, self.output_xr, threshold=self.threshold)

        self.det_score_continuous_pd = DeterministicScore(self.observed_pd, self.output_pd)
        self.det_score_categorical_pd = CategoricalScore(self.observed_pd, self.output_pd, threshold=self.threshold)

    def test_mean_absolute_error(self):
        for score in [self.det_score_continuous_np, self.det_score_continuous_xr, self.det_score_continuous_pd]:
            mae = score.mean_absolute_error()
            self.assertIsInstance(mae, float)

    def test_mean_squared_error(self):
        for score in [self.det_score_continuous_np, self.det_score_continuous_xr, self.det_score_continuous_pd]:
            mse = score.mean_squared_error()
            self.assertIsInstance(mse, float)

    def test_root_mean_squared_error(self):
        for score in [self.det_score_continuous_np, self.det_score_continuous_xr, self.det_score_continuous_pd]:
            rmse = score.root_mean_squared_error()
            self.assertIsInstance(rmse, float)

    def test_bias(self):
        for score in [self.det_score_continuous_np, self.det_score_continuous_xr, self.det_score_continuous_pd]:
            bias = score.bias()
            self.assertIsInstance(bias, float)

    def test_pearson_correlation(self):
        for score in [self.det_score_continuous_np, self.det_score_continuous_xr, self.det_score_continuous_pd]:
            correlation = score.pearson_correlation()
            self.assertIsInstance(correlation, float)

    def test_accuracy(self):
        for score in [self.det_score_categorical_np, self.det_score_categorical_xr, self.det_score_categorical_pd]:
            accuracy = score.accuracy()
            self.assertIsInstance(accuracy, float)

    def test_critical_success_index(self):
        for score in [self.det_score_categorical_np, self.det_score_categorical_xr, self.det_score_categorical_pd]:
            csi = score.critical_success_index()
            self.assertIsInstance(csi, float)

    def test_equitable_threat_score(self):
        for score in [self.det_score_categorical_np, self.det_score_categorical_xr, self.det_score_categorical_pd]:
            ets = score.equitable_threat_score()
            self.assertIsInstance(ets, float)

    def test_false_alarm_ratio(self):
        for score in [self.det_score_categorical_np, self.det_score_categorical_xr, self.det_score_categorical_pd]:
            far = score.false_alarm_ratio()
            self.assertIsInstance(far, float)

    def test_probability_of_detection(self):
        for score in [self.det_score_categorical_np, self.det_score_categorical_xr, self.det_score_categorical_pd]:
            pod = score.probability_of_detection()
            self.assertIsInstance(pod, float)

    def test_gilbert_skill_score(self):
        for score in [self.det_score_categorical_np, self.det_score_categorical_xr, self.det_score_categorical_pd]:
            gss = score.gilbert_skill_score()
            self.assertIsInstance(gss, float)

    def test_heidke_skill_score(self):
        for score in [self.det_score_categorical_np, self.det_score_categorical_xr, self.det_score_categorical_pd]:
            hss = score.heidke_skill_score()
            self.assertIsInstance(hss, float)

    def test_peirce_skill_score(self):
        for score in [self.det_score_categorical_np, self.det_score_categorical_xr, self.det_score_categorical_pd]:
            pss = score.peirce_skill_score()
            self.assertIsInstance(pss, float)

    def test_sedi(self):
        for score in [self.det_score_categorical_np, self.det_score_categorical_xr, self.det_score_categorical_pd]:
            sedi = score.sedi()
            self.assertIsInstance(sedi, float)

if __name__ == '__main__':
    unittest.main()
