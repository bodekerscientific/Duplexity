import unittest
import numpy as np
import xarray as xr
import pandas as pd
import sys
# Add the path to the sys.path
path = '/home/lexi/WeatherValidation/WeatherValidation/'
if path not in sys.path:
    sys.path.append(path)
from duplexity.deterministic import *


class TestDeterministicScore(unittest.TestCase):

    def setUp(self):
        self.observed_data = np.random.rand(192, 144)  # Random data to simulate observations
        self.output_data = self.observed_data + np.random.normal(0, 0.1, self.observed_data.shape)  # Simulate model output with some noise

        self.observed_xr = xr.DataArray(self.observed_data)
        self.output_xr = xr.DataArray(self.output_data)

        self.observed_pd = pd.DataFrame(self.observed_data.reshape(10, -1))
        self.output_pd = pd.DataFrame(self.output_data.reshape(10, -1))

        # Set a threshold for binary classification
        self.threshold = 0.5
        self.scale = 10

    def test_mean_absolute_error(self):
        mae = mean_absolute_error(self.observed_xr, self.output_xr)
        self.assertIsInstance(mae, float)
        mae_pd = mean_absolute_error(self.observed_pd, self.output_pd)
        self.assertIsInstance(mae_pd, float)

    def test_mean_squared_error(self):
        mse = mean_squared_error(self.observed_xr, self.output_xr)
        self.assertIsInstance(mse, float)
        mse_pd = mean_squared_error(self.observed_pd, self.output_pd)
        self.assertIsInstance(mse_pd, float)

    def test_root_mean_squared_error(self):
        rmse = root_mean_squared_error(self.observed_xr, self.output_xr)
        self.assertIsInstance(rmse, float)
        rmse_pd = root_mean_squared_error(self.observed_pd, self.output_pd)
        self.assertIsInstance(rmse_pd, float)

    def test_bias(self):
        bias_value = bias(self.observed_xr, self.output_xr)
        self.assertIsInstance(bias_value, float)
        bias_value_pd = bias(self.observed_pd, self.output_pd)
        self.assertIsInstance(bias_value_pd, float)

    def test_debiased_root_mean_squared_error(self):
        drmse = debiased_root_mean_squared_error(self.observed_xr, self.output_xr)
        self.assertIsInstance(drmse, float)
        drmse_pd = debiased_root_mean_squared_error(self.observed_pd, self.output_pd)
        self.assertIsInstance(drmse_pd, float)

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
