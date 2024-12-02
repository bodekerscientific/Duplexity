import unittest
import numpy as np
import xarray as xr
import pandas as pd
import sys
# Add the path to the sys.path
#path = '/home/lexi/WeatherValidation/WeatherValidation/'
if path not in sys.path:
    sys.path.append(path)
from duplexity.pixelwise import *


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


    def test_continuous_metrics(self):
        # Test continuous metrics
        metrics = ['mae', 'rmse']
        result = calculate_pixelwise_metrics(self.observed, self.output, metrics=metrics, metric_type="continuous")
        expected_result = {'mae': 0.266, 'rmse': 0.275}  # Example values, you need to replace with actual expected results
        self.assertAlmostEqual(result['mae'], expected_result['mae'], places=3)
        self.assertAlmostEqual(result['rmse'], expected_result['rmse'], places=3)

    def test_default_metrics(self):
        # Test with default (all metrics)
        result = calculate_pixelwise_metrics(self.observed, self.output)
        # You need to replace this with actual expected results
        self.assertIn('precision', result)
        self.assertIn('mae', result)

    def test_invalid_metric(self):
        # Test with an invalid metric
        with self.assertRaises(ValueError):
            calculate_pixelwise_metrics(self.observed, self.output, metrics=['invalid_metric'])

    def test_invalid_metric_type(self):
        # Test with an invalid metric type
        with self.assertRaises(ValueError):
            calculate_pixelwise_metrics(self.observed, self.output, metric_type='invalid_type')
    
    def test_conflicting_metric_type(self):
        # Test with conflicting metric type and metrics
        with self.assertRaises(ValueError):
            calculate_pixelwise_metrics(self.observed, self.output, metrics=['precision'], metric_type='continuous')



if __name__ == '__main__':
    unittest.main()
