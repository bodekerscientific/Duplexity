import unittest
import numpy as np
import xarray as xr
import pandas as pd

import sys
# Add the path to the sys.path
path = '/home/lexi/WeatherValidation/WeatherValidation/'
if path not in sys.path:
    sys.path.append(path)

from duplexity.deterministic_score import DeterministicScore

# Import functions from your script
from duplexity.metric_map import (
    initialize_metrics,
    update_metrics,
    _to_numpy,
    grid_point_calculate,
    plot_metrics_map
    )


class TestWeatherValidationFunctions(unittest.TestCase):
    def setUp(self):
        # Create synthetic observed and output data
        self.observed_np = np.random.rand(192, 144)  # Random data to simulate observations
        self.output_np = self.observed_np + np.random.normal(0, 0.1, self.observed_np.shape)  # Simulate model output with some noise

        self.observed_xr = xr.DataArray(self.observed_np)
        self.output_xr = xr.DataArray(self.output_np)

        self.observed_pd = pd.DataFrame(self.observed_np)
        self.output_pd = pd.DataFrame(self.output_np)
        self.det_score= DeterministicScore(self.observed_np, self.output_np)


    def test_initialize_metrics(self):
        shape = (192, 144)
        metrics = initialize_metrics(shape)
        self.assertEqual(metrics.shape, shape)
        self.assertTrue(np.all(metrics == 0))

    def test_update_metrics(self):
        metric_array = np.zeros((192, 144))
        updated_metrics = update_metrics(metric_array, self.observed_np, self.output_np, self.det_score.mae())
        self.assertEqual(updated_metrics.shape, metric_array.shape)
        self.assertTrue(np.any(updated_metrics != 0))

    def test_to_numpy(self):
        np_array = _to_numpy(self.observed_np)
        self.assertTrue(isinstance(np_array, np.ndarray))
        xr_array = _to_numpy(self.observed_xr)
        self.assertTrue(isinstance(xr_array, np.ndarray))
        pd_array = _to_numpy(self.observed_pd)
        self.assertTrue(isinstance(pd_array, np.ndarray))

    def test_grid_point_calculate(self):
        metrics_result = grid_point_calculate(self.observed_np, self.output_np)
        self.assertTrue("MAE" in metrics_result)
        self.assertTrue("MSE" in metrics_result)
        self.assertTrue("RMSE" in metrics_result)
        self.assertTrue("Bias" in metrics_result)

    def test_plot_metrics_map(self):
        metrics_result = grid_point_calculate(self.observed_np, self.output_np)
        plot_metrics_map(metrics_result, "MAE", "Test MAE Map")


    def test_plot_metrics_map_seaborn(self):
        metrics_result = grid_point_calculate(self.observed_np, self.output_np)
        plot_metrics_map_seaborn(metrics_result, "MAE", "Test MAE Map Seaborn")

if __name__ == '__main__':
    unittest.main()
