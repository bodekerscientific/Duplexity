import unittest
import numpy as np
import xarray as xr
import pandas as pd
import sys
# Add the path to the sys.path
path = '/home/lexi/WeatherValidation/WeatherValidation/'
if path not in sys.path:
    sys.path.append(path)
from duplexity.deterministic_score import *


class TestDeterministicScore(unittest.TestCase):

    def setUp(self):
        self.observed_data = np.random.rand(10, 192, 144)  # Random data to simulate observations
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
        corr = pearson_correlation(self.observed_xr, self.output_xr)
        self.assertIsInstance(corr, float)
        corr_pd = pearson_correlation(self.observed_pd, self.output_pd)
        self.assertIsInstance(corr_pd, float)

    def test_confusion_matrix(self):
        cm = confusion_matrix(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.shape, (2, 2))
        cm_pd = confusion_matrix(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(cm_pd, np.ndarray)
        self.assertEqual(cm_pd.shape, (2, 2))

    def test_precision(self):
        prec = precision(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(prec, float)
        prec_pd = precision(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(prec_pd, float)

    def test_recall(self):
        rec = recall(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(rec, float)
        rec_pd = recall(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(rec_pd, float)

    def test_f1_score(self):
        f1 = f1_score(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(f1, float)
        f1_pd = f1_score(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(f1_pd, float)

    def test_accuracy(self):
        acc = accuracy(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(acc, float)
        acc_pd = accuracy(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(acc_pd, float)

    def test_critical_success_index(self):
        csi = critical_success_index(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(csi, float)
        csi_pd = critical_success_index(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(csi_pd, float)

    def test_equitable_threat_score(self):
        ets = equitable_threat_score(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(ets, float)
        ets_pd = equitable_threat_score(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(ets_pd, float)

    def test_false_alarm_ratio(self):
        far = false_alarm_ratio(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(far, float)
        far_pd = false_alarm_ratio(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(far_pd, float)

    def test_probability_of_detection(self):
        pod = probability_of_detection(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(pod, float)
        pod_pd = probability_of_detection(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(pod_pd, float)

    def test_gilbert_skill_score(self):
        gss = gilbert_skill_score(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(gss, float)
        gss_pd = gilbert_skill_score(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(gss_pd, float)

    def test_heidke_skill_score(self):
        hss = heidke_skill_score(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(hss, float)
        hss_pd = heidke_skill_score(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(hss_pd, float)

    def test_peirce_skill_score(self):
        pss = peirce_skill_score(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(pss, float)
        pss_pd = peirce_skill_score(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(pss_pd, float)

    def test_sedi(self):
        sedi_value = sedi(self.observed_xr, self.output_xr, self.threshold)
        self.assertIsInstance(sedi_value, float)
        sedi_value_pd = sedi(self.observed_pd, self.output_pd, self.threshold)
        self.assertIsInstance(sedi_value_pd, float)

    def test_fss_score(self):
        fss_score = calculate_fss_score(self.output_xr, self.observed_xr, self.threshold, self.scale)
        self.assertIsInstance(fss_score, float)
        fss_score_pd = calculate_fss_score(self.output_pd, self.observed_pd, self.threshold, self.scale)
        self.assertIsInstance(fss_score_pd, float)

    #def test_psd(self):
    #    psd_values = validate_with_psd(self.observed_xr, self.output_xr)
    #    self.assertIsInstance(psd_values, np.ndarray)
    #    self.assertEqual(psd_values.shape[0], self.observed_xr.shape[0])
    #    psd_values_pd = validate_with_psd(self.observed_pd.values.reshape(10, 192, 144), self.output_pd.values.reshape(10, 192, 144))
    #    self.assertIsInstance(psd_values_pd, np.ndarray)
    #    self.assertEqual(psd_values_pd.shape[0], self.observed_pd.shape[0])

    def test_rapsd(self):
        rapsd_values = rapsd(self.observed_xr[0].values)
        self.assertIsInstance(rapsd_values, np.ndarray)
        rapsd_values_pd = rapsd(self.observed_pd.iloc[0].values.reshape(192, 144))
        self.assertIsInstance(rapsd_values_pd, np.ndarray)

if __name__ == '__main__':
    unittest.main()
