
import sys
import os
import unittest
import numpy as np
# Add the path to the sys.path
path = '/home/lexi/deep_weather/deep_weather/validation/'
if path not in sys.path:
    sys.path.append(path)
from nn_validation.deterministic import ContinuousScore, CategoricalScore  



class TestContinuousScore(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.observed_data = np.random.rand(100, 100)
        self.forecasted_data = np.random.rand(100, 100)
        self.cs = ContinuousScore(self.observed_data, self.forecasted_data)

    def test_mean_absolute_error(self):
        result = self.cs.mean_absolute_error()
        self.assertTrue(np.isclose(result, np.mean(np.abs(self.observed_data - self.forecasted_data)), atol=1e-5))

    def test_mean_squared_error(self):
        result = self.cs.mean_squared_error()
        self.assertTrue(np.isclose(result, np.mean((self.observed_data - self.forecasted_data) ** 2), atol=1e-5))

    def test_root_mean_squared_error(self):
        result = self.cs.root_mean_squared_error()
        self.assertTrue(np.isclose(result, np.sqrt(np.mean((self.observed_data - self.forecasted_data) ** 2)), atol=1e-5))

    def test_bias(self):
        result = self.cs.bias()
        self.assertTrue(np.isclose(result, np.mean(self.forecasted_data - self.observed_data), atol=1e-5))

    def test_debiased_root_mean_squared_error(self):
        result = self.cs.debiased_root_mean_squared_error()
        bias = np.mean(self.forecasted_data - self.observed_data)
        debiased_predictions = self.forecasted_data - bias
        expected_result = np.sqrt(np.mean((self.observed_data - debiased_predictions) ** 2))
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

    def test_pearson_correlation(self):
        result = self.cs.pearson_correlation()
        expected_result = np.corrcoef(self.observed_data.flatten(), self.forecasted_data.flatten())[0, 1]
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))


class TestCategoricalScore(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.observed_data = np.random.randint(0, 2, (100, 100))
        self.forecasted_data = np.random.rand(100, 100)
        self.threshold = 0.5
        self.cs = CategoricalScore(self.observed_data, self.forecasted_data, self.threshold)

    def test_accuracy(self):
        result = self.cs.accuracy()
        expected_result = ((self.forecasted_data >= self.threshold) == self.observed_data).mean()
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

    def test_critical_success_index(self):
        result = self.cs.critical_success_index()
        TP, FN, FP, _ = self.cs.calculate()
        expected_result = TP.sum() / (TP.sum() + FN.sum() + FP.sum())
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

    def test_equitable_threat_score(self):
        result = self.cs.equitable_threat_score()
        TP, FN, FP, TN = self.cs.calculate()
        expected_result = (TP.sum() - (TP.sum() + FN.sum()) * (TP.sum() + FP.sum()) / (TP.sum() + FN.sum() + FP.sum() + TN.sum())) / (TP.sum() + FN.sum() + FP.sum() - (TP.sum() + FN.sum()) * (TP.sum() + FP.sum()) / (TP.sum() + FN.sum() + FP.sum() + TN.sum()))
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

    def test_false_alarm_ratio(self):
        result = self.cs.false_alarm_ratio()
        TP, _, FP, _ = self.cs.calculate()
        expected_result = FP.sum() / (TP.sum() + FP.sum())
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

    def test_probability_of_detection(self):
        result = self.cs.probability_of_detection()
        TP, FN, _, _ = self.cs.calculate()
        expected_result = TP.sum() / (TP.sum() + FN.sum())
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

    def test_gilbert_skill_score(self):
        result = self.cs.gilbert_skill_score()
        TP, FN, FP, TN = self.cs.calculate()
        expected_result = (TP.sum() - (TP.sum() + FP.sum()) * (TP.sum() + FN.sum()) / (TP.sum() + FN.sum() + FP.sum() + TN.sum())) / (TP.sum() + FN.sum() + FP.sum())
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

    def test_heidke_skill_score(self):
        result = self.cs.heidke_skill_score()
        TP, FN, FP, TN = self.cs.calculate()
        expected_result = (2 * (TP.sum() * TN.sum() - FN.sum() * FP.sum())) / ((TP.sum() + FN.sum()) * (FN.sum() + TN.sum()) + (TP.sum() + FP.sum()) * (FP.sum() + TN.sum()))
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

    def test_peirce_skill_score(self):
        result = self.cs.peirce_skill_score()
        TP, FN, FP, TN = self.cs.calculate()
        expected_result = (TP.sum() / (TP.sum() + FN.sum())) - (FP.sum() / (FP.sum() + TN.sum()))
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

    def test_sedi(self):
        result = self.cs.sedi()
        TP, FN, FP, TN = self.cs.calculate()
        H = TP.sum() / (TP.sum() + FN.sum())
        F = FP.sum() / (FP.sum() + TN.sum())
        expected_result = (np.log(F) - np.log(H) - np.log(1 - F) + np.log(1 - H)) / (np.log(F) + np.log(H) + np.log(1 - F) + np.log(1 - H))
        self.assertTrue(np.isclose(result, expected_result, atol=1e-5))

if __name__ == '__main__':
    unittest.main()
