
.. automodule:: duplexity.pixelwise
    :members:
    :undoc-members:
    :show-inheritance:


Available Metrics
~~~~~~~~~~~~~~~~~

The following pixel-wise metrics are available:

**Categorical Metrics:**

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Metric
     - Description
   * - ``cm``
     - Confusion Matrix
   * - ``precision``
     - Positive Predictive Value (Precision)
   * - ``recall``
     - True Positive Rate (Recall)
   * - ``F1``
     - Harmonic Mean of Precision and Recall (F1 Score)
   * - ``accuracy``
     - Overall Accuracy
   * - ``csi``
     - Critical Success Index (CSI)
   * - ``far``
     - False Alarm Ratio (FAR)
   * - ``pod``
     - Probability of Detection (POD)
   * - ``gss``
     - Gilbert Skill Score (GSS), also known as Equitable Threat Score (ETS)
   * - ``hss``
     - Heidke Skill Score (HSS)
   * - ``pss``
     - Peirce Skill Score (PSS)
   * - ``sedi``
     - Symmetric Extremal Dependence Index (SEDI)

**Continuous Metrics:**

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Metric
     - Description
   * - ``mae``
     - Mean Absolute Error (MAE)
   * - ``mse``
     - Mean Squared Error (MSE)
   * - ``rmse``
     - Root Mean Squared Error (RMSE)
   * - ``bias``
     - Frequency Bias
   * - ``drmse``
     - Debiased Root Mean Squared Error (DRMSE)
   * - ``corr``
     - Pearson Correlation Coefficient (Correlation)



Example Usage
~~~~~~~~~~~~~
Below are examples of how to calculate pixel-wise metrics using the provided functions:

.. code-block:: python

    from duplexity.pixelwise import calculate_pixelwise_metrics
    import numpy as np

    # Generate some random data to simulate observed and model output
    observed_data = np.random.rand(100, 100)
    model_output = np.random.rand(100, 100)

    # Calculate pixel-wise metrics
    results = calculate_pixelwise_metrics(observed_data, model_output, metrics=["mae", "rmse"])

    print(results)

    # Output:
    # {'mae': 0.123, 'rmse': 0.345}

The `calculate_pixelwise_metrics` function allows you to compute multiple metrics at once by specifying them in the `metrics` argument.
