Metrics Map Plot
================

Plotting utility for visualizing the spatial distribution of pixel-wise metrics across a grid.

.. automodule:: duplexity.plot
    :members:
    :undoc-members:
    :show-inheritance:

Example
-------

.. code-block:: python

    from duplexity.plot import plot_metrics_map

    # Example of plotting a spatial metric map
    metric_map = np.random.rand(100, 100)
    plot_metrics_map({"RMSE": metric_map}, metric_name="RMSE", title="RMSE Map", save_path="rmse_map.png")

