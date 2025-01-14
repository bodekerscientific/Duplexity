Plotting Utilities
=====================

Plotting functions for visualizing metric results, forecasts, and analyses.

This section covers the various plotting utilities available for visualizing the outputs of weather forecast validations and other analyses.

.. automodule:: duplexity.plot
    :members:
    :undoc-members:
    :show-inheritance:

Available Plots
----------------

The following plotting utilities are provided to help visualize different aspects of the weather forecast validation results:

- **plot_fss**: Plot Fraction Skill Score (FSS) results.
- **plot_rapsd**: Plot the Radially Averaged Power Spectral Density (RAPSD).
- **plot_metrics_map**: Visualize the spatial distribution of pixel-wise metrics.

.. toctree::
    :maxdepth: 2
    FSS Plot <fss_plot>
    RAPSD Plot <rapsd_plot>
    Metrics Map Plot <metrics_map_plot>

Examples
---------
Here are some examples of how to use the plotting utilities in conjunction with the validation metrics:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from duplexity.plot import plot_fss, plot_rapsd

    # Example of using plot_fss
    scales = [1, 2, 5, 10]
    fss_results = {0.5: [0.6, 0.7, 0.8, 0.85], 0.75: [0.5, 0.6, 0.7, 0.8]}
    plot_fss(fss_results, scales, title="FSS Plot", xlabel="Scale", ylabel="FSS")

    # Example of using plot_rapsd
    freqs = np.linspace(0.1, 100, 100)
    radial_profile = np.random.random(100)
    plot_rapsd(freqs, radial_profile, x_units="km", y_units="Power", title="RAPSD")

The examples illustrate how to generate Fraction Skill Score plots and RAPSD plots to visualize model performance.
