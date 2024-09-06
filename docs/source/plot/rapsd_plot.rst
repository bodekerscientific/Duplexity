RAPSD Plot
================

Radially Averaged Power Spectral Density (RAPSD) plotting utility for visualizing the power spectral density of forecasts.

.. automodule:: duplexity.plot
    :members:
    :undoc-members:
    :show-inheritance:

Example
-------

.. code-block:: python

    from duplexity.plot import plot_rapsd

    freqs = np.linspace(0.1, 100, 100)
    radial_profile = np.random.random(100)
    plot_rapsd(freqs, radial_profile, x_units="km", y_units="Power", title="RAPSD")

