FSS Plot
================

Fraction Skill Score (FSS) plotting utility for visualizing the performance of forecast models across different spatial scales.

.. automodule:: duplexity.plot
    :members:
    :undoc-members:
    :show-inheritance:

Example
-------

.. code-block:: python

    from duplexity.plot import plot_fss

    scales = [1, 2, 5, 10]
    fss_results = {0.5: [0.6, 0.7, 0.8, 0.85], 0.75: [0.5, 0.6, 0.7, 0.8]}
    plot_fss(fss_results, scales, title="FSS Plot", xlabel="Scale", ylabel="FSS")

