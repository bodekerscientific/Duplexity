.. Duplexity documentation master file, created by
   sphinx-quickstart on Fri Aug  9 11:03:23 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Duplexity's documentation!
=====================================

**Duplexity** is a Python package that provides a set of deterministic and probabilistic metrics for 
evaluating the performance of machine learning models. It is designed to be easy to use and to provide a wide 
range of metrics for comparing data sets

.. note::
   This project is still in the early development stages. This means the code is 
   constantly changing, and we currently don't guarantee backwards compatibility 
   when changes are made. Once a stable release of Duplexity is available, we will
   upload Duplexity to PyPI to make it pip-installable.



Table of Contents
=================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Installation <installation>
   Validation Metrics <metrics/metrics>
   Visualization <plot/plot>
   
Introduction
============

The **Duplexity** package provides a range of metrics for validating the performance of weather forecasting models, machine learning models, and other spatial datasets. It includes:

- **Pixel-Wise Metrics**: Evaluate individual grid points or pixels.
- **Probabilistic Metrics**: Designed to assess probabilistic forecasts.
- **Ensemble Metrics**: Assess the performance of ensemble forecasts.
- **Spatial Metrics**: Focus on capturing the spatial structure of forecasts.
- **Image Metrics**: Used to compare images or spatial datasets.

Each of these metrics can be calculated using a wide variety of input formats such as NumPy arrays, xarray datasets, and pandas dataframes.




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


