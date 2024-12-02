Metrics Overview
=====================

The **Duplexity** package provides a wide range of metrics to evaluate the performance of machine learning models, particularly in the context of weather forecasts and other spatial data.

These metrics are divided into different categories based on the type of data and the nature of the predictions being evaluated.


Metric Categories
~~~~~~~~~~~~~~~~~

The metrics in **Duplexity** are divided into the following categories:

- **Pixel-Wise Metrics**: Metrics calculated at individual grid points or pixels, allowing for detailed evaluation of model performance at each point.
  
- **Probabilistic Metrics**: Metrics designed to evaluate the uncertainty or probabilistic nature of forecasts.

- **Ensemble Metrics**: Metrics designed to evaluate the performance of ensemble forecasts, which consist of multiple simulations representing different possible outcomes.

- **Spatial Metrics**: Metrics that account for the spatial structure of the data, providing a higher-level evaluation of the model's ability to capture spatial patterns.

- **Image Metrics**: Metrics typically used for evaluating image-based data, comparing the visual similarity between observed and predicted images.




.. toctree::
    :maxdepth: 1
    
    PixelWise Metrics <pixelwise>
    Probabilistic Metrics <probabilistic>
    Ensemble Metrics <ensemble>
    Spatial Metrics <spatial>
    Image Metrics <image>
