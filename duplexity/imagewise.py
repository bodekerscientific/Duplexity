# Description: This file contains functions for image quality metrics and image processing tasks.

"""
Image Quality Metrics
=====================
This module contains functions for computing image quality metrics, such as Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM).
These metrics are commonly used in image processing to evaluate the quality of reconstructed or processed images.

Functions:
----------
.. autosummary::
    :toctree: ../generated/

    psnr
    ssim
    image_warp
    rapsd
    gmsd

"""


import numpy as np
import xarray as xr
import pandas as pd
from skimage.metrics import structural_similarity
from scipy.ndimage import affine_transform
from scipy.fftpack import fftshift, fft2
from typing import Union, List
from utils import _to_numpy, _check_2d_data, _check_shapes
from duplexity.pixelwise import *
from typing import Tuple, Union, List, Dict, Any, Optional



###################################################################
#########        Peak Signal-to-Noise Ratio (PSNR)       ##########
###################################################################

def psnr(observed: Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         output: Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         max_pixel_value: int = 255 ) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between observed and output images.

    The PSNR is a widely used metric in image processing that quantifies the quality of a reconstructed image 
    compared to the original, in terms of the mean squared error (MSE). A higher PSNR indicates a higher quality image.

    Parameters
    ----------
    observed : Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        The original or reference image. This can be a 2D NumPy array, xarray DataArray, xarray Dataset, 
        Pandas DataFrame, or a list of any of these types.
    output : Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        The reconstructed or output image to be compared. This can be a 2D NumPy array, xarray DataArray, xarray Dataset, 
        Pandas DataFrame, or a list of any of these types.
    max_pixel_value : int, optional
        The maximum possible pixel value in the images. Default is 255, which is common for 8-bit images.

    Returns
    -------
    float
        The PSNR value in decibels (dB). Higher values indicate better image quality.

    Notes
    -----
    - PSNR is computed as: 20 * log10(max_pixel_value / sqrt(MSE)).
    - Infinite PSNR indicates that the observed and output images are identical.

    Examples
    --------
    >>> observed_image = np.array([[100, 100], [100, 100]])
    >>> output_image = np.array([[90, 110], [100, 100]])
    >>> psnr(observed_image, output_image)
    40.0  # Example value, depends on input images

    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)
    _check_shapes(observed, output)
    
    rmse = root_mean_squared_error(observed, output)
    if rmse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel_value / rmse)


###################################################################
#########   Structural Similarity Index (SSIM)         ############
###################################################################

def ssim(observed: Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
                          output: Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]) -> float:
    """
    Calculate the Structural Similarity Index (SSIM) between observed and output images.

    SSIM is a perceptual metric that measures image quality degradation based on changes in structural information, 
    luminance, and contrast. It ranges from -1 to 1, where 1 indicates perfect similarity.

    Parameters
    ----------
    observed : Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        The original or reference image. This can be a 2D NumPy array, xarray DataArray, xarray Dataset, 
        Pandas DataFrame, or a list of any of these types.
    output : Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        The reconstructed or output image to be compared. This can be a 2D NumPy array, xarray DataArray, xarray Dataset, 
        Pandas DataFrame, or a list of any of these types.

    Returns
    -------
    float
        The SSIM value between the observed and output images. Values close to 1 indicate high structural similarity.

    Notes
    -----
    - SSIM is calculated using the luminance, contrast, and structural components of the images.
    - It is particularly effective for comparing images in scenarios where human visual perception is important.

    Examples
    --------
    >>> observed_image = np.array([[100, 100], [100, 100]])
    >>> output_image = np.array([[90, 110], [100, 100]])
    >>> structural_similarity(observed_image, output_image)
    0.99  # Example value, depends on input images
    """
    observed = _to_numpy(observed)
    output = _to_numpy(output)

    _check_shapes(observed, output)

    # Ensure that the data is normalized before calculating SSIM
    def normalize(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    observed_n = normalize(observed)
    output_n = normalize(output)

    ssim_value, ssim_map = structural_similarity(observed_n, output_n, data_range=output_n.max() - output_n.min(), full=True)


    return ssim_value





def image_warp(observed: Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
               matrix: np.ndarray) -> np.ndarray:
    """
    Apply an affine transformation to the observed image using the provided transformation matrix.

    Affine transformations include operations such as rotation, translation, scaling, and shearing.

    Parameters
    ----------
    observed : Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        The image to be transformed. This can be a 2D NumPy array, xarray DataArray, xarray Dataset, 
        Pandas DataFrame, or a list of any of these types.
    matrix : np.ndarray
        A 2x3 or 3x3 affine transformation matrix.

    Returns
    -------
    np.ndarray
        The transformed image as a NumPy array.

    Notes
    -----
    - The affine transformation is applied using the `scipy.ndimage.affine_transform` function.
    - The transformation matrix should be carefully constructed to achieve the desired transformation.

    Examples
    --------
    >>> observed_image = np.array([[100, 100], [100, 100]])
    >>> transformation_matrix = np.array([[1, 0, 0], [0, 1, 0]])  # Identity matrix (no transformation)
    >>> image_warp(observed_image, transformation_matrix)
    array([[100, 100],
           [100, 100]])  # Example output, same as input for identity matrix
    """
    observed = _to_numpy(observed)
    
    return affine_transform(observed, matrix)




###################################################################
#########   Radially Averaged Power Spectral Density   ############
###################################################################


def compute_centred_coord_array(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two centered coordinate arrays for an image or grid of size HxW.

    Parameters
    ----------
    H : int
        The height of the array.
    W : int
        The width of the array.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing two 2D numpy arrays: the x-coordinates and y-coordinates, 
        centered around the middle of the grid.
        - y_coor: The y-coordinates centered around the middle of the grid.
        - x_coor: The x-coordinates centered around the middle of the grid.

    Notes
    -----
    - The function handles both even and odd dimensions for height and width.
    - The coordinate arrays are centered around (0,0). For example, in a 3x3 grid, 
      the coordinates will range from -1 to 1.
    - The resulting coordinate arrays are useful for tasks that involve distance calculations from the center,
      such as in Fourier transforms, filtering, or computing Radially Averaged Power Spectral Density (RAPSD).
    - This function is particularly useful in image processing tasks where the 
      center of the grid needs to be treated as the origin.


    Examples
    --------
    >>> x_coords, y_coords = compute_centred_coord_array(3, 3)
    >>> x_coords
    array([[-1,  0,  1]])
    >>> y_coords
    array([[-1],
           [ 0],
           [ 1]])
    """

    # Determine the slice range for the y-coordinates based on whether height is odd or even

    if H % 2 == 1:
        y_slice = np.s_[-int(H / 2): int(H / 2) + 1]
    else:
        y_slice = np.s_[-int(H / 2): int(H / 2)]

    if W % 2 == 1:
        x_slice = np.s_[-int(W / 2): int(W / 2) + 1]
    else:
        x_slice = np.s_[-int(W / 2): int(W / 2)]

    y_coor, x_coor = np.ogrid[y_slice, x_slice]

    return y_coor, x_coor



def rapsd(data: Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame], 
            fft_method: Union[str, None] = None,
            return_freq: bool = False,
            pixel_spacing: float = 1.0,
            normalize: bool = False,
            epsilon: float = 1e-10,
            ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    
    """
    Compute the Radially Averaged Power Spectral Density (RAPSD) of a 2D field.

    This function calculates the RAPSD of a 2D data field, which is a measure of the 
    distribution of power into frequency components composing that field. The method 
    uses the 2D Fourier Transform to compute the power spectral density and then 
    averages this radially from the center of the frequency domain.

    Parameters
    ----------
    data : Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]
        The 2D data field for which to compute the RAPSD. This can be a 2D NumPy array, xarray DataArray, xarray Dataset,
        or Pandas DataFrame.
    fft_method : Union[str, None], optional
        The method to use for computing the 2D Fourier Transform. If None, the input data is assumed to be the
        shifted discrete Fourier transform of the input field, where the origin is at the center of the array.
        Default is None. 
    return_freq : bool, optional
        Whether to return the frequency values along with the RAPSD. If True, the function will return a tuple
        containing the RAPSD and the corresponding frequency values. Default is False.
    pixel_spacing : float, optional
        The pixel spacing of the data field. This parameter is used to convert the frequency values to physical units.
        Default is 1.0.
    normalize : bool, optional
        Whether to normalize the RAPSD values by the total power. If True, the RAPSD values will be divided by the total
        power in the field. Default is False.
    epsilon : float, optional
        A small value to add to the denominator to avoid division by zero. Default
        is 1e-10.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        The Radially Averaged Power Spectral Density (RAPSD) of the input data field. If return_freq is True, the function
        will return a tuple containing the RAPSD and the corresponding frequency values.
        - freq: The radial frequency bins.
        - rapsd: The radially averaged power spectral density corresponding to each bin.


    Notes
    -----
    - The RAPSD is computed by first computing the 2D Fourier Transform of the input data field. If the input data is not
        already the shifted discrete Fourier transform, the function will compute the FFT using the specified method.
    - The function then computes the power spectral density (PSD) of the Fourier Transform and averages it radially.
    - The frequency values are computed based on the pixel spacing d and the size of the input data field.
    - The RAPSD values are normalized by the total power if the normalize parameter is set to True.
    - The function is useful for analyzing the frequency content of 2D data fields, such as images or spatial maps.

    Examples
    --------
    >>> data = np.random.rand(256, 256)
    >>> rapsd_values = rapsd(data)
    >>> freqs, rapsd_values = rapsd(data, return_freq=True)

    """

    data = _to_numpy(data)

    # Validate the input data dimensions
    _check_2d_data(data)

    # Check if the input data contains nan values
    if np.isnan(data).any():
        raise ValueError("Input data contains NaN values. Please remove or interpolate these values before computing the RAPSD.")

    H, W = data.shape

    # Compute the centered coordinate arrays
    y_coor, x_coor = compute_centred_coord_array(H, W)

    # compute the radial distance from the center
    radial_dist = np.sqrt(x_coor ** 2 + y_coor ** 2).round()

    # Compute the maximum radial distance
    max_dim = max(H, W)

    if max_dim % 2 == 1:
        radius_range = np.arange(0, int(max_dim / 2) + 1)
    else:
        radius_range = np.arange(0, int(max_dim / 2))
    
    # Determine the fft method to use
    if fft_method is not None:
        if not (hasattr(fft_method, 'fft2') and hasattr(fft_method, 'fftshift')):
            raise ValueError("The provided fft_method does not have the required fft2 and fftshift functions.")

        # Perform 2D FFT on the data and shift the zero frequency component to the center
        psd = fft_method.fftshift(fft_method.fft2(data))
        # Compute the Power Spectral Density (PSD)
        psd = np.abs(psd) ** 2 / psd.size
    else:
        # If no FFT method is provided, just use the original data
        psd = data
    

    # Initialize the RAPSD array
    results = []

    # Compute the RAPSD
    for i, r in enumerate(radius_range):
        mask = radial_dist == r
        results.append(np.mean(psd[mask]))

    results = np.array(results)

    # Normalize the RAPSD values if required
    if normalize:
        total_power = np.sum(results)
        if total_power > epsilon: # Check if total power is greater than epsilon
            results /= total_power
        else:
            results /= (total_power + epsilon)

    # Compute the frequency values
    freq = np.fft.fftfreq(max_dim, d = pixel_spacing)

    if return_freq:
        freq = freq[:len(results)]
        return results, freq
    
    return results






###################################################################
#########        Gradient Magnitude Similarity            #########
###################################################################

def gradient_magnitude(data: np.ndarray) -> np.ndarray:
    """
    Compute the gradient magnitude of an image.

    Parameters
    ----------
    data : np.ndarray
        The input image as a 2D NumPy array.

    Returns
    -------
    np.ndarray
        The gradient magnitude of the image.
    """

    dx = np.gradient(data, axis=0)
    dy = np.gradient(data, axis=1)
    return np.sqrt(dx**2 + dy**2)


def gmsd(observed: Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         output: Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]],
         sigma: float = None) -> float:
    """
    Compute the Gradient Magnitude Similarity Deviation (GMSD) between two images.

    The GMSD is a metric that quantifies the similarity between two images based on the deviation of their gradient magnitudes.
    It is particularly useful for evaluating the quality of images that have undergone compression or distortion.

    Parameters
    ----------
    observed : Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        The original or reference image. This can be a 2D NumPy array, xarray DataArray, xarray Dataset, 
        Pandas DataFrame, or a list of any of these types.
    output : Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame, List[Union[np.ndarray, xr.DataArray, xr.Dataset, pd.DataFrame]]]
        The reconstructed or output image to be compared. This can be a 2D NumPy array, xarray DataArray, xarray Dataset, 
        Pandas DataFrame, or a list of any of these types.
    sigma : float, optional
        The standard deviation of the Gaussian filter used to compute the gradient magnitudes. Default is None.

    Returns
    -------
    float
        The GMSD value between the observed and output images. Lower values indicate higher similarity.

    Notes
    -----
    - The GMSD is computed based on the deviation of the gradient magnitudes of the observed and output images.
    - A lower GMSD value indicates higher similarity between the images.
    - The GMSD is particularly useful for evaluating the quality of images that have undergone compression or distortion.

    Examples
    --------
    >>> observed_image = np.array([[100, 100], [100, 100]])
    >>> output_image = np.array([[90, 110], [100, 100]])
    >>> gmsd(observed_image, output_image)
    0.01  # Example value, depends on input images

    """

    observed = _to_numpy(observed)
    output = _to_numpy(output)

    _check_shapes(observed, output)

    # Apply Gaussian filter to smooth the images
    observed_smooth = gaussian_filter(observed, sigma=sigma)
    output_smooth = gaussian_filter(output, sigma=sigma)


    # Compute the gradient magnitudes of the observed and output images
    observed_gm = gradient_magnitude(observed_smooth)
    output_gm = gradient_magnitude(output_smooth)

    # Compute the mean squared error between the gradient magnitudes
    mse = mean_squared_error(observed_gm, output_gm)

    return np.sqrt(mse)






#############################################################
##          Precipitation Smoothing Distance               ##
#############################################################

def circular_kernel(radius: int) -> np.array:
    """
    Creates a circular kernel with the given radius.
    
    Parameters:
    radius (int): The radius of the circular kernel.
    
    Returns:
    np.array: A 2D array representing the circular kernel.
    """
    # Calculate the size of the kernel
    r_max = int(np.floor(radius))
    size = 2 * r_max + 1
    shape = (size, size)
    kernel = np.zeros((size, size), dtype =np.float32)
    
    # Calculate the center of the kernel
    center = r_max
    
    # Get the row and column coordinates for the disk
    row_coords, col_coords = disk((center, center), radius + 0.0001, shape=shape)

    # Set the pixels inside the disk to 1
    kernel[row_coords, col_coords] = 1

    # Normalize the kernel so that the sum of its elements equals 1
    kernel /= np.sum(kernel)

    return kernel



def calculate_pss(output_smooth: np.array, observed_smooth: np.array, radius: int, Q: float) -> float:
    """
    Calculates the Precipitation Smooth Score (PSS).
    
    Parameters:
    output_smooth (np.array): Smoothed forecast data.
    observed_smooth (np.array): Smoothed observed data.
    radius (float): Radius for smoothing.
    Q (float): Quality factor.
    
    Returns:
    float: Precipitation Symmetry Score (PSS).
    """
    kernel = circular_kernel(radius)
    output_smooth = scipy.signal.fftconvolve(output_smooth, kernel, mode='full')
    observed_smooth = scipy.signal.fftconvolve(observed_smooth, kernel, mode='full')
    PSS = 1.0 - 1.0 / (2.0 * float(output_smooth.size) * Q) * np.abs(output_smooth - observed_smooth).sum()
    return PSS

def calculate_psd(output: np.ndarray, observed: np.ndarray) -> float:
    """
    """
    output = output.copy()
    observed = observed.copy()

    _check_shapes(output, observed)
    
    if output.ndim != 2 or observed.ndim != 2:
        raise ValueError("output and observed data are not two-dimensional.")
    
    if output.size == 0 or observed.size == 0:
        raise ValueError("output and observed data are empty.")
    
    if not np.all(np.isfinite(output)) or not np.all(np.isfinite(observed)):
        raise ValueError("output and observed data contain non-numeric values.")
    
    if isinstance(output, np.ma.MaskedArray) or isinstance(observed, np.ma.MaskedArray):
        raise ValueError("output and observed data are masked arrays which is not allowed.")
    
    if np.any(output < 0) or np.any(observed < 0):
        raise ValueError("output and observed data contain negative values which is not allowed.")
    
    output = output.astype(float)
    observed = observed.astype(float)
    
    output_avg = np.average(output)
    observed_avg = np.average(observed)
    
    if output_avg == 0 or observed_avg == 0:
        return np.nan  # Return NaN for empty fields    
    
    output_norm = output / output_avg
    observed_norm = observed / observed_avg
    
    if np.array_equal(output_norm, observed_norm):
        return 0
    
    output_diff = output_norm - np.minimum(output_norm, observed_norm)
    observed_diff = observed_norm - np.minimum(output_norm, observed_norm)
    
    Q = output_diff.sum() / output_norm.sum()
    
    initial_radius = 1
    PSS_initial = calculate_pss(output_diff, observed_diff, initial_radius, Q)
    
    if PSS_initial > 0.5:
        return 1
    
    diagonal = np.sqrt(output.shape[0]**2 + output.shape[1]**2)
    dr = np.ceil(diagonal * 0.05)
    
    radius2 = initial_radius + dr
    PSS2 = calculate_pss(output_diff, observed_diff, radius2, Q)
    
    while PSS2 < 0.5:
        initial_radius = radius2
        PSS_initial = PSS2
        radius2 = initial_radius + dr
        PSS2 = calculate_pss(output_diff, observed_diff, radius2, Q)
    
    while radius2 - initial_radius > 1:
        new_radius = int((initial_radius + radius2) / 2)
        PSS_new = calculate_pss(output_diff, observed_diff, new_radius, Q)
        
        if PSS_new > 0.5:
            radius2 = new_radius
            PSS2 = PSS_new
        else:
            initial_radius = new_radius
            PSS_initial = PSS_new
    
    PSD = 0.808 * Q * float(radius2)
    
    return PSD


def validate_with_psd(observed: np.ndarray, output: np.ndarray) -> np.ndarray:
    """
    Validate the forecasted data with the Precipitation Symmetry Distance (PSD).
    
    Parameters:
    observed (np.ndarray): Observed data array.
    forecasted (np.ndarray): Forecasted data array.
    
    Returns:
    np.ndarray: Array of PSD values.
    """
    psd_results = []
    for i in tqdm(range(output.shape[0]), desc="Calculating PSD"):
        observed_slice = observed[i]
        output_slice = output[i]
        psd = calculate_psd(observed_slice, output_slice)
        psd_results.append(psd)
    return np.array(psd_results)





