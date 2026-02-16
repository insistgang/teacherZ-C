"""
Utility functions for denoising module.
"""

import numpy as np
from typing import Optional


def compute_psnr(original: np.ndarray, denoised: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Parameters
    ----------
    original : np.ndarray
        Original (clean) image.
    denoised : np.ndarray
        Denoised image to evaluate.
        
    Returns
    -------
    float
        PSNR value in dB.
    """
    mse = np.mean((original.astype(float) - denoised.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = original.max() - original.min()
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return float(psnr)


def compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 11,
    C1: Optional[float] = None,
    C2: Optional[float] = None
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Parameters
    ----------
    img1 : np.ndarray
        First image.
    img2 : np.ndarray
        Second image.
    window_size : int
        Size of the Gaussian window.
    C1, C2 : float, optional
        Stability constants.
        
    Returns
    -------
    float
        SSIM value in range [0, 1].
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    if C1 is None:
        C1 = (0.01 * 255) ** 2
    if C2 is None:
        C2 = (0.03 * 255) ** 2
    
    mu1 = _uniform_filter(img1, window_size)
    mu2 = _uniform_filter(img2, window_size)
    
    sigma1_sq = _uniform_filter(img1 ** 2, window_size) - mu1 ** 2
    sigma2_sq = _uniform_filter(img2 ** 2, window_size) - mu2 ** 2
    sigma12 = _uniform_filter(img1 * img2, window_size) - mu1 * mu2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))


def _uniform_filter(img: np.ndarray, size: int) -> np.ndarray:
    """Apply uniform filter to image."""
    kernel = np.ones((size, size)) / (size * size)
    from scipy.ndimage import convolve
    return convolve(img, kernel, mode='reflect')
