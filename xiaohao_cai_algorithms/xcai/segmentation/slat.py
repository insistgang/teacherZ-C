"""
SLaT (Slice-based Thresholding) Segmentation Implementation

Based on:
- Cai et al.: Slice-based approaches for image segmentation
"""

import numpy as np
from typing import Optional, List, Tuple
from scipy import ndimage


class SLatSegmenter:
    """
    SLaT-based image segmentation using slice-based thresholding.
    
    Parameters
    ----------
    n_classes : int
        Number of segmentation classes.
    method : str
        Thresholding method: 'otsu', 'adaptive', or 'kmeans'.
    connectivity : int
        Connectivity for connected components (4 or 8).
    min_region_size : int
        Minimum size of regions to keep.
    """
    
    def __init__(
        self,
        n_classes: int = 2,
        method: str = 'otsu',
        connectivity: int = 8,
        min_region_size: int = 100
    ):
        self.n_classes = n_classes
        self.method = method
        self.connectivity = connectivity
        self.min_region_size = min_region_size
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment an image using SLaT method.
        
        Parameters
        ----------
        image : np.ndarray
            Input image (2D grayscale or 3D color).
            
        Returns
        -------
        np.ndarray
            Segmentation labels (same shape as input, dtype int).
        """
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        if self.method == 'otsu':
            labels = self._otsu_segment(gray)
        elif self.method == 'adaptive':
            labels = self._adaptive_segment(gray)
        else:
            labels = self._kmeans_segment(gray)
        
        labels = self._post_process(labels)
        
        return labels
    
    def _otsu_segment(self, gray: np.ndarray) -> np.ndarray:
        """Multi-level Otsu thresholding."""
        if self.n_classes == 2:
            threshold = self._otsu_threshold(gray)
            labels = (gray > threshold).astype(int)
        else:
            thresholds = self._multi_otsu(gray, self.n_classes)
            labels = np.digitize(gray, thresholds)
        
        return labels
    
    def _otsu_threshold(self, gray: np.ndarray) -> float:
        """Compute Otsu threshold."""
        hist, bins = np.histogram(gray.flatten(), bins=256)
        hist = hist.astype(float)
        hist /= hist.sum()
        
        omega = np.cumsum(hist)
        mu = np.cumsum(hist * np.arange(len(hist)))
        mu_t = mu[-1]
        
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-10)
        sigma_b[omega == 0] = 0
        sigma_b[omega == 1] = 0
        
        return float(bins[np.argmax(sigma_b)])
    
    def _multi_otsu(self, gray: np.ndarray, n_classes: int) -> np.ndarray:
        """Compute multiple Otsu thresholds."""
        n_thresholds = n_classes - 1
        hist, bins = np.histogram(gray.flatten(), bins=256)
        hist = hist.astype(float)
        prob = hist / hist.sum()
        
        best_thresholds = np.zeros(n_thresholds)
        best_variance = 0
        
        from itertools import combinations
        for combo in combinations(range(1, 255), n_thresholds):
            thresholds = np.array(combo)
            variance = self._compute_inter_class_variance(prob, thresholds)
            if variance > best_variance:
                best_variance = variance
                best_thresholds = thresholds
        
        return bins[best_thresholds]
    
    def _compute_inter_class_variance(self, prob: np.ndarray, thresholds: np.ndarray) -> float:
        """Compute inter-class variance for given thresholds."""
        thresholds = np.concatenate([[0], thresholds, [len(prob)]])
        
        total_mean = np.sum(prob * np.arange(len(prob)))
        variance = 0
        
        for i in range(len(thresholds) - 1):
            start, end = thresholds[i], thresholds[i + 1]
            w = prob[start:end].sum()
            if w > 0:
                m = np.sum(prob[start:end] * np.arange(start, end)) / w
                variance += w * (m - total_mean) ** 2
        
        return variance
    
    def _adaptive_segment(self, gray: np.ndarray) -> np.ndarray:
        """Adaptive thresholding using local windows."""
        from scipy.ndimage import uniform_filter
        
        window_size = max(gray.shape) // 10
        if window_size % 2 == 0:
            window_size += 1
        
        local_mean = uniform_filter(gray, size=window_size)
        offset = 0.9
        binary = gray > (local_mean * offset)
        
        return binary.astype(int)
    
    def _kmeans_segment(self, gray: np.ndarray) -> np.ndarray:
        """Simple k-means based segmentation."""
        flat = gray.flatten().reshape(-1, 1)
        
        idx = np.random.choice(len(flat), self.n_classes, replace=False)
        centers = flat[idx].flatten()
        
        for _ in range(20):
            distances = np.abs(flat - centers.reshape(1, -1))
            labels = np.argmin(distances, axis=1)
            
            new_centers = np.array([
                flat[labels == k].mean() if np.sum(labels == k) > 0 else centers[k]
                for k in range(self.n_classes)
            ])
            
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        
        return labels.reshape(gray.shape)
    
    def _post_process(self, labels: np.ndarray) -> np.ndarray:
        """Post-process segmentation: remove small regions."""
        structure = np.ones((3, 3)) if self.connectivity == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        
        result = labels.copy()
        for label in range(labels.max() + 1):
            mask = labels == label
            labeled, num_features = ndimage.label(mask, structure=structure)
            
            for i in range(1, num_features + 1):
                region = labeled == i
                if np.sum(region) < self.min_region_size:
                    result[region] = 0
        
        return result
