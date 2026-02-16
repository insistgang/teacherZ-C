"""
Varifold Kernel Implementation

Based on:
- Charon & Trouvé (2013): The varifold representation of nonoriented shapes
- Kaltenmark et al. (2017): Currents and varifolds for shape registration
"""

import numpy as np
from typing import Optional, Literal, Tuple
from scipy.spatial.distance import cdist


class VarifoldKernel:
    """
    Varifold kernel for shape comparison and registration.
    
    Represents shapes as measures on position-orientation space,
    enabling robust comparison and matching.
    
    Parameters
    ----------
    sigma_geom : float
        Geometric kernel bandwidth (spatial).
    sigma_orient : float
        Orientation kernel bandwidth.
    kernel_type : str
        Type of kernel: 'gaussian', 'cauchy', or 'laplacian'.
    normalize : bool
        Whether to normalize kernel values.
    """
    
    def __init__(
        self,
        sigma_geom: float = 1.0,
        sigma_orient: float = 1.0,
        kernel_type: Literal['gaussian', 'cauchy', 'laplacian'] = 'gaussian',
        normalize: bool = True
    ):
        self.sigma_geom = sigma_geom
        self.sigma_orient = sigma_orient
        self.kernel_type = kernel_type
        self.normalize = normalize
    
    def compute(
        self,
        points1: np.ndarray,
        normals1: np.ndarray,
        points2: np.ndarray,
        normals2: np.ndarray
    ) -> float:
        """
        Compute varifold kernel between two shapes.
        
        Parameters
        ----------
        points1 : np.ndarray
            First shape points, shape (N, 3).
        normals1 : np.ndarray
            First shape normals, shape (N, 3).
        points2 : np.ndarray
            Second shape points, shape (M, 3).
        normals2 : np.ndarray
            Second shape normals, shape (M, 3).
            
        Returns
        -------
        float
            Varifold kernel value (similarity measure).
        """
        K_pos = self._geometric_kernel(points1, points2)
        K_orient = self._orientation_kernel(normals1, normals2)
        
        K = K_pos * K_orient
        
        if self.normalize:
            norm1 = np.sqrt(self.compute_self(points1, normals1))
            norm2 = np.sqrt(self.compute_self(points2, normals2))
            if norm1 > 0 and norm2 > 0:
                return float(np.sum(K) / (norm1 * norm2))
        
        return float(np.sum(K))
    
    def compute_self(
        self,
        points: np.ndarray,
        normals: np.ndarray
    ) -> float:
        """Compute self-similarity (norm squared)."""
        K_pos = self._geometric_kernel(points, points)
        K_orient = self._orientation_kernel(normals, normals)
        return float(np.sum(K_pos * K_orient))
    
    def compute_matrix(
        self,
        points1: np.ndarray,
        normals1: np.ndarray,
        points2: np.ndarray,
        normals2: np.ndarray
    ) -> np.ndarray:
        """
        Compute full kernel matrix between shapes.
        
        Returns
        -------
        np.ndarray
            Kernel matrix of shape (N, M).
        """
        K_pos = self._geometric_kernel(points1, points2)
        K_orient = self._orientation_kernel(normals1, normals2)
        return K_pos * K_orient
    
    def _geometric_kernel(
        self,
        points1: np.ndarray,
        points2: np.ndarray
    ) -> np.ndarray:
        """Compute geometric (spatial) kernel."""
        D = cdist(points1, points2, metric='euclidean')
        
        if self.kernel_type == 'gaussian':
            K = np.exp(-0.5 * (D / self.sigma_geom) ** 2)
        elif self.kernel_type == 'cauchy':
            K = 1.0 / (1.0 + (D / self.sigma_geom) ** 2)
        else:
            K = np.exp(-D / self.sigma_geom)
        
        return K
    
    def _orientation_kernel(
        self,
        normals1: np.ndarray,
        normals2: np.ndarray
    ) -> np.ndarray:
        """
        Compute orientation kernel.
        
        Uses absolute value of dot product to handle unoriented normals.
        """
        n1 = normals1 / (np.linalg.norm(normals1, axis=1, keepdims=True) + 1e-10)
        n2 = normals2 / (np.linalg.norm(normals2, axis=1, keepdims=True) + 1e-10)
        
        dots = np.abs(n1 @ n2.T)
        
        if self.kernel_type == 'gaussian':
            K = np.exp(-0.5 * ((1 - dots) / self.sigma_orient) ** 2)
        elif self.kernel_type == 'cauchy':
            K = 1.0 / (1.0 + ((1 - dots) / self.sigma_orient) ** 2)
        else:
            K = np.exp(-(1 - dots) / self.sigma_orient)
        
        return K
    
    def gradient(
        self,
        points1: np.ndarray,
        normals1: np.ndarray,
        points2: np.ndarray,
        normals2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient w.r.t. points1 and normals1.
        
        Useful for shape registration/optimization.
        
        Returns
        -------
        grad_points : np.ndarray
            Gradient w.r.t. points, shape (N, 3).
        grad_normals : np.ndarray
            Gradient w.r.t. normals, shape (N, 3).
        """
        K_pos = self._geometric_kernel(points1, points2)
        K_orient = self._orientation_kernel(normals1, normals2)
        
        diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        
        if self.kernel_type == 'gaussian':
            dK_pos = -K_pos[:, :, np.newaxis] * diff / (self.sigma_geom ** 2)
        elif self.kernel_type == 'cauchy':
            factor = -2 * K_pos ** 2 * diff / (self.sigma_geom ** 2)
            dK_pos = factor
        else:
            norm_diff = np.linalg.norm(diff, axis=2, keepdims=True) + 1e-10
            dK_pos = -K_pos[:, :, np.newaxis] * diff / (self.sigma_geom * norm_diff)
        
        grad_points = np.sum(dK_pos * K_orient[:, :, np.newaxis], axis=1)
        
        n1 = normals1 / (np.linalg.norm(normals1, axis=1, keepdims=True) + 1e-10)
        n2 = normals2 / (np.linalg.norm(normals2, axis=1, keepdims=True) + 1e-10)
        
        dots = n1 @ n2.T
        abs_dots = np.abs(dots)
        
        sign = np.sign(dots)
        
        if self.kernel_type == 'gaussian':
            dK_orient = K_orient * (1 - abs_dots) / (self.sigma_orient ** 2)
        else:
            dK_orient = K_orient * (1 - abs_dots) / self.sigma_orient
        
        grad_normals = np.zeros_like(points1)
        for i in range(len(normals1)):
            for j in range(len(normals2)):
                grad_normals[i] += sign[i, j] * dK_orient[i, j] * K_pos[i, j] * n2[j]
        
        return grad_points, grad_normals
    
    def register(
        self,
        source_points: np.ndarray,
        source_normals: np.ndarray,
        target_points: np.ndarray,
        target_normals: np.ndarray,
        n_iter: int = 100,
        step_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Register source to target using gradient descent.
        
        Parameters
        ----------
        source_points, source_normals : np.ndarray
            Source shape.
        target_points, target_normals : np.ndarray
            Target shape.
        n_iter : int
            Number of iterations.
        step_size : float
            Gradient descent step size.
            
        Returns
        -------
        registered_points : np.ndarray
            Deformed source points.
        registered_normals : np.ndarray
            Deformed source normals.
        """
        points = source_points.copy()
        normals = source_normals.copy()
        
        for _ in range(n_iter):
            grad_p, grad_n = self.gradient(points, normals, target_points, target_normals)
            
            points = points + step_size * grad_p
            
            normals = normals + step_size * grad_n
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / (norms + 1e-10)
        
        return points, normals
