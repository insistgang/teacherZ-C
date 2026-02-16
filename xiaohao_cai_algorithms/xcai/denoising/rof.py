"""
ROF (Rudin-Osher-Fatemi) Denoising Implementation

Based on:
- Rudin, Osher, Fatemi (1992): Nonlinear total variation based noise removal algorithms
- Chambolle (2004): An algorithm for total variation minimization and applications
"""

import numpy as np
from typing import Optional, Literal


class ROFDenoiser:
    """
    ROF-based image denoiser using Total Variation regularization.
    
    Parameters
    ----------
    lambda_param : float
        Regularization parameter controlling the strength of TV regularization.
        Larger values preserve more edges, smaller values produce smoother results.
    method : str
        Denoising method: 'chambolle' (default) or 'gradient_descent'.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence.
    """
    
    def __init__(
        self,
        lambda_param: float = 0.1,
        method: Literal['chambolle', 'gradient_descent'] = 'chambolle',
        max_iter: int = 100,
        tol: float = 1e-4
    ):
        self.lambda_param = lambda_param
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise an image using ROF model.
        
        Parameters
        ----------
        image : np.ndarray
            Input noisy image (2D or 3D for color images).
            
        Returns
        -------
        np.ndarray
            Denoised image.
        """
        if self.method == 'chambolle':
            return self._chambolle_denoise(image)
        else:
            return self._gradient_descent_denoise(image)
    
    def _chambolle_denoise(self, f: np.ndarray) -> np.ndarray:
        """
        Chambolle's algorithm for ROF denoising.
        
        Solves: min_u TV(u) + (1/2λ)||u - f||^2
        """
        if f.ndim == 3:
            result = np.zeros_like(f)
            for c in range(f.shape[2]):
                result[:, :, c] = self._chambolle_2d(f[:, :, c])
            return result
        return self._chambolle_2d(f)
    
    def _chambolle_2d(self, f: np.ndarray) -> np.ndarray:
        """Chambolle's algorithm for 2D images."""
        h, w = f.shape
        px = np.zeros((h, w), dtype=f.dtype)
        py = np.zeros((h, w), dtype=f.dtype)
        
        tau = 0.25
        
        for _ in range(self.max_iter):
            px_old = px.copy()
            py_old = py.copy()
            
            div_p = self._divergence(px, py)
            ux = np.roll(div_p - f / self.lambda_param, -1, axis=1) - (div_p - f / self.lambda_param)
            uy = np.roll(div_p - f / self.lambda_param, -1, axis=0) - (div_p - f / self.lambda_param)
            
            px = px + tau * ux
            py = py + tau * uy
            
            norm_p = np.sqrt(px**2 + py**2)
            norm_p = np.maximum(norm_p, 1.0)
            px = px / norm_p
            py = py / norm_p
            
            if np.max(np.abs(px - px_old)) < self.tol and np.max(np.abs(py - py_old)) < self.tol:
                break
        
        return f - self.lambda_param * self._divergence(px, py)
    
    def _divergence(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Compute divergence of vector field (px, py)."""
        dx = px - np.roll(px, 1, axis=1)
        dy = py - np.roll(py, 1, axis=0)
        return dx + dy
    
    def _gradient_descent_denoise(self, f: np.ndarray) -> np.ndarray:
        """
        Gradient descent for ROF denoising.
        
        Solves: min_u TV(u) + (1/2λ)||u - f||^2
        """
        u = f.copy()
        dt = 0.1
        
        for _ in range(self.max_iter):
            u_old = u.copy()
            
            grad_x = np.roll(u, -1, axis=1) - u
            grad_y = np.roll(u, -1, axis=0) - u
            
            norm_grad = np.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            grad_x = grad_x / norm_grad
            grad_y = grad_y / norm_grad
            
            div = (grad_x - np.roll(grad_x, 1, axis=1)) + (grad_y - np.roll(grad_y, 1, axis=0))
            
            u = u + dt * (div + (f - u) / self.lambda_param)
            
            if np.max(np.abs(u - u_old)) < self.tol:
                break
        
        return u
