"""
Tucker Decomposition Implementation

Based on:
- Tucker (1966): Some mathematical notes on three-mode factor analysis
- Lathauwer et al. (2000): A multilinear singular value decomposition
"""

import numpy as np
from typing import Optional, List, Tuple, Union


class TuckerDecomposer:
    """
    Tucker decomposition (Higher-order SVD).
    
    Decomposes a tensor into a core tensor and factor matrices:
    T = G ×_1 U_1 ×_2 U_2 × ... ×_N U_N
    
    Parameters
    ----------
    rank : list or tuple
        Target rank for each mode.
    method : str
        Decomposition method: 'hosvd' or 'hooi'.
    max_iter : int
        Maximum iterations for HOOI.
    tol : float
        Tolerance for convergence.
    """
    
    def __init__(
        self,
        rank: Union[List[int], Tuple[int, ...]],
        method: str = 'hosvd',
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        self.rank = list(rank)
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
    
    def decompose(
        self,
        tensor: np.ndarray,
        return_errors: bool = False
    ) -> Union[Tuple[np.ndarray, List[np.ndarray]], Tuple[np.ndarray, List[np.ndarray], List[float]]]:
        """
        Perform Tucker decomposition on a tensor.
        
        Parameters
        ----------
        tensor : np.ndarray
            Input tensor of arbitrary order.
        return_errors : bool
            If True, return reconstruction errors per iteration.
            
        Returns
        -------
        core : np.ndarray
            Core tensor.
        factors : list of np.ndarray
            Factor matrices for each mode.
        errors : list of float, optional
            Reconstruction errors (if return_errors=True).
        """
        if self.method == 'hosvd':
            return self._hosvd(tensor, return_errors)
        else:
            return self._hooi(tensor, return_errors)
    
    def _hosvd(
        self,
        tensor: np.ndarray,
        return_errors: bool = False
    ) -> Union[Tuple[np.ndarray, List[np.ndarray]], Tuple[np.ndarray, List[np.ndarray], List[float]]]:
        """
        Higher-Order SVD (HOSVD).
        
        Compute factor matrices via SVD of mode-n unfoldings,
        then project tensor onto factor spaces to get core.
        """
        ndim = tensor.ndim
        factors = []
        
        for mode in range(ndim):
            unfolding = self._unfold(tensor, mode)
            r = min(self.rank[mode], unfolding.shape[0])
            U, _, _ = np.linalg.svd(unfolding, full_matrices=False)
            factors.append(U[:, :r])
        
        core = tensor.copy()
        for mode in range(ndim):
            core = self._mode_n_product(core, factors[mode].T, mode)
        
        if return_errors:
            recon = self._reconstruct(core, factors)
            error = float(np.linalg.norm(tensor - recon) / np.linalg.norm(tensor))
            return core, factors, [error]
        
        return core, factors
    
    def _hooi(
        self,
        tensor: np.ndarray,
        return_errors: bool = False
    ) -> Union[Tuple[np.ndarray, List[np.ndarray]], Tuple[np.ndarray, List[np.ndarray], List[float]]]:
        """
        Higher-Order Orthogonal Iteration (HOOI).
        
        Alternating optimization for Tucker decomposition.
        """
        ndim = tensor.ndim
        factors = []
        errors = []
        
        for mode in range(ndim):
            unfolding = self._unfold(tensor, mode)
            r = min(self.rank[mode], unfolding.shape[0])
            U, _, _ = np.linalg.svd(unfolding, full_matrices=False)
            factors.append(U[:, :r])
        
        for iteration in range(self.max_iter):
            new_factors = []
            
            for mode in range(ndim):
                other_modes = [i for i in range(ndim) if i != mode]
                
                Y = tensor.copy()
                for m in other_modes:
                    Y = self._mode_n_product(Y, factors[m].T, m if m < mode else m)
                
                Y_unfold = self._unfold(Y, mode)
                r = min(self.rank[mode], Y_unfold.shape[0])
                U, _, _ = np.linalg.svd(Y_unfold, full_matrices=False)
                new_factors.append(U[:, :r])
            
            factors = new_factors
            
            if return_errors:
                core = tensor.copy()
                for mode in range(ndim):
                    core = self._mode_n_product(core, factors[mode].T, mode)
                recon = self._reconstruct(core, factors)
                error = float(np.linalg.norm(tensor - recon) / np.linalg.norm(tensor))
                errors.append(error)
                
                if iteration > 0 and abs(errors[-2] - errors[-1]) < self.tol:
                    break
        
        core = tensor.copy()
        for mode in range(ndim):
            core = self._mode_n_product(core, factors[mode].T, mode)
        
        if return_errors:
            return core, factors, errors
        
        return core, factors
    
    def _unfold(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """
        Mode-n unfolding of tensor.
        
        Rearranges tensor into matrix by putting mode-n fibers as columns.
        """
        ndim = tensor.ndim
        perm = [mode] + [i for i in range(ndim) if i != mode]
        return np.transpose(tensor, perm).reshape(tensor.shape[mode], -1)
    
    def _mode_n_product(
        self,
        tensor: np.ndarray,
        matrix: np.ndarray,
        mode: int
    ) -> np.ndarray:
        """
        Mode-n product of tensor with matrix.
        
        Computes T ×_n M where M multiplies along mode n.
        """
        ndim = tensor.ndim
        perm = list(range(ndim))
        perm[0], perm[mode] = perm[mode], perm[0]
        
        tensor_reshaped = np.transpose(tensor, perm).reshape(tensor.shape[mode], -1)
        result = matrix @ tensor_reshaped
        
        result_shape = list(tensor.shape)
        result_shape[mode] = matrix.shape[0]
        result = result.reshape([result_shape[p] for p in perm])
        result = np.transpose(result, np.argsort(perm))
        
        return result
    
    def _reconstruct(
        self,
        core: np.ndarray,
        factors: List[np.ndarray]
    ) -> np.ndarray:
        """Reconstruct tensor from core and factors."""
        result = core.copy()
        for mode, factor in enumerate(factors):
            result = self._mode_n_product(result, factor, mode)
        return result
    
    def fit_transform(
        self,
        tensor: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Fit and transform - alias for decompose.
        
        Parameters
        ----------
        tensor : np.ndarray
            Input tensor.
            
        Returns
        -------
        core : np.ndarray
            Core tensor.
        factors : list
            Factor matrices.
        """
        return self.decompose(tensor)
    
    @staticmethod
    def reconstruct(
        core: np.ndarray,
        factors: List[np.ndarray]
    ) -> np.ndarray:
        """
        Reconstruct full tensor from Tucker decomposition.
        
        Parameters
        ----------
        core : np.ndarray
            Core tensor.
        factors : list of np.ndarray
            Factor matrices.
            
        Returns
        -------
        np.ndarray
            Reconstructed tensor.
        """
        ndim = core.ndim
        result = core.copy()
        
        for mode, factor in enumerate(factors):
            perm = [mode] + [i for i in range(ndim) if i != mode]
            inv_perm = np.argsort(perm)
            
            tensor_reshaped = np.transpose(result, perm).reshape(result.shape[mode], -1)
            result = factor @ tensor_reshaped
            
            new_shape = [factor.shape[0]] + [result.shape[1]]
            result = result.reshape(new_shape)
            
            full_shape = list(result.shape[:-1]) + [core.shape[i] for i in range(ndim) if i != mode]
        
        return result
