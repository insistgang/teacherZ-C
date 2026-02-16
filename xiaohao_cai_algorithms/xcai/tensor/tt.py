"""
Tensor Train (TT) Decomposition Implementation

Based on:
- Oseledets (2011): Tensor-train decomposition
- Oseledets & Tyrtyshnikov (2009): Breaking the curse of dimensionality
"""

import numpy as np
from typing import List, Tuple, Optional


class TTDecomposer:
    """
    Tensor Train (TT/MPS) decomposition.
    
    Decomposes tensor into chain of 3-way cores:
    T(i1, i2, ..., id) = G1(i1) × G2(:, i2, :) × ... × Gd(:, id)
    
    Parameters
    ----------
    eps : float
        Relative error tolerance for truncation.
    max_rank : int, optional
        Maximum TT-rank.
    """
    
    def __init__(
        self,
        eps: float = 1e-6,
        max_rank: Optional[int] = None
    ):
        self.eps = eps
        self.max_rank = max_rank
    
    def decompose(self, tensor: np.ndarray) -> List[np.ndarray]:
        """
        Perform TT decomposition using TT-SVD algorithm.
        
        Parameters
        ----------
        tensor : np.ndarray
            Input tensor.
            
        Returns
        -------
        cores : list of np.ndarray
            TT cores.
        """
        return self._tt_svd(tensor)
    
    def _tt_svd(self, tensor: np.ndarray) -> List[np.ndarray]:
        """TT-SVD algorithm."""
        ndim = tensor.ndim
        cores = []
        
        current = tensor.copy()
        ranks = [1]
        
        for k in range(ndim - 1):
            shape = current.shape
            
            if k == 0:
                matrix = current.reshape(shape[0], -1)
            else:
                r_prev = shape[0]
                n_k = shape[1]
                matrix = current.reshape(r_prev * n_k, -1)
            
            U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            
            frob_norm = np.sqrt(np.sum(S ** 2))
            cumsum = np.cumsum(S ** 2)
            threshold = frob_norm ** 2 * (1 - self.eps ** 2)
            
            r_k = np.searchsorted(cumsum, threshold) + 1
            r_k = min(r_k, len(S))
            
            if self.max_rank is not None:
                r_k = min(r_k, self.max_rank)
            
            U_k = U[:, :r_k]
            S_k = S[:r_k]
            Vt_k = Vt[:r_k, :]
            
            if k == 0:
                core = U_k.reshape(1, shape[0], r_k)
                current = np.diag(S_k) @ Vt_k
            else:
                r_prev = shape[0]
                n_k = shape[1]
                core = U_k.reshape(r_prev, n_k, r_k)
                current = np.diag(S_k) @ Vt_k
            
            cores.append(core)
            ranks.append(r_k)
        
        last_shape = list(current.shape)
        if len(last_shape) == 1:
            current = current.reshape(1, -1, 1)
        else:
            current = current.reshape(last_shape[0], last_shape[1], 1)
        cores.append(current)
        
        return cores
    
    @staticmethod
    def reconstruct(cores: List[np.ndarray]) -> np.ndarray:
        """
        Reconstruct full tensor from TT cores.
        
        Parameters
        ----------
        cores : list of np.ndarray
            TT cores.
            
        Returns
        -------
        np.ndarray
            Reconstructed tensor.
        """
        result = cores[0]
        
        for k in range(1, len(cores)):
            r1 = result.shape[-1]
            r2 = cores[k].shape[0]
            
            if r1 != r2:
                min_r = min(r1, r2)
                result = result[..., :min_r]
                core_k = cores[k][:min_r, ...]
            else:
                core_k = cores[k]
            
            result = np.einsum('...i,ijk->...jk', result.squeeze(), core_k)
        
        if result.ndim > 0:
            while result.shape[0] == 1:
                result = result[0]
            while result.shape[-1] == 1:
                result = result[..., 0]
        
        return result
    
    def get_tt_rank(self, cores: List[np.ndarray]) -> List[int]:
        """
        Get TT-rank from cores.
        
        Parameters
        ----------
        cores : list of np.ndarray
            TT cores.
            
        Returns
        -------
        list of int
            TT-ranks including boundary ranks (1, ..., 1).
        """
        ranks = [core.shape[0] for core in cores]
        ranks.append(cores[-1].shape[-1])
        return ranks
    
    def compress(
        self,
        cores: List[np.ndarray],
        new_eps: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Compress TT representation by rounding.
        
        Parameters
        ----------
        cores : list of np.ndarray
            TT cores to compress.
        new_eps : float, optional
            New error tolerance.
            
        Returns
        -------
        list of np.ndarray
            Compressed TT cores.
        """
        if new_eps is not None:
            self.eps = new_eps
        
        tensor = self.reconstruct(cores)
        return self.decompose(tensor)
    
    def get_storage_cost(self, cores: List[np.ndarray]) -> int:
        """
        Compute storage cost of TT representation.
        
        Parameters
        ----------
        cores : list of np.ndarray
            TT cores.
            
        Returns
        -------
        int
            Total number of elements stored.
        """
        return sum(core.size for core in cores)
