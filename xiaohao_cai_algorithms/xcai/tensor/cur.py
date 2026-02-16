"""
CUR Decomposition Implementation

Based on:
- Mahoney & Drineas (2009): CUR matrix decompositions for improved data analysis
- Oseledets & Tyrtyshnikov (2013): TT-cross approximation
"""

import numpy as np
from typing import Tuple, List, Optional, Union


class CURDecomposer:
    """
    CUR/Column-Row decomposition for matrices and tensors.
    
    Represents a matrix as A ≈ C × U × R where:
    - C: subset of columns
    - R: subset of rows  
    - U: intersection matrix
    
    Parameters
    ----------
    rank : int or tuple
        Target rank (or ranks for tensor).
    method : str
        Selection method: 'random', 'pivoted', 'leveraged'.
    max_iter : int
        Maximum iterations for tensor CUR.
    tol : float
        Tolerance for convergence.
    """
    
    def __init__(
        self,
        rank: Union[int, Tuple[int, ...]] = 10,
        method: str = 'pivoted',
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        if isinstance(rank, int):
            self.rank = rank
            self.is_tensor = False
        else:
            self.rank = list(rank)
            self.is_tensor = True
        
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
    
    def decompose(
        self,
        matrix: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]],
               Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]]]]:
        """
        Perform CUR decomposition.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix or tensor.
            
        Returns
        -------
        C : np.ndarray
            Column submatrix.
        U : np.ndarray
            Intersection matrix.
        R : np.ndarray
            Row submatrix.
        indices : list
            Selected column/row indices.
        """
        if matrix.ndim == 2:
            return self._matrix_cur(matrix)
        else:
            return self._tensor_cur(matrix)
    
    def _matrix_cur(
        self,
        matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int]]:
        """CUR decomposition for matrices."""
        m, n = matrix.shape
        k = min(self.rank, min(m, n))
        
        if self.method == 'random':
            col_indices = self._random_selection(matrix, k, axis=1)
            row_indices = self._random_selection(matrix, k, axis=0)
        elif self.method == 'leveraged':
            col_indices = self._leveraged_selection(matrix, k, axis=1)
            row_indices = self._leveraged_selection(matrix, k, axis=0)
        else:
            col_indices, row_indices = self._pivoted_selection(matrix, k)
        
        C = matrix[:, col_indices]
        R = matrix[row_indices, :]
        
        W = matrix[np.ix_(row_indices, col_indices)]
        U = self._compute_intersection_matrix(W)
        
        return C, U, R, col_indices, row_indices
    
    def _random_selection(
        self,
        matrix: np.ndarray,
        k: int,
        axis: int
    ) -> List[int]:
        """Random column/row selection."""
        n = matrix.shape[1] if axis == 1 else matrix.shape[0]
        return list(np.random.choice(n, min(k, n), replace=False))
    
    def _leveraged_selection(
        self,
        matrix: np.ndarray,
        k: int,
        axis: int
    ) -> List[int]:
        """Statistical leverage score based selection."""
        if axis == 1:
            m, n = matrix.shape
            _, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            r = min(k, len(S))
            leverage = np.sum(Vt[:r, :] ** 2, axis=0) / r
        else:
            m, n = matrix.shape
            U, _, _ = np.linalg.svd(matrix, full_matrices=False)
            r = min(k, min(m, n))
            leverage = np.sum(U[:, :r] ** 2, axis=1) / r
        
        leverage = leverage / leverage.sum()
        
        indices = []
        for _ in range(k):
            available = [i for i in range(len(leverage)) if i not in indices]
            if not available:
                break
            probs = leverage[available]
            probs = probs / probs.sum()
            idx = np.random.choice(available, p=probs)
            indices.append(idx)
        
        return indices
    
    def _pivoted_selection(
        self,
        matrix: np.ndarray,
        k: int
    ) -> Tuple[List[int], List[int]]:
        """Pivoted QR / LU based selection."""
        from scipy.linalg import qr
        
        m, n = matrix.shape
        k = min(k, min(m, n))
        
        Q, R, P_col = qr(matrix, pivoting=True)
        col_indices = list(P_col[:k])
        
        Q, R, P_row = qr(matrix.T, pivoting=True)
        row_indices = list(P_row[:k])
        
        return col_indices, row_indices
    
    def _compute_intersection_matrix(self, W: np.ndarray) -> np.ndarray:
        """Compute intersection matrix U from W."""
        try:
            U = np.linalg.pinv(W)
        except np.linalg.LinAlgError:
            U = np.linalg.lstsq(W, np.eye(W.shape[0]), rcond=None)[0]
        return U
    
    def _tensor_cur(
        self,
        tensor: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
        """Tensor CUR using skeleton/TT-cross approximation."""
        ndim = tensor.ndim
        indices = []
        
        for mode in range(ndim):
            unfolding = self._unfold(tensor, mode)
            k = min(self.rank[mode], min(unfolding.shape))
            
            if self.method == 'pivoted':
                _, _, P = self._pivoted_qr(unfolding)
                mode_indices = list(P[:k])
            else:
                mode_indices = self._leveraged_selection(unfolding, k, 1)
            
            indices.append(mode_indices)
        
        core_indices = tuple(indices)
        core = tensor[core_indices]
        
        fibers = []
        for mode in range(ndim):
            mode_fibers = []
            for idx in indices[mode]:
                slicer = [slice(None)] * ndim
                slicer[mode] = idx
                mode_fibers.append(tensor[tuple(slicer)])
            fibers.append(np.array(mode_fibers))
        
        U = self._compute_intersection_matrix(core)
        
        C = fibers[0] if ndim >= 1 else core
        R = fibers[-1] if ndim >= 1 else core
        
        return C, U, R, indices
    
    def _unfold(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Mode-n unfolding."""
        ndim = tensor.ndim
        perm = [mode] + [i for i in range(ndim) if i != mode]
        return np.transpose(tensor, perm).reshape(tensor.shape[mode], -1)
    
    def _pivoted_qr(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pivoted QR decomposition."""
        from scipy.linalg import qr
        return qr(matrix, pivoting=True)
    
    @staticmethod
    def reconstruct(
        C: np.ndarray,
        U: np.ndarray,
        R: np.ndarray
    ) -> np.ndarray:
        """
        Reconstruct matrix from CUR decomposition.
        
        Parameters
        ----------
        C : np.ndarray
            Column submatrix.
        U : np.ndarray
            Intersection matrix.
        R : np.ndarray
            Row submatrix.
            
        Returns
        -------
        np.ndarray
            Reconstructed matrix A ≈ C @ U @ R.
        """
        return C @ U @ R
    
    def fit_transform(
        self,
        matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit and transform - alias for decompose.
        
        Parameters
        ----------
        matrix : np.ndarray
            Input matrix.
            
        Returns
        -------
        C, U, R : tuple
            CUR decomposition.
        """
        C, U, R, _, _ = self.decompose(matrix)
        return C, U, R
