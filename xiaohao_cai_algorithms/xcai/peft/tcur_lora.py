"""
Tensor CUR LoRA Implementation

Based on:
- Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models
- CUR decomposition for structured low-rank approximation
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn


class TCURLoRA:
    """
    Tensor CUR-based Low-Rank Adaptation.
    
    Combines LoRA with tensor CUR decomposition for efficient
    parameter-efficient fine-tuning with structured updates.
    
    Parameters
    ----------
    rank : int
        Target rank for LoRA matrices.
    alpha : float
        Scaling factor for LoRA update.
    cur_method : str
        CUR selection method: 'random', 'pivoted', 'leveraged'.
    dropout : float
        Dropout rate.
    """
    
    def __init__(
        self,
        rank: int = 8,
        alpha: float = 1.0,
        cur_method: str = 'pivoted',
        dropout: float = 0.0
    ):
        self.rank = rank
        self.alpha = alpha
        self.cur_method = cur_method
        self.dropout = dropout
    
    def create_lora_layer(
        self,
        original_weight: Union[np.ndarray, torch.Tensor],
        device: Optional[str] = None
    ) -> 'LoRALayer':
        """
        Create LoRA layer using CUR initialization.
        
        Parameters
        ----------
        original_weight : np.ndarray or torch.Tensor
            Original weight matrix of shape (out_features, in_features).
        device : str, optional
            Device for PyTorch tensors.
            
        Returns
        -------
        LoRALayer
            Configured LoRA layer.
        """
        if isinstance(original_weight, np.ndarray):
            weight = torch.from_numpy(original_weight).float()
        else:
            weight = original_weight.float()
        
        out_features, in_features = weight.shape
        
        lora_A, lora_B = self._cur_initialize(weight)
        
        layer = LoRALayer(
            in_features=in_features,
            out_features=out_features,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout
        )
        
        layer.lora_A.data = lora_A
        layer.lora_B.data = lora_B
        
        if device:
            layer = layer.to(device)
        
        return layer
    
    def _cur_initialize(
        self,
        weight: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LoRA matrices using CUR decomposition.
        
        Uses CUR to find important columns/rows for initialization.
        """
        out_features, in_features = weight.shape
        k = min(self.rank, min(out_features, in_features))
        
        W = weight.detach().cpu().numpy()
        
        if self.cur_method == 'random':
            col_idx = np.random.choice(in_features, k, replace=False)
            row_idx = np.random.choice(out_features, k, replace=False)
        elif self.cur_method == 'leveraged':
            col_idx = self._leveraged_selection(W, k, axis=1)
            row_idx = self._leveraged_selection(W, k, axis=0)
        else:
            col_idx, row_idx = self._pivoted_selection(W, k)
        
        C = W[:, col_idx]
        R = W[row_idx, :]
        
        W_int = W[np.ix_(row_idx, col_idx)]
        
        try:
            U = np.linalg.pinv(W_int)
        except np.linalg.LinAlgError:
            U = np.linalg.lstsq(W_int, np.eye(k), rcond=None)[0]
        
        lora_A = torch.from_numpy(C).float()
        lora_B = torch.from_numpy(U @ R).float()
        
        if lora_A.shape[1] != k:
            lora_A = lora_A[:, :k]
        if lora_B.shape[0] != k:
            lora_B = lora_B[:k, :]
        
        return lora_A.T, lora_B
    
    def _leveraged_selection(
        self,
        matrix: np.ndarray,
        k: int,
        axis: int
    ) -> np.ndarray:
        """Statistical leverage score selection."""
        if axis == 1:
            _, S, Vt = np.linalg.svd(matrix, full_matrices=False)
            r = min(k, len(S))
            leverage = np.sum(Vt[:r, :] ** 2, axis=0) / r
        else:
            U, _, _ = np.linalg.svd(matrix, full_matrices=False)
            r = min(k, min(matrix.shape))
            leverage = np.sum(U[:, :r] ** 2, axis=1) / r
        
        leverage = leverage / leverage.sum()
        indices = np.random.choice(len(leverage), k, replace=False, p=leverage)
        return indices
    
    def _pivoted_selection(
        self,
        matrix: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pivoted selection using QR decomposition."""
        from scipy.linalg import qr
        
        _, _, P_col = qr(matrix, pivoting=True)
        _, _, P_row = qr(matrix.T, pivoting=True)
        
        return P_col[:k], P_row[:k]
    
    @staticmethod
    def merge_lora_weights(
        original_weight: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Merge LoRA weights into original weight.
        
        W' = W + (alpha/r) * B @ A
        
        Parameters
        ----------
        original_weight : torch.Tensor
            Original weight matrix.
        lora_A : torch.Tensor
            LoRA A matrix (rank x in_features).
        lora_B : torch.Tensor
            LoRA B matrix (out_features x rank).
        alpha : float
            Scaling factor.
            
        Returns
        -------
        torch.Tensor
            Merged weight matrix.
        """
        rank = lora_A.shape[0]
        scaling = alpha / rank
        
        delta = lora_B @ lora_A * scaling
        
        return original_weight + delta


class LoRALayer(nn.Module):
    """
    LoRA Layer implementation.
    
    Computes: output = W @ x + (alpha/r) * B @ A @ x
    
    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.
    rank : int
        LoRA rank.
    alpha : float
        Scaling factor.
    dropout : float
        Dropout rate.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_features).
            
        Returns
        -------
        torch.Tensor
            LoRA output of shape (batch, out_features).
        """
        x = self.dropout(x)
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
    
    def merge_with_linear(
        self,
        linear: nn.Linear
    ) -> nn.Linear:
        """
        Merge LoRA into a linear layer.
        
        Parameters
        ----------
        linear : nn.Linear
            Original linear layer.
            
        Returns
        -------
        nn.Linear
            Linear layer with merged weights.
        """
        with torch.no_grad():
            merged_weight = TCURLoRA.merge_lora_weights(
                linear.weight,
                self.lora_A,
                self.lora_B,
                self.alpha
            )
            
            new_linear = nn.Linear(
                self.in_features,
                self.out_features,
                bias=linear.bias is not None
            )
            new_linear.weight.data = merged_weight
            if linear.bias is not None:
                new_linear.bias.data = linear.bias.data.clone()
        
        return new_linear


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA.
    
    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.
    rank : int
        LoRA rank.
    alpha : float
        Scaling factor.
    dropout : float
        Dropout rate.
    bias : bool
        Whether to include bias.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 1.0,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining linear and LoRA."""
        return self.linear(x) + self.lora(x)
    
    def merge(self) -> nn.Linear:
        """Merge LoRA into linear and return new linear layer."""
        return self.lora.merge_with_linear(self.linear)
