"""
Utility functions for PEFT module.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    trainable_only : bool
        Count only trainable parameters.
        
    Returns
    -------
    int
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def freeze_model(model: nn.Module) -> None:
    """Freeze all parameters in model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module) -> None:
    """Unfreeze all parameters in model."""
    for param in model.parameters():
        param.requires_grad = True


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from model.
    
    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
        
    Returns
    -------
    list
        List of LoRA parameters.
    """
    lora_params = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A'):
            lora_params.append(module.lora_A)
        if hasattr(module, 'lora_B'):
            lora_params.append(module.lora_B)
    return lora_params


def set_lora_trainable_only(model: nn.Module) -> None:
    """Freeze all parameters except LoRA parameters."""
    freeze_model(model)
    
    for param in get_lora_parameters(model):
        param.requires_grad = True


def merge_all_lora_layers(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA layers in model.
    
    Parameters
    ----------
    model : nn.Module
        Model with LoRA layers.
        
    Returns
    -------
    nn.Module
        Model with merged weights.
    """
    for name, module in list(model.named_modules()):
        if hasattr(module, 'merge') and callable(module.merge):
            merged = module.merge()
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            
            setattr(parent, child_name, merged)
    
    return model


def compute_rank_utilization(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    threshold: float = 0.01
) -> Dict[str, float]:
    """
    Compute rank utilization metrics.
    
    Parameters
    ----------
    lora_A : torch.Tensor
        LoRA A matrix (rank x in_features).
    lora_B : torch.Tensor
        LoRA B matrix (out_features x rank).
    threshold : float
        Threshold for considering a rank dimension utilized.
        
    Returns
    -------
    dict
        Dictionary with utilization metrics.
    """
    rank = lora_A.shape[0]
    
    A_norms = torch.norm(lora_A, dim=1)
    B_norms = torch.norm(lora_B, dim=0)
    
    combined_norms = A_norms * B_norms
    
    max_norm = combined_norms.max()
    normalized_norms = combined_norms / (max_norm + 1e-10)
    
    utilized = (normalized_norms > threshold).sum().item()
    
    return {
        'total_rank': rank,
        'utilized_rank': utilized,
        'utilization_ratio': utilized / rank,
        'norm_distribution': normalized_norms.tolist()
    }


def estimate_optimal_rank(
    weight: torch.Tensor,
    target_energy: float = 0.95
) -> int:
    """
    Estimate optimal LoRA rank using SVD energy.
    
    Parameters
    ----------
    weight : torch.Tensor
        Weight matrix.
    target_energy : float
        Target fraction of singular value energy to preserve.
        
    Returns
    -------
    int
        Estimated optimal rank.
    """
    with torch.no_grad():
        _, S, _ = torch.linalg.svd(weight, full_matrices=False)
        
        energy = torch.cumsum(S ** 2, dim=0)
        total_energy = energy[-1]
        
        target = total_energy * target_energy
        optimal_rank = (energy <= target).sum().item() + 1
        optimal_rank = min(optimal_rank, len(S))
    
    return optimal_rank
