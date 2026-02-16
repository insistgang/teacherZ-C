"""
Utility functions for segmentation module.
"""

import numpy as np
from typing import Optional


def compute_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    class_label: Optional[int] = None
) -> float:
    """
    Compute Intersection over Union (IoU / Jaccard Index).
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted segmentation.
    gt : np.ndarray
        Ground truth segmentation.
    class_label : int, optional
        If provided, compute IoU for specific class.
        
    Returns
    -------
    float
        IoU value in range [0, 1].
    """
    if class_label is not None:
        pred = (pred == class_label)
        gt = (gt == class_label)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def compute_dice(
    pred: np.ndarray,
    gt: np.ndarray,
    class_label: Optional[int] = None
) -> float:
    """
    Compute Dice coefficient (F1 score).
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted segmentation.
    gt : np.ndarray
        Ground truth segmentation.
    class_label : int, optional
        If provided, compute Dice for specific class.
        
    Returns
    -------
    float
        Dice coefficient in range [0, 1].
    """
    if class_label is not None:
        pred = (pred == class_label)
        gt = (gt == class_label)
    
    intersection = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(2 * intersection / total)


def compute_mean_iou(pred: np.ndarray, gt: np.ndarray, n_classes: int) -> float:
    """
    Compute mean IoU across all classes.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted segmentation.
    gt : np.ndarray
        Ground truth segmentation.
    n_classes : int
        Number of classes.
        
    Returns
    -------
    float
        Mean IoU.
    """
    ious = []
    for c in range(n_classes):
        iou = compute_iou(pred, gt, c)
        ious.append(iou)
    
    return float(np.mean(ious))


def compute_pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Compute pixel-wise accuracy.
    
    Parameters
    ----------
    pred : np.ndarray
        Predicted segmentation.
    gt : np.ndarray
        Ground truth segmentation.
        
    Returns
    -------
    float
        Pixel accuracy in range [0, 1].
    """
    return float(np.mean(pred == gt))
