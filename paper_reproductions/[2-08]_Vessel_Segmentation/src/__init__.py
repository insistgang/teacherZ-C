"""
小波框架血管分割实现包

本包实现了基于小波框架的视网膜血管分割方法。

主要模块:
    - vessel_net: 血管分割网络架构
    - wavelet_frame: 小波框架特征提取
    - evaluate: 评估指标 (AUC, 灵敏度, 特异度等)
    - dataset: DRIVE等数据集处理

使用示例:
    >>> from src.vessel_net import VesselSegNet
    >>> from src.dataset import DRIVEDataset
    >>> model = VesselSegNet(use_wavelet=True)
    >>> dataset = DRIVEDataset(root='./data/DRIVE')

作者: Xiaohao Cai
版本: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Xiaohao Cai"

from .vessel_net import VesselSegNet
from .wavelet_frame import WaveletFrameModule
from .evaluate import (
    compute_auc,
    compute_accuracy,
    compute_sensitivity_specificity,
    compute_metrics
)
from .dataset import DRIVEDataset

__all__ = [
    "VesselSegNet",
    "WaveletFrameModule",
    "compute_auc",
    "compute_accuracy",
    "compute_sensitivity_specificity",
    "compute_metrics",
    "DRIVEDataset",
]
