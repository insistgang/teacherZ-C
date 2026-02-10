"""
小波框架管状结构分割实现包

本包实现了基于小波框架的管状结构分割方法，包括:
- 小波框架变换
- 形状先验建模
- 变分分割算法

主要模块:
    - framelet: 小波框架变换实现
    - shape_prior: 管状结构形状先验
    - segmentation: 分割算法
    - utils: 工具函数

使用示例:
    >>> from src.framelet import FrameletTransform
    >>> from src.segmentation import TubularSegmentation
    >>> framelet = FrameletTransform(level=3)
    >>> segmenter = TubularSegmentation()
    >>> result = segmenter.segment(image)

作者: Xiaohao Cai
版本: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Xiaohao Cai"

from .framelet import FrameletTransform, BiorthogonalFramelet, TightFramelet
from .shape_prior import TubularShapePrior, CurvaturePrior
from .segmentation import TubularSegmentation
from .utils import load_image, save_image, visualize_segmentation

__all__ = [
    "FrameletTransform",
    "BiorthogonalFramelet",
    "TightFramelet",
    "TubularShapePrior",
    "CurvaturePrior",
    "TubularSegmentation",
    "load_image",
    "save_image",
    "visualize_segmentation",
]
