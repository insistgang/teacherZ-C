"""
Segmentation module containing SLaT and GraphCut segmentation algorithms.
"""

from xcai.segmentation.slat import SLatSegmenter
from xcai.segmentation.graphcut import GraphCutSegmenter
from xcai.segmentation.utils import compute_iou, compute_dice

__all__ = ['SLatSegmenter', 'GraphCutSegmenter', 'compute_iou', 'compute_dice']
