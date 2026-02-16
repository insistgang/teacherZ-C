"""
Xiaohao Cai Algorithms Package

Image analysis algorithms based on Xiaohao Cai research.
"""

__version__ = '1.0.0'

from xcai.denoising import ROFDenoiser
from xcai.segmentation import SLatSegmenter, GraphCutSegmenter
from xcai.tensor import TuckerDecomposer, TTDecomposer, CURDecomposer
from xcai.pointcloud import VarifoldKernel
from xcai.peft import TCURLoRA

__all__ = [
    'ROFDenoiser',
    'SLatSegmenter',
    'GraphCutSegmenter',
    'TuckerDecomposer',
    'TTDecomposer',
    'CURDecomposer',
    'VarifoldKernel',
    'TCURLoRA',
]
