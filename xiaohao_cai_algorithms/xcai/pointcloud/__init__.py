"""
Point cloud analysis module containing Varifold-based methods.
"""

from xcai.pointcloud.varifold import VarifoldKernel
from xcai.pointcloud.utils import compute_chamfer_distance, compute_hausdorff_distance

__all__ = ['VarifoldKernel', 'compute_chamfer_distance', 'compute_hausdorff_distance']
