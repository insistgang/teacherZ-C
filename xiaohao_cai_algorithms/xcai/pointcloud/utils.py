"""
Utility functions for point cloud analysis.
"""

import numpy as np
from typing import Optional
from scipy.spatial.distance import cdist


def compute_chamfer_distance(
    points1: np.ndarray,
    points2: np.ndarray,
    reduction: str = 'mean'
) -> float:
    """
    Compute Chamfer distance between two point clouds.
    
    CD(P, Q) = mean(min||p - q||²) + mean(min||q - p||²)
    
    Parameters
    ----------
    points1 : np.ndarray
        First point cloud, shape (N, D).
    points2 : np.ndarray
        Second point cloud, shape (M, D).
    reduction : str
        'mean' or 'sum' for the final reduction.
        
    Returns
    -------
    float
        Chamfer distance.
    """
    D = cdist(points1, points2, metric='sqeuclidean')
    
    min1 = D.min(axis=1)
    min2 = D.min(axis=0)
    
    if reduction == 'mean':
        return float(min1.mean() + min2.mean())
    else:
        return float(min1.sum() + min2.sum())


def compute_hausdorff_distance(
    points1: np.ndarray,
    points2: np.ndarray
) -> float:
    """
    Compute Hausdorff distance between two point clouds.
    
    H(P, Q) = max(max(min||p - q||), max(min||q - p||))
    
    Parameters
    ----------
    points1 : np.ndarray
        First point cloud, shape (N, D).
    points2 : np.ndarray
        Second point cloud, shape (M, D).
        
    Returns
    -------
    float
        Hausdorff distance.
    """
    D = cdist(points1, points2, metric='euclidean')
    
    max1 = D.min(axis=1).max()
    max2 = D.min(axis=0).max()
    
    return float(max(max1, max2))


def compute_earth_mover_distance(
    points1: np.ndarray,
    points2: np.ndarray
) -> float:
    """
    Compute Earth Mover's Distance (Wasserstein-1).
    
    Note: Requires scipy >= 1.0 with linear_sum_assignment.
    
    Parameters
    ----------
    points1 : np.ndarray
        First point cloud, shape (N, D).
    points2 : np.ndarray
        Second point cloud, shape (M, D).
        
    Returns
    -------
    float
        EMD value.
    """
    from scipy.optimize import linear_sum_assignment
    
    D = cdist(points1, points2, metric='euclidean')
    
    row_ind, col_ind = linear_sum_assignment(D)
    
    return float(D[row_ind, col_ind].sum() / len(points1))


def estimate_normals(
    points: np.ndarray,
    k_neighbors: int = 10
) -> np.ndarray:
    """
    Estimate normals from point cloud using PCA.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud, shape (N, 3).
    k_neighbors : int
        Number of neighbors for local PCA.
        
    Returns
    -------
    np.ndarray
        Estimated normals, shape (N, 3).
    """
    from scipy.spatial import cKDTree
    
    tree = cKDTree(points)
    normals = np.zeros_like(points)
    
    for i in range(len(points)):
        distances, indices = tree.query(points[i], k=k_neighbors + 1)
        neighbors = points[indices[1:]]
        
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]
        
        normals[i] = normal / (np.linalg.norm(normal) + 1e-10)
    
    return normals


def orient_normals_consistently(
    points: np.ndarray,
    normals: np.ndarray,
    k_neighbors: int = 10
) -> np.ndarray:
    """
    Orient normals consistently using minimum spanning tree.
    
    Parameters
    ----------
    points : np.ndarray
        Point cloud, shape (N, 3).
    normals : np.ndarray
        Input normals, shape (N, 3).
    k_neighbors : int
        Number of neighbors for propagation.
        
    Returns
    -------
    np.ndarray
        Consistently oriented normals.
    """
    from scipy.spatial import cKDTree
    
    tree = cKDTree(points)
    oriented = normals.copy()
    visited = np.zeros(len(points), dtype=bool)
    
    queue = [0]
    visited[0] = True
    
    while queue:
        i = queue.pop(0)
        
        _, indices = tree.query(points[i], k=k_neighbors + 1)
        
        for j in indices[1:]:
            if not visited[j]:
                if np.dot(oriented[i], oriented[j]) < 0:
                    oriented[j] = -oriented[j]
                visited[j] = True
                queue.append(j)
    
    return oriented


def compute_point_cloud_center(points: np.ndarray) -> np.ndarray:
    """Compute center of point cloud."""
    return points.mean(axis=0)


def normalize_point_cloud(
    points: np.ndarray,
    center: Optional[np.ndarray] = None,
    scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize point cloud to unit sphere.
    
    Parameters
    ----------
    points : np.ndarray
        Input points.
    center : np.ndarray, optional
        Custom center.
    scale : float, optional
        Custom scale.
        
    Returns
    -------
    normalized : np.ndarray
        Normalized points.
    center : np.ndarray
        Used center.
    scale : float
        Used scale.
    """
    if center is None:
        center = points.mean(axis=0)
    
    centered = points - center
    
    if scale is None:
        scale = np.max(np.linalg.norm(centered, axis=1))
    
    normalized = centered / scale
    
    return normalized, center, scale


from typing import Tuple
