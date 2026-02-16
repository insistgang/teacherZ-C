"""
GraphCut Segmentation Implementation

Based on:
- Boykov & Kolmogorov (2004): An experimental comparison of min-cut/max-flow algorithms
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy import sparse


class GraphCutSegmenter:
    """
    GraphCut-based image segmentation.
    
    Parameters
    ----------
    sigma : float
        Parameter for boundary term (smoothness).
    lambda_param : float
        Balance between regional and boundary terms.
    connectivity : int
        Neighborhood connectivity (4 or 8).
    """
    
    def __init__(
        self,
        sigma: float = 10.0,
        lambda_param: float = 1.0,
        connectivity: int = 8
    ):
        self.sigma = sigma
        self.lambda_param = lambda_param
        self.connectivity = connectivity
    
    def segment(
        self,
        image: np.ndarray,
        foreground_seed: Optional[np.ndarray] = None,
        background_seed: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Segment an image using GraphCut.
        
        Parameters
        ----------
        image : np.ndarray
            Input image.
        foreground_seed : np.ndarray, optional
            Binary mask marking foreground pixels.
        background_seed : np.ndarray, optional
            Binary mask marking background pixels.
            
        Returns
        -------
        np.ndarray
            Binary segmentation (0: background, 1: foreground).
        """
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)
        
        h, w = gray.shape
        n_pixels = h * w
        
        if foreground_seed is None:
            foreground_seed = np.zeros((h, w), dtype=bool)
            foreground_seed[:h//4, :w//4] = True
        
        if background_seed is None:
            background_seed = np.zeros((h, w), dtype=bool)
            background_seed[3*h//4:, 3*w//4:] = True
        
        fg_hist, bg_hist = self._compute_histograms(gray, foreground_seed, background_seed)
        
        capacities = self._build_graph(gray, fg_hist, bg_hist, foreground_seed, background_seed)
        
        labels = self._max_flow_min_cut(n_pixels, capacities)
        
        return labels.reshape(h, w)
    
    def _compute_histograms(
        self,
        gray: np.ndarray,
        fg_seed: np.ndarray,
        bg_seed: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute color histograms for foreground and background."""
        n_bins = 256
        
        fg_pixels = gray[fg_seed].flatten()
        bg_pixels = gray[bg_seed].flatten()
        
        fg_hist = np.zeros(n_bins)
        bg_hist = np.zeros(n_bins)
        
        if len(fg_pixels) > 0:
            fg_hist, _ = np.histogram(fg_pixels, bins=n_bins, range=(0, 256))
            fg_hist = fg_hist.astype(float) / fg_hist.sum()
        
        if len(bg_pixels) > 0:
            bg_hist, _ = np.histogram(bg_pixels, bins=n_bins, range=(0, 256))
            bg_hist = bg_hist.astype(float) / bg_hist.sum()
        
        fg_hist = np.clip(fg_hist, 1e-10, None)
        bg_hist = np.clip(bg_hist, 1e-10, None)
        
        return fg_hist, bg_hist
    
    def _build_graph(
        self,
        gray: np.ndarray,
        fg_hist: np.ndarray,
        bg_hist: np.ndarray,
        fg_seed: np.ndarray,
        bg_seed: np.ndarray
    ) -> Dict:
        """Build graph with n-links (neighbor) and t-links (terminal)."""
        h, w = gray.shape
        n_pixels = h * w
        
        edges = []
        
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                intensity = int(np.clip(gray[i, j], 0, 255))
                
                if fg_seed[i, j]:
                    t_link_fg = float('inf')
                    t_link_bg = 0
                elif bg_seed[i, j]:
                    t_link_fg = 0
                    t_link_bg = float('inf')
                else:
                    t_link_fg = -np.log(bg_hist[intensity])
                    t_link_bg = -np.log(fg_hist[intensity])
                
                edges.append(('source', idx, t_link_fg))
                edges.append((idx, 'sink', t_link_bg))
                
                neighbors = self._get_neighbors(i, j, h, w)
                for ni, nj in neighbors:
                    n_idx = ni * w + nj
                    diff = np.exp(-((gray[i, j] - gray[ni, nj]) ** 2) / (2 * self.sigma ** 2))
                    capacity = self.lambda_param * diff
                    edges.append((idx, n_idx, capacity))
        
        return {'edges': edges, 'n_nodes': n_pixels + 2}
    
    def _get_neighbors(self, i: int, j: int, h: int, w: int) -> list:
        """Get neighboring pixel coordinates."""
        if self.connectivity == 4:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        neighbors = []
        for di, dj in offsets:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                neighbors.append((ni, nj))
        
        return neighbors
    
    def _max_flow_min_cut(self, n_nodes: int, capacities: Dict) -> np.ndarray:
        """
        Simplified max-flow min-cut implementation.
        
        For production use, consider using library like PyMaxflow.
        """
        labels = np.zeros(n_nodes - 2, dtype=int)
        
        edges = capacities['edges']
        
        graph = {}
        for edge in edges:
            src, dst, cap = edge
            if src not in graph:
                graph[src] = {}
            if dst not in graph:
                graph[dst] = {}
            graph[src][dst] = cap
            if dst not in graph[src]:
                graph[src][dst] = 0
            if src not in graph[dst]:
                graph[dst][src] = 0
        
        flow = self._ford_fulkerson(graph, 'source', 'sink')
        
        visited = set()
        self._dfs(graph, 'source', visited)
        
        for i in range(n_nodes - 2):
            if i in visited:
                labels[i] = 1
        
        return labels
    
    def _ford_fulkerson(self, graph: dict, source: str, sink: str) -> float:
        """Ford-Fulkerson algorithm for max flow."""
        max_flow = 0
        path = self._find_path(graph, source, sink, set())
        
        iterations = 0
        max_iterations = 1000
        
        while path and iterations < max_iterations:
            path_flow = float('inf')
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                path_flow = min(path_flow, graph[u].get(v, 0))
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                graph[u][v] -= path_flow
                graph[v][u] = graph[v].get(u, 0) + path_flow
            
            max_flow += path_flow
            path = self._find_path(graph, source, sink, set())
            iterations += 1
        
        return max_flow
    
    def _find_path(self, graph: dict, source: str, sink: str, visited: set) -> list:
        """Find augmenting path using DFS."""
        if source == sink:
            return [source]
        
        visited.add(source)
        
        for neighbor, capacity in graph.get(source, {}).items():
            if capacity > 0 and neighbor not in visited:
                path = self._find_path(graph, neighbor, sink, visited)
                if path:
                    return [source] + path
        
        return None
    
    def _dfs(self, graph: dict, node: str, visited: set):
        """Depth-first search to find reachable nodes."""
        visited.add(node)
        for neighbor, capacity in graph.get(node, {}).items():
            if capacity > 0 and neighbor not in visited:
                self._dfs(graph, neighbor, visited)
