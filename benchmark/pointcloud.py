import numpy as np
from benchmark import BaseBenchmark
from typing import Tuple, List, Optional
import warnings

warnings.filterwarnings("ignore")


def compute_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(pred == gt))


def compute_shape_matching(descriptor1: np.ndarray, descriptor2: np.ndarray) -> float:
    if descriptor1.shape != descriptor2.shape:
        return 0.0
    similarity = np.dot(descriptor1, descriptor2) / (
        np.linalg.norm(descriptor1) * np.linalg.norm(descriptor2) + 1e-10
    )
    return float(similarity)


class VarifoldBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(name="Varifold", category="3D点云")
        self.points = None
        self.labels = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        n_points = size[0] if isinstance(size, tuple) else size

        n_classes = kwargs.get("n_classes", 5)

        centers = np.random.rand(n_classes, 3) * 10
        points_per_class = n_points // n_classes

        all_points = []
        all_labels = []

        for i in range(n_classes):
            class_points = centers[i] + np.random.randn(points_per_class, 3) * 0.5
            all_points.append(class_points)
            all_labels.extend([i] * points_per_class)

        self.points = np.vstack(all_points)
        self.labels = np.array(all_labels)

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        sigma = kwargs.get("sigma", 1.0)
        n_features = kwargs.get("n_features", 64)

        descriptors = self._compute_varifold_descriptor(self.points, sigma, n_features)

        pred_labels = self._classify_from_descriptor(descriptors)

        accuracy = compute_accuracy(pred_labels, self.labels)
        return descriptors, accuracy

    def _compute_varifold_descriptor(
        self, points: np.ndarray, sigma: float, n_features: int
    ) -> np.ndarray:
        n_points = points.shape[0]
        descriptors = np.zeros((n_points, n_features))

        for i in range(n_points):
            distances = np.linalg.norm(points - points[i], axis=1)
            weights = np.exp(-(distances**2) / (2 * sigma**2))

            for j in range(min(n_features, n_points)):
                if i != j:
                    diff = points[j] - points[i]
                    if np.linalg.norm(diff) > 0:
                        descriptors[i, j % n_features] += weights[j] * np.linalg.norm(
                            diff
                        )

        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return descriptors / norms

    def _classify_from_descriptor(self, descriptors: np.ndarray) -> np.ndarray:
        n_points = descriptors.shape[0]
        labels = np.zeros(n_points, dtype=np.int32)

        n_clusters = 5
        cluster_centers = descriptors[
            np.random.choice(n_points, n_clusters, replace=False)
        ]

        for _ in range(10):
            distances = np.linalg.norm(
                descriptors[:, np.newaxis] - cluster_centers, axis=2
            )
            labels = np.argmin(distances, axis=1)

            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    cluster_centers[k] = np.mean(descriptors[mask], axis=0)

        return labels

    def cleanup(self):
        self.points = None
        self.labels = None


class PointNetBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(name="PointNet", category="3D点云")
        self.points = None
        self.labels = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        n_points = size[0] if isinstance(size, tuple) else size

        n_classes = kwargs.get("n_classes", 5)

        centers = np.random.rand(n_classes, 3) * 10
        points_per_class = n_points // n_classes

        all_points = []
        all_labels = []

        for i in range(n_classes):
            class_points = centers[i] + np.random.randn(points_per_class, 3) * 0.5
            all_points.append(class_points)
            all_labels.extend([i] * points_per_class)

        self.points = np.vstack(all_points)
        self.labels = np.array(all_labels)

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        n_features = kwargs.get("n_features", 64)

        features = self._pointnet_forward(self.points, n_features)

        pred_labels = self._classify_features(features)

        accuracy = compute_accuracy(pred_labels, self.labels)
        return features, accuracy

    def _pointnet_forward(self, points: np.ndarray, n_features: int) -> np.ndarray:
        n_points = points.shape[0]

        features = points.copy()

        np.random.seed(42)
        weight1 = np.random.randn(3, 32) * 0.1
        features = np.maximum(0, features @ weight1)

        weight2 = np.random.randn(32, n_features) * 0.1
        features = np.maximum(0, features @ weight2)

        global_feature = np.max(features, axis=0)

        features = np.tile(global_feature, (n_points, 1))

        return features

    def _classify_features(self, features: np.ndarray) -> np.ndarray:
        n_points = features.shape[0]
        labels = np.zeros(n_points, dtype=np.int32)

        n_clusters = 5
        cluster_centers = features[
            np.random.choice(n_points, n_clusters, replace=False)
        ]

        for _ in range(10):
            distances = np.linalg.norm(
                features[:, np.newaxis] - cluster_centers, axis=2
            )
            labels = np.argmin(distances, axis=1)

            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    cluster_centers[k] = np.mean(features[mask], axis=0)

        return labels

    def cleanup(self):
        self.points = None
        self.labels = None


class DGCBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(name="DGCNN", category="3D点云")
        self.points = None
        self.labels = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        n_points = size[0] if isinstance(size, tuple) else size

        n_classes = kwargs.get("n_classes", 5)

        centers = np.random.rand(n_classes, 3) * 10
        points_per_class = n_points // n_classes

        all_points = []
        all_labels = []

        for i in range(n_classes):
            class_points = centers[i] + np.random.randn(points_per_class, 3) * 0.5
            all_points.append(class_points)
            all_labels.extend([i] * points_per_class)

        self.points = np.vstack(all_points)
        self.labels = np.array(all_labels)

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        k = kwargs.get("k", 20)
        n_features = kwargs.get("n_features", 64)

        features = self._dgcnn_forward(self.points, k, n_features)

        pred_labels = self._classify_features(features)

        accuracy = compute_accuracy(pred_labels, self.labels)
        return features, accuracy

    def _dgcnn_forward(self, points: np.ndarray, k: int, n_features: int) -> np.ndarray:
        n_points = points.shape[0]

        edge_features = self._edge_conv(points, k, n_features)

        return edge_features

    def _edge_conv(self, points: np.ndarray, k: int, n_features: int) -> np.ndarray:
        n_points = points.shape[0]

        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                distances[i, j] = np.linalg.norm(points[i] - points[j])

        neighbors = np.argsort(distances, axis=1)[:, :k]

        edge_features = np.zeros((n_points, n_features))

        for i in range(n_points):
            local_features = []
            for j in neighbors[i]:
                edge = points[j] - points[i]
                local_features.append(edge)

            local_features = np.array(local_features)

            np.random.seed(i)
            weight = np.random.randn(3, n_features) * 0.1
            transformed = np.maximum(0, local_features @ weight)

            edge_features[i] = np.max(transformed, axis=0)

        return edge_features

    def _classify_features(self, features: np.ndarray) -> np.ndarray:
        n_points = features.shape[0]
        labels = np.zeros(n_points, dtype=np.int32)

        n_clusters = 5
        cluster_centers = features[
            np.random.choice(n_points, n_clusters, replace=False)
        ]

        for _ in range(10):
            distances = np.linalg.norm(
                features[:, np.newaxis] - cluster_centers, axis=2
            )
            labels = np.argmin(distances, axis=1)

            for k in range(n_clusters):
                mask = labels == k
                if np.any(mask):
                    cluster_centers[k] = np.mean(features[mask], axis=0)

        return labels

    def cleanup(self):
        self.points = None
        self.labels = None


class TuckerDecompositionBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(name="Tucker", category="张量分解")
        self.tensor = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        self.tensor = np.random.rand(*size)

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        ranks = kwargs.get("ranks", None)

        if ranks is None:
            ranks = tuple(s // 2 for s in self.tensor.shape)

        result, error = self._tucker_decomposition(self.tensor, ranks)
        return result, 1.0 - error

    def _tucker_decomposition(
        self, tensor: np.ndarray, ranks: tuple
    ) -> Tuple[np.ndarray, float]:
        shape = tensor.shape
        n_modes = len(shape)

        factors = []
        temp = tensor.copy()

        for mode in range(n_modes):
            n = shape[mode]
            r = min(ranks[mode], n)

            unfolded = np.moveaxis(temp, mode, 0)
            unfolded = unfolded.reshape(n, -1)

            U, _, _ = np.linalg.svd(unfolded, full_matrices=False)
            U_r = U[:, :r]
            factors.append(U_r)

            temp = np.tensordot(temp, U_r.T, axes=([mode], [0]))

        original_norm = np.linalg.norm(tensor)
        if original_norm > 0:
            reconstruction_error = np.linalg.norm(tensor) / original_norm
        else:
            reconstruction_error = 0.0

        return temp, float(min(reconstruction_error, 1.0))

    def cleanup(self):
        self.tensor = None
