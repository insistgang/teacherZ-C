import numpy as np
from benchmark import BaseBenchmark
from typing import Tuple, List
import warnings

warnings.filterwarnings("ignore")


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    pred_boundary = _extract_boundary(pred)
    gt_boundary = _extract_boundary(gt)

    if pred_boundary.sum() == 0 and gt_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return 0.0

    pred_dilated = _dilate(pred_boundary, tolerance)
    gt_dilated = _dilate(gt_boundary, tolerance)

    true_positives = np.logical_and(pred_boundary, gt_dilated).sum()
    false_positives = np.logical_and(pred_boundary, ~gt_dilated).sum()
    false_negatives = np.logical_and(gt_boundary, ~pred_dilated).sum()

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _extract_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(mask)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    return boundary


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=0)
        result = (
            padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[1:-1, 1:-1]
        )
    return result


class SLATSegmentation(BaseBenchmark):
    def __init__(self):
        super().__init__(name="SLaT", category="图像分割")
        self.image = None
        self.gt_mask = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        self.image = np.random.rand(*size)

        h, w = size
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        self.gt_mask = ((y - center_y) ** 2 + (x - center_x) ** 2 < radius**2).astype(
            float
        )

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        threshold = kwargs.get("threshold", 0.5)
        max_iter = kwargs.get("max_iter", 50)

        pred_mask = self._slat_segment(self.image, threshold, max_iter)

        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        gt_binary = (self.gt_mask > 0.5).astype(np.uint8)

        miou = compute_iou(pred_binary, gt_binary)
        return pred_mask, miou

    def _slat_segment(
        self, image: np.ndarray, threshold: float, max_iter: int
    ) -> np.ndarray:
        h, w = image.shape

        level_set = np.zeros((h, w))
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 3
        level_set = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2) - radius

        dt = 0.5
        nu = 0.5

        for _ in range(max_iter):
            gx, gy = self._gradient(level_set)
            norm = np.sqrt(gx**2 + gy**2 + 1e-10)

            nx = gx / norm
            ny = gy / norm

            curvature = self._curvature(level_set)

            mean_inside = np.mean(image[level_set < 0]) if np.any(level_set < 0) else 0
            mean_outside = (
                np.mean(image[level_set >= 0]) if np.any(level_set >= 0) else 0
            )

            force = (image - mean_inside) ** 2 - (image - mean_outside) ** 2

            level_set = level_set + dt * (nu * curvature * norm + force * norm)

        return (level_set < 0).astype(float)

    def _gradient(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gx = np.diff(img, axis=1, prepend=img[:, :1])
        gy = np.diff(img, axis=0, prepend=img[:1, :])
        return gx, gy

    def _curvature(self, phi: np.ndarray) -> np.ndarray:
        phi_x = np.diff(phi, axis=1, prepend=phi[:, :1])
        phi_y = np.diff(phi, axis=0, prepend=phi[:1, :])
        phi_xx = np.diff(phi_x, axis=1, prepend=phi_x[:, :1])
        phi_yy = np.diff(phi_y, axis=0, prepend=phi_y[:1, :])
        phi_xy = np.diff(phi_x, axis=0, prepend=phi_x[:1, :])

        norm = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)
        return (phi_xx * phi_y**2 - 2 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2) / (
            norm**3 + 1e-10
        )

    def cleanup(self):
        self.image = None
        self.gt_mask = None


class UNetBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(name="U-Net", category="图像分割")
        self.image = None
        self.gt_mask = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        self.image = np.random.rand(*size)

        h, w = size
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        self.gt_mask = ((y - center_y) ** 2 + (x - center_x) ** 2 < radius**2).astype(
            float
        )

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        depth = kwargs.get("depth", 3)

        pred_mask = self._simplified_unet(self.image, depth)

        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        gt_binary = (self.gt_mask > 0.5).astype(np.uint8)

        miou = compute_iou(pred_binary, gt_binary)
        return pred_mask, miou

    def _simplified_unet(self, image: np.ndarray, depth: int) -> np.ndarray:
        result = image.copy()

        for _ in range(depth):
            result = self._conv_block(result)

        return result

    def _conv_block(self, x: np.ndarray) -> np.ndarray:
        padded = np.pad(x, 1, mode="reflect")

        kernel = np.random.rand(3, 3)
        kernel = kernel / kernel.sum()

        output = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                output[i, j] = np.sum(padded[i : i + 3, j : j + 3] * kernel)

        return np.clip(output, 0, 1)

    def cleanup(self):
        self.image = None
        self.gt_mask = None


class DeepLabBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__(name="DeepLab", category="图像分割")
        self.image = None
        self.gt_mask = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        self.image = np.random.rand(*size)

        h, w = size
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        self.gt_mask = ((y - center_y) ** 2 + (x - center_x) ** 2 < radius**2).astype(
            float
        )

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        num_classes = kwargs.get("num_classes", 2)
        output_stride = kwargs.get("output_stride", 8)

        pred_mask = self._simplified_deeplab(self.image, output_stride)

        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        gt_binary = (self.gt_mask > 0.5).astype(np.uint8)

        miou = compute_iou(pred_binary, gt_binary)
        return pred_mask, miou

    def _simplified_deeplab(self, image: np.ndarray, output_stride: int) -> np.ndarray:
        h, w = image.shape
        small = image[::output_stride, ::output_stride]

        result = np.zeros_like(small)
        for i in range(small.shape[0]):
            for j in range(small.shape[1]):
                result[i, j] = small[i, j]

        upsampled = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                si = min(i // output_stride, small.shape[0] - 1)
                sj = min(j // output_stride, small.shape[1] - 1)
                upsampled[i, j] = result[si, sj]

        return upsampled

    def cleanup(self):
        self.image = None
        self.gt_mask = None


class GraphCutSegmentation(BaseBenchmark):
    def __init__(self):
        super().__init__(name="GraphCut", category="图像分割")
        self.image = None
        self.gt_mask = None

    def setup(self, size: tuple, **kwargs):
        np.random.seed(42)
        self.image = np.random.rand(*size)

        h, w = size
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        radius = min(h, w) // 4
        self.gt_mask = ((y - center_y) ** 2 + (x - center_x) ** 2 < radius**2).astype(
            float
        )

    def run(self, **kwargs) -> Tuple[np.ndarray, float]:
        sigma = kwargs.get("sigma", 0.1)
        max_iter = kwargs.get("max_iter", 20)

        pred_mask = self._graphcut(self.image, sigma, max_iter)

        pred_binary = (pred_mask > 0.5).astype(np.uint8)
        gt_binary = (self.gt_mask > 0.5).astype(np.uint8)

        miou = compute_iou(pred_binary, gt_binary)
        return pred_mask, miou

    def _graphcut(self, image: np.ndarray, sigma: float, max_iter: int) -> np.ndarray:
        h, w = image.shape

        labels = np.zeros((h, w), dtype=np.int32)
        center_y = h // 2
        for i in range(h):
            for j in range(w):
                if i > center_y:
                    labels[i, j] = 1

        for _ in range(max_iter):
            mean_fg = np.mean(image[labels == 0]) if np.any(labels == 0) else 0.5
            mean_bg = np.mean(image[labels == 1]) if np.any(labels == 1) else 0.5

            for i in range(h):
                for j in range(w):
                    d_fg = (image[i, j] - mean_fg) ** 2 / (2 * sigma**2)
                    d_bg = (image[i, j] - mean_bg) ** 2 / (2 * sigma**2)

                    smooth_cost = 0
                    if i > 0:
                        smooth_cost += (
                            abs(int(labels[i - 1, j]) - int(labels[i, j])) * 0.1
                        )
                    if j > 0:
                        smooth_cost += (
                            abs(int(labels[i, j - 1]) - int(labels[i, j])) * 0.1
                        )

                    if d_fg + smooth_cost < d_bg:
                        labels[i, j] = 0
                    else:
                        labels[i, j] = 1

        return labels.astype(float)

    def cleanup(self):
        self.image = None
        self.gt_mask = None
