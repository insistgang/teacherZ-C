"""
ROF迭代阈值分割算法实现
A Multiphase Image Segmentation Based on Approximate ROF Models

论文: IEEE Transactions on Image Processing, 2013
作者: Xiaohao Cai, et al.

核心思想:
    - 将多类分割分解为一系列二值分割
    - 使用标签树组织层次分割
    - 每步求解标准ROF模型
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from scipy.ndimage import gaussian_filter


class ROFDenoiser:
    """ROF模型求解器"""

    def __init__(self, max_iter: int = 100, tol: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, f: np.ndarray, lambda_param: float) -> np.ndarray:
        """
        求解ROF模型: min_u {lambda/2 * ||f - u||^2 + TV(u)}

        使用Chambolle投影算法

        参数:
            f: 输入图像
            lambda_param: 数据项权重

        返回:
            u: 去噪后的图像
        """
        f = f.astype(np.float64)
        H, W = f.shape

        p = np.zeros((H, W, 2), dtype=np.float64)
        tau = 0.25

        for _ in range(self.max_iter):
            div_p = self._divergence(p)
            u = f - lambda_param * div_p

            grad_u = self._gradient(u)

            p_new = p + tau * grad_u
            p_norm = np.sqrt(p_new[:, :, 0] ** 2 + p_new[:, :, 1] ** 2 + 1e-10)
            p[:, :, 0] = p_new[:, :, 0] / np.maximum(p_norm, 1.0)
            p[:, :, 1] = p_new[:, :, 1] / np.maximum(p_norm, 1.0)

        return f - lambda_param * self._divergence(p)

    def _gradient(self, u: np.ndarray) -> np.ndarray:
        H, W = u.shape
        grad = np.zeros((H, W, 2), dtype=np.float64)
        grad[:-1, :, 0] = u[1:, :] - u[:-1, :]
        grad[:, :-1, 1] = u[:, 1:] - u[:, :-1]
        return grad

    def _divergence(self, p: np.ndarray) -> np.ndarray:
        H, W = p.shape[:2]
        div = np.zeros((H, W), dtype=np.float64)
        div[1:, :] += p[1:, :, 0] - p[:-1, :, 0]
        div[0, :] += p[0, :, 0]
        div[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]
        div[:, 0] += p[:, 0, 1]
        return div


class IterativeROFSegmentation:
    """迭代ROF多类分割"""

    def __init__(
        self,
        lambda_param: float = 0.1,
        max_iter: int = 100,
        tree_type: str = "balanced",
    ):
        """
        初始化

        参数:
            lambda_param: ROF正则化参数
            max_iter: ROF迭代次数
            tree_type: 标签树类型 ('balanced' 或 'sequential')
        """
        self.lambda_param = lambda_param
        self.max_iter = max_iter
        self.tree_type = tree_type
        self.rof_solver = ROFDenoiser(max_iter=max_iter)

    def segment(
        self, image: np.ndarray, K: int, data_terms: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        执行迭代ROF多类分割

        参数:
            image: 输入图像 (H, W) 或 (H, W, 3)
            K: 类别数
            data_terms: 可选的数据项 (H, W, K)，若None则自动计算

        返回:
            segmentation: K类分割标签图 (H, W)
        """
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.copy()

        H, W = gray.shape

        if data_terms is None:
            data_terms = self._compute_data_terms(gray, K)

        tree = self._build_label_tree(K)

        segmentation = self._iterative_segment(data_terms, tree, K)

        return segmentation

    def _compute_data_terms(self, gray: np.ndarray, K: int) -> np.ndarray:
        """
        计算数据项 f_k(x) = (I(x) - c_k)^2

        使用K-means初始化聚类中心
        """
        H, W = gray.shape

        centroids = np.linspace(gray.min(), gray.max(), K + 2)[1:-1]

        data_terms = np.zeros((H, W, K), dtype=np.float64)
        for k in range(K):
            data_terms[:, :, k] = (gray - centroids[k]) ** 2

        return data_terms

    def _build_label_tree(self, K: int) -> List[Tuple[set, set]]:
        """
        构建标签树

        返回每层的分割结构 [(set_A, set_B), ...]
        """
        if self.tree_type == "balanced":
            return self._build_balanced_tree(K)
        else:
            return self._build_sequential_tree(K)

    def _build_balanced_tree(self, K: int) -> List[Tuple[set, set]]:
        """构建平衡二叉树"""
        tree = []
        current_sets = [frozenset({frozenset(range(K))})]
        level = 0

        while max(len(s) for group in current_sets for s in group) > 1:
            tree_level = []
            next_groups = []

            for group in current_sets:
                for class_set in group:
                    if len(class_set) == 1:
                        next_groups.append(frozenset({class_set}))
                    else:
                        classes = sorted(class_set)
                        mid = len(classes) // 2
                        set_A = frozenset(classes[:mid])
                        set_B = frozenset(classes[mid:])
                        tree_level.append((set(set_A), set(set_B)))
                        next_groups.append(frozenset({set_A, set_B}))

            if tree_level:
                tree.append(tree_level)
            current_sets = next_groups
            level += 1

            if level > 20:
                break

        return tree

    def _build_sequential_tree(self, K: int) -> List[Tuple[set, set]]:
        """构建顺序树: {1} vs {2,3,...}, {2} vs {3,4,...}, ..."""
        tree = []
        for k in range(K - 1):
            set_A = {k}
            set_B = set(range(k + 1, K))
            tree.append([(set_A, set_B)])
        return tree

    def _iterative_segment(
        self, data_terms: np.ndarray, tree: List[Tuple[set, set]], K: int
    ) -> np.ndarray:
        """
        执行迭代分割

        对每个二值分割问题，求解:
        min_u {∫ u*(f_A - f_B) + λ*TV(u) }
        """
        H, W = data_terms.shape[:2]
        labels = np.zeros((H, W), dtype=np.int32)
        current_masks = [np.ones((H, W), dtype=bool)]
        current_sets = [set(range(K))]

        for level_idx, level in enumerate(tree):
            next_masks = []
            next_sets = []

            for (set_A, set_B), mask, class_set in zip(
                level, current_masks, current_sets
            ):
                f_A = np.sum(data_terms[:, :, list(set_A)], axis=2)
                f_B = np.sum(data_terms[:, :, list(set_B)], axis=2)

                f_diff = f_A - f_B

                f_diff_masked = np.where(mask, f_diff, 0)
                u = self.rof_solver.solve(f_diff_masked, self.lambda_param)

                binary_seg = u > 0.5 * np.mean(u[mask])

                mask_A = mask & binary_seg
                mask_B = mask & ~binary_seg

                next_masks.extend([mask_A, mask_B])

                if len(set_A) == 1:
                    labels[mask_A] = list(set_A)[0]
                    next_sets.append(set_A)
                else:
                    next_sets.append(set_A)

                if len(set_B) == 1:
                    labels[mask_B] = list(set_B)[0]
                    next_sets.append(set_B)
                else:
                    next_sets.append(set_B)

            current_masks = []
            current_sets = []
            for m, s in zip(next_masks, next_sets):
                if np.any(m) and len(s) > 0:
                    current_masks.append(m)
                    current_sets.append(s)

        return labels


class AutomaticThresholdROF:
    """带自动阈值选择的ROF分割"""

    def __init__(self, lambda_param: float = 0.1):
        self.lambda_param = lambda_param
        self.rof_solver = ROFDenoiser()

    def segment_with_auto_threshold(
        self, image: np.ndarray, K: int = 2
    ) -> Tuple[np.ndarray, List[float]]:
        """
        自动选择最优阈值的分割

        使用Otsu类方法在ROF结果上选择阈值

        参数:
            image: 输入灰度图像
            K: 类别数

        返回:
            segmentation: 分割结果
            thresholds: 选择的阈值列表
        """
        denoised = self.rof_solver.solve(image, self.lambda_param)

        thresholds = self._find_optimal_thresholds(denoised, K)

        segmentation = self._apply_thresholds(denoised, thresholds)

        return segmentation, thresholds

    def _find_optimal_thresholds(self, image: np.ndarray, K: int) -> List[float]:
        """
        使用Otsu方法找到最优阈值
        """
        hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 1))
        hist = hist.astype(np.float64)
        hist /= hist.sum()

        thresholds = []
        current_min, current_max = 0, 255

        for _ in range(K - 1):
            best_t = self._otsu_threshold(hist[current_min : current_max + 1])
            thresholds.append(
                (bins[current_min + best_t] + bins[current_min + best_t + 1]) / 2
            )

            current_min = current_min + best_t + 1

        return sorted(thresholds)

    def _otsu_threshold(self, hist: np.ndarray) -> int:
        """Otsu阈值选择"""
        total = hist.sum()
        sum_total = np.sum(np.arange(len(hist)) * hist)

        sum_bg = 0
        w_bg = 0
        max_var = 0
        best_t = 0

        for t in range(len(hist)):
            w_bg += hist[t]
            if w_bg == 0:
                continue

            w_fg = total - w_bg
            if w_fg == 0:
                break

            sum_bg += t * hist[t]
            mean_bg = sum_bg / w_bg
            mean_fg = (sum_total - sum_bg) / w_fg

            var = w_bg * w_fg * (mean_bg - mean_fg) ** 2

            if var > max_var:
                max_var = var
                best_t = t

        return best_t

    def _apply_thresholds(
        self, image: np.ndarray, thresholds: List[float]
    ) -> np.ndarray:
        """应用多阈值"""
        segmentation = np.zeros(image.shape, dtype=np.int32)

        for i, t in enumerate(sorted(thresholds)):
            segmentation[image > t] = i + 1

        return segmentation


def demo_rof_segmentation():
    """ROF迭代分割演示"""
    print("=" * 60)
    print("ROF迭代阈值分割演示")
    print("=" * 60)

    np.random.seed(42)
    H, W = 100, 100

    synthetic = np.zeros((H, W), dtype=np.float64)
    synthetic[: H // 2, : W // 2] = 0.2
    synthetic[: H // 2, W // 2 :] = 0.5
    synthetic[H // 2 :, : W // 2] = 0.7
    synthetic[H // 2 :, W // 2 :] = 0.9

    noise = np.random.randn(H, W) * 0.1
    noisy_image = np.clip(synthetic + noise, 0, 1)

    print("\n--- 方法1: 迭代ROF多类分割 (K=4) ---")
    segmenter = IterativeROFSegmentation(lambda_param=0.2, tree_type="balanced")
    seg_result = segmenter.segment(noisy_image, K=4)

    print(f"输入图像: {noisy_image.shape}")
    print(f"分割结果: {seg_result.shape}")
    print(f"唯一标签: {np.unique(seg_result)}")

    print("\n各类别像素数:")
    for k in range(4):
        count = np.sum(seg_result == k)
        print(f"  类别 {k}: {count} 像素")

    print("\n--- 方法2: 自动阈值ROF分割 ---")
    auto_segmenter = AutomaticThresholdROF(lambda_param=0.15)
    auto_seg, thresholds = auto_segmenter.segment_with_auto_threshold(noisy_image, K=4)

    print(f"自动选择的阈值: {thresholds}")
    print(f"分割结果: {auto_seg.shape}")

    print("\n--- 方法3: 不同树结构对比 ---")

    seg_balanced = IterativeROFSegmentation(
        lambda_param=0.2, tree_type="balanced"
    ).segment(noisy_image, K=4)
    seg_sequential = IterativeROFSegmentation(
        lambda_param=0.2, tree_type="sequential"
    ).segment(noisy_image, K=4)

    print(f"平衡树分割 - 各类像素数: {[np.sum(seg_balanced == k) for k in range(4)]}")
    print(f"顺序树分割 - 各类像素数: {[np.sum(seg_sequential == k) for k in range(4)]}")

    return seg_result


if __name__ == "__main__":
    demo_rof_segmentation()
