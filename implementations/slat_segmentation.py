"""
SLaT 三阶段分割算法实现
A Three-Stage Approach for Segmenting Degraded Color Images: Smoothing, Lifting and Thresholding

论文: Journal of Scientific Computing (2017) 72:1313–1332
作者: Xiaohao Cai, Raymond Chan, Mila Nikolova, Tieyong Zeng

三阶段:
    Stage 1: ROF变分平滑 (TV-L2去噪)
    Stage 2: RGB → Lab色彩空间转换 + 维度提升
    Stage 3: K-means阈值分割
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from sklearn.cluster import KMeans


class SLATSegmentation:
    """SLaT三阶段分割算法"""

    def __init__(
        self,
        lambda_param: float = 1.0,
        mu: float = 1.0,
        rof_iterations: int = 100,
        rof_tol: float = 1e-4,
    ):
        """
        初始化SLaT分割器

        参数:
            lambda_param: 数据保真项权重 (默认1.0)
            mu: H1正则化权重，固定为1 (论文建议)
            rof_iterations: ROF迭代最大次数
            rof_tol: ROF收敛阈值
        """
        self.lambda_param = lambda_param
        self.mu = mu
        self.rof_iterations = rof_iterations
        self.rof_tol = rof_tol

    def segment(
        self, image: np.ndarray, K: int, return_intermediate: bool = False
    ) -> dict:
        """
        执行SLaT三阶段分割

        参数:
            image: 输入RGB图像，值域[0,1]或[0,255]，形状(H,W,3)
            K: 分割类别数
            return_intermediate: 是否返回中间结果

        返回:
            dict包含:
                - segmentation: 分割标签图 (H, W)
                - cluster_centers: 聚类中心 (K, 6)
                - (可选) smoothed: Stage1平滑结果
                - (可选) lifted: Stage2升维结果
        """
        image = self._normalize_image(image)

        smoothed = self._stage1_smoothing(image)
        lifted = self._stage2_lifting(smoothed)
        segmentation, centers = self._stage3_thresholding(lifted, K)

        result = {"segmentation": segmentation, "cluster_centers": centers}

        if return_intermediate:
            result["smoothed"] = smoothed
            result["lifted"] = lifted

        return result

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """归一化图像到[0,1]"""
        if image.max() > 1.0:
            image = image.astype(np.float64) / 255.0
        return np.clip(image, 0, 1)

    def _stage1_smoothing(self, f_rgb: np.ndarray) -> np.ndarray:
        """
        Stage 1: ROF变分平滑

        对每个RGB通道求解:
        min_g {lambda/2 * ||f - g||^2 + mu/2 * ||∇g||^2 + ||∇g||_TV}

        使用Chambolle投影算法求解ROF模型
        """
        H, W, C = f_rgb.shape
        g_bar = np.zeros_like(f_rgb)

        for c in range(C):
            g_bar[:, :, c] = self._rof_denoise(f_rgb[:, :, c], self.lambda_param)

        return np.clip(g_bar, 0, 1)

    def _rof_denoise(self, f: np.ndarray, lambda_param: float) -> np.ndarray:
        """
        ROF模型去噪 (Chambolle投影算法)

        求解: min_u {lambda/2 * ||f - u||^2 + TV(u)}
        其中 TV(u) = ||∇u||

        Chambolle对偶公式:
        u = f - lambda * div(p)
        p^{n+1} = p^n + tau * (∇(f - lambda * div(p^n)) / (1 + tau * |∇(f - lambda * div(p^n))|))
        """
        f = f.astype(np.float64)
        H, W = f.shape

        p = np.zeros((H, W, 2), dtype=np.float64)
        tau = 0.25
        u = f.copy()

        for _ in range(self.rof_iterations):
            div_p = self._divergence(p)
            u = f - lambda_param * div_p

            grad_u = self._gradient(u)
            grad_u_norm = np.sqrt(grad_u[:, :, 0] ** 2 + grad_u[:, :, 1] ** 2 + 1e-10)

            p_new = p + tau * grad_u
            p_norm = np.sqrt(p_new[:, :, 0] ** 2 + p_new[:, :, 1] ** 2 + 1e-10)
            p[:, :, 0] = p_new[:, :, 0] / np.maximum(p_norm, 1.0)
            p[:, :, 1] = p_new[:, :, 1] / np.maximum(p_norm, 1.0)

        return u

    def _gradient(self, u: np.ndarray) -> np.ndarray:
        """计算梯度 ∇u = (∂u/∂x, ∂u/∂y)"""
        H, W = u.shape
        grad = np.zeros((H, W, 2), dtype=np.float64)
        grad[:-1, :, 0] = u[1:, :] - u[:-1, :]
        grad[:, :-1, 1] = u[:, 1:] - u[:, :-1]
        return grad

    def _divergence(self, p: np.ndarray) -> np.ndarray:
        """计算散度 div(p) = ∂p_x/∂x + ∂p_y/∂y"""
        H, W = p.shape[:2]
        div = np.zeros((H, W), dtype=np.float64)
        div[1:, :] += p[1:, :, 0] - p[:-1, :, 0]
        div[0, :] += p[0, :, 0]
        div[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]
        div[:, 0] += p[:, 0, 1]
        return div

    def _stage2_lifting(self, g_rgb: np.ndarray) -> np.ndarray:
        """
        Stage 2: 维度提升 RGB → RGB+Lab (6通道)

        关键创新: Lab空间提供感知均匀的色彩表示，
        a/b通道捕获与RGB互补的色彩信息
        """
        g_rgb = np.clip(g_rgb, 0, 1)
        g_rgb_uint8 = (g_rgb * 255).astype(np.uint8)

        g_lab = cv2.cvtColor(g_rgb_uint8, cv2.COLOR_RGB2Lab)

        g_lab_norm = np.zeros_like(g_lab, dtype=np.float64)
        g_lab_norm[:, :, 0] = g_lab[:, :, 0] / 100.0
        g_lab_norm[:, :, 1] = (g_lab[:, :, 1] + 127) / 254.0
        g_lab_norm[:, :, 2] = (g_lab[:, :, 2] + 127) / 254.0

        g_star = np.concatenate([g_rgb, g_lab_norm], axis=2)

        return g_star

    def _stage3_thresholding(
        self, g_star: np.ndarray, K: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 3: K-means阈值分割

        在6维特征空间中进行聚类，优势:
        1. 调整K无需重算Stage 1/2
        2. 6维特征比3维RGB提供更好的可分性
        """
        H, W, C = g_star.shape
        pixels = g_star.reshape(-1, C)

        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(pixels)

        segmentation = labels.reshape(H, W)
        centers = kmeans.cluster_centers_

        return segmentation, centers


def slat_segment(image: np.ndarray, K: int, lambda_param: float = 1.0) -> np.ndarray:
    """
    SLaT三阶段分割的便捷函数

    参数:
        image: 输入RGB图像 (H, W, 3)
        K: 分割类别数
        lambda_param: 数据项权重，值越大越保真

    返回:
        segmentation: 分割标签图 (H, W)
    """
    segmenter = SLATSegmentation(lambda_param=lambda_param)
    result = segmenter.segment(image, K)
    return result["segmentation"]


def demo_slat():
    """SLaT分割演示"""
    print("=" * 60)
    print("SLaT三阶段分割演示")
    print("=" * 60)

    np.random.seed(42)
    H, W = 128, 128

    synthetic = np.zeros((H, W, 3), dtype=np.float64)
    synthetic[: H // 2, : W // 2, 0] = 0.8
    synthetic[: H // 2, W // 2 :, 1] = 0.7
    synthetic[H // 2 :, : W // 2, 2] = 0.9
    synthetic[H // 2 :, W // 2 :, :] = [0.3, 0.5, 0.4]

    noise = np.random.randn(H, W, 3) * 0.1
    noisy_image = np.clip(synthetic + noise, 0, 1)

    segmenter = SLATSegmentation(lambda_param=2.0)
    result = segmenter.segment(noisy_image, K=4, return_intermediate=True)

    print(f"\n输入图像: {noisy_image.shape}")
    print(f"分割类别数: 4")
    print(f"平滑后形状: {result['smoothed'].shape}")
    print(f"升维后形状: {result['lifted'].shape} (6通道)")
    print(f"分割结果: {result['segmentation'].shape}")
    print(f"聚类中心: {result['cluster_centers'].shape}")

    print("\n各类别像素数:")
    for k in range(4):
        count = np.sum(result["segmentation"] == k)
        print(f"  类别 {k}: {count} 像素")

    return result


if __name__ == "__main__":
    demo_slat()
