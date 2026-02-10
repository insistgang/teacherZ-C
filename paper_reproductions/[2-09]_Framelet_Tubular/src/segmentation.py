"""
管状结构分割算法

本模块实现了基于变分框架的管状结构分割，包括:
- 小波框架正则化
- 形状先验能量
- Split Bregman优化

能量泛函:
    E(u) = E_data(u) + λ₁ E_framelet(u) + λ₂ E_shape(u)

其中:
    E_data: 数据保真项
    E_framelet: 小波框架正则化
    E_shape: 管状结构形状先验
"""

import numpy as np
from typing import Optional, Tuple, Dict
from scipy import ndimage
from scipy.optimize import minimize
import warnings

from .framelet import FrameletTransform
from .shape_prior import TubularShapePrior


class TubularSegmentation:
    """
    管状结构分割器
    
    使用变分方法结合小波框架和形状先验进行管状结构分割。
    
    属性:
        lambda_framelet: 小波框架正则化权重
        lambda_shape: 形状先验权重
        framelet: 小波框架变换实例
        shape_prior: 形状先验实例
    """
    
    def __init__(
        self,
        lambda_framelet: float = 0.1,
        lambda_shape: float = 0.05,
        lambda_data: float = 1.0,
        framelet_level: int = 3,
        framelet_filter: str = 'haar',
        max_iter: int = 100,
        tol: float = 1e-4
    ):
        """
        初始化管状结构分割器
        
        参数:
            lambda_framelet: 小波框架正则化权重
            lambda_shape: 形状先验权重
            lambda_data: 数据保真项权重
            framelet_level: 小波框架分解层数
            framelet_filter: 小波滤波器名称
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        示例:
            >>> segmenter = TubularSegmentation(
            ...     lambda_framelet=0.1,
            ...     lambda_shape=0.05
            ... )
        """
        self.lambda_framelet = lambda_framelet
        self.lambda_shape = lambda_shape
        self.lambda_data = lambda_data
        self.max_iter = max_iter
        self.tol = tol
        
        # 初始化小波框架变换
        self.framelet = FrameletTransform(
            level=framelet_level,
            filter_name=framelet_filter
        )
        
        # 初始化形状先验
        self.shape_prior = TubularShapePrior()
        
        # 收敛历史
        self.history = {
            'energy': [],
            'data_term': [],
            'framelet_term': [],
            'shape_term': []
        }
    
    def segment(
        self,
        image: np.ndarray,
        initial_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        执行管状结构分割
        
        参数:
            image: 输入图像 (H, W) 或 (H, W, C)
            initial_mask: 初始分割掩码 (可选)
            
        返回:
            (segmentation, info) 元组
            - segmentation: 分割结果 (二值图)
            - info: 包含收敛信息的字典
            
        示例:
            >>> segmenter = TubularSegmentation()
            >>> seg, info = segmenter.segment(image)
            >>> print(f"迭代次数: {info['iterations']}")
        """
        # 归一化图像
        image_norm = self._normalize_image(image)
        
        # 初始化
        if initial_mask is None:
            # 使用简单的阈值初始化
            u = self._initialize_levelset(image_norm)
        else:
            u = initial_mask.astype(np.float64)
        
        # 优化循环 (Split Bregman)
        u_old = u.copy()
        
        for iter_idx in range(self.max_iter):
            # 1. 更新u (求解线性系统)
            u = self._update_u(image_norm, u)
            
            # 2. 小波框架正则化
            u = self._apply_framelet_regularization(u)
            
            # 3. 形状先验约束
            u = self._apply_shape_prior(u)
            
            # 4. 投影到 [0, 1]
            u = np.clip(u, 0, 1)
            
            # 计算能量
            energy = self._compute_energy(image_norm, u)
            self._update_history(image_norm, u, energy)
            
            # 检查收敛
            diff = np.max(np.abs(u - u_old))
            if diff < self.tol:
                print(f"收敛于迭代 {iter_idx}")
                break
            
            u_old = u.copy()
        
        # 二值化
        segmentation = (u > 0.5).astype(np.uint8)
        
        info = {
            'iterations': iter_idx + 1,
            'final_energy': energy,
            'history': self.history
        }
        
        return segmentation, info
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像到 [0, 1]
        
        参数:
            image: 输入图像
            
        返回:
            归一化图像
        """
        if image.ndim == 3:
            # 多通道转换为灰度
            image = np.mean(image, axis=-1)
        
        img_min = image.min()
        img_max = image.max()
        
        if img_max > img_min:
            return (image - img_min) / (img_max - img_min)
        else:
            return np.zeros_like(image)
    
    def _initialize_levelset(self, image: np.ndarray) -> np.ndarray:
        """
        初始化水平集函数
        
        使用图像的阈值分割作为初始值。
        
        参数:
            image: 输入图像
            
        返回:
            初始水平集 (0-1之间的值)
        """
        # 使用Otsu阈值
        from skimage.filters import threshold_otsu
        
        try:
            thresh = threshold_otsu(image)
            initial = (image > thresh).astype(np.float64)
        except:
            # Otsu失败时使用简单阈值
            initial = (image > image.mean()).astype(np.float64)
        
        # 应用轻微的高斯平滑
        initial = ndimage.gaussian_filter(initial, sigma=1.0)
        
        return initial
    
    def _update_u(
        self,
        image: np.ndarray,
        u: np.ndarray
    ) -> np.ndarray:
        """
        更新u (数据保真项优化)
        
        求解: min λ_data/2 * ||u - image||² + 正则化
        
        参数:
            image: 输入图像
            u: 当前估计
            
        返回:
            更新后的u
        """
        # 简单的梯度下降步骤
        # 数据保真梯度: λ_data * (u - image)
        data_gradient = self.lambda_data * (u - image)
        
        # 更新 (梯度下降)
        step_size = 0.1
        u_new = u - step_size * data_gradient
        
        return u_new
    
    def _apply_framelet_regularization(self, u: np.ndarray) -> np.ndarray:
        """
        应用小波框架正则化
        
        参数:
            u: 当前估计
            
        返回:
            正则化后的u
        """
        if self.lambda_framelet == 0:
            return u
        
        # 小波框架分解
        coeffs = self.framelet.decompose(u)
        
        # 对高频系数应用阈值 (鼓励稀疏性)
        threshold = self.lambda_framelet * 0.1
        denoised_coeffs = self.framelet.threshold_coeffs(
            coeffs,
            threshold=threshold,
            mode='soft'
        )
        
        # 重构
        u_reg = self.framelet.reconstruct(denoised_coeffs)
        
        # 确保形状一致
        if u_reg.shape != u.shape:
            u_reg = u_reg[:u.shape[0], :u.shape[1]]
        
        # 加权平均
        return (1 - self.lambda_framelet) * u + self.lambda_framelet * u_reg
    
    def _apply_shape_prior(self, u: np.ndarray) -> np.ndarray:
        """
        应用形状先验约束
        
        鼓励管状结构的连通性和光滑性。
        
        参数:
            u: 当前估计
            
        返回:
            应用形状先验后的u
        """
        if self.lambda_shape == 0:
            return u
        
        # 使用形态学操作保持连通性
        from skimage import morphology
        
        # 二值化
        binary = (u > 0.5).astype(np.uint8)
        
        # 移除小物体
        cleaned = morphology.remove_small_objects(
            binary.astype(bool),
            min_size=20
        )
        
        # 形态学闭运算 (填充小洞，连接邻近区域)
        selem = morphology.disk(2)
        closed = morphology.closing(cleaned, selem)
        
        # 形态学开运算 (去除毛刺)
        opened = morphology.opening(closed, morphology.disk(1))
        
        # 混合结果
        shape_term = opened.astype(np.float64)
        u_new = (1 - self.lambda_shape) * u + self.lambda_shape * shape_term
        
        return u_new
    
    def _compute_energy(
        self,
        image: np.ndarray,
        u: np.ndarray
    ) -> float:
        """
        计算总能量
        
        E = E_data + λ₁ E_framelet + λ₂ E_shape
        
        参数:
            image: 输入图像
            u: 当前估计
            
        返回:
            总能量值
        """
        # 数据保真项
        e_data = 0.5 * np.sum((u - image) ** 2)
        
        # 小波框架项 (高频能量)
        coeffs = self.framelet.decompose(u)
        e_framelet = 0
        for level in coeffs[1:]:  # 跳过近似系数
            if isinstance(level, tuple):
                e_framelet += sum(np.sum(c**2) for c in level)
        
        # 形状项 (梯度能量)
        grad_x = np.gradient(u, axis=0)
        grad_y = np.gradient(u, axis=1)
        e_shape = np.sum(grad_x**2) + np.sum(grad_y**2)
        
        # 总能量
        energy = (self.lambda_data * e_data +
                  self.lambda_framelet * e_framelet +
                  self.lambda_shape * e_shape)
        
        return energy
    
    def _update_history(
        self,
        image: np.ndarray,
        u: np.ndarray,
        total_energy: float
    ):
        """
        更新收敛历史
        
        参数:
            image: 输入图像
            u: 当前估计
            total_energy: 总能量
        """
        # 数据项
        e_data = 0.5 * np.sum((u - image) ** 2)
        
        # 框架项
        coeffs = self.framelet.decompose(u)
        e_framelet = 0
        for level in coeffs[1:]:
            if isinstance(level, tuple):
                e_framelet += sum(np.sum(c**2) for c in level)
        
        # 形状项
        grad_x = np.gradient(u, axis=0)
        grad_y = np.gradient(u, axis=1)
        e_shape = np.sum(grad_x**2) + np.sum(grad_y**2)
        
        self.history['energy'].append(total_energy)
        self.history['data_term'].append(self.lambda_data * e_data)
        self.history['framelet_term'].append(self.lambda_framelet * e_framelet)
        self.history['shape_term'].append(self.lambda_shape * e_shape)


if __name__ == "__main__":
    """
    分割算法测试
    """
    print("="*60)
    print("管状结构分割算法测试")
    print("="*60)
    
    # 创建模拟管状结构图像
    print("\n1. 创建模拟图像...")
    size = 128
    np.random.seed(42)
    
    # 创建管状结构 (S形曲线)
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # S形管
    center_y = 0.5 + 0.2 * np.sin(2 * np.pi * X * 2)
    tube = np.abs(Y - center_y) < 0.05
    
    # 添加噪声
    noise = np.random.randn(size, size) * 0.3
    image = tube.astype(np.float64) + noise
    
    print(f"   图像形状: {image.shape}")
    print(f"   信噪比: {np.mean(tube) / np.std(noise):.2f}")
    
    # 执行分割
    print("\n2. 执行分割...")
    segmenter = TubularSegmentation(
        lambda_framelet=0.1,
        lambda_shape=0.05,
        max_iter=50,
        tol=1e-3
    )
    
    seg, info = segmenter.segment(image)
    
    print(f"   迭代次数: {info['iterations']}")
    print(f"   最终能量: {info['final_energy']:.4f}")
    
    # 简单评估
    intersection = np.sum(seg * tube)
    union = np.sum((seg + tube) > 0)
    iou = intersection / union if union > 0 else 0
    print(f"   IoU (与真实值): {iou:.4f}")
    
    print("\n" + "="*60)
    print("✅ 分割测试完成!")
    print("="*60)
    print("\n注意: 这是基础实现，完整版本需要:")
    print("  - 更精细的形状先验建模")
    print("  - 更高效的优化算法")
    print("  - 真实医学图像测试")
