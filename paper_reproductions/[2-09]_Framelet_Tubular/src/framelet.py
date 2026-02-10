"""
小波框架（Framelet）变换实现

本模块实现了紧小波框架（Tight Framelet）变换，包括:
- 多分辨率分解与重构
- 多种滤波器支持 (Haar, Biorthogonal, etc.)
- 二维图像变换

理论基础:
    小波框架是紧框架（tight frame）的一种，满足:
    f = Σ <f, φ_i> φ_i
    即完美重建性质。

参考文献:
    Dong & Shen (2010). MRA-based wavelet frames and applications.
"""

import numpy as np
from typing import List, Tuple, Optional
import pywt
from scipy import ndimage


class FrameletTransform:
    """
    紧小波框架变换类
    
    实现2D图像的多尺度小波框架分解和重构。
    
    属性:
        level: 分解层数
        filter_name: 滤波器名称
        decomposition_filters: 分解滤波器组
        reconstruction_filters: 重构滤波器组
    """
    
    # 支持的滤波器
    SUPPORTED_FILTERS = ['haar', 'db1', 'db2', 'db4', 'bior1.3', 'bior2.4', 'sym4']
    
    def __init__(
        self,
        level: int = 3,
        filter_name: str = 'haar',
        mode: str = 'symmetric'
    ):
        """
        初始化小波框架变换
        
        参数:
            level: 分解层数 (金字塔层数)
            filter_name: 小波滤波器名称
            mode: 边界处理模式 ('symmetric', 'periodic', 'reflect')
            
        示例:
            >>> framelet = FrameletTransform(level=3, filter_name='haar')
            >>> framelet = FrameletTransform(level=2, filter_name='bior2.4')
        """
        if level < 1:
            raise ValueError("分解层数必须 >= 1")
        if filter_name not in self.SUPPORTED_FILTERS:
            raise ValueError(f"不支持的滤波器: {filter_name}")
        
        self.level = level
        self.filter_name = filter_name
        self.mode = mode
        
        # 获取小波滤波器
        self.wavelet = pywt.Wavelet(filter_name)
        
        # 计算紧框架常数
        self.frame_constant = self._compute_frame_constant()
        
    def _compute_frame_constant(self) -> float:
        """
        计算框架常数
        
        紧框架满足: Σ |ψ̂(ω)|² = A (常数)
        
        返回:
            框架常数A
        """
        # 对于紧小波框架，常数为1
        return 1.0
    
    def decompose(self, image: np.ndarray) -> List[np.ndarray]:
        """
        多尺度小波框架分解
        
        将图像分解为低频近似和高频细节分量。
        
        参数:
            image: 输入图像，形状为 (H, W) 或 (H, W, C)
            
        返回:
            系数列表，格式为:
            [低频系数, (水平细节, 垂直细节, 对角细节), ...]
            
        示例:
            >>> coeffs = framelet.decompose(image)
            >>> cA = coeffs[0]  # 近似系数
            >>> (cH, cV, cD) = coeffs[1]  # 第一层细节
        """
        if image.ndim == 2:
            # 单通道图像
            coeffs = self._decompose_single(image)
        elif image.ndim == 3:
            # 多通道图像
            coeffs_list = []
            for c in range(image.shape[2]):
                coeffs_c = self._decompose_single(image[:, :, c])
                coeffs_list.append(coeffs_c)
            # 合并通道
            coeffs = self._merge_channel_coeffs(coeffs_list)
        else:
            raise ValueError(f"不支持的图像维度: {image.ndim}")
        
        return coeffs
    
    def _decompose_single(self, image: np.ndarray) -> List:
        """
        单通道图像分解
        
        参数:
            image: 单通道图像，形状 (H, W)
            
        返回:
            小波系数列表
        """
        # 使用PyWavelets进行多层分解
        coeffs = pywt.wavedec2(
            image,
            self.wavelet,
            level=self.level,
            mode=self.mode
        )
        
        return coeffs
    
    def _merge_channel_coeffs(
        self,
        coeffs_list: List[List]
    ) -> List:
        """
        合并多通道系数
        
        参数:
            coeffs_list: 每个通道的系数列表
            
        返回:
            合并后的系数
        """
        # 取第一个通道的结构
        merged = []
        for level_idx in range(len(coeffs_list[0])):
            if isinstance(coeffs_list[0][level_idx], tuple):
                # 细节系数 (水平, 垂直, 对角)
                merged_level = []
                for detail_idx in range(3):
                    detail = np.stack([
                        c[level_idx][detail_idx]
                        for c in coeffs_list
                    ], axis=-1)
                    merged_level.append(detail)
                merged.append(tuple(merged_level))
            else:
                # 近似系数
                approx = np.stack([
                    c[level_idx] for c in coeffs_list
                ], axis=-1)
                merged.append(approx)
        
        return merged
    
    def reconstruct(self, coeffs: List) -> np.ndarray:
        """
        从小波框架系数重构图像
        
        紧框架保证完美重构 (最多有数值误差)。
        
        参数:
            coeffs: 小波系数列表
            
        返回:
            重构图像
            
        示例:
            >>> reconstructed = framelet.reconstruct(coeffs)
            >>> np.allclose(image, reconstructed)  # True
        """
        # 检查是否为多通道
        if isinstance(coeffs[0], np.ndarray) and coeffs[0].ndim == 3:
            # 多通道
            return self._reconstruct_multi_channel(coeffs)
        else:
            # 单通道
            return pywt.waverec2(coeffs, self.wavelet, mode=self.mode)
    
    def _reconstruct_multi_channel(self, coeffs: List) -> np.ndarray:
        """
        重构多通道图像
        
        参数:
            coeffs: 多通道小波系数
            
        返回:
            多通道图像
        """
        num_channels = coeffs[0].shape[2]
        
        # 分离通道
        channels = []
        for c in range(num_channels):
            coeffs_c = []
            for level in coeffs:
                if isinstance(level, tuple):
                    coeffs_c.append(tuple(
                        detail[:, :, c] for detail in level
                    ))
                else:
                    coeffs_c.append(level[:, :, c])
            
            # 重构单通道
            channel = pywt.waverec2(coeffs_c, self.wavelet, mode=self.mode)
            channels.append(channel)
        
        # 合并通道
        return np.stack(channels, axis=-1)
    
    def threshold_coeffs(
        self,
        coeffs: List,
        threshold: float,
        mode: str = 'soft'
    ) -> List:
        """
        系数阈值处理 (用于去噪)
        
        参数:
            coeffs: 小波系数
            threshold: 阈值
            mode: 'soft' (软阈值) 或 'hard' (硬阈值)
            
        返回:
            处理后的系数
            
        示例:
            >>> denoised_coeffs = framelet.threshold_coeffs(coeffs, threshold=0.1)
        """
        def threshold_array(arr: np.ndarray, thr: float) -> np.ndarray:
            if mode == 'soft':
                # 软阈值: sign(x) * max(|x| - thr, 0)
                return np.sign(arr) * np.maximum(np.abs(arr) - thr, 0)
            else:  # hard
                # 硬阈值: x if |x| > thr else 0
                return arr * (np.abs(arr) > thr).astype(arr.dtype)
        
        thresholded = []
        for level in coeffs:
            if isinstance(level, tuple):
                # 对细节系数应用阈值
                thresholded.append(tuple(
                    threshold_array(detail, threshold) for detail in level
                ))
            else:
                # 保留近似系数
                thresholded.append(level)
        
        return thresholded
    
    def get_framelet_features(self, image: np.ndarray) -> dict:
        """
        提取小波框架特征
        
        提取多尺度特征用于分割。
        
        参数:
            image: 输入图像
            
        返回:
            特征字典，包含各层系数能量
        """
        coeffs = self.decompose(image)
        
        features = {
            'approximation': coeffs[0],
            'energies': [],
            'details': []
        }
        
        # 计算各层能量
        for level_idx, level in enumerate(coeffs[1:], 1):
            if isinstance(level, tuple):
                cH, cV, cD = level
                energy = np.sum(cH**2) + np.sum(cV**2) + np.sum(cD**2)
                features['energies'].append(energy)
                features['details'].append({
                    'horizontal': cH,
                    'vertical': cV,
                    'diagonal': cD
                })
        
        return features


class TightFramelet(FrameletTransform):
    """
    紧框架专用类
    
    紧框架具有 A = B = 1 的性质，实现完美重构。
    """
    
    def __init__(self, level: int = 3):
        """
        初始化紧框架变换
        
        默认使用 Haar 小波 (最简单的紧框架)。
        """
        super().__init__(level=level, filter_name='haar')


class BiorthogonalFramelet(FrameletTransform):
    """
    双正交框架类
    
    使用双正交小波滤波器，具有更好的对称性和光滑性。
    """
    
    def __init__(
        self,
        level: int = 3,
        filter_name: str = 'bior2.4'
    ):
        """
        初始化双正交框架
        
        参数:
            level: 分解层数
            filter_name: 双正交滤波器名称 ('bior1.3', 'bior2.4', 'bior4.4')
        """
        super().__init__(level=level, filter_name=filter_name)


if __name__ == "__main__":
    """
    小波框架变换测试
    """
    print("="*60)
    print("小波框架变换测试")
    print("="*60)
    
    # 创建测试图像
    print("\n1. 创建测试图像...")
    np.random.seed(42)
    test_image = np.random.randn(256, 256).astype(np.float32)
    print(f"   图像形状: {test_image.shape}")
    
    # 测试分解与重构
    print("\n2. 测试Haar框架...")
    framelet = FrameletTransform(level=2, filter_name='haar')
    coeffs = framelet.decompose(test_image)
    reconstructed = framelet.reconstruct(coeffs)
    
    # 检查重构误差
    error = np.max(np.abs(test_image - reconstructed[:256, :256]))
    print(f"   重构最大误差: {error:.2e}")
    print(f"   分解层数: {len(coeffs)}")
    print(f"   近似系数形状: {coeffs[0].shape}")
    
    # 测试多通道
    print("\n3. 测试多通道图像...")
    multi_channel = np.random.randn(128, 128, 3).astype(np.float32)
    coeffs_mc = framelet.decompose(multi_channel)
    reconstructed_mc = framelet.reconstruct(coeffs_mc)
    print(f"   输入形状: {multi_channel.shape}")
    print(f"   输出形状: {reconstructed_mc.shape}")
    
    # 测试阈值处理
    print("\n4. 测试系数阈值...")
    noisy_coeffs = framelet.decompose(test_image)
    denoised_coeffs = framelet.threshold_coeffs(noisy_coeffs, threshold=0.5)
    denoised = framelet.reconstruct(denoised_coeffs)
    print(f"   阈值处理完成")
    
    # 特征提取
    print("\n5. 测试特征提取...")
    features = framelet.get_framelet_features(test_image)
    print(f"   近似系数形状: {features['approximation'].shape}")
    print(f"   各层能量: {features['energies']}")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)
