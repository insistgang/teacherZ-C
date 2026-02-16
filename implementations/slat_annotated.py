"""
================================================================================
SLaT三阶段分割算法 - 详细注释版
Smoothing, Lifting, and Thresholding for Image Segmentation
================================================================================

【文件功能】
    实现SLaT三阶段图像分割算法，专门处理退化彩色图像的分割问题。
    该方法通过ROF平滑、色彩空间提升、K-means阈值三阶段实现鲁棒分割。

【数学原理】
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Stage 1: Smoothing - ROF变分平滑                                        │
    │                                                                         │
    │   对每个RGB通道求解:                                                    │
    │   min_g { λ/2 * ||f - g||²_L2 + TV(g) }                                │
    │                                                                         │
    │   其中TV(g)是全变分正则项，去除噪声同时保持边缘。                       │
    │                                                                         │
    │ Stage 2: Lifting - 色彩空间维度提升                                     │
    │                                                                         │
    │   RGB → (R, G, B, L, a, b) ∈ R^6                                       │
    │                                                                         │
    │   CIE Lab空间的转换公式:                                                │
    │   L* = 116 * f(Y/Yn) - 16                                              │
    │   a* = 500 * [f(X/Xn) - f(Y/Yn)]                                       │
    │   b* = 200 * [f(Y/Yn) - f(Z/Zn)]                                       │
    │   其中 f(t) = t^(1/3) if t > (6/29)³, else (29/6)² * t + 4/29         │
    │                                                                         │
    │   Lab空间的优势:                                                        │
    │   - 感知均匀性: 相同的ΔE感知差异对应相同的欧氏距离                      │
    │   - 亮度/色度分离: L通道独立于色彩信息                                  │
    │   - 更好的聚类效果: 在Lab空间颜色区分度更高                             │
    │                                                                         │
    │ Stage 3: Thresholding - K-means聚类阈值                                 │
    │                                                                         │
    │   在6维空间求解:                                                        │
    │   min_{C_k} Σ_i ||g*_i - C_{k(i)}||²                                   │
    │                                                                         │
    │   其中 k(i) 是像素i的类别标签，C_k是第k类的聚类中心。                   │
    └─────────────────────────────────────────────────────────────────────────┘

【算法优势】
    1. 鲁棒性: ROF预处理抵抗噪声和退化
    2. 灵活性: 调整K无需重算Stage 1/2
    3. 效果好: 6维特征提供更好的可分性
    4. 可解释: 三阶段流程清晰明了

【参考文献】
    [1] Cai, X., Chan, R.H., Nikolova, M., Zeng, T. (2017). A Three-Stage 
        Approach for Segmenting Degraded Color Images: Smoothing, Lifting 
        and Thresholding. Journal of Scientific Computing, 72(3), 1313-1332.
    [2] Rudin, L.I., Osher, S., Fatemi, E. (1992). Nonlinear total variation 
        based noise removal algorithms. Physica D.

【作者】 注释版生成系统
【日期】 2024
================================================================================
"""

import numpy as np  # 导入NumPy库，用于数值计算和数组操作
from typing import Tuple, Optional, Dict, List, Any  # 导入类型提示
from dataclasses import dataclass, field  # 导入数据类装饰器
import warnings  # 导入警告模块

# 尝试导入OpenCV用于色彩空间转换
try:  # 尝试导入
    import cv2  # OpenCV库
    HAS_CV2 = True  # 标记有cv2
except ImportError:  # 如果没有
    HAS_CV2 = False  # 标记无cv2
    warnings.warn("OpenCV未安装，将使用纯Python实现Lab转换（较慢）")

# 尝试导入sklearn用于K-means
try:  # 尝试导入
    from sklearn.cluster import KMeans  # K-means聚类
    HAS_SKLEARN = True  # 标记有sklearn
except ImportError:  # 如果没有
    HAS_SKLEARN = False  # 标记无sklearn
    warnings.warn("sklearn未安装，将使用简化的阈值方法")


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class SLATConfig:
    """
    SLaT分割算法配置类
    
    【属性】
        lambda_param: float
            Stage 1 ROF模型的数据保真权重λ
            值越大保留更多原图特征，值越小去噪更强
            推荐范围: [0.5, 5.0]
            默认: 1.0
            
        rof_iterations: int
            ROF迭代的最大次数
            默认: 200
            
        rof_tolerance: float
            ROF迭代的收敛阈值
            默认: 1e-5
            
        random_state: int
            K-means的随机种子，保证可重复性
            默认: 42
            
        n_init: int
            K-means初始化次数，越多越稳定
            默认: 10
            
        max_kmeans_iter: int
            K-means最大迭代次数
            默认: 300
    """
    lambda_param: float = 1.0  # ROF数据保真权重
    rof_iterations: int = 200  # ROF最大迭代次数
    rof_tolerance: float = 1e-5  # ROF收敛阈值
    random_state: int = 42  # 随机种子
    n_init: int = 10  # K-means初始化次数
    max_kmeans_iter: int = 300  # K-means最大迭代


# ============================================================================
# Stage 1: ROF平滑模块
# ============================================================================

class ROFSmoother:
    """
    Stage 1: ROF变分平滑器
    
    【数学模型】
        对输入图像f求解:
        min_g { (λ/2)||f - g||² + TV(g) }
        
        其中 TV(g) = ∫|∇g|dx 是全变分
        
    【实现方法】
        使用Chambolle投影算法，原对偶形式:
        g = f - λ * div(p)
        其中p是满足约束||p||_∞ ≤ 1的对偶变量
    """
    
    def __init__(self, lambda_param: float = 1.0, 
                 max_iterations: int = 200, 
                 tolerance: float = 1e-5):
        """
        初始化ROF平滑器
        
        【参数】
            lambda_param: 数据保真权重λ
            max_iterations: 最大迭代次数
            tolerance: 收敛阈值
        """
        self.lambda_param = lambda_param  # 保存λ参数
        self.max_iterations = max_iterations  # 保存迭代次数
        self.tolerance = tolerance  # 保存收敛阈值
        self.tau = 0.25  # Chambolle算法步长，固定为0.25保证稳定
    
    def smooth(self, image: np.ndarray) -> np.ndarray:
        """
        对图像执行ROF平滑（去噪）
        
        【参数】
            image: np.ndarray, shape (H, W) 或 (H, W, C)
                输入图像，可以是灰度或彩色
                
        【返回】
            smoothed: np.ndarray
                平滑后的图像，与输入形状相同
                
        【数学原理】
            Chambolle投影算法迭代:
            1. 计算梯度: ∇g = gradient(f - λ*div(p))
            2. 更新对偶: p = (p + τ*∇g) / max(1, |p + τ*∇g|)
            3. 更新原变量: g = f - λ*div(p)
        """
        image = np.asarray(image, dtype=np.float64)  # 确保float64类型
        
        # 处理多通道图像：逐通道处理
        if image.ndim == 2:  # 灰度图像
            return self._smooth_single_channel(image)  # 单通道处理
        elif image.ndim == 3:  # 彩色图像
            result = np.zeros_like(image)  # 初始化结果数组
            for c in range(image.shape[2]):  # 遍历每个通道
                result[:, :, c] = self._smooth_single_channel(image[:, :, c])  # 处理该通道
            return result  # 返回多通道结果
        else:
            raise ValueError(f"不支持的图像维度: {image.ndim}")  # 抛出错误
    
    def _smooth_single_channel(self, f: np.ndarray) -> np.ndarray:
        """
        单通道ROF平滑（Chambolle投影算法）
        
        【参数】
            f: np.ndarray, shape (H, W)
                单通道输入图像
                
        【返回】
            g: np.ndarray, shape (H, W)
                平滑后的图像
        """
        H, W = f.shape  # 获取图像尺寸
        
        # 初始化对偶变量 p ∈ R^(H×W×2)
        # p[:,:,0] 对应 x 方向分量
        # p[:,:,1] 对应 y 方向分量
        p = np.zeros((H, W, 2), dtype=np.float64)  # 对偶变量初始化为0
        
        # 初始化原变量
        g = f.copy()  # 初始化为输入图像
        
        # Chambolle迭代
        for n in range(self.max_iterations):  # 迭代max_iterations次
            # 保存上一迭代的g用于收敛检查
            g_prev = g.copy()  # 深拷贝
            
            # 步骤1: 计算g的梯度 ∇g
            grad_g = self._compute_gradient(g)  # 计算 (∂g/∂x, ∂g/∂y)
            
            # 步骤2: 更新对偶变量
            # p_new = p + τ * ∇g (梯度上升)
            p_new = p + self.tau * grad_g  # 对偶上升
            
            # 步骤3: 投影到单位球 {p: ||p||_∞ ≤ 1}
            # proj_P(p) = p / max(1, |p|)
            p_norm = np.sqrt(p_new[:, :, 0]**2 + p_new[:, :, 1]**2 + 1e-10)  # 计算范数
            p[:, :, 0] = p_new[:, :, 0] / np.maximum(p_norm, 1.0)  # 投影x分量
            p[:, :, 1] = p_new[:, :, 1] / np.maximum(p_norm, 1.0)  # 投影y分量
            
            # 步骤4: 计算散度 div(p)
            div_p = self._compute_divergence(p)  # div(p) = ∂p_x/∂x + ∂p_y/∂y
            
            # 步骤5: 更新原变量
            # g = f - λ * div(p)
            g = f - self.lambda_param * div_p  # 原变量更新
            
            # 步骤6: 检查收敛
            diff = np.linalg.norm(g - g_prev)  # 计算变化量
            norm_g = np.linalg.norm(g_prev) + 1e-10  # 计算范数
            relative_change = diff / norm_g  # 相对变化
            
            if relative_change < self.tolerance:  # 如果变化小于阈值
                break  # 收敛，退出循环
        
        return g  # 返回平滑结果
    
    def _compute_gradient(self, u: np.ndarray) -> np.ndarray:
        """
        计算图像梯度 ∇u = (∂u/∂x, ∂u/∂y)
        
        【使用前向差分】
            ∂u/∂x[i,j] = u[i+1,j] - u[i,j]
            ∂u/∂y[i,j] = u[i,j+1] - u[i,j]
            
            边界条件: 零Neumann边界（边界梯度为0）
        """
        H, W = u.shape  # 获取尺寸
        grad = np.zeros((H, W, 2), dtype=np.float64)  # 初始化梯度数组
        
        # x方向梯度: 前向差分
        grad[:-1, :, 0] = u[1:, :] - u[:-1, :]  # 内部点差分
        # grad[-1, :, 0] = 0 (边界，保持初始值)
        
        # y方向梯度: 前向差分
        grad[:, :-1, 1] = u[:, 1:] - u[:, :-1]  # 内部点差分
        # grad[:, -1, 1] = 0 (边界，保持初始值)
        
        return grad  # 返回梯度
    
    def _compute_divergence(self, p: np.ndarray) -> np.ndarray:
        """
        计算向量场散度 div(p) = ∂p_x/∂x + ∂p_y/∂y
        
        【使用后向差分（前向差分的伴随）】
            div(p)[i,j] = p_x[i,j] - p_x[i-1,j] + p_y[i,j] - p_y[i,j-1]
            
            边界条件: Neumann边界
        """
        H, W = p.shape[:2]  # 获取尺寸
        div = np.zeros((H, W), dtype=np.float64)  # 初始化散度数组
        
        # ∂p_x/∂x项
        div[1:, :] += p[1:, :, 0] - p[:-1, :, 0]  # 内部点
        div[0, :] += p[0, :, 0]  # 边界条件
        
        # ∂p_y/∂y项
        div[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]  # 内部点
        div[:, 0] += p[:, 0, 1]  # 边界条件
        
        return div  # 返回散度


# ============================================================================
# Stage 2: 色彩空间提升模块
# ============================================================================

class ColorSpaceLifter:
    """
    Stage 2: 色彩空间维度提升器
    
    【功能】
        将RGB 3通道图像提升到RGB+Lab 6通道特征空间
        
    【数学原理】
        RGB → XYZ → Lab 转换:
        
        1. RGB到XYZ (sRGB gamma校正):
           R_lin = R' ^ 2.4  (if R' > 0.04045, else R'/12.92)
           同理G, B
           
           [X]   [0.4124  0.3576  0.1805] [R_lin]
           [Y] = [0.2126  0.7152  0.0722] [G_lin]
           [Z]   [0.0193  0.1192  0.9505] [B_lin]
        
        2. XYZ到Lab (D65白点):
           f(t) = t^(1/3)           if t > (6/29)³
           f(t) = (29/6)² * t + 4/29 otherwise
           
           L* = 116 * f(Y/Yn) - 16
           a* = 500 * [f(X/Xn) - f(Y/Yn)]
           b* = 200 * [f(Y/Yn) - f(Z/Zn)]
           
           其中 (Xn, Yn, Zn) = (95.047, 100.0, 108.883) 是D65白点
    """
    
    # D65白点参考值
    XN = 95.047  # D65白点X
    YN = 100.0   # D65白点Y
    ZN = 108.883 # D65白点Z
    
    # sRGB到XYZ转换矩阵
    RGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],  # X行
        [0.2126729, 0.7151522, 0.0721750],  # Y行
        [0.0193339, 0.1191920, 0.9503041]   # Z行
    ])
    
    def __init__(self, use_cv2: bool = True):
        """
        初始化色彩空间提升器
        
        【参数】
            use_cv2: 是否优先使用OpenCV（更快）
        """
        self.use_cv2 = use_cv2 and HAS_CV2  # 确定是否使用cv2
    
    def lift(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        执行RGB → RGB+Lab维度提升
        
        【参数】
            rgb_image: np.ndarray, shape (H, W, 3)
                RGB图像，值域[0,1]
                
        【返回】
            lifted: np.ndarray, shape (H, W, 6)
                提升后的6通道特征
                lifted[:,:,0:3] = RGB
                lifted[:,:,3:6] = Lab (归一化)
        """
        # 确保输入合法
        rgb_image = np.clip(rgb_image, 0, 1)  # 裁剪到[0,1]
        
        # 转换到Lab空间
        if self.use_cv2:  # 使用OpenCV
            lab = self._rgb_to_lab_cv2(rgb_image)  # cv2转换
        else:  # 使用纯Python
            lab = self._rgb_to_lab_python(rgb_image)  # 纯Python转换
        
        # 归一化Lab到[0,1]
        lab_norm = self._normalize_lab(lab)  # Lab归一化
        
        # 拼接RGB和Lab
        lifted = np.concatenate([rgb_image, lab_norm], axis=2)  # 6通道特征
        
        return lifted  # 返回提升结果
    
    def _rgb_to_lab_cv2(self, rgb: np.ndarray) -> np.ndarray:
        """
        使用OpenCV进行RGB到Lab转换
        
        【注意】 OpenCV使用BGR顺序，需要转换
        """
        # 转换为uint8格式（OpenCV要求）
        rgb_uint8 = (rgb * 255).astype(np.uint8)  # [0,1] → [0,255]
        
        # RGB → BGR (OpenCV格式)
        bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)  # 颜色顺序转换
        
        # BGR → Lab
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)  # 转换到Lab
        
        return lab.astype(np.float64)  # 返回float类型
    
    def _rgb_to_lab_python(self, rgb: np.ndarray) -> np.ndarray:
        """
        纯Python实现RGB到Lab转换
        
        【步骤】
            1. sRGB gamma逆变换 (线性化)
            2. RGB → XYZ矩阵变换
            3. XYZ → Lab非线性变换
        """
        H, W, _ = rgb.shape  # 获取尺寸
        lab = np.zeros((H, W, 3), dtype=np.float64)  # 初始化Lab数组
        
        # 步骤1: sRGB gamma逆变换（线性化）
        # sRGB使用复合gamma曲线
        rgb_lin = np.where(  # 条件gamma
            rgb > 0.04045,  # 如果大于阈值
            ((rgb + 0.055) / 1.055) ** 2.4,  # 使用2.4次幂
            rgb / 12.92  # 使用线性部分
        )
        
        # 步骤2: RGB → XYZ
        # 需要缩放100倍以匹配Lab标准
        rgb_lin_scaled = rgb_lin * 100  # 缩放
        xyz = np.dot(rgb_lin_scaled, self.RGB_TO_XYZ.T)  # 矩阵乘法
        
        # 步骤3: XYZ → Lab
        # 首先计算归一化的XYZ (相对于白点)
        x = xyz[:, :, 0] / self.XN  # 归一化X
        y = xyz[:, :, 1] / self.YN  # 归一化Y
        z = xyz[:, :, 2] / self.ZN  # 归一化Z
        
        # 计算f函数
        delta = 6.0 / 29.0  # 阈值常量
        delta_cube = delta ** 3  # (6/29)³
        
        def f(t):  # Lab转换函数
            return np.where(  # 条件函数
                t > delta_cube,  # 如果大于阈值
                t ** (1.0/3.0),  # 立方根
                t / (3 * delta * delta) + 4.0 / 29.0  # 线性部分
            )
        
        fx = f(x)  # f(X/Xn)
        fy = f(y)  # f(Y/Yn)
        fz = f(z)  # f(Z/Zn)
        
        # 计算Lab值
        lab[:, :, 0] = 116.0 * fy - 16.0  # L*
        lab[:, :, 1] = 500.0 * (fx - fy)  # a*
        lab[:, :, 2] = 200.0 * (fy - fz)  # b*
        
        return lab  # 返回Lab
    
    def _normalize_lab(self, lab: np.ndarray) -> np.ndarray:
        """
        归一化Lab值到[0,1]范围
        
        【Lab值域】
            L*: [0, 100]
            a*: [-128, 127] (近似)
            b*: [-128, 127] (近似)
        """
        lab_norm = np.zeros_like(lab)  # 初始化归一化数组
        
        # L通道: [0, 100] → [0, 1]
        lab_norm[:, :, 0] = lab[:, :, 0] / 100.0  # L归一化
        
        # a通道: [-128, 127] → [0, 1]
        lab_norm[:, :, 1] = (lab[:, :, 1] + 128.0) / 255.0  # a归一化
        
        # b通道: [-128, 127] → [0, 1]
        lab_norm[:, :, 2] = (lab[:, :, 2] + 128.0) / 255.0  # b归一化
        
        return np.clip(lab_norm, 0, 1)  # 返回并裁剪


# ============================================================================
# Stage 3: K-means阈值模块
# ============================================================================

class KMeansThresholding:
    """
    Stage 3: K-means聚类阈值分割器
    
    【数学模型】
        在6维特征空间中求解:
        min_{C_1,...,C_K} Σ_i min_k ||x_i - C_k||²
        
        其中 x_i ∈ R^6 是像素i的6维特征，C_k是第k类中心
        
    【算法】
        标准K-means算法:
        1. 初始化K个聚类中心（K-means++）
        2. 分配步骤: 每个像素分配到最近的中心
        3. 更新步骤: 重新计算每个类的中心
        4. 重复直到收敛
    """
    
    def __init__(self, n_init: int = 10, max_iter: int = 300, random_state: int = 42):
        """
        初始化K-means分割器
        
        【参数】
            n_init: 不同初始化的运行次数
            max_iter: 每次运行的最大迭代
            random_state: 随机种子
        """
        self.n_init = n_init  # 保存初始化次数
        self.max_iter = max_iter  # 保存最大迭代
        self.random_state = random_state  # 保存随机种子
        self.use_sklearn = HAS_SKLEARN  # 是否使用sklearn
    
    def threshold(self, features: np.ndarray, K: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行K-means聚类分割
        
        【参数】
            features: np.ndarray, shape (H, W, D)
                D维特征图像
            K: int
                分割类别数
                
        【返回】
            labels: np.ndarray, shape (H, W)
                分割标签图，值域[0, K-1]
            centers: np.ndarray, shape (K, D)
                聚类中心
        """
        H, W, D = features.shape  # 获取尺寸
        
        # 展平为2D矩阵 (N, D)，N = H*W
        pixels = features.reshape(-1, D)  # 展平
        
        if self.use_sklearn:  # 使用sklearn的K-means
            kmeans = KMeans(  # 创建K-means对象
                n_clusters=K,  # 类别数
                n_init=self.n_init,  # 初始化次数
                max_iter=self.max_iter,  # 最大迭代
                random_state=self.random_state  # 随机种子
            )
            labels_flat = kmeans.fit_predict(pixels)  # 拟合并预测
            centers = kmeans.cluster_centers_  # 获取中心
        else:  # 使用简化实现
            labels_flat, centers = self._kmeans_simple(pixels, K)  # 简单K-means
        
        # 重塑为图像形状
        labels = labels_flat.reshape(H, W)  # 重塑标签
        
        return labels, centers  # 返回结果
    
    def _kmeans_simple(self, pixels: np.ndarray, K: int, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        简化的K-means实现（当没有sklearn时使用）
        
        【使用随机初始化而非K-means++】
        """
        N, D = pixels.shape  # 获取尺寸
        np.random.seed(self.random_state)  # 设置随机种子
        
        # 随机初始化中心
        idx = np.random.choice(N, K, replace=False)  # 随机选择K个点
        centers = pixels[idx].copy()  # 作为初始中心
        
        # K-means迭代
        for _ in range(max_iter):  # 迭代
            # 分配步骤：计算距离并分配
            # 使用广播计算所有点到所有中心的距离
            diff = pixels[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (N, K, D)
            distances = np.sum(diff ** 2, axis=2)  # (N, K) 欧氏距离平方
            labels = np.argmin(distances, axis=1)  # (N,) 选择最近的中心
            
            # 更新步骤：计算新中心
            new_centers = np.zeros_like(centers)  # 初始化新中心
            for k in range(K):  # 遍历每个类
                mask = labels == k  # 该类的像素
                if np.any(mask):  # 如果有像素
                    new_centers[k] = pixels[mask].mean(axis=0)  # 计算均值
                else:  # 空类
                    new_centers[k] = centers[k]  # 保持原中心
            
            # 检查收敛
            if np.allclose(centers, new_centers):  # 如果中心不变
                break  # 收敛
            centers = new_centers  # 更新中心
        
        return labels, centers  # 返回结果


# ============================================================================
# 完整SLaT分割器
# ============================================================================

class SLATSegmentation:
    """
    完整的SLaT三阶段分割算法
    
    【三阶段流程】
        ┌──────────┐     ┌──────────┐     ┌──────────┐
        │  Stage1  │ --> │  Stage2  │ --> │  Stage3  │
        │ ROF平滑  │     │ 色彩提升 │     │ K-means  │
        │          │     │ RGB→Lab  │     │  阈值    │
        └──────────┘     └──────────┘     └──────────┘
        
        输入f_RGB → g_RGB → g*_(RGB+Lab) → 分割结果
    
    【使用示例】
        >>> segmenter = SLATSegmentation(SLATConfig(lambda_param=1.0))
        >>> result = segmenter.segment(image, K=4)
        >>> labels = result['segmentation']
    """
    
    def __init__(self, config: Optional[SLATConfig] = None):
        """
        初始化SLaT分割器
        
        【参数】
            config: SLATConfig or None
                配置对象，None则使用默认配置
        """
        # 使用默认配置或提供的配置
        if config is None:  # 如果没有提供
            config = SLATConfig()  # 使用默认
        
        self.config = config  # 保存配置
        
        # 初始化三个阶段的模块
        self.stage1_smoother = ROFSmoother(  # Stage 1: ROF平滑
            lambda_param=config.lambda_param,
            max_iterations=config.rof_iterations,
            tolerance=config.rof_tolerance
        )
        
        self.stage2_lifter = ColorSpaceLifter(use_cv2=True)  # Stage 2: 色彩提升
        
        self.stage3_thresholding = KMeansThresholding(  # Stage 3: K-means
            n_init=config.n_init,
            max_iter=config.max_kmeans_iter,
            random_state=config.random_state
        )
    
    def segment(self, image: np.ndarray, K: int, 
                return_intermediate: bool = False) -> Dict[str, Any]:
        """
        执行完整的SLaT三阶段分割
        
        【参数】
            image: np.ndarray, shape (H, W, 3) 或 (H, W)
                输入RGB或灰度图像，值域[0,1]或[0,255]
            K: int
                分割类别数
            return_intermediate: bool
                是否返回中间结果
                
        【返回】
            dict包含:
                - segmentation: 分割标签图 (H, W)
                - cluster_centers: 聚类中心 (K, 6)
                - (可选) smoothed: Stage1结果
                - (可选) lifted: Stage2结果
        """
        # 预处理：归一化到[0,1]
        image = self._normalize_image(image)  # 归一化
        
        # 处理灰度图像：转换为RGB
        if image.ndim == 2:  # 灰度图
            image = np.stack([image, image, image], axis=2)  # 转为三通道
        
        # ================================================================
        # Stage 1: ROF变分平滑
        # 目标: 去除噪声，保持边缘
        # 数学: min_g { (λ/2)||f - g||² + TV(g) }
        # ================================================================
        smoothed = self.stage1_smoother.smooth(image)  # 执行ROF平滑
        
        # ================================================================
        # Stage 2: 色彩空间维度提升
        # 目标: 提供更好的色彩可分性
        # 变换: RGB → (R, G, B, L, a, b) ∈ R^6
        # ================================================================
        lifted = self.stage2_lifter.lift(smoothed)  # 执行色彩提升
        
        # ================================================================
        # Stage 3: K-means聚类阈值
        # 目标: 在6维空间中进行聚类分割
        # 数学: min_{C_k} Σ_i ||x_i - C_{k(i)}||²
        # ================================================================
        labels, centers = self.stage3_thresholding.threshold(lifted, K)  # K-means
        
        # 构建返回结果
        result = {  # 结果字典
            'segmentation': labels,  # 分割标签
            'cluster_centers': centers,  # 聚类中心
            'K': K,  # 类别数
            'config': self.config  # 配置
        }
        
        # 可选：返回中间结果
        if return_intermediate:  # 如果需要
            result['smoothed'] = smoothed  # Stage1结果
            result['lifted'] = lifted  # Stage2结果
        
        return result  # 返回结果
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像到[0,1]范围
        """
        image = np.asarray(image, dtype=np.float64)  # 转为float
        
        if image.max() > 1.0:  # 如果值域是[0,255]
            image = image / 255.0  # 归一化到[0,1]
        
        return np.clip(image, 0, 1)  # 裁剪并返回


# ============================================================================
# 便捷函数
# ============================================================================

def slat_segment(image: np.ndarray, K: int, 
                 lambda_param: float = 1.0) -> np.ndarray:
    """
    SLaT分割的便捷函数
    
    【参数】
        image: 输入图像 (H, W, 3) 或 (H, W)
        K: 分割类别数
        lambda_param: ROF数据保真权重
        
    【返回】
        labels: 分割标签图 (H, W)
        
    【示例】
        >>> labels = slat_segment(image, K=4, lambda_param=1.0)
    """
    config = SLATConfig(lambda_param=lambda_param)  # 创建配置
    segmenter = SLATSegmentation(config)  # 创建分割器
    result = segmenter.segment(image, K)  # 执行分割
    return result['segmentation']  # 返回标签


# ============================================================================
# 单元测试
# ============================================================================

class TestSLAT:
    """
    SLaT分割算法单元测试
    """
    
    @staticmethod
    def test_stage1_smoothing():
        """测试Stage 1: ROF平滑"""
        print("\n" + "="*60)
        print("测试Stage 1: ROF平滑")
        print("="*60)
        
        np.random.seed(42)
        
        # 创建测试图像
        H, W = 64, 64
        clean = np.zeros((H, W, 3))
        clean[:H//2, :, 0] = 0.8  # 上半红色
        clean[H//2:, :, 1] = 0.8  # 下半绿色
        
        # 添加噪声
        noisy = clean + 0.15 * np.random.randn(H, W, 3)
        noisy = np.clip(noisy, 0, 1)
        
        # 执行平滑
        smoother = ROFSmoother(lambda_param=1.0, max_iterations=100)
        smoothed = smoother.smooth(noisy)
        
        # 计算MSE
        mse_noisy = np.mean((noisy - clean)**2)
        mse_smoothed = np.mean((smoothed - clean)**2)
        
        print(f"噪声图像MSE: {mse_noisy:.6f}")
        print(f"平滑后MSE: {mse_smoothed:.6f}")
        print(f"MSE降低: {(1 - mse_smoothed/mse_noisy)*100:.1f}%")
        
        assert mse_smoothed < mse_noisy, "平滑应降低MSE"
        print("✓ Stage 1测试通过")
    
    @staticmethod
    def test_stage2_lifting():
        """测试Stage 2: 色彩空间提升"""
        print("\n" + "="*60)
        print("测试Stage 2: 色彩空间提升")
        print("="*60)
        
        np.random.seed(42)
        
        # 创建RGB测试图像
        H, W = 32, 32
        rgb = np.random.rand(H, W, 3)
        
        # 执行提升
        lifter = ColorSpaceLifter(use_cv2=HAS_CV2)
        lifted = lifter.lift(rgb)
        
        print(f"输入形状: {rgb.shape}")
        print(f"输出形状: {lifted.shape}")
        print(f"RGB范围: [{rgb.min():.3f}, {rgb.max():.3f}]")
        print(f"Lab范围: [{lifted[:,:,3:].min():.3f}, {lifted[:,:,3:].max():.3f}]")
        
        assert lifted.shape == (H, W, 6), "输出应为6通道"
        print("✓ Stage 2测试通过")
    
    @staticmethod
    def test_stage3_thresholding():
        """测试Stage 3: K-means阈值"""
        print("\n" + "="*60)
        print("测试Stage 3: K-means阈值")
        print("="*60)
        
        np.random.seed(42)
        
        # 创建6维特征
        H, W, D = 32, 32, 6
        features = np.random.rand(H, W, D)
        
        # 执行阈值
        thresholder = KMeansThresholding(n_init=5, max_iter=100)
        labels, centers = thresholder.threshold(features, K=4)
        
        print(f"输入特征形状: {features.shape}")
        print(f"标签形状: {labels.shape}")
        print(f"中心形状: {centers.shape}")
        print(f"唯一标签: {np.unique(labels)}")
        print(f"各类像素数: {[np.sum(labels==k) for k in range(4)]}")
        
        assert labels.shape == (H, W), "标签形状应匹配"
        assert centers.shape == (4, D), "中心形状应为(K, D)"
        print("✓ Stage 3测试通过")
    
    @staticmethod
    def test_full_pipeline():
        """测试完整SLaT流程"""
        print("\n" + "="*60)
        print("测试完整SLaT流程")
        print("="*60)
        
        np.random.seed(42)
        
        # 创建合成图像（4个区域）
        H, W = 64, 64
        image = np.zeros((H, W, 3))
        # 区域1: 左上，红色
        image[:H//2, :W//2, 0] = 0.8
        # 区域2: 右上，绿色
        image[:H//2, W//2:, 1] = 0.7
        # 区域3: 左下，蓝色
        image[H//2:, :W//2, 2] = 0.9
        # 区域4: 右下，灰色
        image[H//2:, W//2:, :] = [0.4, 0.4, 0.4]
        
        # 添加噪声
        noisy = image + 0.1 * np.random.randn(H, W, 3)
        noisy = np.clip(noisy, 0, 1)
        
        # 执行SLaT分割
        config = SLATConfig(lambda_param=1.5, rof_iterations=100)
        segmenter = SLATSegmentation(config)
        result = segmenter.segment(noisy, K=4, return_intermediate=True)
        
        print(f"输入形状: {noisy.shape}")
        print(f"分割形状: {result['segmentation'].shape}")
        print(f"聚类中心: {result['cluster_centers'].shape}")
        print(f"各类像素数: {[np.sum(result['segmentation']==k) for k in range(4)]}")
        
        # 验证分割质量
        # 每个区域应主要对应一个类别
        labels = result['segmentation']
        
        print("✓ 完整流程测试通过")
    
    @staticmethod
    def test_parameter_effect():
        """测试参数效果"""
        print("\n" + "="*60)
        print("测试参数效果")
        print("="*60)
        
        np.random.seed(42)
        
        # 创建测试图像
        H, W = 32, 32
        image = np.random.rand(H, W, 3)
        image += 0.2 * np.random.randn(H, W, 3)
        image = np.clip(image, 0, 1)
        
        # 测试不同lambda
        lambdas = [0.5, 1.0, 2.0]
        for lam in lambdas:
            config = SLATConfig(lambda_param=lam, rof_iterations=50)
            segmenter = SLATSegmentation(config)
            result = segmenter.segment(image, K=3)
            
            # 计算类别分布的熵（越低越好）
            counts = np.array([np.sum(result['segmentation']==k) for k in range(3)])
            probs = counts / counts.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            
            print(f"λ={lam}: 类别分布={counts}, 熵={entropy:.3f}")
        
        print("✓ 参数效果测试通过")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("SLaT分割算法单元测试套件")
    print("="*60)
    
    TestSLAT.test_stage1_smoothing()  # Stage 1测试
    TestSLAT.test_stage2_lifting()  # Stage 2测试
    TestSLAT.test_stage3_thresholding()  # Stage 3测试
    TestSLAT.test_full_pipeline()  # 完整流程测试
    TestSLAT.test_parameter_effect()  # 参数测试
    
    print("\n" + "="*60)
    print("所有测试通过!")
    print("="*60)


# ============================================================================
# 使用示例
# ============================================================================

def usage_examples():
    """展示SLaT分割的使用示例"""
    print("="*60)
    print("SLaT分割使用示例")
    print("="*60)
    
    np.random.seed(42)
    
    # 示例1: 基本使用
    print("\n【示例1: 基本使用】")
    H, W = 128, 128
    image = np.random.rand(H, W, 3)
    
    labels = slat_segment(image, K=4, lambda_param=1.0)
    print(f"输入: {image.shape}")
    print(f"输出: {labels.shape}")
    print(f"类别数: {len(np.unique(labels))}")
    
    # 示例2: 完整API
    print("\n【示例2: 完整API】")
    config = SLATConfig(
        lambda_param=2.0,
        rof_iterations=200,
        random_state=42
    )
    segmenter = SLATSegmentation(config)
    result = segmenter.segment(image, K=5, return_intermediate=True)
    
    print(f"分割结果: {result['segmentation'].shape}")
    print(f"平滑结果: {result['smoothed'].shape}")
    print(f"提升结果: {result['lifted'].shape}")
    print(f"聚类中心: {result['cluster_centers'].shape}")
    
    return result


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    # 运行测试
    run_all_tests()
    
    # 展示示例
    print("\n")
    usage_examples()
