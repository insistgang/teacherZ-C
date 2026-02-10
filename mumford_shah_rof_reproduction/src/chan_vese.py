"""
Chan-Vese 主动轮廓模型（无边缘活动轮廓）

Chan-Vese 模型是 Mumford-Shah 模型的一个特例，假设图像是分段常数的。
这是水平集方法在图像分割中最成功的应用之一。

数学模型:
---------
E(c₁, c₂, C) = μ·Length(C) + ν·Area(inside(C))
               + λ₁∫_{inside(C)} |f - c₁|² dx
               + λ₂∫_{outside(C)} |f - c₂|² dx

其中:
- C: 闭合轮廓曲线
- c₁, c₂: 轮廓内外的平均强度
- μ, ν: 正则化参数
- λ₁, λ₂: 数据项权重

水平集形式:
---------
使用水平集函数 φ 隐式表示轮廓 C = {φ = 0}

E(c₁, c₂, φ) = μ∫δ(φ)|∇φ|dx + ν∫H(φ)dx
              + λ₁∫H(φ)|f-c₁|²dx + λ₂∫(1-H(φ))|f-c₂|²dx

演化方程:
---------
∂φ/∂t = δ(φ)[μ·div(∇φ/|∇φ|) - ν - λ₁(f-c₁)² + λ₂(f-c₂)²]

优点:
-----
1. 不依赖图像梯度，适用于弱边界
2. 可以处理拓扑变化（分裂、合并）
3. 对噪声相对鲁棒
"""

import numpy as np
from scipy import ndimage
from .optimization import compute_gradient, compute_divergence


def chan_vese_segmentation(f, phi0, max_iter=200, dt=0.5, mu=0.1, nu=0.0,
                          lambda1=1.0, lambda2=1.0, tol=1e-6, 
                          reinit_interval=5, verbose=False):
    """
    Chan-Vese 图像分割
    
    使用水平集方法最小化 Chan-Vese 能量函数，实现图像分割。
    适用于前景和背景有明显强度差异的图像。
    
    参数:
    -----------
    f : ndarray
        输入图像 [0, 1]
    phi0 : ndarray
        初始水平集函数（符号距离函数）
        - φ > 0: 前景区域（内部）
        - φ < 0: 背景区域（外部）
        - φ = 0: 轮廓边界
    max_iter : int
        最大迭代次数，默认 200
    dt : float
        时间步长，默认 0.5
        - 太大可能导致不稳定
        - 太小收敛慢
    mu : float
        轮廓长度惩罚权重，默认 0.1
        - 较大值: 更平滑的轮廓，更短的长度
        - 较小值: 更复杂的轮廓，更多细节
    nu : float
        区域面积权重，默认 0.0
        - 正值: 倾向于缩小前景区域
        - 负值: 倾向于扩大前景区域
    lambda1 : float
        前景区域数据项权重，默认 1.0
    lambda2 : float
        背景区域数据项权重，默认 1.0
        - 如果 λ1 > λ2: 更强调拟合前景
        - 如果 λ1 < λ2: 更强调拟合背景
    tol : float
        收敛容差，默认 1e-6
    reinit_interval : int
        水平集重初始化间隔，默认 5
        - 定期重初始化保持 |∇φ| ≈ 1
    verbose : bool
        是否打印进度，默认 False
        
    返回:
    -----------
    segmentation : ndarray
        分割结果（二值图像，1=前景，0=背景）
    phi : ndarray
        最终水平集函数
    energy_history : list
        能量迭代历史
        
    示例:
    -----------
    >>> from src.utils import load_image
    >>> from src.chan_vese import initialize_sdf_circle
    >>> 
    >>> f = load_image('cell.png', gray=True)
    >>> phi0 = initialize_sdf_circle(f.shape, center=(100, 100), radius=30)
    >>> seg, phi, history = chan_vese_segmentation(
    ...     f, phi0, max_iter=300, mu=0.2, lambda1=1, lambda2=1
    ... )
    >>> 
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(f, cmap='gray')
    >>> plt.contour(phi, levels=[0], colors='r')
    >>> plt.show()
    
    参考文献:
    -----------
    Chan, T. F., & Vese, L. A. (2001). Active contours without edges. 
    IEEE Transactions on Image Processing, 10(2), 266-277.
    """
    f = np.asarray(f, dtype=np.float64)
    phi = phi0.copy()
    
    rows, cols = f.shape
    eps = 1e-8
    energy_history = []
    
    for i in range(max_iter):
        phi_old = phi.copy()
        
        # 计算平滑的 Heaviside 和 Dirac delta
        H = heaviside(phi)
        delta = dirac_delta(phi)
        
        # 计算区域平均强度 c1 和 c2
        H_sum = np.sum(H) + eps
        H_neg_sum = np.sum(1 - H) + eps
        
        c1 = np.sum(H * f) / H_sum       # 前景平均
        c2 = np.sum((1 - H) * f) / H_neg_sum  # 背景平均
        
        # 计算曲率（边缘长度项）
        curvature = compute_curvature_cv(phi)
        
        # 计算数据项
        # 前景区域数据项: |f - c1|²
        data_inside = (f - c1)**2
        # 背景区域数据项: |f - c2|²
        data_outside = (f - c2)**2
        
        # 水平集演化方程
        # ∂φ/∂t = δ(φ)[μ·κ - ν - λ1(f-c1)² + λ2(f-c2)²]
        dphi = delta * (
            mu * curvature - nu - lambda1 * data_inside + lambda2 * data_outside
        )
        
        # 更新水平集
        phi = phi + dt * dphi
        
        # 定期重初始化
        if i > 0 and i % reinit_interval == 0:
            phi = reinitialize_sdf(phi, iterations=5)
        
        # 计算能量
        # 长度项
        phi_x, phi_y = compute_gradient(phi)
        grad_phi = np.sqrt(phi_x**2 + phi_y**2 + eps)
        length_term = mu * np.sum(delta * grad_phi)
        
        # 面积项
        area_term = nu * np.sum(H)
        
        # 数据项
        fidelity_term = lambda1 * np.sum(H * data_inside) + \
                       lambda2 * np.sum((1 - H) * data_outside)
        
        energy = length_term + area_term + fidelity_term
        energy_history.append(energy)
        
        # 检查收敛
        max_change = np.max(np.abs(phi - phi_old))
        if max_change < tol:
            if verbose:
                print(f"收敛于迭代 {i}")
            break
        
        if verbose and i % 20 == 0:
            print(f"迭代 {i:3d}: 能量={energy:.4f}, c1={c1:.3f}, c2={c2:.3f}, "
                  f"变化={max_change:.6f}")
    
    # 生成分割结果
    segmentation = (phi >= 0).astype(float)
    
    return segmentation, phi, energy_history


def reinitialize_sdf(phi, iterations=10, dt=0.1):
    """
    重初始化符号距离函数（SDF）
    
    在水平集演化过程中，水平集函数会偏离符号距离函数的性质（|∇φ| = 1）。
    定期重初始化可以保持数值稳定性。
    
    求解方程:
    ∂φ/∂t = sign(φ₀)(1 - |∇φ|)
    
    其中 sign(φ₀) 是初始水平集的符号。
    
    参数:
    -----------
    phi : ndarray
        当前水平集函数
    iterations : int
        重初始化迭代次数，默认 10
    dt : float
        时间步长，默认 0.1
        
    返回:
    -----------
    phi_reinit : ndarray
        重初始化后的符号距离函数
        
    注意:
    -----------
    这是一个简化实现。更复杂的方法包括：
    - 快速行进法 (Fast Marching Method)
    - 快速扫描法 (Fast Sweeping Method)
    """
    phi = phi.copy()
    eps = 1e-8
    
    # 计算符号函数（保持零水平集不变）
    sign_phi = phi / np.sqrt(phi**2 + eps)
    
    for _ in range(iterations):
        # 计算梯度
        phi_x, phi_y = compute_gradient(phi)
        grad_phi = np.sqrt(phi_x**2 + phi_y**2 + eps)
        
        # 演化（保持零水平集位置）
        phi = phi + dt * sign_phi * (1 - grad_phi)
    
    return phi


def heaviside(phi, eps=1.0):
    """
    平滑 Heaviside 函数
    
    H_ε(z) = 1/2 * [1 + (2/π) * arctan(z/ε)]
    
    参数:
    -----------
    phi : ndarray
        水平集函数
    eps : float
        平滑参数
        - 较大值: 更平滑的过渡
        - 较小值: 更陡峭的过渡
        
    返回:
    -----------
    H : ndarray
        Heaviside 函数值 [0, 1]
        
    说明:
    -----------
    当 ε → 0 时，退化为标准 Heaviside:
    H(z) = 1 if z ≥ 0
    H(z) = 0 if z < 0
    """
    return 0.5 * (1 + (2 / np.pi) * np.arctan(phi / eps))


def dirac_delta(phi, eps=1.0):
    """
    平滑 Dirac delta 函数
    
    δ_ε(z) = (1/π) * ε / (ε² + z²)
    
    这是平滑 Heaviside 函数的导数：
    d/dz H_ε(z) = δ_ε(z)
    
    参数:
    -----------
    phi : ndarray
        水平集函数
    eps : float
        平滑参数
        
    返回:
    -----------
    delta : ndarray
        Dirac delta 函数值
        
    说明:
    -----------
    Dirac delta 集中在零水平集 φ = 0 附近，
    用于在轮廓附近施加演化力。
    """
    return (eps / np.pi) / (eps**2 + phi**2)


def compute_curvature_cv(phi):
    """
    计算水平集函数的曲率
    
    曲率公式:
    κ = div(∇φ / |∇φ|)
      = (φ_xx·φ_y² - 2·φ_x·φ_y·φ_xy + φ_yy·φ_x²) / (φ_x² + φ_y²)^(3/2)
    
    参数:
    -----------
    phi : ndarray
        水平集函数
        
    返回:
    -----------
    curvature : ndarray
        曲率值
        
    说明:
    -----------
    曲率驱动轮廓向更平滑的方向演化，消除噪声引起的小突起。
    """
    eps = 1e-8
    
    # 一阶导数
    phi_x, phi_y = compute_gradient(phi)
    
    # 二阶导数
    phi_xx, phi_xy = compute_gradient(phi_x)
    phi_yx, phi_yy = compute_gradient(phi_y)
    
    # 曲率
    grad_norm_sq = phi_x**2 + phi_y**2 + eps
    
    curvature = (
        (phi_xx * phi_y**2 - 2 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2) /
        (grad_norm_sq ** 1.5)
    )
    
    return curvature


def initialize_sdf_circle(shape, center=None, radius=None):
    """
    初始化圆形符号距离函数
    
    创建一个以指定点为中心、指定半径的圆的符号距离函数。
    
    参数:
    -----------
    shape : tuple
        图像尺寸 (height, width)
    center : tuple, optional
        圆心 (y, x)，默认为图像中心
    radius : float, optional
        圆半径，默认为图像短边的 1/4
        
    返回:
    -----------
    phi : ndarray
        符号距离函数
        - 圆内: φ < 0
        - 圆外: φ > 0
        - 圆上: φ = 0
        
    示例:
    -----------
    >>> phi = initialize_sdf_circle((256, 256), center=(128, 128), radius=50)
    >>> # 显示零水平集
    >>> import matplotlib.pyplot as plt
    >>> plt.contour(phi, levels=[0])
    """
    rows, cols = shape
    
    if center is None:
        center = (rows // 2, cols // 2)
    if radius is None:
        radius = min(rows, cols) // 4
    
    center_y, center_x = center
    
    # 创建网格
    Y, X = np.ogrid[:rows, :cols]
    
    # 计算到圆心的距离
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    # 符号距离函数: 距离 - 半径
    phi = distance - radius
    
    return phi


def initialize_sdf_rectangle(shape, top_left=None, bottom_right=None):
    """
    初始化矩形符号距离函数
    
    参数:
    -----------
    shape : tuple
        图像尺寸 (height, width)
    top_left : tuple
        左上角 (y, x)
    bottom_right : tuple
        右下角 (y, x)
        
    返回:
    -----------
    phi : ndarray
        符号距离函数
    """
    rows, cols = shape
    
    if top_left is None:
        top_left = (rows // 4, cols // 4)
    if bottom_right is None:
        bottom_right = (3 * rows // 4, 3 * cols // 4)
    
    y1, x1 = top_left
    y2, x2 = bottom_right
    
    Y, X = np.ogrid[:rows, :cols]
    
    # 计算到矩形边界的距离
    dx = np.maximum(np.maximum(x1 - X, 0), X - x2)
    dy = np.maximum(np.maximum(y1 - Y, 0), Y - y2)
    
    phi = np.maximum(dx, dy).astype(float)
    
    # 矩形内部的点设为负值
    inside = (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
    phi[inside] = -np.minimum(np.minimum(X[inside] - x1, x2 - X[inside]),
                              np.minimum(Y[inside] - y1, y2 - Y[inside]))
    
    return phi


def initialize_sdf_multiple_circles(shape, centers, radii):
    """
    初始化多个圆的符号距离函数
    
    适用于多目标分割。
    
    参数:
    -----------
    shape : tuple
        图像尺寸
    centers : list of tuple
        圆心列表 [(y1, x1), (y2, x2), ...]
    radii : list of float
        半径列表 [r1, r2, ...]
        
    返回:
    -----------
    phi : ndarray
        符号距离函数（到最近圆边界的距离）
    """
    rows, cols = shape
    Y, X = np.ogrid[:rows, :cols]
    
    # 初始化距离为无穷大
    phi = np.full(shape, float('inf'))
    
    for center, radius in zip(centers, radii):
        cy, cx = center
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2) - radius
        phi = np.minimum(phi, dist)
    
    return phi


def multiphase_chan_vese(f, phi_list, max_iter=200, mu=0.1, 
                         lambda_list=None, verbose=False):
    """
    多相 Chan-Vese 分割
    
    使用多个水平集函数将图像分割为多个区域。
    
    参数:
    -----------
    f : ndarray
        输入图像
    phi_list : list of ndarray
        初始水平集函数列表
    max_iter : int
        最大迭代次数
    mu : float
        长度正则化权重
    lambda_list : list of float
        各区域数据项权重
    verbose : bool
        是否打印进度
        
    返回:
    -----------
    segmentation : ndarray
        分割标签图（每个像素标记为区域编号）
    phi_list : list
        最终水平集函数列表
    """
    f = np.asarray(f, dtype=np.float64)
    n_phases = len(phi_list)
    
    if lambda_list is None:
        lambda_list = [1.0] * (2**n_phases)
    
    for i in range(max_iter):
        # 计算区域划分
        # 对于 n 个水平集，有 2^n 个区域
        labels = np.zeros_like(f, dtype=int)
        for j, phi in enumerate(phi_list):
            labels = labels + ((phi >= 0).astype(int) << j)
        
        # 计算各区域平均
        c_values = []
        for region_id in range(2**n_phases):
            mask = (labels == region_id)
            count = np.sum(mask)
            if count > 0:
                c_values.append(np.sum(mask * f) / count)
            else:
                c_values.append(0)
        
        # 更新每个水平集
        for j, phi in enumerate(phi_list):
            # 这里简化处理，实际应推导多相演化方程
            # 更新每个水平集时考虑相邻区域
            H = heaviside(phi)
            delta = dirac_delta(phi)
            curvature = compute_curvature_cv(phi)
            
            # 简化的数据项（实际应更复杂）
            # 这里假设两个主要区域
            data_term = np.zeros_like(f)
            
            phi_list[j] = phi + 0.5 * delta * (mu * curvature + data_term)
        
        if verbose and i % 20 == 0:
            print(f"多相迭代 {i}")
    
    segmentation = labels
    return segmentation, phi_list


if __name__ == "__main__":
    # 测试 Chan-Vese 实现
    from utils import create_synthetic_image, add_noise
    
    print("测试 Chan-Vese 分割...")
    
    # 创建测试图像（两个区域）
    rows, cols = 128, 128
    clean = np.zeros((rows, cols))
    clean[32:96, 32:96] = 0.8  # 中心方形区域
    f = add_noise(clean, 'gaussian', sigma=0.1)
    
    # 初始化水平集
    phi0 = initialize_sdf_circle((rows, cols), center=(64, 64), radius=40)
    
    # Chan-Vese 分割
    print("\n运行 Chan-Vese 分割...")
    seg, phi, history = chan_vese_segmentation(
        f, phi0, max_iter=100, mu=0.2, lambda1=1.0, lambda2=1.0, verbose=True
    )
    
    print(f"\n分割完成:")
    print(f"  最终能量: {history[-1]:.4f}")
    print(f"  前景像素: {np.sum(seg):.0f}")
    print(f"  背景像素: {np.sum(1-seg):.0f}")
    
    # 测试重初始化
    print("\n测试水平集重初始化...")
    phi_distorted = phi * (1 + 0.1 * np.random.randn(rows, cols))
    phi_reinit = reinitialize_sdf(phi_distorted, iterations=20)
    
    # 检查 |∇φ| ≈ 1
    phi_x, phi_y = compute_gradient(phi_reinit)
    grad_norm = np.sqrt(phi_x**2 + phi_y**2)
    print(f"  重初始化后 |∇φ| 均值: {np.mean(grad_norm):.4f}")
    print(f"  重初始化后 |∇φ| 标准差: {np.std(grad_norm):.4f}")
