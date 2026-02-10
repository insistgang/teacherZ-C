"""
Mumford-Shah 图像分割模型实现

Mumford-Shah 模型是图像分割领域最重要的变分模型之一，
将分割问题转化为能量最小化问题。

数学模型:
---------
E(u, K) = ∫_{Ω\K} (u - f)² dx + μ ∫_{Ω\K} |∇u|² dx + ν |K|

其中:
- u: 分段平滑的逼近图像
- f: 原始图像
- K: 边缘集合（不连续点）
- μ: 平滑项权重
- ν: 边缘长度惩罚项

这是一个自由不连续问题（free discontinuity problem），
直接求解非常困难。本模块实现了基于逼近和水平集的方法。
"""

import numpy as np
from scipy import ndimage
from .optimization import compute_gradient, compute_divergence


def ms_energy(u, f, edge_set, mu, nu):
    """
    计算 Mumford-Shah 能量函数
    
    E(u, K) = 数据保真项 + 平滑项 + 边缘长度项
    
    参数:
    -----------
    u : ndarray
        逼近图像
    f : ndarray
        原始图像
    edge_set : ndarray
        边缘指示函数（1表示边缘，0表示非边缘）
    mu : float
        平滑项权重
    nu : float
        边缘长度惩罚
        
    返回:
    -----------
    energy : float
        Mumford-Shah 能量值
    """
    # 数据保真项: ∫(u-f)² (在非边缘区域)
    fidelity = np.sum((1 - edge_set) * (u - f)**2)
    
    # 平滑项: μ ∫|∇u|² (在非边缘区域)
    grad_x, grad_y = compute_gradient(u)
    smoothness = mu * np.sum((1 - edge_set) * (grad_x**2 + grad_y**2))
    
    # 边缘长度项: ν |K|
    edge_length = nu * np.sum(edge_set)
    
    return fidelity + smoothness + edge_length


def mumford_shah_segmentation(f, mu=1.0, nu=0.01, max_iter=100, 
                               tol=1e-5, verbose=False):
    """
    Mumford-Shah 分割（近似实现）
    
    使用交替最小化方法求解 M-S 模型：
    1. 固定边缘 K，求解 u（椭圆型 PDE）
    2. 固定 u，更新边缘 K（阈值操作）
    
    这是一个简化的实现，使用阈值检测边缘。
    完整实现需要更复杂的变分方法或相场近似。
    
    参数:
    -----------
    f : ndarray
        输入图像 [0, 1]
    mu : float
        平滑项权重，控制平滑程度
        - 较大值: 更平滑的逼近
        - 较小值: 更接近原图
    nu : float
        边缘长度惩罚，控制分割简洁性
        - 较大值: 更少的边缘（简洁分割）
        - 较小值: 更多的边缘（精细分割）
    max_iter : int
        最大迭代次数
    tol : float
        收敛容差
    verbose : bool
        是否打印进度
        
    返回:
    -----------
    u : ndarray
        分段平滑的逼近图像
    edge_set : ndarray
        检测到的边缘（二值图像）
    energy_history : list
        能量迭代历史
        
    注意:
    -----------
    这是 M-S 模型的简化实现。完整实现需要：
    - Ambrosio-Tortorelli 相场近似
    - 或基于区域的活动轮廓方法
    """
    f = np.asarray(f, dtype=np.float64)
    rows, cols = f.shape
    
    # 初始化
    u = f.copy()
    edge_set = np.zeros_like(f)
    
    energy_history = []
    eps = 1e-8
    
    for i in range(max_iter):
        u_old = u.copy()
        
        # 步骤 1: 固定边缘，求解 u
        # 求解 (1 - K)u - μΔu = (1 - K)f 在非边缘区域
        # 使用简单的迭代方法
        for _ in range(10):  # 内迭代
            # 计算拉普拉斯
            laplacian = compute_laplacian_ms(u)
            
            # 更新 u
            # u = ((1-K)f + μ(Δu + K*u)) / (1-K + μ*K)
            # 简化为: 在非边缘区域平滑，在边缘区域保持
            u = (1 - edge_set) * (f + mu * laplacian) / (1 + 4*mu + eps) + \
                edge_set * f
        
        # 步骤 2: 固定 u，更新边缘
        # 检测 u 和 f 的梯度边缘
        grad_x, grad_y = compute_gradient(u)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 根据梯度幅值和能量最小化更新边缘
        # 边缘出现在梯度较大的地方
        edge_threshold = np.percentile(grad_mag, 90)  # 自适应阈值
        edge_set = (grad_mag > edge_threshold).astype(float)
        
        # 形态学操作清理边缘
        edge_set = ndimage.binary_opening(edge_set > 0, iterations=1)
        edge_set = ndimage.binary_closing(edge_set, iterations=1)
        edge_set = edge_set.astype(float)
        
        # 计算能量
        energy = ms_energy(u, f, edge_set, mu, nu)
        energy_history.append(energy)
        
        # 检查收敛
        rel_change = np.linalg.norm(u - u_old) / (np.linalg.norm(u_old) + eps)
        if rel_change < tol:
            if verbose:
                print(f"收敛于迭代 {i}")
            break
        
        if verbose and i % 10 == 0:
            print(f"迭代 {i}: 能量 = {energy:.6f}, 边缘像素 = {np.sum(edge_set):.0f}")
    
    return u, edge_set, energy_history


def level_set_evolution(f, phi0, mu=0.1, nu=0.0, lambda1=1.0, lambda2=1.0,
                        dt=0.1, max_iter=200, reinit_interval=20,
                        tol=1e-6, verbose=False):
    """
    基于水平集的 Mumford-Shah 分割
    
    使用水平集函数 φ 隐式表示区域：
    - φ > 0: 区域内部
    - φ < 0: 区域外部
    - φ = 0: 区域边界（边缘）
    
    这种方法避免了显式跟踪边界，可以处理拓扑变化（分裂、合并）。
    
    演化方程（简化形式）:
    ∂φ/∂t = δ(φ)[μ·div(∇φ/|∇φ|) - ν - λ1(f-c1)² + λ2(f-c2)²]
    
    其中 c1, c2 是区域内外的平均强度。
    
    参数:
    -----------
    f : ndarray
        输入图像 [0, 1]
    phi0 : ndarray
        初始水平集函数（符号距离函数）
    mu : float
        曲线长度惩罚权重
    nu : float
        区域面积权重（可为负以扩大区域）
    lambda1, lambda2 : float
        数据项权重
    dt : float
        时间步长
    max_iter : int
        最大迭代次数
    reinit_interval : int
        重初始化间隔（保持水平集为符号距离函数）
    tol : float
        收敛容差
    verbose : bool
        是否打印进度
        
    返回:
    -----------
    u : ndarray
        分段常数逼近图像
    phi : ndarray
        最终水平集函数
    segmentation : ndarray
        分割结果（二值图像）
    energy_history : list
        能量迭代历史
        
    示例:
    -----------
    >>> from src.chan_vese import initialize_sdf_circle
    >>> f = load_image('brain.png', gray=True)
    >>> phi0 = initialize_sdf_circle(f.shape, center=(100, 100), radius=50)
    >>> u, phi, seg, hist = level_set_evolution(f, phi0, mu=0.1, max_iter=200)
    """
    f = np.asarray(f, dtype=np.float64)
    phi = phi0.copy()
    
    rows, cols = f.shape
    eps = 1e-8
    energy_history = []
    
    for i in range(max_iter):
        phi_old = phi.copy()
        
        # 计算 Heaviside 函数和 Dirac delta 函数
        H = heaviside_ms(phi, eps)
        delta = dirac_delta_ms(phi, eps)
        
        # 计算区域平均强度
        c1 = np.sum(H * f) / (np.sum(H) + eps)  # 内部平均
        c2 = np.sum((1 - H) * f) / (np.sum(1 - H) + eps)  # 外部平均
        
        # 计算水平集梯度
        phi_x, phi_y = compute_gradient(phi)
        grad_phi = np.sqrt(phi_x**2 + phi_y**2 + eps)
        
        # 计算曲率（边缘长度项的变分）
        curvature = compute_curvature(phi)
        
        # 计算数据项
        data_term = -lambda1 * (f - c1)**2 + lambda2 * (f - c2)**2
        
        # 水平集演化
        # ∂φ/∂t = δ(φ)[μ·curvature + ν + data_term]
        dphi = delta * (mu * curvature + nu + data_term)
        phi = phi + dt * dphi
        
        # 定期重初始化以保持符号距离函数性质
        if i > 0 and i % reinit_interval == 0:
            phi = reinitialize_sdf_ms(phi, iterations=5)
        
        # 计算能量
        edge_length = mu * np.sum(delta * grad_phi)
        region_term = lambda1 * np.sum(H * (f - c1)**2) + \
                      lambda2 * np.sum((1 - H) * (f - c2)**2)
        energy = edge_length + region_term
        energy_history.append(energy)
        
        # 检查收敛
        max_change = np.max(np.abs(phi - phi_old))
        if max_change < tol:
            if verbose:
                print(f"收敛于迭代 {i}")
            break
        
        if verbose and i % 20 == 0:
            print(f"迭代 {i}: 能量 = {energy:.4f}, c1={c1:.3f}, c2={c2:.3f}")
    
    # 生成分割结果
    segmentation = (phi >= 0).astype(float)
    u = c1 * segmentation + c2 * (1 - segmentation)
    
    return u, phi, segmentation, energy_history


def compute_laplacian_ms(u):
    """
    计算拉普拉斯算子（五点差分）
    
    Δu = ∂²u/∂x² + ∂²u/∂y²
    """
    laplacian = np.zeros_like(u)
    laplacian[1:-1, 1:-1] = (
        u[:-2, 1:-1] + u[2:, 1:-1] + 
        u[1:-1, :-2] + u[1:-1, 2:] - 
        4 * u[1:-1, 1:-1]
    )
    return laplacian


def compute_curvature(phi):
    """
    计算水平集函数的曲率
    
    κ = div(∇φ / |∇φ|)
      = (φ_xx·φ_y² - 2·φ_x·φ_y·φ_xy + φ_yy·φ_x²) / (φ_x² + φ_y²)^(3/2)
    """
    eps = 1e-8
    
    # 计算一阶导数
    phi_x, phi_y = compute_gradient(phi)
    
    # 计算二阶导数
    phi_xx, phi_xy = compute_gradient(phi_x)
    phi_yx, phi_yy = compute_gradient(phi_y)
    
    # 曲率公式
    grad_norm_sq = phi_x**2 + phi_y**2 + eps
    curvature = (
        (phi_xx * phi_y**2 - 2 * phi_x * phi_y * phi_xy + phi_yy * phi_x**2) /
        (grad_norm_sq ** 1.5)
    )
    
    return curvature


def heaviside_ms(phi, eps=1e-8):
    """
    平滑 Heaviside 函数
    
    H(z) = 1/2 * [1 + (2/π) * arctan(z/ε)]
    
    当 ε → 0 时，退化为标准 Heaviside 函数
    """
    return 0.5 * (1 + (2 / np.pi) * np.arctan(phi / eps))


def dirac_delta_ms(phi, eps=1e-8):
    """
    平滑 Dirac delta 函数（Heaviside 的导数）
    
    δ(z) = (1/π) * ε / (ε² + z²)
    """
    return (eps / np.pi) / (eps**2 + phi**2)


def reinitialize_sdf_ms(phi, iterations=10):
    """
    重初始化水平集为符号距离函数
    
    求解: ∂φ/∂t = sign(φ₀)(1 - |∇φ|)
    
    这是一个简化版本，使用快速扫描法。
    完整实现可以使用更复杂的方法。
    """
    phi = phi.copy()
    
    for _ in range(iterations):
        # 计算梯度
        phi_x, phi_y = compute_gradient(phi)
        grad_phi = np.sqrt(phi_x**2 + phi_y**2 + 1e-8)
        
        # 符号函数
        sign_phi = np.sign(phi)
        
        # 演化（简化的重初始化）
        phi = phi + 0.1 * sign_phi * (1 - grad_phi)
    
    return phi


def ambrosio_tortorelli_approximation(f, mu=1.0, nu=0.01, epsilon=0.01, 
                                      max_iter=100):
    """
    Ambrosio-Tortorelli 相场近似
    
    这是 Mumford-Shah 模型的椭圆近似，使用相场 v ∈ [0,1] 代替离散边缘：
    
    E(u, v) = ∫(u-f)² dx + μ∫v²|∇u|² dx + ν∫(ε|∇v|² + v²/4ε) dx
    
    当 ε → 0 时，v 趋近于边缘指示函数。
    
    参数:
    -----------
    f : ndarray
        输入图像
    mu : float
        平滑项权重
    nu : float
        边缘惩罚权重
    epsilon : float
        相场宽度参数
    max_iter : int
        最大迭代次数
        
    返回:
    -----------
    u : ndarray
        分段平滑图像
    v : ndarray
        相场（边缘指示，0=边缘, 1=非边缘）
    """
    f = np.asarray(f, dtype=np.float64)
    u = f.copy()
    v = np.ones_like(f)  # 初始时无边缘
    
    for i in range(max_iter):
        # 固定 v，更新 u
        # 求解: u - μ·div(v²∇u) = f
        # 使用简单的显式迭代
        grad_x, grad_y = compute_gradient(u)
        
        # 扩散系数
        diffusion = v**2
        
        # 更新 u（简化版本）
        laplacian = compute_laplacian_ms(u)
        u = (f + mu * diffusion * laplacian) / (1 + 4 * mu * diffusion + 1e-8)
        
        # 固定 u，更新 v
        # 求解相场方程
        grad_u_sq = grad_x**2 + grad_y**2
        
        # 显式更新 v
        v = 4 * epsilon * mu * grad_u_sq / (1 + 4 * epsilon * mu * grad_u_sq + 1e-8)
        v = np.clip(v, 0, 1)
    
    return u, v


if __name__ == "__main__":
    # 测试 Mumford-Shah 实现
    from utils import create_synthetic_image, add_noise
    
    print("测试 Mumford-Shah 分割...")
    
    # 创建测试图像（带边缘）
    clean = create_synthetic_image((128, 128), 'step')
    f = add_noise(clean, 'gaussian', sigma=0.05)
    
    # 测试简化 M-S 分割
    print("\n1. 简化 Mumford-Shah 分割:")
    u, edge, hist = mumford_shah_segmentation(f, mu=0.5, nu=0.01, max_iter=50)
    print(f"   能量: {hist[-1]:.4f}")
    print(f"   边缘像素: {np.sum(edge):.0f}")
    
    # 测试水平集方法
    print("\n2. 水平集演化:")
    # 初始化水平集（圆形）
    rows, cols = f.shape
    phi0 = np.ones((rows, cols))
    center_x, center_y = rows // 2, cols // 2
    radius = min(rows, cols) // 4
    Y, X = np.ogrid[:rows, :cols]
    phi0 = np.sqrt((X - center_y)**2 + (Y - center_x)**2) - radius
    
    u_ls, phi, seg, hist_ls = level_set_evolution(f, phi0, mu=0.1, max_iter=100)
    print(f"   最终能量: {hist_ls[-1]:.4f}")
    print(f"   前景区域: {np.sum(seg):.0f} 像素")
    
    # 测试 A-T 近似
    print("\n3. Ambrosio-Tortorelli 近似:")
    u_at, v = ambrosio_tortorelli_approximation(f, mu=0.5, nu=0.01)
    print(f"   相场范围: [{v.min():.3f}, {v.max():.3f}]")
