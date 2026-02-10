"""
ROF (Rudin-Osher-Fatemi) 去噪模型实现

ROF 模型是图像去噪的经典变分方法，通过最小化总变分（Total Variation）
来去除噪声同时保留边缘。

数学模型:
---------
min_u ∫_Ω |∇u| dx + (λ/2) ∫_Ω (u - f)² dx

其中:
- u: 去噪后的图像
- f: 含噪图像
- λ: 正则化参数，控制平滑程度
- |∇u|: 图像的总变分（梯度幅值）

本模块实现了三种求解方法:
1. 梯度下降法: 简单直观，但收敛慢
2. Chambolle 投影法: 高效且数值稳定（推荐）
3. Split Bregman 方法: 更快的收敛速度
"""

import numpy as np
from scipy import ndimage
from .optimization import (
    gradient_descent, 
    chambolle_projection,
    compute_gradient,
    compute_divergence
)


def rof_energy(u, f, lambda_param):
    """
    计算 ROF 能量函数值
    
    E(u) = TV(u) + (λ/2) ||u - f||²
         = ∫|∇u| + (λ/2) ∫(u-f)²
    
    参数:
    -----------
    u : ndarray
        当前估计图像
    f : ndarray
        原始含噪图像
    lambda_param : float
        正则化参数
        
    返回:
    -----------
    energy : float
        ROF 能量值
    """
    # 计算总变分项 TV(u) = ∫|∇u|
    grad_x, grad_y = compute_gradient(u)
    tv_term = np.sum(np.sqrt(grad_x**2 + grad_y**2 + 1e-8))
    
    # 计算数据保真项 (λ/2) ||u - f||²
    fidelity_term = (lambda_param / 2.0) * np.sum((u - f)**2)
    
    return tv_term + fidelity_term


def gradient_descent_rof(f, lambda_param, step_size=0.01, max_iter=500, 
                          tol=1e-6, verbose=False):
    """
    使用梯度下降法求解 ROF 模型
    
    通过梯度下降迭代最小化能量函数。由于 TV 项是非光滑的，
    这里使用近似的次梯度。
    
    迭代公式:
    u^{k+1} = u^k - step_size * ( -div(∇u/|∇u|) + λ(u - f) )
    
    参数:
    -----------
    f : ndarray
        输入含噪图像 [0, 1]
    lambda_param : float
        正则化参数
        - 较小值 (如 0.1): 更强的平滑
        - 较大值 (如 10): 更接近原图
    step_size : float
        梯度下降步长，默认 0.01
    max_iter : int
        最大迭代次数，默认 500
    tol : float
        收敛容差，默认 1e-6
    verbose : bool
        是否打印进度，默认 False
        
    返回:
    -----------
    u : ndarray
        去噪后的图像
    energy_history : list
        能量值迭代历史
        
    注意:
    -----------
    梯度下降法收敛较慢，建议使用 chambolle_rof() 或 split_bregman_rof()
    """
    f = np.asarray(f, dtype=np.float64)
    u = f.copy()  # 初始化为输入图像
    energy_history = []
    
    # 使用小的 epsilon 避免除零
    eps = 1e-8
    
    for i in range(max_iter):
        # 计算能量
        energy = rof_energy(u, f, lambda_param)
        energy_history.append(energy)
        
        # 计算 TV 项的次梯度: -div(∇u / |∇u|)
        grad_x, grad_y = compute_gradient(u)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2 + eps)
        
        # 归一化梯度
        px = grad_x / grad_norm
        py = grad_y / grad_norm
        
        # 计算散度
        div_p = compute_divergence(px, py)
        
        # 总梯度: -div(∇u/|∇u|) + λ(u - f)
        gradient = -div_p + lambda_param * (u - f)
        
        # 梯度下降更新
        u_new = u - step_size * gradient
        
        # 检查收敛
        if i > 0:
            rel_change = np.linalg.norm(u_new - u) / (np.linalg.norm(u) + eps)
            if rel_change < tol:
                if verbose:
                    print(f"收敛于迭代 {i}")
                break
        
        u = u_new
        
        if verbose and i % 50 == 0:
            print(f"迭代 {i}: 能量 = {energy:.6f}")
    
    return u, energy_history


def chambolle_rof(f, lambda_param, max_iter=100, tol=1e-4):
    """
    使用 Chambolle 投影法求解 ROF 模型
    
    这是求解 ROF 模型的经典算法，通过对偶问题高效求解。
    相比梯度下降法，收敛更快且更稳定。
    
    算法原理:
    ---------
    原始问题: min_u TV(u) + (λ/2)||u-f||²
    对偶问题: max_p <f, div(p)> - (1/2λ)||div(p)||², s.t. |p|≤1
    
    其中 p 是对偶变量（向量场）。通过在对偶空间投影迭代求解。
    
    参数:
    -----------
    f : ndarray
        输入含噪图像 [0, 1]
    lambda_param : float
        正则化参数，推荐值:
        - 0.01-0.1: 强去噪，适合高噪声
        - 0.1-1.0: 中等去噪
        - 1.0-10.0: 弱去噪，保留更多细节
    max_iter : int
        最大迭代次数，默认 100
    tol : float
        收敛容差，默认 1e-4
        
    返回:
    -----------
    u : ndarray
        去噪后的图像 [0, 1]
    p : ndarray
        对偶变量，可用于分析边缘信息
        
    参考文献:
    -----------
    Chambolle, A. (2004). An algorithm for total variation minimization 
    and applications. Journal of Mathematical Imaging and Vision, 20(1-2), 89-97.
    
    示例:
    -----------
    >>> from src.utils import add_noise, psnr
    >>> clean = np.random.rand(100, 100)
    >>> noisy = add_noise(clean, 'gaussian', sigma=0.1)
    >>> denoised, _ = chambolle_rof(noisy, lambda_param=0.5)
    >>> print(f"PSNR 提升: {psnr(clean, denoised):.2f} dB")
    """
    f = np.asarray(f, dtype=np.float64)
    
    # 调用 optimization 模块的 Chambolle 投影算法
    u, p = chambolle_projection(f, lambda_param, max_iter, tol)
    
    return u, p


def split_bregman_rof(f, lambda_param, max_iter=20, inner_iter=1, 
                       tol=1e-6, mu=0.1):
    """
    使用 Split Bregman 方法求解 ROF 模型
    
    Split Bregman 方法通过引入辅助变量将原问题分解为更易求解的子问题，
    通常比 Chambolle 方法收敛更快，特别适合大规模问题。
    
    算法原理:
    ---------
    将原问题转化为约束问题:
    min_{u,d} |d| + (λ/2)||u-f||²  s.t. d = ∇u
    
    使用增广拉格朗日方法求解，迭代更新:
    1. u-子问题: 求解线性系统
    2. d-子问题: 软阈值操作
    3. 更新拉格朗日乘子
    
    参数:
    -----------
    f : ndarray
        输入含噪图像 [0, 1]
    lambda_param : float
        正则化参数，控制数据保真程度
    max_iter : int
        外循环最大迭代次数，默认 20
    inner_iter : int
        u-子问题内迭代次数，默认 1
    tol : float
        收敛容差，默认 1e-6
    mu : float
        增广拉格朗日参数，默认 0.1
        
    返回:
    -----------
    u : ndarray
        去噪后的图像
    energy_history : list
        能量值迭代历史
        
    参考文献:
    -----------
    Goldstein, T., & Osher, S. (2009). The split Bregman method for 
    L1-regularized problems. SIAM Journal on Imaging Sciences, 2(2), 323-343.
    """
    f = np.asarray(f, dtype=np.float64)
    rows, cols = f.shape
    
    # 初始化
    u = f.copy()
    d_x = np.zeros_like(f)  # 辅助变量 d = (d_x, d_y)
    d_y = np.zeros_like(f)
    b_x = np.zeros_like(f)  # Bregman 变量 b = (b_x, b_y)
    b_y = np.zeros_like(f)
    
    energy_history = []
    
    for k in range(max_iter):
        # 保存旧的 u 用于收敛检查
        u_old = u.copy()
        
        # 内循环: 求解 u-子问题
        for _ in range(inner_iter):
            # 计算 div(d - b)
            div_d_b = compute_divergence(d_x - b_x, d_y - b_y)
            
            # 求解 u: (λ - μΔ)u = λf - μ·div(d - b)
            # 使用傅里叶变换快速求解
            u = solve_rof_subproblem(f, div_d_b, lambda_param, mu)
        
        # 计算 ∇u
        grad_x, grad_y = compute_gradient(u)
        
        # 更新 d: d = shrink(∇u + b, 1/μ)
        # 软阈值操作
        d_x, d_y = soft_threshold(grad_x + b_x, grad_y + b_y, 1.0/mu)
        
        # 更新 Bregman 变量: b = b + ∇u - d
        b_x = b_x + grad_x - d_x
        b_y = b_y + grad_y - d_y
        
        # 计算能量
        energy = rof_energy(u, f, lambda_param)
        energy_history.append(energy)
        
        # 检查收敛
        rel_change = np.linalg.norm(u - u_old) / (np.linalg.norm(u_old) + 1e-8)
        if rel_change < tol:
            break
    
    return u, energy_history


def solve_rof_subproblem(f, div_d_b, lambda_param, mu):
    """
    使用 FFT 快速求解 ROF 子问题
    
    求解方程: (λ - μΔ)u = λf - μ·div(d-b)
    
    在傅里叶域中，拉普拉斯算子变为乘法:
    F(Δu) = -4π²(ξ_x² + ξ_y²)F(u)
    
    因此可以直接求解:
    F(u) = F(λf - μ·div(d-b)) / (λ + 4π²μ(ξ_x² + ξ_y²))
    """
    # 计算右端项
    rhs = lambda_param * f - mu * div_d_b
    
    # 在傅里叶域求解
    rhs_fft = np.fft.fft2(rhs)
    
    # 构造频率网格
    rows, cols = f.shape
    freq_x = np.fft.fftfreq(cols).reshape(1, -1)
    freq_y = np.fft.fftfreq(rows).reshape(-1, 1)
    
    # 计算分母: λ + 4π²μ(ξ_x² + ξ_y²)
    denom = lambda_param + 4 * np.pi**2 * mu * (freq_x**2 + freq_y**2)
    
    # 傅里叶域求解并逆变换
    u_fft = rhs_fft / denom
    u = np.real(np.fft.ifft2(u_fft))
    
    return u


def soft_threshold(x, y, threshold):
    """
    向量软阈值操作（近端算子）
    
    shrink(x, γ) = x/|x| * max(|x| - γ, 0)
    
    这是 L1 范数的近端算子，用于求解:
    min_d |d| + (1/2γ)||d - x||²
    """
    # 计算模长
    magnitude = np.sqrt(x**2 + y**2)
    
    # 软阈值
    scale = np.maximum(magnitude - threshold, 0) / (magnitude + 1e-8)
    
    return x * scale, y * scale


def adaptive_rof(f, lambda_min=0.01, lambda_max=10.0, num_lambda=20):
    """
    自适应参数选择 ROF 去噪
    
    尝试一系列 λ 值，返回最优结果（基于某种准则）。
    
    参数:
    -----------
    f : ndarray
        输入含噪图像
    lambda_min : float
        最小 λ 值
    lambda_max : float
        最大 λ 值
    num_lambda : int
        尝试的 λ 值数量
        
    返回:
    -----------
    best_u : ndarray
        最优去噪结果
    best_lambda : float
        最优 λ 值
    results : dict
        所有结果的字典
    """
    lambdas = np.logspace(np.log10(lambda_min), np.log10(lambda_max), num_lambda)
    
    results = {}
    best_score = float('inf')
    best_u = None
    best_lambda = None
    
    for lam in lambdas:
        u, _ = chambolle_rof(f, lam, max_iter=50)
        
        # 使用简单启发式准则（可以替换为更复杂的准则）
        # 这里使用能量与保真度的平衡
        energy = rof_energy(u, f, lam)
        score = energy
        
        results[lam] = u
        
        if score < best_score:
            best_score = score
            best_u = u
            best_lambda = lam
    
    return best_u, best_lambda, results


if __name__ == "__main__":
    # 测试 ROF 实现
    from utils import create_synthetic_image, add_noise, psnr, ssim
    
    print("测试 ROF 去噪算法...")
    
    # 创建测试图像
    clean = create_synthetic_image((128, 128), 'checkerboard')
    noisy = add_noise(clean, 'gaussian', sigma=0.1)
    
    print(f"噪声图像 PSNR: {psnr(clean, noisy):.2f} dB")
    print(f"噪声图像 SSIM: {ssim(clean, noisy):.4f}")
    
    # 测试梯度下降法
    print("\n1. 梯度下降法:")
    u_gd, hist_gd = gradient_descent_rof(noisy, 0.5, max_iter=200, verbose=False)
    print(f"   PSNR: {psnr(clean, u_gd):.2f} dB")
    print(f"   SSIM: {ssim(clean, u_gd):.4f}")
    print(f"   迭代次数: {len(hist_gd)}")
    
    # 测试 Chambolle 法
    print("\n2. Chambolle 投影法:")
    u_cham, p = chambolle_rof(noisy, 0.5, max_iter=100)
    print(f"   PSNR: {psnr(clean, u_cham):.2f} dB")
    print(f"   SSIM: {ssim(clean, u_cham):.4f}")
    
    # 测试 Split Bregman 法
    print("\n3. Split Bregman 方法:")
    u_sb, hist_sb = split_bregman_rof(noisy, 0.5, max_iter=20)
    print(f"   PSNR: {psnr(clean, u_sb):.2f} dB")
    print(f"   SSIM: {ssim(clean, u_sb):.4f}")
    print(f"   迭代次数: {len(hist_sb)}")
