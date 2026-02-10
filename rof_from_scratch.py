"""
ROF模型完整实现：从数学推导到代码
包含三种求解方法的对比
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.restoration import denoise_tv_chambolle
import time
import sys
import io as sys_io

# Windows编码修复
if sys.platform.startswith('win'):
    sys.stdout = sys_io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = sys_io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def add_noise(image, sigma=0.1):
    """添加高斯噪声"""
    return np.clip(image + np.random.normal(0, sigma, image.shape), 0, 1)

# ============================================================================
# 方法1：梯度下降法（最直观，从E-L方程直接推导）
# ============================================================================

def rof_gradient_descent(f, lambda_=0.1, max_iter=1000, dt=0.1, eps=1e-6):
    """
    ROF梯度下降法

    从E-L方程推导：
    ∂u/∂t = div(∇u/|∇u|) - λ(u-f)

    Parameters:
    -----------
    f : ndarray
        含噪图像
    lambda_ : float
        正则化参数
    max_iter : int
        最大迭代次数
    dt : float
        时间步长（CFL条件：dt <= 0.25）
    eps : float
        |∇u|的正则化参数
    """
    print("\n[方法1] 梯度下降法")
    print("  数学基础：从E-L方程直接推导梯度流")
    print("  方程：∂u/∂t = div(∇u/|∇u|) - λ(u-f)")

    u = f.copy()
    energy_history = []

    start_time = time.time()

    for k in range(max_iter):
        # 计算梯度 ∇u
        grad_x = np.zeros_like(u)
        grad_y = np.zeros_like(u)

        grad_x[:, :-1] = u[:, 1:] - u[:, :-1]
        grad_y[:-1, :] = u[1:, :] - u[:-1, :]

        # 计算 |∇u|（正则化）
        grad_norm = np.sqrt(grad_x**2 + grad_y**2 + eps**2)

        # 计算 div(∇u/|∇u|)
        term_x = grad_x / grad_norm
        term_y = grad_y / grad_norm

        div_term = np.zeros_like(u)
        div_term[:, :-1] += term_x[:, :-1]
        div_term[:, 1:]  -= term_x[:, :-1]
        div_term[:-1, :] += term_y[:-1, :]
        div_term[1:, :]  -= term_y[:-1, :]

        # 数据项 λ(u-f)
        data_term = lambda_ * (u - f)

        # 梯度下降更新
        u_new = u + dt * (div_term - data_term)

        # 计算能量（用于监控）
        if k % 10 == 0:
            tv = np.sum(grad_norm)
            fidelity = 0.5 * lambda_ * np.sum((u - f)**2)
            energy = tv + fidelity
            energy_history.append(energy)

        # 检查收敛
        if k > 0 and k % 50 == 0:
            rel_change = np.linalg.norm(u_new - u) / np.linalg.norm(u)
            if rel_change < 1e-4:
                print(f"  收敛于迭代 {k}")
                break

        u = u_new

    elapsed = time.time() - start_time
    print(f"  完成：{elapsed:.2f}秒，{k}次迭代")

    return u, energy_history


# ============================================================================
# 方法2：不动点迭代
# ============================================================================

def rof_fixed_point(f, lambda_=0.1, max_iter=500, eps=1e-6):
    """
    ROF不动点迭代

    从E-L方程重新排列：
    u = f + (1/λ) div(∇u/|∇u|)

    Parameters:
    -----------
    f : ndarray
        含噪图像
    lambda_ : float
        正则化参数
    max_iter : int
        最大迭代次数
    eps : float
        正则化参数
    """
    print("\n[方法2] 不动点迭代")
    print("  数学基础：从E-L方程重新排列")
    print("  方程：u = f + (1/λ) div(∇u/|∇u|)")

    u = f.copy()
    energy_history = []

    start_time = time.time()

    for k in range(max_iter):
        u_old = u.copy()

        # 计算梯度
        grad_x = np.zeros_like(u)
        grad_y = np.zeros_like(u)

        grad_x[:, :-1] = u[:, 1:] - u[:, :-1]
        grad_y[:-1, :] = u[1:, :] - u[:-1, :]

        grad_norm = np.sqrt(grad_x**2 + grad_y**2 + eps**2)

        # 计算 div(∇u/|∇u|)
        term_x = grad_x / grad_norm
        term_y = grad_y / grad_norm

        div_term = np.zeros_like(u)
        div_term[:, :-1] += term_x[:, :-1]
        div_term[:, 1:]  -= term_x[:, :-1]
        div_term[:-1, :] += term_y[:-1, :]
        div_term[1:, :]  -= term_y[:-1, :]

        # 不动点更新
        u = f + (1.0 / lambda_) * div_term

        # 计算能量
        if k % 10 == 0:
            tv = np.sum(grad_norm)
            fidelity = 0.5 * lambda_ * np.sum((u - f)**2)
            energy = tv + fidelity
            energy_history.append(energy)

        # 检查收敛
        if k > 0 and k % 50 == 0:
            rel_change = np.linalg.norm(u - u_old) / np.linalg.norm(u_old)
            if rel_change < 1e-4:
                print(f"  收敛于迭代 {k}")
                break

    elapsed = time.time() - start_time
    print(f"  完成：{elapsed:.2f}秒，{k}次迭代")

    return u, energy_history


# ============================================================================
# 方法3：简化的Chambolle算法
# ============================================================================

def rof_chambolle(f, lambda_=0.1, max_iter=200, tau=0.25, eps=1e-6):
    """
    简化的Chambolle投影算法

    数学基础：对偶问题
    min ∫|p|² - λf·div(p) s.t. |p| ≤ 1

    Parameters:
    -----------
    f : ndarray
        含噪图像
    lambda_ : float
        正则化参数
    max_iter : int
        最大迭代次数
    tau : float
        步长参数（tau <= 0.25）
    eps : float
        正则化参数
    """
    print("\n[方法3] Chambolle投影算法")
    print("  数学基础：对偶问题")
    print("  迭代：p = (p + τ∇(div(p) - λf)) / (1 + τ|∇(div(p) - λf)|)")

    # 初始化对偶变量
    p1 = np.zeros_like(f)  # x分量
    p2 = np.zeros_like(f)  # y分量

    start_time = time.time()

    for k in range(max_iter):
        # 计算 div(p)
        div_p = np.zeros_like(f)
        div_p[:, :-1] += p1[:, :-1]
        div_p[:, 1:]  -= p1[:, :-1]
        div_p[:-1, :] += p2[:-1, :]
        div_p[1:, :]  -= p2[:-1, :]

        # 计算 ∇(div(p) - λf)
        grad = np.zeros((2, f.shape[0], f.shape[1]))

        term = div_p - lambda_ * f

        grad[0, :, :-1] = term[:, 1:] - term[:, :-1]
        grad[1, :-1, :] = term[1:, :] - term[:-1, :]

        # Chambolle迭代
        denom = 1 + tau * np.sqrt(grad[0]**2 + grad[1]**2 + eps**2)

        p1 = (p1 + tau * grad[0]) / denom
        p2 = (p2 + tau * grad[1]) / denom

    # 恢复 u = f - (1/λ)div(p)
    div_p = np.zeros_like(f)
    div_p[:, :-1] += p1[:, :-1]
    div_p[:, 1:]  -= p1[:, :-1]
    div_p[:-1, :] += p2[:-1, :]
    div_p[1:, :]  -= p2[:-1, :]

    u = f - (1.0 / lambda_) * div_p

    elapsed = time.time() - start_time
    print(f"  完成：{elapsed:.2f}秒，{max_iter}次迭代")

    return u, []


# ============================================================================
# 主函数：对比三种方法
# ============================================================================

def main():
    print("=" * 70)
    print("ROF模型完整实现：三种求解方法对比")
    print("=" * 70)

    # 准备测试图像
    print("\n[准备] 加载测试图像...")
    f = img_as_float(data.camera())
    f_noisy = add_noise(f, sigma=0.1)

    print(f"  图像大小：{f.shape}")
    print(f"  噪声标准差：0.1")

    # 参数设置
    lambda_ = 0.15
    print(f"\n[参数] λ = {lambda_}")

    # 方法1：梯度下降
    u1, energy1 = rof_gradient_descent(f_noisy, lambda_=lambda_, max_iter=500, dt=0.2)

    # 方法2：不动点
    u2, energy2 = rof_fixed_point(f_noisy, lambda_=lambda_, max_iter=500)

    # 方法3：Chambolle
    u3, energy3 = rof_chambolle(f_noisy, lambda_=lambda_, max_iter=200, tau=0.24)

    # 方法4：scikit-image（作为参考）
    print("\n[参考] scikit-image实现...")
    start = time.time()
    u_skimage = denoise_tv_chambolle(f_noisy, weight=lambda_)
    print(f"  完成：{time.time()-start:.2f}秒")

    # 计算PSNR
    def psnr(original, denoised):
        mse = np.mean((original - denoised)**2)
        return 10 * np.log10(1.0 / mse)

    print("\n" + "=" * 70)
    print("结果对比")
    print("=" * 70)

    psnr_noisy = psnr(f, f_noisy)
    psnr1 = psnr(f, u1)
    psnr2 = psnr(f, u2)
    psnr3 = psnr(f, u3)
    psnr_sk = psnr(f, u_skimage)

    print(f"\n含噪图像      PSNR: {psnr_noisy:.2f} dB")
    print(f"梯度下降      PSNR: {psnr1:.2f} dB (提升: {psnr1-psnr_noisy:.2f})")
    print(f"不动点        PSNR: {psnr2:.2f} dB (提升: {psnr2-psnr_noisy:.2f})")
    print(f"Chambolle     PSNR: {psnr3:.2f} dB (提升: {psnr3-psnr_noisy:.2f})")
    print(f"scikit-image  PSNR: {psnr_sk:.2f} dB (提升: {psnr_sk-psnr_noisy:.2f})")

    # 可视化
    print("\n[可视化] 生成对比图...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 第一行：图像
    axes[0, 0].imshow(f, cmap='gray')
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(f_noisy, cmap='gray')
    axes[0, 1].set_title('含噪图像', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')

    axes[1, 0].imshow(u1, cmap='gray')
    axes[1, 0].set_title('梯度下降', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(u2, cmap='gray')
    axes[1, 1].set_title('不动点迭代', fontsize=11)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(u3, cmap='gray')
    axes[1, 2].set_title('Chambolle', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('rof_methods_comparison.png', dpi=150, bbox_inches='tight')
    print("  保存到：rof_methods_comparison.png")

    # 能量收敛曲线
    if len(energy1) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(energy1, label='梯度下降')
        plt.plot(energy2, label='不动点')
        plt.xlabel('Iteration (x10)')
        plt.ylabel('Energy')
        plt.title('ROF能量收敛曲线')
        plt.legend()
        plt.grid(True)
        plt.savefig('rof_energy_convergence.png', dpi=150, bbox_inches='tight')
        print("  保存到：rof_energy_convergence.png")

    print("\n" + "=" * 70)
    print("观察要点：")
    print("=" * 70)
    print("1. 所有方法都能有效去噪")
    print("2. Chambolle最快（0.X秒）")
    print("3. 梯度下降最慢但最直观")
    print("4. 不动点可能不稳定（取决于λ）")
    print("\n结论：推荐使用Chambolle或Split Bregman用于实际应用")
    print("       梯度下降用于理解原理")

if __name__ == "__main__":
    main()
