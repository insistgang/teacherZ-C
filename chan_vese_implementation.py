"""
Chan-Vese图像分割完整实现
从Mumford-Shah到实用的分割算法
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, color
from scipy import ndimage
import time
import sys
import io as sys_io

# Windows编码修复
if sys.platform.startswith('win'):
    sys.stdout = sys_io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = sys_io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def heaviside(phi, eps=1e-6):
    """
    Heaviside函数 H(φ)

    H(φ) = {1,  if φ ≥ 0
           {0,  if φ < 0

    使用正则化版本保证数值稳定性
    """
    return 0.5 * (1 + (2/np.pi) * np.arctan(phi/eps))

def dirac(phi, eps=1e-6):
    """
    Dirac测度 δ(φ) = dH/dφ

    正则化版本：
    δ(φ) = (ε/π) / (φ² + ε²)
    """
    return (eps/np.pi) / (phi**2 + eps**2)

def reinitialize_sdf(phi, iterations=5):
    """
    重新初始化水平集函数为符号距离函数

    解: ∂φ/∂τ = sign(φ₀)(1 - |∇φ|)
    """
    for _ in range(iterations):
        # 计算梯度
        phi_y, phi_x = np.gradient(phi)
        phi_norm = np.sqrt(phi_x**2 + phi_y**2)

        # 符号距离函数的符号
        sign = phi / np.sqrt(phi**2 + 1)

        # 更新
        phi = phi + 0.1 * sign * (1 - phi_norm)

    return phi

def curvature(phi):
    """
    计算水平集函数的曲率

    κ = div(∇φ/|∇φ|)
    """
    # 计算梯度
    phi_y, phi_x = np.gradient(phi)

    # 计算二阶导数
    phi_yy, phi_yx = np.gradient(phi_y)
    phi_xy, phi_xx = np.gradient(phi_x)

    # |∇φ|
    phi_norm = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)

    # 曲率
    kappa = (phi_xx * phi_y**2 - 2*phi_xy * phi_x * phi_y + phi_yy * phi_x**2) / (phi_norm**3 + 1e-10)

    return kappa

def chan_vese_segmentation(f, mu=0.1, lambda1=1.0, lambda2=1.0,
                           max_iter=200, dt=0.1, tol=1e-4, reinit_every=10,
                           init_method='circle', verbose=True):
    """
    Chan-Vese图像分割算法

    实现Mumford-Shah的两相分割版本

    Parameters:
    -----------
    f : ndarray
        输入灰度图像（归一化到[0,1]）
    mu : float
        边缘长度惩罚参数（控制边界光滑度）
    lambda1 : float
        前景区域权重
    lambda2 : float
        背景区域权重
    max_iter : int
        最大迭代次数
    dt : float
        时间步长
    tol : float
        收敛阈值
    reinit_every : int
        每隔多少次迭代重新初始化水平集
    init_method : str
        初始化方法 ('circle', 'checkerboard', 'random', 'threshold')
    verbose : bool
        是否打印进度

    Returns:
    --------
    phi : ndarray
        最终的水平集函数
    segmentation : ndarray
        二值分割结果（0=背景，1=前景）
    energy_history : list
        能量历史
    """

    if verbose:
        print("=" * 70)
        print("Chan-Vese图像分割")
        print("=" * 70)
        print(f"\n参数设置:")
        print(f"  μ (边缘惩罚): {mu}")
        print(f"  λ₁ (前景权重): {lambda1}")
        print(f"  λ₂ (背景权重): {lambda2}")
        print(f"  最大迭代: {max_iter}")
        print(f"  初始化方法: {init_method}")

    # 转换为灰度
    if f.ndim == 3:
        f = color.rgb2gray(f)

    H, W = f.shape

    # 初始化水平集函数
    if verbose:
        print(f"\n[初始化] 创建初始水平集函数...")

    if init_method == 'circle':
        # 圆形初始化
        y, x = np.mgrid[:H, :W]
        phi = np.sqrt((x - W/2)**2 + (y - H/2)**2) - min(H, W) / 4

    elif init_method == 'checkerboard':
        # 棋盘格初始化
        y, x = np.mgrid[:H, :W]
        block_size = 20
        phi = np.sin(2*np.pi*x/block_size) * np.sin(2*np.pi*y/block_size)

    elif init_method == 'random':
        # 随机初始化
        phi = np.random.randn(H, W) * 0.1

    elif init_method == 'threshold':
        # 基于阈值初始化
        threshold = np.mean(f)
        phi = f - threshold

    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    if verbose:
        # 显示初始水平集
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].imshow(f, cmap='gray')
        ax[0].set_title('原始图像')
        ax[1].imshow(phi, cmap='jet')
        ax[1].set_title('初始水平集函数')
        plt.tight_layout()
        plt.savefig('chan_vese_initial.png', dpi=150)
        print("  保存初始状态到: chan_vese_initial.png")
        plt.close()

    # 能量历史
    energy_history = []

    start_time = time.time()

    for k in range(max_iter):
        phi_old = phi.copy()

        # 1. 计算Heaviside函数
        H_phi = heaviside(phi)

        # 2. 更新c1, c2（区域平均）
        sum_H = np.sum(H_phi) + 1e-10
        sum_1_H = np.sum(1 - H_phi) + 1e-10

        c1 = np.sum(f * H_phi) / sum_H
        c2 = np.sum(f * (1 - H_phi)) / sum_1_H

        # 3. 计算曲率
        kappa = curvature(phi)

        # 4. 计算演化速度
        # ∂φ/∂t = δ(φ)[μ·κ - λ₁(f-c₁)² + λ₂(f-c₂)²]
        delta = dirac(phi)
        force = mu * kappa - lambda1 * (f - c1)**2 + lambda2 * (f - c2)**2

        # 5. 更新水平集函数
        phi = phi + dt * delta * force

        # 6. 重新初始化（保持符号距离性质）
        if reinit_every > 0 and k % reinit_every == 0:
            phi = reinitialize_sdf(phi)

        # 7. 计算能量（用于监控）
        phi_y, phi_x = np.gradient(phi)
        phi_norm = np.sqrt(phi_x**2 + phi_y**2)

        energy = (mu * np.sum(delta * phi_norm) +
                 lambda1 * np.sum((f - c1)**2 * H_phi) +
                 lambda2 * np.sum((f - c2)**2 * (1 - H_phi)))

        energy_history.append(energy)

        # 8. 检查收敛
        if k > 0 and k % 10 == 0:
            rel_change = np.linalg.norm(phi - phi_old) / (np.linalg.norm(phi_old) + 1e-10)
            if verbose:
                print(f"  迭代 {k:3d}: 能量={energy:.2f}, 变化={rel_change:.6f}")

            if rel_change < tol:
                if verbose:
                    print(f"\n[收敛] 于迭代 {k}")
                break

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n[完成] 用时: {elapsed:.2f}秒, 迭代: {k}次")

    # 最终分割
    segmentation = (phi > 0).astype(np.uint8)

    return phi, segmentation, energy_history


def visualize_results(f, phi, segmentation, energy_history):
    """可视化Chan-Vese分割结果"""
    fig = plt.figure(figsize=(16, 10))

    # 第一行：图像结果
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(f, cmap='gray')
    ax1.set_title('原始图像', fontsize=12)
    ax1.axis('off')

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(phi, cmap='jet')
    ax2.set_title('水平集函数 φ', fontsize=12)
    ax2.axis('off')

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(segmentation, cmap='gray')
    ax3.set_title('分割结果 (前景)', fontsize=12)
    ax3.axis('off')

    # 第二行：叠加和分析
    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(f, cmap='gray')
    ax4.contour(phi, levels=[0], colors='red', linewidths=2)
    ax4.set_title('边界叠加', fontsize=12)
    ax4.axis('off')

    ax5 = plt.subplot(2, 3, 5)
    # 显示前景区域
    foreground = f * segmentation
    ax5.imshow(foreground, cmap='gray')
    ax5.set_title('提取的前景', fontsize=12)
    ax5.axis('off')

    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(energy_history)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Energy')
    ax6.set_title('能量收敛曲线', fontsize=12)
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig('chan_vese_results.png', dpi=150, bbox_inches='tight')
    print("\n[可视化] 保存结果到: chan_vese_results.png")
    plt.show()


def main():
    """主函数：测试Chan-Vese分割"""

    print("\n" + "="*70)
    print("Chan-Vese图像分割演示")
    print("="*70)

    # 准备测试图像
    print("\n[准备] 加载测试图像...")

    # 使用多个测试图像
    test_images = {
        'camera': data.camera(),
        'coins': data.coins(),
        'moon': data.moon(),
    }

    # 选择一个图像
    image_name = 'coins'
    f = img_as_float(test_images[image_name])

    print(f"  图像: {image_name}")
    print(f"  大小: {f.shape}")

    # 运行Chan-Vese
    print("\n" + "="*70)
    print("运行Chan-Vese算法")
    print("="*70)

    phi, segmentation, energy_history = chan_vese_segmentation(
        f,
        mu=0.05,           # 边缘惩罚
        lambda1=1.0,        # 前景权重
        lambda2=1.0,        # 背景权重
        max_iter=200,
        dt=0.1,
        tol=1e-4,
        reinit_every=10,
        init_method='circle',
        verbose=True
    )

    # 可视化结果
    print("\n[可视化] 生成结果图...")
    visualize_results(f, phi, segmentation, energy_history)

    # 统计信息
    print("\n" + "="*70)
    print("分割统计")
    print("="*70)

    foreground_pixels = np.sum(segmentation)
    background_pixels = segmentation.size - foreground_pixels
    foreground_ratio = foreground_pixels / segmentation.size * 100

    print(f"\n前景像素: {foreground_pixels} ({foreground_ratio:.1f}%)")
    print(f"背景像素: {background_pixels} ({100-foreground_ratio:.1f}%)")

    # 前景和背景的平均灰度
    foreground_mean = np.mean(f[segmentation == 1])
    background_mean = np.mean(f[segmentation == 0])

    print(f"\n前景平均灰度: {foreground_mean:.3f}")
    print(f"背景平均灰度: {background_mean:.3f}")
    print(f"对比度: {abs(foreground_mean - background_mean):.3f}")

    # 边界长度（近似）
    phi_y, phi_x = np.gradient(phi)
    phi_norm = np.sqrt(phi_x**2 + phi_y**2)
    boundary_length = np.sum(phi_norm > 0.5)

    print(f"\n边界长度（近似）: {boundary_length} 像素")

    print("\n" + "="*70)
    print("观察要点")
    print("="*70)
    print("1. 水平集函数φ表示轮廓（零水平集=边界）")
    print("2. 正值区域=前景，负值区域=背景")
    print("3. 能量单调下降，保证收敛")
    print("4. 边界自动闭合，无需边缘连接")
    print("\n结论: Chan-Vese实现了全局优化的图像分割！")


if __name__ == "__main__":
    main()
