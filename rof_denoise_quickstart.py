"""
ROF (Rudin-Osher-Fatemi) 去噪快速实现
基于Chambolle-Pock算法的简化版本

Reference:
- Rudin, Osher, Fatemi (1992): "Nonlinear total variation based noise removal algorithms"
- Chambolle (2004): "An algorithm for total variation denoising and denoising"
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.restoration import denoise_tv_chambolle
import time
import sys
import io as sys_io

# 设置UTF-8编码输出（Windows兼容）
if sys.platform.startswith('win'):
    sys.stdout = sys_io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = sys_io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def add_gaussian_noise(image, sigma=0.1):
    """添加高斯噪声"""
    noisy = image + np.random.normal(0, sigma, image.shape)
    return np.clip(noisy, 0, 1)

def rof_denoise_scikit(noisy, weight=0.1, eps=1e-4):
    """
    使用scikit-image的ROF去噪

    Parameters:
    -----------
    noisy : ndarray
        含噪图像
    weight : float
        正则化参数 λ (lambda)
        值越大，去噪越强，但图像会过度平滑
    eps : float
        收敛阈值

    Returns:
    --------
    denoised : ndarray
        去噪后的图像
    """
    start_time = time.time()
    denoised = denoise_tv_chambolle(noisy, weight=weight, eps=eps)
    elapsed = time.time() - start_time

    print(f"ROF去噪完成: {elapsed:.2f}秒")
    return denoised

def rof_denoise_simple(noisy, lambda_val=0.1, max_iter=100):
    """
    简化版ROF去噪算法（用于理解原理）

    基于梯度下降的TV最小化（简化版，非最优算法）
    只用于演示原理，实际使用请用scikit-image或IPOL的优化算法
    """
    u = noisy.copy()

    for i in range(max_iter):
        # 计算梯度
        grad_x = np.roll(u, -1, axis=1) - u
        grad_y = np.roll(u, -1, axis=0) - u

        # TV范数（简化）
        tv_norm = np.sqrt(grad_x**2 + grad_y**2 + 1e-10)

        # 数据保真项
        fidelity = lambda_val * (u - noisy)

        # 更新
        u_new = u - 0.01 * (fidelity + grad_x / tv_norm)

        # 边界处理
        u_new[:, 0] = u[:, 0]
        u_new[0, :] = u[0, :]

        u = u_new

        if i % 20 == 0:
            print(f"迭代 {i}/{max_iter}")

    return u

def compare_denoising_results(original, noisy, denoised):
    """可视化对比结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    # 含噪图像
    mse_noisy = np.mean((original - noisy)**2)
    axes[1].imshow(noisy, cmap='gray')
    axes[1].set_title(f'含噪图像\nMSE: {mse_noisy:.4f}')
    axes[1].axis('off')

    # 去噪图像
    mse_denoised = np.mean((original - denoised)**2)
    axes[2].imshow(denoised, cmap='gray')
    axes[2].set_title(f'ROF去噪\nMSE: {mse_denoised:.4f}')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('rof_denoise_comparison.png', dpi=150, bbox_inches='tight')
    print("\n结果已保存到: rof_denoise_comparison.png")
    plt.show()

def main():
    """主函数：快速体验ROF去噪"""
    print("=" * 60)
    print("ROF (Rudin-Osher-Fatemi) 去噪快速演示")
    print("=" * 60)

    # 1. 加载或生成测试图像
    print("\n[步骤1] 准备测试图像...")
    try:
        # 尝试加载经典图像
        image_path = "camera_man.png"  # scikit-image的内置图像
        from skimage import data
        image = data.camera()
        image = img_as_float(image)
        print(f"✓ 使用经典测试图像: Camera Man ({image.shape})")
    except:
        # 如果失败，生成合成图像
        print("✗ 加载失败，生成合成测试图像")
        x = np.linspace(-1, 1, 256)
        y = np.linspace(-1, 1, 256)
        X, Y = np.meshgrid(x, y)
        image = 0.5 * (1 + np.sin(3 * np.pi * X) * np.cos(3 * np.pi * Y))
        image = (image - image.min()) / (image.max() - image.min())

    # 2. 添加噪声
    print("\n[步骤2] 添加高斯噪声...")
    sigma = 0.1
    noisy = add_gaussian_noise(image, sigma=sigma)
    print(f"✓ 噪声标准差: {sigma}")

    # 3. ROF去噪
    print("\n[步骤3] 执行ROF去噪...")
    denoised = rof_denoise_scikit(noisy, weight=0.15)

    # 4. 计算PSNR
    mse_original = np.mean((image - noisy)**2)
    mse_denoised = np.mean((image - denoised)**2)
    psnr_original = 10 * np.log10(1.0 / mse_original)
    psnr_denoised = 10 * np.log10(1.0 / mse_denoised)

    print(f"\n[结果对比]")
    print(f"  含噪图像 PSNR: {psnr_original:.2f} dB")
    print(f"  去噪图像 PSNR: {psnr_denoised:.2f} dB")
    print(f"  PSNR提升: {psnr_denoised - psnr_original:.2f} dB")

    # 5. 可视化
    print("\n[步骤4] 生成对比图...")
    compare_denoising_results(image, noisy, denoised)

    print("\n" + "=" * 60)
    print("✓ 演示完成！")
    print("=" * 60)

    # 提示下一步
    print("\n🎯 下一步建议:")
    print("  1. 访问 IPOL 查看完整算法: https://www.ipol.im/pub/art/2013/61/")
    print("  2. 尝试不同的lambda参数观察效果")
    print("  3. 学习Chambolle-Pock算法的数学推导")
    print("  4. 扩展到彩色图像")

if __name__ == "__main__":
    main()
