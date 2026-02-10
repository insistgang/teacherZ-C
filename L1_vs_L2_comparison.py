"""
L1 vs L2 正则化对比实验
直观理解为什么TV(L1)能保持边缘
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage import data
import sys
import io as sys_io

# Windows编码修复
if sys.platform.startswith('win'):
    sys.stdout = sys_io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = sys_io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def create_synthetic_image():
    """创建一个有清晰边缘的合成图像"""
    # 256x256 图像
    image = np.zeros((256, 256))

    # 左半边：黑色 (0)
    image[:, :128] = 0

    # 右半边：白色 (1)
    image[:, 128:] = 1

    # 添加一些细节（小的矩形）
    image[50:100, 20:60] = 0.5
    image[150:200, 180:220] = 0.7

    return image

def add_noise(image, noise_level=0.2):
    """添加高斯噪声"""
    noisy = image + np.random.normal(0, noise_level, image.shape)
    return np.clip(noisy, 0, 1)

def compute_edge_strength(image):
    """计算边缘强度"""
    # 使用Sobel算子计算梯度
    from scipy import ndimage
    gx = ndimage.sobel(image, axis=1)
    gy = ndimage.sobel(image, axis=0)
    gradient = np.sqrt(gx**2 + gy**2)
    return gradient

def main():
    print("=" * 70)
    print("L1 vs L2 正则化对比实验")
    print("=" * 70)

    # 1. 创建合成图像
    print("\n[步骤1] 创建合成图像...")
    clean = create_synthetic_image()

    # 2. 添加噪声
    print("[步骤2] 添加噪声...")
    noisy = add_noise(clean, noise_level=0.15)
    print(f"噪声标准差: 0.15")

    # 3. L1正则化 (TV去噪)
    print("\n[步骤3] L1正则化 (TV去噪)...")
    denoised_l1 = denoise_tv_chambolle(noisy, weight=0.2)

    # 4. L2正则化 (高斯滤波，作为对比)
    print("[步骤4] L2正则化 (高斯滤波，作为对比)...")
    from scipy import ndimage
    denoised_l2 = ndimage.gaussian_filter(noisy, sigma=1.5)

    # 5. 计算边缘强度
    print("\n[步骤5] 计算边缘强度...")
    edge_noisy = compute_edge_strength(noisy)
    edge_l1 = compute_edge_strength(denoised_l1)
    edge_l2 = compute_edge_strength(denoised_l2)

    # 6. 统计信息
    print("\n[结果统计]")
    print(f"原始图像 - 均值: {clean.mean():.3f}, 标准差: {clean.std():.3f}")
    print(f"含噪图像 - 均值: {noisy.mean():.3f}, 标准差: {noisy.std():.3f}")
    print(f"L1去噪   - 均值: {denoised_l1.mean():.3f}, 标准差: {denoised_l1.std():.3f}")
    print(f"L2去噪   - 均值: {denoised_l2.mean():.3f}, 标准差: {denoised_l2.std():.3f}")

    print(f"\n边缘强度统计:")
    print(f"含噪图像 - 最大梯度: {edge_noisy.max():.3f}")
    print(f"L1去噪   - 最大梯度: {edge_l1.max():.3f} (保持: {edge_l1.max()/edge_noisy.max()*100:.1f}%)")
    print(f"L2去噪   - 最大梯度: {edge_l2.max():.3f} (保持: {edge_l2.max()/edge_noisy.max()*100:.1f}%)")

    # 7. 可视化
    print("\n[步骤6] 生成对比图...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    # 第一行：图像对比
    axes[0, 0].imshow(clean, cmap='gray')
    axes[0, 0].set_title('原始图像', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('含噪图像', fontsize=12)
    axes[0, 1].axis('off')

    axes[0, 2].axis('off')

    # 第二行：去噪结果
    axes[1, 0].imshow(denoised_l1, cmap='gray')
    axes[1, 0].set_title('L1正则化 (TV)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(denoised_l2, cmap='gray')
    axes[1, 1].set_title('L2正则化 (高斯)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].axis('off')

    # 第三行：边缘图
    axes[2, 0].imshow(edge_noisy, cmap='hot')
    axes[2, 0].set_title('含噪图像边缘', fontsize=10)
    axes[2, 0].axis('off')

    axes[2, 1].imshow(edge_l1, cmap='hot')
    axes[2, 1].set_title('L1去噪边缘', fontsize=10)
    axes[2, 1].axis('off')

    axes[2, 2].imshow(edge_l2, cmap='hot')
    axes[2, 2].set_title('L2去噪边缘', fontsize=10)
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig('L1_vs_L2_comparison.png', dpi=150, bbox_inches='tight')
    print("结果已保存到: L1_vs_L2_comparison.png")

    # 8. 分析
    print("\n" + "=" * 70)
    print("观察要点:")
    print("=" * 70)
    print("1. L1 (TV): 边缘清晰，噪声去除")
    print("2. L2 (高斯): 边缘模糊，过度平滑")
    print("3. 边缘强度: L1保持更多边缘信息")
    print("\n结论: L1范数更适合图像去噪，因为它能保持边缘！")

if __name__ == "__main__":
    main()
