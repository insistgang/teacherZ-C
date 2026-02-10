#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
算法对比示例

本示例对比 ROF 和 Mumford-Shah 系列算法在不同场景下的表现。

运行方式:
    python example_comparison.py

包含对比:
    1. 不同 ROF 实现方法的性能对比
    2. 不同分割方法的对比
    3. 去噪与分割的联合应用
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    chambolle_rof,
    split_bregman_rof,
    gradient_descent_rof,
    chan_vese_segmentation,
    mumford_shah_segmentation,
    level_set_evolution,
    initialize_sdf_circle,
    create_synthetic_image,
    add_noise,
    psnr,
    ssim
)


def compare_rof_methods():
    """对比不同的 ROF 实现方法"""
    
    print("=" * 70)
    print("对比 1: ROF 实现方法性能对比")
    print("=" * 70)
    
    # 创建测试数据
    print("\n准备测试数据...")
    sizes = [(64, 64), (128, 128), (256, 256)]
    results = []
    
    lambda_param = 0.5
    
    for size in sizes:
        print(f"\n图像尺寸: {size}")
        clean = create_synthetic_image(size, 'checkerboard')
        noisy = add_noise(clean, 'gaussian', sigma=0.1)
        
        # Chambolle 方法
        start = time.time()
        denoised_cham, _ = chambolle_rof(noisy, lambda_param, max_iter=50)
        time_cham = time.time() - start
        psnr_cham = psnr(clean, denoised_cham)
        
        # Split Bregman 方法
        start = time.time()
        denoised_sb, _ = split_bregman_rof(noisy, lambda_param, max_iter=15)
        time_sb = time.time() - start
        psnr_sb = psnr(clean, denoised_sb)
        
        # 梯度下降法（仅对小图像）
        if size[0] <= 128:
            start = time.time()
            denoised_gd, _ = gradient_descent_rof(noisy, lambda_param, max_iter=100)
            time_gd = time.time() - start
            psnr_gd = psnr(clean, denoised_gd)
        else:
            time_gd = float('nan')
            psnr_gd = float('nan')
        
        results.append({
            'size': size,
            'chambolle': {'time': time_cham, 'psnr': psnr_cham},
            'split_bregman': {'time': time_sb, 'psnr': psnr_sb},
            'gradient_descent': {'time': time_gd, 'psnr': psnr_gd}
        })
        
        print(f"  Chambolle:     {time_cham:.3f}s, PSNR: {psnr_cham:.2f} dB")
        print(f"  Split Bregman: {time_sb:.3f}s, PSNR: {psnr_sb:.2f} dB")
        if size[0] <= 128:
            print(f"  Grad Descent:  {time_gd:.3f}s, PSNR: {psnr_gd:.2f} dB")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sizes_labels = [f"{s[0]}x{s[1]}" for s in sizes]
    x = np.arange(len(sizes_labels))
    width = 0.25
    
    # 运行时间对比
    times_cham = [r['chambolle']['time'] for r in results]
    times_sb = [r['split_bregman']['time'] for r in results]
    times_gd = [r['gradient_descent']['time'] for r in results]
    
    axes[0].bar(x - width, times_cham, width, label='Chambolle')
    axes[0].bar(x, times_sb, width, label='Split Bregman')
    axes[0].bar(x + width, times_gd, width, label='Gradient Descent')
    axes[0].set_xlabel('图像尺寸')
    axes[0].set_ylabel('运行时间 (秒)')
    axes[0].set_title('运行时间对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sizes_labels)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # PSNR 对比
    psnr_cham = [r['chambolle']['psnr'] for r in results]
    psnr_sb = [r['split_bregman']['psnr'] for r in results]
    psnr_gd = [r['gradient_descent']['psnr'] for r in results]
    
    axes[1].plot(sizes_labels, psnr_cham, 'o-', linewidth=2, label='Chambolle')
    axes[1].plot(sizes_labels, psnr_sb, 's-', linewidth=2, label='Split Bregman')
    axes[1].plot(sizes_labels, psnr_gd, '^-', linewidth=2, label='Gradient Descent')
    axes[1].set_xlabel('图像尺寸')
    axes[1].set_ylabel('PSNR (dB)')
    axes[1].set_title('去噪质量对比')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'comparison_rof_methods.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {save_path}")
    
    plt.show()
    
    # 打印总结
    print("\n" + "-" * 70)
    print("总结:")
    print("  • Chambolle 方法: 稳定可靠，推荐日常使用")
    print("  • Split Bregman: 收敛最快，适合大规模图像")
    print("  • 梯度下降法: 速度慢，仅供学习参考")
    print("-" * 70)


def compare_segmentation_methods():
    """对比不同的分割方法"""
    
    print("\n\n" + "=" * 70)
    print("对比 2: 分割方法对比")
    print("=" * 70)
    
    # 创建测试图像
    print("\n准备测试图像...")
    rows, cols = 256, 256
    image = np.zeros((rows, cols))
    
    # 前景区域（带孔洞的形状）
    Y, X = np.ogrid[:rows, :cols]
    
    # 主体区域
    mask1 = (X - 128)**2/80**2 + (Y - 128)**2/60**2 <= 1
    image[mask1] = 0.8
    
    # 添加孔洞
    mask2 = (X - 128)**2/30**2 + (Y - 110)**2/20**2 <= 1
    image[mask2] = 0.0
    
    # 添加噪声
    np.random.seed(42)
    image = image + np.random.randn(rows, cols) * 0.05
    image = np.clip(image, 0, 1)
    
    print(f"  图像尺寸: {image.shape}")
    print(f"  测试形状: 带孔洞的椭圆")
    
    # 1. Chan-Vese 分割
    print("\n[1/3] Chan-Vese 分割...")
    phi0 = initialize_sdf_circle(image.shape, radius=70)
    
    start = time.time()
    seg_cv, phi_cv, hist_cv = chan_vese_segmentation(
        image, phi0, max_iter=200, mu=0.15, verbose=False
    )
    time_cv = time.time() - start
    
    print(f"  耗时: {time_cv:.3f}s")
    print(f"  迭代: {len(hist_cv)} 次")
    print(f"  最终能量: {hist_cv[-1]:.4f}")
    
    # 2. 简化 M-S 分割
    print("\n[2/3] Mumford-Shah 简化分割...")
    
    start = time.time()
    u_ms, edges_ms, hist_ms = mumford_shah_segmentation(
        image, mu=0.5, nu=0.01, max_iter=50, verbose=False
    )
    time_ms = time.time() - start
    
    # 从边缘生成分割
    seg_ms = 1 - edges_ms
    
    print(f"  耗时: {time_ms:.3f}s")
    print(f"  边缘像素: {np.sum(edges_ms):.0f}")
    
    # 3. 水平集演化
    print("\n[3/3] 水平集演化分割...")
    
    start = time.time()
    u_ls, phi_ls, seg_ls, hist_ls = level_set_evolution(
        image, phi0, max_iter=150, mu=0.1, verbose=False
    )
    time_ls = time.time() - start
    
    print(f"  耗时: {time_ls:.3f}s")
    print(f"  迭代: {len(hist_ls)} 次")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 第一行: 结果
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].contour(seg_cv, levels=[0.5], colors='r', linewidths=2)
    axes[0, 1].set_title(f'Chan-Vese ({time_cv:.2f}s)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(seg_cv, cmap='gray')
    axes[0, 2].set_title('Chan-Vese 分割')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(edges_ms, cmap='gray')
    axes[1, 0].set_title(f'M-S 边缘 ({time_ms:.2f}s)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(seg_ls, cmap='gray')
    axes[1, 1].set_title(f'水平集分割 ({time_ls:.2f}s)')
    axes[1, 1].axis('off')
    
    # 能量曲线对比
    axes[1, 2].semilogy(hist_cv, 'b-', linewidth=2, label='Chan-Vese')
    axes[1, 2].semilogy(hist_ls, 'r--', linewidth=2, label='Level Set')
    axes[1, 2].set_xlabel('迭代次数')
    axes[1, 2].set_ylabel('能量值 (log)')
    axes[1, 2].set_title('能量收敛对比')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('分割方法对比', fontsize=14)
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_path = os.path.join(results_dir, 'comparison_segmentation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {save_path}")
    
    plt.show()
    
    # 方法特点总结
    print("\n" + "-" * 70)
    print("方法特点总结:")
    print("-" * 70)
    print("Chan-Vese:")
    print("  ✓ 实现简单，效果稳定")
    print("  ✓ 适合前景背景分明的图像")
    print("  ✗ 假设分段常数，不适合渐变图像")
    print("\nMumford-Shah (简化):")
    print("  ✓ 提供平滑逼近")
    print("  ✓ 可同时得到分割和去噪")
    print("  ✗ 简化实现效果有限")
    print("\n水平集演化:")
    print("  ✓ 灵活性高")
    print("  ✓ 可扩展性强")
    print("  ✗ 参数较多，需要调参")
    print("-" * 70)


def denoise_then_segment():
    """去噪后分割的联合应用"""
    
    print("\n\n" + "=" * 70)
    print("对比 3: 去噪+分割联合应用")
    print("=" * 70)
    
    # 创建含噪测试图像
    print("\n准备测试图像...")
    rows, cols = 256, 256
    clean = np.zeros((rows, cols))
    
    # 创建复杂形状
    Y, X = np.ogrid[:rows, :cols]
    
    # 大圆
    mask1 = (X - 100)**2 + (Y - 100)**2 <= 50**2
    clean[mask1] = 0.7
    
    # 旁边的小圆
    mask2 = (X - 170)**2 + (Y - 170)**2 <= 30**2
    clean[mask2] = 0.8
    
    # 添加强噪声
    np.random.seed(123)
    noisy = add_noise(clean, 'gaussian', sigma=0.2)
    
    print(f"  图像尺寸: {noisy.shape}")
    print(f"  噪声水平: sigma=0.2 (强噪声)")
    
    # 方案 1: 直接分割
    print("\n[1/3] 直接对含噪图像分割...")
    phi0 = initialize_sdf_circle(noisy.shape, radius=60)
    seg_direct, _, _ = chan_vese_segmentation(
        noisy, phi0, max_iter=200, mu=0.3, verbose=False
    )
    
    # 方案 2: 去噪后分割
    print("\n[2/3] 先去噪再分割...")
    denoised, _ = chambolle_rof(noisy, lambda_param=0.3, max_iter=50)
    
    phi0 = initialize_sdf_circle(denoised.shape, radius=60)
    seg_denoised, _, _ = chan_vese_segmentation(
        denoised, phi0, max_iter=200, mu=0.15, verbose=False
    )
    
    # 方案 3: 强去噪后分割
    print("\n[3/3] 强去噪后分割...")
    denoised_strong, _ = chambolle_rof(noisy, lambda_param=0.1, max_iter=50)
    
    phi0 = initialize_sdf_circle(denoised_strong.shape, radius=60)
    seg_strong, _, _ = chan_vese_segmentation(
        denoised_strong, phi0, max_iter=200, mu=0.15, verbose=False
    )
    
    # 计算质量评估
    def segmentation_accuracy(seg, ground_truth):
        """简单的分割精度评估"""
        return np.mean(seg == ground_truth)
    
    # 生成参考分割（从无噪声图像）
    phi0 = initialize_sdf_circle(clean.shape, radius=60)
    seg_gt, _, _ = chan_vese_segmentation(
        clean, phi0, max_iter=200, mu=0.15, verbose=False
    )
    
    acc_direct = segmentation_accuracy(seg_direct, seg_gt)
    acc_denoised = segmentation_accuracy(seg_denoised, seg_gt)
    acc_strong = segmentation_accuracy(seg_strong, seg_gt)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(clean, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('含噪图像')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(seg_direct, cmap='gray')
    axes[0, 2].set_title(f'直接分割\n精度: {acc_direct:.3f}')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(denoised, cmap='gray')
    axes[1, 0].set_title('去噪后 (λ=0.3)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(seg_denoised, cmap='gray')
    axes[1, 1].set_title(f'去噪后分割\n精度: {acc_denoised:.3f}')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(seg_strong, cmap='gray')
    axes[1, 2].set_title(f'强去噪分割\n精度: {acc_strong:.3f}')
    axes[1, 2].axis('off')
    
    plt.suptitle('去噪+分割联合应用对比', fontsize=14)
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_path = os.path.join(results_dir, 'comparison_pipeline.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {save_path}")
    
    plt.show()
    
    print("\n" + "-" * 70)
    print("结果分析:")
    print(f"  直接分割精度:    {acc_direct:.3f}")
    print(f"  去噪后分割精度:  {acc_denoised:.3f}")
    print(f"  强去噪分割精度:  {acc_strong:.3f}")
    print("-" * 70)
    print("结论:")
    print("  • 强噪声环境下，先进行去噪预处理能显著提高分割质量")
    print("  • 但过度去噪可能丢失细节，需要根据噪声水平调整参数")
    print("-" * 70)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  算法对比示例 - ROF 与 Mumford-Shah 系列算法".center(56) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 对比 1: ROF 方法
    compare_rof_methods()
    
    # 对比 2: 分割方法
    compare_segmentation_methods()
    
    # 对比 3: 联合应用
    denoise_then_segment()
    
    print("\n\n" + "=" * 70)
    print("所有对比完成！")
    print("=" * 70)
