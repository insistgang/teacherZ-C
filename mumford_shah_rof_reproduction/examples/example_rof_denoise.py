#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ROF 去噪示例

本示例演示如何使用不同方法实现 ROF (Rudin-Osher-Fatemi) 去噪。

运行方式:
    python example_rof_denoise.py

输出:
    - 显示去噪结果对比图
    - 显示能量收敛曲线
    - 保存结果到 results/ 目录
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    chambolle_rof,
    split_bregman_rof,
    gradient_descent_rof,
    create_synthetic_image,
    add_noise,
    psnr,
    ssim
)


def run_rof_example():
    """运行 ROF 去噪示例"""
    
    print("=" * 60)
    print("ROF 去噪算法示例")
    print("=" * 60)
    
    # 1. 创建测试图像
    print("\n[1/5] 创建测试图像...")
    np.random.seed(42)  # 为了结果可复现
    
    # 使用棋盘图案作为测试图像
    clean_image = create_synthetic_image(
        shape=(256, 256),
        pattern='checkerboard'
    )
    
    # 添加高斯噪声
    noise_sigma = 0.15
    noisy_image = add_noise(clean_image, 'gaussian', sigma=noise_sigma)
    
    print(f"  图像尺寸: {clean_image.shape}")
    print(f"  噪声水平 (sigma): {noise_sigma}")
    print(f"  噪声图像 PSNR: {psnr(clean_image, noisy_image):.2f} dB")
    print(f"  噪声图像 SSIM: {ssim(clean_image, noisy_image):.4f}")
    
    # 2. Chambolle 投影法
    print("\n[2/5] 使用 Chambolle 投影法去噪...")
    lambda_param = 0.5  # 正则化参数
    
    denoised_chambolle, p = chambolle_rof(
        noisy_image,
        lambda_param=lambda_param,
        max_iter=100,
        tol=1e-4
    )
    
    psnr_chambolle = psnr(clean_image, denoised_chambolle)
    ssim_chambolle = ssim(clean_image, denoised_chambolle)
    
    print(f"  参数: lambda = {lambda_param}")
    print(f"  结果 PSNR: {psnr_chambolle:.2f} dB")
    print(f"  结果 SSIM: {ssim_chambolle:.4f}")
    
    # 3. Split Bregman 方法
    print("\n[3/5] 使用 Split Bregman 方法去噪...")
    
    denoised_sb, energy_history_sb = split_bregman_rof(
        noisy_image,
        lambda_param=lambda_param,
        max_iter=20,
        mu=0.1
    )
    
    psnr_sb = psnr(clean_image, denoised_sb)
    ssim_sb = ssim(clean_image, denoised_sb)
    
    print(f"  参数: lambda = {lambda_param}, mu = 0.1")
    print(f"  结果 PSNR: {psnr_sb:.2f} dB")
    print(f"  结果 SSIM: {ssim_sb:.4f}")
    print(f"  迭代次数: {len(energy_history_sb)}")
    
    # 4. 梯度下降法（迭代较少以节省时间）
    print("\n[4/5] 使用梯度下降法去噪...")
    
    denoised_gd, energy_history_gd = gradient_descent_rof(
        noisy_image,
        lambda_param=lambda_param,
        step_size=0.01,
        max_iter=200,
        tol=1e-5
    )
    
    psnr_gd = psnr(clean_image, denoised_gd)
    ssim_gd = ssim(clean_image, denoised_gd)
    
    print(f"  参数: lambda = {lambda_param}, step_size = 0.01")
    print(f"  结果 PSNR: {psnr_gd:.2f} dB")
    print(f"  结果 SSIM: {ssim_gd:.4f}")
    print(f"  迭代次数: {len(energy_history_gd)}")
    
    # 5. 可视化结果
    print("\n[5/5] 生成可视化结果...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # 图像对比
    images = [
        (clean_image, '原始图像'),
        (noisy_image, f'噪声图像\nPSNR: {psnr(clean_image, noisy_image):.2f} dB'),
        (denoised_chambolle, f'Chambolle 方法\nPSNR: {psnr_chambolle:.2f} dB, SSIM: {ssim_chambolle:.4f}'),
        (denoised_sb, f'Split Bregman\nPSNR: {psnr_sb:.2f} dB, SSIM: {ssim_sb:.4f}'),
        (denoised_gd, f'梯度下降法\nPSNR: {psnr_gd:.2f} dB, SSIM: {ssim_gd:.4f}'),
    ]
    
    for i, (img, title) in enumerate(images):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(title, fontsize=11)
        plt.axis('off')
    
    # 能量收敛曲线
    plt.subplot(2, 3, 6)
    plt.plot(energy_history_sb, 'b-', linewidth=2, label='Split Bregman')
    plt.plot(energy_history_gd, 'r--', linewidth=2, label='Gradient Descent')
    plt.xlabel('迭代次数', fontsize=11)
    plt.ylabel('能量值', fontsize=11)
    plt.title('能量收敛曲线', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    save_path = os.path.join(results_dir, 'rof_denoise_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  结果已保存到: {save_path}")
    
    plt.show()
    
    # 6. 性能对比
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    print(f"{'方法':<20} {'PSNR (dB)':<15} {'SSIM':<15} {'推荐度':<10}")
    print("-" * 60)
    print(f"{'Chambolle':<20} {psnr_chambolle:<15.2f} {ssim_chambolle:<15.4f} {'★★★★★':<10}")
    print(f"{'Split Bregman':<20} {psnr_sb:<15.2f} {ssim_sb:<15.4f} {'★★★★☆':<10}")
    print(f"{'Gradient Descent':<20} {psnr_gd:<15.2f} {ssim_gd:<15.4f} {'★★★☆☆':<10}")
    print("=" * 60)
    
    print("\n说明:")
    print("  • Chambolle 方法: 最稳定，推荐日常使用")
    print("  • Split Bregman: 收敛快，适合大规模图像")
    print("  • 梯度下降法: 简单直观，适合学习理解")
    
    return {
        'clean': clean_image,
        'noisy': noisy_image,
        'chambolle': denoised_chambolle,
        'split_bregman': denoised_sb,
        'gradient_descent': denoised_gd
    }


def test_different_lambda():
    """测试不同正则化参数的效果"""
    
    print("\n\n" + "=" * 60)
    print("不同正则化参数效果对比")
    print("=" * 60)
    
    # 创建测试图像
    clean = create_synthetic_image((256, 256), 'step')
    noisy = add_noise(clean, 'gaussian', sigma=0.15)
    
    # 测试不同的 lambda 值
    lambda_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    results = []
    
    print("\n测试参数:")
    for lam in lambda_values:
        denoised, _ = chambolle_rof(noisy, lambda_param=lam, max_iter=50)
        psnr_val = psnr(clean, denoised)
        results.append((lam, denoised, psnr_val))
        print(f"  lambda = {lam:5.1f}: PSNR = {psnr_val:.2f} dB")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(clean, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title('噪声图像')
    axes[0, 1].axis('off')
    
    for i, (lam, denoised, psnr_val) in enumerate(results):
        row = (i + 2) // 3
        col = (i + 2) % 3
        axes[row, col].imshow(denoised, cmap='gray')
        axes[row, col].set_title(f'λ = {lam}, PSNR = {psnr_val:.2f} dB')
        axes[row, col].axis('off')
    
    plt.suptitle('不同正则化参数的去噪效果', fontsize=14)
    plt.tight_layout()
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_path = os.path.join(results_dir, 'rof_lambda_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {save_path}")
    
    plt.show()
    
    print("\n参数选择建议:")
    print("  • λ 较小 (0.1-0.5): 强去噪，适合高噪声图像")
    print("  • λ 中等 (1.0-5.0): 平衡选择，适合一般情况")
    print("  • λ 较大 (> 5.0): 弱去噪，保留更多细节")


if __name__ == "__main__":
    # 运行主示例
    results = run_rof_example()
    
    # 运行参数对比示例
    test_different_lambda()
    
    print("\n\n示例运行完成！")
