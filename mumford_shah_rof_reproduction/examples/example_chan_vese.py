#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chan-Vese 图像分割示例

本示例演示如何使用 Chan-Vese 模型进行图像分割。
Chan-Vese 模型是一种基于水平集的活动轮廓方法，适用于前景背景
有明显强度差异的图像。

运行方式:
    python example_chan_vese.py

输出:
    - 显示分割过程动画帧
    - 显示最终分割结果
    - 显示能量收敛曲线
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    chan_vese_segmentation,
    initialize_sdf_circle,
    initialize_sdf_rectangle,
    initialize_sdf_multiple_circles,
    create_synthetic_image
)


def example_basic_segmentation():
    """基础分割示例 - 简单几何形状"""
    
    print("=" * 60)
    print("示例 1: 基础 Chan-Vese 分割")
    print("=" * 60)
    
    # 创建测试图像（中心有方形区域）
    print("\n[1/4] 创建测试图像...")
    rows, cols = 256, 256
    image = np.zeros((rows, cols))
    # 中心方形区域，强度为 0.8
    image[64:192, 64:192] = 0.8
    # 添加一些噪声
    np.random.seed(42)
    image = image + np.random.randn(rows, cols) * 0.05
    image = np.clip(image, 0, 1)
    
    print(f"  图像尺寸: {image.shape}")
    print(f"  前景区域: 中心 128x128 方形")
    
    # 初始化水平集（圆形）
    print("\n[2/4] 初始化水平集...")
    phi0 = initialize_sdf_circle(
        image.shape,
        center=(128, 128),  # 图像中心
        radius=80           # 初始半径
    )
    print(f"  初始化: 圆形，中心 (128, 128)，半径 80")
    
    # 运行 Chan-Vese 分割
    print("\n[3/4] 运行 Chan-Vese 分割...")
    segmentation, phi, history = chan_vese_segmentation(
        image,
        phi0,
        max_iter=200,
        dt=0.5,
        mu=0.2,      # 轮廓长度权重
        nu=0.0,      # 区域面积权重
        lambda1=1.0, # 前景数据权重
        lambda2=1.0, # 背景数据权重
        tol=1e-6,
        reinit_interval=5,
        verbose=True
    )
    
    print(f"\n  分割完成!")
    print(f"  迭代次数: {len(history)}")
    print(f"  最终能量: {history[-1]:.4f}")
    print(f"  前景像素: {np.sum(segmentation):.0f} ({100*np.sum(segmentation)/segmentation.size:.1f}%)")
    
    # 可视化结果
    print("\n[4/4] 生成可视化结果...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原始图像
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    # 初始水平集
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].contour(phi0, levels=[0], colors='r', linewidths=2)
    axes[0, 1].set_title('初始轮廓')
    axes[0, 1].axis('off')
    
    # 最终水平集
    axes[0, 2].imshow(image, cmap='gray')
    axes[0, 2].contour(phi, levels=[0], colors='r', linewidths=2)
    axes[0, 2].set_title('最终轮廓')
    axes[0, 2].axis('off')
    
    # 分割结果
    axes[1, 0].imshow(segmentation, cmap='gray')
    axes[1, 0].set_title('分割结果')
    axes[1, 0].axis('off')
    
    # 叠加显示
    axes[1, 1].imshow(image, cmap='gray')
    axes[1, 1].contour(segmentation, levels=[0.5], colors='r', linewidths=2)
    axes[1, 1].set_title('叠加显示')
    axes[1, 1].axis('off')
    
    # 能量收敛
    axes[1, 2].plot(history, 'b-', linewidth=2)
    axes[1, 2].set_xlabel('迭代次数')
    axes[1, 2].set_ylabel('能量值')
    axes[1, 2].set_title('能量收敛曲线')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Chan-Vese 图像分割示例', fontsize=14)
    plt.tight_layout()
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'chan_vese_basic.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  结果已保存到: {save_path}")
    
    plt.show()
    
    return segmentation, phi, history


def example_multiple_objects():
    """多目标分割示例"""
    
    print("\n\n" + "=" * 60)
    print("示例 2: 多目标分割")
    print("=" * 60)
    
    # 创建包含多个圆形区域的图像
    print("\n[1/3] 创建测试图像...")
    rows, cols = 256, 256
    image = np.zeros((rows, cols))
    Y, X = np.ogrid[:rows, :cols]
    
    # 添加三个圆形区域
    circles = [
        ((64, 64), 30, 0.8),    # 左上
        ((192, 64), 25, 0.7),   # 右上
        ((128, 192), 35, 0.9),  # 下方
    ]
    
    for (cy, cx), r, intensity in circles:
        mask = (X - cx)**2 + (Y - cy)**2 <= r**2
        image[mask] = intensity
    
    # 添加噪声
    np.random.seed(123)
    image = image + np.random.randn(rows, cols) * 0.03
    image = np.clip(image, 0, 1)
    
    print(f"  图像中包含 {len(circles)} 个目标")
    
    # 初始化多个水平集
    print("\n[2/3] 初始化水平集...")
    centers = [(64, 64), (192, 64), (128, 192)]
    radii = [35, 30, 40]
    phi0 = initialize_sdf_multiple_circles(
        image.shape,
        centers=centers,
        radii=radii
    )
    print(f"  初始化: 三个圆形水平集")
    
    # 运行分割
    print("\n[3/3] 运行分割...")
    segmentation, phi, history = chan_vese_segmentation(
        image,
        phi0,
        max_iter=300,
        dt=0.5,
        mu=0.15,     # 稍小的 mu 以保留多个目标
        lambda1=1.0,
        lambda2=1.0,
        verbose=True
    )
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    axes[1].imshow(image, cmap='gray')
    axes[1].contour(phi0, levels=[0], colors='b', linewidths=2, linestyles='dashed')
    axes[1].contour(phi, levels=[0], colors='r', linewidths=2)
    axes[1].set_title('初始(蓝虚线) vs 最终(红实线)')
    axes[1].axis('off')
    
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(segmentation, alpha=0.3, cmap='Reds')
    axes[2].contour(segmentation, levels=[0.5], colors='r', linewidths=2)
    axes[2].set_title('分割结果')
    axes[2].axis('off')
    
    plt.suptitle('多目标分割示例', fontsize=14)
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_path = os.path.join(results_dir, 'chan_vese_multiple.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {save_path}")
    
    plt.show()
    
    return segmentation


def example_parameter_effects():
    """展示不同参数对分割结果的影响"""
    
    print("\n\n" + "=" * 60)
    print("示例 3: 参数影响分析")
    print("=" * 60)
    
    # 创建测试图像（有噪声的圆形）
    rows, cols = 200, 200
    image = np.zeros((rows, cols))
    Y, X = np.ogrid[:rows, :cols]
    center_y, center_x = 100, 100
    radius = 50
    mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
    image[mask] = 0.8
    
    # 添加噪声
    np.random.seed(456)
    image = image + np.random.randn(rows, cols) * 0.1
    image = np.clip(image, 0, 1)
    
    # 不同参数组合
    params = [
        (0.05, "mu=0.05 (弱平滑)"),
        (0.2, "mu=0.2 (标准)"),
        (0.5, "mu=0.5 (强平滑)"),
    ]
    
    print("\n测试不同 mu 参数的影响...")
    results = []
    for mu, label in params:
        phi0 = initialize_sdf_circle(image.shape, radius=60)
        seg, phi, hist = chan_vese_segmentation(
            image, phi0, max_iter=150, mu=mu, verbose=False
        )
        results.append((mu, seg, phi, label))
        print(f"  {label}: 迭代 {len(hist)} 次")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('原始图像')
    axes[0, 0].axis('off')
    
    for i, (mu, seg, phi, label) in enumerate(results):
        row = (i + 1) // 2
        col = (i + 1) % 2
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].contour(phi, levels=[0], colors='r', linewidths=2)
        axes[row, col].set_title(label)
        axes[row, col].axis('off')
    
    plt.suptitle('mu 参数对分割结果的影响', fontsize=14)
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_path = os.path.join(results_dir, 'chan_vese_parameters.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {save_path}")
    
    plt.show()
    
    print("\n参数说明:")
    print("  • mu 较小: 允许更复杂的轮廓，可能受噪声影响")
    print("  • mu 较大: 轮廓更平滑，可能丢失细节")
    print("  • 需要根据具体应用调整")


def example_initialization_comparison():
    """比较不同初始化策略"""
    
    print("\n\n" + "=" * 60)
    print("示例 4: 初始化策略比较")
    print("=" * 60)
    
    # 创建测试图像
    rows, cols = 200, 200
    image = create_synthetic_image((rows, cols), 'circles')
    
    # 不同的初始化
    initializers = [
        ("小圆形", initialize_sdf_circle(image.shape, center=(100, 100), radius=30)),
        ("大圆形", initialize_sdf_circle(image.shape, center=(100, 100), radius=80)),
        ("矩形", initialize_sdf_rectangle(image.shape, 
                                          top_left=(50, 50), 
                                          bottom_right=(150, 150))),
    ]
    
    print("\n测试不同初始化策略...")
    
    fig, axes = plt.subplots(2, len(initializers), figsize=(15, 10))
    
    for i, (name, phi0) in enumerate(initializers):
        print(f"  初始化: {name}")
        
        seg, phi, hist = chan_vese_segmentation(
            image, phi0, max_iter=150, mu=0.1, verbose=False
        )
        
        # 显示初始轮廓
        axes[0, i].imshow(image, cmap='gray')
        axes[0, i].contour(initializers[i][1], levels=[0], colors='r', linewidths=2)
        axes[0, i].set_title(f'{name} (初始)')
        axes[0, i].axis('off')
        
        # 显示最终轮廓
        axes[1, i].imshow(image, cmap='gray')
        axes[1, i].contour(phi, levels=[0], colors='r', linewidths=2)
        axes[1, i].set_title(f'{name} (迭代 {len(hist)} 次)')
        axes[1, i].axis('off')
    
    plt.suptitle('不同初始化策略的比较', fontsize=14)
    plt.tight_layout()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    save_path = os.path.join(results_dir, 'chan_vese_initialization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n结果已保存到: {save_path}")
    
    plt.show()
    
    print("\n初始化建议:")
    print("  • 初始轮廓应包含目标区域")
    print("  • 过大可能包含过多背景")
    print("  • 过小可能需要更多迭代")


if __name__ == "__main__":
    # 运行所有示例
    print("Chan-Vese 图像分割示例")
    print("=" * 60)
    print("本示例包含以下内容：")
    print("  1. 基础分割示例")
    print("  2. 多目标分割")
    print("  3. 参数影响分析")
    print("  4. 初始化策略比较")
    print("=" * 60)
    
    # 示例 1: 基础分割
    seg1, phi1, hist1 = example_basic_segmentation()
    
    # 示例 2: 多目标分割
    seg2 = example_multiple_objects()
    
    # 示例 3: 参数影响
    example_parameter_effects()
    
    # 示例 4: 初始化比较
    example_initialization_comparison()
    
    print("\n\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
