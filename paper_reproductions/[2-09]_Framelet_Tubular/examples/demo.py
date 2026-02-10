"""
小波框架管状结构分割演示

本脚本演示如何使用Framelet Tubular方法进行管状结构分割。
包含完整的流程: 数据准备 → 分割 → 评估 → 可视化

使用方法:
    python examples/demo.py
    
可选参数:
    --image: 输入图像路径 (默认: 生成模拟数据)
    --output: 输出目录 (默认: ./outputs)
    --lambda_framelet: 小波框架权重 (默认: 0.1)
    --lambda_shape: 形状先验权重 (默认: 0.05)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.framelet import FrameletTransform
from src.segmentation import TubularSegmentation
from src.utils import (
    visualize_segmentation,
    compute_metrics,
    plot_energy_history,
    save_image
)


def create_synthetic_tube(
    size: int = 256,
    tube_type: str = 'straight',
    noise_level: float = 0.2
) -> tuple:
    """
    创建合成管状结构图像
    
    参数:
        size: 图像大小
        tube_type: 管状类型 ('straight', 'curved', 'branched')
        noise_level: 噪声水平
        
    返回:
        (image, ground_truth) 元组
    """
    np.random.seed(42)
    
    # 创建空白图像
    image = np.zeros((size, size), dtype=np.float32)
    
    if tube_type == 'straight':
        # 直管
        center_y = size // 2
        width = 10
        image[center_y-width:center_y+width, size//8:7*size//8] = 1.0
        
    elif tube_type == 'curved':
        # S形弯曲管
        x = np.linspace(0, 1, size)
        for i, xi in enumerate(x):
            center_y = int(size / 2 + size / 6 * np.sin(2 * np.pi * xi * 2))
            width = 8
            y_start = max(0, center_y - width)
            y_end = min(size, center_y + width)
            image[y_start:y_end, i] = 1.0
            
    elif tube_type == 'branched':
        # 分支管
        center_y = size // 2
        # 主干
        image[center_y-5:center_y+5, :2*size//3] = 1.0
        # 分支1
        for i in range(2*size//3, size):
            y = int(center_y - (i - 2*size//3) * 0.5)
            if 0 <= y < size:
                image[y-4:y+4, i] = 1.0
        # 分支2
        for i in range(2*size//3, size):
            y = int(center_y + (i - 2*size//3) * 0.5)
            if 0 <= y < size:
                image[y-4:y+4, i] = 1.0
    
    # 保存真实值
    ground_truth = image.copy()
    
    # 添加模糊 (模拟PSF)
    from scipy.ndimage import gaussian_filter
    image = gaussian_filter(image, sigma=2.0)
    
    # 添加噪声
    noise = np.random.randn(size, size) * noise_level
    image = image + noise
    
    # 归一化
    image = (image - image.min()) / (image.max() - image.min())
    
    return image, ground_truth


def demo_framelet_transform(image: np.ndarray):
    """
    演示小波框架变换
    
    参数:
        image: 输入图像
    """
    print("\n" + "="*60)
    print("小波框架变换演示")
    print("="*60)
    
    # 创建框架
    framelet = FrameletTransform(level=2, filter_name='haar')
    
    # 分解
    print("\n1. 执行小波框架分解...")
    coeffs = framelet.decompose(image)
    print(f"   分解层数: {len(coeffs)}")
    print(f"   近似系数形状: {coeffs[0].shape}")
    
    # 重构
    print("\n2. 执行重构...")
    reconstructed = framelet.reconstruct(coeffs)
    error = np.max(np.abs(image - reconstructed[:image.shape[0], :image.shape[1]]))
    print(f"   重构误差: {error:.2e}")
    
    # 特征提取
    print("\n3. 提取小波框架特征...")
    features = framelet.get_framelet_features(image)
    print(f"   近似系数形状: {features['approximation'].shape}")
    print(f"   各层能量: {[f'{e:.2f}' for e in features['energies']]}")
    
    return coeffs, features


def demo_segmentation(
    image: np.ndarray,
    ground_truth: np.ndarray,
    lambda_framelet: float = 0.1,
    lambda_shape: float = 0.05,
    output_dir: Path = None
):
    """
    演示管状结构分割
    
    参数:
        image: 输入图像
        ground_truth: 真实分割图
        lambda_framelet: 小波框架权重
        lambda_shape: 形状先验权重
        output_dir: 输出目录
    """
    print("\n" + "="*60)
    print("管状结构分割演示")
    print("="*60)
    
    print(f"\n参数设置:")
    print(f"   λ_framelet: {lambda_framelet}")
    print(f"   λ_shape: {lambda_shape}")
    
    # 创建分割器
    print("\n1. 创建分割器...")
    segmenter = TubularSegmentation(
        lambda_framelet=lambda_framelet,
        lambda_shape=lambda_shape,
        max_iter=100,
        tol=1e-4
    )
    
    # 执行分割
    print("\n2. 执行分割...")
    segmentation, info = segmenter.segment(image)
    
    print(f"   迭代次数: {info['iterations']}")
    print(f"   最终能量: {info['final_energy']:.4f}")
    
    # 评估
    print("\n3. 评估分割结果...")
    metrics = compute_metrics(segmentation, ground_truth)
    print(f"   IoU:  {metrics['iou']:.4f}")
    print(f"   Dice: {metrics['dice']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    
    # 可视化
    print("\n4. 生成可视化...")
    if output_dir:
        vis_path = output_dir / "segmentation_result.png"
        visualize_segmentation(
            image,
            segmentation,
            title=f"Framelet Tubular Segmentation (IoU={metrics['iou']:.3f})",
            save_path=str(vis_path)
        )
        
        # 能量收敛曲线
        if 'history' in info and info['history']['energy']:
            energy_path = output_dir / "energy_convergence.png"
            plot_energy_history(info['history'], save_path=str(energy_path))
        
        # 保存结果
        save_image(segmentation.astype(float), output_dir / "segmentation.png")
    else:
        visualize_segmentation(image, segmentation)
    
    return segmentation, metrics


def compare_parameters(image: np.ndarray, ground_truth: np.ndarray):
    """
    比较不同参数设置的效果
    
    参数:
        image: 输入图像
        ground_truth: 真实分割图
    """
    print("\n" + "="*60)
    print("参数对比实验")
    print("="*60)
    
    # 不同的参数组合
    param_sets = [
        {'lambda_framelet': 0.05, 'lambda_shape': 0.02},
        {'lambda_framelet': 0.1, 'lambda_shape': 0.05},
        {'lambda_framelet': 0.2, 'lambda_shape': 0.1},
        {'lambda_framelet': 0.0, 'lambda_shape': 0.1},  # 无框架正则化
        {'lambda_framelet': 0.1, 'lambda_shape': 0.0},  # 无形状先验
    ]
    
    results = []
    
    for i, params in enumerate(param_sets):
        print(f"\n参数组合 {i+1}: {params}")
        
        segmenter = TubularSegmentation(
            lambda_framelet=params['lambda_framelet'],
            lambda_shape=params['lambda_shape'],
            max_iter=50,
            tol=1e-3
        )
        
        seg, info = segmenter.segment(image)
        metrics = compute_metrics(seg, ground_truth)
        
        print(f"   IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")
        
        results.append({
            'params': params,
            'segmentation': seg,
            'metrics': metrics,
            'iterations': info['iterations']
        })
    
    # 显示对比结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 原图
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # 各参数结果
    for i, result in enumerate(results):
        axes[i+1].imshow(result['segmentation'], cmap='gray')
        title = f"λ_f={result['params']['lambda_framelet']}, "
        title += f"λ_s={result['params']['lambda_shape']}\n"
        title += f"IoU={result['metrics']['iou']:.3f}"
        axes[i+1].set_title(title)
        axes[i+1].axis('off')
    
    plt.suptitle('Parameter Comparison')
    plt.tight_layout()
    plt.show()
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Framelet Tubular Segmentation Demo'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='输入图像路径 (默认: 生成模拟数据)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./outputs/framelet_tubular',
        help='输出目录'
    )
    parser.add_argument(
        '--lambda_framelet',
        type=float,
        default=0.1,
        help='小波框架权重'
    )
    parser.add_argument(
        '--lambda_shape',
        type=float,
        default=0.05,
        help='形状先验权重'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='运行参数对比实验'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("小波框架管状结构分割演示")
    print("="*70)
    
    # 加载或生成数据
    if args.image and os.path.exists(args.image):
        print(f"\n加载图像: {args.image}")
        from src.utils import load_image
        image = load_image(args.image)
        ground_truth = None
    else:
        print("\n生成模拟管状结构数据...")
        image, ground_truth = create_synthetic_tube(
            size=256,
            tube_type='curved',
            noise_level=0.3
        )
        print(f"   图像大小: {image.shape}")
        print(f"   噪声水平: 0.3")
    
    # 演示小波框架变换
    coeffs, features = demo_framelet_transform(image)
    
    # 执行分割
    segmentation, metrics = demo_segmentation(
        image,
        ground_truth,
        lambda_framelet=args.lambda_framelet,
        lambda_shape=args.lambda_shape,
        output_dir=output_dir
    )
    
    # 参数对比
    if args.compare and ground_truth is not None:
        compare_results = compare_parameters(image, ground_truth)
    
    print("\n" + "="*70)
    print("演示完成!")
    print(f"输出目录: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
