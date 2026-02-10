"""
工具函数模块

提供图像加载、保存、可视化等辅助功能。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union
from pathlib import Path


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    加载图像
    
    支持格式: PNG, JPEG, TIFF, BMP 等常见格式。
    
    参数:
        path: 图像文件路径
        
    返回:
        图像数组，归一化到 [0, 1]
        
    示例:
        >>> image = load_image("data/image.png")
        >>> print(image.shape)
    """
    try:
        from PIL import Image
        img = Image.open(path)
        image = np.array(img).astype(np.float32)
        
        # 归一化到 [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        
        return image
    except Exception as e:
        raise IOError(f"无法加载图像 {path}: {e}")


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    cmap: Optional[str] = None
) -> None:
    """
    保存图像
    
    参数:
        image: 图像数组
        path: 保存路径
        cmap: 如果是单通道，使用的colormap
        
    示例:
        >>> save_image(result, "output/segmentation.png")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        from PIL import Image
        
        # 确保值在 [0, 1] 或 [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # 处理不同维度
        if image.ndim == 2:
            img = Image.fromarray(image, mode='L')
        elif image.ndim == 3:
            if image.shape[2] == 3:
                img = Image.fromarray(image, mode='RGB')
            elif image.shape[2] == 4:
                img = Image.fromarray(image, mode='RGBA')
            else:
                # 多通道，只保存第一通道
                img = Image.fromarray(image[:, :, 0], mode='L')
        else:
            raise ValueError(f"不支持的图像维度: {image.ndim}")
        
        img.save(path)
        
    except Exception as e:
        raise IOError(f"无法保存图像到 {path}: {e}")


def visualize_segmentation(
    image: np.ndarray,
    segmentation: np.ndarray,
    title: str = "Segmentation Result",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    可视化分割结果
    
    显示原图、分割结果、叠加效果。
    
    参数:
        image: 原始图像
        segmentation: 分割结果 (二值图)
        title: 图表标题
        save_path: 保存路径 (可选)
        figsize: 图像大小
        
    示例:
        >>> visualize_segmentation(
        ...     image, segmentation,
        ...     save_path="output/result.png"
        ... )
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 原始图像
    if image.ndim == 2:
        axes[0].imshow(image, cmap='gray')
    else:
        axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 分割结果
    axes[1].imshow(segmentation, cmap='gray')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    # 叠加显示
    overlay = create_overlay(image, segmentation, alpha=0.5)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存: {save_path}")
    
    plt.show()


def create_overlay(
    image: np.ndarray,
    segmentation: np.ndarray,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    创建分割叠加图
    
    将分割结果以半透明颜色叠加到原图上。
    
    参数:
        image: 原始图像 (H, W) 或 (H, W, 3)
        segmentation: 分割掩码 (H, W)
        alpha: 透明度
        color: 叠加颜色 (R, G, B)
        
    返回:
        叠加后的图像 (H, W, 3)
    """
    # 转换为RGB
    if image.ndim == 2:
        image_rgb = np.stack([image] * 3, axis=-1)
    else:
        image_rgb = image.copy()
        if image_rgb.max() <= 1.0:
            image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # 归一化到 [0, 255]
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # 创建叠加
    overlay = image_rgb.copy()
    mask = segmentation > 0
    
    for i, c in enumerate(color):
        overlay[..., i] = np.where(
            mask,
            (1 - alpha) * image_rgb[..., i] + alpha * c,
            image_rgb[..., i]
        ).astype(np.uint8)
    
    return overlay


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    计算IoU (Intersection over Union)
    
    评估分割质量的标准指标。
    
    参数:
        pred: 预测分割图
        target: 真实分割图
        threshold: 二值化阈值
        
    返回:
        IoU值 [0, 1]
        
    示例:
        >>> iou = compute_iou(prediction, ground_truth)
        >>> print(f"IoU: {iou:.4f}")
    """
    pred_binary = (pred > threshold).astype(bool)
    target_binary = (target > threshold).astype(bool)
    
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    计算Dice系数
    
    另一个常用的分割评估指标。
    
    参数:
        pred: 预测分割图
        target: 真实分割图
        threshold: 二值化阈值
        
    返回:
        Dice系数 [0, 1]
    """
    pred_binary = (pred > threshold).astype(bool)
    target_binary = (target > threshold).astype(bool)
    
    intersection = np.logical_and(pred_binary, target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()
    
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(2 * intersection / total)


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    计算多个评估指标
    
    参数:
        pred: 预测分割图
        target: 真实分割图
        threshold: 二值化阈值
        
    返回:
        包含各项指标的字典
    """
    pred_binary = (pred > threshold).astype(bool)
    target_binary = (target > threshold).astype(bool)
    
    # 基本计数
    tp = np.logical_and(pred_binary, target_binary).sum()  # 真阳性
    fp = np.logical_and(pred_binary, ~target_binary).sum()  # 假阳性
    fn = np.logical_and(~pred_binary, target_binary).sum()  # 假阴性
    tn = np.logical_and(~pred_binary, ~target_binary).sum()  # 真阴性
    
    # 指标计算
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # 特异度
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn)
    }


def plot_energy_history(history: dict, save_path: Optional[str] = None) -> None:
    """
    绘制能量收敛曲线
    
    参数:
        history: 包含能量历史的字典
        save_path: 保存路径 (可选)
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    iterations = range(len(history['energy']))
    
    # 总能量
    axes[0, 0].plot(iterations, history['energy'], 'b-', linewidth=2)
    axes[0, 0].set_title('Total Energy')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Energy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 数据项
    if 'data_term' in history:
        axes[0, 1].plot(iterations, history['data_term'], 'r-', linewidth=2)
        axes[0, 1].set_title('Data Term')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Energy')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 框架项
    if 'framelet_term' in history:
        axes[1, 0].plot(iterations, history['framelet_term'], 'g-', linewidth=2)
        axes[1, 0].set_title('Framelet Term')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 形状项
    if 'shape_term' in history:
        axes[1, 1].plot(iterations, history['shape_term'], 'm-', linewidth=2)
        axes[1, 1].set_title('Shape Term')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"能量曲线已保存: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    """
    工具函数测试
    """
    print("="*60)
    print("工具函数测试")
    print("="*60)
    
    # 创建测试数据
    print("\n1. 创建测试数据...")
    test_image = np.random.rand(100, 100)
    test_seg = np.zeros((100, 100))
    test_seg[40:60, 30:70] = 1
    
    # 测试指标计算
    print("\n2. 测试评估指标...")
    metrics = compute_metrics(test_seg, test_seg)
    print(f"   IoU: {metrics['iou']:.4f}")
    print(f"   Dice: {metrics['dice']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    
    # 测试叠加
    print("\n3. 测试图像叠加...")
    overlay = create_overlay(test_image, test_seg, alpha=0.5)
    print(f"   叠加图形状: {overlay.shape}")
    
    # 测试能量历史绘图
    print("\n4. 测试能量曲线绘制...")
    history = {
        'energy': [100, 80, 65, 55, 50, 48, 47, 46.5, 46.2, 46.1],
        'data_term': [50, 45, 42, 40, 39, 38.5, 38.2, 38.1, 38.05, 38.02],
        'framelet_term': [30, 25, 18, 12, 9, 8, 7.5, 7.2, 7.05, 7.0],
        'shape_term': [20, 10, 5, 3, 2, 1.5, 1.3, 1.2, 1.1, 1.08]
    }
    print(f"   能量历史条目: {len(history['energy'])}")
    
    print("\n" + "="*60)
    print("✅ 工具函数测试完成!")
    print("="*60)
