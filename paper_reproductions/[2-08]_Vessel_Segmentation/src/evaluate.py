"""
评估指标模块

实现血管分割常用的评估指标:
- AUC-ROC (受试者工作特征曲线下面积)
- 准确率 (Accuracy)
- 灵敏度 (Sensitivity/Recall)
- 特异度 (Specificity)
- Dice系数
- F1分数

这些指标是医学图像分割的标准评估方法。
"""

import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings


def compute_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_label: int = 1
) -> float:
    """
    计算AUC-ROC (受试者工作特征曲线下面积)
    
    AUC是血管分割中最重要的评估指标之一，衡量模型区分
    血管和非血管像素的能力。
    
    参数:
        y_true: 真实标签 (0或1)
        y_score: 预测概率或分数
        pos_label: 正类标签
        
    返回:
        AUC值 [0, 1]，越接近1越好
        
    示例:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_score = np.array([0.1, 0.4, 0.35, 0.8])
        >>> auc = compute_auc(y_true, y_score)
        >>> print(f"AUC: {auc:.4f}")
    """
    # 展平数组
    y_true_flat = y_true.flatten()
    y_score_flat = y_score.flatten()
    
    # 去除无效值
    valid_mask = ~(np.isnan(y_score_flat) | np.isinf(y_score_flat))
    y_true_flat = y_true_flat[valid_mask]
    y_score_flat = y_score_flat[valid_mask]
    
    if len(np.unique(y_true_flat)) < 2:
        # 只有一类，无法计算AUC
        return 0.5
    
    fpr, tpr, _ = roc_curve(y_true_flat, y_score_flat, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)
    
    return float(roc_auc)


def compute_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    计算准确率 (Accuracy)
    
    正确预测的像素比例。
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率或标签
        threshold: 二值化阈值
        
    返回:
        准确率 [0, 1]
    """
    # 二值化预测
    if y_pred.max() <= 1.0 and y_pred.min() >= 0.0 and not np.array_equal(y_pred, y_pred.astype(bool)):
        y_pred_binary = (y_pred > threshold).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    y_true_flat = y_true.flatten().astype(int)
    y_pred_flat = y_pred_binary.flatten()
    
    correct = (y_true_flat == y_pred_flat).sum()
    total = len(y_true_flat)
    
    return float(correct / total) if total > 0 else 0.0


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    计算灵敏度和特异度
    
    灵敏度 (Sensitivity/Recall): 正确识别的血管像素比例
    特异度 (Specificity): 正确识别的非血管像素比例
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率
        threshold: 二值化阈值
        
    返回:
        (灵敏度, 特异度) 元组
        
    示例:
        >>> sens, spec = compute_sensitivity_specificity(y_true, y_pred)
        >>> print(f"灵敏度: {sens:.4f}, 特异度: {spec:.4f}")
    """
    # 二值化
    y_pred_binary = (y_pred > threshold).astype(int)
    
    y_true_flat = y_true.flatten().astype(int)
    y_pred_flat = y_pred_binary.flatten()
    
    # 计算TP, FP, TN, FN
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    
    # 灵敏度 (召回率)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 特异度
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return float(sensitivity), float(specificity)


def compute_dice(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    计算Dice系数
    
    Dice = 2|X∩Y| / (|X| + |Y|)
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率
        threshold: 二值化阈值
        
    返回:
        Dice系数 [0, 1]
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    
    y_true_flat = y_true.flatten().astype(int)
    y_pred_flat = y_pred_binary.flatten()
    
    intersection = np.sum(y_true_flat * y_pred_flat)
    total = np.sum(y_true_flat) + np.sum(y_pred_flat)
    
    return float(2 * intersection / total) if total > 0 else 0.0


def compute_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    计算F1分数
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率
        threshold: 二值化阈值
        
    返回:
        F1分数 [0, 1]
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    
    y_true_flat = y_true.flatten().astype(int)
    y_pred_flat = y_pred_binary.flatten()
    
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return float(2 * precision * recall / (precision + recall))


def compute_precision_recall(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算精确率-召回率曲线
    
    参数:
        y_true: 真实标签
        y_score: 预测分数
        
    返回:
        (精确率, 召回率, 阈值) 元组
    """
    y_true_flat = y_true.flatten()
    y_score_flat = y_score.flatten()
    
    precision, recall, thresholds = precision_recall_curve(y_true_flat, y_score_flat)
    
    return precision, recall, thresholds


def compute_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    计算IoU (Intersection over Union)
    
    也称为Jaccard指数。
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率
        threshold: 二值化阈值
        
    返回:
        IoU [0, 1]
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    
    y_true_flat = y_true.flatten().astype(int)
    y_pred_flat = y_pred_binary.flatten()
    
    intersection = np.sum(y_true_flat & y_pred_flat)
    union = np.sum(y_true_flat | y_pred_flat)
    
    return float(intersection / union) if union > 0 else 0.0


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    apply_fov_mask: bool = True
) -> Dict[str, float]:
    """
    计算所有评估指标
    
    参数:
        y_true: 真实标签
        y_pred: 预测概率或标签
        y_score: 预测概率 (用于计算AUC，如果y_pred已二值化)
        threshold: 二值化阈值
        apply_fov_mask: 是否只计算FOV区域内
        
    返回:
        包含所有指标的字典
    """
    # 应用FOV掩码
    if apply_fov_mask:
        # 假设FOV掩码是y_true中非255的区域
        fov_mask = (y_true != 255) if y_true.max() > 1 else np.ones_like(y_true, dtype=bool)
        y_true = y_true[fov_mask]
        y_pred = y_pred[fov_mask]
        if y_score is not None:
            y_score = y_score[fov_mask]
    
    # 确保y_true是二值
    y_true_binary = (y_true > 0).astype(int)
    
    # 用于AUC的分数
    if y_score is None:
        y_score = y_pred
    
    # 计算各项指标
    metrics = {
        'auc': compute_auc(y_true_binary, y_score),
        'accuracy': compute_accuracy(y_true_binary, y_pred, threshold),
        'dice': compute_dice(y_true_binary, y_pred, threshold),
        'f1': compute_f1_score(y_true_binary, y_pred, threshold),
        'iou': compute_iou(y_true_binary, y_pred, threshold),
    }
    
    # 灵敏度和特异度
    sens, spec = compute_sensitivity_specificity(y_true_binary, y_pred, threshold)
    metrics['sensitivity'] = sens
    metrics['specificity'] = spec
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "评估结果"):
    """
    格式化打印评估指标
    
    参数:
        metrics: 指标字典
        title: 标题
    """
    print("\n" + "="*50)
    print(f"{title}")
    print("="*50)
    print(f"  AUC:        {metrics['auc']:.4f}")
    print(f"  准确率:     {metrics['accuracy']:.4f}")
    print(f"  灵敏度:     {metrics['sensitivity']:.4f}")
    print(f"  特异度:     {metrics['specificity']:.4f}")
    print(f"  Dice:       {metrics['dice']:.4f}")
    print(f"  F1分数:     {metrics['f1']:.4f}")
    print(f"  IoU:        {metrics['iou']:.4f}")
    print("="*50)


if __name__ == "__main__":
    """
    评估指标测试
    """
    print("="*60)
    print("评估指标测试")
    print("="*60)
    
    # 创建测试数据
    np.random.seed(42)
    
    # 模拟预测和真实值
    y_true = np.zeros((100, 100), dtype=int)
    y_true[40:60, 30:70] = 1  # 模拟血管区域
    
    # 模拟预测 (带一定噪声)
    y_score = y_true.astype(float) + np.random.randn(100, 100) * 0.3
    y_score = np.clip(y_score, 0, 1)
    
    print("\n测试数据:")
    print(f"  图像大小: {y_true.shape}")
    print(f"  血管像素比例: {y_true.mean():.2%}")
    
    # 测试各项指标
    print("\n1. 测试AUC...")
    auc_value = compute_auc(y_true, y_score)
    print(f"   AUC: {auc_value:.4f}")
    
    print("\n2. 测试准确率...")
    acc = compute_accuracy(y_true, y_score, threshold=0.5)
    print(f"   准确率: {acc:.4f}")
    
    print("\n3. 测试灵敏度和特异度...")
    sens, spec = compute_sensitivity_specificity(y_true, y_score, threshold=0.5)
    print(f"   灵敏度: {sens:.4f}")
    print(f"   特异度: {spec:.4f}")
    
    print("\n4. 测试Dice系数...")
    dice = compute_dice(y_true, y_score, threshold=0.5)
    print(f"   Dice: {dice:.4f}")
    
    print("\n5. 测试综合指标...")
    metrics = compute_metrics(y_true, y_score, apply_fov_mask=False)
    print_metrics(metrics, "综合评估结果")
    
    print("\n" + "="*60)
    print("✅ 评估指标测试完成!")
    print("="*60)
