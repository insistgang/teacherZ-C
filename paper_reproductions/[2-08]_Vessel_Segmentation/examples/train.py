"""
血管分割训练脚本

训练VesselSegNet进行视网膜血管分割。

使用方法:
    python examples/train.py --data ./data/DRIVE --epochs 50 --batch_size 4

常用参数:
    --data: 数据集路径
    --epochs: 训练轮数
    --batch_size: 批次大小
    --lr: 学习率
    --use_wavelet: 使用小波框架特征
    --output: 输出目录
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.vessel_net import VesselSegNet, count_parameters
from src.dataset import DRIVEDataset, get_drive_transforms
from src.evaluate import compute_metrics, print_metrics


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='血管分割训练')
    
    # 数据参数
    parser.add_argument('--data', type=str, default='./data/DRIVE',
                        help='数据集路径')
    parser.add_argument('--image_size', type=int, nargs=2, default=[576, 576],
                        help='图像大小 (H W)')
    
    # 模型参数
    parser.add_argument('--use_wavelet', action='store_true',
                        help='使用小波框架特征')
    parser.add_argument('--wavelet_level', type=int, default=2,
                        help='小波分解层数')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='基础通道数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    
    # 其他参数
    parser.add_argument('--output', type=str, default='./outputs/vessel_seg',
                        help='输出目录')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载进程数')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备 (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device: str):
    """获取设备"""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Dice损失函数
    
    参数:
        pred: 预测概率 [B, 1, H, W]
        target: 真实标签 [B, 1, H, W]
        smooth: 平滑因子
        
    返回:
        Dice损失
    """
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    组合损失: BCE + Dice
    
    参数:
        pred: 预测概率
        target: 真实标签
        
    返回:
        组合损失
    """
    bce = nn.functional.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device
) -> dict:
    """
    训练一个epoch
    
    参数:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        
    返回:
        训练指标字典
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return {'loss': avg_loss}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """
    评估模型
    
    参数:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    返回:
        评估指标字典
    """
    model.eval()
    
    all_preds = []
    all_masks = []
    
    for images, masks in tqdm(dataloader, desc='Evaluating'):
        images = images.to(device)
        masks = masks.to(device)
        
        outputs = model(images)
        
        all_preds.append(outputs.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # 计算指标
    metrics = compute_metrics(
        all_masks.squeeze(),
        all_preds.squeeze(),
        y_score=all_preds.squeeze(),
        apply_fov_mask=False
    )
    
    return metrics


def main():
    """主函数"""
    args = parse_args()
    
    # 设置
    set_seed(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("血管分割训练")
    print("="*60)
    print(f"设备: {device}")
    print(f"输出目录: {output_dir}")
    print(f"使用小波框架: {args.use_wavelet}")
    
    # 数据
    print("\n加载数据...")
    train_transform, val_transform = get_drive_transforms(
        augment=True,
        image_size=tuple(args.image_size)
    )
    
    try:
        train_dataset = DRIVEDataset(
            root=args.data,
            split='training',
            transform=None  # 变换在collate中处理
        )
        test_dataset = DRIVEDataset(
            root=args.data,
            split='test',
            transform=None
        )
    except Exception as e:
        print(f"错误: {e}")
        print("\n请确保数据集已准备。运行:")
        print(f"  python data/download_drive.py --create-sample --output {args.data}")
        return
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"训练样本: {len(train_dataset)}")
    print(f"测试样本: {len(test_dataset)}")
    
    # 模型
    print("\n创建模型...")
    model = VesselSegNet(
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
        use_wavelet=args.use_wavelet,
        wavelet_level=args.wavelet_level,
        dropout=0.1
    ).to(device)
    
    print(f"模型参数: {count_parameters(model):,}")
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # 日志
    writer = SummaryWriter(output_dir / 'logs')
    
    # 训练循环
    print("\n开始训练...")
    best_auc = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # 评估
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            test_metrics = evaluate(model, test_loader, device)
            print_metrics(test_metrics, f"Epoch {epoch+1} 测试结果")
            
            # TensorBoard
            for key, value in test_metrics.items():
                writer.add_scalar(f'test/{key}', value, epoch)
            
            # 保存最佳模型
            if test_metrics['auc'] > best_auc:
                best_auc = test_metrics['auc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auc': best_auc,
                }, output_dir / 'best_model.pth')
                print(f"  保存最佳模型 (AUC: {best_auc:.4f})")
        
        # 日志
        writer.add_scalar('train/loss', train_metrics['loss'], epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step()
    
    writer.close()
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"最佳AUC: {best_auc:.4f}")
    print(f"模型保存: {output_dir / 'best_model.pth'}")
    print("="*60)


if __name__ == '__main__':
    main()
