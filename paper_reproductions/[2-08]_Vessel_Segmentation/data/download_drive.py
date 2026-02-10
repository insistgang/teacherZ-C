"""
DRIVE数据集下载脚本

DRIVE (Digital Retinal Images for Vessel Extraction)
是视网膜血管分割的标准基准数据集。

数据集结构:
    DRIVE/
    ├── training/
    │   ├── images/           # 训练图像 (20张)
    │   ├── 1st_manual/       # 第一次手动标注
    │   └── mask/             # FOV掩码
    └── test/
        ├── images/           # 测试图像 (20张)
        ├── 1st_manual/       # 第一次手动标注
        ├── 2nd_manual/       # 第二次手动标注 (用于评估一致性)
        └── mask/             # FOV掩码

注意: DRIVE数据集需要从官方网站下载。
本脚本提供下载指引和数据验证功能。

官方网站: https://drive.grand-challenge.org/
"""

import os
import argparse
from pathlib import Path
import zipfile
import urllib.request
from typing import Optional


DRIVE_URL = "https://drive.grand-challenge.org/download/"
DATASET_INFO = """
=================================================================
DRIVE数据集下载说明
=================================================================

DRIVE数据集需要从官方网站手动下载:

1. 访问: https://drive.grand-challenge.org/
2. 注册账号并登录
3. 下载DRIVE数据集 (training.zip 和 test.zip)
4. 解压到指定目录

预期目录结构:
    {output_dir}/
    ├── training/
    │   ├── images/
    │   ├── 1st_manual/
    │   └── mask/
    └── test/
        ├── images/
        ├── 1st_manual/
        ├── 2nd_manual/
        └── mask/

=================================================================
"""


def check_drive_dataset(root_dir: str) -> bool:
    """
    检查DRIVE数据集是否完整
    
    参数:
        root_dir: 数据集根目录
        
    返回:
        是否完整
    """
    root = Path(root_dir)
    
    required_files = {
        'training/images': 20,
        'training/1st_manual': 20,
        'training/mask': 20,
        'test/images': 20,
        'test/1st_manual': 20,
        'test/2nd_manual': 20,
        'test/mask': 20,
    }
    
    all_exist = True
    print("检查DRIVE数据集完整性...")
    
    for subdir, expected_count in required_files.items():
        path = root / subdir
        if not path.exists():
            print(f"  ✗ 缺失目录: {subdir}")
            all_exist = False
        else:
            # 计算文件数
            files = list(path.glob('*.gif')) + list(path.glob('*.tif')) + list(path.glob('*.png'))
            count = len(files)
            status = "✓" if count >= expected_count else "✗"
            print(f"  {status} {subdir}: {count}/{expected_count} 文件")
            if count < expected_count:
                all_exist = False
    
    return all_exist


def create_directory_structure(root_dir: str):
    """
    创建数据集目录结构
    
    参数:
        root_dir: 根目录路径
    """
    root = Path(root_dir)
    
    dirs = [
        'training/images',
        'training/1st_manual',
        'training/mask',
        'test/images',
        'test/1st_manual',
        'test/2nd_manual',
        'test/mask',
    ]
    
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    
    print(f"已创建目录结构: {root_dir}")


def download_sample_data(output_dir: str):
    """
    下载示例数据用于测试
    
    当无法获取DRIVE数据集时，创建合成数据用于代码测试。
    
    参数:
        output_dir: 输出目录
    """
    import numpy as np
    from PIL import Image
    
    output_path = Path(output_dir)
    
    print("\n创建示例合成数据用于测试...")
    print("(注意: 这不是真实的DRIVE数据，仅用于代码测试)")
    
    # 创建目录
    create_directory_structure(output_dir)
    
    np.random.seed(42)
    
    # 创建训练和测试样本
    for split in ['training', 'test']:
        num_samples = 5 if split == 'training' else 3  # 少量样本用于测试
        
        for i in range(num_samples):
            # 创建模拟血管图像
            size = (584, 565)  # DRIVE图像尺寸
            
            # 背景
            image = np.random.rand(*size).astype(np.float32) * 50 + 20
            
            # 添加模拟血管
            y = np.linspace(0, 1, size[0])
            x = np.linspace(0, 1, size[1])
            X, Y = np.meshgrid(x, y)
            
            # 几条曲线模拟血管
            for j in range(5):
                offset = np.random.rand()
                amplitude = np.random.rand() * 0.2
                width = np.random.rand() * 0.01 + 0.002
                
                vessel_center = offset + amplitude * np.sin(2 * np.pi * X * (j + 1))
                vessel = np.abs(Y - vessel_center) < width
                image[vessel] += np.random.rand(*size)[vessel] * 100 + 50
            
            # 保存图像
            img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
            img_rgb = np.stack([img_uint8] * 3, axis=-1)
            Image.fromarray(img_rgb).save(
                output_path / split / 'images' / f'{i+1:02d}_test.tif'
            )
            
            # 创建模拟标注
            vessel_mask = (image > 80).astype(np.uint8) * 255
            Image.fromarray(vessel_mask).save(
                output_path / split / '1st_manual' / f'{i+1:02d}_manual1.gif'
            )
            
            # 创建FOV掩码
            fov_mask = np.ones(size, dtype=np.uint8) * 255
            center_y, center_x = size[0] // 2, size[1] // 2
            radius = min(size) // 2 - 10
            Y, X = np.ogrid[:size[0], :size[1]]
            dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            fov_mask[dist_from_center > radius] = 0
            Image.fromarray(fov_mask).save(
                output_path / split / 'mask' / f'{i+1:02d}_test_mask.gif'
            )
    
    print(f"\n示例数据已创建: {output_dir}")
    print("包含:")
    print("  - 训练集: 5张模拟图像")
    print("  - 测试集: 3张模拟图像")
    print("\n注意: 这些数据仅用于代码测试，不是真实的视网膜图像!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='DRIVE数据集下载和管理工具'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/DRIVE',
        help='输出目录 (默认: ./data/DRIVE)'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='检查数据集完整性'
    )
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='创建示例数据用于测试'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("DRIVE数据集管理工具")
    print("="*60)
    
    if args.check:
        # 检查数据集
        exists = check_drive_dataset(args.output)
        if exists:
            print("\n✅ DRIVE数据集完整!")
        else:
            print("\n⚠️  DRIVE数据集不完整或不存在")
            print(DATASET_INFO.format(output_dir=args.output))
    
    elif args.create_sample:
        # 创建示例数据
        download_sample_data(args.output)
    
    else:
        # 默认: 显示帮助信息
        print(DATASET_INFO.format(output_dir=args.output))
        
        print("\n可用命令:")
        print("  --check          检查数据集完整性")
        print("  --create-sample  创建示例测试数据")
        print("\n示例:")
        print(f"  python download_drive.py --check --output {args.output}")
        print(f"  python download_drive.py --create-sample --output {args.output}")


if __name__ == "__main__":
    main()
