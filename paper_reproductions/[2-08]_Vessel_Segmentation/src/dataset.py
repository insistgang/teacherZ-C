"""
数据集处理模块

支持DRIVE、STARE等视网膜血管分割数据集。
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DRIVEDataset(Dataset):
    """
    DRIVE数据集
    
    视网膜血管分割标准数据集。
    
    属性:
        root: 数据集根目录
        split: 'train' 或 'test'
        transform: 图像变换
        target_transform: 标签变换
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        """
        初始化DRIVE数据集
        
        参数:
            root: 数据集根目录
            split: 'train' 或 'test'
            transform: 图像变换
            target_transform: 标签变换
            normalize: 是否归一化
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.normalize = normalize
        
        # 验证数据集
        self._validate_dataset()
        
        # 获取图像列表
        self.image_files = self._get_image_files()
        
        print(f"DRIVE {split} 数据集: {len(self.image_files)} 张图像")
    
    def _validate_dataset(self):
        """验证数据集结构"""
        split_dir = self.root / self.split / 'images'
        if not split_dir.exists():
            raise RuntimeError(
                f"数据集不存在: {split_dir}\n"
                "请运行: python data/download_drive.py --create-sample"
            )
    
    def _get_image_files(self):
        """获取图像文件列表"""
        image_dir = self.root / self.split / 'images'
        
        # 支持多种格式
        extensions = ['.tif', '.gif', '.png', '.jpg']
        files = []
        for ext in extensions:
            files.extend(image_dir.glob(f'*{ext}'))
        
        return sorted(files)
    
    def _load_image(self, path: Path) -> np.ndarray:
        """加载图像"""
        img = Image.open(path)
        img_array = np.array(img)
        
        # 确保是RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        
        # 归一化
        if self.normalize:
            img_array = img_array.astype(np.float32) / 255.0
        
        return img_array
    
    def _load_mask(self, image_file: Path) -> np.ndarray:
        """加载分割标签"""
        # 根据图像文件名构造标签文件名
        # 例如: 01_test.tif -> 01_manual1.gif
        image_name = image_file.stem  # e.g., "01_test"
        
        if self.split == 'training':
            mask_name = image_name.replace('_test', '_manual1')
        else:
            mask_name = image_name.replace('_test', '_manual1')
        
        mask_dir = self.root / self.split / '1st_manual'
        mask_file = mask_dir / f'{mask_name}.gif'
        
        # 尝试其他格式
        if not mask_file.exists():
            for ext in ['.tif', '.png', '.jpg']:
                alt_file = mask_dir / f'{mask_name}{ext}'
                if alt_file.exists():
                    mask_file = alt_file
                    break
        
        if not mask_file.exists():
            # 如果没有标签，返回空白掩码
            print(f"警告: 标签不存在 {mask_file}")
            return np.zeros((584, 565), dtype=np.float32)
        
        mask = Image.open(mask_file)
        mask_array = np.array(mask)
        
        # 二值化
        mask_array = (mask_array > 0).astype(np.float32)
        
        return mask_array
    
    def _load_fov_mask(self, image_file: Path) -> np.ndarray:
        """加载FOV (视场) 掩码"""
        image_name = image_file.stem
        
        if self.split == 'training':
            fov_name = image_name.replace('_test', '_training')
        else:
            fov_name = image_name
        
        fov_dir = self.root / self.split / 'mask'
        fov_file = fov_dir / f'{fov_name}_mask.gif'
        
        # 尝试其他格式
        if not fov_file.exists():
            for ext in ['.tif', '.png', '.jpg']:
                alt_file = fov_dir / f'{fov_name}_mask{ext}'
                if alt_file.exists():
                    fov_file = alt_file
                    break
        
        if not fov_file.exists():
            # 如果没有FOV掩码，返回全1
            return np.ones((584, 565), dtype=np.float32)
        
        fov = Image.open(fov_file)
        fov_array = np.array(fov)
        fov_array = (fov_array > 0).astype(np.float32)
        
        return fov_array
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        返回:
            (image, mask) 元组
        """
        image_file = self.image_files[idx]
        
        # 加载图像
        image = self._load_image(image_file)
        mask = self._load_mask(image_file)
        fov = self._load_fov_mask(image_file)
        
        # 应用FOV掩码到标签
        mask = mask * fov
        
        # 转换为Tensor
        image = torch.from_numpy(image.transpose(2, 0, 1))  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # HW -> 1HW
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask


class VesselDataset(Dataset):
    """
    通用血管分割数据集
    
    支持任意格式的图像-标签对。
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        image_suffix: str = '.png',
        mask_suffix: str = '_mask.png'
    ):
        """
        初始化通用数据集
        
        参数:
            image_dir: 图像目录
            mask_dir: 标签目录
            transform: 变换
            image_suffix: 图像后缀
            mask_suffix: 标签后缀
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        
        # 获取图像列表
        self.image_files = sorted(self.image_dir.glob(f'*{image_suffix}'))
        
        print(f"通用数据集: {len(self.image_files)} 张图像")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取样本"""
        image_file = self.image_files[idx]
        
        # 构造标签文件名
        mask_name = image_file.stem + self.mask_suffix
        mask_file = self.mask_dir / mask_name
        
        # 加载
        image = np.array(Image.open(image_file).convert('RGB'))
        mask = np.array(Image.open(mask_file).convert('L'))
        
        # 归一化
        image = image.astype(np.float32) / 255.0
        mask = (mask > 128).astype(np.float32)
        
        # 转换为Tensor
        image = torch.from_numpy(image.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.transform:
            # 对图像和标签应用相同的空间变换
            seed = np.random.randint(2147483647)
            torch.manual_seed(seed)
            image = self.transform(image)
            torch.manual_seed(seed)
            mask = self.transform(mask)
        
        return image, mask


def get_drive_transforms(
    augment: bool = True,
    image_size: Tuple[int, int] = (576, 576)
):
    """
    获取DRIVE数据集的数据增强和预处理
    
    参数:
        augment: 是否使用数据增强
        image_size: 输出图像大小
        
    返回:
        (train_transform, val_transform) 元组
    """
    # 基础变换
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    
    if augment:
        # 训练时增强
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
    else:
        train_transform = base_transform
    
    # 验证时不增强
    val_transform = base_transform
    
    return train_transform, val_transform


if __name__ == "__main__":
    """
    数据集模块测试
    """
    print("="*60)
    print("数据集模块测试")
    print("="*60)
    
    # 测试数据集路径
    test_data_dir = Path(__file__).parent.parent / 'data' / 'DRIVE'
    
    if not test_data_dir.exists():
        print("\n注意: 未找到DRIVE数据集")
        print("运行以下命令创建示例数据:")
        print(f"  python {Path(__file__).parent.parent / 'data' / 'download_drive.py'} --create-sample")
    else:
        print(f"\n1. 测试DRIVE数据集...")
        try:
            dataset = DRIVEDataset(
                root=str(test_data_dir),
                split='training'
            )
            
            print(f"   数据集大小: {len(dataset)}")
            
            # 获取一个样本
            image, mask = dataset[0]
            print(f"   图像形状: {image.shape}")
            print(f"   标签形状: {mask.shape}")
            print(f"   图像值范围: [{image.min():.2f}, {image.max():.2f}]")
            print(f"   标签正样本比例: {mask.mean():.2%}")
            
        except Exception as e:
            print(f"   错误: {e}")
    
    print("\n" + "="*60)
    print("✅ 数据集模块测试完成!")
    print("="*60)
