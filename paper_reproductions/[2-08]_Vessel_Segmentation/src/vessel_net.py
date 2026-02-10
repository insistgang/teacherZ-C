"""
血管分割网络架构

实现基于深度学习的血管分割网络，结合小波框架特征。
网络架构包含:
- 编码器 (特征提取)
- 小波框架分支 (多尺度特征)
- 解码器 (分割图生成)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .wavelet_frame import WaveletFrameModule


class ConvBlock(nn.Module):
    """
    卷积块: Conv + BN + ReLU
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EncoderBlock(nn.Module):
    """
    编码器块: 两个卷积层 + 下采样
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv1 = ConvBlock(in_channels, out_channels, dropout=dropout)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout=dropout)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        返回: (特征, 下采样后的特征)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x, self.pool(x)


class DecoderBlock(nn.Module):
    """
    解码器块: 上采样 + 两个卷积层
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv1 = ConvBlock(
            out_channels + skip_channels, out_channels, dropout=dropout
        )
        self.conv2 = ConvBlock(out_channels, out_channels, dropout=dropout)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        
        # 处理尺寸不匹配
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class VesselSegNet(nn.Module):
    """
    血管分割网络
    
    基于U-Net架构，可选集成小波框架特征。
    
    属性:
        use_wavelet: 是否使用小波框架特征
        encoder_channels: 编码器每层的通道数
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        use_wavelet: bool = True,
        wavelet_level: int = 2,
        dropout: float = 0.1
    ):
        """
        初始化血管分割网络
        
        参数:
            in_channels: 输入通道数 (3 for RGB)
            out_channels: 输出通道数 (1 for binary segmentation)
            base_channels: 基础通道数
            use_wavelet: 是否使用小波框架特征
            wavelet_level: 小波分解层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.use_wavelet = use_wavelet
        
        # 编码器
        self.enc1 = EncoderBlock(in_channels, base_channels, dropout)
        self.enc2 = EncoderBlock(base_channels, base_channels*2, dropout)
        self.enc3 = EncoderBlock(base_channels*2, base_channels*4, dropout)
        self.enc4 = EncoderBlock(base_channels*4, base_channels*8, dropout)
        
        # 小波框架模块 (可选)
        if use_wavelet:
            self.wavelet = WaveletFrameModule(
                in_channels=base_channels,
                level=wavelet_level
            )
            wavelet_out = base_channels * (3 * wavelet_level + 1)
            self.wavelet_fusion = ConvBlock(
                wavelet_out, base_channels, dropout=dropout
            )
        
        # 瓶颈层
        bottleneck_in = base_channels * 8
        if use_wavelet:
            bottleneck_in += base_channels
        
        self.bottleneck = nn.Sequential(
            ConvBlock(bottleneck_in, base_channels*16, dropout=dropout),
            ConvBlock(base_channels*16, base_channels*16, dropout=dropout)
        )
        
        # 解码器
        self.dec4 = DecoderBlock(
            base_channels*16, base_channels*8, base_channels*8, dropout
        )
        self.dec3 = DecoderBlock(
            base_channels*8, base_channels*4, base_channels*4, dropout
        )
        self.dec2 = DecoderBlock(
            base_channels*4, base_channels*2, base_channels*2, dropout
        )
        self.dec1 = DecoderBlock(
            base_channels*2, base_channels, base_channels, dropout
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入图像，形状 (B, C, H, W)
            
        返回:
            分割概率图，形状 (B, 1, H, W)
        """
        # 编码器
        enc1, x1 = self.enc1(x)
        enc2, x2 = self.enc2(x1)
        enc3, x3 = self.enc3(x2)
        enc4, x4 = self.enc4(x3)
        
        # 小波框架特征 (可选)
        if self.use_wavelet:
            wavelet_features = self.wavelet(enc1)
            wavelet_features = self.wavelet_fusion(wavelet_features)
            # 下采样以匹配瓶颈层尺寸
            wavelet_features = F.adaptive_avg_pool2d(
                wavelet_features, x4.shape[2:]
            )
            x4 = torch.cat([x4, wavelet_features], dim=1)
        
        # 瓶颈层
        x = self.bottleneck(x4)
        
        # 解码器
        x = self.dec4(x, enc4)
        x = self.dec3(x, enc3)
        x = self.dec2(x, enc2)
        x = self.dec1(x, enc1)
        
        # 输出
        out = self.output(x)
        
        return out
    
    def predict(
        self,
        x: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        预测分割掩码
        
        参数:
            x: 输入图像
            threshold: 二值化阈值
            
        返回:
            二值分割掩码
        """
        with torch.no_grad():
            prob = self.forward(x)
            mask = (prob > threshold).float()
        return mask


class SimpleVesselNet(nn.Module):
    """
    简化版血管分割网络
    
    用于快速测试和轻量级应用。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 16
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, base_channels),
            nn.MaxPool2d(2),
            ConvBlock(base_channels, base_channels*2),
            nn.MaxPool2d(2),
            ConvBlock(base_channels*2, base_channels*4),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2),
            ConvBlock(base_channels*2, base_channels*2),
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),
            ConvBlock(base_channels, base_channels),
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量
    
    参数:
        model: PyTorch模型
        
    返回:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    """
    网络架构测试
    """
    print("="*60)
    print("血管分割网络测试")
    print("="*60)
    
    # 测试VesselSegNet
    print("\n1. 测试VesselSegNet (使用小波框架)...")
    model = VesselSegNet(
        in_channels=3,
        out_channels=1,
        base_channels=32,
        use_wavelet=True
    )
    print(f"   参数数量: {count_parameters(model):,}")
    
    # 测试前向传播
    x = torch.randn(2, 3, 256, 256)
    output = model(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    
    # 测试SimpleVesselNet
    print("\n2. 测试SimpleVesselNet (简化版)...")
    simple_model = SimpleVesselNet(in_channels=3, out_channels=1)
    print(f"   参数数量: {count_parameters(simple_model):,}")
    
    output_simple = simple_model(x)
    print(f"   输出形状: {output_simple.shape}")
    
    # 测试预测
    print("\n3. 测试预测功能...")
    mask = model.predict(x[:1], threshold=0.5)
    print(f"   预测掩码形状: {mask.shape}")
    print(f"   掩码值范围: [{mask.min():.2f}, {mask.max():.2f}]")
    
    print("\n" + "="*60)
    print("✅ 网络架构测试完成!")
    print("="*60)
