"""
小波框架特征提取模块

实现基于小波框架的多尺度特征提取，用于血管分割。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import pywt


class WaveletTransform(nn.Module):
    """
    可微分小波变换模块
    
    使用PyTorch实现，支持梯度传播。
    """
    
    def __init__(
        self,
        wavelet: str = 'haar',
        level: int = 2,
        mode: str = 'symmetric'
    ):
        super().__init__()
        
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        
        # 获取滤波器系数
        w = pywt.Wavelet(wavelet)
        
        # 分解滤波器
        dec_lo = torch.tensor(w.dec_lo, dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi, dtype=torch.float32)
        
        # 构建2D滤波器
        # 低通滤波器 (近似)
        self.register_buffer('filter_ll', self._create_filter2d(dec_lo, dec_lo))
        # 水平细节
        self.register_buffer('filter_lh', self._create_filter2d(dec_lo, dec_hi))
        # 垂直细节
        self.register_buffer('filter_hl', self._create_filter2d(dec_hi, dec_lo))
        # 对角细节
        self.register_buffer('filter_hh', self._create_filter2d(dec_hi, dec_hi))
    
    def _create_filter2d(self, row_filter: torch.Tensor, col_filter: torch.Tensor) -> torch.Tensor:
        """
        创建2D可分离滤波器
        
        参数:
            row_filter: 行滤波器
            col_filter: 列滤波器
            
        返回:
            2D滤波器，形状 (1, 1, H, W)
        """
        # 外积创建2D滤波器
        filter_2d = torch.outer(row_filter, col_filter)
        return filter_2d.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        执行单级小波分解
        
        参数:
            x: 输入张量，形状 (B, C, H, W)
            
        返回:
            (近似, 水平细节, 垂直细节, 对角细节) 元组
        """
        b, c, h, w = x.shape
        
        # 为每个通道分别计算
        coeffs = []
        for i in range(c):
            x_c = x[:, i:i+1, :, :]
            
            # 应用滤波器
            ll = F.conv2d(x_c, self.filter_ll, padding='same')
            lh = F.conv2d(x_c, self.filter_lh, padding='same')
            hl = F.conv2d(x_c, self.filter_hl, padding='same')
            hh = F.conv2d(x_c, self.filter_hh, padding='same')
            
            coeffs.append((ll, lh, hl, hh))
        
        # 合并通道
        ll = torch.cat([c[0] for c in coeffs], dim=1)
        lh = torch.cat([c[1] for c in coeffs], dim=1)
        hl = torch.cat([c[2] for c in coeffs], dim=1)
        hh = torch.cat([c[3] for c in coeffs], dim=1)
        
        return ll, lh, hl, hh


class WaveletFrameModule(nn.Module):
    """
    小波框架特征提取模块
    
    提取多尺度小波特征用于血管分割。
    """
    
    def __init__(
        self,
        in_channels: int,
        level: int = 2,
        wavelet: str = 'haar',
        feature_fusion: str = 'concat'
    ):
        """
        初始化小波框架模块
        
        参数:
            in_channels: 输入通道数
            level: 分解层数
            wavelet: 小波类型
            feature_fusion: 特征融合方式 ('concat', 'sum', 'attention')
        """
        super().__init__()
        
        self.level = level
        self.feature_fusion = feature_fusion
        
        # 创建多级小波变换
        self.wavelet_transforms = nn.ModuleList([
            WaveletTransform(wavelet=wavelet, level=1)
            for _ in range(level)
        ])
        
        # 特征处理层
        if feature_fusion == 'attention':
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels * (3 * level + 1), in_channels, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels, 3 * level + 1, 1),
                nn.Softmax(dim=1)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取小波框架特征
        
        参数:
            x: 输入特征，形状 (B, C, H, W)
            
        返回:
            小波特征，形状取决于融合方式
        """
        features = [x]  # 原始特征
        
        current = x
        for i in range(self.level):
            # 小波分解
            ll, lh, hl, hh = self.wavelet_transforms[i](current)
            
            # 添加细节特征
            features.extend([lh, hl, hh])
            
            # 继续分解近似系数
            current = ll
        
        # 添加最后一层近似
        features.append(current)
        features = features[1:]  # 移除重复的原始特征
        
        # 调整所有特征到相同尺寸
        target_size = x.shape[2:]
        resized_features = []
        for f in features:
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(f)
        
        # 特征融合
        if self.feature_fusion == 'concat':
            output = torch.cat(resized_features, dim=1)
        
        elif self.feature_fusion == 'sum':
            # 调整通道数相同
            base_channels = x.shape[1]
            normalized_features = []
            for f in resized_features:
                if f.shape[1] != base_channels:
                    f = F.adaptive_avg_pool3d(f.unsqueeze(1), (base_channels, f.shape[2], f.shape[3])).squeeze(1)
                normalized_features.append(f)
            output = sum(normalized_features)
        
        elif self.feature_fusion == 'attention':
            concat_feat = torch.cat(resized_features, dim=1)
            weights = self.attention(concat_feat)
            
            # 加权求和
            output = sum(w * f for w, f in zip(weights.chunk(len(resized_features), dim=1), resized_features))
        
        else:
            raise ValueError(f"未知的融合方式: {self.feature_fusion}")
        
        return output


class MultiScaleWaveletFeatures(nn.Module):
    """
    多尺度小波特征提取
    
    专门用于血管的多尺度特征提取。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: List[int] = [1, 2, 3]
    ):
        super().__init__()
        
        self.scales = scales
        
        # 为每个尺度创建小波变换
        self.wavelet_encoders = nn.ModuleList([
            nn.Sequential(
                WaveletFrameModule(in_channels, level=s),
                nn.Conv2d(in_channels * (3 * s + 1), out_channels // len(scales), 1),
                nn.BatchNorm2d(out_channels // len(scales)),
                nn.ReLU()
            )
            for s in scales
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取多尺度特征
        
        参数:
            x: 输入特征
            
        返回:
            融合后的多尺度特征
        """
        scale_features = []
        
        for encoder in self.wavelet_encoders:
            feat = encoder(x)
            scale_features.append(feat)
        
        # 合并多尺度特征
        multi_scale = torch.cat(scale_features, dim=1)
        output = self.fusion(multi_scale)
        
        return output


if __name__ == "__main__":
    """
    小波框架模块测试
    """
    print("="*60)
    print("小波框架模块测试")
    print("="*60)
    
    # 测试WaveletTransform
    print("\n1. 测试WaveletTransform...")
    wt = WaveletTransform(wavelet='haar', level=1)
    x = torch.randn(2, 3, 64, 64)
    ll, lh, hl, hh = wt(x)
    print(f"   输入形状: {x.shape}")
    print(f"   LL形状: {ll.shape}")
    print(f"   LH形状: {lh.shape}")
    print(f"   HL形状: {hl.shape}")
    print(f"   HH形状: {hh.shape}")
    
    # 测试WaveletFrameModule
    print("\n2. 测试WaveletFrameModule...")
    wfm = WaveletFrameModule(in_channels=3, level=2, feature_fusion='concat')
    output = wfm(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   输出通道: {output.shape[1]} (期望: {3 * (3 * 2 + 1)} = 21)")
    
    # 测试不同融合方式
    print("\n3. 测试不同融合方式...")
    for fusion in ['concat', 'sum', 'attention']:
        wfm = WaveletFrameModule(in_channels=3, level=2, feature_fusion=fusion)
        out = wfm(x)
        print(f"   {fusion}: {out.shape}")
    
    # 测试MultiScaleWaveletFeatures
    print("\n4. 测试MultiScaleWaveletFeatures...")
    mswf = MultiScaleWaveletFeatures(in_channels=3, out_channels=64, scales=[1, 2])
    output = mswf(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    
    print("\n" + "="*60)
    print("✅ 小波框架模块测试完成!")
    print("="*60)
