# [4-02] DNCNet雷达信号去噪 - 精读笔记

> **论文标题**: DNCNet: Deep Neural Network for Radar Signal Denoising
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐ (中等)
> **重要性**: ⭐⭐⭐⭐ (雷达信号处理必读)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | DNCNet: Deep Neural Network for Radar Signal Denoising |
| **作者** | Xiaohao Cai 等人 |
| **发表期刊** | Remote Sensing (MDPI) |
| **发表年份** | 2022 |
| **文章类型** | 全文论文 |
| **关键词** | Radar Denoising, Deep CNN, IQ Data, Signal Processing |
| **影响因子** | Remote Sensing (2022) ~5.0 |

---

## 🎯 研究问题

### 雷达信号去噪挑战

**核心问题**: 如何有效去除雷达信号中的噪声，同时保留目标特征

**雷达信号特点**:
```
雷达IQ数据:
├── I路 (In-phase): 同相分量
├── Q路 (Quadrature): 正交分量
└── 复数形式: s = I + jQ

噪声来源:
├── 热噪声 (接收机内部)
├── 杂波 (地物、气象)
├── 干扰 (电磁干扰)
└── 多径效应
```

**传统方法局限**:
```
1. 频域滤波: 可能滤除有用信号
2. 小波去噪: 参数选择困难
3. 自适应滤波: 复杂场景适应性差
```

---

## 🔬 方法论详解

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                  含噪雷达IQ信号输入                        │
│                    (I + jQ + noise)                       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 多尺度特征提取模块                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 尺度1: 细粒度特征 (小目标、边缘)                   │    │
│  │ 尺度2: 中粒度特征 (目标结构)                       │    │
│  │ 尺度3: 粗粒度特征 (背景、趋势)                     │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  残差学习模块 ⭐核心                       │
│  ┌─────────────────────────────────────────────────┐    │
│  │ 学习残差: noise = input - clean                  │    │
│  │ 而非直接学习 clean signal                        │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  去噪后雷达信号                           │
│                    (clean I + jQ)                         │
└─────────────────────────────────────────────────────────┘
```

---

### 核心组件1: DNCNet网络结构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DNCNet(nn.Module):
    """
    DNCNet: 深度神经网络雷达信号去噪

    核心设计:
    1. 多尺度特征提取
    2. 残差学习
    3. 复数域处理
    """
    def __init__(self, in_channels=2, num_scales=3, base_channels=64):
        """
        Args:
            in_channels: 输入通道 (I, Q = 2)
            num_scales: 多尺度数量
            base_channels: 基础通道数
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_scales = num_scales

        # 多尺度编码器
        self.encoders = nn.ModuleList([
            self._build_encoder(in_channels, base_channels * (2**i))
            for i in range(num_scales)
        ])

        # 多尺度解码器
        self.decoders = nn.ModuleList([
            self._build_decoder(base_channels * (2**i), in_channels)
            for i in range(num_scales)
        ])

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * num_scales, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, padding=1)
        )

        # 残差连接
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _build_encoder(self, in_ch, out_ch):
        """构建编码器块"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def _build_decoder(self, in_ch, out_ch):
        """构建解码器块"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2),
            nn.BatchNorm2d(in_ch//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, out_ch, 3, padding=1)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, 2, H, W) IQ数据 [I, Q]

        Returns:
            denoised: (B, 2, H, W) 去噪后信号
            residual: (B, 2, H, W) 估计的噪声
        """
        # 多尺度处理
        multi_scale_features = []
        multi_scale_outputs = []

        current_x = x
        for i, (encoder, decoder) in enumerate(zip(self.encoders, self.decoders)):
            # 编码
            encoded = encoder(current_x)

            # 解码到原尺寸
            decoded = decoder(encoded)
            if decoded.shape != x.shape:
                decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)

            multi_scale_outputs.append(decoded)

            # 为下一尺度准备
            if i < self.num_scales - 1:
                current_x = F.avg_pool2d(x, kernel_size=2**(i+1))

        # 融合多尺度输出
        fused = torch.cat(multi_scale_outputs, dim=1)
        residual = self.fusion(fused)

        # 残差学习: clean = noisy - residual
        denoised = x - self.residual_scale * residual

        return denoised, residual
```

---

### 核心组件2: 复数域处理

```python
class ComplexConv2d(nn.Module):
    """
    复数卷积层

    直接处理复数雷达信号
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # 实部和虚部分别卷积
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x):
        """
        复数卷积前向传播

        Args:
            x: (B, 2, H, W) [real, imag]

        Returns:
            output: (B, 2, H, W) [real, imag]
        """
        real = x[:, 0:1, :, :]
        imag = x[:, 1:2, :, :]

        # 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        real_out = self.conv_real(real) - self.conv_imag(imag)
        imag_out = self.conv_real(imag) + self.conv_imag(real)

        return torch.cat([real_out, imag_out], dim=1)


class ComplexDNCNet(nn.Module):
    """
    复数域DNCNet

    在复数域直接处理雷达信号
    """
    def __init__(self, base_channels=64):
        super().__init__()

        # 复数卷积层
        self.conv1 = ComplexConv2d(1, base_channels)
        self.conv2 = ComplexConv2d(base_channels, base_channels)
        self.conv3 = ComplexConv2d(base_channels, base_channels)
        self.conv4 = ComplexConv2d(base_channels, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        复数域前向传播

        Args:
            x: (B, 2, H, W) 复数IQ数据

        Returns:
            denoised: (B, 2, H, W) 去噪后复数信号
        """
        residual = x

        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)

        # 残差连接
        denoised = residual - out

        return denoised
```

---

### 核心组件3: 损失函数设计

```python
class RadarDenoisingLoss(nn.Module):
    """
    雷达去噪专用损失函数

    结合MSE和感知损失
    """
    def __init__(self, alpha=0.8, beta=0.2):
        super().__init__()
        self.alpha = alpha  # MSE权重
        self.beta = beta    # 感知损失权重
        self.mse = nn.MSELoss()

    def forward(self, denoised, clean, noisy=None):
        """
        计算去噪损失

        Args:
            denoised: 去噪后信号
            clean: 干净信号(ground truth)
            noisy: 含噪信号(用于计算残差一致性)

        Returns:
            total_loss: 总损失
            loss_dict: 各分量损失
        """
        # 1. MSE损失
        mse_loss = self.mse(denoised, clean)

        # 2. 幅度损失 (雷达信号幅度更重要)
        denoised_mag = torch.sqrt(denoised[:, 0:1]**2 + denoised[:, 1:2]**2)
        clean_mag = torch.sqrt(clean[:, 0:1]**2 + clean[:, 1:2]**2)
        mag_loss = self.mse(denoised_mag, clean_mag)

        # 3. 相位损失
        denoised_phase = torch.atan2(denoised[:, 1:2], denoised[:, 0:1])
        clean_phase = torch.atan2(clean[:, 1:2], clean[:, 0:1])
        phase_loss = self._phase_loss(denoised_phase, clean_phase)

        # 4. 总变差损失 (平滑性约束)
        tv_loss = self._total_variation(denoised)

        # 组合损失
        total_loss = (self.alpha * mse_loss +
                     0.5 * mag_loss +
                     0.3 * phase_loss +
                     0.1 * tv_loss)

        loss_dict = {
            'mse': mse_loss.item(),
            'magnitude': mag_loss.item(),
            'phase': phase_loss.item(),
            'tv': tv_loss.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict

    def _phase_loss(self, pred_phase, target_phase):
        """相位损失 (考虑相位周期性)"""
        diff = pred_phase - target_phase
        # 将相位差限制在[-π, π]
        diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        return torch.mean(diff**2)

    def _total_variation(self, x):
        """总变差损失"""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return torch.mean(diff_h) + torch.mean(diff_w)
```

---

## 📊 实验结果

### 数据集

| 数据集 | 信号类型 | 样本数 | 信噪比范围 |
|:---|:---:|:---:|:---:|
| **合成数据** | 模拟雷达 | 50,000 | -10~20 dB |
| **实测数据** | 实测雷达 | 10,000 | 0~15 dB |

### 去噪性能对比

| 方法 | PSNR (dB) | SSIM | 处理速度 (ms) |
|:---|:---:|:---:|:---:|
| **传统滤波 (Wiener)** | 28.5 | 0.82 | 5 |
| **小波去噪** | 30.2 | 0.85 | 15 |
| **BM3D** | 32.1 | 0.88 | 200 |
| **DnCNN** | 33.5 | 0.90 | 20 |
| **[4-02] DNCNet** | **35.8** | **0.93** | 25 |

### 不同信噪比下的性能

| 输入SNR | 输出SNR (DNCNet) | 提升 (dB) |
|:---:|:---:|:---:|
| -10 dB | 5.2 dB | 15.2 |
| 0 dB | 12.8 dB | 12.8 |
| 10 dB | 22.5 dB | 12.5 |
| 20 dB | 32.1 dB | 12.1 |

---

## 💡 对违建检测的迁移

### 雷达去噪 → 遥感图像去噪

```python
class RemoteSensingDenoiser(nn.Module):
    """
    遥感图像去噪器

    基于[4-02] DNCNet架构
    适配多光谱/高光谱图像
    """
    def __init__(self, in_channels=3, num_bands=1):
        super().__init__()

        # 修改DNCNet以支持多波段
        self.dncnet = DNCNet(
            in_channels=in_channels * num_bands,
            num_scales=3,
            base_channels=64
        )

        # 波段注意力
        self.band_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * num_bands, num_bands, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        遥感图像去噪

        Args:
            x: (B, C*Bands, H, W) 多光谱图像

        Returns:
            denoised: 去噪后图像
        """
        # 波段注意力加权
        band_weights = self.band_attention(x)
        x_weighted = x * band_weights

        # DNCNet去噪
        denoised, residual = self.dncnet(x_weighted)

        return denoised

    def denoise_change_detection_pair(self, img_t1, img_t2):
        """
        对变化检测图像对进行去噪

        保持双时相图像一致性
        """
        # 分别去噪
        denoised_t1, _ = self.forward(img_t1)
        denoised_t2, _ = self.forward(img_t2)

        # 一致性约束 (可选)
        # 确保相似区域的去噪结果一致

        return denoised_t1, denoised_t2
```

---

## 💡 可复用代码组件

### 组件1: 通用残差去噪网络

```python
class ResidualDenoiser(nn.Module):
    """
    通用残差去噪网络

    可复用于任何图像/信号去噪任务
    """
    def __init__(self, in_channels=3, num_layers=10, num_features=64):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, num_features, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, 3, padding=1))
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(num_features, in_channels, 3, padding=1))

        self.denoiser = nn.Sequential(*layers)
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        残差去噪

        clean = noisy - residual
        """
        residual = self.denoiser(x)
        clean = x - self.residual_scale * residual
        return clean, residual
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **IQ数据** | In-phase/Quadrature | 雷达复数基带信号 |
| **残差学习** | Residual Learning | 学习噪声而非信号 |
| **多尺度** | Multi-Scale | 不同分辨率特征提取 |
| **PSNR** | Peak Signal-to-Noise Ratio | 峰值信噪比 |
| **SSIM** | Structural Similarity | 结构相似性指标 |

---

## ✅ 复习检查清单

- [ ] 理解雷达IQ数据的结构
- [ ] 掌握残差学习的原理
- [ ] 理解多尺度特征提取的作用
- [ ] 了解复数域处理的必要性
- [ ] 能将去噪网络迁移到遥感图像

---

## 🤔 思考问题

1. **为什么残差学习比直接学习干净信号更好？**
   - 提示: 噪声比信号更容易学习

2. **多尺度特征如何帮助去噪？**
   - 提示: 不同尺度的噪声特性

3. **复数域处理的优势是什么？**
   - 提示: 保持相位信息

---

## 🔗 相关论文推荐

### 必读
1. **[4-01] 雷达工作模式识别** - 去噪后的信号处理
2. **DnCNN** - 图像去噪基础网络
3. **BM3D** - 传统去噪标杆方法

### 扩展阅读
1. **Complex-valued Neural Networks** - 复数神经网络
2. **Multi-scale Image Denoising** - 多尺度去噪综述

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
