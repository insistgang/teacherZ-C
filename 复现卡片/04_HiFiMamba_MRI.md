# 复现卡片: HiFi-Mamba MRI Reconstruction

> arXiv: 2508.09179 | MRI快速重建 | 复现难度: ★★★★☆

---

## 基本信息

| 项目 | 信息 |
|:---|:---|
| **标题** | HiFi-Mamba: High-Fidelity Mamba for Fast MRI Reconstruction |
| **作者** | Xiaohao Cai et al. |
| **年份** | 2025 |
| **领域** | 医学影像、深度学习 |
| **状态** | 代码未开源 |

---

## 代码可用性

| 检查项 | 状态 | 详情 |
|:---|:---:|:---|
| **开源代码** | ❌ | 未声明代码开源 |
| **代码仓库** | ❌ | 需自行实现 |
| **伪代码** | ⚠️ | 论文有架构图 |
| **预训练模型** | ❌ | 未提供 |

### 建议实现框架

```
主要语言: Python 3.10+
深度学习框架: PyTorch 2.0+
关键依赖:
├── torch >= 2.0
├── mamba-ssm >= 1.0 (Mamba状态空间模型)
├── timm (位置编码等)
├── fastmri (数据加载)
└── pytorch-lightning (训练框架)
```

---

## 数据集可用性

| 检查项 | 状态 | 详情 |
|:---|:---:|:---|
| **数据类型** | MRI k空间 | 医学影像 |
| **获取难度** | 中等 | 需注册下载 |
| **预处理** | ⚠️ | FastMRI标准流程 |

### FastMRI数据集

| 数据集 | 训练集 | 验证集 | 测试集 | 加速因子 |
|:---|:---:|:---:|:---:|:---:|
| knee_singlecoil | 973 | 199 | 108 | 4x, 8x |
| knee_multicoil | 973 | 199 | 108 | 4x, 8x |
| brain_multicoil | 4469 | 1378 | - | 4x, 8x |

**下载地址**: https://fastmri.org/dataset/

---

## 实验复现步骤

### 环境配置

```bash
# 创建环境
conda create -n hifimamba python=3.10
conda activate hifimamba

# 安装PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装Mamba-SSM
pip install mamba-ssm

# 安装FastMRI工具
pip install fastmri

# 其他依赖
pip install pytorch-lightning matplotlib scipy
```

### 核心架构实现

```python
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class HiFiMambaBlock(nn.Module):
    """
    HiFi-Mamba核心块
    结合Mamba状态空间模型和局部卷积
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Mamba SSM
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        # 局部特征提取
        self.local_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # 融合
        self.fusion = nn.Conv2d(dim*2, dim, 1)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 局部分支
        local_feat = self.local_conv(x)
        
        # Mamba分支 (需要序列化)
        x_seq = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        mamba_feat = self.mamba(x_seq)
        mamba_feat = mamba_feat.permute(0, 2, 1).view(B, C, H, W)
        
        # 融合
        out = torch.cat([local_feat, mamba_feat], dim=1)
        out = self.fusion(out)
        out = self.norm(out.flatten(2)).view(B, C, H, W)
        
        return out + x  # 残差


class HiFiMambaEncoder(nn.Module):
    """编码器: 逐步下采样"""
    def __init__(self, in_ch=2, depths=[2, 2, 4, 2], dims=[48, 96, 192, 384]):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, dims[0], 3, 1, 1)
        
        self.stages = nn.ModuleList()
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            stage = nn.Sequential(
                *[HiFiMambaBlock(dim) for _ in range(depth)]
            )
            self.stages.append(stage)
            
            # 下采样层 (除了最后)
            if i < len(dims) - 1:
                self.stages.append(
                    nn.Conv2d(dim, dims[i+1], 4, 2, 1)
                )
    
    def forward(self, x):
        x = self.stem(x)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class HiFiMambaDecoder(nn.Module):
    """解码器: 逐步上采样"""
    def __init__(self, dims=[384, 192, 96, 48]):
        super().__init__()
        self.up_stages = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            up = nn.Sequential(
                nn.ConvTranspose2d(dims[i], dims[i+1], 4, 2, 1),
                HiFiMambaBlock(dims[i+1]),
            )
            self.up_stages.append(up)
        
        self.final = nn.Conv2d(dims[-1], 2, 3, 1, 1)  # 输出实部+虚部
    
    def forward(self, features):
        x = features[-1]
        for i, up in enumerate(self.up_stages):
            x = up(x)
            # 跳跃连接
            skip = features[-(i+2)]
            x = x + skip
        return self.final(x)


class HiFiMambaNet(nn.Module):
    """完整的HiFi-Mamba网络"""
    def __init__(self):
        super().__init__()
        self.encoder = HiFiMambaEncoder()
        self.decoder = HiFiMambaDecoder()
    
    def forward(self, x_under, mask):
        """
        x_under: 欠采样k空间数据 (B, 2, H, W)
        mask: 采样掩码 (B, 1, H, W)
        """
        # 初始IFFT
        x_img = torch.ifft(torch.complex(x_under[:,0], x_under[:,1]), 2)
        x_init = torch.stack([x_img.real, x_img.imag], dim=1)
        
        # 编码-解码
        features = self.encoder(x_init)
        recon = self.decoder(features)
        
        # 数据一致性
        recon_fft = torch.fft(torch.complex(recon[:,0], recon[:,1]), 2)
        recon_fft = recon_fft * mask + x_under * (1 - mask)
        
        # 最终输出
        output = torch.ifft(recon_fft, 2)
        return torch.stack([output.real, output.imag], dim=1)
```

### 训练脚本

```python
import pytorch_lightning as pl
from fastmri.data import SliceDataset
from fastmri.pl_modules import FastMriDataModule

class HiFiMambaModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = HiFiMambaNet()
        self.lr = lr
        self.loss_fn = nn.L1Loss()
    
    def forward(self, x_under, mask):
        return self.model(x_under, mask)
    
    def training_step(self, batch, batch_idx):
        x_under, mask, target, _ = batch
        output = self(x_under, mask)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_under, mask, target, _ = batch
        output = self(x_under, mask)
        
        # 计算SSIM
        output_img = torch.sqrt(output[:,0]**2 + output[:,1]**2)
        target_img = torch.sqrt(target[:,0]**2 + target[:,1]**2)
        ssim = self.compute_ssim(output_img, target_img)
        
        self.log("val_ssim", ssim)
        return ssim
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# 训练
trainer = pl.Trainer(
    gpus=4,
    max_epochs=100,
    precision=16,
    strategy="ddp"
)
model = HiFiMambaModule()
trainer.fit(model, datamodule)
```

---

## 超参数设置

| 参数 | 论文值 | 建议范围 |
|:---|:---:|:---:|
| `d_state` | 16 | 8-32 |
| `d_conv` | 4 | 2-8 |
| `expand` | 2 | 1-4 |
| `lr` | 1e-4 | 1e-5 to 1e-3 |
| `batch_size` | 4 | 2-8 (per GPU) |
| `epochs` | 100 | 50-200 |

---

## 结果验证

### 论文报告指标

| 数据集 | 加速因子 | SSIM | PSNR |
|:---|:---:|:---:|:---:|
| knee_singlecoil | 4x | 0.873 | 33.21 |
| knee_singlecoil | 8x | 0.792 | 29.85 |
| knee_multicoil | 4x | 0.912 | 35.42 |
| knee_multicoil | 8x | 0.845 | 31.28 |

### 验证脚本

```python
from fastmri.metrics import ssim, psnr

def evaluate_model(model, test_loader, device):
    model.eval()
    ssim_scores = []
    psnr_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_under, mask, target, _ = batch
            x_under = x_under.to(device)
            mask = mask.to(device)
            
            output = model(x_under, mask)
            
            # 计算指标
            output_img = torch.sqrt(output[:,0]**2 + output[:,1]**2)
            target_img = torch.sqrt(target[:,0]**2 + target[:,1]**2)
            
            ssim_scores.append(ssim(output_img, target_img).item())
            psnr_scores.append(psnr(output_img, target_img).item())
    
    print(f"SSIM: {np.mean(ssim_scores):.4f}")
    print(f"PSNR: {np.mean(psnr_scores):.2f} dB")
    
    return np.mean(ssim_scores), np.mean(psnr_scores)
```

---

## 常见问题

### Q1: Mamba-SSM安装失败

```bash
# 确保CUDA版本匹配
nvidia-smi  # 检查CUDA版本
pip install mamba-ssm --no-build-isolation
```

### Q2: GPU内存不足

```python
# 使用梯度检查点
from torch.utils.checkpoint import checkpoint

class CheckpointedMamba(nn.Module):
    def forward(self, x):
        return checkpoint(self._forward, x)
```

### Q3: 训练不稳定

- 使用预热: `warmup_epochs=5`
- 梯度裁剪: `clip_grad_norm=1.0`
- 学习率衰减: `CosineAnnealingLR`

---

## 复现时间估计

| 任务 | 时间 |
|:---|:---:|
| 环境配置 | 1小时 |
| 数据下载 | 2小时 |
| 架构实现 | 8小时 |
| 调试 | 4小时 |
| 训练(4x A100) | 2天 |
| **总计** | **4-5天** |

---

## 注意事项

1. **Mamba依赖**: 需要CUDA 11.6+和特定PyTorch版本
2. **数据量**: FastMRI数据集约100GB
3. **计算资源**: 推荐4x A100或等效GPU
4. **复现风险**: 代码未开源，可能有实现细节差异

---

*最后更新: 2026-02-16*
