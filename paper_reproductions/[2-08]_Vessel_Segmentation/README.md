# [2-08] 小波框架血管分割 (Wavelet Frame Vessel Segmentation)

## 论文信息

**标题**: Wavelet Frame Based Retinal Vessel Segmentation

**作者**: Xiaohao Cai 等

**发表**: 医学图像分析 (针对眼底图像血管分割)

**论文路径**: `xiaohao_cai_papers/[2-08] 小波框架血管分割 Vessel Segmentation.pdf`

---

## 核心贡献简介

本论文提出了一种基于小波框架的视网膜血管分割方法，结合了多尺度分析和变分优化：

### 1. 多尺度小波框架分析

**小波框架特征**:
- 捕捉血管在不同尺度的特征
- 大尺度: 主要血管结构
- 小尺度: 细微毛细血管

**多尺度表示**:
```
f = Σ W_j^T (W_j f)
```
其中 W_j 是第 j 层的框架分解算子

### 2. 血管特定先验

**血管几何特性**:
- 长条状结构
- 连通性约束
- 宽度变化范围

**能量泛函**:
```
E(u) = ||u - f||² + λ₁ ||Wu||₁ + λ₂ R_vessel(u)
```

### 3. 可扩展性

- 适用于DRIVE、STARE等公开数据集
- 支持2D视网膜图像
- 可扩展到3D血管分割

---

## 复现状态

| 组件 | 状态 | 说明 |
|:---|:---:|:---|
| 小波框架特征 | 🟡 进行中 | 多尺度特征提取已实现 |
| 血管网络 | 🔴 待完成 | 网络架构待实现 |
| 数据集支持 | 🟡 进行中 | DRIVE数据加载器已搭建 |
| 评估指标 | 🟡 进行中 | 基础指标已实现 |
| 训练脚本 | 🔴 待完成 | 待开发 |

**总体状态**: 🟡 **进行中** (约35%完成)

---

## 文件结构说明

```
[2-08]_Vessel_Segmentation/
├── README.md                    # 本文件
├── requirements.txt             # Python依赖
├── data/                        # 数据处理
│   └── download_drive.py       # DRIVE数据集下载
├── src/                         # 源代码
│   ├── __init__.py             # 包初始化
│   ├── vessel_net.py           # 血管分割网络
│   ├── wavelet_frame.py        # 小波框架模块
│   ├── evaluate.py             # 评估指标
│   └── dataset.py              # 数据集处理
└── examples/                    # 示例代码
    └── train.py                # 训练脚本
```

---

## 使用方法

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 下载DRIVE数据集
python data/download_drive.py --output ./data/DRIVE
```

### 快速开始

```python
# 导入模块
from src.vessel_net import VesselSegNet
from src.dataset import DRIVEDataset
from src.wavelet_frame import WaveletFrameModule

# 创建数据加载器
dataset = DRIVEDataset(root='./data/DRIVE', split='train')

# 创建网络
model = VesselSegNet(
    in_channels=3,
    out_channels=1,
    use_wavelet=True
)

# 训练
python examples/train.py --data ./data/DRIVE --epochs 50
```

### 数据集

**DRIVE数据集**:
- 40张视网膜图像 (训练20张，测试20张)
- 分辨率: 584 × 565
- 手动标注的血管分割图

**下载地址**: https://drive.grand-challenge.org/

---

## 依赖要求

- Python >= 3.8
- PyTorch >= 1.10
- torchvision >= 0.11
- PyWavelets >= 1.1
- scikit-image >= 0.18
- opencv-python >= 4.5
- matplotlib >= 3.3

---

## 参考文献

1. Cai, X., et al. (2013). Wavelet Frame Based Retinal Vessel Segmentation.
2. Staal, J., et al. (2004). Ridge-based vessel segmentation in color images of the retina. IEEE TMI.
3. Hoover, A., et al. (2000). Locating blood vessels in retinal images by piecewise threshold probing.
4. Soares, J. V. B., et al. (2006). Retinal vessel segmentation using the 2-D Gabor wavelet.

---

## 更新日志

- **2024-XX-XX**: 创建复现框架
- **2024-XX-XX**: 实现小波框架模块
- **2024-XX-XX**: 添加DRIVE数据加载器
