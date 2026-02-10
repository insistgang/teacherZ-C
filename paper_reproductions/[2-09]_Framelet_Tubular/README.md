# [2-09] 框架分割管状结构 (Framelet Tubular Structure Segmentation)

## 论文信息

**标题**: Framelet-based Tubular Structure Segmentation with Shape Prior

**作者**: Xiaohao Cai 等

**发表**: 医学图像分割领域

**论文路径**: `xiaohao_cai_papers/[2-09] 框架分割管状结构 Framelet Tubular.pdf`

---

## 核心贡献简介

本论文提出了一种基于小波框架（Framelet）的管状结构分割方法，结合形状先验，主要贡献包括：

### 1. 小波框架理论应用

**小波框架（Framelet）**:
- 比传统小波更灵活的时频分析工具
- 紧框架性质: 完美重建保证
- 多分辨率分析能力

**优势**:
- ✅ 捕捉管状结构的多尺度特征
- ✅ 对噪声具有鲁棒性
- ✅ 计算效率高

### 2. 形状先验集成

**管状结构特点**:
- 长而细的形态
- 特定的曲率分布
- 连续性约束

**形状先验建模**:
```
E_shape = ∫ φ(curvature, thickness, connectivity) ds
```

### 3. 变分框架

**能量泛函**:
```
E(u) = E_data(u) + λ₁ E_framelet(u) + λ₂ E_shape(u)
```

其中:
- **数据项**: 图像强度信息
- **框架项**: 小波框架正则化
- **形状项**: 管状结构先验

---

## 复现状态

| 组件 | 状态 | 说明 |
|:---|:---:|:---|
| 小波框架实现 | 🟡 进行中 | 基础框架变换已实现 |
| 管状形状先验 | 🔴 待完成 | 需要设计形状能量项 |
| 变分优化 | 🟡 进行中 | Split Bregman框架搭建中 |
| 分割算法 | 🔴 待完成 | 待集成 |
| 评估指标 | 🔴 待完成 | 待实现 |

**总体状态**: 🟡 **进行中** (约40%完成)

---

## 文件结构说明

```
[2-09]_Framelet_Tubular/
├── README.md                    # 本文件
├── requirements.txt             # Python依赖
├── src/                         # 源代码
│   ├── __init__.py             # 包初始化
│   ├── framelet.py             # 小波框架实现
│   ├── shape_prior.py          # 形状先验建模
│   ├── segmentation.py         # 分割算法
│   └── utils.py                # 工具函数
└── examples/                    # 示例代码
    └── demo.py                 # 演示脚本
```

---

## 使用方法

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 快速开始

```python
# 导入模块
from src.framelet import FrameletTransform
from src.segmentation import TubularSegmentation

# 创建小波框架变换
framelet = FrameletTransform(level=3, filter_name='haar')

# 加载图像
image = ...  # 医学图像

# 分解
coeffs = framelet.decompose(image)

# 分割
segmenter = TubularSegmentation(lambda_framelet=0.1, lambda_shape=0.05)
segmentation = segmenter.segment(image)
```

### 使用示例脚本

```bash
# 运行演示
python examples/demo.py --input image.tif --output result.png
```

---

## 依赖要求

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- PyWavelets >= 1.1
- scikit-image >= 0.18
- matplotlib >= 3.3

---

## 参考文献

1. Cai, X., et al. (2013). Framelet-based Tubular Structure Segmentation with Shape Prior.
2. Daubechies, I. (1992). Ten Lectures on Wavelets.
3. Dong, B., & Shen, Z. (2010). MRA-based wavelet frames and applications.
4. Osher, S., & Fedkiw, R. (2003). Level Set Methods and Dynamic Implicit Surfaces.

---

## 更新日志

- **2024-XX-XX**: 创建复现框架
- **2024-XX-XX**: 实现基础小波框架变换
