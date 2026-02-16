# 论文核心算法 API 文档

## 目录

1. [概述](#概述)
2. [模块列表](#模块列表)
3. [快速开始](#快速开始)
4. [API 参考](#api-参考)
   - [SLaT 三阶段分割](#1-slat-三阶段分割)
   - [ROF 迭代阈值分割](#2-rof-迭代阈值分割)
   - [Tucker 分解加速](#3-tucker-分解加速)
   - [Neural Varifold 点云核](#4-neural-varifold-点云核)
5. [教程](#教程)
6. [参考](#参考)

---

## 概述

本模块基于以下核心论文实现：

| 模块 | 论文 | 来源 |
|:---|:---|:---|
| `slat_segmentation` | SLaT三阶段分割 | J. Scientific Computing (2017) |
| `rof_iterative_segmentation` | 迭代ROF分割 | IEEE TIP (2013) |
| `tucker_decomposition` | Sketching Tucker | SIAM JSC / JMLR |
| `neural_varifold` | Neural Varifolds | IEEE TPAMI (2022) |

## 模块列表

```
implementations/
├── slat_segmentation.py       # SLaT三阶段分割
├── rof_iterative_segmentation.py  # ROF迭代阈值分割
├── tucker_decomposition.py    # Tucker分解加速
├── neural_varifold.py         # Neural Varifold点云核
└── usage_examples.py          # 使用示例
```

## 快速开始

### 安装依赖

```bash
pip install numpy opencv-python scipy scikit-learn torch
```

### 基本使用

```python
# 1. SLaT 三阶段分割
from slat_segmentation import SLATSegmentation
segmenter = SLATSegmentation(lambda_param=1.0)
result = segmenter.segment(image, K=4)

# 2. ROF 迭代分割
from rof_iterative_segmentation import IterativeROFSegmentation
segmenter = IterativeROFSegmentation(lambda_param=0.1)
seg = segmenter.segment(gray_image, K=3)

# 3. Tucker 分解
from tucker_decomposition import SketchingTucker, reconstruct_tucker
sketchy = SketchingTucker(ranks=[5, 4, 3])
core, factors = sketchy.fit(tensor)
reconstructed = reconstruct_tucker(core, factors)

# 4. Neural Varifold
from neural_varifold import NeuralVarifoldNet
net = NeuralVarifoldNet(num_classes=10)
logits = net(positions, normals)
```

---

## API 参考

### 1. SLaT 三阶段分割

#### 模块概述

SLaT (Smoothing-Lifting-Thresholding) 三阶段分割算法，用于退化彩色图像分割。

**论文**: A Three-Stage Approach for Segmenting Degraded Color Images  
**来源**: Journal of Scientific Computing (2017) 72:1313–1332  
**作者**: Xiaohao Cai, Raymond Chan, Mila Nikolova, Tieyong Zeng

#### 算法流程

```
输入RGB图像 → Stage 1: ROF平滑 → Stage 2: RGB→Lab升维 → Stage 3: K-means分割 → 分割标签图
```

#### 类: `SLATSegmentation`

```python
class SLATSegmentation:
    def __init__(
        self,
        lambda_param: float = 1.0,
        mu: float = 1.0,
        rof_iterations: int = 100,
        rof_tol: float = 1e-4,
    )
```

**参数说明**:

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|:---|:---|:---|:---|:---|
| `lambda_param` | float | 1.0 | (0, ∞) | 数据保真项权重 |
| `mu` | float | 1.0 | (0, ∞) | H1正则化权重 |
| `rof_iterations` | int | 100 | [1, ∞) | ROF迭代最大次数 |
| `rof_tol` | float | 1e-4 | (0, 1) | ROF收敛阈值 |

**方法**:

##### `segment(image, K, return_intermediate=False)`

执行SLaT三阶段分割。

**参数**:

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `image` | np.ndarray | 输入RGB图像，值域[0,1]或[0,255]，形状(H,W,3) |
| `K` | int | 分割类别数 |
| `return_intermediate` | bool | 是否返回中间结果 |

**返回值**: `dict`

| 键 | 类型 | 说明 |
|:---|:---|:---|
| `segmentation` | np.ndarray | 分割标签图 (H, W) |
| `cluster_centers` | np.ndarray | 聚类中心 (K, 6) |
| `smoothed` | np.ndarray | Stage1平滑结果 (可选) |
| `lifted` | np.ndarray | Stage2升维结果 (可选) |

**异常**:
- `ValueError`: 图像维度不正确

**使用示例**:

```python
import numpy as np
from slat_segmentation import SLATSegmentation

# 创建分割器
segmenter = SLATSegmentation(lambda_param=1.5)

# 加载图像 (H, W, 3)
image = np.random.rand(128, 128, 3)

# 执行分割
result = segmenter.segment(image, K=4, return_intermediate=True)

print(f"分割结果: {result['segmentation'].shape}")
print(f"聚类中心: {result['cluster_centers'].shape}")
print(f"平滑后: {result['smoothed'].shape}")
print(f"升维后: {result['lifted'].shape}")  # 6通道
```

#### 便捷函数: `slat_segment()`

```python
def slat_segment(
    image: np.ndarray, 
    K: int, 
    lambda_param: float = 1.0
) -> np.ndarray
```

**使用示例**:

```python
from slat_segmentation import slat_segment

segmentation = slat_segment(image, K=4, lambda_param=1.0)
```

#### 数学原理

**Stage 1: ROF 变分平滑**

对每个 RGB 通道求解:

$$\min_g \frac{\lambda}{2} \|f - g\|^2 + \frac{\mu}{2} \|\nabla g\|^2 + \|\nabla g\|_{TV}$$

**Chambolle 对偶算法**:

$$u = f - \lambda \cdot \text{div}(p)$$

$$p^{n+1} = \frac{p^n + \tau \nabla u}{1 + \tau |\nabla u|}$$

**Stage 2: 维度提升**

RGB → RGB + Lab (6通道):

$$g^* = [g_R, g_G, g_B, g_L, g_a, g_b]$$

**Stage 3: K-means 分割**

$$\min_{\{c_k\}} \sum_{k=1}^{K} \sum_{x \in C_k} \|g^*(x) - c_k\|^2$$

---

### 2. ROF 迭代阈值分割

#### 模块概述

基于近似 ROF 模型的多类图像分割算法。

**论文**: A Multiphase Image Segmentation Based on Approximate ROF Models  
**来源**: IEEE Transactions on Image Processing, 2013

#### 核心思想

- 将 K 类分割分解为 K-1 个二值分割
- 使用标签树组织层次分割
- 每步求解标准 ROF 模型

#### 类: `ROFDenoiser`

ROF 模型求解器。

```python
class ROFDenoiser:
    def __init__(self, max_iter: int = 100, tol: float = 1e-4)
```

**方法**:

##### `solve(f, lambda_param)`

求解ROF模型。

**参数**:

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `f` | np.ndarray | 输入图像 |
| `lambda_param` | float | 数据项权重 |

**返回值**: `np.ndarray` - 去噪后的图像

**使用示例**:

```python
from rof_iterative_segmentation import ROFDenoiser

denoiser = ROFDenoiser(max_iter=100, tol=1e-4)
denoised = denoiser.solve(noisy_image, lambda_param=0.2)
```

#### 类: `IterativeROFSegmentation`

```python
class IterativeROFSegmentation:
    def __init__(
        self,
        lambda_param: float = 0.1,
        max_iter: int = 100,
        tree_type: str = "balanced",
    )
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `lambda_param` | float | 0.1 | ROF正则化参数 |
| `max_iter` | int | 100 | ROF迭代次数 |
| `tree_type` | str | 'balanced' | 'balanced' 或 'sequential' |

**方法**:

##### `segment(image, K, data_terms=None)`

**参数**:

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `image` | np.ndarray | 输入图像 (H, W) 或 (H, W, 3) |
| `K` | int | 类别数 |
| `data_terms` | np.ndarray | 可选的数据项 (H, W, K) |

**返回值**: `np.ndarray` - K类分割标签图 (H, W)

**使用示例**:

```python
from rof_iterative_segmentation import IterativeROFSegmentation

segmenter = IterativeROFSegmentation(lambda_param=0.15, tree_type="balanced")
segmentation = segmenter.segment(gray_image, K=4)
```

#### 类: `AutomaticThresholdROF`

```python
class AutomaticThresholdROF:
    def __init__(self, lambda_param: float = 0.1)
```

**方法**:

##### `segment_with_auto_threshold(image, K=2)`

**返回值**: `Tuple[np.ndarray, List[float]]` - (分割结果, 阈值列表)

**使用示例**:

```python
from rof_iterative_segmentation import AutomaticThresholdROF

auto_seg = AutomaticThresholdROF(lambda_param=0.15)
segmentation, thresholds = auto_seg.segment_with_auto_threshold(image, K=3)
print(f"自动选择的阈值: {thresholds}")
```

#### 数学原理

**ROF 模型**:

$$\min_u \frac{\lambda}{2} \|f - u\|^2 + \text{TV}(u)$$

**迭代分割**:

$$\min_u \int_\Omega u(x) (f_A(x) - f_B(x)) \, dx + \lambda \cdot \text{TV}(u)$$

---

### 3. Tucker 分解加速

#### 模块概述

基于随机 Sketching 的低秩 Tucker 分解算法。

**论文**: Low-Rank Tucker Approximation with Sketching

#### Tucker 分解形式

$$\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{A}^{(1)} \times_2 \mathbf{A}^{(2)} \cdots \times_N \mathbf{A}^{(N)}$$

#### 类: `TensorOperations`

张量操作工具类。

```python
class TensorOperations:
    @staticmethod
    def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray
    
    @staticmethod
    def unfold(tensor: np.ndarray, mode: int) -> np.ndarray
    
    @staticmethod
    def fold(matrix: np.ndarray, mode: int, shape: Tuple[int, ...]) -> np.ndarray
```

**使用示例**:

```python
from tucker_decomposition import TensorOperations as TO

# Mode-n 乘积
Y = TO.mode_n_product(tensor, matrix, mode=0)

# 展开与折叠
matrix = TO.unfold(tensor, mode=0)
tensor_recovered = TO.fold(matrix, mode=0, shape=tensor.shape)
```

#### 类: `RandomizedSVD`

```python
class RandomizedSVD:
    def __init__(self, n_oversamples: int = 10, n_power_iter: int = 2)
```

**方法**:

##### `fit(A, rank)`

**返回值**: `Tuple[np.ndarray, np.ndarray, np.ndarray]` - (U, S, Vh)

#### 类: `SketchingTucker`

```python
class SketchingTucker:
    def __init__(
        self, 
        ranks: List[int], 
        sketch_multipliers: float = 2.0, 
        seed: int = 42
    )
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `ranks` | List[int] | - | 目标Tucker秩 [R_1, R_2, ..., R_N] |
| `sketch_multipliers` | float | 2.0 | Sketch尺寸 = multiplier × rank |
| `seed` | int | 42 | 随机种子 |

**方法**:

##### `fit(tensor, max_iter=50, tol=1e-6)`

**返回值**: `Tuple[np.ndarray, List[np.ndarray]]` - (核张量, 因子矩阵列表)

**使用示例**:

```python
from tucker_decomposition import SketchingTucker, reconstruct_tucker

# 创建分解器
tucker = SketchingTucker(ranks=[5, 4, 3], sketch_multipliers=2.0)

# 执行分解
core, factors = tucker.fit(tensor, max_iter=50, tol=1e-6)

# 重构张量
reconstructed = reconstruct_tucker(core, factors)
```

#### 类: `HOOIDecomposition`

```python
class HOOIDecomposition:
    def __init__(self, ranks: List[int], max_iter: int = 100, tol: float = 1e-6)
```

#### 函数: `reconstruct_tucker()`

```python
def reconstruct_tucker(core: np.ndarray, factors: List[np.ndarray]) -> np.ndarray
```

#### 数学原理

**Mode-n 乘积**:

$$Y_{i_1, \ldots, j, \ldots, i_N} = \sum_{i_n} X_{i_1, \ldots, i_n, \ldots, i_N} A_{j, i_n}$$

**HOOI 算法**:

$$\mathbf{A}^{(n)} = \text{SVD}_R \left( \mathcal{X} \times_1 {\mathbf{A}^{(1)}}^\top \cdots \times_{n-1} {\mathbf{A}^{(n-1)}}^\top \times_{n+1} {\mathbf{A}^{(n+1)}}^\top \cdots \times_N {\mathbf{A}^{(N)}}^\top \right)$$

---

### 4. Neural Varifold 点云核

#### 模块概述

Neural Varifold 是一种用于 3D 点云处理的神经表示方法。

**论文**: Neural Varifolds for 3D Point Cloud Processing  
**来源**: IEEE TPAMI, 2022

#### 核心概念

Varifold 将点云表示为带权重的分布:

$$V = \sum_i w_i \cdot \phi(p_i, n_i) \cdot \delta_{p_i}$$

#### 类: `PositionKernel`

```python
class PositionKernel(nn.Module):
    def __init__(self, sigma: float = 1.0, learnable: bool = True)
```

**参数说明**:

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `sigma` | float | 1.0 | 高斯核带宽参数 |
| `learnable` | bool | True | 是否可学习sigma |

**数学定义**:

$$k(p, q) = \exp\left(-\frac{\|p - q\|^2}{2\sigma^2}\right)$$

**使用示例**:

```python
from neural_varifold import PositionKernel

pos_kernel = PositionKernel(sigma=0.5, learnable=True)
K_pos = pos_kernel(positions, positions)  # (B, N, N)
```

#### 类: `NormalKernel`

```python
class NormalKernel(nn.Module):
    def __init__(self, exponent: int = 1)
```

**数学定义**:

$$k_n(n_1, n_2) = \langle n_1, n_2 \rangle^k$$

#### 类: `VarifoldRepresentation`

```python
class VarifoldRepresentation(nn.Module):
    def __init__(
        self, 
        feat_dim: int = 64, 
        use_normals: bool = True, 
        sigma_init: float = 0.5
    )
```

**方法**:

##### `forward(positions, normals=None)`

**参数**:

| 参数 | 类型 | 说明 |
|:---|:---|:---|
| `positions` | torch.Tensor | (B, N, 3) 点位置 |
| `normals` | torch.Tensor | (B, N, 3) 点法向量 (可选) |

**返回值**: `Tuple[torch.Tensor, torch.Tensor, torch.Tensor]` - (features, weights, positions)

#### 类: `VarifoldKernel`

```python
class VarifoldKernel(nn.Module):
    def __init__(self, sigma_pos: float = 0.5, sigma_feat: float = 1.0)
```

**数学定义**:

$$K(V_1, V_2) = \sum_{i,j} w^1_i \cdot w^2_j \cdot k(p^1_i, p^2_j) \cdot \langle f^1_i, f^2_j \rangle$$

#### 类: `VarifoldDistance`

```python
class VarifoldDistance(nn.Module):
    def __init__(self, sigma_pos: float = 0.5, sigma_feat: float = 1.0)
```

**数学定义**:

$$\|V_1 - V_2\|_V^2 = K(V_1, V_1) + K(V_2, V_2) - 2K(V_1, V_2)$$

#### 类: `NeuralVarifoldNet`

```python
class NeuralVarifoldNet(nn.Module):
    def __init__(
        self, 
        num_classes: int = 10, 
        feat_dim: int = 64, 
        use_normals: bool = True
    )
```

**方法**:

##### `forward(positions, normals=None)`

**返回值**: `torch.Tensor` - (B, N, num_classes) 分割logits

**使用示例**:

```python
import torch
from neural_varifold import NeuralVarifoldNet

# 创建网络
net = NeuralVarifoldNet(num_classes=10, feat_dim=64, use_normals=True)

# 前向传播
positions = torch.randn(2, 256, 3)
normals = torch.nn.functional.normalize(torch.randn(2, 256, 3), dim=-1)

logits = net(positions, normals)
predictions = logits.argmax(dim=-1)
```

#### 类: `VarifoldLayer`

```python
class VarifoldLayer(nn.Module):
    def __init__(self, feat_dim: int, sigma: float = 0.5)
```

**特征更新**:

$$f'_i = f_i + \alpha_i \cdot \text{MLP}\left(\sum_j \frac{k(p_i, p_j) \cdot w_j}{\sum_l k(p_i, p_l) \cdot w_l} \cdot f_j\right)$$

#### 函数: `compute_varifold_norm()`

```python
def compute_varifold_norm(
    positions: torch.Tensor,
    normals: torch.Tensor,
    weights: torch.Tensor,
    sigma: float = 0.5,
) -> torch.Tensor
```

**数学定义**:

$$\|V\|_V^2 = \sum_{i,j} w_i \cdot w_j \cdot k(p_i, p_j) \cdot \langle n_i, n_j \rangle$$

---

## 教程

### 快速开始教程

#### 1. 图像分割 (SLaT)

```python
import cv2
import numpy as np
from slat_segmentation import SLATSegmentation

# 读取图像
image = cv2.imread('input.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float64) / 255.0

# 创建分割器
segmenter = SLATSegmentation(lambda_param=1.5)

# 执行分割
result = segmenter.segment(image, K=4, return_intermediate=True)

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(image)
axes[0].set_title('Original')
axes[1].imshow(result['smoothed'])
axes[1].set_title('Smoothed')
axes[2].imshow(result['segmentation'], cmap='jet')
axes[2].set_title('Segmentation')
axes[3].imshow(result['lifted'][:, :, :3])  # RGB部分
axes[3].set_title('Lifted (RGB)')
plt.show()
```

#### 2. 灰度图像分割 (ROF)

```python
import numpy as np
from rof_iterative_segmentation import IterativeROFSegmentation, AutomaticThresholdROF

# 方法1: 迭代ROF分割
segmenter = IterativeROFSegmentation(lambda_param=0.15, tree_type="balanced")
seg1 = segmenter.segment(gray_image, K=4)

# 方法2: 自动阈值
auto_seg = AutomaticThresholdROF(lambda_param=0.15)
seg2, thresholds = auto_seg.segment_with_auto_threshold(gray_image, K=4)
```

#### 3. 张量分解 (Tucker)

```python
import numpy as np
from tucker_decomposition import SketchingTucker, HOOIDecomposition, reconstruct_tucker

# 创建测试张量
tensor = np.random.randn(50, 40, 30)

# 方法1: Sketching加速
sketchy = SketchingTucker(ranks=[5, 4, 3], sketch_multipliers=2.0)
core, factors = sketchy.fit(tensor)

# 方法2: 标准HOOI
hooi = HOOIDecomposition(ranks=[5, 4, 3])
core2, factors2 = hooi.fit(tensor)

# 重构
reconstructed = reconstruct_tucker(core, factors)

# 计算误差
error = np.linalg.norm(tensor - reconstructed) / np.linalg.norm(tensor)
print(f"相对重构误差: {error:.6f}")
```

#### 4. 点云分割 (Neural Varifold)

```python
import torch
from neural_varifold import NeuralVarifoldNet, VarifoldRepresentation, VarifoldDistance

# 加载点云
positions = torch.randn(2, 256, 3)
normals = torch.nn.functional.normalize(torch.randn(2, 256, 3), dim=-1)

# 点云分割
net = NeuralVarifoldNet(num_classes=10, feat_dim=64)
logits = net(positions, normals)
predictions = logits.argmax(dim=-1)

# 点云相似度计算
encoder = VarifoldRepresentation(feat_dim=64)
feat1, w1, pos1 = encoder(positions[0:1], normals[0:1])
feat2, w2, pos2 = encoder(positions[1:2], normals[1:2])

dist_fn = VarifoldDistance(sigma_pos=0.5, sigma_feat=1.0)
distance = dist_fn((pos1, feat1, w1), (pos2, feat2, w2))
print(f"Varifold距离: {distance.item():.6f}")
```

### 进阶用法

#### 参数调优策略

```python
# SLaT: lambda_param 调优
for lambda_val in [0.5, 1.0, 1.5, 2.0, 3.0]:
    segmenter = SLATSegmentation(lambda_param=lambda_val)
    result = segmenter.segment(image, K=4)
    # 评估分割质量...

# Tucker: 秩选择
for ranks in [[3,3,3], [5,4,3], [10,8,6]]:
    tucker = SketchingTucker(ranks=ranks)
    core, factors = tucker.fit(tensor)
    error = compute_reconstruction_error(tensor, core, factors)
    print(f"Ranks {ranks}: error = {error:.4f}")
```

#### 批量处理

```python
# 批量图像分割
images = [load_image(f) for f in image_files]
segmenter = SLATSegmentation(lambda_param=1.5)

results = []
for image in images:
    result = segmenter.segment(image, K=4)
    results.append(result['segmentation'])
```

### 最佳实践

1. **数据预处理**
   - 图像归一化到 [0, 1]
   - 法向量归一化
   - 点云中心化

2. **参数选择**
   - lambda_param: 从 1.0 开始，根据噪声调整
   - K 值: 使用肘部法则确定
   - Tucker 秩: 从小秩开始逐步增加

3. **性能优化**
   - 大图像: 先降采样，再分割
   - 大点云: 使用随机采样
   - Tucker: 使用 Sketching 加速

---

## 参考

### API 索引

#### SLaT 分割

| 名称 | 类型 | 说明 |
|:---|:---|:---|
| `SLATSegmentation` | 类 | 主分割类 |
| `slat_segment` | 函数 | 便捷函数 |

#### ROF 分割

| 名称 | 类型 | 说明 |
|:---|:---|:---|
| `ROFDenoiser` | 类 | ROF求解器 |
| `IterativeROFSegmentation` | 类 | 迭代分割 |
| `AutomaticThresholdROF` | 类 | 自动阈值 |

#### Tucker 分解

| 名称 | 类型 | 说明 |
|:---|:---|:---|
| `TensorOperations` | 类 | 张量操作 |
| `RandomizedSVD` | 类 | 随机SVD |
| `SketchingTucker` | 类 | Sketching分解 |
| `HOOIDecomposition` | 类 | 标准HOOI |
| `reconstruct_tucker` | 函数 | 张量重构 |

#### Neural Varifold

| 名称 | 类型 | 说明 |
|:---|:---|:---|
| `PositionKernel` | 类 | 位置核 |
| `NormalKernel` | 类 | 法向量核 |
| `VarifoldRepresentation` | 类 | Varifold编码 |
| `VarifoldKernel` | 类 | Varifold核 |
| `VarifoldDistance` | 类 | Varifold距离 |
| `NeuralVarifoldNet` | 类 | 完整网络 |
| `VarifoldLayer` | 类 | Varifold层 |
| `compute_varifold_norm` | 函数 | Varifold范数 |

### 参数速查表

#### SLaT 分割参数

| 参数 | 范围 | 推荐值 | 说明 |
|:---|:---|:---|:---|
| `lambda_param` | 0.1-5.0 | 1.0-2.0 | 数据保真权重 |
| `mu` | 0.1-5.0 | 1.0 | H1正则化权重 |
| `rof_iterations` | 50-500 | 100 | ROF迭代次数 |
| `rof_tol` | 1e-6-1e-3 | 1e-4 | 收敛阈值 |

#### ROF 分割参数

| 参数 | 范围 | 推荐值 | 说明 |
|:---|:---|:---|:---|
| `lambda_param` | 0.05-0.5 | 0.1-0.2 | ROF正则化 |
| `tree_type` | - | 'balanced' | 标签树类型 |

#### Tucker 分解参数

| 参数 | 范围 | 推荐值 | 说明 |
|:---|:---|:---|:---|
| `ranks` | - | 原维度5%-20% | Tucker秩 |
| `sketch_multipliers` | 1.5-3.0 | 2.0 | Sketch倍数 |
| `max_iter` | 20-100 | 50 | HOOI迭代 |

#### Neural Varifold 参数

| 参数 | 范围 | 推荐值 | 说明 |
|:---|:---|:---|:---|
| `sigma` | 0.1-1.0 | 0.3-0.5 | 位置核带宽 |
| `feat_dim` | 32-128 | 64 | 特征维度 |
| `num_classes` | - | 取决于任务 | 分割类别数 |

### FAQ

**Q1: SLaT 适用于哪些图像类型？**
> 适用于自然图像、医学图像、遥感图像等。对噪声和模糊有较好的鲁棒性。

**Q2: 如何选择 lambda_param？**
> 高噪声图像用较小值 (0.5-1.0)，低噪声图像用较大值 (1.5-3.0)。

**Q3: Tucker 分解的精度损失如何？**
> Sketching 方法通常在 1% 以内，速度可提升 5-10 倍。

**Q4: Neural Varifold 与 PointNet 的区别？**
> PointNet 独立处理每个点，Varifold 显式建模点之间的关系。

**Q5: 如何处理大点云？**
> 使用随机采样或网格下采样，保持点数在 1024-4096 之间。

### 参考文献

1. Cai, X., Chan, R., Nikolova, M., & Zeng, T. (2017). A Three-Stage Approach for Segmenting Degraded Color Images. *Journal of Scientific Computing*, 72(3), 1313-1332.

2. Cai, X., et al. (2013). A Multiphase Image Segmentation Based on Approximate ROF Models. *IEEE Transactions on Image Processing*.

3. Sun, W., et al. Low-Rank Tucker Approximation with Sketching. *SIAM Journal on Scientific Computing*.

4. Cai, X., et al. (2022). Neural Varifolds for 3D Point Cloud Processing. *IEEE TPAMI*.
