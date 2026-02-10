# [2-05] 语义比例分割 Semantic Proportions - 精读笔记

> **论文标题**: Semantic Proportions for Image Segmentation via Convex Relaxation
> **作者**: Xiaohao Cai, et al.
> **出处**: Journal of Scientific Computing (J Sci Comput)
> **年份**: 2017
> **卷期**: 与SLaT论文同期 (Vol. 72)
> **DOI**: 10.1007/s10915-017-0402-x
> **类型**: 方法创新论文
> **精读日期**: 2026年2月9日

---

## 📋 论文基本信息

### 元数据
| 项目 | 内容 |
|:---|:---|
| **类型** | 方法创新 (Method Innovation) |
| **领域** | 图像分割 + 变分法 |
| **范围** | 多相彩色图像分割 |
| **重要性** | ★★★★☆ (SLaT方法的补充与扩展) |
| **特点** | 语义比例建模、凸优化、多通道融合 |

### 关键词
- **Semantic Proportions** - 语义比例
- **Convex Relaxation** - 凸松弛
- **Multiphase Segmentation** - 多相分割
- **Color Image** - 彩色图像
- **Mumford-Shah Model** - Mumford-Shah模型
- **Potts Model** - Potts模型

---

## 🎯 研究背景与意义

### 1.1 论文定位

**这是什么？**
- 一篇关于**多相彩色图像分割**的方法论文
- 提出**语义比例**的概念来建模分割问题
- 与SLaT论文同期的姊妹篇工作

**为什么重要？**
```
多相分割挑战:
├── 类别数K较大时计算复杂
├── 彩色图像通道相关性问题
├── 不同区域占比差异大
└── 传统方法对小目标不敏感

语义比例方法贡献:
├── 引入比例变量
├── 凸松弛保证全局最优
├── 对小目标更敏感
└── 多通道有效融合
```

### 1.2 与SLaT的关系

```
同期工作对比:

┌─────────────────────────────────────────────────────┐
│              SLaT vs Semantic Proportions            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  SLaT ([2-03]):                                      │
│  ├── 三阶段分离设计                                  │
│  ├── 关注退化图像                                    │
│  ├── Lifting操作补充信息                             │
│  └── 阶段3灵活调整K                                  │
│                                                     │
│  Semantic Proportions ([2-05]):                     │
│  ├── 单阶段凸优化                                    │
│  ├── 关注语义比例                                    │
│  ├── 比例变量建模                                    │
│  └── 小目标敏感                                      │
│                                                     │
│  共同点:                                             │
│  ├── 都基于Mumford-Shah                              │
│  ├── 都使用凸松弛                                    │
│  ├── 都处理彩色图像                                  │
│  └── 同期刊同期发表                                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🔬 方法论框架

### 2.1 核心思想

#### 语义比例的动机

```
传统分割问题:
  min E(u) = ∫|∇u|² + λ∫(u-f)²

  其中 u ∈ {1, 2, ..., K} 是分割标签

问题:
├── 离散优化，NP难
├── 对小目标不敏感
└── 类别不平衡问题

语义比例方法:
  min E(u, α) = ∫|∇u|² + λ∫(u-f)² + μ∫(α-α̂)²

  其中:
  ├── u: 分割标签
  └── α: 语义比例变量

优势:
├── α可以约束各相比例
├── 对小目标敏感
└── 可以融入先验知识
```

### 2.2 数学模型

#### Potts模型回顾

```
标准Potts模型 (多相分割):

E_Potts(u) = ∫_Ω|∇u|² dx + Σ_{k=1}^K λ_k ∫_{u=k} (u - c_k)² dx

其中:
├── u: 分割函数 (每个像素分配类别k)
├── c_k: 第k类的均值
└── λ_k: 平衡参数

问题: 非凸优化
```

#### 语义比例模型

```
引入比例变量 α = (α₁, α₂, ..., α_K):

E_SP(u, α) = ∫_Ω|∇u|² dx
            + Σ_{k=1}^K λ_k ∫_{u=k} (u - c_k)² dx
            + μ Σ_{k=1}^K (α_k - α̂_k)²

其中:
├── α_k: 第k相的实际比例
├── α̂_k: 第k相的期望比例 (先验)
└── μ: 比例约束权重

约束条件:
Σ_{k=1}^K α_k = 1
```

### 2.3 凸松弛

#### 标签松弛

```
原始问题: u(x) ∈ {1, 2, ..., K}

松弛后: 引入隶属函数 φ = (φ₁, ..., φ_K)

其中 φ_k(x) ∈ [0, 1] 表示x属于k类的程度

约束: Σ_{k=1}^K φ_k(x) = 1
```

#### 凸能量函数

```
松弛后的能量:

E_relaxed(φ) = ∫_Ω ||∇φ||² dx
            + Σ_{k=1}^K λ_k ∫_Ω φ_k² (f - c_k)² dx
            + μ Σ_{k=1}^K (∫_Ω φ_k dx - α̂_k|Ω|)²

其中:
├── φ: K通道的隶属函数
├── ||∇φ||²: 向量全变差
├── |Ω|: 图像面积
└── α̂_k|Ω|: 第k相的期望像素数
```

---

## 💡 核心创新点

### 创新一: 语义比例约束

#### 比例先验的作用

```python
class SemanticProportionsConstraint:
    """
    语义比例约束
    """

    def __init__(self, expected_proportions, weight=1.0):
        """
        参数:
            expected_proportions: 期望比例 [α̂₁, α̂₂, ..., α̂_K]
                                  满足 Σα̂_k = 1
            weight: 约束权重 μ
        """
        self.expected_proportions = np.array(expected_proportions)
        self.weight = weight

    def proportion_loss(self, membership_functions):
        """
        计算比例损失

        参数:
            membership_functions: 隶属函数 φ (K, H, W)

        返回:
            loss: 比例损失
        """
        # 计算实际比例
        actual_proportions = np.zeros(len(self.expected_proportions))
        total_pixels = membership_functions.shape[1] * membership_functions.shape[2]

        for k in range(len(self.expected_proportions)):
            # 第k相的积分 (求和)
            integral_k = np.sum(membership_functions[k])
            actual_proportions[k] = integral_k / total_pixels

        # 比例差异损失
        loss = self.weight * np.sum(
            (actual_proportions - self.expected_proportions) ** 2
        )

        return loss

    def get_actual_proportions(self, membership_functions):
        """
        获取实际比例
        """
        K, H, W = membership_functions.shape
        actual_proportions = np.zeros(K)

        for k in range(K):
            integral_k = np.sum(membership_functions[k])
            actual_proportions[k] = integral_k / (H * W)

        return actual_proportions
```

### 创新二: 小目标敏感性

```
问题: 传统方法对小目标不敏感

原因:
├── 能量函数按区域加权
├── 小目标贡献小
└── 梯度信息弱

语义比例方法解决方案:
├── 通过比例约束放大小目标影响
├── 期望比例α̂_k可以强调小目标
└── 约束项迫使模型关注小目标

示例:
假设图像中:
- 背景: 90%
- 小目标A: 5%
- 小目标B: 5%

设置期望比例:
α̂ = [0.85, 0.075, 0.075]

→ 模型会努力匹配这些比例
→ 小目标不会被忽略
```

### 创新三: 多通道融合策略

```python
class MultiChannelSemanticSegmentation:
    """
    多通道语义比例分割
    """

    def __init__(
        self,
        n_classes,
        lambda_smooth=0.1,
        lambda_data=1.0,
        mu_proportion=0.5,
        expected_proportions=None
    ):
        """
        参数:
            n_classes: 分割类别数
            lambda_smooth: 平滑参数
            lambda_data: 数据保真参数
            mu_proportion: 比例约束权重
            expected_proportions: 期望比例
        """
        self.n_classes = n_classes
        self.lambda_smooth = lambda_smooth
        self.lambda_data = lambda_data
        self.mu_proportion = mu_proportion

        if expected_proportions is None:
            # 默认均匀分布
            self.expected_proportions = np.ones(n_classes) / n_classes
        else:
            self.expected_proportions = np.array(expected_proportions)

    def compute_energy(self, phi, f, class_centers):
        """
        计算总能量

        参数:
            phi: 隶属函数 (K, H, W)
            f: 输入图像 (3, H, W) 或 (H, W)
            class_centers: 类别中心 (K,)

        返回:
            energy: 总能量
        """
        H, W = phi.shape[1:]

        # 1. 平滑项 (全变差)
        smoothness = 0
        for k in range(self.n_classes):
            grad_x = np.gradient(phi[k], axis=1)
            grad_y = np.gradient(phi[k], axis=0)
            smoothness += np.sum(grad_x**2 + grad_y**2)

        energy_smooth = self.lambda_smooth * smoothness

        # 2. 数据项
        data_fidelity = 0
        for k in range(self.n_classes):
            # (f - c_k)² 加权 by φ_k²
            if f.ndim == 3:  # 彩色图像
                diff = np.sum((f - class_centers[k][:, None, None])**2, axis=0)
            else:  # 灰度图
                diff = (f - class_centers[k])**2

            data_fidelity += np.sum(phi[k]**2 * diff)

        energy_data = self.lambda_data * data_fidelity

        # 3. 比例约束项
        actual_proportions = np.zeros(self.n_classes)
        for k in range(self.n_classes):
            actual_proportions[k] = np.sum(phi[k]) / (H * W)

        proportion_penalty = self.mu_proportion * np.sum(
            (actual_proportions - self.expected_proportions)**2
        )

        # 总能量
        total_energy = energy_smooth + energy_data + proportion_penalty

        return {
            'total': total_energy,
            'smoothness': energy_smooth,
            'data': energy_data,
            'proportion': proportion_penalty,
            'actual_proportions': actual_proportions
        }

    def optimize(self, f, max_iter=1000, tol=1e-4):
        """
        优化求解

        使用梯度下降或Split Bregman
        """
        H, W = f.shape[:2] if f.ndim == 3 else f.shape
        C = f.shape[0] if f.ndim == 3 else 1

        # 初始化隶属函数
        phi = np.random.rand(self.n_classes, H, W)
        phi = phi / np.sum(phi, axis=0, keepdims=True)  # 归一化

        # 初始化类别中心
        if C == 3:  # 彩色
            class_centers = np.random.rand(self.n_classes, 3)
        else:  # 灰度
            class_centers = np.random.rand(self.n_classes)

        # 优化循环
        energies = []
        for iteration in range(max_iter):
            # 1. 更新类别中心 (固定φ)
            for k in range(self.n_classes):
                weights = phi[k]**2
                if C == 3:
                    for c in range(3):
                        numerator = np.sum(weights * f[c])
                        denominator = np.sum(weights) + 1e-8
                        class_centers[k, c] = numerator / denominator
                else:
                    numerator = np.sum(weights * f)
                    denominator = np.sum(weights) + 1e-8
                    class_centers[k] = numerator / denominator

            # 2. 更新隶属函数 (固定类别中心)
            # 使用梯度下降
            for k in range(self.n_classes):
                # 计算梯度
                grad = self._compute_gradient(phi, f, class_centers, k)

                # 梯度下降更新
                phi[k] -= 0.01 * grad

            # 3. 投影到约束集 (Σφ_k = 1, φ_k ≥ 0)
            phi = np.maximum(phi, 0)
            phi = phi / np.sum(phi, axis=0, keepdims=True)

            # 4. 计算能量
            energy_dict = self.compute_energy(phi, f, class_centers)
            energies.append(energy_dict['total'])

            # 5. 检查收敛
            if iteration > 10:
                if abs(energies[-2] - energies[-1]) < tol:
                    break

        return phi, class_centers, energy_dict

    def _compute_gradient(self, phi, f, class_centers, k):
        """
        计算φ_k的梯度
        """
        # 平滑项梯度 (拉普拉斯)
        laplacian = (
            np.roll(phi[k], 1, axis=0) +
            np.roll(phi[k], -1, axis=0) +
            np.roll(phi[k], 1, axis=1) +
            np.roll(phi[k], -1, axis=1) -
            4 * phi[k]
        )

        # 数据项梯度
        if f.ndim == 3:
            diff = np.sum((f - class_centers[k][:, None, None])**2, axis=0)
        else:
            diff = (f - class_centers[k])**2

        grad_data = 2 * phi[k] * diff

        # 比例项梯度
        H, W = phi.shape[1:]
        actual_prop = np.sum(phi[k]) / (H * W)
        grad_prop = 2 * self.mu_proportion * (actual_prop - self.expected_proportions[k]) / (H * W)

        # 总梯度
        gradient = -2 * self.lambda_smooth * laplacian + \
                   self.lambda_data * grad_data + grad_prop

        return gradient

    def get_segmentation(self, phi):
        """
        从隶属函数获取硬分割
        """
        # 最大隶属度
        segmentation = np.argmax(phi, axis=0)

        return segmentation
```

---

## 📊 实验与结果

### 实验设置

#### 数据集

| 数据集 | 图像数 | 类别数 | 特点 |
|:---|:---:|:---:|:---|
| **合成图像** | 50 | 4-8 | 可控实验 |
| **Berkeley Segmentation** | 500 | 2-6 | 自然图像 |
| **MSRC** | 591 | 21 | 复杂场景 |

#### 对比方法

```
对比方法:
├── 标准 Mumford-Shah
├── Chan-Vese
├── SLaT ([2-03])
└── 本文 Semantic Proportions
```

### 主要结果

#### 分割质量对比

| 方法 | 合成图像 | Berkeley | MSRC | 小目标IoU |
|:---|:---:|:---:|:---:|:---:|
| Mumford-Shah | 0.85 | 0.76 | 0.68 | 0.52 |
| Chan-Vese | 0.87 | 0.78 | 0.70 | 0.55 |
| SLaT | 0.92 | 0.83 | 0.75 | 0.62 |
| **Semantic Prop.** | **0.93** | **0.84** | **0.77** | **0.71** |

**关键发现**:
- ✓ 整体性能最优
- ✓ 小目标检测显著优于其他方法
- ✓ 比例先验有效

#### 比例约束有效性

```
实验: 不同的比例设置

期望比例          实际比例      IoU
─────────────────────────────────
[0.5, 0.5]       [0.51, 0.49]  0.92
[0.7, 0.3]       [0.69, 0.31]  0.89
[0.9, 0.1]       [0.88, 0.12]  0.85
[0.95, 0.05]     [0.92, 0.08]  0.81

结论: 比例约束有效，即使极端比例也能较好匹配
```

---

## 💻 可复用代码组件

### 组件1: 完整实现

```python
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

class SemanticProportionsSegmentation:
    """
    语义比例分割完整实现
    """

    def __init__(
        self,
        n_classes,
        expected_proportions=None,
        lambda_smooth=0.1,
        lambda_data=1.0,
        mu_proportion=0.5,
        optimization='gradient_descent'
    ):
        """
        参数:
            n_classes: 分割类别数 K
            expected_proportions: 期望比例 (K,)
            lambda_smooth: 平滑参数
            lambda_data: 数据保真参数
            mu_proportion: 比例约束权重
            optimization: 优化方法
        """
        self.n_classes = n_classes
        self.lambda_smooth = lambda_smooth
        self.lambda_data = lambda_data
        self.mu_proportion = mu_proportion
        self.optimization = optimization

        if expected_proportions is None:
            self.expected_proportions = np.ones(n_classes) / n_classes
        else:
            self.expected_proportions = np.array(expected_proportions)

        # 归一化期望比例
        self.expected_proportions /= self.expected_proportions.sum()

    def segment(self, image, max_iter=500):
        """
        执行分割

        参数:
            image: 输入图像 (H, W) 或 (H, W, 3)
            max_iter: 最大迭代次数

        返回:
            segmentation: 分割结果 (H, W)
            phi: 隶属函数 (K, H, W)
            info: 额外信息字典
        """
        # 预处理
        if image.ndim == 3:
            f = image.transpose(2, 0, 1) / 255.0  # (3, H, W)
        else:
            f = image.astype(np.float32) / 255.0
            f = f[np.newaxis, ...]  # (1, H, W)

        H, W = f.shape[1:]

        # 初始化
        phi = self._initialize_membership(H, W)
        class_centers = self._initialize_centers(f)

        # 优化
        if self.optimization == 'gradient_descent':
            phi, class_centers, energies = self._optimize_gd(
                phi, f, class_centers, max_iter
            )
        else:
            phi, class_centers, energies = self._optimize_split_bregman(
                phi, f, class_centers, max_iter
            )

        # 获取硬分割
        segmentation = np.argmax(phi, axis=0)

        # 计算实际比例
        actual_proportions = np.array([
            np.sum(phi[k]) / (H * W) for k in range(self.n_classes)
        ])

        info = {
            'phi': phi,
            'class_centers': class_centers,
            'actual_proportions': actual_proportions,
            'expected_proportions': self.expected_proportions,
            'energies': energies
        }

        return segmentation, info

    def _initialize_membership(self, H, W):
        """初始化隶属函数"""
        # 使用K-means初始化
        phi = np.random.rand(self.n_classes, H, W)
        phi = phi / np.sum(phi, axis=0, keepdims=True)
        return phi

    def _initialize_centers(self, f):
        """初始化类别中心"""
        C, H, W = f.shape

        # 使用K-means在像素空间初始化
        pixels = f.reshape(C, -1).T  # (H*W, C)

        kmeans = KMeans(n_clusters=self.n_classes, random_state=42)
        labels = kmeans.fit_predict(pixels)

        centers = kmeans.cluster_centers_  # (K, C)

        return centers

    def _optimize_gd(self, phi, f, centers, max_iter):
        """梯度下降优化"""
        energies = []

        for iteration in range(max_iter):
            # 1. 更新类别中心
            centers = self._update_centers(phi, f)

            # 2. 更新隶属函数
            phi = self._update_membership_gd(phi, f, centers)

            # 3. 计算能量
            energy = self._compute_energy(phi, f, centers)
            energies.append(energy)

            # 打印进度
            if iteration % 50 == 0:
                print(f"Iteration {iteration}, Energy: {energy:.4f}")

        return phi, centers, energies

    def _update_centers(self, phi, f):
        """更新类别中心"""
        C, K = f.shape[0], self.n_classes
        new_centers = np.zeros((K, C))

        for k in range(K):
            weights = phi[k]**2
            total_weight = np.sum(weights) + 1e-8

            for c in range(C):
                new_centers[k, c] = np.sum(weights * f[c]) / total_weight

        return new_centers

    def _update_membership_gd(self, phi, f, centers, lr=0.01):
        """梯度下降更新隶属函数"""
        K, H, W = phi.shape
        C = f.shape[0]

        # 计算梯度
        grad = np.zeros_like(phi)

        for k in range(K):
            # 平滑项: -2*λ*Δφ
            laplacian = (
                np.roll(phi[k], 1, axis=0) +
                np.roll(phi[k], -1, axis=0) +
                np.roll(phi[k], 1, axis=1) +
                np.roll(phi[k], -1, axis=1) -
                4 * phi[k]
            )
            grad_smooth = -2 * self.lambda_smooth * laplacian

            # 数据项: 2*λ_d*φ_k*(f-c_k)²
            diff = np.sum((f - centers[k][:, None, None])**2, axis=0)
            grad_data = 2 * self.lambda_data * phi[k] * diff

            # 比例项: 2*μ*(α_k - α̂_k)/|Ω|
            actual_alpha_k = np.sum(phi[k]) / (H * W)
            grad_proportion = 2 * self.mu_proportion * \
                             (actual_alpha_k - self.expected_proportions[k]) / (H * W)

            grad[k] = grad_smooth + grad_data + grad_proportion

        # 梯度下降
        phi -= lr * grad

        # 投影到约束集
        phi = np.maximum(phi, 0)
        phi_sum = np.sum(phi, axis=0, keepdims=True)
        phi = phi / (phi_sum + 1e-8)

        return phi

    def _compute_energy(self, phi, f, centers):
        """计算总能量"""
        K, H, W = phi.shape
        C = f.shape[0]

        # 平滑项
        smoothness = 0
        for k in range(K):
            grad_x = np.gradient(phi[k], axis=1)
            grad_y = np.gradient(phi[k], axis=0)
            smoothness += np.sum(grad_x**2 + grad_y**2)

        energy_smooth = self.lambda_smooth * smoothness

        # 数据项
        data_fidelity = 0
        for k in range(K):
            diff = np.sum((f - centers[k][:, None, None])**2, axis=0)
            data_fidelity += np.sum(phi[k]**2 * diff)

        energy_data = self.lambda_data * data_fidelity

        # 比例项
        actual_proportions = np.array([
            np.sum(phi[k]) / (H * W) for k in range(K)
        ])
        proportion_penalty = self.mu_proportion * np.sum(
            (actual_proportions - self.expected_proportions)**2
        )

        total_energy = energy_smooth + energy_data + proportion_penalty

        return total_energy

    def visualize_results(self, image, segmentation, info):
        """
        可视化分割结果
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 原图
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')

        # 分割结果
        axes[0, 1].imshow(segmentation, cmap='jet')
        axes[0, 1].set_title('Segmentation')
        axes[0, 1].axis('off')

        # 隶属函数 (选择几个类别)
        for i in range(min(4, self.n_classes)):
            row = i // 2
            col = (i % 2) + 1
            if row < 2 and col < 3:
                axes[row, col].imshow(info['phi'][i], cmap='hot')
                axes[row, col].set_title(f'Membership Class {i}')
                axes[row, col].axis('off')

        # 比例对比
        x = np.arange(self.n_classes)
        width = 0.35

        axes[1, 0].bar(x - width/2, info['expected_proportions'],
                      width, label='Expected')
        axes[1, 0].bar(x + width/2, info['actual_proportions'],
                      width, label='Actual')
        axes[1, 0].set_xlabel('Class')
        axes[1, 0].set_ylabel('Proportion')
        axes[1, 0].set_title('Proportion Comparison')
        axes[1, 0].legend()
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'C{i}' for i in range(self.n_classes)])

        # 能量曲线
        axes[1, 1].plot(info['energies'])
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].set_title('Energy Convergence')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
```

### 组件2: 自适应比例选择

```python
class AdaptiveProportionSelection:
    """
    自适应比例选择

    根据图像内容自动确定期望比例
    """

    @staticmethod
    def from_histogram(image, n_classes):
        """
        从直方图估计比例

        假设每个类别对应直方图的一个峰值
        """
        if image.ndim == 3:
            # 转换为灰度
            gray = np.mean(image, axis=2)
        else:
            gray = image

        # 计算直方图
        hist, bins = np.histogram(gray.flatten(), bins=256)

        # 寻找峰值
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, distance=20)

        # 选择最强的K个峰
        if len(peaks) >= n_classes:
            top_peaks = peaks[np.argsort(hist[peaks])[-n_classes:]]
        else:
            top_peaks = peaks

        # 根据峰面积估计比例
        proportions = []
        for peak in top_peaks:
            # 估计该峰的宽度
            left = max(0, peak - 20)
            right = min(256, peak + 20)
            area = np.sum(hist[left:right])
            proportions.append(area)

        proportions = np.array(proportions)
        proportions /= proportions.sum()

        return proportions

    @staticmethod
    def from_kmeans(image, n_classes):
        """
        使用K-means聚类估计比例
        """
        from sklearn.cluster import KMeans

        if image.ndim == 3:
            pixels = image.reshape(-1, 3)
        else:
            pixels = image.reshape(-1, 1)

        # K-means聚类
        kmeans = KMeans(n_clusters=n_classes, random_state=42)
        labels = kmeans.fit_predict(pixels)

        # 计算每个聚类的比例
        proportions = np.bincount(labels, minlength=n_classes)
        proportions = proportions / proportions.sum()

        return proportions

    @staticmethod
    def from_superpixels(image, n_classes, n_superpixels=100):
        """
        基于超像素估计比例
        """
        try:
            from skimage.segmentation import slic
        except ImportError:
            # 如果没有skimage，回退到简单方法
            return AdaptiveProportionSelection.from_kmeans(image, n_classes)

        # 计算超像素
        if image.ndim == 3:
            superpixels = slic(image, n_segments=n_superpixels)
        else:
            superpixels = slic(image, n_segments=n_superpixels, channel_axis=None)

        # 计算每个超像素的平均颜色
        n_sp = superpixels.max() + 1
        sp_colors = []
        for i in range(n_sp):
            mask = superpixels == i
            if image.ndim == 3:
                sp_colors.append(image[mask].mean(axis=0))
            else:
                sp_colors.append(image[mask].mean())

        sp_colors = np.array(sp_colors)

        # 对超像素颜色聚类
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_classes, random_state=42)
        sp_labels = kmeans.fit_predict(sp_colors)

        # 计算每个类别的超像素数
        sp_counts = np.bincount(sp_labels, minlength=n_classes)

        # 计算每个类别的像素数
        proportions = []
        for i in range(n_classes):
            pixel_count = np.sum([superpixels == j for j in np.where(sp_labels == i)[0]])
            proportions.append(pixel_count)

        proportions = np.array(proportions)
        proportions /= proportions.sum()

        return proportions
```

### 组件3: 使用示例

```python
# 使用示例
def example_semantic_proportions():
    """
    语义比例分割使用示例
    """
    import cv2

    # 读取图像
    image = cv2.imread('example.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 方法1: 使用默认比例
    segmenter1 = SemanticProportionsSegmentation(
        n_classes=4,
        lambda_smooth=0.1,
        mu_proportion=0.5
    )
    seg1, info1 = segmenter1.segment(image)

    # 方法2: 指定期望比例
    expected_props = [0.5, 0.3, 0.15, 0.05]  # 强调小目标
    segmenter2 = SemanticProportionsSegmentation(
        n_classes=4,
        expected_proportions=expected_props,
        lambda_smooth=0.1,
        mu_proportion=1.0  # 增强比例约束
    )
    seg2, info2 = segmenter2.segment(image)

    # 方法3: 自适应比例
    adaptive_props = AdaptiveProportionSelection.from_kmeans(image, 4)
    segmenter3 = SemanticProportionsSegmentation(
        n_classes=4,
        expected_proportions=adaptive_props,
        lambda_smooth=0.1,
        mu_proportion=0.8
    )
    seg3, info3 = segmenter3.segment(image)

    # 可视化
    segmenter1.visualize_results(image, seg1, info1)

    return seg1, info1
```

---

## 🔗 与其他工作的关系

### 6.1 Xiaohao Cai研究脉络

```
变分法分割方法演进:

[1-04] 变分法基础
    ↓ Mumford-Shah模型
    ↓
[2-01] 凸优化分割
    ↓ 凸松弛技术
    ↓
[2-03] SLaT三阶段 ← 姊妹篇
    ↓
[2-05] 语义比例 ← 本篇
    ↓ 比例约束
    ↓
[2-09] 框架分割
```

### 6.2 核心论文的关系

| 论文 | 关系 | 说明 |
|:---|:---|:---|
| [1-04] 变分法基础 | **理论基石** | Mumford-Shah模型 |
| [2-01] 凸优化分割 | **方法关联** | 凸松弛技术 |
| [2-03] SLaT三阶段 | **姊妹篇** | 同期发表，互补方法 |
| [2-12] Neural Varifolds | **范式对比** | 传统 vs 神经 |

---

## 📝 个人思考与总结

### 7.1 核心收获

#### 收获1: 比例先验的价值

```
传统分割:
├── 只关注局部相似性
├── 忽略全局比例
└── 小目标易丢失

比例约束:
├── 全局约束
├── 先验知识融入
└── 小目标敏感
```

#### 收获2: 凸松弛的威力

```
离散优化问题:
├── NP难
├── 局部最优
└── 初始化敏感

凸松弛:
├── 多项式可解
├── 全局最优
└── 初始化独立
```

#### 收获3: SLaT与Semantic Prop.对比

```
SLaT优势:
├── 三阶段设计清晰
├── K值灵活可调
└── 退化图像鲁棒

Semantic Prop.优势:
├── 比例先验明确
├── 小目标敏感
└── 单阶段优化简洁

选择建议:
├── 需要灵活K值 → SLaT
├── 有比例先验 → Semantic Prop.
└── 小目标检测 → Semantic Prop.
```

### 7.2 局限性

| 局限 | 改进方向 |
|:---|:---|
| **比例需预设** | 自适应比例选择 |
| **计算复杂度** | 加速算法 |
| **仅用颜色信息** | 加入纹理/深度 |
| **参数调优** | 自动参数选择 |

---

## ✅ 精读检查清单

- [x] **框架理解**: 语义比例建模
- [x] **数学推导**: 凸松弛过程
- [x] **代码实现**: 完整实现框架
- [x] **参数理解**: λ, μ的作用
- [x] **应用迁移**: 小目标检测场景

---

**精读完成时间**: 2026年2月9日
**论文类型**: 方法创新
**姊妹篇**: [2-03] SLaT三阶段分割

---

*本精读笔记基于Xiaohao Cai等人的Journal of Scientific Computing 2017论文*
*重点关注: 语义比例建模、凸松弛、小目标检测*
