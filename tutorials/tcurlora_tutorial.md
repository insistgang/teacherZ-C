# tCURLoRA张量CUR低秩适应教程

## 目录
1. [理论讲解](#1-理论讲解)
2. [算法详解](#2-算法详解)
3. [代码实现](#3-代码实现)
4. [实验指南](#4-实验指南)
5. [习题与答案](#5-习题与答案)

---

## 1. 理论讲解

### 1.1 大模型微调背景

**全参数微调的挑战**

大语言模型（LLM）参数量巨大（数十亿到千亿），全参数微调面临：
1. **存储开销**：每个任务需要存储完整模型
2. **计算成本**：反向传播需要大量GPU显存
3. **灾难性遗忘**：微调可能损害预训练知识

**参数高效微调（PEFT）**

核心思想：只训练少量参数，冻结大部分预训练权重：
$$W_{\text{finetuned}} = W_0 + \Delta W$$

其中 $\Delta W$ 的参数量远小于 $W_0$。

### 1.2 LoRA基础

**低秩适应（LoRA）**

LoRA（Low-Rank Adaptation）假设权重更新 $\Delta W$ 是低秩的：

$$\Delta W = B A$$

其中：
- $B \in \mathbb{R}^{d \times r}$：下投影矩阵
- $A \in \mathbb{R}^{r \times k}$：上投影矩阵
- $r \ll \min(d, k)$：秩

**前向传播**

$$h = W_0 x + \Delta W x = W_0 x + B(Ax)$$

**参数量对比**

| 方法 | 原始 | LoRA |
|------|------|------|
| 参数量 | $d \times k$ | $r \times (d + k)$ |
| 压缩比 | 1 | $\frac{rk}{d \times k} = \frac{r}{d}$ |

**优势**

1. 训练参数减少10-100倍
2. 不增加推理延迟（可合并到原权重）
3. 可切换不同任务的适配器

### 1.3 CUR分解

**矩阵CUR分解**

将矩阵 $M$ 分解为：
$$M \approx C U R$$

其中：
- $C \in \mathbb{R}^{m \times c}$：$M$ 的 $c$ 列（实际列）
- $R \in \mathbb{R}^{r \times n}$：$M$ 的 $r$ 行（实际行）
- $U \in \mathbb{R}^{c \times r}$：交矩阵

**与SVD的对比**

| 特性 | SVD | CUR |
|------|-----|-----|
| 基底 | 抽象向量 | 原矩阵行/列 |
| 可解释性 | 低 | 高 |
| 稀疏保持 | 否 | 是 |

**列选择算法**

使用杠杆得分（Leverage Score）采样：
$$\pi_j = \frac{1}{k}\sum_{i=1}^{k} (v_j^{(i)})^2$$

其中 $v_j^{(i)}$ 是第 $i$ 个右奇异向量的第 $j$ 个分量。

### 1.4 张量CUR分解

**张量CUR**

将矩阵CUR推广到张量 $\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$：

$$\mathcal{X} \approx \mathcal{C} \times_{(1)} \mathbf{U}^{(1)} \times_{(2)} \mathbf{U}^{(2)} \times \cdots \times_{(N)} \mathbf{U}^{(N)}$$

其中 $\mathcal{C}$ 是从原张量采样的纤维/切片构成的子张量。

**核心张量采样**

通过各模态的杠杆得分采样：
$$\pi_i^{(n)} = \frac{1}{R_n}\sum_{j=1}^{R_n} (u_{ij}^{(n)})^2$$

### 1.5 tCURLoRA模型

**核心思想**

将LoRA的低秩更新与张量CUR的稀疏结构结合：

1. **张量视角**：将权重矩阵视为张量的一部分
2. **CUR采样**：只选择重要的行/列进行更新
3. **低秩投影**：在采样子空间中学习低秩适应

**数学形式**

对于权重张量 $\mathcal{W} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_N}$：

$$\Delta \mathcal{W} = \mathcal{C} \times_1 \mathbf{B}^{(1)} \times_2 \mathbf{B}^{(2)} \times \cdots \times_N \mathbf{B}^{(N)}$$

其中：
- $\mathcal{C}$：采样的核心张量（稀疏）
- $\mathbf{B}^{(n)}$：第 $n$ 模态的低秩投影

**优势**

1. **更少的参数**：利用张量结构进一步压缩
2. **更好的泛化**：CUR采样保持数据结构
3. **可解释性**：采样的行/列有物理意义

### 1.6 存在性与收敛性

**存在性定理**

给定权重更新 $\Delta W$ 和目标秩 $r$，存在CUR分解：
$$\Delta W = CUR + E, \quad \|E\| \leq O(\sigma_{r+1})$$

其中 $\sigma_{r+1}$ 是第 $r+1$ 个奇异值。

**收敛性分析**

tCURLoRA的收敛性由以下保证：
1. CUR分解的采样复杂度：$O(k^2 \log k / \epsilon^2)$
2. 梯度下降的收敛率：$O(1/\sqrt{T})$
3. 组合后：$O(k^2 \log k / \epsilon^2 + T)$

---

## 2. 算法详解

### 2.1 标准LoRA算法

**算法步骤**

```
输入: 预训练权重 W₀ ∈ ℝ^(d×k), 秩 r, 缩放因子 α

初始化:
    A ∈ ℝ^(r×k) 使用随机高斯
    B ∈ ℝ^(d×r) 初始化为0

训练循环:
    For each batch (x, y):
        # 前向传播
        h = W₀ @ x + α · (B @ A @ x)
        
        # 计算损失
        loss = criterion(h, y)
        
        # 反向传播（只更新A和B）
        A ← A - lr · ∂loss/∂A
        B ← B - lr · ∂loss/∂B

输出: 微调后的 A, B
```

**合并权重（推理时）**

$$W_{\text{merged}} = W_0 + \alpha \cdot BA$$

### 2.2 CUR采样算法

**杠杆得分计算**

```
输入: 矩阵 M ∈ ℝ^(m×n), 目标列数 c

1. 计算SVD: M = U Σ Vᵀ
2. 计算列杠杆得分:
   For j = 1 to n:
       πⱼ = (1/k) Σᵢ₌₁ᵏ (V[j,i])²
3. 归一化: πⱼ = πⱼ / Σⱼ πⱼ
4. 按概率采样c列（无放回）

输出: 选择的列索引
```

**交矩阵计算**

给定采样的列矩阵 $C$ 和行矩阵 $R$：
$$U = C^\dagger M R^\dagger$$

其中 $\dagger$ 表示伪逆。

### 2.3 张量CUR算法

**多模态采样**

```
输入: 张量 X ∈ ℝ^(I₁×I₂×...×I_N), 各模态采样数 (c₁, c₂, ..., c_N)

For n = 1 to N:
    1. 模-n展开: X₍ₙ₎ ∈ ℝ^(Iₙ × ∏ₖ≠ₙ Iₖ)
    2. 计算SVD: X₍ₙ₎ = U Σ Vᵀ
    3. 计算杠杆得分并采样cₙ个索引

构建核心张量:
    C = X[I₁_sample, I₂_sample, ..., I_N_sample]

计算交矩阵:
    For n = 1 to N:
        U⁽ⁿ⁾ = pseudo_inverse(X₍ₙ₎[sample_indices, :])

输出: C, {U⁽ⁿ⁾}
```

### 2.4 tCURLoRA完整算法

```
输入: 
    - 预训练权重张量 W₀
    - 目标秩 (r₁, r₂, ..., r_N)
    - CUR采样数 (c₁, c₂, ..., c_N)
    - 训练数据 D

初始化:
    1. 计算初始CUR分解:
       W₀ ≈ C₀ ×₁ U₀⁽¹⁾ ×₂ U₀⁽²⁾ ... ×_N U₀⁽ᴺ⁾
    
    2. 初始化低秩适应矩阵:
       For n = 1 to N:
           B⁽ⁿ⁾ ∈ ℝ^(cₙ × rₙ) 随机初始化
           A⁽ⁿ⁾ ∈ ℝ^(rₙ × cₙ) 初始化为0

训练:
    For each batch:
        # 计算权重更新
        ΔW = learnable_core ×₁ B⁽¹⁾A⁽¹⁾ ×₂ B⁽²⁾A⁽²⁾ ... ×_N B⁽ᴺ⁾A⁽ᴺ⁾
        
        # 前向传播
        h = (W₀ + ΔW) @ x
        
        # 计算损失并反向传播
        ...

输出: 微调后的 {B⁽ⁿ⁾, A⁽ⁿ⁾}
```

### 2.5 复杂度分析

**参数量对比**

| 方法 | 参数量 | 相对LoRA |
|------|--------|----------|
| 全参数 | $O(d \cdot k)$ | 100% |
| LoRA | $O(r \cdot (d + k))$ | $r/d \cdot 100\%$ |
| tCURLoRA | $O(\sum_n c_n \cdot r_n)$ | 更低 |

**计算复杂度**

- 前向传播：$O(\prod_n c_n \cdot \sum_n r_n)$
- CUR采样：$O(N \cdot I^N)$（预处理，一次）
- 反向传播：同前向

---

## 3. 代码实现

### 3.1 基础LoRA实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


class LoRALayer(nn.Module):
    """
    标准LoRA层实现
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        """
        参数:
            in_features: 输入维度
            out_features: 输出维度
            rank: 低秩秩
            alpha: 缩放因子
            dropout: dropout率
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: (batch, in_features)
        
        返回:
            (batch, out_features)
        """
        # 低秩适应
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_out * self.scaling
    
    def merge_with_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        合并到原权重（推理优化）
        """
        return weight + self.lora_B @ self.lora_A * self.scaling


class LoRALinear(nn.Module):
    """
    带LoRA的线性层
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        
        # 原始线性层（冻结）
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False
        if bias:
            self.linear.bias.requires_grad = False
        
        # LoRA层
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)
    
    def merge_weights(self) -> torch.Tensor:
        """合并权重用于高效推理"""
        return self.lora.merge_with_weight(self.linear.weight)
```

### 3.2 CUR分解实现

```python
def compute_leverage_scores(matrix: np.ndarray, k: int) -> np.ndarray:
    """
    计算列杠杆得分
    
    参数:
        matrix: (m, n) 输入矩阵
        k: 使用的奇异向量数
    
    返回:
        (n,) 杠杆得分
    """
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
    
    # 使用前k个右奇异向量
    V = Vh[:k].T  # (n, k)
    
    # 计算杠杆得分
    leverage = np.sum(V ** 2, axis=1) / k
    
    return leverage


def cur_sample_columns(matrix: np.ndarray, 
                       n_columns: int, 
                       k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    CUR列采样
    
    参数:
        matrix: (m, n) 输入矩阵
        n_columns: 要采样的列数
        k: SVD截断秩
    
    返回:
        indices: 采样的列索引
        C: 采样的列矩阵
    """
    m, n = matrix.shape
    
    if k is None:
        k = min(n_columns * 2, n)
    
    # 计算杠杆得分
    leverage = compute_leverage_scores(matrix, k)
    
    # 归一化为概率
    probs = leverage / leverage.sum()
    
    # 采样（无放回，倾向于高分）
    indices = np.random.choice(n, size=n_columns, replace=False, p=probs)
    
    # 提取列
    C = matrix[:, indices]
    
    return indices, C


def cur_decomposition(matrix: np.ndarray, 
                      rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    完整CUR分解
    
    参数:
        matrix: (m, n) 输入矩阵
        rank: 目标秩
    
    返回:
        C: 列矩阵 (m, c)
        U: 交矩阵 (c, r)
        R: 行矩阵 (r, n)
    """
    m, n = matrix.shape
    c = r = rank
    
    # 列采样
    col_indices, C = cur_sample_columns(matrix, c)
    
    # 行采样（在转置上采样列）
    row_indices, R = cur_sample_columns(matrix.T, r)
    R = R.T
    
    # 计算交矩阵
    # U = C⁺ @ M @ R⁺
    C_pinv = np.linalg.pinv(C)
    R_pinv = np.linalg.pinv(R)
    U = C_pinv @ matrix @ R_pinv
    
    return C, U, R, col_indices, row_indices
```

### 3.3 张量CUR实现

```python
class TensorCUR:
    """
    张量CUR分解
    """
    
    def __init__(self, 
                 tensor: np.ndarray, 
                 sample_sizes: List[int],
                 svd_ranks: Optional[List[int]] = None):
        """
        参数:
            tensor: 输入张量
            sample_sizes: 各模态采样数
            svd_ranks: SVD截断秩
        """
        self.tensor = tensor
        self.n_modes = tensor.ndim
        self.sample_sizes = sample_sizes
        
        if svd_ranks is None:
            self.svd_ranks = [s * 2 for s in sample_sizes]
        else:
            self.svd_ranks = svd_ranks
        
        self.sampled_indices = []
        self.core_tensor = None
        self.intersection_matrices = []
    
    def decompose(self) -> Tuple[np.ndarray, List[np.ndarray], List[List[int]]]:
        """
        执行张量CUR分解
        
        返回:
            core: 核心张量
            U_list: 交矩阵列表
            indices_list: 采样索引列表
        """
        # 各模态采样
        for mode in range(self.n_modes):
            # 模-n展开
            unfolded = np.moveaxis(self.tensor, mode, 0)
            unfolded = unfolded.reshape(self.tensor.shape[mode], -1)
            
            # 计算杠杆得分并采样
            k = min(self.svd_ranks[mode], unfolded.shape[1])
            leverage = compute_leverage_scores(unfolding.T, k)  # 在列方向
            probs = leverage / leverage.sum()
            
            indices = np.random.choice(
                self.tensor.shape[mode],
                size=self.sample_sizes[mode],
                replace=False,
                p=probs
            )
            self.sampled_indices.append(indices)
        
        # 提取核心张量
        self.core_tensor = self.tensor[np.ix_(*self.sampled_indices)]
        
        # 计算交矩阵
        for mode in range(self.n_modes):
            # 伪逆投影
            mode_indices = self.sampled_indices[mode]
            other_indices = [self.sampled_indices[i] for i in range(self.n_modes) if i != mode]
            
            # 构建投影矩阵
            U = self._compute_intersection_matrix(mode, mode_indices, other_indices)
            self.intersection_matrices.append(U)
        
        return self.core_tensor, self.intersection_matrices, self.sampled_indices
    
    def _compute_intersection_matrix(self, mode, mode_idx, other_idx):
        """计算交矩阵"""
        # 简化实现：使用随机投影
        return np.random.randn(self.sample_sizes[mode], self.tensor.shape[mode]) * 0.01
    
    def reconstruct(self) -> np.ndarray:
        """
        从CUR分解重建张量
        """
        result = self.core_tensor.copy()
        
        for mode in range(self.n_modes):
            # 模-n乘积
            result = np.tensordot(result, self.intersection_matrices[mode], axes=([0], [0]))
            # 重新排列轴
            result = np.moveaxis(result, -1, 0)
        
        return result


def tensor_cur_sample(tensor: np.ndarray, 
                      sample_sizes: List[int]) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    简化的张量CUR采样
    
    参数:
        tensor: 输入张量
        sample_sizes: 各模态采样数
    
    返回:
        core: 采样的核心张量
        indices: 各模态采样索引
    """
    n_modes = tensor.ndim
    indices = []
    
    for mode in range(n_modes):
        # 模-n展开
        shape = tensor.shape
        perm = [mode] + [i for i in range(n_modes) if i != mode]
        unfolded = np.transpose(tensor, perm).reshape(shape[mode], -1)
        
        # SVD计算杠杆得分
        k = min(sample_sizes[mode] * 2, unfolded.shape[1])
        U, s, Vh = np.linalg.svd(unfolded, full_matrices=False)
        
        leverage = np.sum(Vh[:k].T ** 2, axis=1) / k
        probs = leverage / leverage.sum()
        
        # 采样
        idx = np.random.choice(shape[mode], size=sample_sizes[mode], 
                               replace=False, p=probs)
        indices.append(idx)
    
    # 提取核心张量
    core = tensor[np.ix_(*indices)]
    
    return core, indices
```

### 3.4 tCURLoRA实现

```python
class TCURLoRALayer(nn.Module):
    """
    tCURLoRA层：结合张量CUR和LoRA
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 tensor_shape: Tuple[int, ...],
                 cur_samples: List[int],
                 lora_rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        """
        参数:
            in_features: 输入维度
            out_features: 输出维度
            tensor_shape: 权重张量形状
            cur_samples: 各模态CUR采样数
            lora_rank: LoRA秩
            alpha: 缩放因子
            dropout: dropout率
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tensor_shape = tensor_shape
        self.n_modes = len(tensor_shape)
        self.cur_samples = cur_samples
        self.lora_rank = lora_rank
        self.alpha = alpha
        self.scaling = alpha / lora_rank
        
        # 模态相关的LoRA
        self.lora_A_list = nn.ModuleList([
            nn.Linear(cur_samples[i], lora_rank, bias=False)
            for i in range(self.n_modes)
        ])
        
        self.lora_B_list = nn.ModuleList([
            nn.Linear(lora_rank, cur_samples[i], bias=False)
            for i in range(self.n_modes)
        ])
        
        # 可学习的核心张量（采样后的子张量）
        core_size = int(np.prod(cur_samples))
        self.core = nn.Parameter(torch.randn(core_size) * 0.01)
        
        # 采样索引（预计算）
        self.register_buffer('sample_indices', self._compute_sample_indices())
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for lora_A in self.lora_A_list:
            nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
        for lora_B in self.lora_B_list:
            nn.init.zeros_(lora_B.weight)
    
    def _compute_sample_indices(self) -> torch.Tensor:
        """计算CUR采样索引"""
        indices = []
        for i, (size, sample) in enumerate(zip(self.tensor_shape, self.cur_samples)):
            # 均匀采样（实际中可以用杠杆得分）
            idx = torch.linspace(0, size - 1, sample).long()
            indices.append(idx)
        return torch.stack(indices, dim=0)
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: (batch, in_features)
            weight: 原始权重 (out_features, in_features)
        
        返回:
            (batch, out_features)
        """
        batch_size = x.shape[0]
        
        # 使用可学习核心张量和LoRA调制
        # 简化版本：直接使用线性变换
        x_dropped = self.dropout(x)
        
        # 逐模态处理
        delta = torch.zeros(batch_size, self.out_features, device=x.device)
        
        # 使用第一个模态的LoRA（简化实现）
        # 完整实现需要张量收缩
        for i, (lora_A, lora_B) in enumerate(zip(self.lora_A_list, self.lora_B_list)):
            # 调整输入维度
            if i == 0:
                temp = x_dropped @ lora_A.weight.T
                temp = temp @ lora_B.weight.T
                # 扩展到输出维度
                if temp.shape[1] < self.out_features:
                    temp = F.pad(temp, (0, self.out_features - temp.shape[1]))
                else:
                    temp = temp[:, :self.out_features]
                delta = delta + temp * self.scaling
        
        return delta
    
    def get_delta_weight(self) -> torch.Tensor:
        """
        获取权重更新量（用于合并）
        """
        # 简化实现
        delta = self.lora_B_list[0].weight @ self.lora_A_list[0].weight * self.scaling
        
        # 填充到完整尺寸
        full_delta = torch.zeros(self.out_features, self.in_features, device=delta.device)
        
        # 使用采样索引放置
        out_idx = self.sample_indices[0][:min(delta.shape[0], self.out_features)]
        in_idx = self.sample_indices[1][:min(delta.shape[1], self.in_features)]
        
        full_delta[out_idx[:delta.shape[0]][:, None], in_idx[:delta.shape[1]][None, :]] = delta
        
        return full_delta


class TCURLoRALinear(nn.Module):
    """
    带tCURLoRA的线性层
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 cur_samples: List[int] = None,
                 lora_rank: int = 8,
                 alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        
        # 原始线性层
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        self.linear.bias.requires_grad = False
        
        # 确定张量形状（假设将权重重塑为张量）
        if cur_samples is None:
            # 自动确定：寻找接近的因子分解
            factors = _factorize(in_features * out_features)
            # 简化：使用2D
            cur_samples = [min(in_features, 64), min(out_features, 64)]
        
        tensor_shape = (out_features, in_features)
        
        # tCURLoRA层
        self.tcurlora = TCURLoRALayer(
            in_features, out_features, tensor_shape,
            cur_samples, lora_rank, alpha, dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.linear(x)
        lora_output = self.tcurlora(x, self.linear.weight)
        return base_output + lora_output


def _factorize(n: int) -> List[int]:
    """将数分解为接近的因子"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors
```

### 3.5 模型转换工具

```python
def convert_to_tcurlora(model: nn.Module, 
                        target_modules: List[str],
                        cur_samples: List[int] = None,
                        lora_rank: int = 8,
                        alpha: float = 16.0) -> nn.Module:
    """
    将模型中的线性层转换为tCURLoRA
    
    参数:
        model: 原始模型
        target_modules: 目标模块名称列表
        cur_samples: CUR采样数
        lora_rank: LoRA秩
        alpha: 缩放因子
    
    返回:
        转换后的模型
    """
    for name, module in model.named_modules():
        # 检查是否为目标模块
        is_target = any(target in name for target in target_modules)
        
        if is_target and isinstance(module, nn.Linear):
            # 创建替换层
            new_module = TCURLoRALinear(
                module.in_features,
                module.out_features,
                cur_samples,
                lora_rank,
                alpha
            )
            
            # 复制权重
            new_module.linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                new_module.linear.bias.data = module.bias.data.clone()
            
            # 替换
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            
            setattr(parent, child_name, new_module)
    
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    计算参数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_trainable_parameters(model: nn.Module):
    """
    打印可训练参数统计
    """
    trainable = 0
    total = 0
    
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            print(f"  {name}: {param.shape}, {param.numel()}")
    
    print(f"\n可训练参数: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"总参数: {total:,}")
```

### 3.6 完整示例

```python
class SimpleTransformer(nn.Module):
    """
    简单Transformer用于演示
    """
    
    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)


def demo_tcurlora():
    """
    tCURLoRA演示
    """
    print("=" * 60)
    print("tCURLoRA微调演示")
    print("=" * 60)
    
    # 创建模型
    vocab_size = 10000
    d_model = 256
    n_heads = 8
    n_layers = 4
    
    model = SimpleTransformer(vocab_size, d_model, n_heads, n_layers)
    
    print(f"\n原始模型:")
    print(f"总参数: {count_parameters(model, trainable_only=False):,}")
    
    # 转换为tCURLoRA
    target_modules = ['output', 'in_proj', 'out_proj', 'linear1', 'linear2']
    
    model_tcurlora = convert_to_tcurlora(
        model,
        target_modules=target_modules,
        cur_samples=[64, 64],  # CUR采样数
        lora_rank=8,
        alpha=16.0
    )
    
    print(f"\ntCURLoRA模型:")
    print_trainable_parameters(model_tcurlora)
    
    # 测试前向传播
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        output = model_tcurlora(x)
    
    print(f"\n输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    return model_tcurlora


def demo_lora_comparison():
    """
    对比LoRA和tCURLoRA
    """
    print("\n" + "=" * 60)
    print("LoRA vs tCURLoRA 对比")
    print("=" * 60)
    
    in_features = 1024
    out_features = 1024
    batch_size = 32
    rank = 8
    
    # 标准线性层
    linear = nn.Linear(in_features, out_features)
    
    # LoRA
    lora_linear = LoRALinear(in_features, out_features, rank=rank, alpha=16.0)
    lora_linear.linear.weight.data = linear.weight.data.clone()
    
    # tCURLoRA
    tcurlora_linear = TCURLoRALinear(
        in_features, out_features,
        cur_samples=[64, 64],
        lora_rank=rank, alpha=16.0
    )
    tcurlora_linear.linear.weight.data = linear.weight.data.clone()
    
    # 参数统计
    print(f"\n参数对比:")
    print(f"  原始: {count_parameters(linear, False):,}")
    print(f"  LoRA: {count_parameters(lora_linear, True):,}")
    print(f"  tCURLoRA: {count_parameters(tcurlora_linear, True):,}")
    
    # 测试输入
    x = torch.randn(batch_size, in_features)
    
    with torch.no_grad():
        out_linear = linear(x)
        out_lora = lora_linear(x)
        out_tcurlora = tcurlora_linear(x)
    
    print(f"\n输出形状: {out_linear.shape}")
    print(f"LoRA输出范围: [{out_lora.min():.4f}, {out_lora.max():.4f}]")
    print(f"tCURLoRA输出范围: [{out_tcurlora.min():.4f}, {out_tcurlora.max():.4f}]")


def demo_cur_decomposition():
    """
    CUR分解演示
    """
    print("\n" + "=" * 60)
    print("CUR分解演示")
    print("=" * 60)
    
    # 创建测试矩阵
    np.random.seed(42)
    m, n, rank = 100, 80, 10
    
    # 低秩矩阵
    U = np.random.randn(m, rank)
    V = np.random.randn(rank, n)
    M = U @ V
    
    # 添加噪声
    M_noisy = M + 0.01 * np.random.randn(m, n)
    
    print(f"矩阵形状: {M.shape}")
    print(f"真实秩: {rank}")
    
    # CUR分解
    C, U_cur, R, col_idx, row_idx = cur_decomposition(M_noisy, rank)
    
    # 重建
    M_reconstructed = C @ U_cur @ R
    
    # 误差
    error = np.linalg.norm(M - M_reconstructed, 'fro') / np.linalg.norm(M, 'fro')
    print(f"\n相对重建误差: {error:.4f}")
    print(f"采样列数: {len(col_idx)}")
    print(f"采样行数: {len(row_idx)}")
    
    # 对比SVD
    U_svd, s, Vh = np.linalg.svd(M_noisy, full_matrices=False)
    M_svd = U_svd[:, :rank] @ np.diag(s[:rank]) @ Vh[:rank]
    error_svd = np.linalg.norm(M - M_svd, 'fro') / np.linalg.norm(M, 'fro')
    print(f"SVD重建误差: {error_svd:.4f}")


if __name__ == "__main__":
    demo_tcurlora()
    demo_lora_comparison()
    demo_cur_decomposition()
```

---

## 4. 实验指南

### 4.1 数据集

| 任务 | 数据集 | 评估指标 |
|------|--------|----------|
| 文本分类 | GLUE, SST-2 | Accuracy |
| 问答 | SQuAD | F1 |
| 生成 | GSM8K | Accuracy |
| 翻译 | WMT | BLEU |

### 4.2 超参数推荐

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| LoRA秩 $r$ | 4-64 | 越大越强但更贵 |
| 缩放因子 $\alpha$ | $2r$ | 通常设为2倍秩 |
| CUR采样数 | 32-128 | 平衡精度和效率 |
| 学习率 | 1e-4 到 5e-4 | 比全参数大 |

### 4.3 评估脚本

```python
def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), 
                                   labels.view(-1))
            
            total_loss += loss.item()
            total_correct += (outputs.argmax(-1) == labels).sum().item()
            total_samples += labels.numel()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_correct / total_samples
    }
```

### 4.4 消融实验

- 不同LoRA秩
- 不同CUR采样率
- 目标模块选择
- 初始化策略

---

## 5. 习题与答案

### 5.1 理论题

**题目1**: 分析LoRA为什么使用零初始化 $B$。

**答案**:
1. **训练稳定性**：初始时 $\Delta W = B_0 A = 0$，输出与预训练模型相同
2. **渐进学习**：从预训练状态开始微调，避免大扰动
3. **避免灾难性遗忘**：初始不偏离预训练表示
4. 如果 $B$ 随机初始化，初始输出会偏离，需要更多训练

**题目2**: 解释CUR分解相对于SVD的可解释性优势。

**答案**:
1. **实际基底**：CUR使用原矩阵的行/列，SVD使用抽象向量
2. **稀疏保持**：原矩阵稀疏则C、R也稀疏，SVD会破坏稀疏性
3. **特征对应**：采样的列对应实际数据特征（如词向量）
4. **可追溯性**：可以追踪每个采样元素的影响

**题目3**: 推导tCURLoRA的参数量。

**答案**:
对于权重 $W \in \mathbb{R}^{d \times k}$：

标准LoRA参数量：
$$P_{\text{LoRA}} = r \cdot (d + k)$$

tCURLoRA（2D张量）：
- CUR采样：$c_d$ 行，$c_k$ 列
- 核心张量：$c_d \times c_k$
- 各模态LoRA：$r \cdot c_d + r \cdot c_k$

$$P_{\text{tCUR}} = c_d \cdot c_k + r \cdot (c_d + c_k)$$

当 $c_d, c_k \ll d, k$ 且 $r$ 较小时：
$$P_{\text{tCUR}} \approx P_{\text{LoRA}} \cdot \frac{c}{d}$$

### 5.2 编程题

**题目1**: 实现AdaLoRA（自适应秩LoRA）。

**答案**:
```python
class AdaLoRALayer(nn.Module):
    """自适应秩LoRA"""
    
    def __init__(self, in_features, out_features, max_rank=16, alpha=32.0):
        super().__init__()
        self.max_rank = max_rank
        
        # 可学习的秩重要性
        self.rank_importance = nn.Parameter(torch.ones(max_rank))
        
        # LoRA矩阵
        self.lora_A = nn.Parameter(torch.randn(max_rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, max_rank))
        
        self.scaling = alpha / max_rank
    
    def forward(self, x):
        # 使用sigmoid归一化重要性
        importance = torch.sigmoid(self.rank_importance)
        
        # 加权的低秩投影
        weighted_A = self.lora_A * importance.unsqueeze(1)
        weighted_B = self.lora_B * importance.unsqueeze(0)
        
        delta = x @ weighted_A.T @ weighted_B.T
        return delta * self.scaling
    
    def prune_rank(self, threshold=0.1):
        """剪枝不重要的秩"""
        with torch.no_grad():
            importance = torch.sigmoid(self.rank_importance)
            mask = importance > threshold
            # 应用mask
            self.lora_A.data *= mask.unsqueeze(1).float()
            self.lora_B.data *= mask.unsqueeze(0).float()
```

**题目2**: 实现多模态tCURLoRA。

**答案**:
```python
class MultimodalTCURLoRA(nn.Module):
    """多模态tCURLoRA"""
    
    def __init__(self, modality_dims, shared_rank=8, modality_ranks=None):
        """
        参数:
            modality_dims: 各模态维度字典 {'text': 768, 'image': 1024}
            shared_rank: 共享秩
            modality_ranks: 各模态特定秩
        """
        super().__init__()
        self.modalities = list(modality_dims.keys())
        
        # 共享的低秩适应
        self.shared_A = nn.ParameterDict({
            m: nn.Parameter(torch.randn(shared_rank, dim) * 0.01)
            for m, dim in modality_dims.items()
        })
        
        # 模态特定的CUR采样
        self.cur_indices = nn.ParameterDict({
            m: nn.Parameter(torch.arange(min(64, dim)), requires_grad=False)
            for m, dim in modality_dims.items()
        })
        
        # 交叉注意力融合
        self.fusion = nn.MultiheadAttention(shared_rank, num_heads=4)
    
    def forward(self, modality_inputs):
        """
        参数:
            modality_inputs: 各模态输入字典
        """
        # 各模态投影
        projected = {}
        for m, x in modality_inputs.items():
            # CUR采样
            idx = self.cur_indices[m]
            x_sampled = x[..., idx] if x.dim() > 1 else x[idx]
            
            # LoRA投影
            projected[m] = x_sampled @ self.shared_A[m].T
        
        # 融合
        stacked = torch.stack([projected[m] for m in self.modalities], dim=0)
        fused, _ = self.fusion(stacked, stacked, stacked)
        
        return fused
```

---

## 参考文献

1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *arXiv:2106.09685*.

2. Mahoney, M. W., & Drineas, P. (2009). CUR matrix decompositions for improved data analysis. *PNAS*, 106(3), 697-702.

3. Cai, X., et al. (2024). tCURLoRA: Tensor CUR Low-Rank Adaptation for Large Models.

4. Zhang, Q., et al. (2023). AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. *ACL*.
