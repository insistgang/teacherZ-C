# Neural Varifold神经变分教程

## 目录
1. [理论讲解](#1-理论讲解)
2. [算法详解](#2-算法详解)
3. [代码实现](#3-代码实现)
4. [实验指南](#4-实验指南)
5. [习题与答案](#5-习题与答案)

---

## 1. 理论讲解

### 1.1 Varifold基础

**Varifold的定义**

Varifold（变分流形）是测度论和几何分析中的重要概念，推广了流形和测度的概念。在图像处理中，Varifold用于描述具有几何结构的信号。

**数学定义**

Varifold $V$ 是一个测度，定义在乘积空间 $\Omega \times G$ 上：
$$V \in \mathcal{M}(\Omega \times G)$$

其中：
- $\Omega$：位置空间（图像定义域）
- $G$：方向/梯度空间（如单位球 $S^{d-1}$）

**离散表示**

对于离散点集 $\{x_i, n_i\}_{i=1}^N$：
$$V = \sum_{i=1}^{N} m_i \delta_{x_i, n_i}$$

其中 $m_i$ 为质量权重，$\delta$ 为Dirac测度。

### 1.2 Varifold度量

**Hilbert空间嵌入**

通过核函数将Varifold嵌入到再生核Hilbert空间（RKHS）：

$$\langle V_1, V_2 \rangle_{\mathcal{H}^*} = \int_{\Omega \times G} \int_{\Omega \times G} K((x, n), (y, m)) dV_1(x, n) dV_2(y, m)$$

**核函数设计**

Varifold核通常分解为位置核和方向核的乘积：
$$K((x, n), (y, m)) = K_{\text{space}}(x, y) \cdot K_{\text{orient}}(n, m)$$

**位置核**

常用高斯核：
$$K_{\text{space}}(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma_s^2}\right)$$

**方向核**

1. **无方向性**：$K_{\text{orient}}(n, m) = 1$

2. **方向敏感**：
$$K_{\text{orient}}(n, m) = (n \cdot m)^2$$

3. **旋转不变**：
$$K_{\text{orient}}(n, m) = (n \cdot m)^{2k}$$

### 1.3 神经Varifold

**核心思想**

将Varifold表示与神经网络结合：
1. 用神经网络学习Varifold的特征表示
2. 端到端优化几何匹配任务
3. 可微分，支持反向传播

**神经嵌入**

$$\Phi_\theta(V) = \sum_{i=1}^{N} m_i \cdot \phi_\theta(x_i, n_i)$$

其中 $\phi_\theta: \Omega \times G \to \mathbb{R}^d$ 是神经网络编码器。

**神经Varifold度量**

$$d_\theta(V_1, V_2) = \|\Phi_\theta(V_1) - \Phi_\theta(V_2)\|^2$$

### 1.4 变分公式

**形状匹配问题**

给定两个形状 $S_1, S_2$，寻找最优变形 $\phi$ 使得变形后的 $S_1$ 与 $S_2$ 匹配：

$$\min_\phi E(\phi) = \frac{1}{2}\|v\|_{V}^2 + \lambda \cdot \|\phi_\# V_1 - V_2\|_{\mathcal{H}^*}^2$$

其中：
- $v$：速度场，$\phi$ 由 $v$ 生成
- $\phi_\# V_1$：推前测度
- $\|\cdot\|_V$：速度场的正则化

**Euler-Lagrange方程**

对速度场求变分得到梯度下降方向：
$$\partial_t v = -\frac{\delta E}{\delta v}$$

### 1.5 与传统方法的关系

**与形状上下文的比较**

| 方法 | 表示 | 度量 | 优化 |
|------|------|------|------|
| 形状上下文 | 直方图 | $\chi^2$距离 | 离散匹配 |
| Varifold | 测度 | RKHS范数 | 连续优化 |
| 神经Varifold | 学习嵌入 | 神经距离 | 端到端 |

**与点云网络的关系**

神经Varifold可以看作：
- PointNet的几何扩展
- 加入方向信息的点云处理
- 可微分的形状表示

---

## 2. 算法详解

### 2.1 Varifold特征提取

**输入**：点云或形状边界 $\{x_i, n_i\}_{i=1}^N$

**步骤**：

```
1. 位置编码:
   p_i = MLP_position(x_i)

2. 方向编码:
   d_i = MLP_direction(n_i)

3. 特征融合:
   f_i = concat(p_i, d_i) 或 p_i * d_i

4. 聚合:
   V = mean(f_i) 或 weighted_sum(f_i, m_i)
```

### 2.2 Varifold匹配算法

**目标**：匹配两个形状 $S_1$ 和 $S_2$

**算法**：

```
输入: 形状1的点 {x_i^1, n_i^1}, 形状2的点 {x_j^2, n_j^2}
输出: 匹配矩阵 P

1. 计算Varifold嵌入:
   V1 = VarifoldEmbed(S1)
   V2 = VarifoldEmbed(S2)

2. 计算点级特征:
   F1 = PointFeatures(S1)  # (N1, d)
   F2 = PointFeatures(S2)  # (N2, d)

3. 计算相似度矩阵:
   S = F1 @ F2.T  # (N1, N2)

4. 最优传输:
   P = Sinkhorn(exp(S/τ), ε)

返回 P
```

### 2.3 可微分Varifold层

**前向传播**

```python
def forward(points, normals, weights):
    # 位置编码
    pos_embed = position_mlp(points)  # (N, d1)
    
    # 方向编码
    dir_embed = direction_mlp(normals)  # (N, d2)
    
    # Varifold特征
    varifold_features = pos_embed * dir_embed  # (N, d)
    
    # 加权聚合
    varifold_vector = (weights[:, None] * varifold_features).sum(dim=0)
    
    return varifold_vector
```

**反向传播**

由于所有操作都是可微的，梯度可以直接回传：
- $\frac{\partial L}{\partial \text{weights}}$
- $\frac{\partial L}{\partial \text{points}}$
- $\frac{\partial L}{\partial \text{normals}}$

### 2.4 Sinkhorn算法

**熵正则化最优传输**

$$\min_P \langle P, C \rangle - \epsilon H(P)$$
$$\text{s.t. } P\mathbf{1} = \mu, P^\top\mathbf{1} = \nu$$

**迭代更新**

```
初始化: K = exp(-C/ε)

Repeat:
    u = μ / (K @ v)
    v = ν / (K.T @ u)

Until 收敛

传输矩阵: P = diag(u) @ K @ diag(v)
```

### 2.5 训练策略

**对比学习**

$$\mathcal{L} = \sum_{(i,j) \in \mathcal{P}} d(V_i, V_j) - \sum_{(i,k) \in \mathcal{N}} d(V_i, V_k) + \text{margin}$$

**形状重建**

$$\mathcal{L} = \|\text{Decode}(\text{Varifold}(S)) - S\|^2$$

**匹配监督**

$$\mathcal{L} = \|P - P^*\|_F^2$$

---

## 3. 代码实现

### 3.1 Varifold核函数

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def gaussian_kernel(x: torch.Tensor, 
                    y: torch.Tensor, 
                    sigma: float) -> torch.Tensor:
    """
    高斯位置核
    
    参数:
        x: (N, D) 位置张量
        y: (M, D) 位置张量
        sigma: 带宽
    
    返回:
        (N, M) 核矩阵
    """
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # (N, M, D)
    dist_sq = (diff ** 2).sum(dim=-1)  # (N, M)
    return torch.exp(-dist_sq / (2 * sigma ** 2))


def orientation_kernel(n: torch.Tensor, 
                       m: torch.Tensor, 
                       k: int = 1) -> torch.Tensor:
    """
    方向核
    
    参数:
        n: (N, D) 法向量张量
        m: (M, D) 法向量张量
        k: 幂次参数
    
    返回:
        (N, M) 方向相似度矩阵
    """
    cos_angle = (n.unsqueeze(1) * m.unsqueeze(0)).sum(dim=-1)
    return (cos_angle ** 2) ** k  # (n·m)^{2k}


def varifold_kernel(x: torch.Tensor, n: torch.Tensor,
                    y: torch.Tensor, m: torch.Tensor,
                    sigma_s: float, k: int = 1) -> torch.Tensor:
    """
    完整Varifold核
    
    参数:
        x, n: 第一组点的位置和法向量
        y, m: 第二组点的位置和法向量
        sigma_s: 空间带宽
        k: 方向敏感度
    
    返回:
        (N, M) Varifold核矩阵
    """
    K_space = gaussian_kernel(x, y, sigma_s)
    K_orient = orientation_kernel(n, m, k)
    return K_space * K_orient
```

### 3.2 Varifold表示

```python
class VarifoldRepresentation:
    """
    Varifold表示类
    """
    
    def __init__(self, 
                 points: np.ndarray, 
                 normals: Optional[np.ndarray] = None,
                 weights: Optional[np.ndarray] = None):
        """
        初始化Varifold表示
        
        参数:
            points: (N, D) 点位置
            normals: (N, D) 法向量（可选）
            weights: (N,) 权重（可选）
        """
        self.points = np.asarray(points)
        self.n_points = len(points)
        
        if normals is None:
            self.normals = np.ones_like(points)
        else:
            self.normals = np.asarray(normals)
        
        if weights is None:
            self.weights = np.ones(self.n_points) / self.n_points
        else:
            self.weights = np.asarray(weights)
    
    def compute_kernel_matrix(self, 
                               other: 'VarifoldRepresentation',
                               sigma_s: float = 1.0,
                               k: int = 1) -> np.ndarray:
        """
        计算与另一个Varifold的核矩阵
        """
        K_space = self._gaussian_kernel(self.points, other.points, sigma_s)
        K_orient = self._orientation_kernel(self.normals, other.normals, k)
        return K_space * K_orient
    
    def inner_product(self, 
                      other: 'VarifoldRepresentation',
                      sigma_s: float = 1.0,
                      k: int = 1) -> float:
        """
        计算Varifold内积
        """
        K = self.compute_kernel_matrix(other, sigma_s, k)
        return np.sum(self.weights[:, None] * K * other.weights[None, :])
    
    def squared_norm(self, sigma_s: float = 1.0, k: int = 1) -> float:
        """
        计算Varifold范数的平方
        """
        return self.inner_product(self, sigma_s, k)
    
    def distance(self, 
                 other: 'VarifoldRepresentation',
                 sigma_s: float = 1.0,
                 k: int = 1) -> float:
        """
        计算Varifold距离
        """
        n1 = self.squared_norm(sigma_s, k)
        n2 = other.squared_norm(sigma_s, k)
        inner = self.inner_product(other, sigma_s, k)
        return np.sqrt(max(n1 + n2 - 2 * inner, 0))
    
    @staticmethod
    def _gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
        diff = x[:, None, :] - y[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        return np.exp(-dist_sq / (2 * sigma ** 2))
    
    @staticmethod
    def _orientation_kernel(n: np.ndarray, m: np.ndarray, k: int) -> np.ndarray:
        cos_angle = np.sum(n[:, None, :] * m[None, :, :], axis=-1)
        return (cos_angle ** 2) ** k


def compute_varifold_descriptor(points: np.ndarray,
                                 normals: Optional[np.ndarray] = None,
                                 landmarks: Optional[np.ndarray] = None,
                                 sigma_s: float = 1.0) -> np.ndarray:
    """
    计算Varifold描述子
    
    参数:
        points: (N, D) 点
        normals: (N, D) 法向量
        landmarks: (M, D) 参考点
        sigma_s: 带宽
    
    返回:
        (M,) 或 (M, M) 描述子
    """
    if normals is None:
        normals = np.ones_like(points)
    
    if landmarks is None:
        landmarks = points
    
    weights = np.ones(len(points)) / len(points)
    
    V = VarifoldRepresentation(points, normals, weights)
    
    # 计算与每个landmark的内积
    descriptor = np.zeros(len(landmarks))
    for i, lm in enumerate(landmarks):
        lm_V = VarifoldRepresentation(lm.reshape(1, -1), 
                                      np.ones((1, points.shape[1])))
        descriptor[i] = V.inner_product(lm_V, sigma_s)
    
    return descriptor
```

### 3.3 神经Varifold网络

```python
class PositionalEncoding(nn.Module):
    """
    位置编码模块
    """
    
    def __init__(self, d_model: int, max_freq: int = 10):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        
        # 频率参数
        freqs = 2.0 ** torch.linspace(0, max_freq - 1, d_model // 2)
        self.register_buffer('freqs', freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (B, N, 3) 或 (N, 3) 位置
        
        返回:
            (B, N, d_model) 或 (N, d_model) 编码
        """
        orig_shape = x.shape
        if len(orig_shape) == 2:
            x = x.unsqueeze(0)
        
        B, N, D = x.shape
        
        # 正弦/余弦编码
        x_freq = x.unsqueeze(-1) * self.freqs.view(1, 1, 1, -1)  # (B, N, D, d_model//2)
        sin_enc = torch.sin(x_freq)
        cos_enc = torch.cos(x_freq)
        enc = torch.cat([sin_enc, cos_enc], dim=-1)  # (B, N, D, d_model)
        
        enc = enc.reshape(B, N, -1)
        
        if len(orig_shape) == 2:
            enc = enc.squeeze(0)
        
        return enc


class NeuralVarifoldEncoder(nn.Module):
    """
    神经Varifold编码器
    """
    
    def __init__(self, 
                 pos_dim: int = 3,
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 use_direction: bool = True):
        super().__init__()
        
        self.use_direction = use_direction
        
        # 位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        if use_direction:
            # 方向编码
            self.dir_encoder = nn.Sequential(
                nn.Linear(pos_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2 if use_direction else output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, 
                points: torch.Tensor, 
                normals: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            points: (B, N, 3) 点位置
            normals: (B, N, 3) 法向量
            weights: (B, N) 权重
        
        返回:
            (B, output_dim) Varifold嵌入
        """
        B, N, _ = points.shape
        
        # 位置特征
        pos_feat = self.pos_encoder(points)  # (B, N, output_dim)
        
        if self.use_direction and normals is not None:
            # 方向特征
            dir_feat = self.dir_encoder(normals)  # (B, N, output_dim)
            
            # 融合
            combined = torch.cat([pos_feat, dir_feat], dim=-1)
            point_features = self.fusion(combined)
        else:
            point_features = pos_feat
        
        # 加权聚合
        if weights is None:
            weights = torch.ones(B, N, device=points.device) / N
        
        weights = weights.unsqueeze(-1)  # (B, N, 1)
        varifold_embed = (weights * point_features).sum(dim=1)  # (B, output_dim)
        
        # 归一化
        varifold_embed = F.normalize(varifold_embed, dim=-1)
        
        return varifold_embed


class NeuralVarifoldNetwork(nn.Module):
    """
    完整的神经Varifold网络
    """
    
    def __init__(self, 
                 pos_dim: int = 3,
                 hidden_dim: int = 128,
                 output_dim: int = 256):
        super().__init__()
        
        self.encoder = NeuralVarifoldEncoder(pos_dim, hidden_dim, output_dim)
        
        # 匹配头
        self.matching_head = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, 
                points1: torch.Tensor, 
                normals1: torch.Tensor,
                points2: torch.Tensor, 
                normals2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算两个形状的Varifold嵌入和匹配分数
        
        返回:
            embed1, embed2: Varifold嵌入
            similarity: 相似度分数
        """
        embed1 = self.encoder(points1, normals1)
        embed2 = self.encoder(points2, normals2)
        
        # 计算相似度
        combined = torch.cat([embed1, embed2], dim=-1)
        similarity = self.matching_head(combined).squeeze(-1)
        
        return embed1, embed2, similarity
    
    def compute_distance(self, 
                         embed1: torch.Tensor, 
                         embed2: torch.Tensor) -> torch.Tensor:
        """
        计算Varifold嵌入间的距离
        """
        return torch.norm(embed1 - embed2, dim=-1)
```

### 3.4 Sinkhorn层

```python
class SinkhornLayer(nn.Module):
    """
    可微分Sinkhorn层
    """
    
    def __init__(self, 
                 n_iter: int = 20, 
                 epsilon: float = 0.1,
                 gamma: float = 1.0):
        super().__init__()
        self.n_iter = n_iter
        self.epsilon = epsilon
        self.gamma = gamma
    
    def forward(self, 
                cost_matrix: torch.Tensor,
                mu: Optional[torch.Tensor] = None,
                nu: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sinkhorn算法
        
        参数:
            cost_matrix: (B, N, M) 代价矩阵
            mu: (B, N) 源分布
            nu: (B, M) 目标分布
        
        返回:
            (B, N, M) 传输矩阵
        """
        B, N, M = cost_matrix.shape
        
        if mu is None:
            mu = torch.ones(B, N, device=cost_matrix.device) / N
        if nu is None:
            nu = torch.ones(B, M, device=cost_matrix.device) / M
        
        # 核矩阵
        K = torch.exp(-cost_matrix / self.epsilon)
        
        # Sinkhorn迭代
        u = torch.ones(B, N, device=cost_matrix.device)
        v = torch.ones(B, M, device=cost_matrix.device)
        
        for _ in range(self.n_iter):
            u = mu / (K @ v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
            v = nu / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        
        # 传输矩阵
        P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
        
        return P
```

### 3.5 形状匹配

```python
class ShapeMatcher:
    """
    基于Varifold的形状匹配器
    """
    
    def __init__(self, 
                 sigma_s: float = 1.0,
                 k: int = 1,
                 n_sinkhorn_iter: int = 20,
                 sinkhorn_eps: float = 0.1):
        self.sigma_s = sigma_s
        self.k = k
        self.sinkhorn = SinkhornLayer(n_sinkhorn_iter, sinkhorn_eps)
    
    def compute_cost_matrix(self, 
                            points1: torch.Tensor, 
                            normals1: torch.Tensor,
                            points2: torch.Tensor, 
                            normals2: torch.Tensor) -> torch.Tensor:
        """
        计算Varifold代价矩阵
        """
        B, N, D = points1.shape
        M = points2.shape[1]
        
        # 展开用于广播
        p1 = points1.unsqueeze(2)  # (B, N, 1, D)
        p2 = points2.unsqueeze(1)  # (B, 1, M, D)
        n1 = normals1.unsqueeze(2)
        n2 = normals2.unsqueeze(1)
        
        # 空间核
        dist_sq = ((p1 - p2) ** 2).sum(dim=-1)  # (B, N, M)
        K_space = torch.exp(-dist_sq / (2 * self.sigma_s ** 2))
        
        # 方向核
        cos_angle = (n1 * n2).sum(dim=-1)
        K_orient = (cos_angle ** 2) ** self.k
        
        # Varifold核
        K_varifold = K_space * K_orient
        
        # 代价矩阵（负相似度）
        cost_matrix = -K_varifold
        
        return cost_matrix
    
    def match(self, 
              points1: torch.Tensor, 
              normals1: torch.Tensor,
              points2: torch.Tensor, 
              normals2: torch.Tensor) -> torch.Tensor:
        """
        执行形状匹配
        
        返回:
            (B, N, M) 匹配矩阵
        """
        cost_matrix = self.compute_cost_matrix(points1, normals1, points2, normals2)
        transport = self.sinkhorn(cost_matrix)
        return transport
    
    def match_numpy(self,
                    points1: np.ndarray,
                    normals1: np.ndarray,
                    points2: np.ndarray,
                    normals2: np.ndarray) -> np.ndarray:
        """
        NumPy版本的匹配
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        p1 = torch.from_numpy(points1).float().unsqueeze(0).to(device)
        n1 = torch.from_numpy(normals1).float().unsqueeze(0).to(device)
        p2 = torch.from_numpy(points2).float().unsqueeze(0).to(device)
        n2 = torch.from_numpy(normals2).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            transport = self.match(p1, n1, p2, n2)
        
        return transport.cpu().numpy()[0]
```

### 3.6 完整示例

```python
def demo_varifold_matching():
    """
    Varifold形状匹配演示
    """
    print("=" * 60)
    print("神经Varifold形状匹配演示")
    print("=" * 60)
    
    # 生成测试形状
    np.random.seed(42)
    
    # 形状1：球面上的点
    n_points = 100
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    
    points1 = np.stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ], axis=1)
    normals1 = points1.copy()  # 法向量指向外
    
    # 形状2：变形后的球面
    deformation = np.array([1.5, 1.0, 0.8])  # 各向异性缩放
    points2 = points1 * deformation
    normals2 = normals1.copy()
    
    # 添加噪声
    points2 += 0.05 * np.random.randn(*points2.shape)
    
    print(f"形状1: {n_points} 点")
    print(f"形状2: {n_points} 点")
    
    # 计算Varifold表示
    V1 = VarifoldRepresentation(points1, normals1)
    V2 = VarifoldRepresentation(points2, normals2)
    
    # 计算距离
    distance = V1.distance(V2, sigma_s=0.5)
    print(f"\nVarifold距离: {distance:.4f}")
    
    # 执行匹配
    print("\n执行Sinkhorn匹配...")
    matcher = ShapeMatcher(sigma_s=0.5, n_sinkhorn_iter=50)
    transport = matcher.match_numpy(points1, normals1, points2, normals2)
    
    # 可视化
    visualize_matching(points1, points2, transport)
    
    return transport


def visualize_matching(points1: np.ndarray, 
                       points2: np.ndarray,
                       transport: np.ndarray,
                       n_show: int = 10):
    """
    可视化匹配结果
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 5))
    
    # 形状1
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='blue', s=10)
    ax1.set_title('Shape 1')
    
    # 形状2
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='red', s=10)
    ax2.set_title('Shape 2')
    
    # 匹配
    ax3 = fig.add_subplot(133, projection='3d')
    
    # 显示最强的n_show个匹配
    for i in range(n_show):
        j = np.argmax(transport[i])
        if transport[i, j] > 0.01:
            ax3.plot([points1[i, 0], points2[j, 0]],
                     [points1[i, 1], points2[j, 1]],
                     [points1[i, 2], points2[j, 2]], 
                     'g-', alpha=0.5, linewidth=0.5)
    
    ax3.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='blue', s=5, alpha=0.3)
    ax3.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='red', s=5, alpha=0.3)
    ax3.set_title('Correspondences')
    
    plt.tight_layout()
    plt.show()


def demo_neural_varifold():
    """
    神经Varifold网络演示
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建网络
    model = NeuralVarifoldNetwork(pos_dim=3, hidden_dim=128, output_dim=256).to(device)
    
    # 生成训练数据
    n_samples = 32
    n_points = 64
    
    # 随机点云
    points = torch.randn(n_samples, n_points, 3)
    normals = F.normalize(torch.randn(n_samples, n_points, 3), dim=-1)
    
    # 移到设备
    points = points.to(device)
    normals = normals.to(device)
    
    # 前向传播
    with torch.no_grad():
        embed1, embed2, sim = model(points[:16], normals[:16], 
                                     points[16:], normals[16:])
    
    print(f"嵌入维度: {embed1.shape}")
    print(f"相似度范围: [{sim.min():.4f}, {sim.max():.4f}]")
    
    # 计算距离
    distances = model.compute_distance(embed1, embed2)
    print(f"Varifold距离: {distances.mean():.4f} ± {distances.std():.4f}")


if __name__ == "__main__":
    demo_varifold_matching()
    demo_neural_varifold()
```

---

## 4. 实验指南

### 4.1 数据集

| 数据集 | 类型 | 用途 |
|--------|------|------|
| FAUST | 人体网格 | 形状对应 |
| SHREC'16 | 形状匹配 | 基准测试 |
| ModelNet40 | 点云分类 | 特征学习 |
| ShapeNet | 3D形状 | 大规模评估 |

### 4.2 评估指标

```python
def evaluate_correspondence(pred_match, gt_match):
    """
    评估形状对应
    
    参数:
        pred_match: (N,) 预测对应索引
        gt_match: (N,) 真实对应索引
    """
    # 正确率
    accuracy = np.mean(pred_match == gt_match)
    
    # 平均测地误差
    # geodesic_error = ...
    
    return {
        'accuracy': accuracy,
        # 'geodesic_error': geodesic_error
    }
```

### 4.3 参数调优

- $\sigma_s$（空间带宽）：与形状尺度相关
- $k$（方向敏感度）：通常1-2
- $\epsilon$（Sinkhorn正则化）：0.05-0.5

### 4.4 可视化

- 点云匹配连线
- 传输矩阵热力图
- 嵌入空间t-SNE

---

## 5. 习题与答案

### 5.1 理论题

**题目1**: 解释Varifold核为何使用 $(n \cdot m)^{2k}$ 而非 $(n \cdot m)^k$。

**答案**:
1. **方向模糊性**：法向量 $n$ 和 $-n$ 表示相同曲面，需要旋转不变
2. $(n \cdot m)^{2k}$ 对 $n \to -n$ 不变
3. 当 $k=1$，$(n \cdot m)^2 = \cos^2\theta$，在0°和180°时相等
4. 确保相同曲面的不同朝向有相同的Varifold表示

**题目2**: 证明Varifold范数的非负性。

**答案**:
Varifold范数通过正定核定义：
$$\|V\|^2 = \int\int K((x,n),(y,m)) dV(x,n) dV(y,m)$$

对于高斯核和方向核的乘积：
1. 高斯核正定：$K_{space}(x,y) = \exp(-\|x-y\|^2/2\sigma^2)$
2. 方向核半正定：$K_{orient}(n,m) = (n \cdot m)^{2k}$（可分解）
3. 乘积保持正定性
4. 因此 $\|V\|^2 \geq 0$

**题目3**: 比较Sinkhorn算法的计算复杂度。

**答案**:
- 朴素最优传输：$O(n^3)$（线性规划）
- Sinkhorn：$O(n^2 \cdot T)$，$T$为迭代次数
- 对于 $n$ 个点，$T$ 通常20-100次
- 空间复杂度：$O(n^2)$ 存储核矩阵
- 可通过低秩近似加速到 $O(n \cdot r \cdot T)$，$r \ll n$

### 5.2 编程题

**题目1**: 实现Varifold形状插值。

**答案**:
```python
def varifold_interpolation(V1, V2, alpha, sigma_s=1.0):
    """
    Varifold空间中的形状插值
    
    参数:
        V1, V2: 两个Varifold表示
        alpha: 插值系数 [0, 1]
    
    返回:
        插值后的Varifold
    """
    # 在RKHS中的线性插值
    # V_alpha = (1-alpha) * V1 + alpha * V2
    
    # 实际实现：对嵌入向量插值
    embed1 = compute_varifold_embedding(V1, sigma_s)
    embed2 = compute_varifold_embedding(V2, sigma_s)
    
    embed_interp = (1 - alpha) * embed1 + alpha * embed2
    
    return embed_interp


def compute_varifold_embedding(V, sigma_s, landmarks=None):
    """计算Varifold到RKHS的嵌入"""
    if landmarks is None:
        # 使用自身作为landmark
        landmarks = V.points
    
    embed = np.zeros(len(landmarks))
    for i, lm in enumerate(landmarks):
        lm_V = VarifoldRepresentation(lm.reshape(1, -1), 
                                       np.ones((1, 3)))
        embed[i] = V.inner_product(lm_V, sigma_s)
    
    return embed
```

**题目2**: 实现基于Varifold的形状分类器。

**答案**:
```python
class VarifoldClassifier(nn.Module):
    """基于Varifold的形状分类"""
    
    def __init__(self, n_classes, hidden_dim=256, output_dim=128):
        super().__init__()
        
        self.encoder = NeuralVarifoldEncoder(
            pos_dim=3, hidden_dim=hidden_dim, output_dim=output_dim
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, points, normals, weights=None):
        embed = self.encoder(points, normals, weights)
        logits = self.classifier(embed)
        return logits


def train_varifold_classifier(model, train_loader, n_epochs=50):
    """训练分类器"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            points, normals, labels = batch
            
            optimizer.zero_grad()
            logits = model(points, normals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
```

---

## 参考文献

1. Charon, N., & Trouvé, A. (2013). The varifold representation of non-oriented shapes for diffeomorphic registration. *SIAM J. Imaging Sciences*, 6(4), 2547-2580.

2. Kaltenmark, I., Charlier, B., & Charon, N. (2017). A framework for curve and surface comparison using the varifold distances.

3. Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. *NeurIPS*.

4. Qi, C. R., et al. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. *CVPR*.
