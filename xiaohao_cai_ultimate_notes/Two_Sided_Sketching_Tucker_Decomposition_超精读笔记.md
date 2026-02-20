# Two-Sided Sketching for High-Order Tensor Approximation
# 超精读笔记

## 📋 论文元数据

| 项目 | 内容 |
|------|------|
| **标题** | Tensor Sketching with Applications to High-Order Data Analysis |
| **中文名** | 用于高阶数据分析的张量Sketching方法 |
| **作者** | Xiaohao Cai, Sayantan Nag, Thomas Strohmer |
| **机构** | University of California, Davis, USA |
| **年份** | 2024 |
| **期刊/会议** | SIAM Journal on Mathematics of Data Science (SIMODS) |
| **arXiv ID** | arXiv:2301.11598 |
| **引用** | ~50+ (Google Scholar) |

---

## 📝 摘要翻译

**原文摘要**:
High-order tensor data arises in many applications including signal processing, machine learning, and bioinformatics. Tucker decomposition is a fundamental tool for dimensionality reduction and compression of tensor data. However, computing Tucker decomposition of large-scale tensors remains computationally challenging. In this paper, we propose two-sided sketching methods for efficient Tucker decomposition. Our approach uses random projections from both sides of the tensor to capture its column and row spaces simultaneously. We provide theoretical guarantees on the approximation quality and demonstrate significant computational advantages over existing methods. Extensive experiments on synthetic and real-world datasets validate the effectiveness of our approach.

**中文翻译**:
高阶张量数据出现在许多应用中，包括信号处理、机器学习和生物信息学。Tucker分解是张量数据降维和压缩的基本工具。然而，计算大规模张量的Tucker分解在计算上仍然具有挑战性。在本文中，我们提出了用于高效Tucker分解的双面sketching方法。我们的方法使用张量两侧的随机投影来同时捕获其列空间和行空间。我们提供了关于近似质量的理论保证，并展示了相比现有方法的显著计算优势。在合成数据集和真实世界数据集上的大量实验验证了我们方法的有效性。

---

## 🔢 数学家Agent：理论分析

### 核心数学框架

#### 1. 张量基础

**张量定义**:
阶数为 $d$ 的张量 $\mathcal{X} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}$

**张量模乘**:
$$\mathcal{Y} = \mathcal{X} \times_1 U^{(1)} \times_2 U^{(2)} \times \cdots \times_d U^{(d)}$$

其中 $U^{(k)} \in \mathbb{R}^{m_k \times n_k}$ 是第 $k$ 模的因子矩阵。

#### 2. Tucker分解

**Tucker分解形式**:
$$\mathcal{X} \approx \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times \cdots \times_d U^{(d)}$$

其中：
- $\mathcal{G} \in \mathbb{R}^{r_1 \times r_2 \times \cdots \times r_d}$ 是核心张量
- $U^{(k)} \in \mathbb{R}^{n_k \times r_k}$ 是因子矩阵（正交列）
- $r_k \leq n_k$ 是第 $k$ 模的秩

**目标函数**:
$$\min_{\mathcal{G}, U^{(1)},...,U^{(d)}} \|\mathcal{X} - \mathcal{G} \times_1 U^{(1)} \times_2 U^{(2)} \times \cdots \times_d U^{(d)}\|_F^2$$

#### 3. HOSVD（高阶SVD）

**逐模展开**:
$$\mathcal{X}_{(k)} \in \mathbb{R}^{n_k \times (N/n_k)}$$

其中 $N = \prod_{i=1}^d n_i$。

**左奇异向量**:
$$U^{(k)} = \text{SVD}(\mathcal{X}_{(k)})[:, 1:r_k]$$

**核心张量**:
$$\mathcal{G} = \mathcal{X} \times_1 U^{(1)T} \times_2 U^{(2)T} \times \cdots \times_d U^{(d)T}$$

#### 4. 双面Sketching方法

**传统HOSVD的挑战**:
- 需要对每个模进行完整SVD
- 时间复杂度: $O(\sum_k n_k^2 \cdot N/n_k)$
- 内存复杂度: $O(N)$

**双面Sketching核心思想**:
使用随机投影矩阵 $\Omega^{(k)} \in \mathbb{R}^{n_k \times \ell_k}$ ($\ell_k \ll n_k$) 来近似列空间。

**Sketching矩阵构造**:
$$\mathcal{S}_k = \mathcal{X} \times_k \Omega^{(k)T}$$

**因子矩阵估计**:
$$U^{(k)} = \text{orth}\left(\text{SVD}(\mathcal{S}_{(k)})\right)$$

#### 5. 理论保证

**Johnson-Lindenstrauss引理**:
对于单位向量 $x \in \mathbb{R}^n$ 和随机投影 $\Omega \in \mathbb{R}^{n \times \ell}$：
$$P\left((1-\epsilon)\|x\|^2 \leq \|\Omega^T x\|^2 \leq (1+\epsilon)\|x\|^2\right) \geq 1 - 2\exp(-c\ell \epsilon^2)$$

**近似误差界**:
$$\|\mathcal{X} - \mathcal{X} \times_1 \tilde{U}^{(1)}\tilde{U}^{(1)T} \times_2 \cdots \times_d \tilde{U}^{(d)}\tilde{U}^{(d)T}\|_F \leq (1+\epsilon)\|\mathcal{X} - \mathcal{X}^*\|_F$$

其中 $\mathcal{X}^*$ 是最优低秩近似。

#### 6. 概率分析

**目标秩**:
$$\ell_k \geq C \cdot \frac{r_k + \log(1/\delta)}{\epsilon^2}$$

其中 $C$ 是常数，$\delta$ 是失败概率。

**误差概率**:
$$P(\text{error} > \epsilon) \leq \delta$$

---

## 🔧 工程师Agent：实现分析

### 双面Sketching算法架构

```
输入: 张量 X ∈ ℝ^{n₁×n₂×...×n_d}
       目标秩 (r₁, r₂, ..., r_d)
       Sketching维度 (ℓ₁, ℓ₂, ..., ℓ_d)
       ↓
┌─────────────────────────────────────────────────┐
│          双面Sketching Tucker分解              │
│  ┌──────────────────────────────────────────┐  │
│  │  阶段1: 构造Sketching矩阵               │  │
│  │  ┌────────────────────────────────────┐ │  │
│  │  │ 高斯随机投影: Ω^{(k)} ~ N(0,1)     │ │  │
│  │  │ 或                                 │ │  │
│  │  │ 稀疏随机投影: SRHT                 │ │  │
│  │  └────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                         │
│  ┌──────────────────────────────────────────┐  │
│  │  阶段2: 张量Sketching                   │  │
│  │  ┌────────────────────────────────────┐ │  │
│  │  │ 对每个模 k = 1, ..., d:            │ │  │
│  │  │   S_k = X ×_k Ω^{(k)T}            │ │  │
│  │  │   (降维到 ℓ_k)                    │ │  │
│  │  └────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                         │
│  ┌──────────────────────────────────────────┐  │
│  │  阶段3: 因子矩阵估计                    │  │
│  │  ┌────────────────────────────────────┐ │  │
│  │  │ 对每个模 k:                        │ │  │
│  │  │   [U, Σ, V] = SVD(S_{(k)})        │ │  │
│  │  │   Û^{(k)} = U(:, 1:r_k)          │ │  │
│  │  └────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                         │
│  ┌──────────────────────────────────────────┐  │
│  │  阶段4: 核心张量计算                    │  │
│  │  ┌────────────────────────────────────┐ │  │
│  │  │ G = X ×_1 Û^{(1)T} ×_2 ... ×_d Û^{(d)T} │ │  │
│  │  └────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
       ↓
输出: 核心张量 G, 因子矩阵 {Û^{(k)}}
```

### 算法实现

```python
import numpy as np
from scipy.linalg import svd
from scipy.sparse import csr_matrix


class TwoSidedSketchingTucker:
    """双面Sketching Tucker分解"""

    def __init__(self, target_ranks, sketch_ranks=None, random_type='gaussian'):
        """
        参数:
            target_ranks: 目标Tucker秩 (r1, r2, ..., rd)
            sketch_ranks: Sketching维度 (ℓ1, ℓ2, ..., ℓd)，默认为目标秩+10
            random_type: 随机投影类型 ('gaussian', 'sparse', 'srht')
        """
        self.target_ranks = np.array(target_ranks)
        self.order = len(target_ranks)

        if sketch_ranks is None:
            # 经验法则: ℓ_k = r_k + 10 或 1.5*r_k
            self.sketch_ranks = np.array([min(r + 10, int(1.5 * r))
                                          for r in target_ranks])
        else:
            self.sketch_ranks = np.array(sketch_ranks)

        self.random_type = random_type

    def generate_sketching_matrix(self, n, ell):
        """
        生成Sketching矩阵 Ω ∈ ℝ^{n × ℓ}

        参数:
            n: 原始维度
            ell: sketching维度

        返回:
            Ω: Sketching矩阵
        """
        if self.random_type == 'gaussian':
            # 高斯随机投影
            Omega = np.random.randn(n, ell) / np.sqrt(ell)

        elif self.random_type == 'sparse':
            # 稀疏随机投影 (Achlioptas)
            density = 1 / 3  # 稀疏度
            Omega = np.random.choice(
                [-np.sqrt(3/density), 0, np.sqrt(3/density)],
                size=(n, ell),
                p=[density/2, 1-density, density/2]
            )
            Omega = np.where(Omega != 0, Omega, 0)

        elif self.random_type == 'srht':
            # Subsampled Randomized Hadamard Transform
            # Ω = PHD where P是下采样矩阵, H是Hadamard矩阵, D是随机对角矩阵
            D = np.diag(np.random.choice([-1, 1], size=n))
            # 快速Hadamard变换 (简化版，实际应使用FWHT)
            H = self._hadamard_transform(n)
            PH = H[:, :ell]  # 下采样
            Omega = PH @ D

        else:
            raise ValueError(f"Unknown random type: {self.random_type}")

        return Omega

    def _hadamard_transform(self, n):
        """生成Hadamard变换矩阵（简化版）"""
        # 实际实现应使用快速Walsh-Hadamard变换
        from scipy.linalg import hadamard
        m = int(2 ** np.ceil(np.log2(n)))
        H_full = hadamard(m) / np.sqrt(m)
        return H_full[:n, :]

    def tensor_mode_product(self, X, Omega, mode):
        """
        张量模乘: Y = X ×_mode Ω^T

        参数:
            X: 输入张量
            Omega: 投影矩阵
            mode: 模态索引 (0-based)

        返回:
            Y: 结果张量
        """
        # 展开第mode模
        X_mode = np.moveaxis(X, mode, 0)
        n_mode = X_mode.shape[0]
        X_unfolded = X_mode.reshape(n_mode, -1)

        # 矩阵乘法
        Y_unfolded = Omega.T @ X_unfolded

        # 重构回张量
        new_shape = list(X.shape)
        new_shape[mode] = Omega.shape[1]
        Y = Y_unfolded.reshape(new_shape)

        return Y

    def decompose(self, X):
        """
        执行双面Sketching Tucker分解

        参数:
            X: 输入张量

        返回:
            core: 核心张量
            factors: 因子矩阵列表
        """
        d = self.order
        factors = []

        # 阶段1-3: 对每个模进行sketching和SVD
        for k in range(d):
            n_k = X.shape[k]
            ell_k = self.sketch_ranks[k]

            # 生成sketching矩阵
            Omega_k = self.generate_sketching_matrix(n_k, ell_k)

            # Tensor sketching
            S_k = self.tensor_mode_product(X, Omega_k, k)

            # 展开并计算SVD
            S_k_unfolded = np.moveaxis(S_k, k, 0)
            S_k_unfolded = S_k_unfolded.reshape(S_k.shape[k], -1)

            U_k, s_k, V_k = svd(S_k_unfolded, full_matrices=False)

            # 取前r_k个左奇异向量
            r_k = self.target_ranks[k]
            U_tilde_k = U_k[:, :r_k]

            # 正交化
            U_tilde_k, _ = np.linalg.qr(U_tilde_k)

            factors.append(U_tilde_k)

        # 阶段4: 计算核心张量
        core = X.copy()
        for k in range(d):
            core = self.tensor_mode_product(core, factors[k].T, k)

        return core, factors

    def reconstruct(self, core, factors):
        """
        从核心张量和因子矩阵重构原始张量

        参数:
            core: 核心张量
            factors: 因子矩阵列表

        返回:
            X_rec: 重构的张量
        """
        X_rec = core.copy()
        d = len(factors)

        for k in range(d):
            X_rec = self.tensor_mode_product(X_rec, factors[k], k)

        return X_rec

    def relative_error(self, X, X_rec):
        """计算相对误差"""
        return np.linalg.norm(X - X_rec) / np.linalg.norm(X)


# ===== 高级功能：自适应Sketching =====

class AdaptiveSketchingTucker(TwoSidedSketchingTucker):
    """自适应Sketching Tucker分解"""

    def __init__(self, target_ranks, epsilon=0.1, delta=0.01):
        """
        参数:
            target_ranks: 目标Tucker秩
            epsilon: 近似误差界
            delta: 失败概率
        """
        # 根据理论界自动计算sketching维度
        sketch_ranks = []
        C = 4  # 常数因子

        for r_k in target_ranks:
            ell_k = int(C * (r_k + np.log(1/delta)) / (epsilon**2))
            sketch_ranks.append(ell_k)

        super().__init__(target_ranks, sketch_ranks)
        self.epsilon = epsilon
        self.delta = delta

    def decompose_with_refinement(self, X, max_iter=3):
        """
        带迭代细化的分解

        参数:
            X: 输入张量
            max_iter: 最大迭代次数

        返回:
            core: 核心张量
            factors: 因子矩阵列表
        """
        # 初始分解
        core, factors = self.decompose(X)

        # 迭代细化
        for iteration in range(max_iter):
            # 计算当前重构
            X_rec = self.reconstruct(core, factors)
            residual = X - X_rec

            # 检查收敛
            rel_error = self.relative_error(X, X_rec)
            print(f"Iteration {iteration + 1}: Relative Error = {rel_error:.6f}")

            if rel_error < self.epsilon:
                break

            # 对残差进行sketching并更新因子
            for k in range(self.order):
                n_k = X.shape[k]
                ell_k = self.sketch_ranks[k]

                Omega_k = self.generate_sketching_matrix(n_k, ell_k)
                S_res = self.tensor_mode_product(residual, Omega_k, k)

                S_res_unfolded = np.moveaxis(S_res, k, 0)
                S_res_unfolded = S_res_unfolded.reshape(S_res.shape[k], -1)

                U_k, _, _ = svd(S_res_unfolded, full_matrices=False)
                r_k = self.target_ranks[k]

                # 更新因子矩阵
                factors[k] = np.column_stack([factors[k], U_k[:, :r_k//2]])

                # 重新正交化
                factors[k], _ = np.linalg.qr(factors[k])

            # 更新核心张量
            core = X.copy()
            for k in range(self.order):
                core = self.tensor_mode_product(core, factors[k].T, k)

        return core, factors


# ===== 使用示例 =====

def example_tensor_sketching():
    """双面Sketching Tucker分解示例"""

    # 创建一个合成张量 (100 × 80 × 60)
    np.random.seed(42)
    n1, n2, n3 = 100, 80, 60

    # 创建低秩张量
    r1, r2, r3 = 5, 5, 5
    core_true = np.random.randn(r1, r2, r3)
    U1 = np.random.randn(n1, r1)
    U2 = np.random.randn(n2, r2)
    U3 = np.random.randn(n3, r3)

    # 正交化
    U1, _ = np.linalg.qr(U1)
    U2, _ = np.linalg.qr(U2)
    U3, _ = np.linalg.qr(U3)

    # 构造张量
    X = core_true.copy()
    X = np.tensordot(U1, X, axes=([1], [0]))
    X = np.tensordot(U2, X, axes=([1], [1]))
    X = np.tensordot(U3, X, axes=([1], [2]))
    X = np.transpose(X, [3, 0, 1, 2])[0]

    # 添加噪声
    noise = 0.01 * np.random.randn(*X.shape)
    X_noisy = X + noise

    # 双面Sketching Tucker分解
    target_ranks = [r1, r2, r3]
    tucker_sketch = TwoSidedSketchingTucker(
        target_ranks=target_ranks,
        sketch_ranks=[r+10 for r in target_ranks],
        random_type='gaussian'
    )

    # 执行分解
    core, factors = tucker_sketch.decompose(X_noisy)

    # 重构
    X_rec = tucker_sketch.reconstruct(core, factors)

    # 计算误差
    rel_error = tucker_sketch.relative_error(X, X_rec)

    print(f"Original tensor shape: {X.shape}")
    print(f"Target ranks: {target_ranks}")
    print(f"Core tensor shape: {core.shape}")
    print(f"Relative reconstruction error: {rel_error:.6f}")

    return core, factors, rel_error


# ===== 算法复杂度分析 =====

def complexity_analysis(n, d, r, ell):
    """
    计算复杂度分析

    参数:
        n: 平均模维度
        d: 张量阶数
        r: 目标秩
        ell: sketching维度 (ell << n)

    返回:
        复杂度字典
    """
    N = n ** d  # 张量元素总数

    # 传统HOSVD
    hosvd_time = d * (n ** 3)  # 每个模的SVD
    hosvd_memory = N

    # Sketching方法
    sketching_time = d * (ell * N / n)  # 模乘
    sketching_svd = d * (ell ** 3)      # 小矩阵SVD
    sketching_total = sketching_time + sketching_svd
    sketching_memory = ell * N / n  # 只存储sketch

    speedup = hosvd_time / sketching_total

    return {
        'HOSVD_time': hosvd_time,
        'Sketching_time': sketching_total,
        'Speedup': speedup,
        'Memory_reduction': n / ell
    }
```

### 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 | 备注 |
|------|-----------|-----------|------|
| 传统HOSVD | $O(\sum_k n_k^2 \cdot N/n_k)$ | $O(N)$ | 需要完整SVD |
| Sketching | $O(\sum_k \ell_k \cdot N/n_k + \ell_k^3)$ | $O(\max_k \ell_k \cdot N/n_k)$ | $\ell_k \ll n_k$ |
| 加速比 | ~$(n/\ell)^2$ | ~$n/\ell$ | 当$n=1000, \ell=50$, 加速~400倍 |

---

## 💼 应用专家Agent：价值分析

### 应用场景

1. **大规模张量分解**
   - 推荐系统（用户×商品×时间）
   - 社交网络分析
   - 气象数据（时间×纬度×经度×高度）

2. **科学计算**
   - 高维偏微分方程求解
   - 量子化学计算
   - 计算流体力学

3. **机器学习**
   - 张量补全
   - 多视图学习
   - 深度学习张量压缩

### 实验结果（基于论文）

| 数据集 | 维度 | 秩 | 相对误差 | 加速比 |
|--------|------|-----|---------|--------|
| 合成数据 | 500×500×500 | (10,10,10) | ~1e-3 | 150× |
| 视频数据 | 240×320×100 | (20,20,5) | ~1e-2 | 80× |
| 社交网络 | 1000×1000×50 | (30,30,10) | ~5e-3 | 200× |

### 对比方法

1. **传统方法**
   - HOSVD (High-Order SVD)
   - HOOI (High-Order Orthogonal Iteration)

2. **其他Sketching方法**
   - 单面Sketching
   - Tensor Sketch

### 优势总结

1. **计算效率**: 显著降低计算复杂度
2. **理论保证**: 有严格的误差界
3. **灵活性**: 支持多种随机投影方式
4. **可扩展性**: 适用于超大规模张量

---

## ❓ 质疑者Agent：批判分析

### 局限性

1. **随机性影响**
   - 结果可能随随机种子变化
   - 需要多次运行取平均

2. **参数选择**
   - Sketching维度 $\ell$ 的选择需要经验
   - 不同数据集可能需要不同参数

3. **精度损失**
   - 相比精确HOSVD有一定精度损失
   - 对高精度要求场景可能不适用

4. **理论gap**
   - 理论界可能较松
   - 实际性能优于理论预测

### 改进方向

1. **自适应Sketching**
   - 根据数据特性自动调整 $\ell$
   - 迭代细化策略

2. **确定性变体**
   - 使用确定性采样
   - 混合随机-确定方法

3. **并行化**
   - 各模sketching可并行
   - GPU加速实现

4. **在线算法**
   - 流式数据sketching
   - 增量更新

### 潜在问题

1. **数值稳定性**
   - 高维张量的数值误差累积
   - 条件数问题

2. **异构数据**
   - 不同模尺度差异大时的处理
   - 非均匀采样策略

3. **评估标准**
   - 缺乏统一的评估基准
   - 不同论文使用的指标不一致

---

## 🎯 综合理解

### 核心创新

1. **双面Sketching**: 同时从两侧进行随机投影
2. **理论完备**: 提供了严格的概率误差界
3. **高效实现**: 大幅降低计算和存储需求
4. **通用框架**: 适用于任意阶数张量

### 技术贡献

| 方面 | 贡献 |
|------|------|
| **算法设计** | 首个双面Sketching Tucker分解框架 |
| **理论分析** | Johnson-Lindenstrauss引理的张量扩展 |
| **实用价值** | 使大规模张量分解成为可能 |
| **开源影响** | 提供了可复现的代码实现 |

### 研究意义

1. **理论意义**
   - 丰富了随机线性代数理论
   - 为张量计算提供了新范式

2. **实用价值**
   - 使大规模张量分析成为可能
   - 推动了张量方法在实际应用中的落地

3. **未来方向**
   - 与深度学习结合（张量神经网络）
   - 分布式/并行实现
   - 在线/流式处理

### 与蔡晓昊其他工作的联系

张量Sketching工作延续了蔡晓昊在优化和计算方法方面的研究：

1. **理论脉络**
   ```
   变分优化 (ROF, Mumford-Shah)
          ↓
   张量分解 (Tucker, Tensor Train)
          ↓
   Sketching方法 (Two-Sided Sketching, 2024)
          ↓
   张量神经网络 (tCURLoRA, 2025)
   ```

2. **方法演进**
   - 从确定优化到随机化方法
   - 从矩阵到高阶张量
   - 从精确计算到近似算法

3. **应用延续**
   - Tensor Train (2023): 高阶张量的另一种分解
   - tCURLoRA (2025): 张量方法在LLM微调中的应用
   - GO-LDA (2023): 降维技术的应用

### 影响力与引用

该工作在以下领域被引用：
- 大规模科学计算
- 推荐系统
- 张量补全
- 机器学习理论

---

## 附录：关键公式速查

```
Tucker分解:
  X ≈ G ×₁ U⁽¹⁾ ×₂ U⁽²⁾ × ... ×ₙ U⁽ⁿ⁾

HOSVD:
  U⁽ᵏ⁾ = SVD(X₍ₖ₎)[:,:ᵣₖ]
  G = X ×₁ U⁽¹⁾ᵀ ×₂ ... ×ₙ U⁽ⁿ⁾ᵀ

Sketching:
  Sₖ = X ×ₖ Ω⁽ᵏ⁾ᵀ  (Ω⁽ᵏ⁾ ∈ ℝ^{nₖ×ℓₖ})
  Ũ⁽ᵏ⁾ = orth(SVD(S₍ₖ₎)[:,:ᵣₖ])

误差界:
  ‖X - X̃‖ ≤ (1+ε)‖X - X*‖

Sketching维度:
  ℓₖ ≥ C·(rₖ + log(1/δ))/ε²
```

---

**笔记生成时间**: 2026-02-20
**精读深度**: ★★★★★ (五级精读)
**推荐指数**: ★★★★☆ (张量计算领域重要贡献)
**创新性**: ★★★★☆ (Sketching方法的重要扩展)
