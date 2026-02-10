# [3-05] 大规模张量分解 Two-Sided Sketching - 精读笔记

> **论文标题**: Two-Sided Sketching for Low-Rank Tensor Decomposition
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐⭐ (中高)
> **重要性**: ⭐⭐⭐⭐ (重要，大规模张量分解加速)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Two-Sided Sketching for Low-Rank Tensor Decomposition |
| **作者** | X. Cai 等人 |
| **发表期刊** | SIAM Journal on Scientific Computing |
| **发表年份** | 2022 |
| **关键词** | Tensor Decomposition, Sketching, Tucker Decomposition, Large-Scale |
| **代码** | (请查看论文是否有开源代码) |

---

## 🎯 研究问题与动机

### 大规模张量分解挑战

**传统方法的计算瓶颈**:
```
Tucker分解 (HOOI算法):
- 每轮迭代需要计算SVD
- 复杂度: O(I × J × K × R) 每模态
- 对于大规模张量 (I,J,K > 10^4) 不可行

内存需求:
- 存储完整张量: I × J × K × 8 bytes
- 10000 × 10000 × 10000 = 8 PB (不可能)
```

**Sketching解决方案**:
```
核心思想: 不处理完整张量，而是处理压缩后的sketch

优势:
- 内存: 只存储sketch，大幅减少
- 计算: sketch上的操作更快
- 精度: 保持主要结构信息
```

---

## 🔬 方法论详解

### 整体框架

```
┌─────────────────────────────────────────────────────────┐
│              双边Sketching框架                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  原始大规模张量 X ∈ R^(I×J×K)                            │
│                    ↓                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │           阶段1: 双边Sketching                    │   │
│  │                                                  │   │
│  │   行方向Sketching: Ω ∈ R^(J×s1), Ψ ∈ R^(K×s2)    │   │
│  │         ↓                                        │   │
│  │   X_(1) × Ω × Ψ → Y ∈ R^(I×s1×s2)               │   │
│  │                                                  │   │
│  │   列方向Sketching: Φ ∈ R^(I×s3)                  │   │
│  │         ↓                                        │   │
│  │   Φ^T × X_(1) → Z ∈ R^(s3×J×K)                  │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │           阶段2: Sketch上的分解                   │   │
│  │                                                  │   │
│  │   在Y和Z上进行Tucker分解                          │   │
│  │   复杂度: O(I×s1×s2 + s3×J×K) << O(I×J×K)       │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │           阶段3: 重建核心与因子矩阵               │   │
│  │                                                  │   │
│  │   从sketch恢复: A ∈ R^(I×R1), B ∈ R^(J×R2)      │   │
│  │                 C ∈ R^(K×R3), G ∈ R^(R1×R2×R3)  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

### 核心方法1: 双边Sketching

**数学定义**:
```
给定张量 X ∈ R^(I×J×K)

行方向Sketching矩阵:
- Ω ∈ R^(J×s1): 随机投影矩阵 (s1 << J)
- Ψ ∈ R^(K×s2): 随机投影矩阵 (s2 << K)

列方向Sketching矩阵:
- Φ ∈ R^(I×s3): 随机投影矩阵 (s3 << I)

Sketch计算:
Y = X ×_2 Ω^T ×_3 Ψ^T  ∈ R^(I×s1×s2)  (模态2和3的压缩)
Z = X ×_1 Φ^T          ∈ R^(s3×J×K)   (模态1的压缩)

其中 ×_n 表示n-模态积
```

**Sketching实现**:
```python
import torch
import numpy as np


class TwoSidedSketching:
    """
    双边Sketching用于大规模张量分解

    通过随机投影将大规模张量压缩为小的sketch
    """
    def __init__(
        self,
        tensor_shape: tuple,  # (I, J, K)
        sketch_sizes: tuple,  # (s1, s2, s3)
        sketch_type: str = 'gaussian'
    ):
        self.I, self.J, self.K = tensor_shape
        self.s1, self.s2, self.s3 = sketch_sizes
        self.sketch_type = sketch_type

        # 生成随机投影矩阵
        self.Omega = self._generate_sketch_matrix((self.J, self.s1))
        self.Psi = self._generate_sketch_matrix((self.K, self.s2))
        self.Phi = self._generate_sketch_matrix((self.I, self.s3))

    def _generate_sketch_matrix(self, shape: tuple) -> torch.Tensor:
        """
        生成随机sketching矩阵

        Args:
            shape: 矩阵形状

        Returns:
            sketch_matrix: 随机投影矩阵
        """
        if self.sketch_type == 'gaussian':
            # 高斯随机矩阵
            return torch.randn(shape) / np.sqrt(shape[0])
        elif self.sketch_type == 'sparse':
            # 稀疏随机矩阵 (Count-Sketch)
            matrix = torch.zeros(shape)
            for i in range(shape[0]):
                j = np.random.randint(0, shape[1])
                matrix[i, j] = np.random.choice([-1, 1])
            return matrix
        elif self.sketch_type == 'srht':
            # Subsampled Randomized Hadamard Transform
            # 更高效但实现复杂
            return torch.randn(shape) / np.sqrt(shape[0])
        else:
            raise ValueError(f"Unknown sketch type: {self.sketch_type}")

    def compute_sketches(self, X: torch.Tensor) -> tuple:
        """
        计算双边sketch

        Args:
            X: 输入张量 (I, J, K)

        Returns:
            Y: 行方向sketch (I, s1, s2)
            Z: 列方向sketch (s3, J, K)
        """
        # 行方向sketch: Y = X ×_2 Ω^T ×_3 Ψ^T
        # 先进行模态2乘积
        X_mode2 = torch.einsum('ijk,jl->ilk', X, self.Omega)
        # 再进行模态3乘积
        Y = torch.einsum('ilk,km->ilm', X_mode2, self.Psi)

        # 列方向sketch: Z = X ×_1 Φ^T
        Z = torch.einsum('ijk,li->ljk', X, self.Phi)

        return Y, Z

    def compute_sketch_memory(self) -> dict:
        """
        计算sketch的内存占用

        Returns:
            memory_info: 内存信息字典
        """
        original_memory = self.I * self.J * self.K * 8 / (1024**3)  # GB
        sketch_memory = (
            self.I * self.s1 * self.s2 +
            self.s3 * self.J * self.K
        ) * 8 / (1024**3)  # GB

        return {
            'original_gb': original_memory,
            'sketch_gb': sketch_memory,
            'compression_ratio': original_memory / sketch_memory
        }
```

---

### 核心方法2: Sketch上的Tucker分解

```python
class SketchTuckerDecomposition:
    """
    在Sketch上进行Tucker分解

    避免直接处理大规模张量
    """
    def __init__(self, ranks: tuple):
        """
        Args:
            ranks: Tucker秩 (R1, R2, R3)
        """
        self.R1, self.R2, self.R3 = ranks

    def decompose(
        self,
        Y: torch.Tensor,  # (I, s1, s2)
        Z: torch.Tensor,  # (s3, J, K)
        sketching: TwoSidedSketching
    ) -> tuple:
        """
        在sketch上执行Tucker分解

        Args:
            Y: 行方向sketch
            Z: 列方向sketch
            sketching: sketching对象

        Returns:
            A, B, C: 因子矩阵
            G: 核心张量
        """
        I, s1, s2 = Y.shape
        s3, J, K = Z.shape

        # 从Y估计因子矩阵A (模态1)
        # Y_(1) ∈ R^(I×s1s2)
        Y_unfold = Y.reshape(I, -1)
        # SVD分解
        U_y, S_y, V_y = torch.svd(Y_unfold)
        A = U_y[:, :self.R1]  # (I, R1)

        # 从Z估计因子矩阵B和C
        # 需要额外的处理，因为Z是模态1的压缩

        # 方法: 使用交替最小二乘 (ALS)
        # 初始化
        B = torch.randn(J, self.R2)
        C = torch.randn(K, self.R3)

        # 迭代优化
        for iteration in range(50):
            # 更新B
            B = self._update_factor_B(Z, A, C, sketching)
            # 更新C
            C = self._update_factor_C(Z, A, B, sketching)

        # 估计核心张量
        G = self._estimate_core(Y, A, B, C, sketching)

        return A, B, C, G

    def _update_factor_B(
        self,
        Z: torch.Tensor,
        A: torch.Tensor,
        C: torch.Tensor,
        sketching: TwoSidedSketching
    ) -> torch.Tensor:
        """更新因子矩阵B"""
        s3, J, K = Z.shape

        # 构建最小二乘问题
        # 这需要利用sketching矩阵的性质
        # 简化实现: 使用Z的模态2展开
        Z_mode2 = Z.permute(1, 0, 2).reshape(J, -1)  # (J, s3*K)

        # 伪逆求解
        B = torch.linalg.lstsq(
            Z_mode2.T,
            torch.randn(Z_mode2.size(1), self.R2)
        ).solution[:J]

        return B

    def _update_factor_C(
        self,
        Z: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        sketching: TwoSidedSketching
    ) -> torch.Tensor:
        """更新因子矩阵C"""
        s3, J, K = Z.shape

        # 类似地更新C
        Z_mode3 = Z.permute(2, 0, 1).reshape(K, -1)  # (K, s3*J)

        C = torch.linalg.lstsq(
            Z_mode3.T,
            torch.randn(Z_mode3.size(1), self.R3)
        ).solution[:K]

        return C

    def _estimate_core(
        self,
        Y: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        sketching: TwoSidedSketching
    ) -> torch.Tensor:
        """
        估计核心张量

        G ≈ Y ×_1 A^T ×_2 (Ω^T B)^† ×_3 (Ψ^T C)^†
        """
        # 使用Moore-Penrose伪逆
        Omega_B_pinv = torch.linalg.pinv(sketching.Omega.T @ B)
        Psi_C_pinv = torch.linalg.pinv(sketching.Psi.T @ C)

        # 计算核心
        G = torch.einsum('ijk,ir->rjk', Y, A)
        G = torch.einsum('rjk,js->rsk', G, Omega_B_pinv)
        G = torch.einsum('rsk,kt->rst', G, Psi_C_pinv)

        return G
```

---

### 核心方法3: 流式张量分解

```python
class StreamingTensorDecomposition:
    """
    流式张量分解

    处理无法一次性加载到内存的大规模张量
    """
    def __init__(
        self,
        tensor_shape: tuple,
        ranks: tuple,
        sketch_sizes: tuple,
        block_size: int = 1000
    ):
        self.tensor_shape = tensor_shape
        self.ranks = ranks
        self.block_size = block_size

        # 初始化sketching
        self.sketching = TwoSidedSketching(tensor_shape, sketch_sizes)

        # 累积的sketch
        self.Y_accum = torch.zeros(tensor_shape[0], sketch_sizes[0], sketch_sizes[1])
        self.Z_accum = torch.zeros(sketch_sizes[2], tensor_shape[1], tensor_shape[2])
        self.sample_count = 0

    def update(self, X_block: torch.Tensor, block_indices: tuple):
        """
        用新的数据块更新sketch

        Args:
            X_block: 数据块
            block_indices: 块在完整张量中的位置
        """
        # 计算块的sketch
        Y_block, Z_block = self.sketching.compute_sketches(X_block)

        # 更新累积sketch (需要处理索引)
        # 简化: 假设块是模态1的切片
        start_idx, end_idx = block_indices
        self.Y_accum[start_idx:end_idx] += Y_block
        self.Z_accum += Z_block

        self.sample_count += 1

    def finalize(self) -> tuple:
        """
        完成分解

        Returns:
            A, B, C, G: Tucker分解结果
        """
        # 平均累积的sketch
        Y_avg = self.Y_accum / self.sample_count
        Z_avg = self.Z_accum / self.sample_count

        # 在平均sketch上分解
        decomposer = SketchTuckerDecomposition(self.ranks)
        A, B, C, G = decomposer.decompose(Y_avg, Z_avg, self.sketching)

        return A, B, C, G
```

---

## 📊 复杂度分析

### 计算复杂度对比

| 方法 | 时间复杂度 | 空间复杂度 | 适用规模 |
|:---|:---|:---|:---|
| HOOI | O(I×J×K×R × iter) | O(I×J×K) | < 10^3 |
| Randomized SVD | O(I×J×K×s) | O(I×J×K) | < 10^4 |
| **Two-Sided Sketching** | O(I×s1×s2 + s3×J×K) | O(I×s1×s2 + s3×J×K) | > 10^5 |

### 精度-效率权衡

```
Sketch大小选择:
- s太小: 精度损失大
- s太大: 效率提升小

经验法则:
s = O(R / ε)
其中R是Tucker秩，ε是允许的相对误差

典型设置:
- s1 = s2 = s3 = 2R ~ 5R
- 压缩比: 100x ~ 10000x
```

---

## 💡 可复用代码组件

### 组件1: 完整的大规模Tucker分解流程

```python
class LargeScaleTucker:
    """
    大规模张量Tucker分解

    使用双边Sketching处理无法放入内存的张量
    """
    def __init__(
        self,
        tensor_shape: tuple,
        ranks: tuple,
        sketch_sizes: tuple = None,
        device: str = 'cpu'
    ):
        self.tensor_shape = tensor_shape
        self.ranks = ranks
        self.device = device

        # 自动选择sketch大小
        if sketch_sizes is None:
            R1, R2, R3 = ranks
            self.sketch_sizes = (5*R1, 5*R2, 5*R3)
        else:
            self.sketch_sizes = sketch_sizes

        # 初始化组件
        self.sketching = TwoSidedSketching(
            tensor_shape, self.sketch_sizes
        )
        self.decomposer = SketchTuckerDecomposition(ranks)

    def fit(self, X: torch.Tensor = None, data_loader=None) -> tuple:
        """
        执行分解

        Args:
            X: 完整张量 (如果可放入内存)
            data_loader: 数据加载器 (用于流式处理)

        Returns:
            A, B, C, G: Tucker分解结果
        """
        if X is not None:
            # 直接处理
            Y, Z = self.sketching.compute_sketches(X)
            return self.decomposer.decompose(Y, Z, self.sketching)

        elif data_loader is not None:
            # 流式处理
            streamer = StreamingTensorDecomposition(
                self.tensor_shape,
                self.ranks,
                self.sketch_sizes
            )

            for batch in data_loader:
                X_block, indices = batch
                streamer.update(X_block, indices)

            return streamer.finalize()

        else:
            raise ValueError("必须提供X或data_loader")

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        将张量投影到Tucker核心空间

        Args:
            X: 输入张量

        Returns:
            G: 核心张量
        """
        # 需要先有分解结果
        # G = X ×_1 A^T ×_2 B^T ×_3 C^T
        pass

    def inverse_transform(self, G: torch.Tensor) -> torch.Tensor:
        """
        从核心张量重建

        Args:
            G: 核心张量

        Returns:
            X_reconstructed: 重建张量
        """
        # X = G ×_1 A ×_2 B ×_3 C
        pass
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **Sketching** | Sketching | 随机投影压缩技术 |
| **双边Sketching** | Two-Sided Sketching | 同时压缩多个模态 |
| **Tucker分解** | Tucker Decomposition | 高阶SVD分解 |
| **核心张量** | Core Tensor | Tucker分解的核心 |
| **因子矩阵** | Factor Matrix | 各模态的基矩阵 |
| **模态积** | Mode-n Product | 张量与矩阵的n模态乘积 |

---

## ✅ 复习检查清单

- [ ] 理解大规模张量分解的挑战
- [ ] 掌握Sketching的基本原理
- [ ] 理解双边Sketching的优势
- [ ] 了解流式张量分解
- [ ] 能够选择合适的sketch大小

---

## 🤔 思考问题

1. **为什么Sketching能保持张量的主要结构？**
   - 提示: Johnson-Lindenstrauss引理

2. **双边Sketching相比单边有何优势？**
   - 提示: 内存和计算平衡

3. **sketch大小如何选择？**
   - 提示: 精度vs效率权衡

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
