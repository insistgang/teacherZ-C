# Tucker张量分解教程

## 目录
1. [理论讲解](#1-理论讲解)
2. [算法详解](#2-算法详解)
3. [代码实现](#3-代码实现)
4. [实验指南](#4-实验指南)
5. [习题与答案](#5-习题与答案)

---

## 1. 理论讲解

### 1.1 张量基础

**张量的定义**

张量是多维数组的推广：
- 0阶张量：标量
- 1阶张量：向量
- 2阶张量：矩阵
- N阶张量：N维数组

**记号**

- 张量：$\mathcal{X} \in \mathbb{R}^{I_1 \times I_2 \times \cdots \times I_N}$
- 矩阵：$\mathbf{U} \in \mathbb{R}^{I \times J}$
- 向量：$\mathbf{v} \in \mathbb{R}^{I}$
- 元素：$x_{i_1 i_2 \ldots i_N}$

**模-n展开（Matricization）**

将N阶张量 $\mathcal{X}$ 沿第n模展开为矩阵 $\mathbf{X}_{(n)}$：

$$\mathbf{X}_{(n)} \in \mathbb{R}^{I_n \times \prod_{k \neq n} I_k}$$

**模-n乘积**

张量 $\mathcal{X}$ 与矩阵 $\mathbf{U}$ 的模-n乘积：

$$\mathcal{Y} = \mathcal{X} \times_n \mathbf{U}$$

其中：
$$y_{i_1 \ldots i_{n-1} j i_{n+1} \ldots i_N} = \sum_{i_n} x_{i_1 \ldots i_N} u_{j i_n}$$

### 1.2 Tucker分解模型

**分解形式**

Tucker分解将张量分解为核心张量与各模态因子矩阵的乘积：

$$\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times \cdots \times_N \mathbf{U}^{(N)}$$

其中：
- $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times \cdots \times R_N}$：核心张量
- $\mathbf{U}^{(n)} \in \mathbb{R}^{I_n \times R_n}$：第n模态因子矩阵
- $R_n$：第n模态的Tucker秩

**展开形式**

$$\mathbf{X}_{(n)} \approx \mathbf{U}^{(n)} \mathbf{G}_{(n)} (\mathbf{U}^{(N)} \otimes \cdots \otimes \mathbf{U}^{(n+1)} \otimes \mathbf{U}^{(n-1)} \otimes \cdots \otimes \mathbf{U}^{(1)})^\top$$

**元素形式**

$$x_{i_1 i_2 \ldots i_N} \approx \sum_{r_1, r_2, \ldots, r_N} g_{r_1 r_2 \ldots r_N} u_{i_1 r_1}^{(1)} u_{i_2 r_2}^{(2)} \cdots u_{i_N r_N}^{(N)}$$

### 1.3 与其他分解的关系

**CP分解（CANDECOMP/PARAFAC）**

CP是Tucker的特殊情况，当：
- 核心张量为超对角张量
- 所有模态秩相等：$R_1 = R_2 = \cdots = R_N = R$

$$\mathcal{X} \approx \sum_{r=1}^{R} \lambda_r \mathbf{a}_r^{(1)} \circ \mathbf{a}_r^{(2)} \circ \cdots \circ \mathbf{a}_r^{(N)}$$

**矩阵SVD**

对于矩阵（2阶张量），Tucker分解退化为SVD：
$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

其中 $\mathbf{U}$ 和 $\mathbf{V}$ 对应因子矩阵，$\mathbf{\Sigma}$ 对应核心张量。

**高阶SVD（HOSVD）**

当因子矩阵正交时，Tucker分解称为HOSVD：
$$\mathcal{X} = \mathcal{S} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times \cdots \times_N \mathbf{U}^{(N)}$$

其中 $\mathbf{U}^{(n)\top} \mathbf{U}^{(n)} = \mathbf{I}$。

### 1.4 存在性与唯一性

**存在性**

Tucker分解总是存在，因为任何张量都可以精确表示（当秩足够大时）。

**唯一性**

Tucker分解一般不唯一，因为存在变换模糊性：
$$\mathcal{X} = \mathcal{G} \times_1 \mathbf{U}^{(1)} \times \cdots \times_N \mathbf{U}^{(N)} = (\mathcal{G} \times_1 \mathbf{Q}^{(1)}) \times_1 (\mathbf{U}^{(1)} \mathbf{Q}^{(1)-1}) \times \cdots$$

其中 $\mathbf{Q}^{(n)}$ 为可逆矩阵。

### 1.5 最优性条件

**最小二乘问题**

$$\min_{\mathcal{G}, \{\mathbf{U}^{(n)}\}} \|\mathcal{X} - \mathcal{G} \times_1 \mathbf{U}^{(1)} \times \cdots \times_N \mathbf{U}^{(N)}\|_F^2$$

**交替最小二乘（ALS）的最优性**

固定其他变量，对 $\mathbf{U}^{(n)}$ 的最优解：
$$\mathbf{U}^{(n)} = \mathbf{X}_{(n)} \mathbf{Z}_{(n)} (\mathbf{Z}_{(n)}^\top \mathbf{Z}_{(n)})^{-1}$$

其中 $\mathbf{Z}_{(n)} = \mathcal{G} \times_{k \neq n} \mathbf{U}^{(k)}$ 的模-n展开。

---

## 2. 算法详解

### 2.1 高阶奇异值分解（HOSVD）

**算法步骤**

```
输入: 张量 X ∈ ℝ^(I₁×I₂×...×I_N), 目标秩 (R₁, R₂, ..., R_N)
输出: 核心张量 G, 因子矩阵 {U⁽ⁿ⁾}

For n = 1 to N:
    1. 计算模-n展开: X₍ₙ₎
    2. 计算SVD: X₍ₙ₎ = U Σ Vᵀ
    3. 取前Rₙ个左奇异向量: U⁽ⁿ⁾ = U[:, 1:Rₙ]

计算核心张量:
    G = X ×₁ U⁽¹⁾ᵀ ×₂ U⁽²⁾ᵀ ... ×_N U⁽ᴺ⁾ᵀ
```

**复杂度分析**

- 模-n SVD: $O(I_n^2 \prod_{k \neq n} I_k)$
- 总复杂度: $O(N \cdot I^N)$（假设 $I_n \approx I$）

**性质**

1. 提供秩的近似
2. 因子矩阵正交
3. 可作为其他算法的初始化

### 2.2 高阶正交迭代（HOOI）

**算法思想**

交替优化各模态因子矩阵，保持正交性约束。

**算法步骤**

```
输入: 张量 X, 目标秩 (R₁, ..., R_N)
输出: 最优核心张量和因子矩阵

初始化: 使用HOSVD初始化 {U⁽ⁿ⁾}

Repeat until 收敛:
    For n = 1 to N:
        1. 计算中间张量: Y = X ×₁ U⁽¹⁾ᵀ ... ×ₙ₋₁ U⁽ⁿ⁻¹⁾ᵀ ×ₙ₊₁ U⁽ⁿ⁺¹⁾ᵀ ... ×_N U⁽ᴺ⁾ᵀ
        2. 模-n展开: Y₍ₙ₎
        3. SVD: Y₍ₙ₎ = U Σ Vᵀ
        4. 更新: U⁽ⁿ⁾ = U[:, 1:Rₙ]

计算核心: G = X ×₁ U⁽¹⁾ᵀ ×₂ U⁽²⁾ᵀ ... ×_N U⁽ᴺ⁾ᵀ
```

**收敛条件**

相对重构误差变化：
$$\frac{\|\mathcal{X}^{(k)} - \mathcal{X}^{(k-1)}\|_F}{\|\mathcal{X}\|_F} < \epsilon$$

### 2.3 交替最小二乘（ALS）

**非正交约束版本**

```
输入: 张量 X, 目标秩 (R₁, ..., R_N)

初始化: 随机或HOSVD初始化 {U⁽ⁿ⁾}, G

Repeat:
    For n = 1 to N:
        # 固定其他变量，更新U⁽ⁿ⁾
        计算: Aₙ = G ×ₖ₌₁,ₖ≠ₙ U⁽ᵏ⁾
        求解: min ||X₍ₙ₎ - U⁽ⁿ⁾ A₍ₙ₎||²_F
        更新: U⁽ⁿ⁾ = X₍ₙ₎ A₍ₙ₎ᵀ (A₍ₙ₎ A₍ₙ₎ᵀ)⁻¹
    
    # 更新核心张量
    G = X ×₁ U⁽¹⁾⁺ ×₂ U⁽²⁾⁺ ... ×_N U⁽ᴺ⁾⁺  (+表示伪逆)

Until 收敛
```

### 2.4 随机化算法

**随机投影方法**

```
输入: 张量 X, 目标秩 (R₁, ..., R_N)

For n = 1 to N:
    1. 生成随机矩阵: Ω⁽ⁿ⁾ ∈ ℝ^(∏ₖ≠ₙIₖ × Rₙ)
    2. 计算: Y⁽ⁿ⁾ = X₍ₙ₎ Ω⁽ⁿ⁾
    3. QR分解: Y⁽ⁿ⁾ = Q⁽ⁿ⁾ R⁽ⁿ⁾
    4. 投影: B⁽ⁿ⁾ = Q⁽ⁿ⁾ᵀ X₍ₙ₎
    5. SVD: B⁽ⁿ⁾ = U Σ Vᵀ
    6. 因子矩阵: U⁽ⁿ⁾ = Q⁽ⁿ⁾ U

计算核心: G = X ×₁ U⁽¹⁾ᵀ ... ×_N U⁽ᴺ⁾ᵀ
```

**优势**

- 复杂度从 $O(I^N)$ 降到 $O(R \cdot I^N)$
- 适合大规模张量

### 2.5 秩确定方法

**基于奇异值的方法**

对每个模态计算模-n展开的奇异值，使用肘部法则或阈值：

$$R_n = \max\{r : \sigma_r / \sigma_1 > \epsilon\}$$

**核一致诊断（CORCONDIA）**

计算核心一致性指标，选择使指标最大的秩。

**信息准则（AIC/BIC）**

$$\text{BIC} = N \cdot \log(\text{MSE}) + p \cdot \log(N)$$

其中 $p = \sum_n R_n (I_n + R_n)$ 为参数数。

---

## 3. 代码实现

### 3.1 张量操作工具

```python
import numpy as np
from typing import Tuple, List, Optional
from scipy.linalg import svd, qr


def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    """
    张量模-n展开（Matricization）
    
    参数:
        tensor: 输入张量
        mode: 模态索引（0-based）
    
    返回:
        展开后的矩阵
    """
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def fold(matrix: np.ndarray, mode: int, shape: Tuple[int, ...]) -> np.ndarray:
    """
    矩阵折叠回张量
    
    参数:
        matrix: 输入矩阵
        mode: 模态索引
        shape: 目标张量形状
    
    返回:
        张量
    """
    new_shape = [shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
    tensor = matrix.reshape(new_shape)
    return np.moveaxis(tensor, 0, mode)


def mode_n_product(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """
    模-n乘积：张量与矩阵的乘积
    
    参数:
        tensor: 输入张量
        matrix: 矩阵 (R, I_n)
        mode: 模态索引
    
    返回:
        结果张量
    """
    # 展开张量
    unfolded = unfold(tensor, mode)
    # 矩阵乘法
    result = matrix @ unfolded
    # 折叠回来
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    return fold(result, mode, tuple(new_shape))


def kronecker(matrices: List[np.ndarray]) -> np.ndarray:
    """
    多个矩阵的Kronecker积
    
    参数:
        matrices: 矩阵列表
    
    返回:
        Kronecker积结果
    """
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result


def khatri_rao(matrices: List[np.ndarray]) -> np.ndarray:
    """
    多个矩阵的Khatri-Rao积（列向量的Kronecker积）
    
    参数:
        matrices: 矩阵列表，每个矩阵有相同的列数
    
    返回:
        Khatri-Rao积结果
    """
    n_cols = matrices[0].shape[1]
    result = np.ones((1, n_cols))
    
    for mat in matrices:
        temp = np.zeros((result.shape[0] * mat.shape[0], n_cols))
        for c in range(n_cols):
            temp[:, c] = np.kron(result[:, c], mat[:, c])
        result = temp
    
    return result


def norm(tensor: np.ndarray) -> float:
    """计算张量的Frobenius范数"""
    return np.linalg.norm(tensor)


def tucker_to_tensor(core: np.ndarray, factors: List[np.ndarray]) -> np.ndarray:
    """
    从Tucker分解重建张量
    
    参数:
        core: 核心张量
        factors: 因子矩阵列表
    
    返回:
        重建的张量
    """
    result = core.copy()
    for i, factor in enumerate(factors):
        result = mode_n_product(result, factor, i)
    return result
```

### 3.2 HOSVD实现

```python
def hosvd(tensor: np.ndarray, 
          ranks: Optional[List[int]] = None,
          tol: float = 1e-7) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    高阶奇异值分解（HOSVD）
    
    参数:
        tensor: 输入张量
        ranks: 目标秩列表，None则自动确定
        tol: 奇异值截断阈值
    
    返回:
        core: 核心张量
        factors: 因子矩阵列表
    """
    n_modes = tensor.ndim
    factors = []
    
    for mode in range(n_modes):
        # 模-n展开
        unfolded = unfold(tensor, mode)
        
        # SVD
        U, s, Vh = svd(unfolded, full_matrices=False)
        
        # 确定秩
        if ranks is not None:
            r = ranks[mode]
        else:
            # 自动确定：保留能量99%
            cumulative = np.cumsum(s**2) / np.sum(s**2)
            r = np.searchsorted(cumulative, 1 - tol) + 1
            r = min(r, len(s))
        
        factors.append(U[:, :r])
    
    # 计算核心张量
    core = tensor.copy()
    for mode, factor in enumerate(factors):
        core = mode_n_product(core, factor.T, mode)
    
    return core, factors


def hosvd_compress(tensor: np.ndarray, 
                   compression_ratio: float = 0.1) -> Tuple[np.ndarray, List[np.ndarray], Tuple]:
    """
    基于压缩比的HOSVD
    
    参数:
        tensor: 输入张量
        compression_ratio: 压缩比（存储/原始）
    
    返回:
        core: 核心张量
        factors: 因子矩阵
        original_shape: 原始形状
    """
    original_shape = tensor.shape
    original_size = np.prod(original_shape)
    
    # 估计目标秩
    n_modes = tensor.ndim
    total_factor_size = 0
    ranks = []
    
    for mode in range(n_modes):
        unfolded = unfold(tensor, mode)
        U, s, _ = svd(unfolded, full_matrices=False)
        
        # 二分搜索找到合适的秩
        low, high = 1, len(s)
        while low < high:
            mid = (low + high + 1) // 2
            rank_sizes = ranks + [mid]
            factor_size = sum(original_shape[i] * rank_sizes[i] for i in range(len(rank_sizes)))
            factor_size += sum(rank_sizes)  # 核心张量部分估计
            
            if factor_size < compression_ratio * original_size:
                low = mid
            else:
                high = mid - 1
        
        ranks.append(low)
    
    # 补全剩余模态的秩
    while len(ranks) < n_modes:
        ranks.append(1)
    
    return hosvd(tensor, ranks), original_shape
```

### 3.3 HOOI实现

```python
def hooi(tensor: np.ndarray,
         ranks: List[int],
         n_iter: int = 100,
         tol: float = 1e-6,
         init: str = 'hosvd',
         verbose: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    高阶正交迭代（HOOI）
    
    参数:
        tensor: 输入张量
        ranks: 目标秩列表
        n_iter: 最大迭代次数
        tol: 收敛阈值
        init: 初始化方法
        verbose: 打印信息
    
    返回:
        core: 核心张量
        factors: 因子矩阵列表
    """
    n_modes = tensor.ndim
    
    # 初始化
    if init == 'hosvd':
        _, factors = hosvd(tensor, ranks)
    else:
        factors = [np.random.randn(tensor.shape[i], ranks[i]) for i in range(n_modes)]
    
    # 初始重构误差
    core = _compute_core(tensor, factors)
    prev_error = norm(tensor - tucker_to_tensor(core, factors))
    
    for iteration in range(n_iter):
        for mode in range(n_modes):
            # 计算中间张量 Y = X ×_k U_k^T (k != mode)
            Y = tensor.copy()
            for k in range(n_modes):
                if k != mode:
                    Y = mode_n_product(Y, factors[k].T, k)
            
            # 模-n展开并计算SVD
            Y_n = unfold(Y, mode)
            U, _, _ = svd(Y_n, full_matrices=False)
            
            # 更新因子矩阵
            factors[mode] = U[:, :ranks[mode]]
        
        # 更新核心张量
        core = _compute_core(tensor, factors)
        
        # 检查收敛
        error = norm(tensor - tucker_to_tensor(core, factors))
        improvement = (prev_error - error) / prev_error
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1}: Error = {error:.6f}, Improvement = {improvement:.6f}")
        
        if improvement < tol:
            if verbose:
                print(f"Converged at iteration {iteration+1}")
            break
        
        prev_error = error
    
    return core, factors


def _compute_core(tensor: np.ndarray, factors: List[np.ndarray]) -> np.ndarray:
    """计算核心张量"""
    core = tensor.copy()
    for mode, factor in enumerate(factors):
        core = mode_n_product(core, factor.T, mode)
    return core
```

### 3.4 ALS实现

```python
def tucker_als(tensor: np.ndarray,
               ranks: List[int],
               n_iter: int = 100,
               tol: float = 1e-6,
               reg: float = 1e-8,
               verbose: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    交替最小二乘法求解Tucker分解
    
    参数:
        tensor: 输入张量
        ranks: 目标秩
        n_iter: 最大迭代次数
        tol: 收敛阈值
        reg: 正则化参数
        verbose: 打印信息
    
    返回:
        core: 核心张量
        factors: 因子矩阵
    """
    n_modes = tensor.ndim
    
    # 初始化
    core, factors = hosvd(tensor, ranks)
    
    prev_error = norm(tensor - tucker_to_tensor(core, factors))
    
    for iteration in range(n_iter):
        # 更新因子矩阵
        for mode in range(n_modes):
            # 计算其他因子矩阵的Kronecker积
            other_factors = [factors[k] for k in range(n_modes) if k != mode]
            K = kronecker(other_factors[::-1])  # 逆序
            
            # 展开张量
            X_n = unfold(tensor, mode)
            
            # 计算核心张量的模-n展开
            G_n = unfold(core, mode)
            
            # 求解线性系统
            A = G_n @ K.T @ K @ G_n.T + reg * np.eye(ranks[mode])
            b = X_n @ K.T @ G_n.T
            factors[mode] = np.linalg.solve(A, b.T).T
        
        # 更新核心张量
        core = _compute_core_als(tensor, factors, ranks, reg)
        
        # 检查收敛
        error = norm(tensor - tucker_to_tensor(core, factors))
        improvement = (prev_error - error) / (prev_error + 1e-10)
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"Iter {iteration+1}: Error = {error:.6f}")
        
        if improvement < tol and iteration > 0:
            break
        
        prev_error = error
    
    return core, factors


def _compute_core_als(tensor: np.ndarray, 
                      factors: List[np.ndarray],
                      ranks: List[int],
                      reg: float) -> np.ndarray:
    """使用伪逆计算核心张量"""
    n_modes = tensor.ndim
    
    # 计算 A = U^(1) ⊗ U^(2) ⊗ ... ⊗ U^(N)
    K = kronecker([f.T for f in factors[::-1]])
    
    # 向量化张量
    x = tensor.flatten()
    
    # 求解 min ||K^T @ g - x||^2
    # g = (K @ K^T)^{-1} @ K @ x
    KKT = K.T @ K + reg * np.eye(K.shape[1])
    core_vec = np.linalg.solve(KKT, K.T @ x)
    
    return core_vec.reshape(ranks)
```

### 3.5 随机化Tucker

```python
def randomized_tucker(tensor: np.ndarray,
                      ranks: List[int],
                      n_oversamples: int = 10,
                      n_iter: int = 2) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    随机化Tucker分解
    
    参数:
        tensor: 输入张量
        ranks: 目标秩
        n_oversamples: 过采样数
        n_iter: 幂迭代次数
    
    返回:
        core: 核心张量
        factors: 因子矩阵
    """
    n_modes = tensor.ndim
    factors = []
    
    for mode in range(n_modes):
        # 模-n展开
        X_n = unfold(tensor, mode)
        m, n = X_n.shape
        r = ranks[mode]
        
        # 随机投影
        Omega = np.random.randn(n, r + n_oversamples)
        Y = X_n @ Omega
        
        # 幂迭代增强精度
        for _ in range(n_iter):
            Q, _ = qr(Y, mode='economic')
            Y = X_n @ (X_n.T @ Q)
        
        # QR分解
        Q, _ = qr(Y, mode='economic')
        
        # 投影并做SVD
        B = Q.T @ X_n
        U, s, Vh = svd(B, full_matrices=False)
        
        # 因子矩阵
        factors.append((Q @ U)[:, :r])
    
    # 计算核心张量
    core = _compute_core(tensor, factors)
    
    return core, factors
```

### 3.6 应用函数

```python
def tucker_denoise(tensor: np.ndarray,
                   ranks: List[int],
                   method: str = 'hooi') -> np.ndarray:
    """
    使用Tucker分解去噪
    
    参数:
        tensor: 噪声张量
        ranks: 保留秩
        method: 分解方法
    
    返回:
        去噪后的张量
    """
    if method == 'hooi':
        core, factors = hooi(tensor, ranks)
    elif method == 'hosvd':
        core, factors = hosvd(tensor, ranks)
    else:
        core, factors = tucker_als(tensor, ranks)
    
    return tucker_to_tensor(core, factors)


def tucker_completion(tensor: np.ndarray,
                      mask: np.ndarray,
                      ranks: List[int],
                      n_iter: int = 100) -> np.ndarray:
    """
    使用Tucker分解进行张量补全
    
    参数:
        tensor: 有缺失值的张量（缺失位置用0填充）
        mask: 观测掩码（1表示观测，0表示缺失）
        ranks: 目标秩
        n_iter: 迭代次数
    
    返回:
        补全后的张量
    """
    # 初始化
    filled = tensor.copy()
    filled[mask == 0] = np.random.rand(np.sum(mask == 0))
    
    for _ in range(n_iter):
        # Tucker分解
        core, factors = hosvd(filled, ranks)
        
        # 重构
        reconstructed = tucker_to_tensor(core, factors)
        
        # 只更新缺失位置
        filled[mask == 0] = reconstructed[mask == 0]
    
    return filled


def estimate_tucker_rank(tensor: np.ndarray, 
                         max_rank: int = 50,
                         threshold: float = 0.01) -> List[int]:
    """
    估计Tucker秩
    
    参数:
        tensor: 输入张量
        max_rank: 最大秩
        threshold: 能量保留阈值
    
    返回:
        估计的秩列表
    """
    n_modes = tensor.ndim
    ranks = []
    
    for mode in range(n_modes):
        X_n = unfold(tensor, mode)
        _, s, _ = svd(X_n, full_matrices=False)
        
        # 累积能量
        energy = np.cumsum(s**2) / np.sum(s**2)
        
        # 找到保留(1-threshold)能量的秩
        r = np.searchsorted(energy, 1 - threshold) + 1
        ranks.append(min(r, max_rank, len(s)))
    
    return ranks
```

### 3.7 完整示例

```python
def demo_tucker():
    """
    Tucker分解完整演示
    """
    print("=" * 60)
    print("Tucker张量分解演示")
    print("=" * 60)
    
    # 创建测试张量
    np.random.seed(42)
    I, J, K = 50, 60, 40
    R1, R2, R3 = 5, 6, 4
    
    # 生成低秩张量
    core_true = np.random.randn(R1, R2, R3)
    U1 = np.random.randn(I, R1)
    U2 = np.random.randn(J, R2)
    U3 = np.random.randn(K, R3)
    
    tensor_true = tucker_to_tensor(core_true, [U1, U2, U3])
    
    # 添加噪声
    noise_level = 0.1
    tensor_noisy = tensor_true + noise_level * np.random.randn(*tensor_true.shape)
    
    print(f"张量形状: {tensor_true.shape}")
    print(f"真实秩: ({R1}, {R2}, {R3})")
    print(f"噪声级别: {noise_level}")
    
    # HOSVD
    print("\n[HOSVD]")
    core_hosvd, factors_hosvd = hosvd(tensor_noisy, [R1, R2, R3])
    recon_hosvd = tucker_to_tensor(core_hosvd, factors_hosvd)
    error_hosvd = norm(tensor_true - recon_hosvd) / norm(tensor_true)
    print(f"相对重构误差: {error_hosvd:.4f}")
    print(f"核心张量形状: {core_hosvd.shape}")
    print(f"压缩比: {np.prod(core_hosvd.shape) + sum(f.shape[0]*f.shape[1] for f in factors_hosvd)} / {np.prod(tensor_true.shape)}")
    
    # HOOI
    print("\n[HOOI]")
    core_hooi, factors_hooi = hooi(tensor_noisy, [R1, R2, R3], n_iter=50, verbose=True)
    recon_hooi = tucker_to_tensor(core_hooi, factors_hooi)
    error_hooi = norm(tensor_true - recon_hooi) / norm(tensor_true)
    print(f"相对重构误差: {error_hooi:.4f}")
    
    # 秩估计
    print("\n[秩估计]")
    estimated_ranks = estimate_tucker_rank(tensor_noisy, threshold=0.05)
    print(f"估计秩: {estimated_ranks}")
    
    return core_hooi, factors_hooi


def demo_image_compression():
    """
    图像压缩演示
    """
    from skimage import data, img_as_float
    import matplotlib.pyplot as plt
    
    # 加载彩色图像
    image = img_as_float(data.astronaut())
    print(f"图像形状: {image.shape}")
    
    # 不同压缩比
    rank_configs = [
        ([10, 10, 3], "Low (10,10,3)"),
        ([30, 30, 3], "Medium (30,30,3)"),
        ([50, 50, 3], "High (50,50,3)"),
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')
    
    for i, (ranks, label) in enumerate(rank_configs):
        core, factors = hooi(image, ranks, n_iter=20)
        reconstructed = np.clip(tucker_to_tensor(core, factors), 0, 1)
        
        # 计算压缩比
        original_size = np.prod(image.shape)
        compressed_size = (np.prod(core.shape) + 
                         sum(f.shape[0] * f.shape[1] for f in factors))
        ratio = compressed_size / original_size
        
        axes[0, i+1].imshow(reconstructed)
        axes[0, i+1].set_title(f'{label}\nRatio: {ratio:.2%}')
        axes[0, i+1].axis('off')
        
        # 差异图
        diff = np.abs(image - reconstructed).mean(axis=2)
        axes[1, i+1].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
        axes[1, i+1].set_title(f'Error: {np.mean(diff):.4f}')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_tucker()
    demo_image_compression()
```

---

## 4. 实验指南

### 4.1 测试数据集

| 数据集 | 维度 | 用途 |
|--------|------|------|
| 合成张量 | 可变 | 算法验证 |
| 图像集 | H×W×3 | 压缩 |
| 视频集 | T×H×W | 时序分析 |
| 知识图谱 | E×R×E | 链接预测 |

### 4.2 评估指标

```python
def evaluate_tucker(original, core, factors):
    """评估Tucker分解质量"""
    reconstructed = tucker_to_tensor(core, factors)
    
    return {
        'relative_error': norm(original - reconstructed) / norm(original),
        'rmse': np.sqrt(np.mean((original - reconstructed)**2)),
        'psnr': 10 * np.log10(1 / np.mean((original - reconstructed)**2)),
        'compression_ratio': _calc_compression_ratio(original.shape, core.shape, 
                                                      [f.shape for f in factors]),
        'core_sparsity': np.sum(np.abs(core) < 1e-6) / np.prod(core.shape)
    }
```

### 4.3 参数选择

- **秩选择**：从HOSVD的奇异值分布估计
- **正则化**：防止过拟合，通常 $10^{-6} \sim 10^{-4}$
- **迭代次数**：50-200次，视收敛情况

### 4.4 可视化

- 核心张量切片
- 因子矩阵
- 奇异值衰减曲线
- 误差收敛曲线

---

## 5. 习题与答案

### 5.1 理论题

**题目1**: 证明Tucker分解的表达式在元素形式下的正确性。

**答案**:
Tucker分解：$\mathcal{X} \approx \mathcal{G} \times_1 \mathbf{U}^{(1)} \times_2 \mathbf{U}^{(2)} \times_3 \mathbf{U}^{(3)}$

模-1乘积的元素形式：
$$(\mathcal{G} \times_1 \mathbf{U}^{(1)})_{j,i_2,i_3} = \sum_{r_1} g_{r_1,i_2,i_3} u_{j,r_1}^{(1)}$$

继续模-2乘积：
$$((\mathcal{G} \times_1 \mathbf{U}^{(1)}) \times_2 \mathbf{U}^{(2)})_{j,k,i_3} = \sum_{r_1,r_2} g_{r_1,r_2,i_3} u_{j,r_1}^{(1)} u_{k,r_2}^{(2)}$$

最后模-3乘积：
$$x_{i_1,i_2,i_3} \approx \sum_{r_1,r_2,r_3} g_{r_1,r_2,r_3} u_{i_1,r_1}^{(1)} u_{i_2,r_2}^{(2)} u_{i_3,r_3}^{(3)}$$

**题目2**: 解释Tucker分解与CP分解的关系。

**答案**:
CP分解是Tucker的特殊情况：
1. 核心张量为超对角：$g_{r_1,...,r_N} \neq 0$ 仅当 $r_1 = ... = r_N$
2. 所有模态秩相等：$R_1 = ... = R_N$
3. CP的权重 $\lambda_r$ 对应超对角元素

**题目3**: 分析HOSVD的时间复杂度。

**答案**:
对于 $N$ 阶张量 $\mathcal{X} \in \mathbb{R}^{I_1 \times \cdots \times I_N}$：

每个模态n的SVD：
- 模-n展开：$\mathbb{R}^{I_n \times \prod_{k \neq n} I_k}$
- SVD复杂度：$O(I_n^2 \cdot \prod_{k \neq n} I_k)$

假设 $I_n \approx I$：
- 单模SVD：$O(I^{N+1})$
- 总复杂度：$O(N \cdot I^{N+1})$

### 5.2 编程题

**题目1**: 实现增量式Tucker分解（支持新数据添加）。

**答案**:
```python
def incremental_tucker(existing_core, existing_factors, new_tensor, mode=0, alpha=0.9):
    """
    增量式Tucker分解
    
    参数:
        existing_core: 现有核心张量
        existing_factors: 现有因子矩阵
        new_tensor: 新增数据
        mode: 增量方向
        alpha: 历史权重
    """
    n_modes = len(existing_factors)
    
    # 新数据的Tucker分解
    new_core, new_factors = hosvd(new_tensor, 
                                   [f.shape[1] for f in existing_factors])
    
    # 合并因子矩阵
    updated_factors = []
    for i in range(n_modes):
        if i == mode:
            # 增量方向：追加因子
            updated_factors.append(np.vstack([
                alpha * existing_factors[i],
                (1-alpha) * new_factors[i]
            ]))
        else:
            # 非增量方向：加权平均
            updated_factors.append(alpha * existing_factors[i] + (1-alpha) * new_factors[i])
    
    # 合并核心张量
    updated_core = alpha * existing_core + (1-alpha) * new_core
    
    return updated_core, updated_factors
```

**题目2**: 实现稀疏Tucker分解。

**答案**:
```python
def sparse_tucker_als(tensor, ranks, sparsity_lambda=0.1, n_iter=100):
    """
    带L1正则化的稀疏Tucker分解
    """
    n_modes = tensor.ndim
    core, factors = hosvd(tensor, ranks)
    
    for _ in range(n_iter):
        # 更新因子矩阵（带L1正则化）
        for mode in range(n_modes):
            X_n = unfold(tensor, mode)
            G_n = unfold(core, mode)
            other_factors = [factors[k] for k in range(n_modes) if k != mode]
            K = kronecker(other_factors[::-1])
            
            # 软阈值
            A = X_n @ K.T @ G_n.T
            factors[mode] = np.sign(A) * np.maximum(np.abs(A) - sparsity_lambda, 0)
            
            # 归一化
            for r in range(ranks[mode]):
                norm = np.linalg.norm(factors[mode][:, r])
                if norm > 0:
                    factors[mode][:, r] /= norm
        
        # 更新核心张量
        core = _compute_core(tensor, factors)
        core = np.sign(core) * np.maximum(np.abs(core) - sparsity_lambda, 0)
    
    return core, factors
```

---

## 参考文献

1. Tucker, L. R. (1966). Some mathematical notes on three-mode factor analysis. *Psychometrika*, 31(3), 279-311.

2. De Lathauwer, L., De Moor, B., & Vandewalle, J. (2000). A multilinear singular value decomposition. *SIAM J. Matrix Anal. Appl.*, 21(4), 1253-1278.

3. Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications. *SIAM Review*, 51(3), 455-500.

4. Oseledets, I. V. (2011). Tensor-train decomposition. *SIAM J. Sci. Comput.*, 33(5), 2295-2317.
