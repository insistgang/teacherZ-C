# Challenge 4: Tucker分解 ⭐⭐⭐⭐

## 题目描述

用单个函数实现Tucker分解(HOSVD - Higher-Order SVD)。

Tucker分解将张量分解为核心张量和各模态的因子矩阵：
```
X ≈ G ×₁ A ×₂ B ×₃ C
```

其中G是核心张量，A、B、C是因子矩阵。

## 输入输出格式

**输入**:
```python
X: np.ndarray  # 3阶张量，shape=(I, J, K)
ranks: Tuple[int, int, int]  # 各模态的秩，默认(None,None,None)表示全秩
```

**输出**:
```python
G: np.ndarray  # 核心张量，shape=(R1, R2, R3)
A: np.ndarray  # 因子矩阵1，shape=(I, R1)
B: np.ndarray  # 因子矩阵2，shape=(J, R2)
C: np.ndarray  # 因子矩阵3，shape=(K, R3)
```

## 示例

```python
import numpy as np

X = np.random.rand(4, 5, 6)
G, A, B, C = tucker_decompose(X, ranks=(3, 3, 3))

print(G.shape)  # (3, 3, 3)
print(A.shape)  # (4, 3)
print(B.shape)  # (5, 3)
print(C.shape)  # (6, 3)

# 重建误差
X_reconstructed = tucker_reconstruct(G, A, B, C)
error = np.linalg.norm(X - X_reconstructed) / np.linalg.norm(X)
print(f"相对误差: {error:.4f}")  # 通常<0.1
```

## 数据范围

- 张量维度: 2×2×2 到 256×256×256
- 秩: 1 到 各维度大小
- 数据类型: float64

## 限制条件

| 限制项 | 要求 |
|--------|------|
| 有效代码行数 | ≤50行 |
| Python版本 | 3.8+ |
| 允许使用 | numpy.linalg.svd |
| 禁止使用 | tensorly等张量库 |

## 评分标准

| 维度 | 分数 | 说明 |
|------|------|------|
| 代码行数 | 30分 | ≤30行满分 |
| 正确性 | 40分 | 重建误差<5%得满分 |
| 效率 | 30分 | 64×64×64张量<5秒 |

## 参考答案

### HOSVD实现 (约35行)
```python
import numpy as np

def tucker_decompose(X, ranks=None):
    """
    Tucker分解 (HOSVD)
    
    Args:
        X: 输入张量 shape=(I, J, K)
        ranks: 目标秩 (R1, R2, R3)
    
    Returns:
        G: 核心张量
        A, B, C: 因子矩阵
    """
    I, J, K = X.shape
    if ranks is None:
        ranks = (I, J, K)
    
    # 模态1展开
    X1 = X.reshape(I, J*K)
    U1, _, _ = np.linalg.svd(X1, full_matrices=False)
    A = U1[:, :ranks[0]]
    
    # 模态2展开
    X2 = np.transpose(X, (1, 0, 2)).reshape(J, I*K)
    U2, _, _ = np.linalg.svd(X2, full_matrices=False)
    B = U2[:, :ranks[1]]
    
    # 模态3展开
    X3 = np.transpose(X, (2, 0, 1)).reshape(K, I*J)
    U3, _, _ = np.linalg.svd(X3, full_matrices=False)
    C = U3[:, :ranks[2]]
    
    # 计算核心张量
    G = X.copy()
    G = np.tensordot(G, A.T, axes=([0], [1]))
    G = np.tensordot(G, B.T, axes=([0], [1]))
    G = np.tensordot(G, C.T, axes=([0], [1]))
    
    return G, A, B, C

def tucker_reconstruct(G, A, B, C):
    """重建张量"""
    X = G.copy()
    X = np.tensordot(X, A, axes=([0], [1]))
    X = np.tensordot(X, B, axes=([0], [1]))
    X = np.tensordot(X, C, axes=([0], [1]))
    return X
```

### 紧凑版 (约20行)
```python
import numpy as np

def tucker_decompose(X, ranks=None):
    s = X.shape
    r = ranks or s
    
    def mode_n_svd(x, n, rank):
        x_n = np.moveaxis(x, n, 0).reshape(s[n], -1)
        U, _, _ = np.linalg.svd(x_n, full_matrices=False)
        return U[:, :rank]
    
    factors = [mode_n_svd(X, i, r[i]) for i in range(3)]
    
    core = X.copy()
    for i, U in enumerate(factors):
        core = np.tensordot(core, U.T, axes=([0], [0]))
    
    return (core,) + tuple(factors)
```

### 极简版 (约15行)
```python
import numpy as np

def tucker_decompose(X,r=None):
    s=X.shape;r=r or s
    U=[np.linalg.svd(np.moveaxis(X,i,0).reshape(s[i],-1),0)[0][:,:r[i]]for i in range(3)]
    G=X
    for i,u in enumerate(U):G=np.tensordot(G,u.T,([0],[0]))
    return G,U[0],U[1],U[2]
```

## 优化提示

1. **模态展开技巧**:
   - 使用`np.moveaxis`或`np.transpose`
   - `reshape`时注意内存布局
   - 考虑Fortran顺序提高效率

2. **SVD优化**:
   - 只计算需要的奇异向量
   - 使用`full_matrices=False`
   - 考虑截断SVD

3. **核心张量计算**:
   - 使用`np.tensordot`代替显式循环
   - 注意axes参数的正确设置
   - 可以用`np.einsum`优化

4. **内存优化**:
   - 分批处理大张量
   - 使用原地操作
   - 考虑稀疏张量

## 测试用例

### 测试1: 基本功能
```python
def test_basic():
    X = np.random.rand(4, 5, 6)
    G, A, B, C = tucker_decompose(X)
    assert G.shape == (4, 5, 6)
    assert A.shape == (4, 4)
    assert B.shape == (5, 5)
    assert C.shape == (6, 6)
```

### 测试2: 低秩分解
```python
def test_low_rank():
    X = np.random.rand(10, 10, 10)
    G, A, B, C = tucker_decompose(X, ranks=(3, 3, 3))
    
    # 重建
    X_hat = np.einsum('abc,ia,jb,kc->ijk', G, A, B, C)
    
    # 检查形状
    assert X_hat.shape == X.shape
    
    # 检查重建误差
    rel_error = np.linalg.norm(X - X_hat) / np.linalg.norm(X)
    print(f"相对误差: {rel_error:.4f}")
```

### 测试3: 正交性
```python
def test_orthogonality():
    X = np.random.rand(8, 8, 8)
    G, A, B, C = tucker_decompose(X)
    
    # 因子矩阵应近似正交
    assert np.allclose(A.T @ A, np.eye(A.shape[1]), atol=1e-10)
    assert np.allclose(B.T @ B, np.eye(B.shape[1]), atol=1e-10)
    assert np.allclose(C.T @ C, np.eye(C.shape[1]), atol=1e-10)
```

### 测试4: 性能测试
```python
def test_performance():
    import time
    X = np.random.rand(64, 64, 64)
    
    start = time.time()
    G, A, B, C = tucker_decompose(X, ranks=(32, 32, 32))
    elapsed = time.time() - start
    
    assert elapsed < 5.0, f"耗时{elapsed:.2f}s超过5秒"
```

### 测试5: 代码行数
```python
def test_line_count():
    import inspect
    source = inspect.getsource(tucker_decompose)
    lines = [l for l in source.split('\n') 
             if l.strip() and not l.strip().startswith(('#','"""',"'''"))]
    # 排除def和return
    effective = [l for l in lines if 'def ' not in l and 'return' not in l]
    assert len(effective) <= 50
```

## 扩展思考

- [ ] 实现n阶张量通用版本
- [ ] 添加非负Tucker分解
- [ ] 实现稀疏Tucker分解
- [ ] 比较HOOI与HOSVD精度差异
