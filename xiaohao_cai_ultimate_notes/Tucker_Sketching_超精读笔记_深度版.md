# Tucker近似: 实用Sketching算法

> **超精读笔记** | J Sci Comput 2023
> 作者：Wandi Dong, Gaohang Yu, Liqun Qi, **Xiaohao Cai** (4th)
> 领域：张量分解、随机算法、高维数据

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | Practical Sketching Algorithms for Low-Rank Tucker Approximation of Large Tensors |
| **期刊** | Journal of Scientific Computing (2023) 95:52 |
| **领域** | 数值计算、张量分析 |

---

## 🎯 核心创新

1. **Sketch-STHOSVD**：单次sketching算法
2. **sub-Sketch-STHOSVD**：子空间幂迭代加速
3. **严格误差界**：理论保证
4. **高维扩展**：适用于大规模张量

---

## 📊 方法

### Sketch-STHOSVD

用双面sketching替代SVD：

$$Y = A\Omega, \quad W = \Psi A$$

### 子空间幂迭代

$$Y = (AA^T)^q A\Omega$$

---

## 💡 实验结果

| 算法 | 时间 | 相对误差 |
|------|------|---------|
| THOSVD | 基准 | 基准 |
| R-STHOSVD | -60% | +2% |
| **sub-Sketch** | **-80%** | **+0.5%** |

---

*本笔记基于完整PDF深度阅读生成*
