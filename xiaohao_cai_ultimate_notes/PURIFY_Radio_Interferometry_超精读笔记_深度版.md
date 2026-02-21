# PURIFY: 分布式射电干涉成像

> **超精读笔记** | Astronomy and Computing 2019
> 作者：Luke Pratley, Jason D. McEwen, **Xiaohao Cai** (4th) 等
> 领域：射电天文、分布式计算、凸优化

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | Distributed and parallel sparse convex optimization for radio interferometry with PURIFY |
| **期刊** | Astronomy and Computing |
| **arXiv** | 1903.04502v2 |
| **规模** | 500亿可见度数据 |

---

## 🎯 核心创新

1. **分布式ADMM**：MPI实现的大规模优化
2. **稀疏正则化**：ℓ1范数+小波表示
3. **预采样核**：减少内存开销
4. **GPU/OpenMP**：多级并行

---

## 📊 方法

### 测量方程

$$y = WGFZSx$$

### 分布式优化

$$\min_x g(x) \quad \text{s.t.} \quad \|y - \Phi x\|_{\ell_2} \leq \epsilon$$

---

## 💡 性能

| 数据规模 | 节点数 | 时间/迭代 |
|---------|--------|---------|
| 1Gb | 25 | 100ms |
| 2.4Tb | 100 | 3min |

---

*本笔记基于完整PDF深度阅读生成*
