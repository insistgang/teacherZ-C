# LL4G: 自监督动态优化人格检测

> **超精读笔记** | arXiv 2504.02146v1
> 作者：Lingzhi Shen, Yunfei Long, **Xiaohao Cai** (3rd) 等
> 领域：人格检测、大语言模型、图神经网络

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | LL4G: Self-Supervised Dynamic Optimization for Graph-Based Personality Detection |
| **年份** | 2025 |
| **arXiv** | 2504.02146v1 |
| **任务** | MBTI人格类型预测 |

---

## 🎯 核心创新

1. **LLM语义提取**：Llama 3提取帖子语义特征
2. **动态图更新**：自适应添加节点和边
3. **显隐式关系推断**：语义相似+矛盾语言
4. **自监督训练**：节点重构+边预测+对比学习

---

## 📊 方法架构

### 节点语义特征

$$h_i = \text{Llama}(T_i) \in \mathbb{R}^d$$

### 显式边

$$A^{explicit}_{ij} = \text{sim}(h_i, h_j) + \lambda_p \cdot P_{ij}$$

### 隐式边

$$A^{implicit}_{ij} = f_{Llama}(T_i, T_j) + \lambda_c \cdot C_{ij}$$

### 最终邻接矩阵

$$A = A^{explicit} + A^{implicit}$$

---

## 💡 实验结果

| 数据集 | LL4G | 最佳基线 | 提升 |
|--------|------|---------|------|
| Kaggle | 76.8% | 68.3% | +8.47% |
| Pandora | 72.5% | 67.7% | +4.80% |

**在4个MBTI维度均超越SOTA**

---

*本笔记基于完整PDF深度阅读生成*
