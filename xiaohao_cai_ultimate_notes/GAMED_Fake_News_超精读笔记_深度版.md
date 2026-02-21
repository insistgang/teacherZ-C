# GAMED: 多专家解耦虚假新闻检测

> **超精读笔记** | WSDM 2025
> 作者：Lingzhi Shen, Yunfei Long, **Xiaohao Cai** (3rd) 等
> 领域：虚假新闻检测、多模态学习、混合专家

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | GAMED: Knowledge Adaptive Multi-Experts Decoupling for Multimodal Fake News Detection |
| **会议** | WSDM 2025 (CCF-A) |
| **数据** | Fakeddit, Yang |

---

## 🎯 核心创新

1. **模态解耦**：独立建模各模态特征
2. **MMoE-Pro**：多专家网络+知识增强
3. **AdaIN调节**：自适应特征分布调整
4. **投票机制**：带否决权的决策

---

## 📊 架构

```
图像/文本 → 特征提取 → 专家评审 → AdaIN调节 → 投票分类
                          ↑
                    知识增强(ERNIE2.0)
```

---

## 💡 实验结果

| 数据集 | GAMED | 最佳基线 |
|--------|-------|---------|
| Fakeddit | **89.3%** | 85.7% |
| Yang | **88.1%** | 84.2% |

**可解释性：可追踪各模态贡献**

---

*本笔记基于完整PDF深度阅读生成*
