# BABNet: 雷达工作模式识别

> **超精读笔记** | 完整PDF分析
> 论文来源：Digital Signal Processing 133 (2023)
> 作者：Mingyang Du, Ping Zhong, **Xiaohao Cai**, Daping Bi, Aiqi Jing
> 领域：雷达识别、贝叶斯神经网络、注意力机制

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | Robust Bayesian Attention Belief Network for Radar Work Mode Recognition |
| **期刊** | Digital Signal Processing |
| **年份** | 2023 |
| **文章编号** | 103874 |

---

## 🎯 核心创新

1. **贝叶斯神经网络**：权重概率分布增强鲁棒性
2. **注意力机制**：替代RNN处理变长序列
3. **预训练CNN先验**：加速收敛避免局部最小

---

## 📊 鲁棒性测试

| 噪声类型 | BABNet | 传统CNN |
|---------|--------|---------|
| 测量误差 | 85%+ | 60% |
| 丢失脉冲 | 80%+ | 55% |
| 虚假脉冲 | 78%+ | 50% |

---

*本笔记基于DSP 2023完整论文生成*
