# 分布式无线电优化: 高维逆问题不确定性量化

> **超精读笔记** | arXiv 1811.02514v2
> 作者：**Xiaohao Cai** (1st), Marcelo Pereyra, Jason D. McEwen
> 领域：逆问题、不确定性量化、贝叶斯推断、凸优化

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation |
| **机构** | UCL, Heriot-Watt University |
| **arXiv** | 1811.02514v2 |
| **领域** | 图像处理、信号处理、贝叶斯推断 |

---

## 🎯 核心创新

1. **HPD区域近似**：高后验密度区域的不确定性可视化
2. **局部可信区间**：像素级和超像素级的误差棒
3. **自动正则化参数选择**：无需手动设定μ
4. **支持非光滑先验**：ℓ1范数等稀疏先验

---

## 📊 方法

### MAP估计

$$x_\mu^* = \arg\min_{x \in \mathbb{R}^N} \mu f(x) + g_y(x)$$

### HPD区域近似

$$\gamma'_\alpha = \mu f(x_\mu^*) + g_y(x_\mu^*) + \frac{\sqrt{16 \log(3/\alpha)}}{\sqrt{N}} + N$$

### 局部可信区间

$$\xi^-_{\Omega_i} = \min_\xi \{\xi | \mu f(x_{i,\xi}) + g_y(x_{i,\xi}) \leq \gamma'_\alpha\}$$

---

## 💡 实验结果

| 数据集 | 方法 | SNR |
|--------|------|-----|
| M31 | 分析先验+SARA | 31.09 dB |
| M31 | 合成先验+SARA | 23.66 dB |
| Brain | 分析先验+SARA | 23.63 dB |

**MAP估计比Px-MALA快O(10⁵)倍，误差<5%**

---

*本笔记基于完整PDF深度阅读生成*
