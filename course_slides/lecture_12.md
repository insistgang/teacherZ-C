# 第十二讲：贝叶斯推断

## Bayesian Inference

---

### 📋 本讲大纲

1. 贝叶斯框架介绍
2. 先验分布选择
3. 后验分布计算
4. MAP估计
5. 图像处理中的应用

---

### 12.1 贝叶斯定理

#### 基本形式

$$\boxed{p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)}}$$

#### 分解

- $p(\theta | D)$：**后验分布** (Posterior)
- $p(D | \theta)$：**似然函数** (Likelihood)
- $p(\theta)$：**先验分布** (Prior)
- $p(D)$：**边缘似然** (Evidence)

**动画建议**：展示先验如何通过数据更新为后验

---

### 12.2 图像处理的贝叶斯框架

#### 观测模型

$$y = Ax + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

#### 似然函数

$$p(y|x) = \frac{1}{(2\pi\sigma^2)^{n/2}} \exp\left(-\frac{\|Ax - y\|_2^2}{2\sigma^2}\right)$$

#### 后验分布

$$p(x|y) \propto p(y|x) p(x)$$

---

### 12.3 先验分布

#### 高斯先验

$$p(x) \propto \exp\left(-\frac{\lambda}{2}\|Lx\|_2^2\right)$$

- 对应Tikhonov正则化
- $L$ 可以是梯度算子

#### 拉普拉斯先验

$$p(x) \propto \exp\left(-\lambda\|x\|_1\right)$$

- 促进稀疏性
- 对应$\ell_1$正则化

#### TV先验

$$p(x) \propto \exp\left(-\lambda\|x\|_{TV}\right)$$

- 边缘保持
- 非高斯分布

---

### 12.4 先验与正则化的关系

| 先验分布 | 正则化项 | 对应模型 |
|----------|----------|----------|
| 高斯 | $\|\cdot\|_2^2$ | Tikhonov |
| 拉普拉斯 | $\|\cdot\|_1$ | LASSO |
| 指数 | $\|\cdot\|_{TV}$ | ROF |
| 学生t | $\log(1+x^2)$ | 重尾分布 |

#### 统一框架

$$-\log p(x|y) = \text{似然项} + \text{先验项} + \text{常数}$$

---

### 12.5 MAP估计

#### 定义

Maximum A Posteriori：
$$x_{MAP} = \arg\max_x p(x|y) = \arg\min_x \left[-\log p(y|x) - \log p(x)\right]$$

#### 等价形式

$$x_{MAP} = \arg\min_x \left[\frac{1}{2\sigma^2}\|Ax - y\|_2^2 + \lambda \Phi(x)\right]$$

#### 与变分方法的联系

MAP估计 = 变分能量最小化

---

### 12.6 高斯情形的解析解

#### 模型

- 似然：$p(y|x) = \mathcal{N}(Ax, \sigma^2 I)$
- 先验：$p(x) = \mathcal{N}(0, \Lambda^{-1})$

#### 后验分布

$$p(x|y) = \mathcal{N}(\mu, \Sigma)$$

其中：
$$\Sigma = (A^T A / \sigma^2 + \Lambda)^{-1}$$
$$\mu = \Sigma A^T y / \sigma^2$$

---

### 12.7 共轭先验

#### 定义

若后验分布与先验分布同族，则称先验是共轭的

#### 常见共轭对

| 似然 | 共轭先验 | 后验 |
|------|----------|------|
| 高斯(已知方差) | 高斯 | 高斯 |
| 高斯(已知均值) | 逆Gamma | 逆Gamma |
| 伯努利 | Beta | Beta |
| 泊松 | Gamma | Gamma |

---

### 12.8 超参数估计

#### 经验贝叶斯

最大化边缘似然：
$$\lambda^* = \arg\max_\lambda p(y|\lambda) = \arg\max_\lambda \int p(y|x)p(x|\lambda) dx$$

#### 证据近似

$$p(y) \approx p(y|\lambda^*)$$

#### 应用

- 自动选择正则化参数
- 噪声水平估计

---

### 12.9 层次贝叶斯模型

#### 结构

```
超参数 → 先验参数 → 隐变量 → 观测
   θ    →    λ    →    x   →   y
```

#### 例子：稀疏贝叶斯学习

$$p(y|x) = \mathcal{N}(Ax, \sigma^2 I)$$
$$p(x|\alpha) = \mathcal{N}(0, \text{diag}(\alpha)^{-1})$$
$$p(\alpha) = \prod_i \text{Gamma}(\alpha_i | a, b)$$

#### 优势

- 自动确定稀疏度
- 不需要交叉验证

---

### 12.10 贝叶斯推断 vs 优化

| 方面 | 贝叶斯推断 | 正则化优化 |
|------|------------|------------|
| 输出 | 后验分布 | 点估计 |
| 不确定性 | 可量化 | 不可 |
| 参数选择 | 自动/层次 | 需调参 |
| 计算复杂度 | 高 | 低 |
| 模型比较 | 证据框架 | 交叉验证 |

---

### 12.11 应用实例

#### 图像去噪

$$p(x|y) \propto \exp\left(-\frac{1}{2\sigma^2}\|x-y\|_2^2 - \lambda\|x\|_{TV}\right)$$

MAP估计 = ROF去噪

#### 图像分割

$$p(\ell|y) \propto p(y|\ell) p(\ell)$$

其中 $\ell$ 是标签场，先验可以是Potts模型

#### 压缩感知

$$p(x|y) \propto p(y|x) p(x|\lambda)$$

稀疏先验促进稀疏解

---

### 12.12 计算方法

#### 确定性方法

- MAP估计（优化）
- 变分推断（VI）
- 期望传播（EP）

#### 随机方法

- MCMC采样
- Gibbs采样
- HMC

**动画建议**：展示后验分布的采样过程

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           贝叶斯推断框架                         │
├─────────────────────────────────────────────────┤
│                                                 │
│   贝叶斯定理：                                   │
│   posterior ∝ likelihood × prior               │
│                                                 │
│   MAP估计：                                     │
│   x_MAP = argmin[-log p(y|x) - log p(x)]       │
│         = argmin[数据项 + 正则项]              │
│                                                 │
│   先验选择：                                     │
│   • 高斯 → ℓ₂正则化                            │
│   • 拉普拉斯 → ℓ₁正则化                        │
│   • TV先验 → TV正则化                          │
│                                                 │
│   优势：不确定性量化、自动参数选择               │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **推导题**：推导高斯似然+高斯先验的后验分布

2. **实现题**：实现贝叶斯图像去噪（MAP估计）

3. **比较题**：比较不同先验对MAP估计的影响

4. **分析题**：分析层次贝叶斯模型如何自动选择参数

---

### 📖 扩展阅读

1. **经典教材**：
   - Bishop, *Pattern Recognition and Machine Learning*
   - Murphy, *Machine Learning: A Probabilistic Perspective*

2. **论文**：
   - Tipping, "Sparse Bayesian Learning and the Relevance Vector Machine", JMLR, 2001

3. **Cai相关论文**：
   - Bayesian框架下的图像分割与重建

---

### 📖 参考文献

1. Bishop, C.M. (2006). *Pattern Recognition and Machine Learning*. Springer.

2. Murphy, K.P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.

3. Tipping, M.E. (2001). Sparse Bayesian learning and the relevance vector machine. *JMLR*, 1, 211-244.

4. Geman, S. & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *IEEE PAMI*.
