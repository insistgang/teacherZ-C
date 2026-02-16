# 第十三讲：MCMC方法

## Markov Chain Monte Carlo Methods

---

### 📋 本讲大纲

1. 蒙特卡洛方法基础
2. 马尔可夫链理论
3. Metropolis-Hastings算法
4. Gibbs采样
5. 图像处理中的应用

---

### 13.1 为什么需要采样？

#### 贝叶斯推断的挑战

后验分布通常：
- 没有解析形式
- 高维
- 复杂的相关结构

#### 蒙特卡洛积分

$$\mathbb{E}[f(X)] = \int f(x) p(x) dx \approx \frac{1}{N}\sum_{i=1}^N f(x^{(i)})$$

其中 $x^{(i)} \sim p(x)$

---

### 13.2 马尔可夫链基础

#### 定义

序列 $\{X_t\}$ 满足：
$$P(X_{t+1}|X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1}|X_t)$$

#### 转移矩阵

$$P_{ij} = P(X_{t+1} = j | X_t = i)$$

#### 平稳分布

若 $\pi P = \pi$，则 $\pi$ 是平稳分布

**动画建议**：展示马尔可夫链收敛到平稳分布的过程

---

### 13.3 MCMC核心思想

#### 构造马尔可夫链

设计转移核使得平稳分布 $\pi$ 等于目标分布 $p(x)$

#### 收敛条件

1. **不可约**：任意状态可到达任意其他状态
2. **非周期**：不陷入周期循环
3. **细致平衡**：$\pi(x)P(x'|x) = \pi(x')P(x|x')$

---

### 13.4 Metropolis-Hastings算法

#### 算法

```
输入: 目标分布 p(x), 提议分布 q(x'|x)
初始化 x_0
for t = 1, 2, ..., N do
  1. 提议: x' ~ q(x'|x_{t-1})
  2. 计算接受率:
     α = min(1, p(x')q(x_{t-1}|x') / (p(x_{t-1})q(x'|x_{t-1})))
  3. 接受/拒绝:
     if u < α (u ~ Uniform(0,1)):
       x_t = x'
     else:
       x_t = x_{t-1}
end for
```

#### 关键点

- 接受率保证细致平衡
- 只需要 p 的相对值（不需要归一化常数）

---

### 13.5 特殊MH采样器

#### 独立MH

$$q(x'|x) = q(x')$$

与当前状态无关

#### 随机游走MH

$$x' = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

#### 自适应MH

根据历史样本调整提议分布

---

### 13.6 Gibbs采样

#### 算法

对于多维变量 $x = (x_1, \ldots, x_d)$：

```
for t = 1, 2, ..., N do
  for i = 1, 2, ..., d do
    x_i^{(t)} ~ p(x_i | x_{-i}^{(t-1)})
  end for
end for
```

其中 $x_{-i} = (x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_d)$

#### 特点

- 不需要计算接受率（总是接受）
- 需要条件分布可采样
- 适用于高维问题

---

### 13.7 图像处理中的Gibbs采样

#### 图像模型

Gibbs分布：
$$p(x) = \frac{1}{Z}\exp\left(-\frac{1}{T}E(x)\right)$$

#### 图像分割

Potts模型：
$$E(\ell) = -\sum_{<i,j>} \beta \delta(\ell_i, \ell_j)$$

Gibbs采样更新每个像素标签：
$$P(\ell_i | \ell_{-i}) \propto \exp\left(\beta \sum_{j \in N(i)} \delta(\ell_i, \ell_j)\right)$$

---

### 13.8 图像去噪的MCMC

#### 模型

$$p(x|y) \propto \exp\left(-\frac{1}{2\sigma^2}\|x-y\|_2^2 - \lambda\|x\|_{TV}\right)$$

#### 采样策略

1. **逐像素更新**：
   $$x_i \sim p(x_i | x_{-i}, y)$$

2. **块更新**：一次更新一个区域

#### 计算效率

- 慢收敛
- 高维空间探索困难

---

### 13.9 Hamiltonian Monte Carlo

#### 思想

引入动量变量 $p$，定义哈密顿量：
$$H(x, p) = U(x) + K(p) = -\log p(x) + \frac{1}{2}p^T M^{-1} p$$

#### 演化

$$\frac{dx}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial x}$$

#### 优势

- 更高效的探索
- 减少自相关
- 适合高维问题

---

### 13.10 收敛诊断

#### 自相关

$$\rho_k = \frac{\text{Cov}(X_t, X_{t+k})}{\text{Var}(X_t)}$$

#### 有效样本量

$$ESS = \frac{N}{1 + 2\sum_{k=1}^{\infty}\rho_k}$$

#### Gelman-Rubin统计量

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

$\hat{R} \approx 1$ 表示收敛

---

### 13.11 实际考虑

#### Burn-in

丢弃初始样本，等待链收敛

#### Thinning

每隔 k 步保存一个样本，减少自相关

#### 多链

运行多条独立链，增加可靠性

---

### 13.12 变分推断 vs MCMC

| 方面 | MCMC | 变分推断 |
|------|------|----------|
| 准确性 | 渐近精确 | 近似 |
| 计算成本 | 高 | 低 |
| 可扩展性 | 差 | 好 |
| 不确定性 | 完整 | 受限 |

#### 选择建议

- 小规模、精确结果：MCMC
- 大规模、快速结果：变分推断

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           MCMC方法要点                           │
├─────────────────────────────────────────────────┤
│                                                 │
│   核心：构造马尔可夫链，平稳分布=目标分布         │
│                                                 │
│   主要算法：                                     │
│   • Metropolis-Hastings：通用，需接受率         │
│   • Gibbs采样：条件分布可采样，无拒绝           │
│   • HMC：高效探索，适合高维                     │
│                                                 │
│   图像处理应用：                                 │
│   • 图像分割：Potts模型采样                     │
│   • 图像去噪：TV先验采样                        │
│   • 不确定性量化                                │
│                                                 │
│   诊断：自相关、ESS、Gelman-Rubin               │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **实现题**：实现Metropolis-Hastings算法采样高斯混合分布

2. **实现题**：实现Gibbs采样用于图像分割

3. **分析题**：比较MH和Gibbs的收敛速度

4. **应用题**：用MCMC实现贝叶斯图像去噪

---

### 📖 扩展阅读

1. **经典教材**：
   - Brooks et al., *Handbook of Markov Chain Monte Carlo*
   - Robert & Casella, *Monte Carlo Statistical Methods*

2. **软件工具**：
   - PyMC3/PyMC4 (Python)
   - Stan (通用)
   - JAGS (贝叶斯分析)

3. **论文**：
   - Neal, "MCMC using Hamiltonian dynamics", 2011

---

### 📖 参考文献

1. Brooks, S., et al. (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press.

2. Robert, C.P. & Casella, G. (2004). *Monte Carlo Statistical Methods*. Springer.

3. Neal, R.M. (2011). MCMC using Hamiltonian dynamics. In *Handbook of MCMC*.

4. Geman, S. & Geman, D. (1984). Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. *IEEE PAMI*.
