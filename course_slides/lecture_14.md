# 第十四讲：近端算法

## Proximal Algorithms

---

### 📋 本讲大纲

1. 近端算子定义
2. 近端梯度法
3. ADMM算法
4. Split Bregman方法
5. 图像处理应用

---

### 14.1 近端算子

#### 定义

函数 $f$ 的近端算子：
$$\text{prox}_{\lambda f}(v) = \arg\min_x \left( f(x) + \frac{1}{2\lambda}\|x - v\|_2^2 \right)$$

#### 直观理解

- 输入 $v$，输出一个在 $f$ 的值和与 $v$ 的距离之间权衡的点
- $\lambda$ 控制权衡强度

**动画建议**：展示近端算子在简单函数上的作用

---

### 14.2 常见近端算子

#### 软阈值 (Soft Thresholding)

对于 $f(x) = \|x\|_1$：
$$\text{prox}_{\lambda\|\cdot\|_1}(v) = \text{sign}(v) \max(|v| - \lambda, 0) = S_\lambda(v)$$

#### 硬阈值

对于 $f(x) = \|x\|_0$（非凸）：
$$\text{prox}_{\lambda\|\cdot\|_0}(v) = v \cdot \mathbf{1}_{|v| > \sqrt{2\lambda}}$$

#### 投影

对于约束集 $\mathcal{C}$：
$$\text{prox}_{I_\mathcal{C}}(v) = \text{proj}_\mathcal{C}(v)$$

---

### 14.3 更多近端算子

| 函数 $f(x)$ | 近端算子 |
|-------------|----------|
| $\frac{1}{2}\|x\|_2^2$ | $\frac{v}{1+\lambda}$ |
| $\|x\|_1$ | $S_\lambda(v)$ |
| $\|x\|_2$ | $\max(1 - \lambda/\|v\|_2, 0) \cdot v$ |
| $I_{\|x\|_2 \leq 1}$ | $v / \max(1, \|v\|_2)$ |
| $I_{x \geq 0}$ | $\max(v, 0)$ |

---

### 14.4 近端算子的性质

#### 恒等式

$$\text{prox}_{\lambda f}(v) = (I + \lambda \partial f)^{-1}(v)$$

#### Moreau分解

$$\text{prox}_{\lambda f}(v) + \lambda \text{prox}_{f^*/\lambda}(v/\lambda) = v$$

其中 $f^*$ 是 $f$ 的Fenchel共轭

#### 组合

$$\text{prox}_{\lambda(f+g)} \neq \text{prox}_{\lambda f} \circ \text{prox}_{\lambda g}$$

（一般不成立）

---

### 14.5 近端梯度法

#### 问题形式

$$\min_x f(x) + g(x)$$

其中 $f$ 可微（Lipschitz梯度），$g$ 不可微但近端算子已知

#### 算法

```
x^{k+1} = prox_{λg}(x^k - λ∇f(x^k))
```

#### 收敛性

若 $\lambda \leq 1/L$（$L$是$\nabla f$的Lipschitz常数）：
$$f(x^k) + g(x^k) - f(x^*) - g(x^*) = O(1/k)$$

---

### 14.6 加速近端梯度

#### FISTA (Beck & Teboulle, 2009)

```
y^1 = x^0, t^1 = 1
for k = 1, 2, ... do
  x^k = prox_{λg}(y^k - λ∇f(y^k))
  t^{k+1} = (1 + sqrt(1 + 4(t^k)^2)) / 2
  y^{k+1} = x^k + ((t^k - 1) / t^{k+1})(x^k - x^{k-1})
end for
```

#### 收敛率

$$O(1/k^2)$$

---

### 14.7 ADMM

#### 问题形式

$$\min_{x,z} f(x) + g(z) \quad \text{s.t.} \quad Ax + Bz = c$$

#### 增广Lagrangian

$$L_\rho(x, z, y) = f(x) + g(z) + y^T(Ax + Bz - c) + \frac{\rho}{2}\|Ax + Bz - c\|_2^2$$

#### 算法

```
repeat
  x^{k+1} = argmin_x L_ρ(x, z^k, y^k)
  z^{k+1} = argmin_z L_ρ(x^{k+1}, z, y^k)
  y^{k+1} = y^k + ρ(Ax^{k+1} + Bz^{k+1} - c)
until 收敛
```

---

### 14.8 ADMM的特殊形式

#### 标准形式

$$\min_x f(x) + g(Ax)$$

引入 $z = Ax$：
```
x^{k+1} = prox_{f}(x^k - τ A^T y^k)
z^{k+1} = prox_{g/ρ}(Ax^{k+1} + y^k/ρ)
y^{k+1} = y^k + ρ(Ax^{k+1} - z^{k+1})
```

#### 共识形式

并行优化多个子问题：
$$\min \sum_i f_i(x) + g(x)$$

---

### 14.9 Split Bregman

#### 问题形式

$$\min_x f(x) + g(Ax)$$

#### 等价形式

$$\min_{x,d} f(x) + g(d) \quad \text{s.t.} \quad d = Ax$$

#### 算法

```
repeat
  x^{k+1} = argmin_x f(x) + (μ/2)||Ax - d^k + b^k||²
  d^{k+1} = prox_{g/μ}(Ax^{k+1} + b^k)
  b^{k+1} = b^k + Ax^{k+1} - d^{k+1}
until 收敛
```

#### 与ADMM的关系

Split Bregman = ADMM的对偶形式

---

### 14.10 TV去噪的Split Bregman

#### 问题

$$\min_u \frac{1}{2}\|u - f\|_2^2 + \lambda \|u\|_{TV}$$

#### 等价形式

$$\min_{u,d} \frac{1}{2}\|u - f\|_2^2 + \lambda \|d\|_1 \quad \text{s.t.} \quad d = \nabla u$$

#### 算法

```
repeat
  u^{k+1} = (I - μΔ)^{-1}(f + div(d^k - b^k))  // Poisson求解
  d^{k+1} = shrink(∇u^{k+1} + b^k, λ/μ)        // 软阈值
  b^{k+1} = b^k + ∇u^{k+1} - d^{k+1}
until 收敛
```

---

### 14.11 算法比较

| 算法 | 适用问题 | 收敛率 | 每步代价 |
|------|----------|--------|----------|
| 近端梯度 | $f+g$ | $O(1/k)$ | 低 |
| FISTA | $f+g$ | $O(1/k^2)$ | 低 |
| ADMM | $f(x)+g(z)$ | $O(1/k)$ | 中 |
| Split Bregman | $f+g(Ax)$ | $O(1/k)$ | 中 |

#### 选择建议

- 单一非光滑项：近端梯度/FISTA
- 多个非光滑项：ADMM/Split Bregman
- 大规模：并行ADMM

---

### 14.12 Primal-Dual方法

#### Chambolle-Pock算法

对于 $\min_x f(x) + g(Ax)$：

```
repeat
  x^{k+1} = prox_{τf}(x^k - τ A^T y^k)
  ȳ^{k+1} = prox_{σg^*}(y^k + σ A(2x^{k+1} - x^k))
  y^{k+1} = ȳ^{k+1}
until 收敛
```

#### 优势

- 不需要求逆
- $O(1/k)$ 收敛
- 适用于各种问题

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           近端算法框架                           │
├─────────────────────────────────────────────────┤
│                                                 │
│   近端算子：                                     │
│   prox_{λf}(v) = argmin f(x) + (1/2λ)||x-v||²  │
│                                                 │
│   核心算法：                                     │
│   • 近端梯度：x+ = prox(x - λ∇f)               │
│   • FISTA：加速 O(1/k²)                        │
│   • ADMM：交替方向乘子法                        │
│   • Split Bregman：分裂+辅助变量               │
│                                                 │
│   图像处理应用：                                 │
│   • TV去噪/去模糊                               │
│   • 框架稀疏                                    │
│   • 约束优化                                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **推导题**：推导 $\ell_2$ 范数的近端算子

2. **实现题**：实现FISTA算法用于LASSO问题

3. **实现题**：实现Split Bregman用于TV去噪

4. **比较题**：比较近端梯度法和ADMM的计算效率

---

### 📖 扩展阅读

1. **经典论文**：
   - Parikh & Boyd, "Proximal Algorithms", Foundations and Trends in Optimization, 2014
   - Beck & Teboulle, "A fast iterative shrinkage-thresholding algorithm", SIAM J. Imaging Sci., 2009

2. **教材**：
   - Boyd & Vandenberghe, *Convex Optimization*

3. **Cai相关论文**：
   - Split Bregman方法的收敛性分析

---

### 📖 参考文献

1. Parikh, N. & Boyd, S. (2014). Proximal algorithms. *Foundations and Trends in Optimization*, 1(3), 127-239.

2. Beck, A. & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM J. Imaging Sci.*, 2(1), 183-202.

3. Goldstein, T. & Osher, S. (2009). The split Bregman method for L1-regularized problems. *SIAM J. Imaging Sci.*, 2(2), 323-343.

4. Chambolle, A. & Pock, T. (2011). A first-order primal-dual algorithm for convex problems. *JMIV*, 40(1), 120-145.
