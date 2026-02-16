---
marp: true
theme: default
paginate: true
---

# 计算机视觉前沿方法

## 第二讲：凸优化理论

**凸集、凸函数、KKT条件、对偶理论**

---

# 本讲大纲

1. 凸集的定义与性质
2. 凸函数及其判定
3. 凸优化问题的一般形式
4. KKT最优性条件
5. 对偶理论与应用

---

# 一、凸集

## 定义

集合 $C \subseteq \mathbb{R}^n$ 是**凸集**，若对任意 $x, y \in C$ 和 $\theta \in [0,1]$：

$$\theta x + (1-\theta)y \in C$$

## 几何意义

集合中任意两点的连线仍在集合内

## 例子

- 仿射集：直线、平面、超平面
- 半空间：$\{x : a^Tx \leq b\}$
- 多面体：$\{x : Ax \leq b\}$
- 范数球：$\{x : \|x\|_p \leq 1\}$

---

# 凸集的性质

## 保持凸性的操作

1. **交集**：$C_1 \cap C_2$ 是凸集
2. **仿射变换**：$AC + b$ 是凸集
3. **透视变换**：保持凸性
4. **Minkowski和**：$C_1 + C_2 = \{x+y : x \in C_1, y \in C_2\}$

## 分离定理

若 $C$ 和 $D$ 是不相交的凸集，则存在 $a \neq 0$ 和 $b$ 使得：

$$a^Tx \leq b, \quad \forall x \in C$$
$$a^Tx \geq b, \quad \forall x \in D$$

---

# 凸集的例子

```
    凸集                  非凸集
   ┌───────┐              ┌───┐
   │       │              │   │
   │   ●   │              │ ● │   ╱
   │       │              │   │  ╱
   └───────┘              └───┘
   
   圆/椭圆                 星形
```

## 常见凸集

| 凸集 | 定义 |
|------|------|
| 超平面 | $\{x : a^Tx = b\}$ |
| 半空间 | $\{x : a^Tx \leq b\}$ |
| 范数球 | $\{x : \|x\| \leq r\}$ |
| 范数锥 | $\{(x,t) : \|x\| \leq t\}$ |
| 半正定锥 | $\mathbb{S}_+^n = \{X \succeq 0\}$ |

---

# 二、凸函数

## 定义

函数 $f: \mathbb{R}^n \to \mathbb{R}$ 是**凸函数**，若：
1. $\text{dom}(f)$ 是凸集
2. 对所有 $x, y \in \text{dom}(f)$ 和 $\theta \in [0,1]$：

$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$

## 一阶条件

若 $f$ 可微，则 $f$ 是凸函数当且仅当：

$$f(y) \geq f(x) + \nabla f(x)^T(y-x), \quad \forall x, y$$

几何意义：切平面始终在函数图像下方

---

# 凸函数的判定

## 二阶条件

若 $f$ 二阶可微，则 $f$ 是凸函数当且仅当：

$$\nabla^2 f(x) \succeq 0, \quad \forall x \in \text{dom}(f)$$

即Hessian矩阵半正定

## 例子

| 函数 | 条件 | 凸性 |
|------|------|------|
| $ax^2 + bx + c$ | $a \geq 0$ | 凸 |
| $e^{ax}$ | 任意 $a$ | 凸 |
| $\|x\|_p$ | $p \geq 1$ | 凸 |
| $\log(x)$ | $x > 0$ | 凹 |
| $x\log(x)$ | $x > 0$ | 凸 |

---

# 保持凸性的操作

## 1. 非负组合

若 $f_1, \ldots, f_m$ 是凸函数，则：

$$f(x) = \sum_{i=1}^m w_i f_i(x), \quad w_i \geq 0$$

是凸函数

## 2. 逐点最大值

$$f(x) = \max\{f_1(x), \ldots, f_m(x)\}$$

是凸函数

## 3. 复合函数

若 $g$ 是凸函数，$h$ 是凸且非减，则 $h(g(x))$ 是凸函数

---

# 三、凸优化问题

## 标准形式

$$\begin{aligned}
\min_x \quad & f_0(x) \\
\text{s.t.} \quad & f_i(x) \leq 0, \quad i = 1, \ldots, m \\
& Ax = b
\end{aligned}$$

其中 $f_0, f_1, \ldots, f_m$ 是凸函数

## 重要性质

- **局部最优 = 全局最优**
- 可行域是凸集
- 凸优化问题有高效算法

---

# 常见凸优化问题

## 1. 线性规划 (LP)

$$\min_x \quad c^Tx \quad \text{s.t.} \quad Ax \leq b$$

## 2. 二次规划 (QP)

$$\min_x \quad \frac{1}{2}x^TPx + q^Tx \quad \text{s.t.} \quad Ax \leq b, \quad P \succeq 0$$

## 3. 半定规划 (SDP)

$$\min_X \quad \text{tr}(CX) \quad \text{s.t.} \quad \text{tr}(A_iX) = b_i, \quad X \succeq 0$$

---

# 四、KKT条件

## 问题形式

$$\begin{aligned}
\min_x \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}$$

## Lagrangian函数

$$L(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$

---

# KKT条件详解

## 完整KKT条件

**必要条件**（凸问题时也是充分条件）：

1. **原问题可行性**：$g_i(x^*) \leq 0$, $h_j(x^*) = 0$

2. **对偶可行性**：$\lambda_i^* \geq 0$

3. **互补松弛**：$\lambda_i^* g_i(x^*) = 0$

4. **平稳性**：$\nabla_x L(x^*, \lambda^*, \nu^*) = 0$

$$\nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$$

---

# KKT条件示例

## 例子：带约束二次优化

$$\min_x \frac{1}{2}x^2 \quad \text{s.t.} \quad x \geq 1$$

**Lagrangian**：$L = \frac{1}{2}x^2 + \lambda(1-x)$

**KKT条件**：
- 平稳性：$x - \lambda = 0$
- 原可行性：$1 - x \leq 0$
- 对偶可行性：$\lambda \geq 0$
- 互补松弛：$\lambda(1-x) = 0$

**解**：$x^* = 1$, $\lambda^* = 1$

---

# 五、对偶理论

## Lagrange对偶函数

$$g(\lambda, \nu) = \inf_x L(x, \lambda, \nu)$$

## 对偶问题

$$\max_{\lambda \geq 0, \nu} g(\lambda, \nu)$$

## 弱对偶性

对任意可行点 $(x, \lambda, \nu)$：

$$g(\lambda, \nu) \leq f(x)$$

即**对偶值 ≤ 原值**

---

# 强对偶性

## Slater条件

若存在严格可行点 $x$ 使得：
- $g_i(x) < 0$（严格不等式）
- $Ax = b$

则强对偶性成立：$p^* = d^*$

## 对偶间隙

$$p^* - d^* \geq 0$$

- 凸问题满足Slater条件时，对偶间隙为零
- 非凸问题通常有正的对偶间隙

---

# 对偶的应用：LASSO

## 原问题

$$\min_x \frac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1$$

## 对偶形式

$$\max_\nu \quad -\frac{1}{2}\|\nu\|_2^2 + b^T\nu \quad \text{s.t.} \quad \|A^T\nu\|_\infty \leq \lambda$$

## 优势

- 对偶问题光滑且约束简单
- 可用于验证最优性
- 提供收敛界

---

# 实例分析：支持向量机

## 原问题

$$\min_{w,b} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i$$
$$\text{s.t.} \quad y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

## 对偶问题

$$\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i\alpha_j y_i y_j x_i^T x_j$$
$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

---

# Python实现：CVXPY

```python
import cvxpy as cp
import numpy as np

# LASSO问题
n, m = 100, 20
A = np.random.randn(m, n)
b = np.random.randn(m)
lam = 0.1

x = cp.Variable(n)
objective = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + 
                        lam * cp.norm(x, 1))
prob = cp.Problem(objective)
prob.solve()

print(f"最优值: {prob.value}")
print(f"非零元素: {np.sum(np.abs(x.value) > 1e-4)}")
```

---

# 本讲总结

## 核心要点

1. **凸集**：任意两点连线在集合内
2. **凸函数**：满足Jensen不等式，二阶导数非负
3. **凸优化**：局部最优即全局最优
4. **KKT条件**：凸问题的充要最优性条件
5. **对偶理论**：提供下界和验证手段

---

# 课后作业

1. 证明半正定锥 $\mathbb{S}_+^n$ 是凸集

2. 判断函数 $f(x) = \log(1 + e^x)$ 的凸性

3. 用KKT条件求解：$\min_x \frac{1}{2}\|x\|^2$ s.t. $Ax = b$

4. 对比原始和对偶SVM的计算复杂度

5. 编程实现ADMM求解LASSO问题

---

# 扩展阅读

1. **教材**：
   - Boyd & Vandenberghe, *Convex Optimization*, Cambridge, 2004
   - Nesterov, *Introductory Lectures on Convex Optimization*, Springer, 2004

2. **软件**：
   - CVXPY: https://www.cvxpy.org/
   - CVX: http://cvxr.com/cvx/

3. **课程**：
   - Stanford EE364a: Convex Optimization I

---

# 参考文献

1. Boyd, S., & Vandenberghe, L. *Convex Optimization*. Cambridge University Press, 2004.

2. Rockafellar, R. T. *Convex Analysis*. Princeton University Press, 1970.

3. Nesterov, Y. *Introductory Lectures on Convex Optimization: A Basic Course*. Springer, 2004.

4. Bertsekas, D. P. *Convex Optimization Theory*. Athena Scientific, 2009.

5. Hiriart-Urruty, J. B., & Lemaréchal, C. *Fundamentals of Convex Analysis*. Springer, 2001.
