# 第四讲: 正则化方法

## 课程信息
- 课程: 计算机视觉前沿方法
- 讲次: 4/24
- 学时: 2学时

## 教学目标
1. 理解正则化的数学基础
2. 掌握Tikhonov正则化方法
3. 了解稀疏正则化及其优化算法
4. 理解正则化参数的选择策略

## 内容大纲

### Part 1: 正则化理论基础 (25分钟)

#### 1.1 逆问题与不适定性

**正问题**: 给定原因，求结果
$$y = Ax$$

**逆问题**: 给定结果，求原因
$$x = A^{-1}y$$

**不适定性示例**: 图像去模糊

**Hadamard适定性**:
1. 存在性：解必须存在
2. 唯一性：解必须唯一
3. 稳定性：解必须连续依赖于数据

#### 1.2 正则化的必要性

**不适定问题的例子**:

求积分方程：$\int_0^1 K(s,t)x(t)dt = y(s)$

若 $y_\epsilon(s) = y(s) + \epsilon \sin(ns)$，则：
$$x_\epsilon(t) \approx x(t) + n\epsilon \cdot (\text{某种放大})$$

高频噪声被剧烈放大！

**正则化思想**: 添加先验约束，稳定化逆问题

$$\min_x \|Ax - y\|^2 + \lambda R(x)$$

#### 1.3 变分正则化框架

**一般形式**:
$$\hat{x} = \arg\min_x \left\{ D(Ax, y) + \lambda R(x) \right\}$$

其中：
- $D(\cdot,\cdot)$: 数据保真项
- $R(\cdot)$: 正则化项（惩罚项）
- $\lambda > 0$: 正则化参数

**贝叶斯解释**:
$$p(x|y) \propto p(y|x) \cdot p(x)$$

- $D(Ax,y) \leftrightarrow -\log p(y|x)$ (似然)
- $R(x) \leftrightarrow -\log p(x)$ (先验)

### Part 2: Tikhonov正则化 (30分钟)

#### 2.1 经典Tikhonov正则化

**模型**:
$$\min_x \left\{ \|Ax - y\|_2^2 + \lambda \|x\|_2^2 \right\}$$

**解析解**:
$$\hat{x} = (A^TA + \lambda I)^{-1} A^T y$$

**SVD分析**:

设 $A = U\Sigma V^T$，则：
$$\hat{x} = V \cdot \text{diag}\left(\frac{\sigma_i}{\sigma_i^2 + \lambda}\right) \cdot U^T y$$

**滤波因子**:
$$f_i = \frac{\sigma_i^2}{\sigma_i^2 + \lambda}$$

当 $\sigma_i \ll \sqrt{\lambda}$ 时，$f_i \to 0$，小奇异值被抑制。

#### 2.2 广义Tikhonov正则化

**模型**:
$$\min_x \left\{ \|Ax - y\|_2^2 + \lambda \|Lx\|_2^2 \right\}$$

其中 $L$ 是正则化矩阵。

**常见选择**:
- $L = I$: 标准正则化
- $L = \nabla$: 一阶差分（平滑性）
- $L = \Delta$: 二阶差分（更高平滑性）

**解析解**:
$$\hat{x} = (A^TA + \lambda L^TL)^{-1} A^T y$$

#### 2.3 迭代正则化

**Landweber迭代**:
$$x_{k+1} = x_k + \tau A^T(y - Ax_k)$$

其中 $\tau < 2/\sigma_{\max}^2(A)$ 是步长。

**半迭代方法** (CG, GMRES):
收敛更快，但需要早停以实现正则化效果。

**Landweber的滤波解释**:
$$x_k = V \cdot \text{diag}\left(1 - (1 - \tau\sigma_i^2)^k\right) \cdot \frac{1}{\sigma_i} U^T y$$

迭代次数 $k$ 起到正则化参数的作用。

#### 2.4 截断SVD

**方法**:
$$\hat{x}_r = \sum_{i=1}^{r} \frac{(u_i^T y)}{\sigma_i} v_i$$

只保留前 $r$ 个最大的奇异值。

**截断数 $r$ 的作用**: 相当于正则化参数

### Part 3: 稀疏正则化 (30分钟)

#### 3.1 $L^0$与$L^1$正则化

**$L^0$问题** (NP-hard):
$$\min_x \|Ax - y\|_2^2 + \lambda \|x\|_0$$

**$L^1$松弛** (凸优化):
$$\min_x \|Ax - y\|_2^2 + \lambda \|x\|_1$$

**为什么$L^1$促进稀疏性**:

几何解释：$L^1$球是有棱角的，更容易与等高线在最优点（坐标轴上）相切。

#### 3.2 LASSO与Basis Pursuit

**LASSO** (Least Absolute Shrinkage and Selection Operator):
$$\min_x \|Ax - y\|_2^2 \quad \text{s.t.} \quad \|x\|_1 \leq t$$

**Basis Pursuit Denoising**:
$$\min_x \|x\|_1 \quad \text{s.t.} \quad \|Ax - y\|_2 \leq \epsilon$$

**等价性**: 在一定条件下，LASSO和BPDN等价于无约束$L^1$正则化。

#### 3.3 优化算法

**(1) 软阈值 (Soft Thresholding)**

对于 $\min_x \frac{1}{2}\|x - y\|_2^2 + \lambda\|x\|_1$:

**解**:
$$\hat{x}_i = S_\lambda(y_i) = \text{sign}(y_i)\max(|y_i| - \lambda, 0)$$

**(2) ISTA (Iterative Shrinkage-Thresholding Algorithm)**

$$x_{k+1} = S_{\lambda/L}(x_k - \frac{1}{L}A^T(Ax_k - y))$$

其中 $L \geq \lambda_{\max}(A^TA)$。

**(3) FISTA (Fast ISTA)**

引入动量项，收敛速度从 $O(1/k)$ 提升到 $O(1/k^2)$。

$$\begin{aligned}
x_{k+1} &= S_{\lambda/L}(y_k - \frac{1}{L}A^T(Ay_k - y)) \\
t_{k+1} &= \frac{1 + \sqrt{1 + 4t_k^2}}{2} \\
y_{k+1} &= x_{k+1} + \frac{t_k - 1}{t_{k+1}}(x_{k+1} - x_k)
\end{aligned}$$

**(4) ADMM (Alternating Direction Method of Multipliers)**

$$\begin{aligned}
x_{k+1} &= (A^TA + \rho I)^{-1}(A^Ty + \rho(z_k - u_k)) \\
z_{k+1} &= S_{\lambda/\rho}(x_{k+1} + u_k) \\
u_{k+1} &= u_k + x_{k+1} - z_{k+1}
\end{aligned}$$

#### 3.4 压缩感知理论

**RIP条件** (Restricted Isometry Property):

矩阵 $A$ 满足 $(k, \delta)$-RIP，若对所有 $k$-稀疏向量 $x$：
$$(1-\delta)\|x\|_2^2 \leq \|Ax\|_2^2 \leq (1+\delta)\|x\|_2^2$$

**定理**: 若 $A$ 满足 $(2k, \delta < \sqrt{2}-1)$-RIP，则$L^1$最小化精确恢复$k$-稀疏信号。

### Part 4: 正则化参数选择 (15分钟)

#### 4.1 偏差-方差权衡

**均方误差分解**:
$$E[\|\hat{x} - x_{\text{true}}\|^2] = \text{Bias}^2 + \text{Variance}$$

- $\lambda \to 0$: 方差大（过拟合）
- $\lambda \to \infty$: 偏差大（欠拟合）

**最优 $\lambda$**: 平衡偏差与方差

#### 4.2 交叉验证

**K折交叉验证**:
1. 数据分为 $K$ 份
2. 轮流用 $K-1$ 份训练，1份验证
3. 选择使验证误差最小的 $\lambda$

**留一交叉验证 (LOOCV)**: $K = n$

#### 4.3 广义交叉验证 (GCV)

$$GCV(\lambda) = \frac{\|A\hat{x}_\lambda - y\|^2}{(n - \text{trace}(A(A^TA + \lambda I)^{-1}A^T))^2}$$

**优点**: 不需要实际分割数据

#### 4.4 L曲线法

**L曲线**: $\log\|A\hat{x}_\lambda - y\|$ vs $\log\|L\hat{x}_\lambda\|$

**最优 $\lambda$**: L曲线的拐角处

**原理**: 拐角代表数据拟合与正则化的最佳平衡

#### 4.5 Morozov偏差原理

选择 $\lambda$ 使得：
$$\|A\hat{x}_\lambda - y\| = \delta$$

其中 $\delta$ 是噪声水平的估计。

## 核心公式汇总

| 方法 | 形式 | 解/算法 |
|------|------|---------|
| Tikhonov | $\|Ax-y\|^2 + \lambda\|x\|^2$ | $(A^TA+\lambda I)^{-1}A^Ty$ |
| Landweber | 迭代 | $x_{k+1} = x_k + \tau A^T(y-Ax_k)$ |
| LASSO | $\|Ax-y\|^2 + \lambda\|x\|_1$ | ISTA/FISTA |
| 软阈值 | - | $S_\lambda(x) = \text{sign}(x)\max(|x|-\lambda,0)$ |

## 课后习题

### 1. 证明题
证明Tikhonov正则化解 $(A^TA + \lambda I)^{-1}A^Ty$ 当 $\lambda \to 0$ 时收敛到最小范数解。

### 2. 计算题
给定 $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$，$y = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$，计算 $\lambda = 0.1$ 时的Tikhonov正则化解。

### 3. 推导题
推导软阈值算子的表达式：证明 $\arg\min_x \frac{1}{2}(x-y)^2 + \lambda|x| = S_\lambda(y)$。

### 4. 编程题
```python
import numpy as np

def ista(A, y, lambda_, max_iter=1000):
    """
    ISTA算法实现
    """
    # 计算Lipschitz常数
    L = np.linalg.norm(A, 2)**2
    x = np.zeros(A.shape[1])
    
    for _ in range(max_iter):
        # 梯度步
        grad = A.T @ (A @ x - y)
        # 软阈值
        # 请完成此函数
        pass
    return x
```

## 扩展阅读

1. **教材**
   - Engl, Hanke & Neubauer "Regularization of Inverse Problems", 1996
   - Hastie, Tibshirani & Friedman "Elements of Statistical Learning"

2. **稀疏优化**
   - Beck & Teboulle "A Fast Iterative Shrinkage-Thresholding Algorithm", SIAM J. Imaging Sci., 2009
   - Boyd et al. "Distributed Optimization and Statistical Learning via ADMM", 2011

3. **压缩感知**
   - Candès & Wakin "An Introduction To Compressive Sampling", IEEE Signal Processing Magazine, 2008

## 下节预告
下一讲将深入讲解图像去噪算法，包括各种经典和现代去噪方法的实现。
