# ROF 与 Mumford-Shah 模型理论基础

## 目录
1. [变分法基础](#变分法基础)
2. [ROF 去噪模型](#rof-去噪模型)
3. [Mumford-Shah 分割模型](#mumford-shah-分割模型)
4. [Chan-Vese 模型](#chan-vese-模型)
5. [数值方法](#数值方法)

---

## 变分法基础

### 变分问题的一般形式

变分法是研究泛函极值问题的数学分支。在图像处理中，我们常常需要最小化如下形式的能量泛函：

$$E(u) = \int_\Omega F(x, u(x), \nabla u(x)) \, dx$$

其中：
- $\Omega$ 是图像定义域（通常是矩形区域）
- $u$ 是我们要求解的函数（如去噪后的图像）
- $F$ 是能量密度函数

### Euler-Lagrange 方程

根据变分法的基本定理，若 $u$ 是能量 $E$ 的极小值点，则 $u$ 满足 Euler-Lagrange 方程：

$$\frac{\partial F}{\partial u} - \nabla \cdot \frac{\partial F}{\partial \nabla u} = 0$$

这个方程是泛函取极值的必要条件。

### 梯度流（Gradient Flow）

为了数值求解 Euler-Lagrange 方程，我们通常引入**梯度流**：

$$\frac{\partial u}{\partial t} = -\frac{\delta E}{\delta u}$$

其中 $\frac{\delta E}{\delta u}$ 是能量 $E$ 关于 $u$ 的变分导数。

当 $t \to \infty$ 时，$u$ 收敛到能量的稳定点。

---

## ROF 去噪模型

### 模型提出

ROF (Rudin-Osher-Fatemi) 模型于 1992 年提出，是图像去噪领域最重要的变分模型之一。

### 数学模型

给定含噪图像 $f$，ROF 模型寻找去噪图像 $u$ 最小化如下能量：

$$\min_u \left\{ E(u) = \int_\Omega |\nabla u| \, dx + \frac{\lambda}{2} \int_\Omega (u - f)^2 \, dx \right\}$$

### 两项的解释

1. **总变分项** $TV(u) = \int_\Omega |\nabla u| \, dx$
   - 惩罚图像的变化
   - 保持边缘（不像拉普拉斯平滑那样模糊边缘）
   - 允许不连续解

2. **数据保真项** $\frac{\lambda}{2} \int_\Omega (u - f)^2 \, dx$
   - 保持去噪图像接近原始图像
   - 参数 $\lambda$ 控制平滑强度

### Euler-Lagrange 方程

$$- \nabla \cdot \left(\frac{\nabla u}{|\nabla u|}\right) + \lambda(u - f) = 0$$

其中 $\frac{\nabla u}{|\nabla u|}$ 称为**归一化梯度**或**边缘指示**。

### 数值特性

- 方程在 $|\nabla u| = 0$ 处是奇异的
- 需要使用正则化：$|\nabla u|_\epsilon = \sqrt{|\nabla u|^2 + \epsilon^2}$
- 解是唯一的（严格凸问题）

---

## Mumford-Shah 分割模型

### 模型提出

Mumford-Shah 模型于 1989 年提出，将图像分割问题转化为能量最小化问题。

### 数学模型

$$E(u, K) = \int_{\Omega \setminus K} (u - f)^2 \, dx + \mu \int_{\Omega \setminus K} |\nabla u|^2 \, dx + \nu |K|$$

其中：
- $u$：分段光滑的逼近图像
- $K$：边缘集合（不连续点集）
- $|K|$：边缘长度（Hausdorff 测度）
- $\mu, \nu$：正则化参数

### 三项的解释

1. **数据保真项** $\int (u-f)^2$
   - 逼近图像 $u$ 应该接近原始图像 $f$

2. **平滑项** $\mu \int |\nabla u|^2$
   - 在每个区域内 $u$ 应该是光滑的
   - 但在边缘 $K$ 上不惩罚（允许跳跃）

3. **边缘惩罚项** $\nu |K|$
   - 限制边缘的长度
   - 防止过分割
   - 参数 $\nu$ 控制分割复杂度

### 自由不连续问题

Mumford-Shah 模型是一个**自由不连续问题**（free discontinuity problem）：

- 边缘集合 $K$ 不是预先给定的
- 需要同时优化 $u$ 和 $K$
- 理论分析和数值求解都非常困难

### 解的存在性

Mumford-Shah 问题在适当条件下存在解，但：

- 解可能不唯一
- 边缘集合 $K$ 可能有复杂的拓扑结构
- 全局优化是 NP 难问题

---

## Chan-Vese 模型

### 模型动机

Chan-Vese 模型是 Mumford-Shah 模型的一个重要特例，假设图像可以被分成两个近似常数的区域。

### 数学模型

对于闭合轮廓 $C$ 将区域 $\Omega$ 分成内部 $\Omega_1$ 和外部 $\Omega_2$：

$$E(c_1, c_2, C) = \mu \cdot \text{Length}(C) + \nu \cdot \text{Area}(\Omega_1)$$
$$+ \lambda_1 \int_{\Omega_1} |f - c_1|^2 \, dx + \lambda_2 \int_{\Omega_2} |f - c_2|^2 \, dx$$

其中 $c_1, c_2$ 是区域内外的平均强度：

$$c_1 = \frac{1}{|\Omega_1|} \int_{\Omega_1} f \, dx, \quad c_2 = \frac{1}{|\Omega_2|} \int_{\Omega_2} f \, dx$$

### 水平集表述

使用水平集函数 $\phi$ 隐式表示轮廓：

- $C = \{x \in \Omega : \phi(x) = 0\}$
- $\Omega_1 = \{x : \phi(x) > 0\}$（内部）
- $\Omega_2 = \{x : \phi(x) < 0\}$（外部）

能量函数变为：

$$E(c_1, c_2, \phi) = \mu \int_\Omega \delta(\phi)|\nabla\phi| \, dx + \nu \int_\Omega H(\phi) \, dx$$
$$+ \lambda_1 \int_\Omega H(\phi)|f - c_1|^2 \, dx + \lambda_2 \int_\Omega (1-H(\phi))|f - c_2|^2 \, dx$$

其中：
- $H(z)$ 是 Heaviside 函数
- $\delta(z) = H'(z)$ 是 Dirac delta 函数

### 演化方程

最小化能量得到水平集演化方程：

$$\frac{\partial \phi}{\partial t} = \delta(\phi) \left[ \mu \cdot \text{div}\left(\frac{\nabla\phi}{|\nabla\phi|}\right) - \nu - \lambda_1(f - c_1)^2 + \lambda_2(f - c_2)^2 \right]$$

### 各项的作用

1. **曲率项** $\mu \cdot \text{div}(\frac{\nabla\phi}{|\nabla\phi|})$：平滑轮廓
2. **面积项** $-\nu$：控制区域大小（可选）
3. **数据项** $-\lambda_1(f-c_1)^2 + \lambda_2(f-c_2)^2$：驱动轮廓向边界移动

### 优点与局限

**优点：**
- 不依赖图像梯度，适用于弱边界
- 可以自动处理拓扑变化
- 对噪声相对鲁棒

**局限：**
- 假设图像是分段常数的
- 对强度不均匀的图像效果不佳
- 参数选择需要经验

---

## 数值方法

### 梯度下降法

直接离散化 Euler-Lagrange 方程：

$$u^{n+1} = u^n - \Delta t \cdot \frac{\delta E}{\delta u}(u^n)$$

**优点**：简单直观
**缺点**：收敛慢，需要小步长

### Chambolle 投影法

利用凸对偶理论，将问题转化为对偶空间：

$$\min_u \max_{|p| \leq 1} \int u \, \text{div}(p) + \frac{\lambda}{2}(u-f)^2$$

通过交替投影高效求解。

**优点**：快速、数值稳定
**缺点**：仅适用于 TV 模型

### Split Bregman 方法

通过引入辅助变量将问题分解：

$$\min_{u,d} |d| + \frac{\lambda}{2}\|u-f\|^2 \quad \text{s.t.} \quad d = \nabla u$$

使用 Bregman 迭代交替求解子问题。

**优点**：收敛快，适用于大规模问题
**缺点**：需要求解线性系统

### 水平集方法

数值实现要点：

1. **空间离散化**：有限差分
2. **时间离散化**：显式欧拉或 Runge-Kutta
3. **重新初始化**：保持 $|\nabla\phi| \approx 1$
4. **延拓方法**：处理窄带计算

### 收敛性分析

| 方法 | 收敛速率 | 存储需求 | 适用范围 |
|------|---------|---------|---------|
| 梯度下降 | $O(1/k)$ | 低 | 一般 |
| Chambolle | $O(1/k)$ | 中 | TV 模型 |
| Split Bregman | 线性 | 高 | 一般 L1 问题 |

---

## 参考文献

1. Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D*, 60(1-4), 259-268.

2. Mumford, D., & Shah, J. (1989). Optimal approximations by piecewise smooth functions and associated variational problems. *Comm. Pure Appl. Math.*, 42(5), 577-685.

3. Chan, T. F., & Vese, L. A. (2001). Active contours without edges. *IEEE Trans. Image Processing*, 10(2), 266-277.

4. Chambolle, A. (2004). An algorithm for total variation minimization and applications. *J. Math. Imaging Vision*, 20(1-2), 89-97.

5. Goldstein, T., & Osher, S. (2009). The split Bregman method for L1-regularized problems. *SIAM J. Imaging Sci.*, 2(2), 323-343.

6. Aubert, G., & Kornprobst, P. (2006). *Mathematical Problems in Image Processing*. Springer.
