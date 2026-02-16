# 第一讲: 变分法基础

## 课程信息
- 课程: 计算机视觉前沿方法
- 讲次: 1/24
- 学时: 2学时
- 主讲教师: [教师姓名]

## 教学目标
1. 理解泛函的基本概念与性质
2. 掌握变分的计算方法
3. 能够推导Euler-Lagrange方程
4. 理解变分法在图像处理中的应用

## 内容大纲

### Part 1: 泛函基础 (25分钟)

#### 1.1 什么是泛函？

**定义**: 设 $V$ 是一个函数空间，泛函 $J: V \to \mathbb{R}$ 将函数映射到实数。

泛函可以理解为"函数的函数"，它以函数作为输入，输出一个标量值。

**数学表述**:
$$J[u] = \int_\Omega F(x, u(x), \nabla u(x)) dx$$

其中：
- $u(x)$ 是待求函数
- $F$ 是被积函数（泛函密度）
- $\Omega$ 是定义域

#### 1.2 经典泛函例子

**例1: 弧长泛函**
$$J[y] = \int_a^b \sqrt{1 + \left(\frac{dy}{dx}\right)^2} dx$$

曲线 $y(x)$ 从点 $(a, y(a))$ 到点 $(b, y(b))$ 的弧长。

**例2: 能量泛函**
$$J[u] = \int_\Omega |\nabla u|^2 dx = \int_\Omega \left[\left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial u}{\partial y}\right)^2\right] dx$$

Dirichlet能量，衡量函数的平滑程度。

**例3: 面积泛函**
$$J[u] = \int_\Omega \sqrt{1 + |\nabla u|^2} dx$$

曲面 $z = u(x,y)$ 的表面积。

#### 1.3 图像处理中的泛函

**图像去噪能量泛函**:
$$J[u] = \underbrace{\int_\Omega (u - f)^2 dx}_{\text{数据保真项}} + \lambda \underbrace{\int_\Omega |\nabla u|^2 dx}_{\text{正则化项}}$$

其中：
- $f$ 是观测到的噪声图像
- $u$ 是待恢复的清晰图像
- $\lambda > 0$ 是正则化参数

**图像分割能量泛函** (Mumford-Shah):
$$J[u, C] = \int_\Omega (u - f)^2 dx + \mu \int_{\Omega \setminus C} |\nabla u|^2 dx + \nu |C|$$

### Part 2: 变分原理 (30分钟)

#### 2.1 函数的变分

**定义**: 函数 $u(x)$ 的变分 $\delta u$ 是函数的一个微小改变量。

设 $u_\epsilon(x) = u(x) + \epsilon \eta(x)$，其中：
- $\epsilon$ 是小参数
- $\eta(x)$ 是任意检验函数（通常要求在边界为零）

则变分为：
$$\delta u = \epsilon \eta(x)$$

#### 2.2 泛函的一阶变分

**定义**: 泛函 $J[u]$ 的一阶变分为：
$$\delta J = \left.\frac{d}{d\epsilon} J[u + \epsilon \eta]\right|_{\epsilon=0}$$

**计算步骤**:
1. 令 $u_\epsilon = u + \epsilon \eta$
2. 将 $J[u_\epsilon]$ 对 $\epsilon$ 求导
3. 令 $\epsilon = 0$

**例子**: 计算 $J[u] = \int_a^b F(x, u, u') dx$ 的变分

$$\delta J = \int_a^b \left(\frac{\partial F}{\partial u} \eta + \frac{\partial F}{\partial u'} \eta'\right) dx$$

利用分部积分：
$$\delta J = \int_a^b \left[\frac{\partial F}{\partial u} - \frac{d}{dx}\frac{\partial F}{\partial u'}\right] \eta dx + \left.\frac{\partial F}{\partial u'} \eta\right|_a^b$$

#### 2.3 极值条件

**定理**: 泛函 $J[u]$ 在 $u^*$ 处取得极值的必要条件是：
$$\delta J[u^*] = 0$$

**证明思路**:
设 $J[u^*]$ 是极小值，则对任意 $\eta$：
$$J[u^* + \epsilon \eta] \geq J[u^*]$$

定义 $\phi(\epsilon) = J[u^* + \epsilon \eta]$，则 $\phi(0)$ 是极小值点。

由费马定理：$\phi'(0) = 0$，即 $\delta J = 0$。

### Part 3: Euler-Lagrange方程 (35分钟)

#### 3.1 标量情形的推导

**问题**: 求 $J[y] = \int_a^b F(x, y, y') dx$ 的极值曲线。

**推导过程**:

由一阶变分为零：
$$\delta J = \int_a^b \left[\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'}\right] \eta dx = 0$$

由于 $\eta$ 任意，根据**变分法基本引理**：

**引理**: 若 $\int_a^b f(x) \eta(x) dx = 0$ 对所有满足 $\eta(a) = \eta(b) = 0$ 的连续函数 $\eta$ 成立，则 $f(x) \equiv 0$。

因此得到 **Euler-Lagrange方程**:
$$\boxed{\frac{\partial F}{\partial y} - \frac{d}{dx}\frac{\partial F}{\partial y'} = 0}$$

#### 3.2 展开形式

将全导数展开：
$$\frac{d}{dx}\frac{\partial F}{\partial y'} = \frac{\partial^2 F}{\partial y' \partial x} + \frac{\partial^2 F}{\partial y' \partial y} y' + \frac{\partial^2 F}{\partial y'^2} y''$$

完整形式：
$$F_y - F_{xy'} - F_{yy'} y' - F_{y'y'} y'' = 0$$

这是一个二阶常微分方程。

#### 3.3 多元函数情形

对于 $J[u] = \int_\Omega F(x_1, ..., x_n, u, u_{x_1}, ..., u_{x_n}) dx$

**Euler-Lagrange方程**:
$$\frac{\partial F}{\partial u} - \sum_{i=1}^{n} \frac{\partial}{\partial x_i}\frac{\partial F}{\partial u_{x_i}} = 0$$

或用梯度算子表示：
$$\frac{\partial F}{\partial u} - \nabla \cdot \frac{\partial F}{\partial \nabla u} = 0$$

#### 3.4 向量值函数情形

对于 $\mathbf{u} = (u_1, ..., u_m)^T$：
$$J[\mathbf{u}] = \int_\Omega F(x, \mathbf{u}, \nabla \mathbf{u}) dx$$

对每个分量有：
$$\frac{\partial F}{\partial u_i} - \nabla \cdot \frac{\partial F}{\partial \nabla u_i} = 0, \quad i = 1, ..., m$$

### Part 4: 应用实例 (20分钟)

#### 4.1 最速下降线问题

**问题**: 在重力作用下，质点从点 $A$ 沿曲线滑到点 $B$，求最速下降曲线。

**建模**:
下降时间：
$$T[y] = \int_A^B \frac{ds}{v} = \int_0^{x_1} \frac{\sqrt{1 + y'^2}}{\sqrt{2gy}} dx$$

设 $F = \frac{\sqrt{1 + y'^2}}{\sqrt{y}}$

**求解E-L方程**:

由于 $F$ 不显含 $x$，有首次积分：
$$F - y' \frac{\partial F}{\partial y'} = C$$

计算得：
$$\frac{1}{\sqrt{y(1+y'^2)}} = C$$

解为**摆线（旋轮线）**：
$$\begin{cases}
x = a(\theta - \sin\theta) \\
y = a(1 - \cos\theta)
\end{cases}$$

#### 4.2 图像去噪能量泛函

**Tikhonov正则化**:
$$J[u] = \int_\Omega (u - f)^2 dx + \lambda \int_\Omega |\nabla u|^2 dx$$

**推导E-L方程**:

$F = (u-f)^2 + \lambda |\nabla u|^2 = (u-f)^2 + \lambda (u_x^2 + u_y^2)$

$$\frac{\partial F}{\partial u} = 2(u-f)$$
$$\frac{\partial F}{\partial u_x} = 2\lambda u_x, \quad \frac{\partial F}{\partial u_y} = 2\lambda u_y$$

E-L方程：
$$2(u-f) - 2\lambda \Delta u = 0$$

即：
$$\boxed{u - \lambda \Delta u = f}$$

这是**Helmholtz方程**，可用数值方法求解。

#### 4.3 测地线问题

**问题**: 求曲面上两点间的最短路径。

**曲面上的弧长**:
$$J[\gamma] = \int_a^b \sqrt{g_{ij} \dot{\gamma}^i \dot{\gamma}^j} dt$$

**测地线方程**:
$$\ddot{\gamma}^k + \Gamma^k_{ij} \dot{\gamma}^i \dot{\gamma}^j = 0$$

其中 $\Gamma^k_{ij}$ 是Christoffel符号。

## 核心公式汇总

| 公式名称 | 表达式 | 说明 |
|---------|--------|------|
| 泛函定义 | $J[u] = \int_\Omega F(x, u, \nabla u) dx$ | 基本形式 |
| 一阶变分 | $\delta J = \int_\Omega \left(\frac{\partial F}{\partial u} \delta u + \frac{\partial F}{\partial \nabla u} \cdot \nabla \delta u\right) dx$ | 极值必要条件 |
| E-L方程(1D) | $\frac{\partial F}{\partial u} - \frac{d}{dx}\frac{\partial F}{\partial u'} = 0$ | 一维情形 |
| E-L方程(nD) | $\frac{\partial F}{\partial u} - \nabla \cdot \frac{\partial F}{\partial \nabla u} = 0$ | 多维情形 |
| 边界条件 | $\left.\frac{\partial F}{\partial u'} \delta u\right|_a^b = 0$ | 自然边界条件 |

## 课后习题

### 1. 证明题
证明：若 $F$ 不显含 $x$，即 $F = F(y, y')$，则E-L方程有首次积分：
$$F - y' \frac{\partial F}{\partial y'} = C$$

### 2. 计算题
求泛函 $J[y] = \int_0^1 (y'^2 + 2xy) dx$ 满足 $y(0) = 0, y(1) = 1$ 的极值曲线。

### 3. 推导题
推导泛函 $J[u] = \int_\Omega \left[\frac{1}{2}|\nabla u|^2 - fu\right] dx$ 的Euler-Lagrange方程，并说明其物理意义。

### 4. 编程题
用Python实现基于E-L方程的图像去噪算法：
```python
# 提示：求解 u - lambda * Delta(u) = f
# 使用Jacobi或Gauss-Seidel迭代
```

## 扩展阅读

1. **经典教材**
   - Gelfand, I.M. & Fomin, S.V. "Calculus of Variations", Dover, 2000
   - Weinstock, R. "Calculus of Variations", Dover, 1974

2. **图像处理应用**
   - Aubert, G. & Kornprobst, P. "Mathematical Problems in Image Processing", Springer, 2006
   - Chan, T.F. & Shen, J. "Image Processing and Analysis", SIAM, 2005

3. **数值方法**
   - Osher, S. & Fedkiw, R. "Level Set Methods and Dynamic Implicit Surfaces", Springer, 2003

4. **在线资源**
   - MIT OpenCourseWare: Calculus of Variations
   - Stanford CS194: Computational Photography

## 下节预告
下一讲将介绍图像复原基础，包括图像退化模型、噪声类型以及经典的图像复原方法。
