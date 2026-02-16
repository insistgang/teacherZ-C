---
marp: true
theme: default
paginate: true
---

# 计算机视觉前沿方法

## 第一讲：变分法基础

**泛函、Euler-Lagrange方程、变分问题**

---

# 本讲大纲

1. 泛函的基本概念与定义
2. 变分原理与Euler-Lagrange方程
3. 经典变分问题及其求解
4. 变分法在图像处理中的应用
5. 数值求解方法

---

# 一、泛函的基本概念

## 什么是泛函？

**定义**：泛函是函数空间到实数的映射

$$J: \mathcal{F} \rightarrow \mathbb{R}$$

其中 $\mathcal{F}$ 是某个函数空间（如 $C^1[a,b]$）

## 例子

- 弧长泛函：$J[y] = \int_a^b \sqrt{1 + y'^2} \, dx$
- 能量泛函：$J[u] = \int_\Omega |\nabla u|^2 \, dx$

---

# 变分的定义

## 一阶变分

设 $J$ 是定义在函数空间上的泛函，对于函数 $u$ 和任意扰动 $v$：

$$\delta J(u; v) = \lim_{\epsilon \to 0} \frac{J[u + \epsilon v] - J[u]}{\epsilon} = \left.\frac{d}{d\epsilon} J[u + \epsilon v]\right|_{\epsilon=0}$$

## Gateaux导数

$$\delta J(u; v) = \langle J'(u), v \rangle$$

其中 $J'(u)$ 是泛函的梯度（Fréchet导数）

---

# 二、Euler-Lagrange方程

## 基本问题

求泛函 $J[y] = \int_a^b L(x, y, y') \, dx$ 的极值

## 推导

设 $y^*$ 是极值点，对任意 $\eta \in C_c^1[a,b]$：

$$\left.\frac{d}{d\epsilon} J[y^* + \epsilon \eta]\right|_{\epsilon=0} = 0$$

展开并分部积分得：

---

# Euler-Lagrange方程

## 标准形式

$$\frac{\partial L}{\partial y} - \frac{d}{dx}\frac{\partial L}{\partial y'} = 0$$

## 边界条件

- 固定边界：$y(a) = y_a$, $y(b) = y_b$
- 自然边界：$\left.\frac{\partial L}{\partial y'}\right|_{x=a} = 0$

## 多变量情形

对于 $J[u] = \int_\Omega L(x, u, \nabla u) \, dx$：

$$\frac{\partial L}{\partial u} - \nabla \cdot \frac{\partial L}{\partial \nabla u} = 0$$

---

# 三、经典变分问题

## 1. 最速降线问题

求从A点到B点的曲线，使小球沿曲线滑下时间最短

$$T[y] = \int_0^a \frac{\sqrt{1 + y'^2}}{\sqrt{2gy}} \, dx$$

**解**：摆线（Cycloid）

$$x = a(\theta - \sin\theta), \quad y = a(1 - \cos\theta)$$

---

## 2. 测地线问题

求曲面上两点间的最短路径

$$J[\gamma] = \int_a^b \sqrt{g_{ij}\dot{\gamma}^i\dot{\gamma}^j} \, dt$$

**测地线方程**：

$$\ddot{\gamma}^k + \Gamma^k_{ij}\dot{\gamma}^i\dot{\gamma}^j = 0$$

其中 $\Gamma^k_{ij}$ 是Christoffel符号

---

## 3. 极小曲面问题

求边界固定的面积最小曲面

$$A[u] = \int_\Omega \sqrt{1 + |\nabla u|^2} \, dx$$

**极小曲面方程**：

$$(1 + u_y^2)u_{xx} - 2u_x u_y u_{xy} + (1 + u_x^2)u_{yy} = 0$$

或简写为：

$$\nabla \cdot \frac{\nabla u}{\sqrt{1 + |\nabla u|^2}} = 0$$

---

# 四、变分法在图像处理中的应用

## 图像去噪的能量泛函

给定含噪图像 $f$，求去噪图像 $u$：

$$J[u] = \frac{1}{2}\int_\Omega |u - f|^2 \, dx + \lambda \int_\Omega |\nabla u|^2 \, dx$$

- **数据保真项**：$\int_\Omega |u - f|^2 \, dx$
- **正则化项**：$\int_\Omega |\nabla u|^2 \, dx$

## Euler-Lagrange方程

$$u - f - \lambda \Delta u = 0$$

即 $(I - \lambda\Delta)u = f$（Helmholtz方程）

---

## Perona-Malik扩散

**各向异性扩散**：

$$\frac{\partial u}{\partial t} = \nabla \cdot (g(|\nabla u|)\nabla u)$$

其中扩散系数：

$$g(|\nabla u|) = \frac{1}{1 + (|\nabla u|/\kappa)^2}$$

**特点**：
- 边缘处（$|\nabla u|$大）：扩散弱，保持边缘
- 平坦区域（$|\nabla u|$小）：扩散强，去除噪声

---

# 五、数值求解方法

## 1. 梯度下降法

$$\frac{\partial u}{\partial t} = -\frac{\delta J}{\delta u}$$

离散化：

$$u^{n+1} = u^n - \tau \frac{\delta J}{\delta u}(u^n)$$

## 2. 有限差分

梯度 $\nabla u$ 的离散化：

- 前向差分：$(D_+ u)_{i,j} = u_{i+1,j} - u_{i,j}$
- 后向差分：$(D_- u)_{i,j} = u_{i,j} - u_{i-1,j}$
- 中心差分：$(D_0 u)_{i,j} = \frac{u_{i+1,j} - u_{i-1,j}}{2}$

---

## 3. Python实现示例

```python
import numpy as np

def laplacian(u, h=1.0):
    """离散Laplacian算子"""
    return (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 
            4 * u) / h**2

def gradient_descent_denoise(f, lambda_reg, tau, max_iter=1000):
    """梯度下降去噪"""
    u = f.copy()
    for _ in range(max_iter):
        # Euler-Lagrange: u - f - lambda*Delta(u) = 0
        grad_J = u - f - lambda_reg * laplacian(u)
        u = u - tau * grad_J
    return u
```

---

# 实例分析：图像去噪

## 实验设置

```python
import matplotlib.pyplot as plt
from skimage import data, util

# 加载测试图像
image = data.camera()
noisy = util.random_noise(image, mode='gaussian', var=0.01)

# 变分去噪
denoised = gradient_descent_denoise(noisy, lambda_reg=0.1, 
                                     tau=0.01, max_iter=500)
```

## 结果分析

- PSNR提升：约2-3 dB
- 边缘保持良好
- 平滑区域噪声有效抑制

---

# 本讲总结

## 核心要点

1. **泛函**是函数空间到实数的映射，变分是泛函的"导数"
2. **Euler-Lagrange方程**是泛函极值的必要条件
3. 图像处理中的变分方法通过能量泛函建模
4. **梯度下降**是求解变分问题的常用数值方法
5. 变分法提供了图像处理的数学理论基础

---

# 课后作业

1. 证明弧长泛函 $J[y] = \int_a^b \sqrt{1 + y'^2} dx$ 的Euler-Lagrange方程的解是直线

2. 对于泛函 $J[u] = \int_\Omega \left(\frac{1}{2}|\nabla u|^2 - fu\right) dx$，推导其Euler-Lagrange方程

3. 编程实现Perona-Malik扩散，比较不同参数 $\kappa$ 的效果

4. 将Tikhonov正则化推广到向量值图像（彩色图像）

---

# 扩展阅读

1. **教材**：
   - Gelfand & Fomin, *Calculus of Variations*, Dover, 2000
   - Aubert & Kornprobst, *Mathematical Problems in Image Processing*, Springer, 2006

2. **论文**：
   - Rudin, Osher & Fatemi, "Nonlinear total variation based noise removal algorithms", Physica D, 1992
   - Perona & Malik, "Scale-space and edge detection using anisotropic diffusion", PAMI, 1990

---

# 参考文献

1. Weinstock, R. *Calculus of Variations*. Dover Publications, 1974.

2. Sapiro, G. *Geometric Partial Differential Equations and Image Analysis*. Cambridge, 2001.

3. Chan, T. F., & Shen, J. *Image Processing and Analysis: Variational, PDE, Wavelet, and Stochastic Methods*. SIAM, 2005.

4. Osher, S., & Paragios, N. *Geometric Level Set Methods in Imaging, Vision, and Graphics*. Springer, 2003.

5. Evans, L. C. *Partial Differential Equations*. AMS, 2010.
