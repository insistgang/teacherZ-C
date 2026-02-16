# 第三讲: 全变分模型

## 课程信息
- 课程: 计算机视觉前沿方法
- 讲次: 3/24
- 学时: 2学时

## 教学目标
1. 理解全变分(TV)的数学定义
2. 掌握TV正则化的物理意义
3. 能够推导TV去噪模型的Euler-Lagrange方程
4. 了解TV模型的优缺点

## 内容大纲

### Part 1: 全变分的定义与性质 (25分钟)

#### 1.1 有界变差函数空间

**定义**: 函数 $u \in L^1(\Omega)$ 的全变分定义为：
$$TV(u) = \sup\left\{\int_\Omega u \, \text{div} \phi \, dx : \phi \in C_c^1(\Omega, \mathbb{R}^n), |\phi(x)| \leq 1\right\}$$

**有界变差函数空间**:
$$BV(\Omega) = \{u \in L^1(\Omega) : TV(u) < \infty\}$$

#### 1.2 光滑函数的全变分

对于光滑函数 $u \in C^1(\Omega)$：
$$TV(u) = \int_\Omega |\nabla u| dx = \int_\Omega \sqrt{u_x^2 + u_y^2} \, dx$$

**解释**:
- $|\nabla u|$ 是梯度幅值
- TV是所有梯度幅值的积分
- 度量图像"变化总量"

#### 1.3 TV的几何解释

**水平集**: 给定阈值 $t$，水平集 $L_t = \{x : u(x) = t\}$

**Co-Area公式**:
$$TV(u) = \int_{-\infty}^{+\infty} \text{Per}(L_t) \, dt$$

其中 $\text{Per}(L_t)$ 是水平集的周长。

**意义**: 全变分等于所有水平集周长的积分。

#### 1.4 TV的性质

**性质1: 旋转不变性**
TV在坐标旋转下保持不变。

**性质2: 边缘保持**
TV允许函数有跳跃（边缘），不同于 $H^1$ 范数。

**性质3: 凸性**
$TV(u)$ 是 $u$ 的凸泛函。

**性质4: 齐次性**
$TV(cu) = |c| TV(u)$，一次齐次。

### Part 2: Rudin-Osher-Fatemi (ROF) 模型 (30分钟)

#### 2.1 模型建立

**ROF模型** (1992):
$$\min_u \left\{ \int_\Omega |\nabla u| dx + \frac{\lambda}{2} \int_\Omega (u - f)^2 dx \right\}$$

其中：
- $f$: 观测（噪声）图像
- $u$: 待求（去噪）图像
- $\lambda > 0$: 正则化参数

**能量泛函**:
$$J[u] = TV(u) + \frac{\lambda}{2}\|u - f\|_{L^2}^2$$

#### 2.2 Euler-Lagrange方程推导

**变分计算**:
$$\delta J = \int_\Omega \left[ \frac{\nabla u}{|\nabla u|} \cdot \nabla \delta u + \lambda(u-f)\delta u \right] dx$$

利用分部积分（散度定理）：
$$\delta J = \int_\Omega \left[ -\text{div}\left(\frac{\nabla u}{|\nabla u|}\right) + \lambda(u-f) \right] \delta u \, dx$$

**Euler-Lagrange方程**:
$$\boxed{-\text{div}\left(\frac{\nabla u}{|\nabla u|}\right) + \lambda(u - f) = 0}$$

#### 2.3 曲率流解释

记 $\kappa = \text{div}\left(\frac{\nabla u}{|\nabla u|}\right)$，这是水平集的**平均曲率**。

**E-L方程改写**:
$$u - f = \frac{1}{\lambda}\kappa$$

**物理意义**:
- 曲率为正的区域（凸），$u < f$
- 曲率为负的区域（凹），$u > f$
- 曲率越大，偏离越明显

#### 2.4 梯度下降流

引入人工时间 $t$，梯度下降流：
$$\frac{\partial u}{\partial t} = \text{div}\left(\frac{\nabla u}{|\nabla u|}\right) - \lambda(u - f)$$

**稳态**: $\frac{\partial u}{\partial t} = 0$ 即得到E-L方程。

### Part 3: TV正则化的特点 (25分钟)

#### 3.1 边缘保持能力

**对比Tikhonov正则化**:

| 正则化 | 形式 | 边缘处理 |
|--------|------|----------|
| Tikhonov | $\|\nabla u\|_2^2$ | 过度平滑边缘 |
| TV | $\|\nabla u\|_1$ | 保持边缘 |

**原因分析**:
- $L^2$ 对大梯度惩罚重 → 边缘被平滑
- $L^1$ 对大梯度惩罚轻 → 边缘被保留

#### 3.2 阶梯效应 (Staircasing Effect)

**问题**: TV正则化在平滑区域产生虚假边缘

**原因**: TV偏好分段常数函数

**示例**: 斜坡函数经TV去噪后变成阶梯状

**解决方案**:
- 高阶TV: $TV^2(u) = \int|\nabla^2 u|$
- 广义TV: $\alpha TV(u) + \beta TV^2(u)$

#### 3.3 对比度不变性

**定理**: 设 $u^*$ 是ROF问题的解，则对任意单调递增函数 $g$，$g(u^*)$ 是以 $g(f)$ 为输入的ROF问题解。

**意义**: TV去噪结果与图像对比度线性变换无关。

#### 3.4 参数选择

**Morozov偏差原理**: 选择 $\lambda$ 使得：
$$\|u - f\|_{L^2} = \sigma\sqrt{|\Omega|}$$

其中 $\sigma$ 是噪声标准差。

**广义交叉验证(GCV)**:
$$\lambda^* = \arg\min_\lambda \frac{\|u_\lambda - f\|^2}{(\text{trace}(I - A_\lambda))^2}$$

### Part 4: TV模型的扩展 (20分钟)

#### 4.1 加权TV

**各向异性TV**:
$$TV_{\alpha,\beta}(u) = \int_\Omega \sqrt{\alpha u_x^2 + \beta u_y^2} \, dx$$

**加权TV**:
$$TV_w(u) = \int_\Omega w(x)|\nabla u| dx$$

其中 $w(x)$ 是空间权重函数。

#### 4.2 非局部TV

**非局部梯度**:
$$(\nabla_{NL} u)(x,y) = u(y) - u(x), \quad y \in \text{邻域}(x)$$

**非局部TV**:
$$TV_{NL}(u) = \int_\Omega \sqrt{\int_\Omega w(x,y)(u(y)-u(x))^2 dy} \, dx$$

**优势**: 利用图像的自相似性，更好地保持纹理。

#### 4.3 向量值TV (Color TV)

**彩色图像**: $u = (u_1, u_2, u_3)^T$

**通道耦合TV**:
$$TV(u) = \int_\Omega \sqrt{\sum_{i=1}^3 |\nabla u_i|^2} \, dx$$

**意义**: 所有通道共享边缘，避免颜色边缘错位。

#### 4.4 TV-L1模型

**模型**:
$$\min_u \left\{ TV(u) + \lambda \|u - f\|_{L^1} \right\}$$

**特点**:
- 对椒盐噪声更鲁棒
- 更好的对比度保持
- 更强的边缘保持

## 核心公式汇总

| 公式名称 | 表达式 | 说明 |
|---------|--------|------|
| TV定义 | $TV(u) = \int_\Omega \|\nabla u\| dx$ | 全变分 |
| ROF模型 | $\min_u TV(u) + \frac{\lambda}{2}\|u-f\|^2$ | 经典去噪 |
| E-L方程 | $-\text{div}(\nabla u/\|\nabla u\|) + \lambda(u-f) = 0$ | 最优性条件 |
| 曲率流 | $\partial_t u = \text{div}(\nabla u/\|\nabla u\|)$ | 梯度下降 |
| TV-L1 | $\min_u TV(u) + \lambda\|u-f\|_1$ | 鲁棒版本 |

## 课后习题

### 1. 证明题
证明ROF模型能量泛函的凸性和下半连续性。

### 2. 计算题
验证下列函数的TV值：
- $u(x) = ax + b$ on $[0,1]$
- $u(x) = \begin{cases} 0, x < c \\ 1, x \geq c \end{cases}$ on $[0,1]$

### 3. 推导题
推导TV-L1模型的Euler-Lagrange方程（提示：需要引入辅助变量）。

### 4. 编程题
```python
import numpy as np

def tv_denoise(f, lambda_, n_iter=100):
    """
    TV去噪的梯度下降实现
    f: 输入图像
    lambda_: 正则化参数
    n_iter: 迭代次数
    """
    u = f.copy()
    dt = 0.1
    
    for _ in range(n_iter):
        # 计算梯度
        # 计算曲率项 div(grad_u/|grad_u|)
        # 梯度下降更新
        pass
    return u
```

## 扩展阅读

1. **经典论文**
   - Rudin, Osher & Fatemi "Nonlinear Total Variation Based Noise Removal Algorithms", Physica D, 1992
   - Chambolle "An Algorithm for Total Variation Minimization and Applications", JMIV, 2004

2. **高阶TV**
   - Bredies, Kunisch, Pock "Total Generalized Variation", SIAM J. Imaging Sci., 2010

3. **应用**
   - Compressive Sensing中的TV最小化
   - 医学图像重建

## 下节预告
下一讲将介绍正则化方法，深入探讨各种正则化策略及其数学基础。
