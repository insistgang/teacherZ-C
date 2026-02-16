# 第二讲: 图像复原基础

## 课程信息
- 课程: 计算机视觉前沿方法
- 讲次: 2/24
- 学时: 2学时

## 教学目标
1. 理解图像退化模型
2. 掌握常见噪声类型及其统计特性
3. 了解经典图像复原方法
4. 能够建立基本的图像复原能量泛函

## 内容大纲

### Part 1: 图像退化模型 (25分钟)

#### 1.1 图像退化的物理过程

**一般退化模型**:
$$f(x,y) = h(x,y) * u(x,y) + n(x,y)$$

其中：
- $u(x,y)$: 原始图像（理想图像）
- $h(x,y)$: 点扩散函数（PSF）/ 退化函数
- $n(x,y)$: 加性噪声
- $f(x,y)$: 观测到的退化图像
- $*$: 卷积运算

**矩阵-向量形式**:
$$\mathbf{f} = \mathbf{H}\mathbf{u} + \mathbf{n}$$

#### 1.2 常见退化类型

**1. 运动模糊**
$$h(x,y) = \begin{cases}
\frac{1}{L} & \text{if } x\cos\theta + y\sin\theta = 0, |x| \leq \frac{L}{2} \\
0 & \text{otherwise}
\end{cases}$$

其中 $L$ 是运动距离，$\theta$ 是运动角度。

**2. 散焦模糊** (Out-of-focus)
$$h(x,y) = \begin{cases}
\frac{1}{\pi R^2} & \text{if } x^2 + y^2 \leq R^2 \\
0 & \text{otherwise}
\end{cases}$$

其中 $R$ 是散焦半径。

**3. 大气湍流模糊**
$$H(u,v) = \exp\left[-k(u^2 + v^2)^{5/6}\right]$$

**4. 高斯模糊**
$$h(x,y) = \frac{1}{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)$$

#### 1.3 频域分析

**卷积定理**:
$$f = h * u \quad \Leftrightarrow \quad F = H \cdot U$$

在频域中：
$$F(u,v) = H(u,v)U(u,v) + N(u,v)$$

**逆滤波的问题**:
$$\hat{U}(u,v) = \frac{F(u,v)}{H(u,v)} = U(u,v) + \frac{N(u,v)}{H(u,v)}$$

当 $H(u,v) \to 0$ 时，噪声被剧烈放大！

### Part 2: 噪声模型 (25分钟)

#### 2.1 高斯噪声

**概率密度函数**:
$$p(x) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**特性**:
- 均值: $E[X] = \mu$
- 方差: $\text{Var}[X] = \sigma^2$
- 加性噪声，与信号无关

**应用场景**: 电子电路热噪声、传感器暗电流噪声

#### 2.2 泊松噪声（光子噪声）

**概率质量函数**:
$$P(k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**特性**:
- 均值: $E[X] = \lambda$
- 方差: $\text{Var}[X] = \lambda$
- **信号相关**: 方差等于均值

**应用场景**: 低光照成像、X射线成像、荧光显微镜

**近似**: 当 $\lambda$ 较大时，泊松分布近似高斯分布：
$$\text{Poisson}(\lambda) \approx \mathcal{N}(\lambda, \lambda)$$

#### 2.3 椒盐噪声

**概率模型**:
$$p(x) = \begin{cases}
p_s & \text{if } x = s_{\max} \text{ (盐)} \\
p_n & \text{if } x = s_{\min} \text{ (椒)} \\
1 - p_s - p_n & \text{if } x = \text{原始值}
\end{cases}$$

**特性**:
- 脉冲式噪声
- 与信号无关
- 由比特错误、像素损坏引起

#### 2.4 散斑噪声（Speckle Noise）

**乘性噪声模型**:
$$f = u \cdot n$$

其中 $n$ 服从均值为1的分布。

**对数变换**:
$$\log f = \log u + \log n$$

乘性噪声转为加性噪声。

**应用场景**: 合成孔径雷达(SAR)、超声波成像

#### 2.5 噪声统计量的估计

**方法1**: 从均匀区域估计
$$\hat{\sigma}^2 = \frac{1}{N-1}\sum_{i=1}^{N}(f_i - \bar{f})^2$$

**方法2**: 基于局部方差
$$\hat{\sigma}^2 = E[\text{Var}_{\text{local}}(f)] - E[f]^2 \cdot \text{Var}_{\text{signal}}$$

### Part 3: 经典复原方法 (30分钟)

#### 3.1 逆滤波

**基本思想**: 直接对退化求逆

$$\hat{u} = \mathcal{F}^{-1}\left[\frac{F(u,v)}{H(u,v)}\right]$$

**问题**: 
- 零点和近零点导致数值不稳定
- 噪声放大

**改进**: 截止逆滤波
$$\hat{U}(u,v) = \begin{cases}
\frac{F(u,v)}{H(u,v)} & \text{if } |H(u,v)| > \epsilon \\
0 & \text{otherwise}
\end{cases}$$

#### 3.2 Wiener滤波

**最优准则**: 最小化均方误差
$$\min_{\hat{u}} E[|u - \hat{u}|^2]$$

**Wiener滤波器**:
$$\hat{U}(u,v) = \left[\frac{H^*(u,v)}{|H(u,v)|^2 + \frac{S_n(u,v)}{S_u(u,v)}}\right] F(u,v)$$

其中：
- $H^*$: $H$的复共轭
- $S_n$: 噪声功率谱
- $S_u$: 信号功率谱
- $S_n/S_u$: 噪信比

**简化形式** (当噪声和信号功率谱未知):
$$\hat{U}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + K} F(u,v)$$

$K$ 是需要调节的常数。

#### 3.3 约束最小二乘滤波

**正则化形式**:
$$\min_u \|Hu - f\|^2 + \lambda \|Lu\|^2$$

其中 $L$ 是正则化算子（通常是Laplacian）。

**频域解**:
$$\hat{U}(u,v) = \frac{H^*(u,v)}{|H(u,v)|^2 + \gamma |L(u,v)|^2} F(u,v)$$

#### 3.4 Lucy-Richardson算法

**适用于泊松噪声**，基于最大似然估计。

**迭代公式**:
$$u_{k+1} = u_k \cdot \left[h^* * \frac{f}{h * u_k}\right]$$

**特性**:
- 非负性保持
- 非线性方法
- 需要控制迭代次数防止噪声放大

### Part 4: 复原的能量泛函方法 (20分钟)

#### 4.1 一般形式

**最大后验概率(MAP)框架**:
$$\hat{u} = \arg\max_u p(u|f) = \arg\max_u p(f|u) p(u)$$

取负对数：
$$\hat{u} = \arg\min_u \left[-\log p(f|u) - \log p(u)\right]$$

**能量泛函**:
$$J[u] = \underbrace{E_{\text{data}}(u)}_{\text{数据项}} + \lambda \underbrace{E_{\text{reg}}(u)}_{\text{正则项}}$$

#### 4.2 不同噪声对应的数据项

**高斯噪声**:
$$E_{\text{data}}(u) = \int_\Omega (Hu - f)^2 dx$$

**泊松噪声**:
$$E_{\text{data}}(u) = \int_\Omega (Hu - f \log(Hu)) dx$$

**椒盐噪声**:
$$E_{\text{data}}(u) = \int_\Omega |Hu - f| dx$$

#### 4.3 常用正则项

**Tikhonov正则化** ($L^2$范数):
$$E_{\text{reg}}(u) = \int_\Omega |\nabla u|^2 dx$$

**全变分(TV)正则化** ($L^1$范数):
$$E_{\text{reg}}(u) = \int_\Omega |\nabla u| dx$$

**高阶正则化**:
$$E_{\text{reg}}(u) = \int_\Omega |\Delta u|^2 dx$$

#### 4.4 复原问题的不适定性

**Hadamard适定性三条件**:
1. 解存在
2. 解唯一
3. 解稳定（连续依赖于数据）

**图像复原的不适定性**:
- 逆问题通常不适定
- 解不唯一
- 噪声敏感

**正则化的作用**: 将不适定问题转化为适定问题

## 核心公式汇总

| 公式名称 | 表达式 | 说明 |
|---------|--------|------|
| 退化模型 | $f = h * u + n$ | 卷积加噪声 |
| 逆滤波 | $\hat{U} = F/H$ | 需要正则化 |
| Wiener滤波 | $\hat{U} = \frac{H^*F}{|H|^2 + S_n/S_u}$ | MMSE最优 |
| L-R迭代 | $u_{k+1} = u_k \cdot [h^* * (f/(h*u_k))]$ | 泊松噪声 |
| 能量泛函 | $J[u] = \|Hu-f\|^2 + \lambda\|\nabla u\|^2$ | 变分形式 |

## 课后习题

### 1. 概念题
解释为什么逆滤波在实际应用中不稳定，Wiener滤波如何解决这个问题？

### 2. 计算题
给定退化系统 $H = \text{diag}(1, 0.8, 0.5, 0.1, 0.01)$，噪声方差 $\sigma^2 = 0.1$，信号功率 $S_u = 1$，计算Wiener滤波器系数。

### 3. 编程题
```python
import numpy as np
from scipy import fft2, ifft2

def wiener_filter(f, H, K):
    """
    实现Wiener滤波
    f: 退化图像
    H: 退化函数的傅里叶变换
    K: 噪信比参数
    """
    # 请完成此函数
    pass
```

### 4. 推导题
对于泊松噪声模型，推导MAP估计对应的负对数似然函数。

## 扩展阅读

1. **经典教材**
   - Gonzalez & Woods "Digital Image Processing", 第5章
   - Banham & Katsaggelos "Digital Image Restoration", IEEE Signal Processing Magazine, 1997

2. **最新进展**
   - 联合复原与分割方法
   - 深度学习复原方法 (DnCNN, DeblurGAN)

3. **开源代码**
   - skimage.restoration模块
   - OpenCV中的复原函数

## 下节预告
下一讲将深入讲解全变分(TV)模型，这是图像处理中最重要的正则化方法之一。
