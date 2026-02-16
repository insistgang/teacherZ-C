# 凸松弛：当非凸问题变得可解

在前面的文章中，我们介绍了ROF去噪、主动轮廓和水平集方法。这些方法都涉及到**能量最小化**，而大多数能量函数都是**非凸**的。

非凸优化是一个困难的问题：可能有多个局部最优解，找到全局最优解是NP困难的。

但是，如果我们能把非凸问题转化为凸问题呢？

这就是**凸松弛**（Convex Relaxation）的核心思想：通过某种变换，把非凸问题"松弛"为凸问题，然后高效求解。

## 为什么凸性如此重要？

凸优化有许多美妙的性质：

1. **局部最优即全局最优**：不存在多个局部最优解
2. **高效算法**：有成熟、快速的求解方法
3. **收敛保证**：算法保证收敛到最优解

相比之下，非凸优化：

- 可能陷入局部最优
- 没有通用的高效算法
- 收敛性难以保证

## 一个简单的例子：二值分割

考虑最简单的图像分割问题：把图像分成前景和背景两部分。

### 组合优化形式

设 $u$ 是分割标签，$u_i \in \{0, 1\}$ 表示像素 $i$ 属于背景还是前景。

能量函数可以是：

$$E(u) = \sum_{i} c_i u_i + \sum_{i,j} w_{ij} |u_i - u_j|$$

第一项是**区域项**：$c_i$ 是像素 $i$ 属于前景的代价。
第二项是**边界项**：相邻像素标签不同时付出代价，鼓励平滑的边界。

这是一个**组合优化**问题，因为 $u_i \in \{0, 1\}$ 是离散的。当图像有 $N$ 个像素时，有 $2^N$ 种可能的分割，穷举是不可能的。

### 凸松弛

关键观察：$u_i \in \{0, 1\}$ 等价于 $u_i$ 是实数且 $u_i(u_i-1) = 0$。

后一个条件是非凸的。如果我们把它松弛为 $0 \leq u_i \leq 1$，问题就变成了凸的！

$$\min_{0 \leq u \leq 1} \sum_{i} c_i u_i + \sum_{i,j} w_{ij} |u_i - u_j|$$

这是一个**凸优化**问题。尽管边界项看起来有绝对值，但它是凸的（因为 $u_i - u_j$ 是线性的，绝对值函数是凸的）。

### 阈值化得到二值解

凸松弛的解 $u^*$ 通常在 $(0, 1)$ 之间（分数值）。我们需要把它"舍入"到 $\{0, 1\}$：

$$\hat{u}_i = \begin{cases} 1 & \text{if } u^*_i > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

令人惊讶的是，在很多情况下，凸松弛的解恰好是整数（0或1），这时它就是原始问题的精确解！

## Rudin-Osher-Fatemi (ROF) 模型的凸性

ROF模型：

$$\min_u \|\nabla u\|_1 + \frac{\lambda}{2}\|f - u\|_2^2$$

这个模型已经是凸的！TV项 $\|\nabla u\|_1$ 是凸的（它是线性算子的1-范数），保真项是凸的（它是二次函数）。

所以ROF模型不需要松弛——它本身就是凸的。

## Chan-Vese模型的凸松弛

Chan-Vese模型是非凸的。原始形式：

$$\min_{C, c_1, c_2} \text{Length}(C) + \lambda_1\int_{inside}|I-c_1|^2 + \lambda_2\int_{outside}|I-c_2|^2$$

使用水平集表示，设 $u \in \{0, 1\}$ 表示区域标签：

$$E(u, c_1, c_2) = \|\nabla u\|_1 + \lambda_1\int |I-c_1|^2 u \, dx + \lambda_2\int |I-c_2|^2 (1-u) \, dx$$

固定 $c_1, c_2$ 后，对 $u$ 来说：

- 第一项是凸的（TV）
- 第二、三项是线性的（关于 $u$），所以也是凸的

但是，$u \in \{0, 1\}$ 是非凸约束。

### 凸松弛

把 $u \in \{0, 1\}$ 松弛为 $u \in [0, 1]$：

$$\min_{u \in [0,1]} \|\nabla u\|_1 + \int (r_1 u + r_2 (1-u)) \, dx$$

其中 $r_1 = \lambda_1|I-c_1|^2$，$r_2 = \lambda_2|I-c_2|^2$。

### 全局最优性

Chan等人证明了：如果 $c_1, c_2$ 固定，那么上述凸松弛的解（在阈值化后）是原始问题的全局最优解。

这是一个惊人的结果：一个看起来很困难的非凸问题，竟然可以通过凸松弛精确求解！

## 对偶公式

凸优化问题通常有多种等价形式。**对偶形式**往往能提供新的洞察和更高效的算法。

### TV的对偶

TV可以写成对偶形式：

$$\|u\|_{TV} = \sup_{p \in K} \int u \cdot \text{div}(p) \, dx = \sup_{p \in K} \langle \nabla u, p \rangle$$

其中 $K = \{p : |p|_\infty \leq 1\}$ 是允许的对偶变量集合。

### ROF模型的对偶

ROF模型可以写成**鞍点问题**：

$$\min_u \max_{p \in K} \langle \nabla u, p \rangle + \frac{\lambda}{2}\|f-u\|_2^2$$

这个形式催生了**原始-对偶算法**。

## 原始-对偶算法

Chambolle和Pock在2011年提出了一种高效的原始-对偶算法，专门用于解决这类鞍点问题。

### 算法框架

对于一般形式：

$$\min_x \max_y \langle Kx, y \rangle + G(x) - F^*(y)$$

算法如下：

1. 初始化 $x^0, y^0, \bar{x}^0 = x^0$
2. 迭代：
   - $y^{n+1} = \text{prox}_{\sigma F^*}(y^n + \sigma K\bar{x}^n)$
   - $x^{n+1} = \text{prox}_{\tau G}(x^n - \tau K^* y^{n+1})$
   - $\bar{x}^{n+1} = 2x^{n+1} - x^n$

其中 $\text{prox}$ 是**近端算子**（proximal operator）。

### 收敛性

当 $\tau\sigma\|K\|^2 < 1$ 时，算法保证收敛。

## 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

def prox_conv_conj(y, sigma, f):
    """
    F^*(y) = (1/2)||y - f||^2 的近端算子
    
    prox_{σF^*}(y) = (y + σf) / (1 + σ)
    """
    return (y + sigma * f) / (1 + sigma)


def prox_tv_dual(p, sigma):
    """
    TV对偶的投影: proj_{|·|≤1}
    
    p / max(1, |p|)
    """
    norm = np.sqrt(p[0]**2 + p[1]**2)
    norm = np.maximum(norm, 1.0)
    return p[0] / norm, p[1] / norm


def gradient(u):
    """计算图像梯度 (前向差分)"""
    ux = np.roll(u, -1, axis=1) - u
    uy = np.roll(u, -1, axis=0) - u
    return ux, uy


def divergence(px, py):
    """计算散度 (后向差分)"""
    div_x = px - np.roll(px, 1, axis=1)
    div_y = py - np.roll(py, 1, axis=0)
    return div_x + div_y


def rof_primal_dual(f, lambda_, n_iter=200, verbose=False):
    """
    ROF模型的原始-对偶求解
    
    min_u ||∇u||_1 + (λ/2)||f - u||^2
    
    参数:
        f: 输入图像
        lambda_: 正则化参数
        n_iter: 迭代次数
        verbose: 是否打印进度
    """
    # 初始化
    u = f.copy()
    px = np.zeros_like(f)  # 对偶变量 (x方向)
    py = np.zeros_like(f)  # 对偶变量 (y方向)
    u_bar = f.copy()
    
    # 步长 (满足 τσ||K||^2 < 1, ||K||^2 = 8 for 2D gradient)
    tau = 0.125
    sigma = 0.125
    
    for i in range(n_iter):
        # 对偶更新
        gux, guy = gradient(u_bar)
        px = px + sigma * gux
        py = py + sigma * guy
        px, py = prox_tv_dual((px, py), sigma)
        
        # 原始更新
        div_p = divergence(px, py)
        u_new = (u - tau * (-div_p) + tau * lambda_ * f) / (1 + tau * lambda_)
        
        # 外推
        u_bar = 2 * u_new - u
        u = u_new
        
        if verbose and (i + 1) % 50 == 0:
            tv = np.sum(np.sqrt(gux**2 + guy**2))
            fidelity = 0.5 * lambda_ * np.sum((f - u)**2)
            print(f"Iter {i+1}: TV = {tv:.2f}, Fidelity = {fidelity:.2f}")
    
    return u


def chan_vese_convex(image, lambda1=1.0, lambda2=1.0, mu=0.1, 
                     n_outer=10, n_inner=20):
    """
    Chan-Vese模型的凸松弛求解
    
    参数:
        image: 输入图像
        lambda1, lambda2: 区域权重
        mu: 长度正则化
        n_outer: 外层迭代次数 (更新c1, c2)
        n_inner: 内层迭代次数 (求解u)
    """
    image = img_as_float(image)
    
    # 初始化
    h, w = image.shape
    u = np.ones((h, w)) * 0.5
    
    c1 = np.mean(image) + 0.1
    c2 = np.mean(image) - 0.1
    
    for outer in range(n_outer):
        # 计算区域项
        r1 = lambda1 * (image - c1)**2
        r2 = lambda2 * (image - c2)**2
        
        # 求解凸松弛问题
        # min_u μ||∇u||_1 + <r1 - r2, u> + <r2, 1>
        # s.t. 0 ≤ u ≤ 1
        
        u = solve_binary_segmentation(u, r1 - r2, mu, n_iter=n_inner)
        
        # 更新 c1, c2
        eps = 1e-10
        c1 = np.sum(image * u) / (np.sum(u) + eps)
        c2 = np.sum(image * (1 - u)) / (np.sum(1 - u) + eps)
        
        print(f"Outer {outer+1}: c1 = {c1:.3f}, c2 = {c2:.3f}")
    
    # 阈值化
    segmentation = (u > 0.5).astype(float)
    
    return u, segmentation, c1, c2


def solve_binary_segmentation(u_init, c, mu, n_iter=100):
    """
    求解二值分割的凸松弛
    
    min_u μ||∇u||_1 + <c, u>
    s.t. 0 ≤ u ≤ 1
    
    使用原始-对偶方法
    """
    u = u_init.copy()
    px = np.zeros_like(u)
    py = np.zeros_like(u)
    u_bar = u.copy()
    
    tau = 0.1
    sigma = 0.1
    
    for _ in range(n_iter):
        # 对偶更新
        gux, guy = gradient(u_bar)
        px = px + sigma * gux
        py = py + sigma * guy
        
        # 投影到 |p| ≤ μ
        norm = np.sqrt(px**2 + py**2)
        factor = np.minimum(1.0, mu / (norm + 1e-10))
        px = px * factor
        py = py * factor
        
        # 原始更新
        div_p = divergence(px, py)
        u_new = u - tau * (-div_p + c)
        
        # 投影到 [0, 1]
        u_new = np.clip(u_new, 0, 1)
        
        # 外推
        u_bar = 2 * u_new - u
        u = u_new
    
    return u


def graph_cut_convex(image, foreground_seed, background_seed, 
                     lambda_=1.0, mu=0.1, n_iter=100):
    """
    基于凸松弛的图割分割
    
    参数:
        image: 输入图像
        foreground_seed: 前景种子点 (mask)
        background_seed: 背景种子点 (mask)
        lambda_: 区域项权重
        mu: 边界项权重
        n_iter: 迭代次数
    """
    image = img_as_float(image)
    
    # 计算区域项 (基于颜色模型)
    fg_color = np.mean(image[foreground_seed > 0])
    bg_color = np.mean(image[background_seed > 0])
    
    c_fg = lambda_ * (image - fg_color)**2
    c_bg = lambda_ * (image - bg_color)**2
    
    # 区域项: c = c_fg - c_bg (u=1为前景, u=0为背景)
    c = c_fg - c_bg
    
    # 求解
    u_init = np.ones_like(image) * 0.5
    u_init[foreground_seed > 0] = 1
    u_init[background_seed > 0] = 0
    
    u = solve_binary_segmentation(u_init, c, mu, n_iter)
    
    # 阈值化
    segmentation = (u > 0.5).astype(float)
    
    # 强制种子点
    segmentation[foreground_seed > 0] = 1
    segmentation[background_seed > 0] = 0
    
    return u, segmentation


class ConvexRelaxation:
    """凸松弛方法的统一接口"""
    
    def __init__(self, image):
        self.image = img_as_float(image)
        
    def rof_denoise(self, lambda_=0.1, n_iter=200):
        """ROF去噪"""
        return rof_primal_dual(self.image, lambda_, n_iter)
    
    def binary_segmentation(self, c1, c2, mu=0.1, n_iter=100):
        """二值分割"""
        c = (self.image - c1)**2 - (self.image - c2)**2
        u = solve_binary_segmentation(np.ones_like(self.image) * 0.5, c, mu, n_iter)
        return u
    
    def chan_vese(self, **kwargs):
        """Chan-Vese分割"""
        return chan_vese_convex(self.image, **kwargs)


# 示例使用
if __name__ == "__main__":
    # 测试ROF去噪
    print("=" * 50)
    print("ROF去噪测试")
    print("=" * 50)
    
    from skimage.util import random_noise
    image = data.camera()
    noisy = random_noise(image, var=0.01)
    
    denoised = rof_primal_dual(noisy, lambda_=0.15, n_iter=200, verbose=True)
    
    # 测试Chan-Vese分割
    print("\n" + "=" * 50)
    print("Chan-Vese分割测试")
    print("=" * 50)
    
    image = data.coins()
    u, seg, c1, c2 = chan_vese_convex(image, lambda1=1.0, lambda2=1.0, 
                                       mu=0.5, n_outer=5, n_inner=50)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(noisy, cmap='gray')
    axes[0, 0].set_title('噪声图像')
    
    axes[0, 1].imshow(denoised, cmap='gray')
    axes[0, 1].set_title('ROF去噪 (原始-对偶)')
    
    axes[1, 0].imshow(image, cmap='gray')
    axes[1, 0].contour(u, levels=[0.5], colors='r', linewidths=2)
    axes[1, 0].set_title(f'Chan-Vese分割 (c1={c1:.1f}, c2={c2:.1f})')
    
    axes[1, 1].imshow(u, cmap='RdBu')
    axes[1, 1].set_title('凸松弛解 u ∈ [0,1]')
    
    plt.tight_layout()
    plt.savefig('convex_relaxation_results.png', dpi=150)
    plt.show()
```

## 松弛的精确性

凸松弛的一个重要问题是：松弛后的解与原始问题的解有多接近？

### 精确松弛

如果凸松弛的解恰好满足原始问题的约束（即 $u \in \{0, 1\}$），那么它就是原始问题的精确解。

这种情况在某些条件下成立：

1. **两个区域差异足够大**：前景和背景的颜色分布有足够的分离
2. **正则化参数适中**：既不太大（过度平滑）也不太小（过拟合噪声）

### 近似松弛

如果松弛的解不是整数，我们需要阈值化。这时，阈值化后的解可能不是原始问题的最优解。

但是，Chan等人证明了：阈值化后的解满足某种"近似最优性"——它的能量不超过最优解的某个倍数。

## 多区域分割的凸松弛

对于多区域分割，凸松弛变得更复杂。

### 松弛排列约束

设 $u_i \in \{0, 1\}$ 表示像素属于区域 $i$，约束为每个像素恰好属于一个区域：

$$\sum_i u_i = 1$$

这是一个排列约束。凸松弛为：

$$\sum_i u_i = 1, \quad u_i \geq 0$$

这把每个像素的标签从 $\{e_1, ..., e_k\}$ 松弛为单纯形 $\{u : \sum_i u_i = 1, u_i \geq 0\}$。

### 多区域TV

多区域分割的TV项是：

$$\text{TV}(u) = \sum_i \|\nabla u_i\|_1$$

或者使用更精细的边界长度度量：

$$\text{Length}(\partial \Omega) = \int \sqrt{\sum_i |\nabla u_i|^2} \, dx$$

后者是**向量值TV**，更准确地度量边界长度。

## 凸松弛与深度学习

凸松弛的思想正在与深度学习结合：

### 深度展开

把凸优化的迭代过程"展开"成神经网络：

- 每一层是一次优化迭代
- 参数（如步长、正则化权重）可以学习
- 结合数据驱动和模型驱动

### 可学习的能量函数

用神经网络学习能量函数的各项：

$$E_\theta(u) = \text{TV}(u) + \langle f_\theta(I), u \rangle$$

其中 $f_\theta$ 是由网络学习的区域项。

## 总结

凸松弛是一种强大的技术，它把困难的非凸问题转化为可高效求解的凸问题。

核心思想是：

1. **识别非凸约束**：通常是整数约束（$u \in \{0, 1\}$）
2. **松弛为凸约束**：通常是区间约束（$u \in [0, 1]$）
3. **求解凸问题**：使用原始-对偶、ADMM等算法
4. **舍入**：如果解不是整数，进行阈值化

凸松弛的美妙之处在于：在很多情况下，松弛是"紧"的，即松弛后的解恰好满足原始约束。

即使松弛不是紧的，凸松弛也提供了一个良好的近似，而且有理论保证。

---

## 参考文献

1. Chan, T. F., Esedoglu, S., & Nikolova, M. (2006). Algorithms for finding global minimizers of image segmentation and denoising models. *SIAM Journal on Applied Mathematics*, 66(5), 1632-1648.

2. Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for convex problems with applications to imaging. *Journal of Mathematical Imaging and Vision*, 40(1), 120-145.

3. Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26(9), 1124-1137.

4. Lellmann, J., & Schnörr, C. (2011). Continuous multiclass labeling approaches and algorithms. *SIAM Journal on Imaging Sciences*, 4(4), 1049-1096.

---

*下一篇，我们将深入探讨全变分正则化，理解为什么它在图像处理中如此有效。*
