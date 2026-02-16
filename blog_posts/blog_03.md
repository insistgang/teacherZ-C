# 水平集方法：让曲线自由流动

在上一篇文章中，我们介绍了主动轮廓模型。传统的主动轮廓（Snake）有一个根本性的局限：它是一条显式参数曲线，无法自动处理拓扑变化。

什么是拓扑变化？简单说，就是曲线的"结构"发生了改变。比如：

- 一条曲线分裂成两条
- 两条曲线合并成一条
- 曲线中出现空洞

这些变化在图像分割中经常发生：一个物体可能被遮挡分成两部分，两个物体可能靠得很近需要分开。

水平集方法（Level Set Method）正是为解决这些问题而生的。

## 从显式到隐式

传统参数曲线的表示是"显式"的：直接给出曲线上的点 $(x(s), y(s))$。

水平集方法采用"隐式"表示：曲线被定义为某个高维函数的"等高线"。

具体来说，曲线 $C$ 是函数 $\phi(x,y)$ 的**零水平集**：

$$C = \{(x,y) : \phi(x,y) = 0\}$$

函数 $\phi$ 叫做**水平集函数**或**符号距离函数**（Signed Distance Function, SDF）。

### 符号距离函数

$\phi$ 的值表示点到曲线的距离，符号表示在曲线的哪一边：

- $\phi > 0$：在曲线内部
- $\phi < 0$：在曲线外部
- $\phi = 0$：在曲线上

更重要的是，$|\nabla\phi| = 1$ 处处成立（在曲线附近）。

### 为什么这种表示更好？

1. **自动处理拓扑变化**：曲线分裂或合并，只需要 $\phi$ 相应变化，不需要特殊处理
2. **计算简单**：曲线的法向量就是 $\nabla\phi / |\nabla\phi|$
3. **数值稳定**：可以在固定网格上计算，不需要处理曲线的参数化

## 曲线演化的水平集方程

假设曲线 $C(t)$ 随时间演化。设水平集函数为 $\phi(x,y,t)$，那么：

$$\phi(x,y,t) = 0 \quad \text{当} \quad (x,y) \in C(t)$$

对时间求导，并注意到曲线上的点满足 $\phi = 0$，可以得到：

$$\frac{\partial \phi}{\partial t} + v_n |\nabla\phi| = 0$$

其中 $v_n$ 是曲线在法方向上的速度。

这是**水平集方程**的一般形式。

### 几何主动轮廓

Caselles、Kimmel和Sapiro在1997年提出了**测地线主动轮廓**（Geodesic Active Contour）：

$$\frac{\partial \phi}{\partial t} = g(|\nabla I|)|\nabla\phi|\left(\text{div}\left(\frac{\nabla\phi}{|\nabla\phi|}\right) + \nu\right) + \nabla g \cdot \nabla\phi$$

其中 $g$ 是边缘检测函数，如 $g(|\nabla I|) = 1/(1+|\nabla I|^2)$。

这个方程可以分解为几项：

1. **曲率流** $g \cdot \kappa \cdot |\nabla\phi|$：平滑曲线
2. **膨胀/收缩** $\nu g |\nabla\phi|$：使曲线膨胀或收缩
3. **边缘吸引** $\nabla g \cdot \nabla\phi$：把曲线拉向边缘

## 水平集的数值实现

水平集方法的数值实现需要考虑几个关键问题。

### 重新初始化

随着演化进行，$\phi$ 会逐渐偏离符号距离函数的性质（$|\nabla\phi| \neq 1$）。这会导致数值不稳定。

解决方法是周期性地**重新初始化** $\phi$ 为符号距离函数。常用的方法是求解：

$$\frac{\partial \phi}{\partial t} = \text{sign}(\phi_0)(1 - |\nabla\phi|)$$

直到收敛。这个方程会驱使 $|\nabla\phi|$ 趋近于1。

### 窄带方法

水平集在整个图像域上定义，但曲线演化只发生在曲线附近。**窄带方法**只更新曲线附近的一个窄带区域，大大提高效率。

### 上风格式

在曲线演化中，信息是沿某一方向传播的。**上风格式**（Upwind Scheme）根据传播方向选择差分方向，保证数值稳定性。

对于方程 $\phi_t + v|\nabla\phi| = 0$，使用：

$$|\nabla\phi| \approx \sqrt{\max(D^{-x}\phi, 0)^2 + \min(D^{+x}\phi, 0)^2 + \max(D^{-y}\phi, 0)^2 + \min(D^{+y}\phi, 0)^2}$$

其中 $D^{-x}\phi$ 是后向差分，$D^{+x}\phi$ 是前向差分。

## 代码实现

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, color
from scipy.ndimage import gaussian_filter

class LevelSetSegmentation:
    """水平集图像分割"""
    
    def __init__(self, image):
        self.image = img_as_float(image)
        if len(self.image.shape) == 3:
            self.image = color.rgb2gray(self.image)
        self.h, self.w = self.image.shape
        
    def initialize_phi(self, center=None, radius=None):
        """初始化水平集函数为圆形"""
        phi = np.ones((self.h, self.w))
        if center is None:
            center = (self.h // 2, self.w // 2)
        if radius is None:
            radius = min(self.h, self.w) // 4
            
        y, x = np.ogrid[:self.h, :self.w]
        mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
        phi[mask] = -1
        return phi
    
    def compute_edge_function(self, sigma=1.0):
        """计算边缘检测函数 g(|∇I|)"""
        # 计算图像梯度
        Ix = np.roll(self.image, -1, axis=1) - self.image
        Iy = np.roll(self.image, -1, axis=0) - self.image
        
        # 高斯平滑
        Ix = gaussian_filter(Ix, sigma)
        Iy = gaussian_filter(Iy, sigma)
        
        # 梯度模长
        grad_I = np.sqrt(Ix**2 + Iy**2 + 1e-10)
        
        # 边缘函数
        g = 1.0 / (1.0 + grad_I**2)
        
        # 计算 ∇g
        gx = np.roll(g, -1, axis=1) - np.roll(g, 1, axis=1)
        gy = np.roll(g, -1, axis=0) - np.roll(g, 1, axis=0)
        gx /= 2
        gy /= 2
        
        return g, gx, gy
    
    def compute_curvature(self, phi):
        """计算曲率 κ = div(∇φ/|∇φ|)"""
        # 中心差分计算梯度
        phi_x = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / 2
        phi_y = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / 2
        
        # 梯度模长
        norm = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)
        
        # 归一化梯度
        nx = phi_x / norm
        ny = phi_y / norm
        
        # 散度
        div_n = ((np.roll(nx, -1, axis=1) - np.roll(nx, 1, axis=1)) / 2 + 
                 (np.roll(ny, -1, axis=0) - np.roll(ny, 1, axis=0)) / 2)
        
        return div_n
    
    def compute_gradient_norm_upwind(self, phi, speed):
        """使用上风格式计算 |∇φ|"""
        # 后向差分
        phi_x_minus = phi - np.roll(phi, 1, axis=1)
        phi_y_minus = phi - np.roll(phi, 1, axis=0)
        
        # 前向差分
        phi_x_plus = np.roll(phi, -1, axis=1) - phi
        phi_y_plus = np.roll(phi, -1, axis=0) - phi
        
        # 根据速度方向选择差分
        if speed >= 0:
            grad_norm = np.sqrt(
                np.maximum(phi_x_minus, 0)**2 + np.minimum(phi_x_plus, 0)**2 +
                np.maximum(phi_y_minus, 0)**2 + np.minimum(phi_y_plus, 0)**2
            )
        else:
            grad_norm = np.sqrt(
                np.minimum(phi_x_minus, 0)**2 + np.maximum(phi_x_plus, 0)**2 +
                np.minimum(phi_y_minus, 0)**2 + np.maximum(phi_y_plus, 0)**2
            )
        
        return grad_norm
    
    def reinitialize(self, phi, n_iter=5, dt=0.5):
        """重新初始化为符号距离函数"""
        for _ in range(n_iter):
            phi_x_minus = phi - np.roll(phi, 1, axis=1)
            phi_y_minus = phi - np.roll(phi, 1, axis=0)
            phi_x_plus = np.roll(phi, -1, axis=1) - phi
            phi_y_plus = np.roll(phi, -1, axis=0) - phi
            
            # Sussman符号函数
            s = phi / np.sqrt(phi**2 + 1)
            
            # 使用Godunov格式
            a = np.maximum(phi_x_plus, 0)
            b = np.minimum(phi_x_minus, 0)
            c = np.maximum(phi_y_plus, 0)
            d = np.minimum(phi_y_minus, 0)
            
            grad_plus = np.sqrt(a**2 + b**2 + c**2 + d**2)
            
            a = np.minimum(phi_x_plus, 0)
            b = np.maximum(phi_x_minus, 0)
            c = np.minimum(phi_y_plus, 0)
            d = np.maximum(phi_y_minus, 0)
            
            grad_minus = np.sqrt(a**2 + b**2 + c**2 + d**2)
            
            # 根据符号选择
            phi = phi - dt * s * (
                np.maximum(phi, 0) * (grad_plus - 1) + 
                np.minimum(phi, 0) * (grad_minus - 1)
            )
        
        return phi
    
    def geodesic_active_contour(self, phi_init, nu=0.0, lambda_=5.0, 
                                 n_iter=300, dt=0.1, reinit_interval=20):
        """
        测地线主动轮廓
        
        参数:
            phi_init: 初始水平集函数
            nu: 膨胀/收缩系数 (正为收缩)
            lambda_: 边缘吸引力系数
            n_iter: 迭代次数
            dt: 时间步长
            reinit_interval: 重新初始化间隔
        """
        phi = phi_init.copy()
        
        # 预计算边缘函数
        g, gx, gy = self.compute_edge_function(sigma=1.5)
        
        for i in range(n_iter):
            # 计算梯度
            phi_x = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / 2
            phi_y = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / 2
            
            # 梯度模长
            grad_phi = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)
            
            # 归一化梯度
            nx = phi_x / grad_phi
            ny = phi_y / grad_phi
            
            # 计算曲率
            kappa = self.compute_curvature(phi)
            
            # 演化方程
            # ∂φ/∂t = g·κ·|∇φ| + ν·g·|∇φ| + ∇g·∇φ
            term1 = g * kappa * grad_phi          # 曲率流
            term2 = nu * g * grad_phi              # 膨胀/收缩
            term3 = lambda_ * (gx * phi_x + gy * phi_y)  # 边缘吸引
            
            phi = phi + dt * (term1 + term2 + term3)
            
            # 重新初始化
            if (i + 1) % reinit_interval == 0:
                phi = self.reinitialize(phi)
        
        return phi
    
    def chan_vese(self, phi_init, mu=0.25, lambda1=1.0, lambda2=1.0,
                  n_iter=200, dt=0.5, epsilon=1.0):
        """
        Chan-Vese主动轮廓
        
        参数:
            phi_init: 初始水平集函数
            mu: 长度正则化参数
            lambda1, lambda2: 区域一致性权重
            n_iter: 迭代次数
            dt: 时间步长
            epsilon: Heaviside函数的正则化参数
        """
        phi = phi_init.copy()
        
        def heaviside(z):
            return 0.5 * (1 + (2/np.pi) * np.arctan(z/epsilon))
        
        def delta(z):
            return epsilon / (np.pi * (z**2 + epsilon**2))
        
        for i in range(n_iter):
            # Heaviside函数
            H = heaviside(phi)
            
            # 区域平均值
            c1 = np.sum(self.image * H) / (np.sum(H) + 1e-10)
            c2 = np.sum(self.image * (1-H)) / (np.sum(1-H) + 1e-10)
            
            # 曲率
            kappa = self.compute_curvature(phi)
            
            # Dirac函数
            D = delta(phi)
            
            # 演化
            phi = phi + dt * D * (
                mu * kappa 
                - lambda1 * (self.image - c1)**2 
                + lambda2 * (self.image - c2)**2
            )
        
        return phi
    
    def get_contour(self, phi):
        """提取零水平集轮廓"""
        from skimage import measure
        contours = measure.find_contours(phi, 0)
        return contours[0] if len(contours) > 0 else None


def demonstrate_topology_change():
    """演示水平集处理拓扑变化的能力"""
    # 创建两个分离的圆形目标
    h, w = 100, 100
    image = np.zeros((h, w))
    
    # 两个圆
    y, x = np.ogrid[:h, :w]
    circle1 = (x - 30)**2 + (y - 50)**2 <= 15**2
    circle2 = (x - 70)**2 + (y - 50)**2 <= 15**2
    image[circle1 | circle2] = 1
    
    # 初始化一个大圆包围两个目标
    ls = LevelSetSegmentation(image)
    phi = ls.initialize_phi(center=(50, 50), radius=40)
    
    # 演化
    phi_final = ls.chan_vese(phi, mu=0.1, n_iter=300)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始图像 (两个目标)')
    
    axes[1].imshow(phi < 0, cmap='gray')
    axes[1].set_title('初始水平集 (一个圆)')
    
    axes[2].imshow(image, cmap='gray')
    axes[2].contour(phi_final, levels=[0], colors='r', linewidths=2)
    axes[2].set_title('演化结果 (自动分裂)')
    
    plt.tight_layout()
    plt.savefig('levelset_topology.png', dpi=150)
    plt.show()


# 示例使用
if __name__ == "__main__":
    # 加载测试图像
    image = data.coins()
    
    # 水平集分割
    ls = LevelSetSegmentation(image)
    phi = ls.initialize_phi(center=(100, 150), radius=50)
    
    # 测地线主动轮廓
    phi_gac = ls.geodesic_active_contour(phi, nu=0.5, n_iter=300)
    
    # Chan-Vese
    phi_cv = ls.chan_vese(phi, mu=0.3, n_iter=200)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].contour(phi, levels=[0], colors='r', linewidths=2)
    axes[0, 0].set_title('初始轮廓')
    
    axes[0, 1].imshow(image, cmap='gray')
    axes[0, 1].contour(phi_gac, levels=[0], colors='r', linewidths=2)
    axes[0, 1].set_title('测地线主动轮廓')
    
    axes[1, 0].imshow(image, cmap='gray')
    axes[1, 0].contour(phi_cv, levels=[0], colors='r', linewidths=2)
    axes[1, 0].set_title('Chan-Vese')
    
    axes[1, 1].imshow(phi_cv, cmap='RdBu')
    axes[1, 1].set_title('水平集函数 φ')
    
    plt.tight_layout()
    plt.savefig('levelset_results.png', dpi=150)
    plt.show()
    
    # 演示拓扑变化
    demonstrate_topology_change()
```

## 水平集的变体与改进

### 变分水平集

传统水平集方法的演化方程来自曲线的几何性质。另一种方法是从**变分原理**出发，设计能量函数，然后推导演化方程。

能量函数的形式通常是：

$$E(\phi) = \int_\Omega g(|\nabla I|) \delta(\phi) |\nabla\phi| dx + \nu \int_\Omega g(|\nabla I|) H(-\phi) dx$$

对 $\phi$ 求变分，得到演化方程。

### 无需重新初始化的水平集

重新初始化增加了计算量，也可能引入误差。Li等人提出了一个巧妙的方法：在能量函数中加入惩罚项：

$$P(\phi) = \int_\Omega \frac{1}{2}(|\nabla\phi| - 1)^2 dx$$

这一项会自动驱使 $|\nabla\phi|$ 趋近于1，无需显式重新初始化。

### 距离正则化水平集（DRLSE）

这是上述思想的进一步发展，使用更平滑的正则化函数：

$$R(\phi) = \int_\Omega r(|\nabla\phi|) dx$$

其中 $r(p)$ 被设计为在 $p=1$ 处取最小值，且在 $p \to 0$ 和 $p \to \infty$ 时都发散。

## 水平集的应用

### 医学图像分割

水平集在医学图像分析中广泛应用：

- **脑部MRI分割**：区分灰质、白质、脑脊液
- **心脏分割**：追踪心脏壁的运动
- **血管分割**：提取血管网络

### 视频目标跟踪

水平集可以自然地处理目标的形变和遮挡：

- 轮廓随时间平滑演化
- 自动处理目标的出现和消失

### 三维重建

水平集可以扩展到三维，用于表面重建：

$$C = \{(x,y,z) : \phi(x,y,z) = 0\}$$

从点云或切片数据重建三维表面。

## 水平集与深度学习的融合

近年来，水平集方法与深度学习结合，产生了强大的混合方法：

### 深度学习预测水平集

用神经网络直接预测水平集函数 $\phi$，而不是通过迭代演化：

- 输入：原始图像
- 输出：水平集函数
- 损失函数：可以包含水平集相关的正则化项

### 深度主动轮廓

神经网络学习主动轮廓的"速度函数"：

$$\frac{\partial \phi}{\partial t} = F_\theta(I, \phi) |\nabla\phi|$$

其中 $F_\theta$ 是由神经网络学习的速度函数。

## 计算效率的改进

水平集方法的主要缺点是计算量大。以下是一些加速方法：

### 多分辨率方法

在粗网格上快速演化，然后在细网格上精化。

### 窄带方法

只更新曲线附近的像素，而不是整个图像域。

### GPU加速

水平集演化中的计算高度并行，适合GPU实现。

## 总结

水平集方法是计算机视觉和图像处理中最优雅的技术之一。它的核心思想是：

**用高维函数的零水平集来隐式表示曲线，通过演化这个函数来实现曲线的变形。**

这种方法的优势是：

1. **自动处理拓扑变化**：曲线可以自由分裂和合并
2. **数值稳定**：在固定网格上计算
3. **几何直观**：曲率、法向量等几何量容易计算

它的挑战是：

1. **计算效率**：需要多次迭代
2. **参数敏感**：演化速度和正则化参数需要调整
3. **初始化依赖**：结果可能依赖于初始曲线的位置

尽管有这些挑战，水平集方法仍然是许多应用的首选方法，尤其是在医学图像分析和需要精确边界的场景中。

---

## 参考文献

1. Osher, S., & Sethian, J. A. (1988). Fronts propagating with curvature-dependent speed: algorithms based on Hamilton-Jacobi formulations. *Journal of Computational Physics*, 79(1), 12-49.

2. Caselles, V., Kimmel, R., & Sapiro, G. (1997). Geodesic active contours. *International Journal of Computer Vision*, 22(1), 61-79.

3. Li, C., Xu, C., Gui, C., & Fox, M. D. (2010). Distance regularized level set evolution and its application to image segmentation. *IEEE Transactions on Image Processing*, 19(12), 3243-3254.

4. Sethian, J. A. (1999). *Level set methods and fast marching methods: evolving interfaces in computational geometry, fluid mechanics, computer vision, and materials science*. Cambridge University Press.

---

*下一篇，我们将探讨凸松弛技术，看看如何把非凸的变分问题转化为可高效求解的凸优化问题。*
