# ROF图像去噪从零实现教程

## 目录
1. [理论讲解](#1-理论讲解)
2. [算法详解](#2-算法详解)
3. [代码实现](#3-代码实现)
4. [实验指南](#4-实验指南)
5. [习题与答案](#5-习题与答案)

---

## 1. 理论讲解

### 1.1 ROF模型推导

ROF（Rudin-Osher-Fatemi）模型是图像处理领域最具影响力的变分去噪模型之一，由Rudin、Osher和Fatemi于1992年提出。该模型基于全变分（Total Variation, TV）正则化，能够有效去除噪声同时保持图像边缘。

**问题背景**

设观测到的噪声图像为 $f$，其与原始图像 $u$ 的关系为：
$$f = u + n$$

其中 $n$ 为加性噪声。去噪的目标是从观测图像 $f$ 恢复原始图像 $u$。

**ROF能量泛函**

ROF模型通过最小化以下能量泛函来求解去噪问题：

$$\min_u E(u) = \int_\Omega |\nabla u| \, dx + \frac{\lambda}{2} \int_\Omega (u - f)^2 \, dx$$

其中：
- $\Omega$ 为图像定义域
- $|\nabla u|$ 为图像梯度的模（全变分）
- $\lambda > 0$ 为正则化参数
- 第一项为正则化项（平滑约束）
- 第二项为数据保真项（保持与观测数据的相似性）

**全变分的优势**

与传统的Tikhonov正则化（$|\nabla u|^2$）相比，全变分 $|\nabla u|$ 具有以下优势：
1. 允许图像存在不连续性（边缘）
2. 属于凸泛函，存在唯一解
3. 对阶梯状区域保持良好

### 1.2 变分原理

**变分导数**

为了求解ROF模型，我们需要计算能量泛函的变分导数（Gateaux导数）。设 $v$ 为任意测试函数，变分导数为：

$$\delta E(u; v) = \lim_{\epsilon \to 0} \frac{E(u + \epsilon v) - E(u)}{\epsilon}$$

对于正则化项，其变分导数为：
$$\delta \int_\Omega |\nabla u| \, dx = -\int_\Omega \text{div}\left(\frac{\nabla u}{|\nabla u|}\right) v \, dx$$

对于数据保真项，其变分导数为：
$$\delta \frac{\lambda}{2} \int_\Omega (u - f)^2 \, dx = \lambda \int_\Omega (u - f) v \, dx$$

### 1.3 Euler-Lagrange方程

**最优性条件**

令总变分为零，得到Euler-Lagrange方程：
$$-\text{div}\left(\frac{\nabla u}{|\nabla u|}\right) + \lambda(u - f) = 0$$

配合Neumann边界条件：
$$\left.\frac{\partial u}{\partial n}\right|_{\partial\Omega} = 0$$

**正则化处理**

当 $|\nabla u| = 0$ 时，系数 $\frac{1}{|\nabla u|}$ 无定义。引入正则化参数 $\epsilon > 0$：
$$|\nabla u|_\epsilon = \sqrt{|\nabla u|^2 + \epsilon^2}$$

正则化后的Euler-Lagrange方程：
$$-\text{div}\left(\frac{\nabla u}{\sqrt{|\nabla u|^2 + \epsilon^2}}\right) + \lambda(u - f) = 0$$

### 1.4 存在唯一性

**存在性**

ROF能量泛函的极小化问题存在解，其证明基于直接变分法：
1. 能量泛函在 $BV(\Omega)$ 空间（有界变差函数空间）中是下半连续的
2. 能量泛函具有强制性（coercive）
3. 根据变分法的存在性定理，极小值点存在

**唯一性**

ROF能量泛函是严格凸的，因为：
- 全变分项是凸的
- 数据保真项 $\frac{\lambda}{2}(u-f)^2$ 是严格凸的
- 凸函数之和为凸函数，且严格凸项保证唯一性

**正则化解的性质**

1. **对比度不变性**：若 $g = \alpha f + \beta$，则对应解为 $u_g = \alpha u_f + \beta$
2. **尺度空间性质**：随着 $\lambda$ 减小，解趋向于常数图像
3. **边缘保持性**：在边缘处，梯度不被过度平滑

---

## 2. 算法详解

### 2.1 梯度下降法

**基本思想**

梯度下降法将Euler-Lagrange方程转化为演化方程：
$$\frac{\partial u}{\partial t} = \text{div}\left(\frac{\nabla u}{|\nabla u|_\epsilon}\right) - \lambda(u - f)$$

**离散格式**

时间和空间离散化：
$$u^{n+1}_{i,j} = u^n_{i,j} + \Delta t \left[ D_x\left(\frac{D_x u^n}{|D_x u^n|_\epsilon}\right) + D_y\left(\frac{D_y u^n}{|D_y u^n|_\epsilon}\right) - \lambda(u^n_{i,j} - f_{i,j}) \right]$$

其中 $D_x$, $D_y$ 为离散差分算子。

**CFL条件**

稳定性要求时间步长满足：
$$\Delta t \leq \frac{1}{4 + \lambda}$$

**优缺点**
- 优点：实现简单，直观
- 缺点：收敛慢，迭代次数多

### 2.2 Chambolle对偶算法

**对偶形式**

Chambolle于2004年提出了一种基于对偶问题的算法。ROF问题的对偶形式为：

$$\min_{p \in C} \frac{1}{2}\left\|f - \frac{1}{\lambda}\text{div}(p)\right\|_2^2$$

其中 $C$ 为约束集：
$$C = \{p : \Omega \to \mathbb{R}^2 : |p(x)| \leq 1, \forall x \in \Omega\}$$

**投影梯度下降**

对偶变量的更新公式：
$$p^{n+1} = \Pi_C\left(p^n + \tau \nabla(\text{div}(p^n) - \lambda f)\right)$$

其中 $\Pi_C$ 为到约束集 $C$ 的投影，$\tau$ 为步长。

**投影操作**

$$\Pi_C(p) = \frac{p}{\max(1, |p|)}$$

**步长选择**

收敛性要求：
$$\tau \leq \frac{1}{8}$$

**恢复原图**

从对偶变量恢复原图：
$$u = f - \frac{1}{\lambda}\text{div}(p)$$

### 2.3 原对偶算法

**鞍点问题**

将ROF问题转化为原对偶形式：
$$\min_u \max_p \langle p, \nabla u \rangle + \frac{\lambda}{2}\|u - f\|_2^2 - \delta_C(p)$$

其中 $\delta_C(p)$ 为约束集 $C$ 的指示函数。

**交替更新**

原变量和对偶变量交替更新：
$$p^{n+1} = \Pi_C(p^n + \sigma \nabla \bar{u}^n)$$
$$u^{n+1} = u^n - \tau(\text{div}(p^{n+1}) + \lambda(u^n - f))$$
$$\bar{u}^{n+1} = 2u^{n+1} - u^n$$

**参数选择**

收敛条件：
$$\sigma \tau \leq \frac{1}{4}$$

**收敛性保证**

该算法收敛率为 $O(1/n)$，对于光滑问题可达 $O(1/n^2)$。

### 2.4 收敛性分析

**收敛速度**

| 算法 | 收敛速度 | 复杂度/迭代 |
|------|----------|-------------|
| 梯度下降 | $O(1/\sqrt{n})$ | $O(N)$ |
| Chambolle | $O(1/n)$ | $O(N)$ |
| 原对偶 | $O(1/n)$ | $O(N)$ |

**停止准则**

常用停止条件：
1. 相对误差：$\frac{\|u^{n+1} - u^n\|}{\|u^n\|} < \epsilon$
2. 能量变化：$|E(u^{n+1}) - E(u^n)| < \epsilon$
3. 最大迭代次数

---

## 3. 代码实现

### 3.1 基础工具函数

```python
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def gradient(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算图像的前向差分梯度
    
    参数:
        u: 输入图像 (H, W)
    
    返回:
        ux: x方向梯度 (H, W)
        uy: y方向梯度 (H, W)
    """
    ux = np.roll(u, -1, axis=1) - u
    uy = np.roll(u, -1, axis=0) - u
    return ux, uy


def divergence(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """
    计算对偶变量的散度（后向差分）
    
    参数:
        px: x方向对偶变量 (H, W)
        py: y方向对偶变量 (H, W)
    
    返回:
        div: 散度场 (H, W)
    """
    divx = px - np.roll(px, 1, axis=1)
    divy = py - np.roll(py, 1, axis=0)
    return divx + divy


def tv_norm(u: np.ndarray, eps: float = 1e-8) -> float:
    """
    计算图像的全变分
    
    参数:
        u: 输入图像
        eps: 防止除零的小常数
    
    返回:
        tv: 全变分值
    """
    ux, uy = gradient(u)
    return np.sum(np.sqrt(ux**2 + uy**2 + eps**2))


def project_to_ball(p: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """
    将向量场投影到单位球内
    
    参数:
        p: 向量场 (H, W, 2)
        radius: 球半径
    
    返回:
        投影后的向量场
    """
    norm = np.sqrt(p[:,:,0]**2 + p[:,:,1]**2)
    factor = np.minimum(radius, norm) / (norm + 1e-10)
    return p * factor[:,:,np.newaxis]
```

### 3.2 Chambolle对偶算法实现

```python
def rof_chambolle(f: np.ndarray, 
                  lambda_: float, 
                  n_iter: int = 100,
                  tau: Optional[float] = None,
                  verbose: bool = False) -> np.ndarray:
    """
    Chambolle对偶算法求解ROF模型
    
    参数:
        f: 输入噪声图像 [0,1]
        lambda_: 正则化参数
        n_iter: 迭代次数
        tau: 步长（默认1/8）
        verbose: 是否打印收敛信息
    
    返回:
        u: 去噪后图像
    """
    if tau is None:
        tau = 1.0 / 8.0
    
    # 初始化对偶变量
    h, w = f.shape
    px = np.zeros((h, w))
    py = np.zeros((h, w))
    
    for i in range(n_iter):
        # 计算当前恢复图像
        div_p = divergence(px, py)
        u_temp = f - div_p / lambda_
        
        # 计算恢复图像的梯度
        gx, gy = gradient(u_temp)
        
        # 更新对偶变量（梯度上升步）
        px = px + tau * lambda_ * gx
        py = py + tau * lambda_ * gy
        
        # 投影到约束集
        norm_p = np.sqrt(px**2 + py**2)
        factor = np.minimum(1.0, 1.0 / (norm_p + 1e-10))
        px = px * factor
        py = py * factor
        
        if verbose and (i + 1) % 20 == 0:
            div_p = divergence(px, py)
            u_current = f - div_p / lambda_
            energy = tv_norm(u_current) + lambda_ / 2 * np.sum((u_current - f)**2)
            print(f"Iter {i+1}: Energy = {energy:.6f}")
    
    # 恢复最终图像
    div_p = divergence(px, py)
    u = f - div_p / lambda_
    
    return u


def rof_chambolle_accelerated(f: np.ndarray,
                               lambda_: float,
                               n_iter: int = 100,
                               verbose: bool = False) -> np.ndarray:
    """
    加速版Chambolle算法（FISTA风格）
    
    参数:
        f: 输入噪声图像 [0,1]
        lambda_: 正则化参数
        n_iter: 迭代次数
        verbose: 是否打印收敛信息
    
    返回:
        u: 去噪后图像
    """
    tau = 1.0 / 8.0
    h, w = f.shape
    
    # 初始化
    px = np.zeros((h, w))
    py = np.zeros((h, w))
    px_old = px.copy()
    py_old = py.copy()
    t = 1.0
    
    for i in range(n_iter):
        # FISTA外推
        t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
        beta = (t - 1) / t_new
        
        # 外推点
        px_extrap = px + beta * (px - px_old)
        py_extrap = py + beta * (py - py_old)
        
        # 保存旧值
        px_old = px.copy()
        py_old = py.copy()
        
        # 在外推点计算梯度
        div_p = divergence(px_extrap, py_extrap)
        u_temp = f - div_p / lambda_
        gx, gy = gradient(u_temp)
        
        # 更新对偶变量
        px = px_extrap + tau * lambda_ * gx
        py = py_extrap + tau * lambda_ * gy
        
        # 投影
        norm_p = np.sqrt(px**2 + py**2)
        factor = np.minimum(1.0, 1.0 / (norm_p + 1e-10))
        px = px * factor
        py = py * factor
        
        t = t_new
        
        if verbose and (i + 1) % 20 == 0:
            div_p = divergence(px, py)
            u_current = f - div_p / lambda_
            energy = tv_norm(u_current) + lambda_ / 2 * np.sum((u_current - f)**2)
            print(f"Iter {i+1}: Energy = {energy:.6f}")
    
    div_p = divergence(px, py)
    u = f - div_p / lambda_
    
    return u
```

### 3.3 原对偶算法实现

```python
def rof_primal_dual(f: np.ndarray,
                    lambda_: float,
                    n_iter: int = 100,
                    sigma: Optional[float] = None,
                    tau: Optional[float] = None,
                    verbose: bool = False) -> np.ndarray:
    """
    原对偶算法求解ROF模型
    
    参数:
        f: 输入噪声图像 [0,1]
        lambda_: 正则化参数
        n_iter: 迭代次数
        sigma: 对偶步长
        tau: 原步长
        verbose: 是否打印收敛信息
    
    返回:
        u: 去噪后图像
    """
    h, w = f.shape
    
    # 默认参数
    if sigma is None:
        sigma = 1.0 / np.sqrt(8.0)
    if tau is None:
        tau = 1.0 / np.sqrt(8.0)
    
    # 初始化变量
    u = f.copy()
    u_bar = f.copy()
    px = np.zeros((h, w))
    py = np.zeros((h, w))
    
    for i in range(n_iter):
        # 对偶变量更新
        gx, gy = gradient(u_bar)
        px = px + sigma * gx
        py = py + sigma * gy
        
        # 投影到约束集
        norm_p = np.sqrt(px**2 + py**2)
        factor = np.minimum(1.0, 1.0 / (norm_p + 1e-10))
        px = px * factor
        py = py * factor
        
        # 原变量更新
        u_old = u.copy()
        div_p = divergence(px, py)
        u = (u - tau * div_p + tau * lambda_ * f) / (1 + tau * lambda_)
        
        # 外推
        u_bar = 2 * u - u_old
        
        if verbose and (i + 1) % 20 == 0:
            energy = tv_norm(u) + lambda_ / 2 * np.sum((u - f)**2)
            print(f"Iter {i+1}: Energy = {energy:.6f}")
    
    return u
```

### 3.4 梯度下降法实现

```python
def rof_gradient_descent(f: np.ndarray,
                         lambda_: float,
                         n_iter: int = 500,
                         dt: Optional[float] = None,
                         eps: float = 1e-4,
                         verbose: bool = False) -> np.ndarray:
    """
    显式梯度下降法求解ROF模型
    
    参数:
        f: 输入噪声图像 [0,1]
        lambda_: 正则化参数
        n_iter: 迭代次数
        dt: 时间步长
        eps: 正则化参数（防止除零）
        verbose: 是否打印收敛信息
    
    返回:
        u: 去噪后图像
    """
    h, w = f.shape
    
    # CFL条件确定步长
    if dt is None:
        dt = 1.0 / (4.0 + lambda_)
    
    u = f.copy()
    
    for i in range(n_iter):
        # 计算正则化梯度
        ux, uy = gradient(u)
        norm_grad = np.sqrt(ux**2 + uy**2 + eps**2)
        
        # 归一化梯度
        nx = ux / norm_grad
        ny = uy / norm_grad
        
        # 计算散度
        div_n = divergence(nx, ny)
        
        # 梯度下降更新
        u = u + dt * (div_n - lambda_ * (u - f))
        
        if verbose and (i + 1) % 50 == 0:
            energy = tv_norm(u, eps) + lambda_ / 2 * np.sum((u - f)**2)
            print(f"Iter {i+1}: Energy = {energy:.6f}")
    
    return u
```

### 3.5 噪声添加与评估工具

```python
def add_gaussian_noise(image: np.ndarray, 
                       sigma: float = 0.1,
                       seed: Optional[int] = None) -> np.ndarray:
    """
    添加高斯噪声
    
    参数:
        image: 干净图像 [0,1]
        sigma: 噪声标准差
        seed: 随机种子
    
    返回:
        noisy: 带噪图像
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.randn(*image.shape) * sigma
    noisy = image + noise
    return np.clip(noisy, 0, 1)


def evaluate_denoising(clean: np.ndarray, 
                       noisy: np.ndarray,
                       denoised: np.ndarray) -> dict:
    """
    评估去噪效果
    
    参数:
        clean: 原始干净图像
        noisy: 噪声图像
        denoised: 去噪后图像
    
    返回:
        指标字典
    """
    metrics = {
        'psnr_noisy': psnr(clean, noisy, data_range=1.0),
        'psnr_denoised': psnr(clean, denoised, data_range=1.0),
        'ssim_noisy': ssim(clean, noisy, data_range=1.0),
        'ssim_denoised': ssim(clean, denoised, data_range=1.0),
        'tv_clean': tv_norm(clean),
        'tv_noisy': tv_norm(noisy),
        'tv_denoised': tv_norm(denoised)
    }
    
    metrics['psnr_gain'] = metrics['psnr_denoised'] - metrics['psnr_noisy']
    metrics['ssim_gain'] = metrics['ssim_denoised'] - metrics['ssim_noisy']
    
    return metrics


def plot_results(clean: np.ndarray,
                 noisy: np.ndarray,
                 denoised: np.ndarray,
                 title: str = "ROF Denoising Results"):
    """
    可视化去噪结果
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(clean, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title(f'Noisy (PSNR: {psnr(clean, noisy):.2f}dB)')
    axes[1].axis('off')
    
    axes[2].imshow(denoised, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Denoised (PSNR: {psnr(clean, denoised):.2f}dB)')
    axes[2].axis('off')
    
    diff = np.abs(denoised - clean)
    axes[3].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
    axes[3].set_title('Absolute Error')
    axes[3].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
```

### 3.6 完整示例

```python
def demo_rof_denoising():
    """
    ROF去噪完整演示
    """
    # 加载测试图像
    image = img_as_float(data.camera())
    
    # 添加噪声
    sigma = 0.1
    noisy = add_gaussian_noise(image, sigma=sigma, seed=42)
    
    # 设置正则化参数
    lambda_ = 1.0 / sigma  # 经验公式
    
    print("=" * 50)
    print("ROF图像去噪演示")
    print("=" * 50)
    print(f"图像尺寸: {image.shape}")
    print(f"噪声级别: {sigma}")
    print(f"正则化参数: {lambda_:.2f}")
    print("=" * 50)
    
    # Chambolle算法
    print("\n[1] Chambolle对偶算法")
    u_chambolle = rof_chambolle(noisy, lambda_, n_iter=200, verbose=True)
    m_chambolle = evaluate_denoising(image, noisy, u_chambolle)
    print(f"PSNR: {m_chambolle['psnr_noisy']:.2f} -> {m_chambolle['psnr_denoised']:.2f} dB")
    print(f"SSIM: {m_chambolle['ssim_noisy']:.4f} -> {m_chambolle['ssim_denoised']:.4f}")
    
    # 原对偶算法
    print("\n[2] 原对偶算法")
    u_pd = rof_primal_dual(noisy, lambda_, n_iter=200, verbose=True)
    m_pd = evaluate_denoising(image, noisy, u_pd)
    print(f"PSNR: {m_pd['psnr_noisy']:.2f} -> {m_pd['psnr_denoised']:.2f} dB")
    print(f"SSIM: {m_pd['ssim_noisy']:.4f} -> {m_pd['ssim_denoised']:.4f}")
    
    # 可视化
    plot_results(image, noisy, u_chambolle, "ROF Denoising (Chambolle)")
    
    return u_chambolle, u_pd


def parameter_sweep():
    """
    参数扫描实验
    """
    image = img_as_float(data.camera())
    noisy = add_gaussian_noise(image, sigma=0.1, seed=42)
    
    lambdas = [5, 10, 15, 20, 30]
    results = []
    
    for lam in lambdas:
        u = rof_chambolle(noisy, lam, n_iter=200)
        m = evaluate_denoising(image, noisy, u)
        results.append({
            'lambda': lam,
            'psnr': m['psnr_denoised'],
            'ssim': m['ssim_denoised']
        })
        print(f"lambda={lam}: PSNR={m['psnr_denoised']:.2f}dB, SSIM={m['ssim_denoised']:.4f}")
    
    # 绘制参数曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot([r['lambda'] for r in results], [r['psnr'] for r in results], 'o-')
    ax1.set_xlabel('Lambda')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('PSNR vs Lambda')
    ax1.grid(True)
    
    ax2.plot([r['lambda'] for r in results], [r['ssim'] for r in results], 'o-')
    ax2.set_xlabel('Lambda')
    ax2.set_ylabel('SSIM')
    ax2.set_title('SSIM vs Lambda')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return results


if __name__ == "__main__":
    demo_rof_denoising()
    parameter_sweep()
```

---

## 4. 实验指南

### 4.1 数据集选择

**标准测试图像**
- `skimage.data`: camera, lena, house, barbara
- BSDD500: 自然图像数据集
- Set12: 经典去噪测试集

**噪声类型**
- 高斯噪声：最常用
- 泊松噪声：医学成像
- 混合噪声：实际场景

### 4.2 参数调优

**正则化参数 $\lambda$**

经验公式：
$$\lambda \approx \frac{1}{\sigma}$$

其中 $\sigma$ 为噪声标准差。

**调优策略**
1. 网格搜索
2. 交叉验证
3. 无参方法（噪声估计）

### 4.3 评估指标

| 指标 | 公式 | 范围 |
|------|------|------|
| PSNR | $10\log_{10}\frac{1}{\text{MSE}}$ | 越高越好 |
| SSIM | 结构相似度 | [0,1] |
| TV | 全变分 | 越低越平滑 |

### 4.4 可视化

- 去噪前后对比
- 误差热力图
- 能量收敛曲线
- 参数敏感性曲线

---

## 5. 习题与答案

### 5.1 理论题

**题目1**: 证明ROF能量泛函是凸函数。

**答案**:
1. 全变分项 $\int_\Omega |\nabla u| dx$ 是凸的，因为 $|\cdot|$ 是凸函数，且梯度是线性算子。
2. 数据保真项 $\frac{\lambda}{2}\int_\Omega (u-f)^2 dx$ 是严格凸的，因为平方函数是严格凸的。
3. 凸函数之和仍为凸函数，且存在严格凸项，因此ROF能量泛函是严格凸的。

**题目2**: 解释为什么全变分正则化比Tikhonov正则化更适合图像去噪。

**答案**:
- Tikhonov正则化 $\int |\nabla u|^2$ 惩罚大梯度，导致边缘模糊
- 全变分 $\int |\nabla u|$ 允许大的不连续性（边缘）
- TV保持分段常数结构，适合卡通类图像
- Tikhonov产生过度平滑效果

**题目3**: 推导离散梯度下降法的CFL条件。

**答案**:
离散拉普拉斯算子的最大特征值约为4（二维网格），因此显式格式的稳定性条件为：
$$\Delta t \cdot \lambda_{max} \leq 2$$
$$\Delta t \leq \frac{2}{4 + \lambda} \approx \frac{1}{4 + \lambda}$$

### 5.2 编程题

**题目1**: 实现ROF模型的彩色图像扩展（向量值TV）。

**答案**:
```python
def rof_color_chambolle(f: np.ndarray, lambda_: float, n_iter: int = 100) -> np.ndarray:
    """
    彩色图像ROF去噪（向量值TV）
    
    参数:
        f: 输入RGB图像 (H, W, 3) [0,1]
        lambda_: 正则化参数
        n_iter: 迭代次数
    """
    h, w, c = f.shape
    tau = 1.0 / 8.0
    
    px = np.zeros((h, w))
    py = np.zeros((h, w))
    
    for _ in range(n_iter):
        div_p = divergence(px, py)
        u_temp = f - div_p[:,:,np.newaxis] / lambda_
        
        # 向量值TV：联合梯度
        gx = np.zeros((h, w))
        gy = np.zeros((h, w))
        for ch in range(c):
            gxc, gyc = gradient(u_temp[:,:,ch])
            gx += gxc
            gy += gyc
        
        px = px + tau * lambda_ * gx / c
        py = py + tau * lambda_ * gy / c
        
        norm_p = np.sqrt(px**2 + py**2)
        factor = np.minimum(1.0, 1.0 / (norm_p + 1e-10))
        px *= factor
        py *= factor
    
    div_p = divergence(px, py)
    u = f - div_p[:,:,np.newaxis] / lambda_
    return np.clip(u, 0, 1)
```

**题目2**: 实现自适应参数选择的ROF去噪。

**答案**:
```python
def estimate_noise_sigma(image: np.ndarray) -> float:
    """
    使用MAD方法估计噪声标准差
    """
    from scipy.ndimage import median_filter
    filtered = median_filter(image, size=3)
    diff = image - filtered
    sigma = 1.4826 * np.median(np.abs(diff - np.median(diff)))
    return sigma


def rof_adaptive(f: np.ndarray, n_iter: int = 200) -> np.ndarray:
    """
    自适应参数ROF去噪
    """
    sigma = estimate_noise_sigma(f)
    lambda_ = 1.0 / max(sigma, 0.01)
    print(f"Estimated noise sigma: {sigma:.4f}")
    print(f"Selected lambda: {lambda_:.2f}")
    return rof_chambolle(f, lambda_, n_iter)
```

---

## 参考文献

1. Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D*, 60(1-4), 259-268.

2. Chambolle, A. (2004). An algorithm for total variation minimization and applications. *Journal of Mathematical Imaging and Vision*, 20(1-2), 89-97.

3. Chambolle, A., & Pock, T. (2011). A first-order primal-dual algorithm for convex problems with applications to imaging. *Journal of Mathematical Imaging and Vision*, 40(1), 120-145.

4. Beck, A., & Teboulle, M. (2009). Fast gradient-based algorithms for constrained total variation image denoising and deblurring problems. *IEEE Transactions on Image Processing*, 18(11), 2419-2434.
