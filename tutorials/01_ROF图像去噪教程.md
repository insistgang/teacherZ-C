# ROF图像去噪从零实现教程

## 目录
1. [理论部分](#理论部分)
2. [实现部分](#实现部分)
3. [实验部分](#实验部分)
4. [习题](#习题)

---

## 理论部分

### 1.1 ROF模型推导

#### 1.1.1 图像去噪的变分框架

图像去噪问题可以表述为变分优化问题：给定噪声图像 $f$，寻找去噪图像 $u$，使得：

$$\min_u E(u) = \underbrace{\frac{\lambda}{2}\|f - u\|_{L^2}^2}_{\text{数据保真项}} + \underbrace{TV(u)}_{\text{正则项}}$$

其中：
- $f: \Omega \to \mathbb{R}$ 是观测到的噪声图像
- $u: \Omega \to \mathbb{R}$ 是待求解的去噪图像
- $\Omega \subset \mathbb{R}^2$ 是图像域
- $\lambda > 0$ 是正则化参数
- $TV(u)$ 是全变分（Total Variation）

#### 1.1.2 全变分（Total Variation）定义

对于光滑函数 $u$，全变分定义为：

$$TV(u) = \int_\Omega |\nabla u| \, dx = \int_\Omega \sqrt{\left(\frac{\partial u}{\partial x}\right)^2 + \left(\frac{\partial u}{\partial y}\right)^2} \, dx$$

**为什么选择TV正则化？**

1. **边缘保持性**：TV惩罚梯度的幅度，但不惩罚梯度的方向。在边缘处，梯度大但TV允许这种"跳跃"。

2. **对比L2正则化**：
   - L2正则化：$\int |\nabla u|^2 dx$ 导致边缘模糊
   - TV正则化：$\int |\nabla u| dx$ 保持边缘锐利

3. **数学性质**：TV是L1范数在梯度空间的对偶，具有稀疏促进性质。

#### 1.1.3 ROF模型的历史

Rudin、Osher和Fatemi在1992年的经典论文中提出：

> **核心洞察**：图像的边缘承载重要信息，不应该被过度平滑。TV正则化恰好能够：
> - 在平滑区域（$\nabla u \approx 0$）强烈去噪
> - 在边缘区域（$|\nabla u|$ 大）保持结构

### 1.2 变分原理

#### 1.2.1 变分问题的一般形式

考虑泛函 $E(u) = \int_\Omega L(x, u, \nabla u) dx$，其中 $L$ 是拉格朗日函数。

**一阶变分**：
$$\delta E = \lim_{\epsilon \to 0} \frac{E(u + \epsilon v) - E(u)}{\epsilon}$$

其中 $v$ 是测试函数（变分方向）。

#### 1.2.2 ROF模型的变分形式

对于ROF模型：
$$E(u) = \frac{\lambda}{2}\int_\Omega (f-u)^2 dx + \int_\Omega |\nabla u| dx$$

计算变分：

**数据项变分**：
$$\delta \left(\frac{\lambda}{2}\int (f-u)^2 dx\right) = \lambda \int (u-f) v \, dx$$

**TV项变分**（形式上）：
$$\delta \left(\int |\nabla u| dx\right) = \int \frac{\nabla u}{|\nabla u|} \cdot \nabla v \, dx$$

### 1.3 Euler-Lagrange方程

#### 1.3.1 标准推导

对于泛函 $E(u) = \int_\Omega L(x, u, \nabla u) dx$，Euler-Lagrange方程为：

$$\frac{\partial L}{\partial u} - \nabla \cdot \frac{\partial L}{\partial \nabla u} = 0$$

对于ROF模型的TV项，$L = |\nabla u| = \sqrt{u_x^2 + u_y^2}$：

$$\frac{\partial L}{\partial u_x} = \frac{u_x}{|\nabla u|}, \quad \frac{\partial L}{\partial u_y} = \frac{u_y}{|\nabla u|}$$

因此：
$$\nabla \cdot \frac{\nabla u}{|\nabla u|} = 0$$

#### 1.3.2 ROF的Euler-Lagrange方程

结合数据项，得到：

$$\lambda(u - f) - \nabla \cdot \left(\frac{\nabla u}{|\nabla u|}\right) = 0$$

**问题**：当 $|\nabla u| = 0$ 时，分母为零！

#### 1.3.3 正则化处理

引入小参数 $\epsilon > 0$：

$$\nabla \cdot \left(\frac{\nabla u}{\sqrt{|\nabla u|^2 + \epsilon^2}}\right) + \lambda(f - u) = 0$$

或使用**对偶变量** $p = \nabla u / |\nabla u|$（见Chambolle算法）。

### 1.4 Chambolle对偶算法

#### 1.4.1 对偶问题推导

ROF原问题：
$$\min_u \left\{ \frac{\lambda}{2}\|f-u\|^2 + TV(u) \right\}$$

引入对偶变量 $p: \Omega \to \mathbb{R}^2$，利用：
$$TV(u) = \sup_{\|p\|_\infty \leq 1} \int_\Omega u \cdot \nabla \cdot p \, dx$$

得到**鞍点问题**：
$$\min_u \max_{\|p\|_\infty \leq 1} \left\{ \frac{\lambda}{2}\|f-u\|^2 + \langle \nabla u, p \rangle \right\}$$

#### 1.4.2 Chambolle投影算法

**算法步骤**：

1. 初始化 $p^0 = 0$

2. 迭代更新：
   $$p^{n+1} = \text{proj}_P\left(p^n + \tau \nabla(f - \lambda \nabla \cdot p^n)\right)$$

3. 投影算子：
   $$\text{proj}_P(p) = \frac{p}{\max(1, |p|)}$$

4. 恢复 $u$：
   $$u = f - \lambda \nabla \cdot p$$

**收敛条件**：步长 $\tau \leq 1/8$（2D情况）

---

## 实现部分

### 2.1 梯度下降法

```python
"""
ROF模型求解 - 梯度下降法实现

最直接的数值方法，使用显式Euler迭代求解Euler-Lagrange方程。
"""

import numpy as np
from typing import Tuple, Optional

def gradient_descent_rof(
    f: np.ndarray,
    lambda_param: float = 0.1,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tol: float = 1e-5,
    dt: float = 0.01
) -> Tuple[np.ndarray, dict]:
    """
    使用梯度下降法求解ROF模型
    
    参数:
        f: 输入噪声图像 (H, W)
        lambda_param: 数据保真项权重
        epsilon: TV正则化小参数，避免除零
        max_iter: 最大迭代次数
        tol: 收敛阈值
        dt: 时间步长
        
    返回:
        u: 去噪图像
        info: 收敛信息字典
    """
    f = f.astype(np.float64)
    u = f.copy()
    H, W = f.shape
    
    residuals = []
    
    for n in range(max_iter):
        u_old = u.copy()
        
        # 计算梯度
        ux = np.zeros_like(u)
        uy = np.zeros_like(u)
        ux[:-1, :] = u[1:, :] - u[:-1, :]  # ∂u/∂x (前向差分)
        uy[:, :-1] = u[:, 1:] - u[:, :-1]  # ∂u/∂y (前向差分)
        
        # 计算梯度幅度
        grad_norm = np.sqrt(ux**2 + uy**2 + epsilon**2)
        
        # 计算 curv = ∇·(∇u/|∇u|)
        # 使用中心差分的散度
        nx = ux / grad_norm  # n_x = u_x/|∇u|
        ny = uy / grad_norm  # n_y = u_y/|∇u|
        
        curv = np.zeros_like(u)
        # ∂n_x/∂x (后向差分)
        curv[1:, :] += nx[1:, :] - nx[:-1, :]
        # ∂n_y/∂y (后向差分)
        curv[:, 1:] += ny[:, 1:] - ny[:, :-1]
        
        # 梯度下降更新
        # ∂u/∂t = curv - λ(u - f)
        u = u + dt * (curv - lambda_param * (u - f))
        
        # 计算残差
        residual = np.linalg.norm(u - u_old) / (np.linalg.norm(u_old) + 1e-10)
        residuals.append(residual)
        
        if residual < tol:
            break
    
    info = {
        'iterations': n + 1,
        'converged': residual < tol,
        'final_residual': residual,
        'residual_history': residuals
    }
    
    return u, info


# 测试梯度下降法
def test_gradient_descent():
    np.random.seed(42)
    
    # 创建合成图像
    H, W = 64, 64
    clean = np.zeros((H, W))
    clean[:H//2, :] = 1.0
    
    # 添加噪声
    noisy = clean + 0.2 * np.random.randn(H, W)
    
    # 去噪
    denoised, info = gradient_descent_rof(noisy, lambda_param=0.1)
    
    print(f"梯度下降法去噪:")
    print(f"  迭代次数: {info['iterations']}")
    print(f"  收敛状态: {info['converged']}")
    print(f"  最终残差: {info['final_residual']:.2e}")
    
    return denoised, info

if __name__ == "__main__":
    test_gradient_descent()
```

### 2.2 Chambolle对偶算法

```python
"""
ROF模型求解 - Chambolle投影算法

基于对偶形式的高效求解方法，收敛速度快，数值稳定。
"""

import numpy as np
from typing import Tuple, Optional

class ChambolleROF:
    """
    Chambolle投影算法求解ROF模型
    
    数学原理:
        对偶问题: p^{n+1} = proj_P(p^n + τ∇(f - λ div(p^n)))
        其中 P = {p : |p| ≤ 1} 是单位球
        
        恢复: u = f - λ div(p)
    """
    
    def __init__(self, lambda_param: float = 0.1, tau: float = 0.25, 
                 max_iter: int = 100, tol: float = 1e-4):
        """
        参数:
            lambda_param: 数据保真项权重λ
            tau: 迭代步长，需满足 τ ≤ 1/4 (2D情况更安全用 ≤ 1/8)
            max_iter: 最大迭代次数
            tol: 收敛阈值
        """
        self.lambda_param = lambda_param
        self.tau = tau
        self.max_iter = max_iter
        self.tol = tol
        
    def solve(self, f: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        求解ROF模型
        
        参数:
            f: 输入图像 (H, W) 或 (H, W, C)
            
        返回:
            u: 去噪图像
            info: 求解信息
        """
        f = f.astype(np.float64)
        original_shape = f.shape
        
        # 处理多通道
        if f.ndim == 2:
            u = self._solve_single(f)
        else:
            u = np.zeros_like(f)
            for c in range(f.shape[2]):
                u[:, :, c] = self._solve_single(f[:, :, c])
        
        return u, self.info
    
    def _solve_single(self, f: np.ndarray) -> np.ndarray:
        """单通道ROF求解"""
        H, W = f.shape
        
        # 初始化对偶变量 p = (p_x, p_y)
        p = np.zeros((H, W, 2), dtype=np.float64)
        
        for n in range(self.max_iter):
            p_old = p.copy()
            
            # 计算 u = f - λ div(p)
            div_p = self._divergence(p)
            u = f - self.lambda_param * div_p
            
            # 计算梯度 ∇u
            grad_u = self._gradient(u)
            
            # 对偶更新: p_new = p + τ ∇u
            p_new = p + self.tau * grad_u
            
            # 投影到单位球
            p = self._project_unit_ball(p_new)
            
            # 检查收敛
            residual = np.linalg.norm(p - p_old) / (np.linalg.norm(p_old) + 1e-10)
            if residual < self.tol:
                break
        
        # 最终恢复
        div_p = self._divergence(p)
        u = f - self.lambda_param * div_p
        
        self.info = {
            'iterations': n + 1,
            'converged': residual < self.tol,
            'final_residual': residual
        }
        
        return u
    
    def _gradient(self, u: np.ndarray) -> np.ndarray:
        """
        计算梯度 ∇u = (∂u/∂x, ∂u/∂y)
        使用前向差分，Neumann边界条件
        """
        H, W = u.shape
        grad = np.zeros((H, W, 2), dtype=np.float64)
        
        # ∂u/∂x[i,j] = u[i+1,j] - u[i,j]
        grad[:-1, :, 0] = u[1:, :] - u[:-1, :]
        # 边界: grad[-1,:,0] = 0
        
        # ∂u/∂y[i,j] = u[i,j+1] - u[i,j]
        grad[:, :-1, 1] = u[:, 1:] - u[:, :-1]
        # 边界: grad[:,-1,1] = 0
        
        return grad
    
    def _divergence(self, p: np.ndarray) -> np.ndarray:
        """
        计算散度 div(p) = ∂p_x/∂x + ∂p_y/∂y
        这是梯度的负伴随算子，使用后向差分
        """
        H, W = p.shape[:2]
        div = np.zeros((H, W), dtype=np.float64)
        
        # ∂p_x/∂x (后向差分)
        div[1:, :] += p[1:, :, 0] - p[:-1, :, 0]
        div[0, :] += p[0, :, 0]  # 边界条件
        
        # ∂p_y/∂y (后向差分)
        div[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]
        div[:, 0] += p[:, 0, 1]  # 边界条件
        
        return div
    
    def _project_unit_ball(self, p: np.ndarray) -> np.ndarray:
        """
        投影到单位球 P = {p : |p|_∞ ≤ 1}
        proj_P(p) = p / max(1, |p|)
        """
        # 计算每点的向量范数
        p_norm = np.sqrt(p[:, :, 0]**2 + p[:, :, 1]**2 + 1e-10)
        
        # 归一化因子
        scale = np.maximum(p_norm, 1.0)
        
        # 执行投影
        p_proj = np.zeros_like(p)
        p_proj[:, :, 0] = p[:, :, 0] / scale
        p_proj[:, :, 1] = p[:, :, 1] / scale
        
        return p_proj


# 测试Chambolle算法
def test_chambolle():
    np.random.seed(42)
    
    H, W = 128, 128
    clean = np.zeros((H, W))
    clean[:H//2, :] = 1.0
    
    noisy = clean + 0.2 * np.random.randn(H, W)
    
    # 测试不同lambda
    for lam in [0.05, 0.1, 0.2, 0.5]:
        solver = ChambolleROF(lambda_param=lam, tau=0.25)
        denoised, info = solver.solve(noisy)
        
        mse_noisy = np.mean((noisy - clean)**2)
        mse_denoised = np.mean((denoised - clean)**2)
        
        print(f"λ = {lam:.2f}: MSE {mse_noisy:.4f} → {mse_denoised:.4f}, "
              f"迭代 {info['iterations']}")

if __name__ == "__main__":
    test_chambolle()
```

### 2.3 Chambolle-Pock原对偶算法

```python
"""
ROF模型求解 - Chambolle-Pock原对偶算法

一阶原对偶方法，收敛速度快，可扩展到更复杂的问题。
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class SolverConfig:
    """求解器配置"""
    lambda_param: float = 0.1
    tau: float = 0.01        # 原步长
    sigma: float = 0.1       # 对偶步长
    max_iter: int = 500
    tol: float = 1e-5
    verbose: bool = False


class ChambollePockROF:
    """
    Chambolle-Pock原对偶算法求解ROF模型
    
    算法迭代:
        1. 对偶更新: p^{n+1} = proj_P(p^n + σ∇ū^n)
        2. 原更新: u^{n+1} = (u^n + τλf + τdiv(p^{n+1})) / (1 + τλ)
        3. 外推: ū^{n+1} = 2u^{n+1} - u^n
    
    收敛条件: τσ < 1/||K||², 对于梯度算子 ||K||² ≈ 8
    """
    
    def __init__(self, config: Optional[SolverConfig] = None):
        if config is None:
            config = SolverConfig()
        self.config = config
        
        # 验证步长条件
        if config.tau * config.sigma >= 0.125:
            print(f"警告: τσ = {config.tau * config.sigma:.4f} 可能过大")
    
    def solve(self, f: np.ndarray) -> Tuple[np.ndarray, dict]:
        """求解ROF模型"""
        cfg = self.config
        f = f.astype(np.float64)
        
        # 初始化
        u = f.copy()           # 原变量
        u_bar = f.copy()       # 外推变量
        p = np.zeros((*f.shape, 2), dtype=np.float64)  # 对偶变量
        
        residuals = []
        
        for n in range(cfg.max_iter):
            u_prev = u.copy()
            
            # 步骤1: 对偶更新
            grad_u_bar = self._gradient(u_bar)
            p_new = p + cfg.sigma * grad_u_bar
            p = self._project_unit_ball(p_new)
            
            # 步骤2: 原更新
            div_p = self._divergence(p)
            u = (u + cfg.tau * cfg.lambda_param * f + cfg.tau * div_p) / \
                (1.0 + cfg.tau * cfg.lambda_param)
            
            # 步骤3: 外推
            u_bar = 2.0 * u - u_prev
            
            # 收敛检查
            residual = np.linalg.norm(u - u_prev) / (np.linalg.norm(u_prev) + 1e-10)
            residuals.append(residual)
            
            if residual < cfg.tol:
                if cfg.verbose:
                    print(f"第 {n+1} 次迭代收敛")
                break
        
        info = {
            'iterations': n + 1,
            'converged': residual < cfg.tol,
            'final_residual': residual,
            'residual_history': residuals
        }
        
        return u, info
    
    def _gradient(self, u: np.ndarray) -> np.ndarray:
        """前向差分梯度"""
        H, W = u.shape
        grad = np.zeros((H, W, 2), dtype=np.float64)
        grad[:-1, :, 0] = u[1:, :] - u[:-1, :]
        grad[:, :-1, 1] = u[:, 1:] - u[:, :-1]
        return grad
    
    def _divergence(self, p: np.ndarray) -> np.ndarray:
        """后向差分散度"""
        H, W = p.shape[:2]
        div = np.zeros((H, W), dtype=np.float64)
        div[1:, :] += p[1:, :, 0] - p[:-1, :, 0]
        div[0, :] += p[0, :, 0]
        div[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]
        div[:, 0] += p[:, 0, 1]
        return div
    
    def _project_unit_ball(self, p: np.ndarray) -> np.ndarray:
        """投影到单位无穷范数球"""
        p_norm = np.sqrt(p[..., 0]**2 + p[..., 1]**2 + 1e-10)
        scale = np.maximum(p_norm, 1.0)
        p_proj = np.zeros_like(p)
        p_proj[..., 0] = p[..., 0] / scale
        p_proj[..., 1] = p[..., 1] / scale
        return p_proj


# 性能对比
def compare_algorithms():
    """对比三种算法的性能"""
    import time
    
    np.random.seed(42)
    H, W = 256, 256
    
    # 创建测试图像
    clean = np.zeros((H, W))
    clean[:H//3, :] = 0.3
    clean[H//3:2*H//3, :] = 0.6
    clean[2*H//3:, :] = 0.9
    
    noisy = clean + 0.15 * np.random.randn(H, W)
    noisy = np.clip(noisy, 0, 1)
    
    lambda_param = 0.1
    
    print("=" * 60)
    print("算法性能对比")
    print("=" * 60)
    
    # 方法1: 梯度下降
    start = time.time()
    u1, info1 = gradient_descent_rof(noisy, lambda_param, max_iter=2000)
    t1 = time.time() - start
    mse1 = np.mean((u1 - clean)**2)
    print(f"梯度下降法: 时间={t1:.2f}s, 迭代={info1['iterations']}, MSE={mse1:.6f}")
    
    # 方法2: Chambolle投影
    start = time.time()
    solver2 = ChambolleROF(lambda_param, max_iter=500)
    u2, info2 = solver2.solve(noisy)
    t2 = time.time() - start
    mse2 = np.mean((u2 - clean)**2)
    print(f"Chambolle投影: 时间={t2:.2f}s, 迭代={info2['iterations']}, MSE={mse2:.6f}")
    
    # 方法3: Chambolle-Pock
    start = time.time()
    config = SolverConfig(lambda_param=lambda_param, verbose=False)
    solver3 = ChambollePockROF(config)
    u3, info3 = solver3.solve(noisy)
    t3 = time.time() - start
    mse3 = np.mean((u3 - clean)**2)
    print(f"Chambolle-Pock: 时间={t3:.2f}s, 迭代={info3['iterations']}, MSE={mse3:.6f}")

if __name__ == "__main__":
    compare_algorithms()
```

---

## 实验部分

### 3.1 不同噪声水平测试

```python
"""
实验1: 不同噪声水平下的ROF去噪效果
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def experiment_noise_levels():
    """测试不同噪声水平"""
    np.random.seed(42)
    
    # 创建测试图像
    H, W = 128, 128
    clean = np.zeros((H, W))
    clean[:H//2, :] = 1.0
    
    noise_levels = [0.05, 0.1, 0.15, 0.2, 0.3]
    results = []
    
    for sigma in noise_levels:
        # 添加噪声
        noisy = clean + sigma * np.random.randn(H, W)
        
        # 去噪 (使用最优lambda)
        lambda_opt = sigma * 2  # 经验公式
        solver = ChambolleROF(lambda_param=lambda_opt, max_iter=200)
        denoised, info = solver.solve(noisy)
        
        # 计算指标
        mse_noisy = np.mean((noisy - clean)**2)
        mse_denoised = np.mean((denoised - clean)**2)
        psnr_improvement = 10 * np.log10(mse_noisy / mse_denoised)
        
        results.append({
            'sigma': sigma,
            'mse_noisy': mse_noisy,
            'mse_denoised': mse_denoised,
            'psnr_gain': psnr_improvement,
            'iterations': info['iterations']
        })
        
        print(f"噪声σ={sigma:.2f}: MSE {mse_noisy:.4f}→{mse_denoised:.4f}, "
              f"PSNR提升 {psnr_improvement:.2f}dB")
    
    return results

def plot_results(results: List[dict]):
    """绘制结果曲线"""
    sigmas = [r['sigma'] for r in results]
    mse_noisy = [r['mse_noisy'] for r in results]
    mse_denoised = [r['mse_denoised'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # MSE对比
    axes[0].plot(sigmas, mse_noisy, 'o-', label='噪声图像')
    axes[0].plot(sigmas, mse_denoised, 's-', label='去噪图像')
    axes[0].set_xlabel('噪声水平 σ')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE随噪声水平变化')
    axes[0].legend()
    axes[0].grid(True)
    
    # PSNR提升
    psnr_gain = [r['psnr_gain'] for r in results]
    axes[1].bar(range(len(sigmas)), psnr_gain, tick_label=[f'σ={s}' for s in sigmas])
    axes[1].set_ylabel('PSNR提升 (dB)')
    axes[1].set_title('去噪效果(PSNR提升)')
    axes[1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('rof_noise_experiment.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    results = experiment_noise_levels()
    plot_results(results)
```

### 3.2 参数调优指南

```python
"""
实验2: λ参数调优
"""

def parameter_tuning_guide():
    """λ参数选择指南"""
    np.random.seed(42)
    
    H, W = 128, 128
    clean = np.zeros((H, W))
    clean[:H//2, :] = 1.0
    noisy = clean + 0.15 * np.random.randn(H, W)
    
    lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    print("λ参数调优指南:")
    print("-" * 50)
    print(f"{'λ':>8} {'MSE':>10} {'TV':>10} {'说明':>20}")
    print("-" * 50)
    
    for lam in lambda_values:
        solver = ChambolleROF(lambda_param=lam, max_iter=200)
        denoised, _ = solver.solve(noisy)
        
        mse = np.mean((denoised - clean)**2)
        
        # 计算TV
        grad_x = np.abs(denoised[1:, :] - denoised[:-1, :]).sum()
        grad_y = np.abs(denoised[:, 1:] - denoised[:, :-1]).sum()
        tv = grad_x + grad_y
        
        if lam < 0.05:
            note = "过强平滑"
        elif lam < 0.2:
            note = "适中"
        elif lam < 1.0:
            note = "弱平滑"
        else:
            note = "几乎无平滑"
        
        print(f"{lam:>8.2f} {mse:>10.6f} {tv:>10.2f} {note:>20}")
    
    print("-" * 50)
    print("\n推荐规则:")
    print("  - 弱噪声 (σ < 0.05): λ = 0.5 - 2.0")
    print("  - 中等噪声 (σ ≈ 0.1): λ = 0.1 - 0.3")
    print("  - 强噪声 (σ > 0.2): λ = 0.01 - 0.1")

if __name__ == "__main__":
    parameter_tuning_guide()
```

### 3.3 性能对比

```python
"""
实验3: 与其他去噪方法对比
"""

from scipy.ndimage import gaussian_filter, median_filter

def compare_denoising_methods():
    """对比多种去噪方法"""
    np.random.seed(42)
    
    H, W = 128, 128
    clean = np.zeros((H, W))
    clean[:H//2, :] = 1.0
    noisy = clean + 0.15 * np.random.randn(H, W)
    
    methods = {}
    
    # 1. ROF去噪
    solver = ChambolleROF(lambda_param=0.15, max_iter=200)
    methods['ROF'] = solver.solve(noisy)[0]
    
    # 2. 高斯滤波
    methods['Gaussian (σ=2)'] = gaussian_filter(noisy, sigma=2)
    
    # 3. 中值滤波
    methods['Median (3x3)'] = median_filter(noisy, size=3)
    
    # 4. TV去噪（正则化版本）
    methods['TV (ε=0.01)'], _ = gradient_descent_rof(noisy, lambda_param=0.15, epsilon=0.01)
    
    print("=" * 60)
    print("去噪方法对比")
    print("=" * 60)
    print(f"{'方法':<20} {'MSE':<12} {'PSNR (dB)':<12} {'边缘保持':<12}")
    print("-" * 60)
    
    # 计算边缘保持指标
    def edge_preservation(denoised, clean):
        clean_edge = np.abs(clean[1:, :] - clean[:-1, :]).mean()
        denoised_edge = np.abs(denoised[1:, :] - denoised[:-1, :]).mean()
        return denoised_edge / (clean_edge + 1e-10)
    
    for name, result in methods.items():
        mse = np.mean((result - clean)**2)
        psnr = 10 * np.log10(1.0 / mse)
        edge_score = edge_preservation(result, clean)
        
        print(f"{name:<20} {mse:<12.6f} {psnr:<12.2f} {edge_score:<12.4f}")
    
    print("-" * 60)

if __name__ == "__main__":
    compare_denoising_methods()
```

---

## 习题

### 习题1: 理论推导

**题目**：证明Chambolle投影算法的收敛性条件 $\tau \leq \frac{1}{4}$。

**提示**：
1. 考虑梯度算子的范数 $\|\nabla\|_2$
2. 利用压缩映射定理
3. 对于离散前向差分，$\|\nabla\|_2^2 \leq 4$

### 习题2: 算法实现

**题目**：修改Chambolle-Pock算法，实现加权TV去噪：
$$TV_w(u) = \int_\Omega w(x)|\nabla u|dx$$

其中 $w(x)$ 是空间变化的权重函数。

```python
# TODO: 实现加权TV去噪
class WeightedChambollePockROF:
    def __init__(self, weight_map: np.ndarray):
        """
        参数:
            weight_map: 权重图 (H, W)，值越大平滑越强
        """
        self.weight_map = weight_map
        # 完成实现...
```

### 习题3: 实验分析

**题目**：分析ROF模型在彩色图像上的处理策略，比较以下方法：

1. **通道独立处理**：对RGB每个通道分别求解ROF
2. **向量TV**：$\|\nabla u\|_F = \sqrt{\sum_c |\nabla u_c|^2}$
3. **色彩空间转换**：先转到Lab空间再处理

**要求**：
- 实现三种方法
- 在彩色噪声图像上测试
- 分析优缺点

### 习题4: 扩展应用

**题目**：将ROF模型扩展到图像分割问题。

**模型**：
$$\min_u \left\{ \int_\Omega (c_1 - f)^2 u + (c_2 - f)^2 (1-u) dx + \lambda TV(u) \right\}$$

其中 $u \in [0,1]$ 是分割函数，$c_1, c_2$ 是两类的平均灰度。

**提示**：这是Chan-Vese模型与ROF的结合。

### 习题5: 数值稳定性

**题目**：分析ROF求解中可能出现的数值问题，并提出解决方案：

1. **梯度消失**：$|\nabla u| \to 0$ 时的处理
2. **迭代发散**：步长选择不当导致发散
3. **边界效应**：图像边界处的伪影

**要求**：编写测试代码验证你的解决方案。

---

## 附录

### A. 常用公式速查

| 公式 | 表达式 | 说明 |
|------|--------|------|
| ROF能量 | $E(u) = \frac{\lambda}{2}\|f-u\|^2 + TV(u)$ | 目标函数 |
| TV定义 | $TV(u) = \int \|\nabla u\| dx$ | 全变分 |
| 梯度(前向) | $u_x[i,j] = u[i+1,j] - u[i,j]$ | 数值离散 |
| 散度(后向) | $div(p)[i,j] = p_x[i,j] - p_x[i-1,j] + ...$ | 伴随算子 |
| 投影 | $proj_P(p) = p/\max(1, \|p\|)$ | 单位球投影 |

### B. 参数选择经验

| 噪声水平 | 推荐λ | 迭代次数 |
|----------|-------|----------|
| σ = 0.05 | 0.5 - 1.0 | 100-200 |
| σ = 0.10 | 0.1 - 0.3 | 200-300 |
| σ = 0.15 | 0.1 - 0.2 | 200-400 |
| σ = 0.20 | 0.05 - 0.1 | 300-500 |

### C. 参考文献

1. Rudin, L.I., Osher, S., Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D*, 60(1-4), 259-268.

2. Chambolle, A. (2004). An algorithm for total variation minimization and applications. *Journal of Mathematical Imaging and Vision*, 20(1-2), 89-97.

3. Chambolle, A., Pock, T. (2011). A first-order primal-dual algorithm for convex problems with applications to imaging. *Journal of Mathematical Imaging and Vision*, 40(1), 120-145.

4. Chan, T., Shen, J. (2005). *Image Processing and Analysis: Variational, PDE, Wavelet, and Stochastic Methods*. SIAM.
