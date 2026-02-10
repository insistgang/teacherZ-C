"""
优化算法模块

本模块实现了图像处理中常用的优化算法，包括：
- 梯度下降法
- Chambolle 投影法
- 梯度与散度的数值计算

这些算法是 ROF 和 Mumford-Shah 模型的基础组件。
"""

import numpy as np
from scipy import ndimage


def gradient_descent(energy_func, gradient_func, x0, step_size=0.01, 
                     max_iter=1000, tol=1e-6, verbose=False):
    """
    梯度下降优化算法
    
    用于最小化能量函数的基本梯度下降方法。在每一步迭代中，
    沿着能量函数的负梯度方向更新变量。
    
    参数:
    -----------
    energy_func : callable
        能量函数 E(x)，接受一个参数返回标量能量值
    gradient_func : callable
        梯度函数 grad_E(x)，返回能量函数在当前点的梯度
    x0 : ndarray
        初始值
    step_size : float, 可选
        步长（学习率），默认 0.01
    max_iter : int, 可选
        最大迭代次数，默认 1000
    tol : float, 可选
        收敛容差，当能量变化小于此值时停止，默认 1e-6
    verbose : bool, 可选
        是否打印迭代信息，默认 False
        
    返回:
    -----------
    x : ndarray
        优化后的结果
    energy_history : list
        每次迭代的能量值历史记录
        
    示例:
    -----------
    >>> # 最小化 f(x) = (x-2)^2
    >>> energy = lambda x: (x - 2)**2
    >>> grad = lambda x: 2*(x - 2)
    >>> x_opt, history = gradient_descent(energy, grad, np.array([0.0]))
    """
    x = x0.copy()
    energy_history = []
    
    for i in range(max_iter):
        # 计算当前能量值
        energy = energy_func(x)
        energy_history.append(energy)
        
        # 计算梯度
        grad = gradient_func(x)
        
        # 梯度下降更新
        x_new = x - step_size * grad
        
        # 检查收敛
        if i > 0 and abs(energy_history[-1] - energy_history[-2]) < tol:
            if verbose:
                print(f"收敛于迭代 {i}")
            break
            
        x = x_new
        
        if verbose and i % 100 == 0:
            print(f"迭代 {i}: 能量 = {energy:.6f}")
    
    return x, energy_history


def chambolle_projection(f, lambda_param, max_iter=100, tol=1e-4):
    """
    Chambolle 对偶投影算法
    
    这是求解 ROF 模型的高效算法，通过在对偶空间中进行投影来
    避免直接处理非光滑的总变分项。
    
    原始问题: min_u ∫|∇u| + (λ/2)∫(u-f)²
    对偶问题: max_p ∫f·div(p) - (1/2λ)∫(div(p))², s.t. |p| ≤ 1
    
    参数:
    -----------
    f : ndarray
        输入图像（已归一化到 [0, 1]）
    lambda_param : float
        正则化参数，控制平滑程度
        - 较小的 λ: 更强的平滑，更多噪声被去除
        - 较大的 λ: 更接近原图，保留更多细节
    max_iter : int, 可选
        最大迭代次数，默认 100
    tol : float, 可选
        收敛容差，默认 1e-4
        
    返回:
    -----------
    u : ndarray
        去噪后的图像
    p : ndarray
        对偶变量（梯度场）
        
    参考文献:
    -----------
    Chambolle, A. (2004). An algorithm for total variation minimization 
    and applications. Journal of Mathematical Imaging and Vision, 20(1-2), 89-97.
    """
    f = np.asarray(f, dtype=np.float64)
    
    # 初始化对偶变量 p = (px, py)
    # p 是一个向量场，每个像素点有一个二维向量
    px = np.zeros_like(f)
    py = np.zeros_like(f)
    
    # 计算时间步长，Chambolle 建议使用 1/4
    dt = 0.25 / lambda_param if lambda_param > 0 else 0.25
    
    for i in range(max_iter):
        # 保存旧的 p 用于收敛检查
        px_old = px.copy()
        py_old = py.copy()
        
        # 计算 div(p) - 对偶变量的散度
        div_p = compute_divergence(px, py)
        
        # 计算梯度: ∇(div(p) - f/λ)
        # 注意: 这里实际上是 ∇(div(p) - λ*f) 的对偶形式
        grad_x, grad_y = compute_gradient(div_p - f * lambda_param)
        
        # 更新对偶变量（梯度上升）
        px = px + dt * grad_x
        py = py + dt * grad_y
        
        # 投影到约束集 |p| ≤ 1
        # 计算 p 的模长
        norm_p = np.sqrt(px**2 + py**2)
        norm_p = np.maximum(norm_p, 1.0)  # 避免除以小于1的数
        
        # 投影: p = p / max(|p|, 1)
        px = px / norm_p
        py = py / norm_p
        
        # 检查收敛
        if i > 0 and i % 10 == 0:
            change = np.sqrt(np.sum((px - px_old)**2 + (py - py_old)**2))
            if change < tol:
                break
    
    # 计算最终结果: u = f - (1/λ) * div(p)
    div_p = compute_divergence(px, py)
    u = f - div_p / lambda_param
    
    return u, np.stack([px, py], axis=-1)


def compute_gradient(u):
    """
    计算标量场的梯度（中心差分）
    
    使用中心差分近似计算图像的梯度：
    ∂u/∂x ≈ (u(x+1,y) - u(x-1,y)) / 2
    ∂u/∂y ≈ (u(x,y+1) - u(x,y-1)) / 2
    
    边界采用前向或后向差分。
    
    参数:
    -----------
    u : ndarray
        输入的二维标量场（如灰度图像）
        
    返回:
    -----------
    grad_x : ndarray
        x 方向的梯度（水平方向）
    grad_y : ndarray
        y 方向的梯度（垂直方向）
        
    示例:
    -----------
    >>> u = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
    >>> gx, gy = compute_gradient(u)
    >>> # gx 应该在水平方向为常数 1
    """
    u = np.asarray(u, dtype=np.float64)
    
    # 初始化梯度数组
    grad_x = np.zeros_like(u)
    grad_y = np.zeros_like(u)
    
    # 内部点使用中心差分
    # x 方向梯度 (水平)
    grad_x[1:-1, :] = (u[2:, :] - u[:-2, :]) / 2.0
    # y 方向梯度 (垂直)
    grad_y[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / 2.0
    
    # 边界使用前向或后向差分
    # x 方向边界
    grad_x[0, :] = u[1, :] - u[0, :]      # 前向差分（上边界）
    grad_x[-1, :] = u[-1, :] - u[-2, :]   # 后向差分（下边界）
    
    # y 方向边界
    grad_y[:, 0] = u[:, 1] - u[:, 0]      # 前向差分（左边界）
    grad_y[:, -1] = u[:, -1] - u[:, -2]   # 后向差分（右边界）
    
    return grad_x, grad_y


def compute_divergence(px, py):
    """
    计算向量场的散度
    
    散度是梯度的伴随算子（adjoint operator），满足：
    <∇u, p> = -<u, div(p)>
    
    其中 <.,.> 表示内积。这个性质在推导对偶问题中很重要。
    
    参数:
    -----------
    px : ndarray
        向量场的 x 分量
    py : ndarray
        向量场的 y 分量
        
    返回:
    -----------
    div : ndarray
        向量场的散度
        
    示例:
    -----------
    >>> # 如果 p = ∇u，则 div(p) = Δu（拉普拉斯算子）
    >>> u = np.random.randn(10, 10)
    >>> px, py = compute_gradient(u)
    >>> div_p = compute_divergence(px, py)
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    
    div = np.zeros_like(px)
    
    # 内部点使用中心差分
    # ∂px/∂x
    div[1:-1, :] += (px[2:, :] - px[:-2, :]) / 2.0
    # ∂py/∂y
    div[:, 1:-1] += (py[:, 2:] - py[:, :-2]) / 2.0
    
    # 边界处理
    # x 方向边界
    div[0, :] += px[0, :]           # 前向差分
    div[-1, :] -= px[-1, :]         # 后向差分
    
    # y 方向边界
    div[:, 0] += py[:, 0]           # 前向差分
    div[:, -1] -= py[:, -1]         # 后向差分
    
    return div


def compute_laplacian(u):
    """
    计算拉普拉斯算子 Δu = ∂²u/∂x² + ∂²u/∂y²
    
    使用五点差分格式：
    Δu(i,j) ≈ [u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1) - 4*u(i,j)] / h²
    
    参数:
    -----------
    u : ndarray
        输入的二维标量场
        
    返回:
    -----------
    laplacian : ndarray
        拉普拉斯算子结果
    """
    u = np.asarray(u, dtype=np.float64)
    
    laplacian = np.zeros_like(u)
    
    # 内部点
    laplacian[1:-1, 1:-1] = (
        u[2:, 1:-1] + u[:-2, 1:-1] + 
        u[1:-1, 2:] + u[1:-1, :-2] - 
        4 * u[1:-1, 1:-1]
    )
    
    return laplacian


def line_search(f, x, grad, direction, alpha=0.3, beta=0.8, max_iter=20):
    """
    回溯线搜索（Backtracking Line Search）
    
    用于确定合适的步长，满足 Armijo 条件：
    f(x + t*d) ≤ f(x) + α*t*<∇f(x), d>
    
    参数:
    -----------
    f : callable
        目标函数
    x : ndarray
        当前点
    grad : ndarray
        当前梯度
    direction : ndarray
        搜索方向
    alpha : float
        Armijo 参数 (0 < α < 0.5)
    beta : float
        收缩因子 (0 < β < 1)
    max_iter : int
        最大迭代次数
        
    返回:
    -----------
    t : float
        合适的步长
    """
    t = 1.0
    fx = f(x)
    grad_dot_dir = np.sum(grad * direction)
    
    for _ in range(max_iter):
        x_new = x + t * direction
        fx_new = f(x_new)
        
        # Armijo 条件
        if fx_new <= fx + alpha * t * grad_dot_dir:
            return t
        
        t *= beta
    
    return t


if __name__ == "__main__":
    # 简单的测试
    print("测试梯度计算...")
    u = np.array([[0.0, 1.0, 2.0], 
                  [0.0, 1.0, 2.0], 
                  [0.0, 1.0, 2.0]])
    gx, gy = compute_gradient(u)
    print("输入图像:")
    print(u)
    print("x 方向梯度:")
    print(gx)
    print("y 方向梯度:")
    print(gy)
    
    print("\n测试散度计算...")
    div = compute_divergence(gx, gy)
    print("散度（近似拉普拉斯）:")
    print(div)
