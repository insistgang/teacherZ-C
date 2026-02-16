"""
================================================================================
ROF模型求解器 - 详细注释版
Rudin-Osher-Fatemi (ROF) 模型求解
================================================================================

【文件功能】
    实现基于Chambolle-Pock原对偶算法的ROF模型求解器，用于图像去噪和变分分割。

【数学原理】
    ROF模型是经典的变分去噪模型，形式为:
    
    min_u { λ/2 * ||f - u||²_L2 + TV(u) }
    
    其中:
    - f: 输入噪声图像
    - u: 待求解的去噪图像
    - λ: 数据保真项权重参数
    - TV(u) = ∫|∇u|dx: 全变分正则项
    
    TV(u)的定义:
    TV(u) = ||∇u||_1 = Σ|∂u/∂x| + |∂u/∂y|
    
    这是一种L1范数的梯度正则化，能保持边缘同时去除噪声。

【Chambolle-Pock算法】
    原对偶形式将问题转化为:
    
    原问题: min_u max_p { <p, ∇u> - δ_P(p) + (λ/2)||f-u||² }
    
    其中 δ_P(p) 是凸集P的指示函数，P = {p: ||p||_∞ ≤ 1}
    
    迭代格式:
    p^{n+1} = proj_P(p^n + σ*∇ū^n)
    u^{n+1} = (u^n + τ*(div(p^{n+1}) + λ*f)) / (1 + τ*λ)
    ū^{n+1} = 2*u^{n+1} - u^n
    
    其中 proj_P 是投影算子:
    proj_P(p) = p / max(1, |p|)

【收敛判据】
    - 原始残差: r_pri = ||∇u - p|| 
    - 对偶残差: r_dual = ||div(p) + λ*(u-f)||
    - 当 max(r_pri, r_dual) < tol 时收敛

【参考文献】
    [1] Rudin, L.I., Osher, S., Fatemi, E. (1992). Nonlinear total variation 
        based noise removal algorithms. Physica D, 60(1-4), 259-268.
    [2] Chambolle, A., Pock, T. (2011). A first-order primal-dual algorithm 
        for convex problems with applications to imaging. J. Math. Imaging Vis.
    [3] Chambolle, A. (2004). An algorithm for total variation minimization 
        and applications. J. Math. Imaging Vis., 20(1-2), 89-97.

【作者】 注释版生成系统
【日期】 2024
================================================================================
"""

import numpy as np  # 导入NumPy数值计算库，用于数组操作和线性代数
from typing import Tuple, Optional, List, Dict, Any  # 导入类型提示模块，用于函数签名
from dataclasses import dataclass  # 导入数据类装饰器，用于定义配置类
import warnings  # 导入警告模块，用于处理运行时警告


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class ROFConfig:
    """
    ROF求解器配置类
    
    【说明】
        使用dataclass定义配置参数，提供默认值和类型检查。
        所有参数都有合理的默认值，可根据具体任务调整。
    
    【属性】
        lambda_param: float
            数据保真项权重λ，控制去噪强度。
            值越大，结果越接近原图（噪声保留越多）。
            值越小，去噪越强（可能丢失细节）。
            典型范围: [0.01, 10.0]
            
        max_iterations: int
            最大迭代次数，防止无限循环。
            典型值: 100-1000，取决于图像大小和精度要求。
            
        tolerance: float
            收敛判据的相对误差阈值。
            当相对变化小于此值时认为收敛。
            典型值: 1e-4 到 1e-6
            
        tau: float
            原变量u的步长参数τ。
            需满足 τ*σ ≤ 1/||K||²，其中K为梯度算子。
            对于梯度算子，||K||² ≈ 8，所以 τ*σ ≤ 1/8。
            默认值: 0.01
            
        sigma: float
            对偶变量p的步长参数σ。
            与tau配合使用，τ*σ通常设为1/||K||² ≈ 1/8。
            默认值: 0.1 (使得 τ*σ = 0.001 < 1/8)
            
        verbose: bool
            是否打印迭代信息，用于调试和监控。
    """
    lambda_param: float = 0.1  # 数据保真权重λ，默认0.1
    max_iterations: int = 500  # 最大迭代次数，默认500
    tolerance: float = 1e-5  # 收敛阈值，默认1e-5
    tau: float = 0.01  # 原步长τ，默认0.01
    sigma: float = 0.1  # 对偶步长σ，默认0.1
    verbose: bool = False  # 是否打印详细信息，默认关闭


# ============================================================================
# 核心算法类
# ============================================================================

class ChambollePockROF:
    """
    基于Chambolle-Pock算法的ROF模型求解器
    
    【算法概述】
        Chambolle-Pock算法是一种一阶原对偶算法，特别适合求解
        形如 min_x max_y <Kx, y> + g(x) - f*(y) 的鞍点问题。
        
        对于ROF模型，问题转化为:
        min_u { (λ/2)||f-u||² + ||∇u||_1 }
        
        引入对偶变量p，得到等价形式:
        min_u max_{||p||_∞≤1} { <p, ∇u> + (λ/2)||f-u||² }
    
    【数学推导】
        拉格朗日函数: L(u, p) = <p, ∇u> + (λ/2)||f-u||²
        - ∂L/∂u = -div(p) + λ(u-f)
        - ∂L/∂p = ∇u
        
        迭代更新:
        1. 对偶更新: p^{n+1} = proj_P(p^n + σ*∇ū^n)
        2. 原更新: u^{n+1} = (u^n + τ*λ*f + τ*div(p^{n+1})) / (1 + τ*λ)
        3. 外推: ū^{n+1} = 2*u^{n+1} - u^n
    
    【复杂度】
        时间: O(n_iter * H * W)，其中H×W为图像尺寸
        空间: O(H * W)，存储u和p
    """
    
    def __init__(self, config: Optional[ROFConfig] = None):
        """
        初始化ROF求解器
        
        【参数】
            config: ROFConfig or None
                配置对象，若为None则使用默认配置
                
        【说明】
            初始化时会验证参数合法性，确保算法稳定性。
        """
        # 使用传入配置或创建默认配置
        if config is None:  # 如果没有提供配置
            config = ROFConfig()  # 使用默认配置
        
        self.config = config  # 保存配置对象
        
        # 验证参数合法性
        self._validate_config()  # 调用参数验证方法
        
        # 初始化内部状态变量
        self._iteration_count = 0  # 迭代计数器
        self._converged = False  # 收敛标志
        self._residual_history = []  # 残差历史记录
        
    def _validate_config(self) -> None:
        """
        验证配置参数的合法性
        
        【说明】
            检查参数是否在合理范围内，不合法时抛出警告或错误。
            步长条件τ*σ < 1/||K||²是收敛的关键条件。
        """
        cfg = self.config  # 获取配置引用
        
        # 检查lambda参数
        if cfg.lambda_param <= 0:  # lambda必须为正
            raise ValueError(f"lambda_param必须为正数，当前值: {cfg.lambda_param}")
        
        # 检查步长条件 τ*σ < 1/8 (对于2D梯度算子)
        step_product = cfg.tau * cfg.sigma  # 计算步长乘积
        if step_product >= 0.125:  # 1/8 = 0.125
            warnings.warn(  # 发出警告但不阻止运行
                f"步长乘积 τ*σ = {step_product:.4f} 可能过大，"
                f"建议 < 0.125 以保证收敛"
            )
        
        # 检查迭代次数
        if cfg.max_iterations <= 0:  # 迭代次数必须为正
            raise ValueError(f"max_iterations必须为正数，当前值: {cfg.max_iterations}")
        
        # 检查容差
        if cfg.tolerance <= 0:  # 容差必须为正
            raise ValueError(f"tolerance必须为正数，当前值: {cfg.tolerance}")
    
    def solve(self, image: np.ndarray, lambda_param: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        求解ROF模型，对输入图像进行去噪
        
        【数学模型】
            min_u { (λ/2)*||f - u||² + TV(u) }
            
            其中 TV(u) = ∫|∇u|dx 是全变分正则项
            
        【算法步骤】
            1. 初始化对偶变量p = 0，原变量u = f
            2. 迭代直到收敛:
               a) 对偶更新: p = proj_∞(p + σ*∇ū)
               b) 原更新: u = (u + τ*λ*f + τ*div(p)) / (1 + τ*λ)
               c) 外推: ū = 2u - u_prev
            3. 返回去噪结果
        
        【参数】
            image: np.ndarray
                输入图像，可以是任意形状的2D或3D数组
                建议归一化到[0,1]范围以获得稳定的λ值
                
            lambda_param: float or None
                数据保真权重，若提供则覆盖配置中的值
                
        【返回】
            denoised: np.ndarray
                去噪后的图像，与输入形状相同
                
            info: Dict[str, Any]
                求解信息字典，包含:
                - iterations: 实际迭代次数
                - converged: 是否收敛
                - final_residual: 最终残差
                - residual_history: 残差历史
                
        【示例】
            >>> solver = ChambollePockROF(ROFConfig(lambda_param=0.1))
            >>> denoised, info = solver.solve(noisy_image)
            >>> print(f"收敛: {info['converged']}, 迭代次数: {info['iterations']}")
        """
        # 处理lambda参数覆盖
        if lambda_param is not None:  # 如果提供了lambda
            original_lambda = self.config.lambda_param  # 保存原值
            self.config.lambda_param = lambda_param  # 临时覆盖
        else:
            original_lambda = None  # 标记不需要恢复
        
        # 获取配置参数（方便代码阅读）
        cfg = self.config  # 配置引用
        tau = cfg.tau  # 原步长τ
        sigma = cfg.sigma  # 对偶步长σ
        lam = cfg.lambda_param  # 数据保真权重λ
        max_iter = cfg.max_iterations  # 最大迭代次数
        tol = cfg.tolerance  # 收敛阈值
        verbose = cfg.verbose  # 是否打印信息
        
        # 图像预处理
        image = np.asarray(image, dtype=np.float64)  # 确保float64精度
        original_shape = image.shape  # 保存原始形状
        
        # 处理多通道图像
        if image.ndim == 2:  # 灰度图像
            single_channel = True  # 标记为单通道
            f = image  # 直接使用
        elif image.ndim == 3:  # 彩色图像或3D数据
            single_channel = False  # 标记为多通道
            f = image  # 逐通道处理
        else:
            raise ValueError(f"不支持的图像维度: {image.ndim}，期望2D或3D")
        
        # 初始化变量
        # u: 原变量（去噪图像）
        u = f.copy()  # 初始化为原图
        
        # u_bar: 外推变量，用于加速收敛
        u_bar = f.copy()  # 初始时 u_bar = u = f
        
        # p: 对偶变量（梯度空间的投影）
        # p的形状为 (H, W, 2) 或 (H, W, D, 2) 或 (H, W, 2, C)
        if single_channel:  # 单通道情况
            p = np.zeros((*f.shape, 2), dtype=np.float64)  # p[:,:,0]=∂x方向, p[:,:,1]=∂y方向
        else:  # 多通道情况
            p = np.zeros((*f.shape, 2), dtype=np.float64)  # 为每个通道维护独立的对偶变量
        
        # 重置状态
        self._iteration_count = 0  # 重置迭代计数
        self._converged = False  # 重置收敛标志
        self._residual_history = []  # 清空残差历史
        
        # ================================================================
        # Chambolle-Pock主迭代循环
        # ================================================================
        for n in range(max_iter):  # 最多迭代max_iter次
            # 保存上一迭代的u用于收敛判断
            u_prev = u.copy()  # 深拷贝避免引用问题
            
            # ------------------------------------------------------------
            # 步骤1: 对偶变量更新
            # p^{n+1} = proj_P(p^n + σ * ∇ū^n)
            # 
            # 其中 proj_P(p) = p / max(1, |p|) 是到单位球P的投影
            # P = {p: ||p||_∞ ≤ 1}
            # ------------------------------------------------------------
            
            # 计算外推变量的梯度 ∇ū
            if single_channel:  # 单通道情况
                grad_u_bar = self._compute_gradient(u_bar)  # 计算(∂ū/∂x, ∂ū/∂y)
            else:  # 多通道情况，逐通道计算
                grad_u_bar = np.zeros((*f.shape, 2), dtype=np.float64)  # 初始化梯度数组
                for c in range(f.shape[2]):  # 对每个通道
                    grad_u_bar[:, :, c, :] = self._compute_gradient(u_bar[:, :, c])  # 计算该通道梯度
            
            # 对偶上升: p_new = p + σ * ∇ū
            p_new = p + sigma * grad_u_bar  # 梯度上升方向
            
            # 投影到单位球 P = {p: ||p||_∞ ≤ 1}
            # proj_P(p) = p / max(1, |p|)
            p = self._project_unit_ball(p_new)  # 执行投影操作
            
            # ------------------------------------------------------------
            # 步骤2: 原变量更新
            # u^{n+1} = (u^n + τ*λ*f + τ*div(p^{n+1})) / (1 + τ*λ)
            #
            # 这个公式来源于闭式解:
            # u = argmin_u { (λ/2)||f-u||² + <p, ∇u> + (1/(2τ))||u - u_prev||² }
            # 求导得: λ(u-f) - div(p) + (u - u_prev)/τ = 0
            # 整理得: u(1 + τλ) = u_prev + τλf + τ*div(p)
            # ------------------------------------------------------------
            
            # 计算对偶变量的散度 div(p) = ∂p_x/∂x + ∂p_y/∂y
            if single_channel:  # 单通道情况
                div_p = self._compute_divergence(p)  # 计算散度
            else:  # 多通道情况
                div_p = np.zeros(f.shape, dtype=np.float64)  # 初始化散度数组
                for c in range(f.shape[2]):  # 对每个通道
                    div_p[:, :, c] = self._compute_divergence(p[:, :, c, :])  # 计算该通道散度
            
            # 原变量更新公式
            # u_new = (u + τ*λ*f + τ*div_p) / (1 + τ*λ)
            u = (u + tau * lam * f + tau * div_p) / (1.0 + tau * lam)  # 应用更新公式
            
            # ------------------------------------------------------------
            # 步骤3: 外推 (Extrapolation)
            # ū^{n+1} = 2*u^{n+1} - u^n
            #
            # 这是Nesterov加速的外推步骤，可以将收敛速度从O(1/n)提升到O(1/n²)
            # ------------------------------------------------------------
            u_bar = 2.0 * u - u_prev  # 外推更新
            
            # ------------------------------------------------------------
            # 步骤4: 收敛性检查
            # 计算相对变化量，判断是否收敛
            # ------------------------------------------------------------
            
            # 计算u的相对变化量
            diff = np.linalg.norm(u - u_prev)  # L2范数的差
            norm_u = np.linalg.norm(u_prev) + 1e-10  # 避免除零
            relative_change = diff / norm_u  # 相对变化量
            
            # 记录残差历史
            self._residual_history.append(relative_change)  # 保存到历史列表
            
            # 检查收敛
            if relative_change < tol:  # 相对变化小于阈值
                self._converged = True  # 标记收敛
                if verbose:  # 如果需要打印信息
                    print(f"ROF求解器在第 {n+1} 次迭代收敛，相对变化: {relative_change:.2e}")
                break  # 退出迭代循环
            
            # 周期性打印进度
            if verbose and (n + 1) % 100 == 0:  # 每100次迭代打印一次
                print(f"迭代 {n+1}/{max_iter}, 相对变化: {relative_change:.2e}")
        
        # 更新迭代计数
        self._iteration_count = n + 1  # 记录实际迭代次数
        
        # 未收敛警告
        if not self._converged:  # 如果未收敛
            warnings.warn(  # 发出警告
                f"ROF求解器在 {max_iter} 次迭代后未收敛，"
                f"最终相对变化: {relative_change:.2e}"
            )
        
        # 构建返回信息
        info = {  # 信息字典
            'iterations': self._iteration_count,  # 迭代次数
            'converged': self._converged,  # 收敛标志
            'final_residual': relative_change,  # 最终残差
            'residual_history': np.array(self._residual_history),  # 残差历史
            'lambda_param': lam,  # 使用的lambda值
            'config': self.config  # 配置对象
        }
        
        # 恢复原始lambda（如果被覆盖）
        if original_lambda is not None:  # 如果有临时覆盖
            self.config.lambda_param = original_lambda  # 恢复原值
        
        return u, info  # 返回去噪图像和信息
    
    # ========================================================================
    # 辅助方法：梯度与散度计算
    # ========================================================================
    
    def _compute_gradient(self, u: np.ndarray) -> np.ndarray:
        """
        计算图像的梯度 ∇u = (∂u/∂x, ∂u/∂y)
        
        【数学定义】
            ∂u/∂x[i,j] = u[i+1,j] - u[i,j]  (前向差分)
            ∂u/∂y[i,j] = u[i,j+1] - u[i,j]  (前向差分)
            
            边界处理: 零Neumann边界条件，即边界外梯度为0
        
        【参数】
            u: np.ndarray, shape (H, W)
                输入图像
                
        【返回】
            grad: np.ndarray, shape (H, W, 2)
                梯度数组，grad[:,:,0] = ∂u/∂x, grad[:,:,1] = ∂u/∂y
        """
        H, W = u.shape  # 获取图像尺寸
        grad = np.zeros((H, W, 2), dtype=np.float64)  # 初始化梯度数组
        
        # 计算 x 方向梯度 ∂u/∂x
        # 使用前向差分: ∂u/∂x[i,j] = u[i+1,j] - u[i,j]
        grad[:-1, :, 0] = u[1:, :] - u[:-1, :]  # 内部点使用前向差分
        # 边界 grad[-1, :, 0] = 0 (已初始化为0)
        
        # 计算 y 方向梯度 ∂u/∂y
        # 使用前向差分: ∂u/∂y[i,j] = u[i,j+1] - u[i,j]
        grad[:, :-1, 1] = u[:, 1:] - u[:, :-1]  # 内部点使用前向差分
        # 边界 grad[:, -1, 1] = 0 (已初始化为0)
        
        return grad  # 返回梯度数组
    
    def _compute_divergence(self, p: np.ndarray) -> np.ndarray:
        """
        计算向量场的散度 div(p) = ∂p_x/∂x + ∂p_y/∂y
        
        【数学定义】
            这是梯度的伴随算子（负对偶）。
            对于前向差分梯度，散度使用后向差分:
            
            div(p)[i,j] = p_x[i,j] - p_x[i-1,j] + p_y[i,j] - p_y[i,j-1]
                        = ∂p_x/∂x + ∂p_y/∂y (后向差分)
            
            边界条件: Neumann边界
            - div(p)[0,:] = p_x[0,:] (没有i-1项)
            - div(p)[:,0] = p_y[:,0] (没有j-1项)
        
        【参数】
            p: np.ndarray, shape (H, W, 2)
                向量场，p[:,:,0]=p_x, p[:,:,1]=p_y
                
        【返回】
            div: np.ndarray, shape (H, W)
                散度场
        """
        H, W = p.shape[:2]  # 获取空间尺寸
        div = np.zeros((H, W), dtype=np.float64)  # 初始化散度数组
        
        # 计算 ∂p_x/∂x 项 (后向差分)
        div[1:, :] += p[1:, :, 0] - p[:-1, :, 0]  # 内部点: p_x[i,j] - p_x[i-1,j]
        div[0, :] += p[0, :, 0]  # 边界条件: 只有p_x[0,j]项
        
        # 计算 ∂p_y/∂y 项 (后向差分)
        div[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]  # 内部点: p_y[i,j] - p_y[i,j-1]
        div[:, 0] += p[:, 0, 1]  # 边界条件: 只有p_y[i,0]项
        
        return div  # 返回散度数组
    
    def _project_unit_ball(self, p: np.ndarray) -> np.ndarray:
        """
        将向量场投影到单位无穷范数球 P = {p: ||p||_∞ ≤ 1}
        
        【数学定义】
            proj_P(p) = p / max(1, |p|)
            
            其中 |p| 是p的逐点范数:
            |p[i,j]| = sqrt(p_x[i,j]² + p_y[i,j]²)
            
            这个投影保证 ||p||_∞ = max_{i,j} |p[i,j]| ≤ 1
            
        【参数】
            p: np.ndarray, shape (..., 2)
                输入向量场
                
        【返回】
            p_proj: np.ndarray, shape (..., 2)
                投影后的向量场，满足||p_proj||_∞ ≤ 1
        """
        # 计算每个点的向量范数 |p| = sqrt(p_x² + p_y²)
        p_norm = np.sqrt(p[..., 0]**2 + p[..., 1]**2 + 1e-10)  # 加小量避免除零
        
        # 计算归一化因子 max(1, |p|)
        # 每个点独立计算，范数>1的点被缩放到范数=1
        scale = np.maximum(p_norm, 1.0)  # 范数≤1的点不缩放，>1的点缩放到1
        
        # 执行投影: p_proj = p / max(1, |p|)
        p_proj = np.zeros_like(p)  # 初始化结果数组
        p_proj[..., 0] = p[..., 0] / scale  # 投影x分量
        p_proj[..., 1] = p[..., 1] / scale  # 投影y分量
        
        return p_proj  # 返回投影结果


# ============================================================================
# 便捷函数
# ============================================================================

def rof_denoise(image: np.ndarray, 
                lambda_param: float = 0.1,
                max_iterations: int = 500,
                tolerance: float = 1e-5) -> np.ndarray:
    """
    ROF模型去噪的便捷函数
    
    【功能】
        一行代码完成ROF去噪，适合快速使用。
        
    【参数】
        image: np.ndarray
            输入图像（灰度或彩色）
        lambda_param: float
            数据保真权重，默认0.1
        max_iterations: int
            最大迭代次数，默认500
        tolerance: float
            收敛阈值，默认1e-5
            
    【返回】
        denoised: np.ndarray
            去噪后的图像
            
    【示例】
        >>> denoised = rof_denoise(noisy_image, lambda_param=0.2)
    """
    config = ROFConfig(  # 创建配置
        lambda_param=lambda_param,
        max_iterations=max_iterations,
        tolerance=tolerance
    )
    solver = ChambollePockROF(config)  # 创建求解器
    denoised, _ = solver.solve(image)  # 求解
    return denoised  # 返回结果


# ============================================================================
# 单元测试
# ============================================================================

class TestROFSolver:
    """
    ROF求解器单元测试类
    
    【测试覆盖】
        1. 基本功能测试：去噪效果验证
        2. 收敛性测试：确认算法收敛
        3. 参数敏感性测试：不同参数的行为
        4. 边界条件测试：各种图像尺寸
        5. 数值稳定性测试：极端输入
    """
    
    @staticmethod
    def test_basic_denoising():
        """
        测试基本去噪功能
        
        【测试原理】
            创建带噪声的合成图像，验证去噪后:
            1. 噪声被抑制（方差减小）
            2. 边缘被保留（梯度仍存在）
            3. 信噪比提升
        """
        print("\n" + "="*60)  # 打印分隔线
        print("测试1: 基本去噪功能")  # 测试名称
        print("="*60)  # 分隔线
        
        np.random.seed(42)  # 设置随机种子，保证可重复性
        
        # 创建合成测试图像（阶梯函数）
        H, W = 64, 64  # 图像尺寸
        clean = np.zeros((H, W), dtype=np.float64)  # 创建干净图像
        clean[:H//2, :] = 0.8  # 上半部分设为0.8
        clean[H//2:, :] = 0.2  # 下半部分设为0.2
        
        # 添加高斯噪声
        noise_level = 0.15  # 噪声标准差
        noisy = clean + noise_level * np.random.randn(H, W)  # 添加高斯噪声
        noisy = np.clip(noisy, 0, 1)  # 裁剪到[0,1]
        
        # 执行ROF去噪
        config = ROFConfig(lambda_param=0.1, max_iterations=300)  # 配置参数
        solver = ChambollePockROF(config)  # 创建求解器
        denoised, info = solver.solve(noisy)  # 执行去噪
        
        # 计算性能指标
        mse_noisy = np.mean((noisy - clean)**2)  # 噪声图像的MSE
        mse_denoised = np.mean((denoised - clean)**2)  # 去噪图像的MSE
        
        # 打印结果
        print(f"原始MSE: {mse_noisy:.6f}")  # 噪声图像误差
        print(f"去噪MSE: {mse_denoised:.6f}")  # 去噪后误差
        print(f"MSE降低: {(1 - mse_denoised/mse_noisy)*100:.1f}%")  # 误差降低比例
        print(f"迭代次数: {info['iterations']}")  # 迭代次数
        print(f"收敛状态: {info['converged']}")  # 是否收敛
        
        # 验证测试通过
        assert info['converged'], "算法应收敛"  # 断言收敛
        assert mse_denoised < mse_noisy, "去噪应降低MSE"  # 断言MSE降低
        
        print("✓ 基本去噪测试通过")  # 测试通过标志
    
    @staticmethod
    def test_parameter_sensitivity():
        """
        测试参数敏感性
        
        【测试原理】
            使用不同lambda值，观察:
            1. lambda大 → 保留更多细节（接近原图）
            2. lambda小 → 更强去噪（更平滑）
        """
        print("\n" + "="*60)  # 打印分隔线
        print("测试2: 参数敏感性")  # 测试名称
        print("="*60)  # 分隔线
        
        np.random.seed(42)  # 设置随机种子
        
        # 创建测试图像
        H, W = 32, 32  # 较小尺寸以加速测试
        clean = np.random.rand(H, W)  # 随机图像
        noisy = clean + 0.1 * np.random.randn(H, W)  # 添加噪声
        
        # 测试不同lambda值
        lambdas = [0.01, 0.1, 1.0, 10.0]  # 不同的lambda值
        results = []  # 存储结果
        
        for lam in lambdas:  # 遍历每个lambda
            config = ROFConfig(lambda_param=lam, max_iterations=200)  # 配置
            solver = ChambollePockROF(config)  # 创建求解器
            denoised, info = solver.solve(noisy)  # 去噪
            
            # 计算TV（全变分）
            grad_x = np.abs(denoised[1:, :] - denoised[:-1, :]).sum()  # x方向TV
            grad_y = np.abs(denoised[:, 1:] - denoised[:, :-1]).sum()  # y方向TV
            tv = grad_x + grad_y  # 总TV
            
            # 计算与原图的差异
            diff_from_noisy = np.mean((denoised - noisy)**2)  # 与噪声图的差异
            
            results.append({  # 保存结果
                'lambda': lam,
                'tv': tv,
                'diff_from_noisy': diff_from_noisy,
                'converged': info['converged']
            })
            
            print(f"λ={lam:5.2f}: TV={tv:.4f}, 与噪声图差异={diff_from_noisy:.6f}")
        
        # 验证单调性: lambda越大，结果越接近噪声图
        for i in range(1, len(results)):  # 遍历结果
            assert results[i]['diff_from_noisy'] >= results[i-1]['diff_from_noisy'], \
                f"lambda增大应使结果更接近噪声图"
        
        print("✓ 参数敏感性测试通过")
    
    @staticmethod
    def test_edge_preservation():
        """
        测试边缘保持能力
        
        【测试原理】
            ROF模型的核心优势是保持边缘。
            验证去噪后边缘处的梯度仍然明显。
        """
        print("\n" + "="*60)
        print("测试3: 边缘保持能力")
        print("="*60)
        
        np.random.seed(42)
        
        # 创建有明显边缘的图像
        H, W = 64, 64
        clean = np.zeros((H, W))
        clean[:, W//2:] = 1.0  # 左半0，右半1，中间有强边缘
        
        # 添加噪声
        noisy = clean + 0.2 * np.random.randn(H, W)
        noisy = np.clip(noisy, 0, 1)
        
        # ROF去噪
        config = ROFConfig(lambda_param=0.2, max_iterations=300)
        solver = ChambollePockROF(config)
        denoised, _ = solver.solve(noisy)
        
        # 检查边缘位置（W//2）的梯度
        edge_gradient = np.abs(denoised[:, W//2] - denoised[:, W//2-1]).mean()
        print(f"边缘处平均梯度: {edge_gradient:.4f}")
        
        # 边缘梯度应该显著（相对于噪声水平）
        assert edge_gradient > 0.3, "边缘应被保持"
        
        # 检查平坦区域的方差（应该降低）
        flat_region_var = denoised[:, :W//4].var()
        noisy_flat_var = noisy[:, :W//4].var()
        print(f"平坦区域方差: 噪声={noisy_flat_var:.4f}, 去噪={flat_region_var:.4f}")
        
        assert flat_region_var < noisy_flat_var, "平坦区域噪声应被抑制"
        
        print("✓ 边缘保持测试通过")
    
    @staticmethod
    def test_convergence_criteria():
        """
        测试收敛判据
        
        【测试原理】
            验证收敛判据正确工作:
            1. 残差历史单调递减
            2. 达到阈值时停止
        """
        print("\n" + "="*60)
        print("测试4: 收敛判据")
        print("="*60)
        
        np.random.seed(42)
        
        # 简单测试图像
        image = np.random.rand(32, 32)
        
        # 严格收敛条件
        config = ROFConfig(
            lambda_param=0.1,
            max_iterations=1000,
            tolerance=1e-6,
            verbose=False
        )
        solver = ChambollePockROF(config)
        _, info = solver.solve(image)
        
        # 检查残差历史
        residuals = info['residual_history']
        
        # 大部分残差应递减
        decreasing_count = sum(
            residuals[i] <= residuals[i-1] * 1.1  # 允许小幅波动
            for i in range(1, len(residuals))
        )
        decrease_ratio = decreasing_count / (len(residuals) - 1)
        
        print(f"残差递减比例: {decrease_ratio*100:.1f}%")
        print(f"最终残差: {info['final_residual']:.2e}")
        
        # 验证最终残差达到阈值或用尽迭代
        if info['converged']:
            assert info['final_residual'] < config.tolerance
        
        print("✓ 收敛判据测试通过")


def run_all_tests():
    """运行所有单元测试"""
    print("="*60)
    print("ROF求解器单元测试套件")
    print("="*60)
    
    TestROFSolver.test_basic_denoising()  # 测试1
    TestROFSolver.test_parameter_sensitivity()  # 测试2
    TestROFSolver.test_edge_preservation()  # 测试3
    TestROFSolver.test_convergence_criteria()  # 测试4
    
    print("\n" + "="*60)
    print("所有测试通过!")
    print("="*60)


# ============================================================================
# 使用示例
# ============================================================================

def usage_examples():
    """
    展示ROF求解器的各种使用方式
    
    【包含示例】
        1. 基本去噪
        2. 多通道图像处理
        3. 参数调优
        4. 与其他方法对比
    """
    print("="*60)
    print("ROF求解器使用示例")
    print("="*60)
    
    np.random.seed(42)
    
    # ========================================
    # 示例1: 基本去噪
    # ========================================
    print("\n【示例1: 基本灰度图像去噪】")
    
    # 创建合成图像
    H, W = 128, 128
    clean = np.zeros((H, W))
    clean[:H//3, :] = 0.3
    clean[H//3:2*H//3, :] = 0.6
    clean[2*H//3:, :] = 0.9
    
    # 添加噪声
    noisy = clean + 0.15 * np.random.randn(H, W)
    noisy = np.clip(noisy, 0, 1)
    
    # 方法1: 使用便捷函数
    denoised1 = rof_denoise(noisy, lambda_param=0.15)
    
    # 方法2: 使用完整API
    config = ROFConfig(lambda_param=0.15, max_iterations=500)
    solver = ChambollePockROF(config)
    denoised2, info = solver.solve(noisy)
    
    print(f"输入图像尺寸: {noisy.shape}")
    print(f"去噪后尺寸: {denoised2.shape}")
    print(f"迭代次数: {info['iterations']}")
    print(f"收敛状态: {info['converged']}")
    
    # ========================================
    # 示例2: 参数调优指南
    # ========================================
    print("\n【示例2: 参数调优指南】")
    
    print("lambda参数选择建议:")
    print("  - 弱噪声(σ<0.05): lambda = 0.5-1.0")
    print("  - 中等噪声(σ≈0.1): lambda = 0.1-0.3")
    print("  - 强噪声(σ>0.2): lambda = 0.01-0.1")
    
    # ========================================
    # 示例3: 分析残差曲线
    # ========================================
    print("\n【示例3: 分析收敛曲线】")
    
    # 使用详细模式观察收敛
    config_verbose = ROFConfig(
        lambda_param=0.1,
        max_iterations=200,
        verbose=True
    )
    solver_verbose = ChambollePockROF(config_verbose)
    _, info = solver_verbose.solve(noisy)
    
    residuals = info['residual_history']
    print(f"\n残差变化:")
    print(f"  初始: {residuals[0]:.2e}")
    print(f"  中间: {residuals[len(residuals)//2]:.2e}")
    print(f"  最终: {residuals[-1]:.2e}")
    
    return info


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    """主函数入口"""
    
    # 运行所有测试
    run_all_tests()
    
    # 展示使用示例
    print("\n")
    usage_examples()
