"""
Mumford-Shah 与 ROF 模型复现项目

本项目实现了经典的图像分割和去噪模型：
- ROF (Rudin-Osher-Fatemi) 去噪模型
- Mumford-Shah 分割模型
- Chan-Vese 水平集分割模型

模块说明：
- rof_models: ROF 模型的各种实现（梯度下降、Chambolle、Split Bregman）
- mumford_shah_models: Mumford-Shah 模型的实现
- chan_vese: Chan-Vese 水平集分割方法
- utils: 工具函数（噪声添加、图像加载、质量评估等）
- optimization: 优化算法（梯度下降、Chambolle 投影等）

作者: [Your Name]
日期: 2026-02-10
"""

# 版本信息
__version__ = '1.0.0'
__author__ = 'Your Name'

# 导出主要函数，方便用户使用
from .rof_models import (
    gradient_descent_rof,
    chambolle_rof,
    split_bregman_rof,
    rof_energy
)

from .mumford_shah_models import (
    mumford_shah_segmentation,
    level_set_evolution,
    ms_energy
)

from .chan_vese import (
    chan_vese_segmentation,
    reinitialize_sdf,
    heaviside,
    dirac_delta
)

from .utils import (
    add_noise,
    gaussian_noise,
    salt_pepper_noise,
    psnr,
    ssim,
    load_image,
    save_image,
    normalize_image
)

from .optimization import (
    gradient_descent,
    chambolle_projection,
    compute_gradient,
    compute_divergence
)

# 定义 __all__ 变量，明确公开接口
__all__ = [
    # ROF 模型
    'gradient_descent_rof',
    'chambolle_rof',
    'split_bregman_rof',
    'rof_energy',
    
    # Mumford-Shah 模型
    'mumford_shah_segmentation',
    'level_set_evolution',
    'ms_energy',
    
    # Chan-Vese 模型
    'chan_vese_segmentation',
    'reinitialize_sdf',
    'heaviside',
    'dirac_delta',
    
    # 工具函数
    'add_noise',
    'gaussian_noise',
    'salt_pepper_noise',
    'psnr',
    'ssim',
    'load_image',
    'save_image',
    'normalize_image',
    
    # 优化算法
    'gradient_descent',
    'chambolle_projection',
    'compute_gradient',
    'compute_divergence',
]
