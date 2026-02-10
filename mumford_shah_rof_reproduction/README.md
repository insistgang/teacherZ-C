# Mumford-Shah 与 ROF 模型复现项目

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-1.19+-green.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.5+-green.svg)](https://scipy.org/)

这是一个用于复现和研究 **Mumford-Shah 分割模型** 和 **Rudin-Osher-Fatemi (ROF) 去噪模型** 的完整 Python 项目。项目实现了多种数值优化算法，并提供了详细的文档和示例。

## 📚 理论基础

### Mumford-Shah 模型
Mumford-Shah 模型是图像分割领域最重要的变分模型之一，由 David Mumford 和 Jayant Shah 于 1989 年提出。模型将图像分割问题转化为能量最小化问题：

$$E(u, K) = \int_{\Omega \setminus K} (u - f)^2 dx + \mu \int_{\Omega \setminus K} |\nabla u|^2 dx + \nu |K|$$

其中：
- $u$：平滑后的图像
- $f$：原始图像
- $K$：图像中的边缘（不连续点集合）
- $\mu$：平滑项权重
- $\nu$：边缘长度惩罚项

### ROF 模型
Rudin-Osher-Fatemi (ROF) 模型，也称为 TV-L2 模型，是图像去噪的经典方法：

$$\min_u \int_{\Omega} |\nabla u| dx + \frac{\lambda}{2} \int_{\Omega} (u - f)^2 dx$$

其中 $\lambda$ 控制保真项和正则化项之间的平衡。

### Chan-Vese 模型
Chan-Vese 模型是 Mumford-Shah 模型的一个简化版本，使用水平集方法实现：

$$E(c_1, c_2, \phi) = \mu \cdot \text{Length}(C) + \nu \cdot \text{Area}(\text{inside}(C))$$
$$+ \lambda_1 \int_{\text{inside}(C)} |f - c_1|^2 dx + \lambda_2 \int_{\text{outside}(C)} |f - c_2|^2 dx$$

## 🚀 安装说明

### 环境要求
- Python 3.7 或更高版本
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Matplotlib >= 3.3.0
- scikit-image >= 0.17.0
- Pillow >= 8.0.0

### 安装步骤

1. **克隆仓库**
```bash
cd mumford_shah_rof_reproduction
```

2. **创建虚拟环境（推荐）**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

## 📝 快速开始

### 示例 1: ROF 去噪
```python
import numpy as np
import matplotlib.pyplot as plt
from src.rof_models import chambolle_rof
from src.utils import add_noise, load_image

# 加载图像
image = load_image('data/sample.png', gray=True)

# 添加高斯噪声
noisy = add_noise(image, noise_type='gaussian', sigma=0.1)

# ROF 去噪
denoised = chambolle_rof(noisy, lambda_param=0.1, max_iter=100)

# 显示结果
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray'); axes[0].set_title('原始图像')
axes[1].imshow(noisy, cmap='gray'); axes[1].set_title('噪声图像')
axes[2].imshow(denoised, cmap='gray'); axes[2].set_title('ROF 去噪')
plt.show()
```

### 示例 2: Chan-Vese 分割
```python
from src.chan_vese import chan_vese_segmentation
from src.utils import load_image

# 加载图像
image = load_image('data/sample.png', gray=True)

# 初始化水平集（圆形）
rows, cols = image.shape
phi = np.ones((rows, cols))
center_x, center_y = rows // 2, cols // 2
radius = min(rows, cols) // 4
Y, X = np.ogrid[:rows, :cols]
phi = np.sqrt((X - center_y)**2 + (Y - center_x)**2) - radius

# Chan-Vese 分割
segmentation, phi_final, energies = chan_vese_segmentation(
    image, phi, max_iter=200, dt=0.5, mu=0.1, lambda1=1.0, lambda2=1.0
)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(121); plt.imshow(image, cmap='gray'); plt.title('原始图像')
plt.subplot(122); plt.imshow(segmentation, cmap='gray'); plt.title('分割结果')
plt.show()
```

## 📂 项目结构

```
mumford_shah_rof_reproduction/
├── README.md                   # 项目说明文档
├── requirements.txt            # Python 依赖
├── src/                        # 源代码
│   ├── __init__.py
│   ├── rof_models.py          # ROF 模型实现
│   ├── mumford_shah_models.py # Mumford-Shah 模型实现
│   ├── chan_vese.py           # Chan-Vese 水平集方法
│   ├── utils.py               # 工具函数
│   └── optimization.py        # 优化算法
├── docs/                       # 文档
│   ├── theory.md              # 理论文档
│   ├── algorithm_guide.md     # 算法指南
│   └── api_reference.md       # API 参考
├── data/                       # 数据目录
├── results/                    # 结果输出目录
├── examples/                   # 示例脚本
│   ├── example_rof_denoise.py
│   ├── example_chan_vese.py
│   └── example_comparison.py
└── tests/                      # 单元测试
    ├── test_rof.py
    └── test_mumford_shah.py
```

## 🔧 核心功能

### ROF 模型实现
- **梯度下降法** (`gradient_descent_rof`): 基础实现，易于理解
- **Chambolle 投影法** (`chambolle_rof`): 快速且数值稳定
- **Split Bregman 方法** (`split_bregman_rof`): 更快的收敛速度

### Mumford-Shah 模型实现
- **分段平滑逼近** (`mumford_shah_segmentation`): 完整的 M-S 模型
- **水平集演化** (`level_set_evolution`): 基于水平集的实现

### Chan-Vese 模型实现
- **水平集分割** (`chan_vese_segmentation`): 经典的 C-V 方法
- **符号距离函数重初始化** (`reinitialize_sdf`): 保持水平集性质

## 📖 文档

- [理论文档](docs/theory.md) - ROF 和 M-S 模型的数学理论
- [算法指南](docs/algorithm_guide.md) - 如何使用各种算法
- [API 参考](docs/api_reference.md) - 完整的 API 文档

## 🧪 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_rof.py
```

## 📊 示例结果

项目包含多个示例脚本，可以直接运行：

```bash
# ROF 去噪示例
python examples/example_rof_denoise.py

# Chan-Vese 分割示例
python examples/example_chan_vese.py

# 算法对比示例
python examples/example_comparison.py
```

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！请确保：
1. 代码遵循 PEP 8 规范
2. 添加适当的单元测试
3. 更新相关文档
4. 使用中文注释

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 📚 参考文献

1. Mumford, D., & Shah, J. (1989). Optimal approximations by piecewise smooth functions and associated variational problems. *Communications on Pure and Applied Mathematics*, 42(5), 577-685.

2. Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D: Nonlinear Phenomena*, 60(1-4), 259-268.

3. Chan, T. F., & Vese, L. A. (2001). Active contours without edges. *IEEE Transactions on Image Processing*, 10(2), 266-277.

4. Chambolle, A. (2004). An algorithm for total variation minimization and applications. *Journal of Mathematical Imaging and Vision*, 20(1-2), 89-97.

5. Goldstein, T., & Osher, S. (2009). The split Bregman method for L1-regularized problems. *SIAM Journal on Imaging Sciences*, 2(2), 323-343.

## 📧 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 提交 GitHub Issue
- 发送邮件至：your.email@example.com

---

**祝您使用愉快！**
