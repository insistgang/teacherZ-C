# API 参考文档

## 目录
- [ROF 模型 (`rof_models`)](#rof-模型)
- [Mumford-Shah 模型 (`mumford_shah_models`)](#mumford-shah-模型)
- [Chan-Vese 模型 (`chan_vese`)](#chan-vese-模型)
- [工具函数 (`utils`)](#工具函数)
- [优化算法 (`optimization`)](#优化算法)

---

## ROF 模型

### `chambolle_rof`

```python
chambolle_rof(f, lambda_param, max_iter=100, tol=1e-4)
```

使用 Chambolle 投影法求解 ROF 去噪模型。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `f` | ndarray | - | 输入含噪图像，值域 [0, 1] |
| `lambda_param` | float | - | 正则化参数，控制平滑程度 |
| `max_iter` | int | 100 | 最大迭代次数 |
| `tol` | float | 1e-4 | 收敛容差 |

**返回**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `u` | ndarray | 去噪后的图像 |
| `p` | ndarray | 对偶变量，形状为 (H, W, 2) |

**示例**：
```python
from src import chambolle_rof
u, p = chambolle_rof(noisy_image, lambda_param=0.5, max_iter=100)
```

---

### `split_bregman_rof`

```python
split_bregman_rof(f, lambda_param, max_iter=20, inner_iter=1, tol=1e-6, mu=0.1)
```

使用 Split Bregman 方法求解 ROF 模型。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `f` | ndarray | - | 输入含噪图像 |
| `lambda_param` | float | - | 正则化参数 |
| `max_iter` | int | 20 | 外循环最大迭代次数 |
| `inner_iter` | int | 1 | u-子问题内迭代次数 |
| `tol` | float | 1e-6 | 收敛容差 |
| `mu` | float | 0.1 | 增广拉格朗日参数 |

**返回**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `u` | ndarray | 去噪后的图像 |
| `energy_history` | list | 能量值迭代历史 |

---

### `gradient_descent_rof`

```python
gradient_descent_rof(f, lambda_param, step_size=0.01, max_iter=500, tol=1e-6, verbose=False)
```

使用梯度下降法求解 ROF 模型。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `f` | ndarray | - | 输入含噪图像 |
| `lambda_param` | float | - | 正则化参数 |
| `step_size` | float | 0.01 | 梯度下降步长 |
| `max_iter` | int | 500 | 最大迭代次数 |
| `tol` | float | 1e-6 | 收敛容差 |
| `verbose` | bool | False | 是否打印进度 |

**返回**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `u` | ndarray | 去噪后的图像 |
| `energy_history` | list | 能量值迭代历史 |

---

### `rof_energy`

```python
rof_energy(u, f, lambda_param)
```

计算 ROF 能量函数值。

**公式**：
$$E(u) = \int |\nabla u| + \frac{\lambda}{2} \int (u-f)^2$$

---

## Mumford-Shah 模型

### `mumford_shah_segmentation`

```python
mumford_shah_segmentation(f, mu=1.0, nu=0.01, max_iter=100, tol=1e-5, verbose=False)
```

Mumford-Shah 图像分割（简化实现）。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `f` | ndarray | - | 输入图像 [0, 1] |
| `mu` | float | 1.0 | 平滑项权重 |
| `nu` | float | 0.01 | 边缘长度惩罚 |
| `max_iter` | int | 100 | 最大迭代次数 |
| `tol` | float | 1e-5 | 收敛容差 |
| `verbose` | bool | False | 是否打印进度 |

**返回**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `u` | ndarray | 分段平滑的逼近图像 |
| `edge_set` | ndarray | 检测到的边缘（二值图像） |
| `energy_history` | list | 能量迭代历史 |

---

### `level_set_evolution`

```python
level_set_evolution(f, phi0, mu=0.1, nu=0.0, lambda1=1.0, lambda2=1.0, 
                    dt=0.1, max_iter=200, reinit_interval=20, tol=1e-6, verbose=False)
```

基于水平集的 Mumford-Shah 分割。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `f` | ndarray | - | 输入图像 |
| `phi0` | ndarray | - | 初始水平集函数 |
| `mu` | float | 0.1 | 曲线长度惩罚权重 |
| `nu` | float | 0.0 | 区域面积权重 |
| `lambda1` | float | 1.0 | 内部数据项权重 |
| `lambda2` | float | 1.0 | 外部数据项权重 |
| `dt` | float | 0.1 | 时间步长 |
| `max_iter` | int | 200 | 最大迭代次数 |
| `reinit_interval` | int | 20 | 重初始化间隔 |
| `tol` | float | 1e-6 | 收敛容差 |
| `verbose` | bool | False | 是否打印进度 |

**返回**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `u` | ndarray | 分段常数逼近图像 |
| `phi` | ndarray | 最终水平集函数 |
| `segmentation` | ndarray | 分割结果（二值图像） |
| `energy_history` | list | 能量迭代历史 |

---

### `ms_energy`

```python
ms_energy(u, f, edge_set, mu, nu)
```

计算 Mumford-Shah 能量函数值。

---

### `ambrosio_tortorelli_approximation`

```python
ambrosio_tortorelli_approximation(f, mu=1.0, nu=0.01, epsilon=0.01, max_iter=100)
```

Ambrosio-Tortorelli 相场近似。

**返回**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `u` | ndarray | 分段平滑图像 |
| `v` | ndarray | 相场（边缘指示） |

---

## Chan-Vese 模型

### `chan_vese_segmentation`

```python
chan_vese_segmentation(f, phi0, max_iter=200, dt=0.5, mu=0.1, nu=0.0,
                      lambda1=1.0, lambda2=1.0, tol=1e-6, 
                      reinit_interval=5, verbose=False)
```

Chan-Vese 图像分割。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `f` | ndarray | - | 输入图像 [0, 1] |
| `phi0` | ndarray | - | 初始水平集函数 |
| `max_iter` | int | 200 | 最大迭代次数 |
| `dt` | float | 0.5 | 时间步长 |
| `mu` | float | 0.1 | 轮廓长度惩罚权重 |
| `nu` | float | 0.0 | 区域面积权重 |
| `lambda1` | float | 1.0 | 前景数据项权重 |
| `lambda2` | float | 1.0 | 背景数据项权重 |
| `tol` | float | 1e-6 | 收敛容差 |
| `reinit_interval` | int | 5 | 重初始化间隔 |
| `verbose` | bool | False | 是否打印进度 |

**返回**：
| 返回值 | 类型 | 说明 |
|--------|------|------|
| `segmentation` | ndarray | 分割结果（1=前景，0=背景） |
| `phi` | ndarray | 最终水平集函数 |
| `energy_history` | list | 能量迭代历史 |

---

### `initialize_sdf_circle`

```python
initialize_sdf_circle(shape, center=None, radius=None)
```

初始化圆形符号距离函数。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `shape` | tuple | - | 图像尺寸 (H, W) |
| `center` | tuple | None | 圆心 (y, x)，默认图像中心 |
| `radius` | float | None | 半径，默认 min(H,W)/4 |

**返回**：`ndarray` - 符号距离函数

---

### `initialize_sdf_rectangle`

```python
initialize_sdf_rectangle(shape, top_left=None, bottom_right=None)
```

初始化矩形符号距离函数。

---

### `initialize_sdf_multiple_circles`

```python
initialize_sdf_multiple_circles(shape, centers, radii)
```

初始化多个圆的符号距离函数。

**参数**：
| 参数 | 类型 | 说明 |
|------|------|------|
| `centers` | list | 圆心列表 [(y1, x1), (y2, x2), ...] |
| `radii` | list | 半径列表 [r1, r2, ...] |

---

### `reinitialize_sdf`

```python
reinitialize_sdf(phi, iterations=10, dt=0.1)
```

重初始化符号距离函数。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `phi` | ndarray | - | 当前水平集函数 |
| `iterations` | int | 10 | 迭代次数 |
| `dt` | float | 0.1 | 时间步长 |

**返回**：`ndarray` - 重初始化后的符号距离函数

---

### `heaviside`

```python
heaviside(phi, eps=1.0)
```

平滑 Heaviside 函数。

**公式**：$H_\epsilon(z) = \frac{1}{2}[1 + \frac{2}{\pi}\arctan(\frac{z}{\epsilon})]$

---

### `dirac_delta`

```python
dirac_delta(phi, eps=1.0)
```

平滑 Dirac delta 函数。

**公式**：$\delta_\epsilon(z) = \frac{\epsilon}{\pi(\epsilon^2 + z^2)}$

---

## 工具函数

### `add_noise`

```python
add_noise(image, noise_type='gaussian', **kwargs)
```

为图像添加噪声。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | ndarray | - | 输入图像 [0, 1] |
| `noise_type` | str | 'gaussian' | 噪声类型 |
| `**kwargs` | - | - | 噪声参数 |

**支持的噪声类型**：
- `'gaussian'`：`sigma` (标准差，默认 0.1)
- `'salt_pepper'`：`amount` (噪声比例，默认 0.05)
- `'poisson'`：无额外参数

---

### `gaussian_noise`

```python
gaussian_noise(image, sigma=0.1)
```

添加高斯噪声。

---

### `salt_pepper_noise`

```python
salt_pepper_noise(image, amount=0.05, salt_vs_pepper=0.5)
```

添加椒盐噪声。

---

### `psnr`

```python
psnr(original, denoised, max_val=1.0)
```

计算峰值信噪比。

**公式**：$PSNR = 10 \log_{10}(\frac{MAX^2}{MSE})$

---

### `ssim`

```python
ssim(original, denoised, window_size=11, K1=0.01, K2=0.03, max_val=1.0)
```

计算结构相似性指数。

---

### `load_image`

```python
load_image(path, gray=True, resize=None)
```

加载图像文件。

**参数**：
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | str | - | 图像文件路径 |
| `gray` | bool | True | 是否转换为灰度 |
| `resize` | tuple | None | 调整大小 (W, H) |

**返回**：`ndarray` - 图像数组 [0, 1]

---

### `save_image`

```python
save_image(image, path, quality=95)
```

保存图像到文件。

---

### `create_synthetic_image`

```python
create_synthetic_image(shape=(256, 256), pattern='checkerboard')
```

创建合成测试图像。

**支持的图案**：
- `'checkerboard'`：棋盘
- `'circles'`：同心圆
- `'gradient'`：线性渐变
- `'step'`：阶梯函数

---

## 优化算法

### `gradient_descent`

```python
gradient_descent(energy_func, gradient_func, x0, step_size=0.01, 
                 max_iter=1000, tol=1e-6, verbose=False)
```

通用梯度下降优化。

---

### `chambolle_projection`

```python
chambolle_projection(f, lambda_param, max_iter=100, tol=1e-4)
```

Chambolle 对偶投影算法。

---

### `compute_gradient`

```python
compute_gradient(u)
```

计算标量场的梯度（中心差分）。

**返回**：`(grad_x, grad_y)`

---

### `compute_divergence`

```python
compute_divergence(px, py)
```

计算向量场的散度。

**公式**：$\text{div}(p) = \frac{\partial p_x}{\partial x} + \frac{\partial p_y}{\partial y}$

---

### `compute_laplacian`

```python
compute_laplacian(u)
```

计算拉普拉斯算子。

**公式**：$\Delta u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}$

---

## 类型别名

```python
import numpy as np
from typing import Tuple, List, Optional

# 图像类型
Image = np.ndarray  # 形状为 (H, W) 或 (H, W, C)

# 水平集类型
LevelSet = np.ndarray  # 形状为 (H, W)

# 对偶变量类型
DualVariable = np.ndarray  # 形状为 (H, W, 2)

# 能量历史
EnergyHistory = List[float]
```

---

## 异常说明

| 异常 | 触发条件 | 处理建议 |
|------|----------|----------|
| `ValueError` | 参数不合法 | 检查参数范围 |
| `IOError` | 图像加载失败 | 检查文件路径 |
| `RuntimeError` | 算法未收敛 | 调整参数或增加迭代次数 |

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0.0 | 2026-02-10 | 初始版本 |
