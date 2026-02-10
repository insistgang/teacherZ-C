# 算法使用指南

本文档提供项目中各种算法的详细使用指南。

## 目录
1. [快速开始](#快速开始)
2. [ROF 去噪算法](#rof-去噪算法)
3. [Chan-Vese 分割](#chan-vese-分割)
4. [Mumford-Shah 分割](#mumford-shah-分割)
5. [参数调优指南](#参数调优指南)
6. [常见问题](#常见问题)

---

## 快速开始

### 安装和导入

```python
import numpy as np
import matplotlib.pyplot as plt
from src import (
    chambolle_rof,          # ROF 去噪
    chan_vese_segmentation, # Chan-Vese 分割
    add_noise,              # 添加噪声
    load_image,             # 加载图像
    psnr, ssim              # 质量评估
)
```

### 基本工作流程

```python
# 1. 加载图像
image = load_image('path/to/image.png', gray=True)

# 2. 添加噪声（测试用）
noisy = add_noise(image, 'gaussian', sigma=0.1)

# 3. 应用算法
result = your_algorithm(noisy, ...)

# 4. 评估质量
print(f"PSNR: {psnr(image, result):.2f} dB")
print(f"SSIM: {ssim(image, result):.4f}")

# 5. 可视化
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(image, cmap='gray'); plt.title('原始')
plt.subplot(132); plt.imshow(noisy, cmap='gray'); plt.title('噪声')
plt.subplot(133); plt.imshow(result, cmap='gray'); plt.title('结果')
plt.show()
```

---

## ROF 去噪算法

### 1. Chambolle 投影法（推荐）

**适用场景**：一般去噪任务，需要快速稳定的结果

```python
from src import chambolle_rof

denoised, dual_var = chambolle_rof(
    noisy_image,
    lambda_param=0.5,  # 正则化参数
    max_iter=100,      # 最大迭代次数
    tol=1e-4          # 收敛容差
)
```

**参数说明**：
- `lambda_param`：控制平滑程度
  - 0.01-0.1：强去噪，适合高噪声
  - 0.1-1.0：中等去噪（推荐值）
  - 1.0-10.0：弱去噪，保留细节

**返回值**：
- `denoised`：去噪后的图像
- `dual_var`：对偶变量（可用于分析边缘）

### 2. Split Bregman 方法

**适用场景**：大规模图像，需要更快收敛

```python
from src import split_bregman_rof

denoised, history = split_bregman_rof(
    noisy_image,
    lambda_param=0.5,
    max_iter=20,       # 通常 20 次即可收敛
    mu=0.1            # 增广拉格朗日参数
)
```

### 3. 梯度下降法

**适用场景**：教学演示，理解算法原理

```python
from src import gradient_descent_rof

denoised, history = gradient_descent_rof(
    noisy_image,
    lambda_param=0.5,
    step_size=0.01,    # 步长需要仔细调整
    max_iter=500
)
```

### 选择合适的算法

| 算法 | 速度 | 稳定性 | 适用场景 |
|------|------|--------|----------|
| Chambolle | 快 | 高 | 日常使用 |
| Split Bregman | 很快 | 高 | 大规模图像 |
| 梯度下降 | 慢 | 中 | 学习/研究 |

---

## Chan-Vese 分割

### 基本使用

```python
from src import chan_vese_segmentation, initialize_sdf_circle

# 初始化水平集（圆形）
phi0 = initialize_sdf_circle(
    image.shape,
    center=(100, 100),  # 圆心 (y, x)
    radius=50           # 半径
)

# 运行分割
segmentation, phi, history = chan_vese_segmentation(
    image,
    phi0,
    max_iter=200,
    mu=0.2,        # 轮廓长度权重
    nu=0.0,        # 区域面积权重
    lambda1=1.0,   # 前景数据权重
    lambda2=1.0,   # 背景数据权重
    dt=0.5         # 时间步长
)
```

### 初始化策略

**1. 圆形初始化**
```python
from src import initialize_sdf_circle

# 单个圆
phi = initialize_sdf_circle(shape, center=(y, x), radius=r)

# 多个圆（多目标分割）
from src import initialize_sdf_multiple_circles
phi = initialize_sdf_multiple_circles(
    shape,
    centers=[(50, 50), (150, 150)],
    radii=[30, 40]
)
```

**2. 矩形初始化**
```python
from src import initialize_sdf_rectangle

phi = initialize_sdf_rectangle(
    shape,
    top_left=(y1, x1),
    bottom_right=(y2, x2)
)
```

**3. 自定义初始化**
```python
# 手动创建符号距离函数
rows, cols = shape
Y, X = np.ogrid[:rows, :cols]
phi = np.sin(X/10) * np.cos(Y/10)  # 任意形状
```

### 参数调优

**mu（轮廓长度权重）**：
- 作用：控制轮廓的光滑程度
- 小值 (0.01-0.1)：允许复杂轮廓，更多细节
- 大值 (0.5-1.0)：平滑轮廓，减少噪声
- 推荐：从 0.1 开始调整

**nu（区域面积权重）**：
- 作用：控制前景区域大小
- 正值：倾向于缩小前景
- 负值：倾向于扩大前景
- 通常设为 0

**lambda1, lambda2（数据权重）**：
- 作用：控制对前景/背景的拟合程度
- 默认都设为 1.0
- 如果前景更重要，增大 lambda1
- 如果背景更重要，增大 lambda2

### 收敛判断

```python
segmentation, phi, history = chan_vese_segmentation(
    image, phi0, max_iter=500, tol=1e-6
)

# 检查是否收敛
if len(history) < 500:
    print(f"算法在 {len(history)} 次迭代后收敛")
else:
    print("算法达到最大迭代次数")

# 绘制能量曲线
plt.plot(history)
plt.xlabel('迭代次数')
plt.ylabel('能量')
plt.title('能量收敛曲线')
plt.show()
```

---

## Mumford-Shah 分割

### 简化版 M-S 分割

```python
from src import mumford_shah_segmentation

u, edges, history = mumford_shah_segmentation(
    image,
    mu=0.5,      # 平滑权重
    nu=0.01,     # 边缘长度惩罚
    max_iter=100
)
```

**返回值**：
- `u`：分段平滑的逼近图像
- `edges`：检测到的边缘（二值图像）
- `history`：能量历史

### 水平集方法

```python
from src import level_set_evolution, initialize_sdf_circle

phi0 = initialize_sdf_circle(image.shape, radius=50)

u, phi, segmentation, history = level_set_evolution(
    image,
    phi0,
    mu=0.1,       # 边缘长度权重
    nu=0.0,       # 区域面积权重
    lambda1=1.0,  # 内部数据权重
    lambda2=1.0,  # 外部数据权重
    max_iter=200
)
```

### Ambrosio-Tortorelli 近似

```python
from src import ambrosio_tortorelli_approximation

u, v = ambrosio_tortorelli_approximation(
    image,
    mu=1.0,       # 平滑权重
    nu=0.01,      # 边缘惩罚
    epsilon=0.01  # 相场宽度
)

# v 是相场，0 表示边缘，1 表示非边缘
```

---

## 参数调优指南

### ROF 去噪参数选择

**根据噪声水平选择 lambda**：

| 噪声水平 | sigma | 推荐 lambda | 说明 |
|----------|-------|-------------|------|
| 低 | 0.05 | 1.0-5.0 | 轻微软化 |
| 中 | 0.10 | 0.5-1.0 | 平衡选择 |
| 高 | 0.20 | 0.1-0.5 | 强去噪 |

**自适应参数选择**：

```python
from src import adaptive_rof

best_result, best_lambda, all_results = adaptive_rof(
    noisy_image,
    lambda_min=0.01,
    lambda_max=10.0,
    num_lambda=20
)
print(f"最优 lambda: {best_lambda}")
```

### Chan-Vese 参数选择

**根据图像类型选择参数**：

| 图像类型 | mu | lambda1/lambda2 | 说明 |
|----------|-----|-----------------|------|
| 干净图像 | 0.05-0.1 | 1.0/1.0 | 最小正则化 |
| 噪声图像 | 0.2-0.5 | 1.0/1.0 | 增加平滑 |
| 复杂形状 | 0.01-0.05 | 1.0/1.0 | 保留细节 |
| 不均匀强度 | 0.1-0.2 | 2.0/1.0 | 强调前景 |

### 通用调参流程

```python
def grid_search_cv_params(image, phi0, mu_range, lambda_range):
    """网格搜索 Chan-Vese 参数"""
    best_score = float('inf')
    best_params = None
    best_result = None
    
    for mu in mu_range:
        for lam in lambda_range:
            seg, phi, _ = chan_vese_segmentation(
                image, phi0, max_iter=100,
                mu=mu, lambda1=lam, lambda2=lam
            )
            
            # 评估：能量越低越好
            score = evaluate_segmentation(image, seg)
            
            if score < best_score:
                best_score = score
                best_params = (mu, lam)
                best_result = seg
    
    return best_params, best_result
```

---

## 常见问题

### Q1: ROF 去噪后图像出现阶梯效应？

**原因**：TV 正则化倾向于产生分段常数解。

**解决**：
- 增大 lambda 减少阶梯效应
- 使用高阶 TV（如 TGV）
- 考虑其他正则化项

### Q2: Chan-Vese 分割不收敛？

**可能原因**：
1. 时间步长 dt 太大
2. 初始水平集与目标区域差距太大
3. 图像对比度太低

**解决**：
```python
# 减小时步
segmentation, phi, history = chan_vese_segmentation(
    image, phi0, dt=0.1  # 减小时间步长
)

# 更好的初始化
phi0 = initialize_sdf_circle(image.shape, 
                             center=estimate_center(image), 
                             radius=estimate_radius(image))
```

### Q3: 分割结果包含太多小区域？

**解决**：增大 mu 值
```python
segmentation, phi, _ = chan_vese_segmentation(image, phi0, mu=0.5)  # 增大平滑
```

### Q4: 如何加速计算？

**优化建议**：
1. 使用 Chambolle 而非梯度下降
2. 减小 max_iter（观察收敛曲线）
3. 使用多尺度方法
4. 考虑 GPU 实现（项目未包含）

```python
# 多尺度策略
def multiscale_cv(image, scales=[0.25, 0.5, 1.0]):
    for scale in scales:
        small = resize(image, scale)
        # 在低分辨率上快速计算
        ...
```

### Q5: 如何处理彩色图像？

**方法**：对每个通道分别处理或转换为灰度

```python
from skimage import color

# 转换为灰度
ggray = color.rgb2gray(rgb_image)
result = chambolle_rof(gray, lambda_param=0.5)

# 或分别处理各通道
result_rgb = np.zeros_like(rgb_image)
for i in range(3):
    result_rgb[:,:,i] = chambolle_rof(
        rgb_image[:,:,i], lambda_param=0.5
    )
```

---

## 性能优化建议

### 内存优化

```python
# 使用 float32 而非 float64
image = image.astype(np.float32)

# 及时释放不用的变量
del large_intermediate_result
```

### 计算优化

```python
# 1. 使用更快的算法
from src import split_bregman_rof  # 比梯度下降快

# 2. 减少迭代次数（如果已收敛）
denoised, _ = chambolle_rof(image, lambda_param=0.5, max_iter=50)

# 3. 使用小图像测试参数，再应用到全图
```

### 批量处理

```python
def batch_denoise(images, lambda_param=0.5):
    """批量去噪"""
    results = []
    for img in images:
        result, _ = chambolle_rof(img, lambda_param)
        results.append(result)
    return results
```

---

## 更多示例

参见 `examples/` 目录：
- `example_rof_denoise.py`：ROF 去噪完整示例
- `example_chan_vese.py`：Chan-Vese 分割完整示例
- `example_comparison.py`：算法对比示例
