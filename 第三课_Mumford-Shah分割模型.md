# 第三课：Mumford-Shah分割模型 - 从去噪到分割

> **学习目标**: 理解图像分割的变分方法，掌握Mumford-Shah和Chan-Vese
> **数学难度**: ⭐⭐⭐⭐ (比ROF更复杂)
> **实用价值**: ⭐⭐⭐⭐⭐ (医学图像、遥感、计算机视觉)

---

## Part 1: 从ROF到Mumford-Shah的动机

### 1.1 ROF模型的局限

**ROF解决的问题**：
```
min_u ∫|∇u| + (λ/2)(u-f)²
```
- 目标：去噪
- 输出：平滑后的图像 u
- 假设：图像本身连续，只有噪声

**关键问题**：ROF不告诉我们在哪里分割！

```
ROF输出: 平滑的图像
我们需要: 明确的区域边界
```

**示例**：医学图像分析
```
输入: 脑部MRI扫描
ROF: 平滑的MRI
需求: 肿瘤在哪里？（需要分割）
```

---

### 1.2 什么是图像分割？

**定义**：将图像域 Ω 分割为不相交的区域

```
Ω = Ω₁ ∪ Ω₂ ∪ ... ∪ Ω_K
Ω_i ∩ Ω_j = ∅ (i ≠ j)
```

**每个区域的特性**：
- **内部光滑**：强度变化小
- **边界清晰**：强度跳跃明显

**分割的数学表示**：
- **显式**：边界集合 Γ = ∪Γ_ij
- **隐式**：水平集函数 φ(x)

---

### 1.3 Mumford-Shah的洞察

**关键创新**：同时优化两样东西

1. **分割边界 Γ**：在哪里分割？
2. **近似图像 u**：每个区域用什么值？

**统一的能量泛函**：
```
E(u, Γ) = ∫_{Ω\Γ} |∇u|² dx + μ∫_Ω (u-f)² dx + ν|Γ|
         ↑                ↑              ↑
      光滑性           保真度         边缘长度
```

**三项的作用**：
1. **光滑性项**：u在非边缘处光滑
2. **保真项**：u接近观察图像f
3. **边缘长度**：惩罚过长的边缘

---

## Part 2: Mumford-Shah泛函详解

### 2.1 与ROF的对比

| 特性 | ROF模型 | Mumford-Shah模型 |
|:---|:---|:---|
| **变量** | u（图像） | u + Γ（图像+边界） |
| **目标** | 去噪 | 分割+去噪 |
| **正则化** | ∫\|∇u\| | ∫\|∇u\|² + ν\|Γ\| |
| **不连续** | 隐式允许 | 显式建模 |
| **光滑性** | 分段常数 | 分段光滑 |
| **难度** | 凸优化 | 非凸+自由边界 |

---

### 2.2 Mumford-Shah的三种形式

#### **形式1：完整Mumford-Shah**

```
E(u, Γ) = ∫_{Ω\Γ} |∇u|² + μ∫_Ω (u-f)² + ν|Γ|
```

- u：分段光滑函数
- Γ：边缘集合（不连续点集）
- 难度：自由边界问题，极难求解

#### **形式2：分片常数Mumford-Shah**

假设 u在每个区域内是常数：
```
E(c₁,...,c_K, Γ) = Σ_i ∫_{Ω_i} (c_i - f)² + ν|Γ|
```

- 简化：不需要求u，只需求常数c_i
- 这是**K-means的连续版本**

#### **形式3：Chan-Vese（两相分割）**

```
E(c₁, c₂, Γ) = μ|Γ| + λ₁∫_{in(Γ)}(f-c₁)² + λ₂∫_{out(Γ)}(f-c₂)²
```

- 最实用：两相（前景/背景）分割
- 可用水平集方法求解

---

### 2.3 Mumford-Shah的几何意义

**物理类比**：弹性膜问题

```
想象 u 是一个弹性薄膜：
├── 膜可以沿着Γ"撕裂"
├── 膜的张力使其光滑（|∇u|²项）
├── 数据f吸引膜（保真项）
└── 撕裂需要能量（ν|Γ|项）
```

**最优平衡**：
```
如果 |∇f| 大 → 值得撕裂 → Γ经过
如果 |∇f| 小 → 不撕裂 → u光滑
```

---

## Part 3: 自由边界问题

### 3.1 什么是自由边界？

**定义**：边界不是预先给定的，而是解的一部分

**对比**：
```
固定边界: Ω已知 → 在∂Ω上求解PDE
自由边界: Ω未知 → Ω和PDE解同时优化
```

**图像分割中的自由边界**：
- Γ是物体与背景的边界
- Γ的位置由图像内容决定
- Γ的形状可以任意复杂

---

### 3.2 为什么自由边界难？

**挑战1：拓扑复杂**
```
Γ可以是：
├── 单条闭合曲线
├── 多条曲线
├── 有分叉
└── 任意形状
```

**挑战2：非凸性**
```
能量泛函非凸 → 多个局部极小值
├── 不同的Γ对应不同的能量
├── 难以找到全局最优
└── 需要好的初始化
```

**挑战3：变量耦合**
```
u和Γ相互依赖：
├── 给定Γ，求u容易（Poisson方程）
├── 给定u，求Γ困难（边缘检测）
└── 需要交替优化
```

---

## Part 4: Chan-Vese模型（Mumford-Shah的实用版本）

### 4.1 简化假设

**从完整M-S到Chan-Vese**：

```
完整M-S: u在Ω\Γ上光滑
Chan-Vese: u在Ω\Γ上是常数
```

**为什么这样简化？**
- 常数假设使u可解析求解
- 大多数分割问题只需分区域
- 数学上更易处理

---

### 4.2 Chan-Vese能量泛函

**两相分割**（前景vs背景）：

```
E(c₁, c₂, Γ) = μ|Γ| + λ₁∫_{inside(Γ)} (f-c₁)² + λ₂∫_{outside(Γ)} (f-c₂)²
```

**变量**：
- c₁：前景平均灰度
- c₂：背景平均灰度
- Γ：分割轮廓

**参数**：
- μ：边缘长度惩罚
- λ₁, λ₂：前景/背景权重

---

### 4.3 水平集方法

**问题**：如何表示Γ？

**方案A：显式表示**
```
Γ = {γ(t): t ∈ [0,1]}
```
- 直观
- 拓扑变化困难

**方案B：隐式表示（水平集）**

**定义**：水平集函数 φ(x)
```
φ(x) > 0  →  x在Γ内部（前景）
φ(x) < 0  →  x在Γ外部（背景）
φ(x) = 0  →  x在Γ上（边界）
```

**优势**：
- 自动处理拓扑变化
- 数值稳定
- 易于计算几何量

---

### 4.4 Heaviside和Dirac函数

**Heaviside函数 H(φ)**：
```
H(φ) = {1,  if φ ≥ 0
       {0,  if φ < 0
```

作用：指示函数，标记内部/外部

**Dirac测度 δ(φ)**：
```
δ(φ) = dH/dφ
```

作用：限制演化在边界附近

**正则化版本**（数值稳定）：
```
H_ε(φ) = (1/2)[1 + (2/π)arctan(φ/ε)]
δ_ε(φ) = (ε/π) / (φ² + ε²)
```

---

### 4.5 Chan-Vese的演化方程

**能量用水平集表示**：
```
E(φ) = μ∫δ(φ)|∇φ| + λ₁∫(f-c₁)²H(φ) + λ₂∫(f-c₂)²(1-H(φ))
```

**欧拉-拉格朗日方程**：
```
∂φ/∂t = δ(φ)[μ·div(∇φ/|∇φ|) - λ₁(f-c₁)² + λ₂(f-c₂)²]
```

**各项解释**：
```
δ(φ): 限制在边界附近演化
μ·div(∇φ/|∇φ|): 曲率项（平滑边界）
-λ₁(f-c₁)²: 前景拉力
+λ₂(f-c₂)²: 背景拉力
```

---

### 4.6 c₁和c₂的更新

**给定φ，解析求解c₁, c₂**：

```
∂E/∂c₁ = -2λ₁∫(f-c₁)²H(φ) = 0

→ c₁ = ∫f·H(φ) / ∫H(φ)
```

**直观理解**：
```
c₁ = 前景区域的平均灰度
c₂ = 背景区域的平均灰度
```

---

### 4.7 Chan-Vese算法流程

```
初始化:
├── φ⁰（初始水平集函数）
├── 设置参数 μ, λ₁, λ₂
└── 计算初始c₁, c₂

For k = 0, 1, 2, ...:
    1. 更新c₁, c₂（区域平均）
    2. 更新φ（演化方程）
    3. 重新初始化φ（保持符号距离）
    4. 检查收敛

收敛后:
    Γ = {x: φ(x) = 0}
```

---

## Part 5: 数值实现细节

### 5.1 水平集初始化

**方法1：几何形状**
```python
# 圆形初始化
y, x = np.mgrid[:H, :W]
phi = np.sqrt((x-cx)² + (y-cy)²) - radius
```

**方法2：随机**
```python
phi = np.random.randn(H, W)
```

**方法3：基于阈值**
```python
phi = f - threshold
```

---

### 5.2 曲率计算

**公式**：
```
κ = div(∇φ/|∇φ|) = (φ_xxφ_y² - 2φ_xyφ_xφ_y + φ_yyφ_x²) / |∇φ|³
```

**离散化**：
```python
# 计算梯度
phi_x = (phi[:, 1:] - phi[:, :-1]) / 2
phi_y = (phi[1:, :] - phi[:-1, :]) / 2

# 计算二阶导数
phi_xx = phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]
phi_yy = phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]
phi_xy = ...

# 曲率
kappa = (phi_xx * phi_y² - 2*phi_xy * phi_x * phi_y + phi_yy * phi_x²) / (phi_x² + phi_y² + eps)^(3/2)
```

---

### 5.3 重新初始化

**问题**：φ在演化中失去符号距离性质

**解决**：解重新初始化方程
```
∂φ/∂τ = sign(φ₀)(1 - |∇φ|)
```

**快速方法**：直接符号距离
```python
from scipy import ndimage
phi = ndimage.distance_transform_edt(phi) - ndimage.distance_transform_edt(1-phi)
```

---

## Part 6: 完整实现代码

### 6.1 Chan-Vese算法实现

```python
def chan-vese(f, mu=0.1, lambda1=1.0, lambda2=1.0,
              max_iter=200, dt=0.1, tol=1e-4):
    """
    Chan-Vese图像分割

    Parameters:
    -----------
    f : ndarray
        输入灰度图像
    mu : float
        边缘长度惩罚
    lambda1, lambda2 : float
        前景/背景权重
    max_iter : int
        最大迭代次数
    dt : float
        时间步长
    tol : float
        收敛阈值

    Returns:
    --------
    phi : ndarray
        水平集函数
    segmentation : ndarray
        分割结果（0/1）
    """

    # 初始化水平集函数
    H, W = f.shape
    y, x = np.mgrid[:H, :W]
    phi = np.sqrt((x-W/2)² + (y-H/2)²) - min(H, W)/4

    for k in range(max_iter):
        phi_old = phi.copy()

        # 1. 计算Heaviside函数
        H_phi = 0.5 * (1 + (2/np.pi) * np.arctan(phi/1e-6))

        # 2. 更新c1, c2
        c1 = np.sum(f * H_phi) / (np.sum(H_phi) + 1e-10)
        c2 = np.sum(f * (1-H_phi)) / (np.sum(1-H_phi) + 1e-10)

        # 3. 计算曲率
        phi_x = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / 2
        phi_y = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / 2
        phi_norm = np.sqrt(phi_x² + phi_y² + 1e-10)

        kappa = (phi_xx * phi_y² - 2*phi_xy * phi_x * phi_y + phi_yy * phi_x²) / phi_norm³

        # 4. 更新phi
        delta = (1e-6) / (phi² + 1e-12)  # 正则化Dirac
        force = mu * kappa - lambda1 * (f-c1)² + lambda2 * (f-c2)²

        phi = phi + dt * delta * force

        # 5. 重新初始化
        if k % 10 == 0:
            phi = reinitialize_sdf(phi)

        # 6. 检查收敛
        if np.linalg.norm(phi - phi_old) / np.linalg.norm(phi_old) < tol:
            print(f"收敛于迭代 {k}")
            break

    # 分割结果
    segmentation = (phi > 0).astype(np.uint8)

    return phi, segmentation
```

---

### 6.2 测试与可视化

```python
from skimage import data, img_as_float
import matplotlib.pyplot as plt

# 测试图像
f = img_as_float(data.camera())

# 运行Chan-Vese
phi, seg = chan_vese(f, mu=0.05, max_iter=200)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(f, cmap='gray')
axes[0].set_title('原始图像')

axes[1].imshow(phi, cmap='jet')
axes[1].set_title('水平集函数')

axes[2].imshow(seg, cmap='gray')
axes[2].set_title('分割结果')

plt.show()
```

---

## Part 7: 与其他分割方法的对比

### 7.1 方法对比

| 方法 | 类型 | 优点 | 缺点 |
|:---|:---|:---|:---|
| **Chan-Vese** | 变分法 | 全局优化，不依赖梯度 | 分段常数假设 |
| **图割** | 组合优化 | 全局最优，快速 | 需要用户交互 |
| **分水岭** | 形态学 | 快速，简单 | 过分割 |
| **深度学习** | 数据驱动 | 效果最好 | 需要大量标注 |

---

### 7.2 Chan-Vese vs 传统边缘检测

**传统方法**（Canny等）：
```
步骤:
1. 计算梯度|∇f|
2. 非极大值抑制
3. 双阈值
4. 边缘连接
```

**问题**：
- 梯度阈值难以选择
- 边缘不连续
- 没有全局优化

**Chan-Vese优势**：
- 全局能量泛函
- 自动闭合边缘
- 考虑区域信息

---

## Part 8: 实际应用案例

### 8.1 医学图像：血管分割

```python
from skimage import io

# 加载视网膜图像
f = io.imread('retina_image.png', as_gray=True)

# 运行Chan-Vese
phi, seg = chan_vese(f, mu=0.01, lambda1=1.0, lambda2=1.0)

# 后处理
from scipy import ndimage
seg_clean = ndimage.remove_small_objects(seg.astype(bool), min_size=100)
```

### 8.2 遥感图像：道路提取

```python
# 加载卫星图像
f = io.imread('satellite.png', as_gray=True)

# Chan-Vese提取道路
phi, seg = chan_vese(f, mu=0.1, max_iter=300)
```

---

## Part 9: 高级主题

### 9.1 多相Chan-Vese

**扩展到K个区域**：

使用K-1个水平集函数：
```
φ₁, φ₂, ..., φ_{K-1}

每个区域定义为:
Ω_i = {x: φ_j(x) 的符号组合}
```

### 9.2 局部Chan-Vese

**动机**：处理强度不均匀

**方法**：在每个像素周围的小窗口内拟合
```
E_i(x) = ∫_{N(x)} (f(y)-c₁)²H(φ(y)) + (f(y)-c₂)²(1-H(φ(y))) dy
```

### 9.3 活动轮廓模型的现代发展

**从Chan-Vese到深度学习**：

1. **经典变分法**：手工设计能量
2. **学习变分法**：学习能量参数
3. **深度学习**：端到端学习

---

## 课后练习

### 练习1：理解推导

推导Chan-Vese的演化方程：
```
从 E(φ) = μ∫δ(φ)|∇φ| + λ₁∫(f-c₁)²H(φ) + λ₂∫(f-c₂)²(1-H(φ))
到 ∂φ/∂t = δ(φ)[μ·div(∇φ/|∇φ|) - λ₁(f-c₁)² + λ₂(f-c₂)²]
```

### 练习2：代码实现

实现并测试：
1. 基础Chan-Vese
2. 多相Chan-Vese
3. 局部Chan-Vese

### 练习3：参数分析

测试不同参数：
- μ对边界光滑度的影响
- λ₁, λ₂对区域比例的影响
- 初始化对收敛的影响

---

## 参考文献深入阅读

1. **Mumford-Shah (1989)**: 原始论文，第1-3章
2. **Chan-Vese (2001)**: 经典论文，必读
3. **水平集方法**: Osher & Sethian (1988)

---

**下一课预告**：凸松弛与图割 - 让M-S可求解
