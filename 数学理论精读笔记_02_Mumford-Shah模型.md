# 数学理论精读笔记 02：Mumford-Shah分割模型

> **目标**: 深入理解Mumford-Shah泛函的数学理论与分割原理
> **前置知识**: 精读笔记01 (ROF模型)、泛函分析、偏微分方程
> **学习时间**: 3-4周

---

## 目录

1. [从ROF到Mumford-Shah：问题的扩展](#1-从rof到mumford-shah问题的扩展)
2. [Mumford-Shah泛函的完整形式](#2-mumford-shah泛函的完整形式)
3. [自由边界问题](#3-自由边界问题)
4. [Mumford-Shah的简化版本：Chan-Vese](#4-mumford-shah的简化版本chan-vese)
5. [凸松弛方法](#5-凸松弛方法)
6. [数值实现策略](#6-数值实现策略)
7. [理论与应用的联系](#7-理论与应用的联系)
8. [习题与研究问题](#8-习题与研究问题)

---

## 1. 从ROF到Mumford-Shah：问题的扩展

### 1.1 ROF的局限性

**ROF解决的问题**: 去噪
$$\min_u \int |\nabla u| + \frac{\lambda}{2}(u-f)^2$$

**核心假设**: 图像本身是连续的，只有噪声需要去除

**但实际问题**: 图像不仅需要去噪，更需要**分割**

**示例**: 医学图像分析
- 输入: 脑部MRI扫描
- 目标: 识别肿瘤区域
- ROF输出: 平滑的图像（但不知道肿瘤在哪里）
- 需要: 明确的边界（分割）

### 1.2 分割的本质

**什么是图像分割？**

数学定义: 将图像域 Ω 分割为不相交的子集
$$\Omega = \Omega_1 \cup \Omega_2 \cup \cdots \cup \Omega_K$$
$$\Omega_i \cap \Omega_j = \emptyset \quad (i \neq j)$$

**每个子集的特性**:
- 内部光滑（低频）
- 边界清晰（高频、不连续）

**Mumford-Shah的洞察**:
- 同时优化**分割边界** Γ 和 **图像近似** u
- Γ 是边缘集合（不连续点集）
- u 在 Ω\Γ 上光滑

### 1.3 经典分割方法的数学表述

**边缘检测方法** (Canny, Sobel等):
```
步骤1: 计算梯度 |∇f|
步骤2: 阈值化
步骤3: 边缘连接
```

**问题**:
- 梯度阈值难以选择
- 边缘不连续
- 没有全局优化

**区域增长方法**:
```
步骤1: 选择种子点
步骤2: 基于相似性增长
步骤3: 合并/分裂区域
```

**问题**:
- 种子点选择敏感
- 没有能量泛函指导
- 容易过分割或欠分割

**Mumford-Shah的优势**:
- 全局能量泛函
- 同时优化边界和近似
- 数学理论完备

---

## 2. Mumford-Shah泛函的完整形式

### 2.1 原始定义 (1989)

**Mumford-Shah泛函**:
$$E(u, \Gamma) = \int_{\Omega \setminus \Gamma} |\nabla u|^2 \, dx + \mu \int_{\Omega} (u - f)^2 \, dx + \nu |\Gamma|$$

其中:
- **u**: 分段光滑的近似图像
- **Γ**: 边缘集合（不连续点集）
- **f**: 原始观察图像

**三项的解释**:

1. **光滑性项**: ∫\_{Ω\Γ} |∇u|² dx
   - 作用: 使u在非边缘处光滑
   - 为什么是 L₂ 范数: 保证 u 在 Ω\Γ 上是调和函数
   - 与ROF的对比: ROF用 L₁，M-S用 L₂

2. **保真项**: μ ∫\_Ω (u-f)² dx
   - 作用: 保持 u 与 f 相似
   - μ: 平滑参数（控制保真度）

3. **边缘长度项**: ν |Γ|
   - |Γ|: 边缘的Hausdorff测度（1维长度）
   - 作用: 惩罚过长的边缘（正则化）
   - ν: 边缘惩罚参数

### 2.2 Mumford-Shah vs ROF

| 特性 | ROF模型 | Mumford-Shah模型 |
|:---|:---|:---|
| **问题** | 去噪 | 分割+去噪 |
| **变量** | u (图像) | u + Γ (图像+边缘) |
| **正则化** | TV = ∫\|∇u\| | ∫\|∇u\|² + ν\|Γ\| |
| **不连续** | 允许，但隐式 | 显式建模 Γ |
| **光滑性** | 分段常数 | 分片光滑 |
| **难度** | 凸优化 | 非凸、自由边界 |

### 2.3 为什么Mumford-Shah难解？

**挑战1: 非凸性**
- 变量 Γ 是离散的（边缘存在或不存在）
- 能量泛函非凸
- 多个局部极小值

**挑战2: 自由边界问题**
- Γ 的位置和形状未知
- Γ 的拓扑（连通性）未知
- 没有固定网格可以表示任意 Γ

**挑战3: 优化变量耦合**
- u 和 Γ 相互依赖
- 给定 Γ，求 u 容易
- 给定 u，求 Γ 困难

**数学难度**: MS问题是NP难的（即使离散化）

---

## 3. 自由边界问题

### 3.1 什么是自由边界？

**定义**: 在偏微分方程中，边界不是预先给定的，而是解的一部分。

**对比**:
```
固定边界问题:
Ω 已知 → 在 ∂Ω 上求解 PDE

自由边界问题:
Ω 未知 → Ω 和 PDE解同时优化
```

**图像分割中的自由边界**:
- Γ 是图像中物体与背景的边界
- Γ 的位置由图像内容决定
- Γ 的形状可以任意复杂

### 3.2 Mumford-Shah的几何意义

**能量泛函的物理解释**:
$$E(u, \Gamma) = \text{内部光滑} + \text{数据保真} + \text{边界长度}$$

**类比**: 弹性膜问题
- 想象 u 是一个弹性膜
- 膜可以沿着边缘 Γ "撕裂"
- 膜的张力使其光滑（第一项）
- 数据 f 吸引膜（第二项）
- 撕裂需要能量（第三项）

**最优解的平衡**:
- 如果 |∇f| 大 → 值得撕裂 → Γ 经过
- 如果 |∇f| 小 → 不撕裂 → u 光滑

### 3.3 边界Γ的数学刻画

**Γ 的性质**:
1. Γ 是有限长度的 1 维集合
2. Γ 可以是曲线、折线、或更一般的集合
3. Γ 的 Hausdorff 维数是 1
4. Γ 可以有分叉、端点

**Γ 的正则性理论**:
- Mumford-Shah证明了Γ的某些正则性
- Γ 几乎处处是 C^1 曲线
- 在奇点处可能有限制

---

## 4. Mumford-Shah的简化版本：Chan-Vese

### 4.1 简化的动机

**完整M-S的困难**:
- 自由边界问题难解
- Γ 的表示复杂
- 数值优化困难

**Chan-Vese的简化** (2001):
- **假设**: u 是分段常数（piecewise constant）
- 即: u = c₁ 在 Ω₁ 内，u = c₂ 在 Ω₂ 内
- 边界 Γ 用水平集函数表示

### 4.2 Chan-Vese能量泛函

**两相分割** (foreground vs background):
$$E(c_1, c_2, \Gamma) = \mu \cdot |\Gamma| + \lambda_1 \int_{\text{inside}(\Gamma)} (f - c_1)^2 \, dx + \lambda_2 \int_{\text{outside}(\Gamma)} (f - c_2)^2 \, dx$$

**变量**:
- c₁: 前景平均灰度
- c₂: 背景平均灰度
- Γ: 分割轮廓

**与M-S的关系**:
- Chan-Vese是M-S的特殊情况
- 假设 u 在每个区域内是常数
- |∇u| = 0 几乎处处
- 因此去掉 ∫|∇u|² 项

### 4.3 水平集方法 (Level Set Method)

**思想**: 用隐函数表示边界

**水平集函数** φ(x):
$$ \begin{cases} \phi(x) > 0 & \text{if } x \in \Omega_1 \text{ (inside)} \\ \phi(x) < 0 & \text{if } x \in \Omega_2 \text{ (outside)} \\ \phi(x) = 0 & \text{if } x \in \Gamma \text{ (boundary)} \end{cases} $$

**Heaviside函数** H(φ):
$$H(\phi) = \begin{cases} 1 & \text{if } \phi \geq 0 \\ 0 & \text{if } \phi < 0 \end{cases}$$

**Dirac测度** δ(φ):
$$\delta(\phi) = \frac{d}{d\phi} H(\phi)$$

**Chan-Vese能量用水平集表示**:
$$E(\phi, c_1, c_2) = \mu \int_{\Omega} \delta(\phi)|\nabla \phi| \, dx + \lambda_1 \int_{\Omega} (f-c_1)^2 H(\phi) \, dx + \lambda_2 \int_{\Omega} (f-c_2)^2 (1-H(\phi)) \, dx$$

### 4.4 Chan-Vese的演化方程

**欧拉-拉格朗日方程**:
$$\frac{\partial \phi}{\partial t} = \delta(\phi) \left[ \mu \, \text{div}\left(\frac{\nabla \phi}{|\nabla \phi|}\right) - \lambda_1(f-c_1)^2 + \lambda_2(f-c_2)^2 \right]$$

**曲率项**:
$$\kappa = \text{div}\left(\frac{\nabla \phi}{|\nabla \phi|}\right)$$

**解释**:
- μκ: 曲率力（使边界平滑）
- -λ₁(f-c₁)²: 前景拉力
- +λ₂(f-c₂)²: 背景拉力
- δ(φ): 限制演化只在边界附近

**更新c₁和c₂**:
$$c_1 = \frac{\int f \cdot H(\phi) \, dx}{\int H(\phi) \, dx}$$
$$c_2 = \frac{\int f \cdot (1-H(\phi)) \, dx}{\int (1-H(\phi)) \, dx}$$

### 4.5 Chan-Vese的优缺点

**优点**:
- ✅ 不依赖梯度（可以检测模糊边缘）
- ✅ 全局优化（能量泛函）
- ✅ 水平集方法自动处理拓扑变化
- ✅ 可以扩展到多相分割

**缺点**:
- ❌ 假设分段常数（对强度变化敏感）
- ❌ 计算量大（需要重新初始化水平集）
- ❌ 参数调节困难

---

## 5. 凸松弛方法

### 5.1 为什么需要凸松弛？

**Mumford-Shah的问题**: 非凸 → 多个局部极小值 → 难以全局优化

**凸松弛的思想**:
- 将原问题嵌入到更大的凸问题中
- 在凸问题中找到全局最小值
- 如果松弛是"紧"的，则解对应原问题的解

### 5.2 两类松弛方法

#### **方法1: 变量松弛（Variable Relaxation）**

**原问题**:
$$\min_u \int_{\Omega} |\nabla u| \, dx + \frac{\lambda}{2} \int_{\Omega} (u-f)^2 \, dx$$

问题: |∇u| 在 ∇u=0 处不可微

**松弛**: 引入辅助变量 v = ∇u
$$\min_{u,v} \int_{\Omega} |v| \, dx + \frac{\lambda}{2} \int_{\Omega} (u-f)^2 \, dx$$
$$\text{s.t. } v = \nabla u$$

**拉格朗日形式**:
$$L(u,v) = \int_{\Omega} |v| \, dx + \frac{\lambda}{2} \int_{\Omega} (u-f)^2 \, dx + \frac{\beta}{2} \int_{\Omega} |v - \nabla u|^2 \, dx$$

其中 β 是惩罚参数。

#### **方法2: 值域松弛（Range Relaxation）**

**思想**: 允许 u 取[0,1]之间的值，而不是{0,1}

**原问题** (二值分割):
$$u \in \{0, 1\}$$

**松弛问题**:
$$u \in [0, 1]$$

**关键**:
- 如果最优解 u* ∈ {0,1} 几乎处处，则松弛是紧的
- 这需要某些条件（如数据保真项的凸性）

### 5.3 Split Bregman算法

**问题**: 优化TV正则化问题

**辅助变量**: v = ∇u

**增广拉格朗日**:
$$L(u,v,b) = \int_{\Omega} |v| \, dx + \frac{\lambda}{2} \int_{\Omega} (u-f)^2 \, dx + \frac{\mu}{2} \int_{\Omega} |v - \nabla u - b|^2 \, dx$$

**交替优化**:

**子问题1**: u更新（线性Poisson方程）
$$(\lambda I - \mu \Delta) u = \lambda f + \mu \, \text{div}(v-b)$$

**子问题2**: v更新（收缩公式）
$$v = \text{shrink}(\nabla u + b, 1/\mu)$$

其中收缩函数:
$$\text{shrink}(x, \kappa) = \frac{x}{|x|} \max(|x|-\kappa, 0)$$

**子问题3**: b更新（Bregman迭代）
$$b = b + \nabla u - v$$

**收敛性**:
- Bregman迭代保证收敛
- 通常2-10次迭代即可收敛

---

## 6. 数值实现策略

### 6.1 离散化方法

#### **有限差分法**

**前向差分**:
$$u_x \approx \frac{u_{i+1,j} - u_{i,j}}{h}$$

**中心差分**:
$$u_x \approx \frac{u_{i+1,j} - u_{i-1,j}}{2h}$$

**散度**:
$$\text{div}(p) \approx \frac{p_{i,j}^x - p_{i-1,j}^x}{h} + \frac{p_{i,j}^y - p_{i,j-1}^y}{h}$$

#### **有限元法 (FEM)**

**思想**: 将域分解为三角形，在每个三角形上用基函数近似

**优点**:
- 更精确
- 适合复杂几何
- 可以分析收敛性

**缺点**:
- 实现复杂
- 计算量大

### 6.2 边界Γ的表示方法

#### **显式表示** (参数化曲线)
- Γ = {γ(t): t ∈ [0,1]}
- 优点: 直观
- 缺点: 拓扑变化困难

#### **隐式表示** (水平集)
- Γ = {x: φ(x) = 0}
- 优点: 自动处理拓扑
- 缺点: 需要重新初始化

#### **离散表示** (图割)
- Γ 是像素网格上的边界
- 优点: 快速（max-flow算法）
- 缺点: 网格偏差

### 6.3 实现流程

**Chan-Vese实现**:
```
1. 初始化水平集函数 φ
2. For k = 1, 2, ... until convergence:
   a. 计算c₁, c₂（平均灰度）
   b. 更新φ（演化方程）
   c. 重新初始化φ（保持为符号距离函数）
   d. 检查收敛
3. 提取边界Γ = {x: φ(x) = 0}
```

**图割实现**:
```
1. 构建图:
   - 每个像素是一个节点
   - 源点s和汇点t
   - 边权重 = 数据项 + 平滑项
2. 运行max-flow/min-cut算法
3. 分割 = {x: x在s侧}
```

---

## 7. 理论与应用的联系

### 7.1 理论保证什么？

**存在性** (Existence):
- Mumford-Shah泛函在BV空间有最小值
- 解(u*, Γ*)存在

**正则性** (Regularity):
- Γ几乎处处是C¹曲线
- u在Ω\Γ上是光滑的

**唯一性** (Uniqueness):
- 一般不唯一（多个局部极小值）
- 但在凸松弛后唯一

### 7.2 实际应用中的问题

**参数选择**:
- μ, ν, λ如何选择？
- 没有理论指导
- 实践中用交叉验证或经验

**计算复杂度**:
- 理论算法慢
- 实际需要启发式
- 图割快但可能有网格偏差

**评估指标**:
- IoU (Intersection over Union)
- Dice系数
- Rand指数
- 但这些与能量泛函不完全对应

### 7.3 从理论到实践的鸿沟

**理论假设**:
- 图像是连续函数
- 噪声是高斯的
- 边缘是清晰的

**实际情况**:
- 图像是离散的
- 噪声可能非高斯
- 边缘模糊、纹理复杂

**桥接方法**:
- 数据驱动的方法
- 深度学习结合变分法
- 自适应参数

---

## 8. 习题与研究问题

### 基础题

1. **推导Chan-Vese演化方程**
   - 从能量泛函出发
   - 计算变分
   - 得到梯度下降流

2. **实现Chan-Vese**
   - 使用scikit-image的segmentation模块
   - 测试合成图像（圆形、方形）
   - 调节参数观察效果

3. **对比ROF和Chan-Vese**
   - ROF: 去噪
   - Chan-Vese: 分割
   - 它们的关系是什么？

### 进阶题

4. **证明**: 如果u是分段常数，Chan-Vese等价于Mumford-Shah

5. **分析**: 水平集函数重新初始化的必要性

6. **设计**: 多相Chan-Vese（K>2个区域）

7. **比较**: Chan-Vese vs 图割 vs 深度学习

### 研究题

8. **改进**: 如何处理强度不均匀性？
   - 提示: 局部拟合能量

9. **自适应参数**: 如何自动选择μ, ν, λ？
   - 提示: 使用尺度空间理论

10. **深度学习+变分法**:
    - 用神经网络学习能量泛函的参数
    - 或学习正则化项

---

## 参考文献与进一步阅读

**经典论文**:
1. Mumford, Shah (1989): "Optimal approximations by piecewise smooth functions and associated variational problems", Communications on Pure and Applied Mathematics
2. Chan, Vese (2001): "Active contours without edges", IEEE Trans. Image Processing

**凸松弛**:
3. Chan, Esedoḡlu, Nikolova (2006): "Algorithms for finding global minimizers of image segmentation and denoising models", SIAM J. Applied Mathematics

**教材**:
1. Chan, Shen (2005): "Image Processing and Analysis"
2. Osher, Sethian (1988): "Fronts propagating with curvature-dependent speed: algorithms based on Hamilton-Jacobi formulations" (Level Set经典论文)

---

**下一步**: 阅读"数学理论精读笔记03：凸优化与对偶理论"
