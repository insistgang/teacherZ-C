# 第十一讲：图像逆问题

## Image Inverse Problems

---

### 📋 本讲大纲

1. 逆问题概述
2. 图像去噪
3. 图像去模糊
4. 图像超分辨率
5. 统一优化框架

---

### 11.1 什么是逆问题？

#### 定义

已知观测 $g$，求解原始信号 $f$：
$$g = \mathcal{A}(f) + \eta$$

其中 $\mathcal{A}$ 是正向算子，$\eta$ 是噪声。

#### 不适定性

大多数图像逆问题是**不适定(ill-posed)**的：
- 解不存在
- 解不唯一
- 解不连续依赖于数据

---

### 11.2 图像退化模型

#### 一般模型

$$g = Hf + n$$

| 问题 | $H$ | 物理意义 |
|------|-----|----------|
| 去噪 | $I$ | 加性噪声 |
| 去模糊 | 卷积核 | 运动模糊、散焦 |
| 超分辨 | 下采样+模糊 | 低分辨率采集 |
| 压缩感知 | 随机测量 | 欠采样 |

**动画建议**：展示不同退化类型的效果

---

### 11.3 图像去噪

#### 问题

$$g = f + n, \quad n \sim \mathcal{N}(0, \sigma^2)$$

#### 变分框架

$$\min_u \frac{1}{2}\|u - g\|_2^2 + \lambda \Phi(u)$$

#### 常用正则化

| 方法 | $\Phi(u)$ | 特点 |
|------|-----------|------|
| Tikhonov | $\|\nabla u\|_2^2$ | 过度平滑 |
| ROF | $\|u\|_{TV}$ | 边缘保持 |
| 框架去噪 | $\|Wu\|_1$ | 多尺度稀疏 |
| 非局部 | NL-means | 纹理保持 |

---

### 11.4 图像去模糊

#### 问题

$$g = h * f + n$$

其中 $h$ 是模糊核（已知或未知）

#### 变分模型

$$\min_u \frac{1}{2}\|h * u - g\|_2^2 + \lambda \|u\|_{TV}$$

#### Euler-Lagrange方程

$$(h^T * (h * u - g)) - \lambda \nabla \cdot \left(\frac{\nabla u}{|\nabla u|}\right) = 0$$

#### FFT加速

利用卷积定理：
$$h * u = \mathcal{F}^{-1}(\hat{h} \cdot \hat{u})$$

---

### 11.5 盲去模糊

#### 问题

模糊核 $h$ 未知：
$$g = h * f + n$$

#### 联合优化

$$\min_{u, h} \frac{1}{2}\|h * u - g\|_2^2 + \lambda_1 \|u\|_{TV} + \lambda_2 \|h\|_1$$

#### 交替优化

```
repeat
  u-step: 固定h，更新u
  h-step: 固定u，更新h
until 收敛
```

---

### 11.6 图像超分辨率

#### 问题

从低分辨率图像 $g$ 恢复高分辨率图像 $f$：
$$g = D H f + n$$

- $H$：模糊矩阵
- $D$：下采样矩阵

#### 插值方法

- 双线性插值
- 双三次插值
- Lanczos重采样

#### 变分方法

$$\min_u \frac{1}{2}\|DHu - g\|_2^2 + \lambda \|u\|_{TV}$$

---

### 11.7 单图像超分辨率

#### 挑战

只有一张低分辨率图像，问题是严重不适定的

#### 自相似性方法

```
1. 在图像内寻找相似块
2. 利用高分辨率块关系
3. 聚合重建
```

#### 深度学习方法

- SRCNN
- ESRGAN
- SwinIR

---

### 11.8 压缩感知

#### 信号稀疏性

自然图像在适当变换下稀疏：
$$f = W^T \alpha, \quad \|\alpha\|_0 \ll N$$

#### 测量模型

$$g = \Phi f = \Phi W^T \alpha$$

#### 重构

$$\min_\alpha \|g - \Phi W^T \alpha\|_2^2 + \lambda \|\alpha\|_1$$

#### RIP条件

测量矩阵 $\Phi W^T$ 需要满足约束等距性质(RIP)

---

### 11.9 统一优化框架

#### 一般形式

$$\min_u E_{data}(u) + E_{prior}(u)$$

| 项 | 形式 | 作用 |
|---|------|------|
| 数据项 | $\|Au - g\|_2^2$ | 数据一致性 |
| 先验项 | $\Phi(u)$ | 正则化/先验知识 |

#### 常用求解器

- 梯度下降
- 近端梯度
- ADMM
- Split Bregman

---

### 11.10 算法选择指南

| 问题规模 | 推荐算法 |
|----------|----------|
| 小规模 | 内点法 |
| 中等规模 | ADMM/Split Bregman |
| 大规模 | 随机梯度下降 |

| 正则化类型 | 推荐算法 |
|------------|----------|
| $\ell_2$ | 共轭梯度 |
| $\ell_1$ | 近端梯度/ISTA |
| TV | Split Bregman/Chambolle |

---

### 11.11 图像修复(Inpainting)

#### 问题

图像部分区域缺失，需要恢复：
$$\min_u \int_\Omega (u - g)^2 dx + \lambda \|u\|_{TV}$$

约束：$u|_{\Omega \setminus D} = g|_{\Omega \setminus D}$

#### 变分方法

$$E(u) = \int_D |\nabla u| dx + \frac{1}{2} \int_{\Omega \setminus D} (u - g)^2 dx$$

#### Cai的框架方法

$$\min_u \|W u\|_1 \quad \text{s.t.} \quad u|_{\Omega \setminus D} = g$$

---

### 11.12 性能评估指标

#### 评估指标

| 指标 | 公式 | 范围 |
|------|------|------|
| PSNR | $10\log_{10}(\frac{MAX^2}{MSE})$ | 越高越好 |
| SSIM | 结构相似度 | [0,1] |
| LPIPS | 感知相似度 | 越低越好 |

#### 主观评估

- MOS (Mean Opinion Score)
- 用户研究

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           图像逆问题框架                         │
├─────────────────────────────────────────────────┤
│                                                 │
│   一般模型：g = Hf + n                          │
│                                                 │
│   优化框架：                                     │
│   min_u E_data(u) + λE_prior(u)                │
│                                                 │
│   主要问题：                                     │
│   • 去噪：H = I                                │
│   • 去模糊：H = 卷积核                          │
│   • 超分辨：H = 下采样 + 模糊                   │
│   • 压缩感知：H = 测量矩阵                      │
│                                                 │
│   求解方法：                                     │
│   变分方法 → 近端算法 → ADMM/Split Bregman      │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **实现题**：实现基于ROF的图像去模糊算法

2. **比较题**：比较不同正则化项在去噪中的表现

3. **分析题**：分析PSNR和SSIM的优缺点

4. **编程题**：实现单图像超分辨的自相似性方法

---

### 📖 扩展阅读

1. **经典教材**：
   - Osher & Paragios, *Geometric Level Set Methods in Imaging, Vision, and Graphics*
   - Szeliski, *Computer Vision: Algorithms and Applications*

2. **Cai相关论文**：
   - Cai, Chan, Shen, "Framelet-based image inpainting"

3. **深度学习**：
   - U-Net, ESRGAN, SwinIR

---

### 📖 参考文献

1. Osher, S. & Paragios, N. (2003). *Geometric Level Set Methods*. Springer.

2. Rudin, L.I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. *Physica D*.

3. Cai, J.F., Chan, R.H., & Shen, Z. (2008). A framelet-based image inpainting algorithm. *ACHA*.

4. Yang, C.Y., et al. (2010). Edge-directed single image super-resolution. *CVPR*.
