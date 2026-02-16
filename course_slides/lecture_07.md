# 第七讲：两阶段分割方法

## Two-Stage Segmentation: SaT Framework

---

### 📋 本讲大纲

1. 两阶段方法概述
2. SaT (Smoothing and Thresholding) 框架
3. 理论分析
4. 算法实现
5. Cai的贡献

---

### 7.1 两阶段方法动机

#### 传统方法的挑战

变分分割方法（如Chan-Vese）需要：
- 迭代求解非线性偏微分方程
- 复杂的水平集初始化和重新初始化
- 计算量大

#### 两阶段思想

```
阶段1: 平滑/预处理 → 得到连续的分割函数
阶段2: 阈值化     → 得到最终分割结果
```

**优势**：将复杂问题分解为两个简单子问题

---

### 7.2 SaT框架

#### Smoothing and Thresholding

**Cai, Chan, Shen等**提出的框架：

$$\textbf{SaT: } \underbrace{\text{Smoothing}}_{\text{阶段1}} \rightarrow \underbrace{\text{Thresholding}}_{\text{阶段2}}$$

#### 阶段1：平滑

$$\min_u \frac{1}{2}\|u - f\|_2^2 + \lambda \Phi(u)$$

其中 $\Phi(u)$ 是正则化项（如TV、框架变换）

#### 阶段2：阈值化

$$\Sigma = \{x : u(x) > T\}$$

---

### 7.3 SaT与Chan-Vese的联系

#### 理论发现

**Cai等的贡献**：在适当条件下，SaT方法与Chan-Vese方法等价！

$$\boxed{\text{SaT} \approx \text{Chan-Vese}}$$

#### 关键条件

1. 平滑阶段的正则化强度足够
2. 阈值选择适当
3. 图像具有某种"二值性"

**动画建议**：展示同一图像用两种方法得到相同结果

---

### 7.4 阈值选择

#### Otsu阈值

$$T^* = \arg\max_T \sigma_b^2(T)$$

#### 均值阈值

$$T = \text{mean}(u)$$

#### 与区域均值的关系

对于两相分割，最优阈值满足：
$$T = \frac{c_1 + c_2}{2}$$

其中 $c_1, c_2$ 是两相的均值

---

### 7.5 平滑方法选择

#### 常用平滑方法

| 方法 | 正则项 | 特点 |
|------|--------|------|
| ROF | $\|u\|_{TV}$ | 边缘保持 |
| 框架去噪 | $\|Wu\|_1$ | 多尺度 |
| 双边滤波 | 非局部 | 边缘保持 |
| 非局部均值 | 非局部 | 纹理保持 |

#### Cai的框架方法

$$\min_u \frac{1}{2}\|u - f\|_2^2 + \lambda \|W u\|_1$$

使用紧框架 $W$

---

### 7.6 理论保证

#### 主要定理 (Cai et al.)

设 $f = c_1 \chi_{\Omega_1} + c_2 \chi_{\Omega_2} + \eta$，其中 $\eta$ 是噪声。若：

1. $|c_1 - c_2| > \sigma(\eta)$（对比度足够）
2. 正则化参数 $\lambda$ 适当选择

则SaT方法的输出收敛到真实分割 $\Omega_1, \Omega_2$。

#### 误差界

$$\|\Sigma - \Sigma^*\| \leq C \cdot \frac{\sigma(\eta)}{|c_1 - c_2|}$$

---

### 7.7 算法流程

#### 完整SaT算法

```
输入: 图像 f, 正则化参数 λ
输出: 分割结果 Σ

阶段1 (平滑):
  求解 min_u ½‖u-f‖² + λΦ(u)
  可用方法: ROF, Split Bregman, 框架去噪

阶段2 (阈值化):
  计算阈值 T (Otsu或均值)
  Σ = {x : u(x) > T}

返回 Σ
```

**动画建议**：逐步展示两阶段过程

---

### 7.8 多相分割扩展

#### 多阈值

对于 $K$ 相分割，需要 $K-1$ 个阈值：
$$\Sigma_k = \{x : T_{k-1} < u(x) \leq T_k\}$$

#### 阈值选择方法

- K-means聚类
- 直方图分析
- 自动阈值检测（如Otsu多阈值）

---

### 7.9 SaT与PCMS的关系

#### PCMS模型

Piecewise Constant Mumford-Shah：
$$\min_{\{c_i\}, K} \sum_i \int_{R_i} (f - c_i)^2 dx + |K|$$

#### 关系

$$\text{SaT} \xrightarrow{\text{等价}} \text{PCMS} \xrightarrow{\text{特例}} \text{Chan-Vese}$$

#### 意义

- 提供了简化求解PCMS的途径
- 揭示了阈值方法的数学基础

---

### 7.10 计算效率比较

| 方法 | 时间复杂度 | 实现难度 |
|------|------------|----------|
| Chan-Vese | $O(N \cdot \text{迭代次数})$ | 中等 |
| 水平集 | $O(N \cdot \text{迭代次数})$ | 高 |
| SaT | $O(N \log N)$ | 低 |

其中 $N$ 是像素数

#### SaT优势

- **无迭代阈值化**：阈值操作是$O(N)$
- **快速平滑**：可用FFT或快速滤波器
- **无需初始化**：不依赖初始轮廓

---

### 7.11 实验结果

#### 合成图像

- 简单几何形状：100%准确
- 加噪声后：依赖信噪比

#### 真实图像

- 医学图像：效果良好
- 自然场景：需要预处理

**动画建议**：展示不同类型图像的分割结果对比

---

### 7.12 SaT的局限性

#### 挑战场景

1. **低对比度**：$|c_1 - c_2|$ 很小
2. **弱边界**：边界处平滑导致模糊
3. **多模态**：灰度分布不是双峰
4. **复杂纹理**：无法用灰度区分

#### 解决方向

- 结合边缘信息
- 多特征融合
- 深度学习增强

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           SaT框架核心思想                        │
├─────────────────────────────────────────────────┤
│                                                 │
│   两阶段分解：                                   │
│   f ──[平滑]──→ u ──[阈值]──→ Σ               │
│                                                 │
│   理论联系：                                     │
│   SaT ≈ PCMS ≈ Chan-Vese                       │
│                                                 │
│   优势：                                        │
│   • 计算高效                                     │
│   • 实现简单                                     │
│   • 无需初始化                                   │
│                                                 │
│   局限：                                        │
│   • 低对比度场景                                 │
│   • 复杂纹理                                     │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **实现题**：实现基于ROF的SaT分割算法

2. **比较题**：对比SaT与Chan-Vese在含噪图像上的表现

3. **分析题**：分析阈值选择对分割结果的影响

4. **扩展题**：将SaT扩展到三相位分割

---

### 📖 扩展阅读

1. **Cai相关论文**：
   - Cai, Chan, Zeng, "A two-stage image segmentation method using a convex variant of the Mumford-Shah model and thresholding", SIAM J. Imaging Sci., 2013
   - Cai, "Two-stage image segmentation", 2014

2. **相关工作**：
   - Chan-Vese模型
   - PCMS模型

---

### 📖 参考文献

1. Cai, X., Chan, R.H., & Zeng, T. (2013). A two-stage image segmentation method using a convex variant of the Mumford-Shah model and thresholding. *SIAM J. Imaging Sci.*, 6(1), 368-390.

2. Cai, X. (2014). Two-stage image segmentation: a convex variant of the Mumford-Shah model. *SIAM News*.

3. Chan, T.F. & Vese, L.A. (2001). Active contours without edges. *IEEE TIP*.
