# 第四讲：小波与框架变换

## Wavelets and Frame Transforms

---

### 📋 本讲大纲

1. 小波变换基础
2. 多分辨率分析
3. 紧框架(Framelet)
4. 图像处理应用
5. Cai的框架方法

---

### 4.1 小波变换简介

#### 从傅里叶到小波

| 变换 | 基函数 | 优点 | 缺点 |
|------|--------|------|------|
| 傅里叶 | $e^{i\omega x}$ | 频率定位 | 无空间定位 |
| Gabor | $e^{i\omega x} g(x-x_0)$ | 时空定位 | 冗余 |
| 小波 | $\psi_{j,k}(x)$ | 多分辨率 | - |

**动画建议**：展示三种变换在边缘检测上的表现差异

---

### 4.2 连续小波变换

#### 定义

对于小波母函数 $\psi(t)$：
$$W_f(a,b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} f(t) \psi^*\left(\frac{t-b}{a}\right) dt$$

#### 参数含义

- **a**：尺度参数（伸缩）—— 对应频率
- **b**：平移参数 —— 对应空间位置
- **ψ**：小波母函数

#### 可容许条件

$$C_\psi = \int_0^\infty \frac{|\hat{\psi}(\omega)|^2}{\omega} d\omega < \infty$$

---

### 4.3 离散小波变换

#### 离散化

取 $a = 2^j$, $b = k \cdot 2^j$：
$$\psi_{j,k}(t) = 2^{-j/2} \psi(2^{-j}t - k)$$

#### 两尺度关系

尺度函数 $\phi$ 和小波函数 $\psi$：
$$\phi(t) = \sqrt{2} \sum_k h_k \phi(2t - k)$$
$$\psi(t) = \sqrt{2} \sum_k g_k \phi(2t - k)$$

其中 $g_k = (-1)^k h_{1-k}$

---

### 4.4 多分辨率分析(MRA)

#### 定义

MRA是 $L^2(\mathbb{R})$ 的一列闭子空间 $\{V_j\}_{j \in \mathbb{Z}}$，满足：

```
1. 嵌套性：V_j ⊂ V_{j+1}
2. 稠密性：∪V_j = L²(R)
3. 伸缩性：f(x) ∈ V_j ⟺ f(2x) ∈ V_{j+1}
4. 平移性：f(x) ∈ V_0 ⟺ f(x-k) ∈ V_0
5. Riesz基：存在φ使{φ(x-k)}是V_0的Riesz基
```

**动画建议**：展示MRA的金字塔分解结构

---

### 4.5 小波分解与重构

#### 分解

$$V_{j+1} = V_j \oplus W_j$$

- $V_j$：近似空间（低频）
- $W_j$：细节空间（高频）

#### 快速算法（Mallat算法）

```
分解：          重构：
c_{j+1} → c_j  c_j + d_j → c_{j+1}
       ↘ d_j
```

---

### 4.6 常见小波族

| 小波 | 特点 | 应用 |
|------|------|------|
| Haar | 最简单，不连续 | 边缘检测 |
| Daubechies | 紧支撑，正则 | 信号压缩 |
| Symlets | 近对称 | 去噪 |
| Coiflets | 对称，正则 | 特征提取 |
| Biorthogonal | 对称，线性相位 | 图像压缩(JPEG2000) |

---

### 4.7 框架理论

#### 框架定义

$\{f_i\}_{i \in I}$ 是Hilbert空间 $H$ 的**框架**，若存在 $0 < A \leq B$：
$$A\|x\|^2 \leq \sum_{i \in I} |\langle x, f_i \rangle|^2 \leq B\|x\|^2$$

#### 框架算子

$$Sx = \sum_i \langle x, f_i \rangle f_i$$

#### 重构公式

$$x = S^{-1} S x = \sum_i \langle x, f_i \rangle S^{-1} f_i$$

---

### 4.8 紧框架(Frames/Frames)

#### 定义

若 $A = B$，称为**紧框架**：
$$x = \frac{1}{A} \sum_i \langle x, f_i \rangle f_i$$

#### 特点

- 重构简单（直接加权求和）
- 比正交基更灵活
- 冗余性提供稳定性

#### Framelet（框架小波）

满足紧框架性质的离散小波系统

---

### 4.9 B样条框架小波

#### Cai的方法

基于B样条的紧框架构造：

$$\phi^{(n)}(x) = \underbrace{\chi_{[0,1]} * \chi_{[0,1]} * \cdots * \chi_{[0,1]}}_{n \text{次卷积}}$$

#### 滤波器性质

| 阶数n | 消失矩 | 光滑性 |
|-------|--------|--------|
| 1 | 1 | $C^0$ |
| 2 | 2 | $C^1$ |
| 3 | 3 | $C^2$ |

**动画建议**：展示不同阶数B样条框架小波的波形

---

### 4.10 框架在图像处理中的应用

#### 图像去噪

稀疏框架模型：
$$\min_u \frac{1}{2}\|u - f\|_2^2 + \lambda \|W u\|_1$$

其中 $W$ 是框架变换，$\|W u\|_1$ 促进稀疏性

#### 图像修复(Inpainting)

$$\min_u \|W u\|_1 \quad \text{s.t.} \quad u|_\Omega = f|_\Omega$$

#### 优势

- 多尺度稀疏表示
- 冗余性提高稳定性
- 快速算法（FFT或滤波器组）

---

### 4.11 Split Bregman框架去噪

#### 算法框架

引入 $d = Wu$：

```
1. u子问题： u = (I + μW^T W)^{-1}(f + W^T(d - b))
         或 u = f + W^T(d - b)（紧框架简化）
         
2. d子问题： d = shrink(Wu + b, λ/μ)

3. b更新：  b = b + Wu - d
```

#### Cai的贡献

- 收敛性分析
- 快速实现
- 多种应用扩展

---

### 4.12 图像的框架分解示例

#### 分解层次

```
原始图像 f
    │
    ├── 低频近似 c_J
    │
    └── 高频细节 {d_j}_{j=1}^{J}
         ├── 水平细节
         ├── 垂直细节
         └── 对角细节
```

**动画建议**：展示Lena图像的小波分解各层

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           小波与框架核心概念                      │
├─────────────────────────────────────────────────┤
│                                                 │
│   小波变换：多分辨率时频分析                      │
│   ├── 连续小波：CWT，高冗余                      │
│   └── 离散小波：DWT，高效                        │
│                                                 │
│   多分辨率分析(MRA)：                            │
│   V_{j+1} = V_j ⊕ W_j                           │
│                                                 │
│   紧框架：A=B，简单重构                          │
│   x = (1/A) Σ⟨x, f_i⟩f_i                        │
│                                                 │
│   图像处理应用：                                 │
│   min_u ‖u-f‖²_2 + λ‖Wu‖_1                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **理论题**：证明Haar小波构成$L^2(\mathbb{R})$的正交基

2. **推导题**：推导紧框架的重构公式

3. **编程题**：实现2D小波分解与重构（可用PyWavelets）

4. **实验题**：比较Daubechies-4和Haar小波在图像去噪中的表现

---

### 📖 扩展阅读

1. **经典教材**：
   - Mallat, *A Wavelet Tour of Signal Processing*, Academic Press
   - Daubechies, *Ten Lectures on Wavelets*, SIAM

2. **Cai相关论文**：
   - Cai, Chan, Shen, "A framelet-based image inpainting algorithm", ACHA, 2008
   - Cai, Shen, "Framelets: MRA-based constructions of wavelet frames", ACHA, 2010

3. **软件工具**：
   - PyWavelets (Python)
   - Wavelet Toolbox (MATLAB)

---

### 📖 参考文献

1. Daubechies, I. (1992). *Ten Lectures on Wavelets*. SIAM.

2. Mallat, S. (2008). *A Wavelet Tour of Signal Processing*, 3rd ed. Academic Press.

3. Cai, J.F., Chan, R.H., & Shen, Z. (2008). A framelet-based image inpainting algorithm. *ACHA*, 24(3), 268-283.

4. Cai, J.F., Dong, B., Osher, S., & Shen, Z. (2012). Image restoration: Total variation, wavelet frames, and beyond. *JAMS*, 25(4), 1033-1061.
