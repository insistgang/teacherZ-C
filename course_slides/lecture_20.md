# 第二十讲：图像重建

## Image Reconstruction

---

### 📋 本讲大纲

1. 图像重建问题
2. 解析重建方法
3. 迭代重建方法
4. 压缩感知重建
5. 深度学习重建

---

### 20.1 重建问题

#### 一般形式

$$y = Ax + n$$

- $y$：观测数据（投影/K空间）
- $A$：成像算子
- $x$：待重建图像
- $n$：噪声

#### 逆问题

已知 $y$ 和 $A$，求解 $x$

#### 挑战

- 不适定性
- 噪声放大
- 欠采样

---

### 20.2 CT重建：滤波反投影

#### Radon变换

$$p(\theta, s) = \int f(x) \delta(x \cdot n_\theta - s) dx$$

#### 反投影

$$f_{BP}(x) = \int_0^{\pi} p(\theta, x \cdot n_\theta) d\theta$$

#### 滤波反投影 (FBP)

$$f_{FBP}(x) = \int_0^{\pi} [p(\theta, \cdot) * h](x \cdot n_\theta) d\theta$$

斜坡滤波器 $h$ 的频率响应：$H(\omega) = |\omega|$

---

### 20.3 MRI重建：傅里叶重建

#### K空间采样

$$S(k) = \int \rho(x) e^{-i2\pi k \cdot x} dx$$

#### 直接重建

$$\rho(x) = \mathcal{F}^{-1}\{S(k)\}$$

#### 欠采样问题

- 混叠伪影
- 需要正则化

---

### 20.4 迭代重建

#### 变分框架

$$\min_x \frac{1}{2}\|Ax - y\|_2^2 + \lambda R(x)$$

- 数据项：$\|Ax - y\|_2^2$
- 正则项：$R(x)$（TV、小波稀疏等）

#### 常用算法

- 迭代最小二乘 (ILS)
- 代数重建技术 (ART)
- 同时迭代重建 (SIRT)

---

### 20.5 ART算法

#### 代数重建技术

$$x^{k+1} = x^k + \lambda_k \frac{y_i - a_i^T x^k}{\|a_i\|^2} a_i$$

逐行更新

#### SIRT

$$x^{k+1} = x^k + A^T \Lambda (y - Ax^k)$$

同时更新所有投影

---

### 20.6 压缩感知重建

#### 稀疏性假设

图像在某变换域稀疏：
$$x = \Psi \alpha, \quad \|\alpha\|_0 \ll N$$

#### 欠采样重建

$$\min_\alpha \|A\Psi\alpha - y\|_2^2 + \lambda\|\alpha\|_1$$

#### 条件

- 图像稀疏
- 测量矩阵满足RIP
- 足够采样

---

### 20.7 MRI压缩感知

#### 问题

加速采集 → K空间欠采样

#### 模型

$$\min_x \|Ax - y\|_2^2 + \lambda_1\|Wx\|_1 + \lambda_2\|x\|_{TV}$$

- $A$：欠采样傅里叶变换
- $W$：小波变换

#### 求解

- ADMM
- FISTA
- 非线性共轭梯度

---

### 20.8 CT压缩感知

#### 低剂量CT

减少辐射 → 噪声增加

#### 模型

$$\min_x \|Ax - y\|_2^2 + \lambda\|x\|_{TV} + \mu\|Wx\|_1$$

#### 效果

- 噪声抑制
- 条纹伪影减少
- 保持细节

---

### 20.9 深度学习重建

#### 端到端网络

$$\hat{x} = f_\theta(y)$$

直接学习从观测到图像的映射

#### 展开网络

将迭代算法展开为网络：
$$x^{k+1} = f_\theta(x^k, y)$$

#### 网络架构

| 类型 | 例子 |
|------|------|
| CNN | U-Net |
| Unrolled | ADMM-Net |
| Implicit | NeRF |

---

### 20.10 经典深度学习方法

#### U-Net重建

```
编码器 → 瓶颈 → 解码器
      ↘ 跳跃连接 ↙
```

#### MoDL

Model-Based Deep Learning：
- 数据一致性模块
- 正则化模块（CNN）
- 交替优化

#### Learned PD

学习的原始-对偶：
- 学习步长
- 学习算子

---

### 20.11 重建质量评估

#### 定量指标

| 指标 | 公式 |
|------|------|
| PSNR | $10\log_{10}(MAX^2/MSE)$ |
| SSIM | 结构相似度 |
| RMSE | $\sqrt{\sum(x-\hat{x})^2/N}$ |

#### 任务特定指标

- 分割准确率
- 检测灵敏度
- 临床评分

---

### 20.12 挑战与趋势

#### 挑战

- 泛化能力
- 极端欠采样
- 运动伪影

#### 趋势

- 自监督学习
- 物理约束网络
- 不确定性量化

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           图像重建方法                           │
├─────────────────────────────────────────────────┤
│                                                 │
│   解析方法：                                     │
│   • FBP (CT)                                    │
│   • 傅里叶重建 (MRI)                            │
│                                                 │
│   迭代方法：                                     │
│   • 变分框架：min ½||Ax-y||² + λR(x)           │
│   • ART, SIRT                                  │
│                                                 │
│   压缩感知：                                     │
│   • min ||Ax-y||² + λ||Ψx||₁                  │
│                                                 │
│   深度学习：                                     │
│   • 端到端网络                                  │
│   • 算法展开                                    │
│   • 物理约束                                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **实现题**：实现FBP重建算法

2. **实现题**：实现基于TV的MRI压缩感知重建

3. **实验题**：比较不同正则化项的重建效果

4. **研究题**：调研最新的深度学习重建方法

---

### 📖 扩展阅读

1. **经典教材**：
   - Kak & Slaney, *Principles of Computerized Tomographic Imaging*
   - Lustig et al., "Compressed Sensing MRI", IEEE Signal Processing Magazine, 2008

2. **深度学习**：
   - Jin et al., "Deep Convolutional Neural Network for Inverse Problems", IEEE TIP, 2017
   - Aggarwal et al., "MoDL", IEEE TMI, 2019

3. **代码**：
   - pyLAD - Learning-based重建
   - DeepInPy - 深度逆问题

---

### 📖 参考文献

1. Kak, A.C. & Slaney, M. (2001). *Principles of Computerized Tomographic Imaging*. SIAM.

2. Lustig, M., et al. (2007). Sparse MRI: The application of compressed sensing for rapid MR imaging. *Magnetic Resonance in Medicine*, 58(6), 1182-1195.

3. Jin, K.H., et al. (2017). Deep convolutional neural network for inverse problems in imaging. *IEEE TIP*, 26(9), 4509-4522.

4. Aggarwal, H.K., et al. (2019). MoDL: Model-based deep learning architecture for inverse problems. *IEEE TMI*, 38(2), 394-405.
