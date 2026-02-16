# 第六讲：Mumford-Shah模型

## Mumford-Shah Model

---

### 📋 本讲大纲

1. Mumford-Shah模型介绍
2. 能量泛函的数学形式
3. 理论性质与Γ收敛
4. 特殊情况与近似
5. 数值求解方法

---

### 6.1 模型背景

#### 动机

如何找到一个统一的数学框架来描述：
- 分割区域的平滑性
- 边界的长度
- 与原始图像的拟合

#### Mumford-Shah模型 (1989)

David Mumford和Jayant Shah提出的变分分割模型：
$$\min_{u, K} \left\{ \int_\Omega (u - f)^2 dx + \mu \int_{\Omega \setminus K} |\nabla u|^2 dx + \nu |K| \right\}$$

---

### 6.2 能量泛函分解

#### 三项含义

$$E(u, K) = \underbrace{\int_\Omega (u - f)^2 dx}_{\text{数据保真}} + \underbrace{\mu \int_{\Omega \setminus K} |\nabla u|^2 dx}_{\text{区域平滑}} + \underbrace{\nu |K|}_{\text{边界长度}}$$

| 项 | 含义 | 作用 |
|---|------|------|
| $\int (u-f)^2$ | 数据项 | 保持与原图一致 |
| $\int |\nabla u|^2$ | 平滑项 | 区域内平滑 |
| $\|K\|$ | 边界项 | 惩罚复杂边界 |

**动画建议**：展示三项权重变化对分割结果的影响

---

### 6.3 数学难点

#### 理论挑战

1. **边界的正则性**：$K$ 的形状如何？
2. **存在性**：最优解是否存在？
3. **数值求解**：如何处理不连续集合 $K$？

#### SBV函数空间

Special Bounded Variation空间：
$$BV(\Omega) = \{u : \|u\|_{BV} < \infty\}$$
$$SBV(\Omega) = \{u \in BV : Du = \nabla u dx + (u^+ - u^-) \nu_u d\mathcal{H}^{n-1}\llcorner S_u\}$$

---

### 6.4 Γ收敛理论

#### 定义

泛函序列 $F_n$ Γ收敛到 $F$，记为 $F_n \xrightarrow{\Gamma} F$，如果：

```
1. 下界不等式：对任意 x_n → x，F(x) ≤ lim inf F_n(x_n)
2. 恢复序列：对任意 x，存在 x_n → x 使得 F(x) ≥ lim sup F_n(x_n)
```

#### 意义

Γ收敛保证**极小点收敛**

---

### 6.5 Ambrosio-Tortorelli逼近

#### 核心思想

用辅助函数 $v$ 逼近边界集合 $K$

#### 逼近泛函

$$E_\epsilon(u, v) = \int_\Omega (u-f)^2 dx + \mu \int_\Omega v^2 |\nabla u|^2 dx + \nu \int_\Omega \left( \epsilon |\nabla v|^2 + \frac{(1-v)^2}{4\epsilon} \right) dx$$

#### Γ收敛结果

$$E_\epsilon \xrightarrow{\Gamma} E_{MS} \quad \text{as } \epsilon \to 0$$

**动画建议**：展示ε→0时v函数逼近边界的过程

---

### 6.6 特殊情况：分段常数

#### 当 $\mu \to \infty$ 时

$u$ 在每个区域内为常数：
$$E(K, \{c_i\}) = \sum_i \int_{R_i} (c_i - f)^2 dx + \nu |K|$$

#### 优化条件

最优常数：
$$c_i = \frac{1}{|R_i|} \int_{R_i} f(x) dx = \text{mean}(R_i)$$

---

### 6.7 Chan-Vese模型

#### 两相分段常数

$$E(c_1, c_2, C) = \lambda_1 \int_{inside(C)} (f - c_1)^2 dx + \lambda_2 \int_{outside(C)} (f - c_2)^2 dx + \nu |C|$$

#### 水平集形式

$$E(c_1, c_2, \phi) = \int_\Omega (f - c_1)^2 H(\phi) dx + \int_\Omega (f - c_2)^2 (1-H(\phi)) dx + \nu \int_\Omega |\nabla H(\phi)| dx$$

其中 $H$ 是Heaviside函数

---

### 6.8 Chan-Vese的水平集演化

#### Euler-Lagrange方程

$$\frac{\partial \phi}{\partial t} = \delta_\epsilon(\phi) \left[ \nu \nabla \cdot \left( \frac{\nabla \phi}{|\nabla \phi|} \right) - \lambda_1 (f-c_1)^2 + \lambda_2 (f-c_2)^2 \right]$$

#### 常数更新

$$c_1 = \frac{\int_\Omega f \cdot H(\phi) dx}{\int_\Omega H(\phi) dx}, \quad c_2 = \frac{\int_\Omega f \cdot (1-H(\phi)) dx}{\int_\Omega (1-H(\phi)) dx}$$

---

### 6.9 数值实现细节

#### 正则化

Heaviside函数的正则化：
$$H_\epsilon(\phi) = \frac{1}{2}\left(1 + \frac{2}{\pi}\arctan\left(\frac{\phi}{\epsilon}\right)\right)$$

Delta函数：
$$\delta_\epsilon(\phi) = \frac{d H_\epsilon}{d \phi} = \frac{1}{\pi} \frac{\epsilon}{\epsilon^2 + \phi^2}$$

#### 离散格式

有限差分 + 隐式/半隐式时间步进

---

### 6.10 多相Chan-Vese模型

#### 多水平集

使用 $m$ 个水平集函数 $\phi_1, \ldots, \phi_m$，可表示 $2^m$ 个区域

#### 能量泛函

$$E = \sum_{i=1}^{2^m} \int_{\Omega_i} (f - c_i)^2 dx + \nu \sum_{j=1}^m \int_\Omega |\nabla H(\phi_j)| dx$$

**动画建议**：展示4相分割的演化过程

---

### 6.11 Mumford-Shah的变体

| 变体 | 特点 | 应用 |
|------|------|------|
| Chan-Vese | 分段常数 | 简单目标 |
| 分段光滑 | 区域内平滑 | 灰度渐变 |
| 向量值 | 多通道 | 彩色图像 |
| 纹理 | 纹理特征 | 复杂纹理 |

---

### 6.12 与其他方法的关系

```
Mumford-Shah (一般形式)
        │
        ├── μ→∞ → 分段常数 (Chan-Vese)
        │              │
        │              └── 两相 → Active Contours
        │
        └── 水平集表示 → 变分水平集方法
```

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           Mumford-Shah模型核心                   │
├─────────────────────────────────────────────────┤
│                                                 │
│   能量泛函：                                     │
│   E(u,K) = ∫(u-f)² + μ∫|∇u|² + ν|K|            │
│            数据项   平滑项    边界项             │
│                                                 │
│   关键概念：                                     │
│   • SBV函数空间                                 │
│   • Γ收敛理论                                   │
│   • Ambrosio-Tortorelli逼近                     │
│                                                 │
│   特例：                                        │
│   • Chan-Vese（分段常数）                        │
│   • Active Contours（曲线演化）                  │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **推导题**：推导Chan-Vese模型的水平集演化方程

2. **实现题**：实现两相Chan-Vese分割算法

3. **分析题**：比较分段常数和分段光滑模型的优缺点

4. **思考题**：为什么Mumford-Shah模型的直接数值求解很困难？

---

### 📖 扩展阅读

1. **经典论文**：
   - Mumford & Shah, "Optimal approximations by piecewise smooth functions and associated variational problems", CPAM, 1989
   - Chan & Vese, "Active contours without edges", IEEE TIP, 2001

2. **理论深入**：
   - Ambrosio & Tortorelli, "Approximation of functionals depending on jumps by elliptic functionals", CPAM, 1992

3. **相关代码**：
   - Chan-Vese MATLAB实现
   - OpenCV中的GrabCut

---

### 📖 参考文献

1. Mumford, D. & Shah, J. (1989). Optimal approximations by piecewise smooth functions and associated variational problems. *CPAM*, 42(5), 577-685.

2. Chan, T.F. & Vese, L.A. (2001). Active contours without edges. *IEEE TIP*, 10(2), 266-277.

3. Ambrosio, L. & Tortorelli, V.M. (1992). On the approximation of free discontinuity problems. *Boll. Un. Mat. Ital.*

4. Vese, L.A. & Chan, T.F. (2002). A multiphase level set framework for image segmentation using the Mumford and Shah model. *IJCV*, 50(3), 271-293.
