# Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation

> **超精读笔记** | 5-Agent辩论分析系统
> 分析时间：2026-05-10
> arXiv: 1811.02514

---

## 📋 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation |
| **作者** | Xiaohao Cai (1st), Marcelo Pereyra, Jason D. McEwen |
| **第一作者核验** | 是，PDF 首页标注 1st Xiaohao Cai |
| **年份** | 2018/2019 (v1: 2018-11, v2: 2019-09) |
| **arXiv ID** | 1811.02514 |
| **期刊** | EUSIPCO 2019 (European Signal Processing Conference) |
| **机构** | UCL MSSL, Heriot-Watt University, University of Geneva |
| **领域** | 信号处理、贝叶斯推断、不确定性量化、凸优化 |

### 📝 摘要翻译

反问题在现代图像/信号处理方法中起着关键作用。然而，由于观测不足，它们通常是病态的或不适定的，其解可能具有显著的内蕴不确定性。分析和量化这种不确定性非常具有挑战性，特别是在高维问题和具有非光滑目标泛函（例如稀疏性促进先验）的问题中。

在本文中，我们提出了一系列可视化这种不确定性的策略，例如最高后验密度（HPD）可信区域，以及单个像素和超像素的局部可信区间（相当于误差条）。我们的方法支持反问题的非光滑先验，并且可以扩展到高维设置。此外，我们提出了自动设置正则化参数的策略，使得提出的不确定性量化（UQ）策略更容易使用。此外，我们还使用了不同类型的字典（完全和过完备）来表示图像/信号，并研究了它们在所提出的UQ方法中的性能。

**关键词**: 不确定性量化、反问题、凸优化、贝叶斯推断、稀疏先验、射电干涉成像、X射线成像

---

## 🔢 1. 数学家Agent：理论分析

### 1.1 核心数学框架

**问题设定：**

考虑线性反问题模型：
$$y = \Phi x + n$$

其中：
- $y \in \mathbb{C}^M$：观测数据（如射电干涉可见度、X射线投影）
- $x \in \mathbb{R}^N$：待恢复的未知信号/图像
- $\Phi \in \mathbb{C}^{M \times N}$：正向算子（测量矩阵）
- $n \in \mathbb{C}^M$：加性噪声，假设 $n \sim \mathcal{N}(0, \sigma^2 I_M)$

**字典/稀疏表示：**

信号可通过字典 $\Psi \in \mathbb{R}^{N \times L}$ 表示：
$$x = \Psi a = \sum_{i=1}^{L} \Psi_i a_i$$

- **完备字典**：$L = N$，$\Psi$ 为方阵（如正交小波基）
- **过完备字典**：$L > N$，如 SARA 字典（9个基的拼接）

**贝叶斯后验分布：**

$$p(x|y) = \frac{p(y|x)p(x)}{p(y)} \propto \exp\left(-\frac{1}{2\sigma^2}\|y - \Phi x\|_2^2\right) \cdot \exp\left(-\mu \|\Psi^\dagger x\|_1\right)$$

其中 $\Psi^\dagger$ 为分析算子，$\mu > 0$ 为正则化参数。

### 1.2 关键定理与公式

**定理1：MAP估计的凸优化形式**

MAP估计等价于求解以下凸优化问题：
$$x^*_\mu = \arg\min_{x \in \mathbb{R}^N} \left\{ \underbrace{\mu \|\Psi^\dagger x\|_1}_{f(x): \text{非光滑正则项}} + \underbrace{\frac{1}{2\sigma^2}\|y - \Phi x\|_2^2}_{g_y(x): \text{数据保真项}} \right\}$$

**定理2：HPD可信区域的高维近似**

基于概率集中理论（probability concentration），对于高维对数凹后验分布，HPD可信区域可近似为：

$$C_\alpha = \{x : \mu f(x) + g_y(x) \leq \gamma'_\alpha\}$$

其中阈值 $\gamma'_\alpha$ 的近似公式为：
$$\gamma'_\alpha = \mu f(x^*_\mu) + g_y(x^*_\mu) + \sqrt{16\log(3/\alpha)} \cdot \sqrt{N} + N$$

**近似误差界**：当 $N$ 足够大时，该近似的相对误差趋近于零。

**定理3：局部可信区间**

对于超像素区域 $\Omega_i \subseteq \{1,\ldots,N\}$，定义局部可信区间 $[\xi^{-,\Omega_i}, \xi^{+,\Omega_i}]$：

$$\xi^{-,\Omega_i} = \min_\xi \left\{ \mu f(x_{i,\xi}) + g_y(x_{i,\xi}) \leq \gamma'_\alpha \right\}$$
$$\xi^{+,\Omega_i} = \max_\xi \left\{ \mu f(x_{i,\xi}) + g_y(x_{i,\xi}) \leq \gamma'_\alpha \right\}$$

其中 $x_{i,\xi} = x^*_\mu \odot \mathbb{1}_{\Omega \setminus \Omega_i} + \xi \cdot \mathbb{1}_{\Omega_i}$，即仅在区域 $\Omega_i$ 上将值设为 $\xi$，其余保持 MAP 估计值。

**全局可信区间**：
$$\xi^- = \sum_i \xi^{-,\Omega_i} \cdot \mathbb{1}_{\Omega_i}, \quad \xi^+ = \sum_i \xi^{+,\Omega_i} \cdot \mathbb{1}_{\Omega_i}$$

### 1.3 关键证明思路

**HPD近似的核心论证**：

1. **概率集中不等式**：对于高维对数凹分布，后验概率质量集中在 MAP 估计的邻域内
2. **Moreau-Yosida包络**：非光滑函数 $f(x)$ 的 Moreau-Yosida 正则化 $f_\gamma(x)$ 提供光滑逼近
3. **维度效应**：随着 $N \to \infty$，后验分布的"有效支撑集"收缩，使得 MAP 点附近的局部近似越来越准确
4. **阈值校正项**：$\sqrt{16\log(3/\alpha)} \cdot \sqrt{N} + N$ 来自高维概率集中不等式的精确界

**自动正则化参数选择的推导**：

采用层次贝叶斯模型，将 $\mu$ 视为随机变量，其先验为：
$$p(\mu) \propto \mu^{N/k} \cdot \exp(-\gamma \mu)$$

联合 MAP 估计 $(x^*, \mu^*)$ 满足不动点方程：
$$\mu^{(i)} = \frac{N/k + \gamma^{-1}}{f(x^{(i)}) + \beta}$$

其中 $k$ 与先验 $f$ 的结构相关（$\ell_1$ 范数时 $k=1$）。

### 1.4 理论性质总结

| 性质 | 描述 |
|------|------|
| **适用范围** | 对数凹后验分布（含 $\ell_1$、TV 等非光滑先验） |
| **近似精度** | 随 $N$ 增大而提高（高维渐近精确） |
| **计算优势** | 无需采样，仅需凸优化 |
| **参数自动化** | 联合 MAP 估计避免手动调参 |

---

## 🔧 2. 工程师Agent：实现细节

### 2.1 算法架构

```
输入: 观测数据 y ∈ ℂ^M, 正向算子 Φ, 字典 Ψ, 噪声水平 σ
  ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段1: 自动正则化参数选择                                        │
├─────────────────────────────────────────────────────────────────┤
│  初始化: μ⁽⁰⁾ = 1, x⁽⁰⁾ = 0                                    │
│  迭代 i = 1, 2, ...:                                            │
│    x⁽ⁱ⁾ = argminₓ {μ⁽ⁱ⁻¹⁾‖Ψ†x‖₁ + (1/2σ²)‖y - Φx‖²₂}        │
│    μ⁽ⁱ⁾ = (N/k + γ⁻¹) / (‖Ψ†x⁽ⁱ⁾‖₁ + β)                       │
│  直到收敛                                                       │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段2: MAP估计（凸优化求解）                                     │
├─────────────────────────────────────────────────────────────────┤
│  minₓ {μ*‖Ψ†x‖₁ + (1/2σ²)‖y - Φx‖²₂}                          │
│  算法选择:                                                       │
│    • 前向-后向分裂 (Forward-Backward Splitting)                   │
│    • Douglas-Rachford 分裂                                       │
│    • ADMM (交替方向乘子法)                                        │
│    • Primal-Dual 算法                                            │
│  输出: MAP 估计 x*_μ                                             │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段3: HPD阈值计算                                              │
├─────────────────────────────────────────────────────────────────┤
│  γ'_α = μf(x*_μ) + g_y(x*_μ) + √(16log(3/α))·√N + N           │
│  其中 f(x*_μ) = ‖Ψ†x*_μ‖₁, g_y(x*_μ) = ‖y-Φx*_μ‖²₂/(2σ²)     │
└─────────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────────┐
│  阶段4: 局部可信区间计算                                          │
├─────────────────────────────────────────────────────────────────┤
│  对每个超像素 Ω_i:                                               │
│    ξ⁻ᵢ = min_ξ {μf(x_{i,ξ}) + g_y(x_{i,ξ}) ≤ γ'_α}            │
│    ξ⁺ᵢ = max_ξ {μf(x_{i,ξ}) + g_y(x_{i,ξ}) ≤ γ'_α}            │
│  拼接得到全局可信区间 [ξ⁻, ξ⁺]                                   │
└─────────────────────────────────────────────────────────────────┘
  ↓
输出: MAP估计 x*_μ, 可信区间 [ξ⁻, ξ⁺], HPD区域 C_α
```

### 2.2 计算复杂度

| 阶段 | 复杂度 | 瓶颈 |
|------|--------|------|
| 自动参数选择 | $O(n_{\text{iter}} \times C_{\text{MAP}})$ | 通常 $n_{\text{iter}} \leq 10$ |
| MAP估计（单次） | $O(N \log N)$ ~ $O(N^2)$ | 取决于算子 $\Phi$ 和字典 $\Psi$ |
| HPD阈值 | $O(1)$ | 闭式解 |
| 局部可信区间 | $O(n_{\text{grid}} \times C_{\text{MAP}})$ | 每个超像素需求解凸优化 |
| **总体** | 比 MCMC 快 $O(10^5)$ 倍 | 主要优势 |

**关键加速策略**：
- FFT 加速 $\Phi$ 和 $\Phi^\dagger$ 的计算（射电干涉/X射线场景）
- 近端算子的闭式解（软阈值）
- 超像素并行计算可信区间

### 2.3 实现要点

**前向-后向分裂算法实现**：

```python
def forward_backward_splitting(Phi, Psi, y, mu, sigma, max_iter=1000):
    """
    求解: min_x { mu * ||Psi^† x||_1 + (1/(2*sigma^2)) * ||y - Phi x||_2^2 }
    
    迭代格式:
      x^(k+1) = prox_{delta * mu * f}(x^k - delta * grad_g(x^k))
    """
    # 步长选择: delta < 2 / L，其中 L 是 g 的 Lipschitz 常数
    L = np.linalg.norm(Phi.T @ Phi, ord=2) / sigma**2
    delta = 0.9 / L
    
    x = np.zeros(Phi.shape[1])
    
    for k in range(max_iter):
        # 梯度步
        residual = y - Phi @ x
        grad_g = -Phi.T @ residual / sigma**2
        v = x - delta * grad_g
        
        # 近端步（软阈值）
        if Psi is not None:
            coeffs = Psi.T @ v
            coeffs_prox = np.sign(coeffs) * np.maximum(np.abs(coeffs) - delta * mu, 0)
            x_new = Psi @ coeffs_prox
        else:
            x_new = np.sign(v) * np.maximum(np.abs(v) - delta * mu, 0)
        
        if np.linalg.norm(x_new - x) < 1e-6:
            break
        x = x_new
    
    return x
```

**HPD阈值与可信区间计算**：

```python
def compute_hpd_threshold(x_map, y, Phi, Psi, mu, sigma, alpha=0.01):
    """
    γ'_α = μf(x*_μ) + g_y(x*_μ) + sqrt(16*log(3/α)) * sqrt(N) + N
    """
    N = len(x_map)
    
    # 正则项值
    if Psi is not None:
        f_val = np.linalg.norm(Psi.T @ x_map, ord=1)
    else:
        f_val = np.linalg.norm(x_map, ord=1)
    
    # 数据保真项值
    residual = y - Phi @ x_map
    g_val = 0.5 * np.sum(np.abs(residual)**2) / sigma**2
    
    # HPD阈值
    gamma_alpha = mu * f_val + g_val
    correction = np.sqrt(16 * np.log(3 / alpha)) * np.sqrt(N) + N
    
    return gamma_alpha + correction

def compute_credible_intervals(x_map, y, Phi, Psi, mu, sigma, gamma_alpha, grid_size=8):
    """
    对每个超像素计算局部可信区间
    """
    N = len(x_map)
    n_side = int(np.sqrt(N))
    
    xi_minus = np.zeros(N)
    xi_plus = np.zeros(N)
    
    for i in range(0, n_side, grid_size):
        for j in range(0, n_side, grid_size):
            # 定义超像素区域
            region_indices = []
            for di in range(grid_size):
                for dj in range(grid_size):
                    if i+di < n_side and j+dj < n_side:
                        region_indices.append((i+di) * n_side + (j+dj))
            
            # 二分搜索求下界
            xi_low = binary_search_bound(x_map, y, Phi, Psi, mu, sigma, 
                                         gamma_alpha, region_indices, 'lower')
            # 二分搜索求上界
            xi_high = binary_search_bound(x_map, y, Phi, Psi, mu, sigma, 
                                          gamma_alpha, region_indices, 'upper')
            
            for idx in region_indices:
                xi_minus[idx] = xi_low
                xi_plus[idx] = xi_high
    
    return xi_minus, xi_plus
```

### 2.4 实验应用：射电干涉成像与X射线成像

**射电干涉成像 (RI Imaging)**：
- 正向算子：$\Phi = P \cdot F$，其中 $F$ 为傅里叶变换，$P$ 为欠采样掩模
- 字典：SARA（Daubechies 小波 DB1-DB8 + Dirac 基的拼接）
- 应用：M31 星系观测数据

**X射线成像**：
- 正向算子：$\Phi$ 为 Radon 变换（X射线投影）
- 字典：正交小波基
- 应用：医学 CT 重建

---

## 💼 3. 应用专家Agent：应用价值

### 3.1 应用场景

| 应用领域 | 具体场景 | 正向算子 | 字典选择 |
|----------|----------|----------|----------|
| **射电干涉成像** | SKA、LOFAR、VLA | 傅里叶+欠采样 | SARA（过完备） |
| **X射线成像** | 医学CT、工业检测 | Radon变换 | 正交小波 |
| **MRI** | 快速MRI重建 | 傅里叶+欠采样 | 小波/Curvelet |
| **遥感** | 合成孔径雷达成像 | 傅里叶变换 | 稀疏字典 |

### 3.2 实验结果

**实验设置**：
- 测试图像：M31星系（射电天文）、Shepp-Logan幻影（X射线）
- 对比方法：MCMC采样（作为ground truth参考）
- 评估指标：相对误差、可信区间覆盖率、计算时间

**关键结果**：

| 指标 | 本文方法 | MCMC | 说明 |
|------|----------|------|------|
| 相对误差 | < 5% | - | 与MCMC后验均值对比 |
| 计算时间 | 秒级 | 小时级 | 加速比 $\sim 10^5$ |
| 可信区间覆盖率 | $\geq 1-\alpha$ | $1-\alpha$ | 满足名义覆盖 |
| HPD区域准确度 | 高 | 高 | 视觉一致性好 |

**字典性能对比**：

| 字典类型 | 优势 | 劣势 | 适用场景 |
|----------|------|------|----------|
| 正交小波 | 计算快、实现简单 | 表示能力有限 | 一般图像 |
| SARA（过完备） | 表示能力强、重建质量高 | 计算开销大 | 射电天文 |

### 3.3 与现有方法对比

| 方法 | UQ能力 | 计算效率 | 非光滑先验 | 高维适用 |
|------|--------|----------|------------|----------|
| **MCMC采样** | ✓ 完整后验 | ✗ 极慢 | ✓ | ✗ |
| **变分推断** | △ 近似 | ✓ 快 | △ | ✓ |
| **本文方法** | ✓ MAP+区间 | ✓ 快 | ✓ | ✓ |

**核心优势**：
1. 首个支持非光滑先验的高效 UQ 方法
2. 自动正则化参数选择，降低使用门槛
3. 提供多种不确定性可视化（HPD区域、局部误差条）

---

## 🤨 4. 质疑者Agent：批判性审查

### 4.1 假设局限性

| 假设 | 局限性 | 影响 |
|------|--------|------|
| **后验对数凹** | $\ell_1$ 先验导致后验非严格对数凹 | 理论保证可能不完全成立 |
| **高斯噪声** | 实际噪声可能非高斯（如泊松噪声） | 似然函数建模偏差 |
| **大 $N$ 渐近** | 小规模问题近似可能不准确 | 短图像/信号需谨慎 |
| **算子精确已知** | 实际存在校准误差 | 系统偏差 |

### 4.2 方法局限

1. **对数凹假设的限制**：
   - 强多峰后验分布无法处理
   - 某些非凸先验（如 $\ell_p$ 范数，$p < 1$）不适用

2. **超像素尺度选择**：
   - 需要手动选择网格大小
   - 过大：丢失局部信息；过小：计算开销增加
   - 缺乏自适应选择策略

3. **HPD近似的精度**：
   - 校正项 $\sqrt{N}$ 依赖高维渐近
   - 中等维度（$N \sim 10^2 - 10^3$）可能不够精确

4. **字典依赖性**：
   - 不同字典对 UQ 结果影响显著
   - 缺乏字典选择的理论指导

### 4.3 实验局限

1. **数据集规模有限**：
   - 仅测试了少量图像
   - 缺乏大规模统计验证

2. **对比方法单一**：
   - 仅与一种 MCMC 方法对比
   - 缺乏与其他 UQ 方法（如变分推断、深度学习 UQ）的对比

3. **评估指标不完整**：
   - 缺乏标准 UQ 评估指标（如 CRPS、校准曲线）
   - 可信区间的覆盖率验证不够充分

4. **真实数据有限**：
   - 主要基于合成数据和少量真实观测
   - 需要更多真实世界验证

### 4.4 改进建议

| 方向 | 短期改进 | 长期研究 |
|------|----------|----------|
| **理论** | 小 $N$ 误差分析 | 非对数凹后验扩展 |
| **方法** | 自适应超像素选择 | 与深度学习 UQ 结合 |
| **实验** | 更多数据集和对比方法 | 真实大规模数据验证 |
| **应用** | 更多成像模态 | 实时 UQ 系统 |

---

## 🎯 5. 综合者Agent：共识总结

### 5.1 核心贡献

| 贡献 | 描述 | 重要性 |
|------|------|--------|
| **理论** | 基于概率集中理论的 HPD 区域高维近似 | ★★★★★ |
| **方法** | 支持非光滑先验的 MAP-based UQ 框架 | ★★★★★ |
| **应用** | 自动正则化参数选择策略 | ★★★★☆ |
| **实验** | 射电干涉与X射线成像的 UQ 验证 | ★★★★☆ |

### 5.2 技术演进脉络

```
传统贝叶斯UQ (MCMC采样)
  │
  ├── 优势: 完整后验分布
  └── 劣势: 高维计算不可行
        │
        ▼
MAP估计 (凸优化)
  │
  ├── 优势: 高效、支持非光滑先验
  └── 劣势: 仅单点估计，无不确定性
        │
        ▼
MAP-based UQ (Cai et al., 2018) ← 本文
  │
  ├── 创新: 概率集中理论 → HPD近似
  ├── 创新: 局部可信区间（误差条）
  └── 创新: 自动正则化参数选择
        │
        ▼
后续工作 (2019-至今)
  ├── 近端MCMC (UQ for RI I & II)
  ├── 深度学习UQ
  └── 实时UQ系统
```

### 5.3 与作者其他工作的关系

| 论文 | 关系 | 区别 |
|------|------|------|
| UQ for RI I (MNRAS 2017) | 前序工作 | 近端MCMC，计算较慢 |
| UQ for RI II (MNRAS 2018) | 前序工作 | MAP估计，无UQ |
| **本文 (EUSIPCO 2019)** | 核心贡献 | MAP-based UQ，高效 |
| Proximal Nested Sampling | 后续发展 | 嵌套采样框架 |

### 5.4 未来方向

1. **理论扩展**：
   - 非对数凹后验的 UQ
   - 更紧的近似误差界
   - 自适应字典选择理论

2. **方法改进**：
   - 与深度学习结合（如神经网络作为先验）
   - 实时 UQ 系统
   - 自动超像素尺度选择

3. **应用拓展**：
   - 更多成像模态（MRI、超声）
   - 大规模科学数据处理
   - 在线/流式 UQ

### 5.5 综合评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 理论深度 | ★★★★★ | 概率集中理论基础扎实 |
| 方法创新 | ★★★★★ | MAP-based UQ 框架新颖 |
| 实验验证 | ★★★★☆ | 数据集可更丰富 |
| 应用价值 | ★★★★★ | 解决高维 UQ 的实际难题 |
| 论文写作 | ★★★★☆ | 清晰简洁，5页篇幅限制 |

**总分：★★★★☆ (4.6/5.0)**

---

## 📚 关键参考文献

1. **Pereyra (2016)**. Proximal Markov chain Monte Carlo algorithms. *Statistics and Computing*. — 近端MCMC的理论基础
2. **Cai, Pereyra, McEwen (2017)**. Uncertainty quantification for radio interferometric imaging: I. proximal MCMC methods. *MNRAS*. — 作者前序工作，近端MCMC方法
3. **Cai, Pereyra, McEwen (2018)**. Uncertainty quantification for radio interferometric imaging: II. MAP estimation. *MNRAS*. — 作者前序工作，MAP估计
4. **Carrillo, McEwen, Wiaux (2012)**. Sparsity averaging reweighted analysis (SARA). — SARA字典的提出
5. **Combettes, Pesquet (2011)**. Proximal splitting methods in signal processing. — 近端分裂方法综述
6. **Vershynin (2018)**. High-dimensional probability. — 概率集中理论
7. **Parikh, Boyd (2014)**. Proximal algorithms. — 凸优化基础
8. **Robert (2007)**. The Bayesian choice. — 贝叶斯推断理论

---

## 💻 复现要点

### 环境配置
```
Python >= 3.7
numpy, scipy, matplotlib
pywt (PyWavelets)  # 小波变换
```

### 关键实现步骤

1. **正向算子构建**：
   - 射电干涉：`Phi = PartialFFT(mask)`，利用欠采样傅里叶变换
   - X射线：`Phi = RadonTransform(angles)`

2. **字典构建**：
   - 正交小波：`Psi = WaveletMatrix(wavelet='db4', level=3)`
   - SARA：拼接 DB1-DB8 + Dirac 基

3. **自动参数选择**：
   - 迭代 5-10 次即可收敛
   - 初始 $\mu$ 可设为 1 或根据数据尺度调整

4. **可信区间计算**：
   - 超像素大小建议：$8 \times 8$ 或 $16 \times 16$
   - 可用二分搜索加速边界查找

### 常见陷阱

- **步长选择**：前向-后向分裂的步长需满足 $\delta < 2/L$，其中 $L$ 为 Lipschitz 常数
- **收敛判断**：建议同时监测目标函数值和参数变化
- **字典内存**：过完备字典（如 SARA）内存开销大，需预分配
- **噪声估计**：$\sigma$ 的准确估计对 UQ 结果影响显著

---

## 📝 分析笔记

**核心洞察**：

1. **理论桥梁**：本文的核心贡献是建立了 MAP 估计与完整贝叶斯 UQ 之间的桥梁，利用概率集中理论证明了高维情况下 MAP 点附近的局部近似是准确的

2. **实用性突破**：相比 MCMC 方法快 $10^5$ 倍，使得 UQ 在实际大规模问题中变得可行

3. **参数自动化**：自动正则化参数选择是该方法能被广泛使用的关键，避免了繁琐的手动调参

4. **字典影响**：过完备字典（SARA）在表示能力和计算开销之间存在权衡，需要根据具体应用选择

**开放问题**：

- 如何扩展到非对数凹后验（如 $\ell_p$ 范数，$p < 1$）？
- 如何自适应选择超像素尺度？
- 与深度学习先验（如 DnCNN、深度展开）如何结合？
- 在线/流式场景下的实时 UQ 如何实现？
