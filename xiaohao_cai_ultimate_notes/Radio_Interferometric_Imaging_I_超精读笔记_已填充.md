# Radio Interferometric Imaging: Uncertainty Quantification I

> **超精读笔记** | 5-Agent辩论分析系统
> 分析时间：2026-02-16
> MNRAS 2018 (Vol.480; arXiv 预印本 2017), arXiv: 1711.04818v2

---

## 📋 论文元数据

| 属性 | 信息 |
|------|------|
| **标题** | Uncertainty quantification for radio interferometric imaging: I. proximal MCMC methods |
| **作者** | Xiaohao Cai, Marcelo Pereyra, Jason D. McEwen |
| **第一作者核验** | 是，PDF 首页作者列表以 Xiaohao Cai 开头 |
| **年份** | 2018（MNRAS Vol.480, Issue 3 正式发表；arXiv 预印本 2017） |
| **期刊** | Monthly Notices of the Royal Astronomical Society (MNRAS) |
| **arXiv ID** | 1711.04818v2 |
| **机构** | UCL MSSL, Heriot-Watt University |

### 📝 摘要翻译

不确定量化是射电干涉成像中一个关键缺失的组成部分，随着射电干涉大数据时代的到来，这将变得越来越重要。由于射电干涉成像需要求解一个高维病态逆问题，不确定量化虽然困难，但对于准确科学解释射电观测结果至关重要。统计采样方法如马尔可夫链蒙特卡罗(MCMC)采样执行贝叶斯推断，原则上可以恢复图像的完整后验分布，从而量化不确定性。然而，传统的高维采样方法通常限于光滑(如高斯)先验，不能用于稀疏促进先验。受压缩感知理论激励的稀疏先验已被证明对射电干涉成像非常有效。在本文中，我们开发了用于射电干涉成像的近端MCMC方法，利用近端演算在贝叶斯框架中支持非微分先验(如稀疏先验)。此外，开发了三种使用恢复的后验分布量化不确定性的策略：(i)局部(逐像素)可信区间为每个像素提供误差条；(ii)最高后验密度可信区域；(iii)图像结构的假设检验。这些形式的不确定量化为以统计稳健的方式分析射电干涉观测提供了丰富信息。

**关键词**: 射电干涉成像、不确定量化、近端MCMC、稀疏先验、贝叶斯推断

> PDF 关键词原文（PDF §1 Key words）：techniques: image processing / interferometric；methods: data analysis / numerical / statistical。

### 🧭 一句话定位与阅读地图

- **一句话**：把 proximal calculus（Moreau-Yosida envelope + proximity operator）接进高维 Langevin MCMC，使 MCMC 第一次能在**非光滑 $\ell_1$ 稀疏后验**上采样，并由后验样本产出三类可解释的 UQ 产品（pixel-wise credible interval / HPD region / hypothesis test）。
- **阅读路线（建议）**：式 2 的 $y=\Phi x+n$ → §2.3 似然（式 4）与 analysis/synthesis 先验（式 5、6）→ §2.4 MAP（式 11、12）与 CS 联系 → §3.1 Moreau-Yosida envelope（式 17–19）→ §3.2–§3.4 ULA→MYULA→Px-MALA（式 22、24、25）→ §4 RI 专用 prox/grad（式 29、30）与 Algorithm 1/2 → §5 三类 UQ（式 46–53）→ §6 四张真值图实验（Table 1、Fig. 5/6、Table 2/3）。
- **核心张力**：speed vs exactness。MYULA 快（无 MH）但有可控 bias；Px-MALA 渐近无偏（有 MH 校正）但更贵、链相关性更高、估计方差更大。论文反复用"二者一致"佐证可信度。

---

## 🔢 1. 数学家Agent：理论分析

### 1.1 核心数学框架

**射电干涉成像 (RI Imaging) 数学模型**

观测方程：
```
y(u) = ∫ A(l) x(l) e^{-2πiu·l} d²l
```

离散形式：
```
y = Φx + n
```

其中：
- x ∈ ℝ^N 是天空亮度分布(图像)
- y ∈ ℂ^M 是测量到的可见度
- Φ ∈ ℂ^{M×N} 是线性测量算子
- n ∈ ℂ^M 是仪器噪声

**问题特点**：
- 病态问题
- 高维(N很大)
- 不适定(欠定，M < N)

**稀疏表示**：
```
x = Ψa = Σ Ψ_i a_i
```

如果a只有K个非零系数，则称x在Ψ下是K-稀疏的。

### 1.2 关键公式推导

**核心公式1：贝叶斯推断框架**

**分析模型后验分布**：
```
p(x|y) ∝ exp{-μ||Ψ†x||₁ - ||y - Φx||²₂/(2σ²)}
```

**综合模型后验分布**：
```
p(a|y) ∝ exp{-μ||a||₁ - ||y - ΦΨa||²₂/(2σ²)}
```

其中：
- Ψ† 是分析算子(如小波变换)
- Ψ 是综合算子
- ||·||₁ 是ℓ₁范数(促进稀疏性)
- ||·||₂ 是ℓ₂范数(数据保真度)

**核心公式2：MAP估计**

分析模型MAP：
```
x̂_MAP = argmin_x μ||Ψ†x||₁ + ||y - Φx||²₂/(2σ²)
```

综合模型MAP：
```
â_MAP = argmin_a μ||a||₁ + ||y - ΦΨa||²₂/(2σ²)
```

**核心公式3：近端算子**

对于非光滑函数φ，近端算子prox_φ定义为：
```
prox_φ(v) = argmin_x φ(x) + (1/2)||x - v||²₂
```

**关键近端算子：**

**软阈值(ℓ₁范数)**：
```
prox_{μ||·||₁}(v) = sign(v) ⊙ max(|v| - μ, 0)
```

**投影函数(指示函数)**：
```
prox_{ι_C}(v) = Π_C(v)
```
其中Π_C是到集合C的投影。

### 1.3 理论性质分析

**从 Langevin diffusion 到 MYULA / Px-MALA（PDF §3.2–§3.4，逐式直觉）：**

- **Langevin diffusion（式 21）**：$d\mathcal{L}(t)=\tfrac12\nabla\log\pi(\mathcal{L}(t))\,dt+d\mathcal{W}(t)$，以 $\pi$ 为不变分布；Euler-Maruyama 离散得 **ULA（式 22）** $l^{(m+1)}=l^{(m)}+\tfrac{\delta}{2}\nabla\log\pi(l^{(m)})+\sqrt{\delta}\,w^{(m+1)}$。**关键限制**：要求 $\log\pi$ 处处可微且梯度 Lipschitz——而 $\ell_1$ 先验 $\|\cdot\|_1$ 非光滑，ULA/MALA/HMC **直接用不了**。这正是论文的切入点。
- **MYULA（式 24）**：把非光滑 $f$ 换成 Moreau-Yosida envelope $f^\lambda$（$C^1$，梯度 $\nabla f^\lambda(z)=(z-\mathrm{prox}_f^\lambda(z))/\lambda$，式 18），得
  $$l^{(m+1)}=\Big(1-\tfrac{\delta}{\lambda}\Big)l^{(m)}+\tfrac{\delta}{\lambda}\,\mathrm{prox}_f^\lambda(l^{(m)})-\delta\,\nabla g(l^{(m)})+\sqrt{2\delta}\,w^{(m)}.$$
  直觉：第一项+第二项是"向 prox 不动点收缩"，对应平滑后的先验梯度；第三项是数据项梯度下降；最后是 Langevin 噪声。参数取 $\lambda=2/\beta_{\text{Lip}}$、$\delta\in[1/5\beta_{\text{Lip}},\,1/2\beta_{\text{Lip}}]$（Durmus et al. 2016）。**无 MH 步**，bias 随 $\lambda\to0$ 任意小，scale 到高维。
- **Px-MALA（式 25、43）**：用一次 MYULA 迭代当 proposal（式 43 $x^{(m+1)}=\mathrm{prox}_f^{\delta/2}(x^{(m)})+\sqrt\delta\,w^{(m)}$），再加 **Metropolis-Hastings** 接受概率 $\rho=\min\{1,\,q(l^{(m)}|l^*)\pi(l^*)/q(l^*|l^{(m)})\pi(l^{(m)})\}$（$q$ 为 MYULA transition kernel，式 26）。**有 MH 校正**，渐近以 $\pi$ 为**精确**不变分布；调 $\delta$ 使接受率约 0.5。

**速度 vs 精确性（论文核心权衡）：**

| | MYULA | Px-MALA |
|---|---|---|
| MH 校正 | 无 | 有（accept-reject） |
| 不变分布 | 近似 $\pi$（bias 可控、随 $\lambda$ 任意小） | 精确 $\pi$（渐近无偏） |
| 单步成本 | 低 | 高（多一个 MH） |
| 链相关性 / 估计方差 | 低（更平滑） | 高（MH 去 bias 的代价） |
| 实测 CPU 时间 | 约为 Px-MALA 一半（Table 1） | 约 MYULA 两倍 |
| credible interval | 更宽、更平滑、略 overestimate | 更小、更 noisy |

**Px-MALA**：理论保证收敛到目标分布、具有正确平稳分布，但 MH 步开销高、链相关性高。**MYULA**：通过引入可控偏差消除 MH 步、偏差可任意小、计算开销更低——但代价是过平滑、轻微高估不确定度。

### 1.4 数学创新点

**新的数学工具：**
1. **近端MCMC**：将近端演算与MCMC结合
2. **Moreau-Yosida正则化**：处理非光滑先验
3. **三种不确定量化方法**：可信区间、HPD区域、假设检验

---

## 🔧 2. 工程师Agent：实现分析

### 2.1 算法架构

```
┌─────────────────────────────────────────────────────────────────┐
│              Radio Interferometric Imaging Pipeline              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  输入: 可见度数据 y ∈ ℂ^M, 测量算子 Φ                          │
│                         ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  贝叶斯推断 + 近端MCMC采样                                │   │
│  │  ┌───────────────────────────────────────────────────┐  │   │
│  │  │ 两种模型选择:                                    │  │   │
│  │  │  • 分析模型: p(x|y) ∝ exp{-μ||Ψ†x||₁ + ...}    │  │   │
│  │  │  • 综合模型: p(a|y) ∝ exp{-μ||a||₁ + ...}      │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  │  ┌───────────────────────────────────────────────────┐  │   │
│  │  │ 两种采样算法:                                    │  │   │
│  │  │  • Px-MALA: 高精度，有MH步骤                     │  │   │
│  │  │  • MYULA: 低开销，可控偏差                       │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  │                                                         │   │
│  │  采样: 生成 {x^(i)} ~ p(x|y) 或 {a^(i)} ~ p(a|y)      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  不确定量化 (三种方法)                                   │   │
│  │  ┌───────────────────────────────────────────────────┐  │   │
│  │  │ 方法1: 局部可信区间                                │  │   │
│  │  │   对每个像素: [q_α/2(x_j), q_{1-α/2}(x_j)]       │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  │  ┌───────────────────────────────────────────────────┐  │   │
│  │  │ 方法2: HPD可信区域                                │  │   │
│  │  │   找最小区域C使P(x∈C|y) ≥ 1-α                    │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  │  ┌───────────────────────────────────────────────────┐  │   │
│  │  │ 方法3: 假设检验                                  │  │   │
│  │  │   H0: x在某个区域内无结构                         │  │   │
│  │  └───────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         ↓                                        │
│  输出: 重构图像 + 不确定度量                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键实现要点

**Px-MALA算法：**

```python
import numpy as np

class PxMALA:
    """近端MALA算法"""
    def __init__(self, Phi, Psi, mu, sigma, step_size):
        self.Phi = Phi          # 测量算子
        self.Psi = Psi          # 稀疏基
        self.mu = mu            # 正则化参数
        self.sigma = sigma      # 噪声标准差
        self.step_size = step_size

    def grad_log_posterior(self, x, y):
        """计算后验对数梯度"""
        # 数据项梯度
        residual = y - self.Phi.dot(x)
        grad_data = self.Phi.conj().T.dot(residual) / (self.sigma ** 2)

        # 先验项梯度 (对于分析模型)
        # ℓ₁范数是次微分的，在近端框架中处理

        return grad_data

    def proximal_prior(self, v):
        """先验的近端算子 (软阈值)"""
        # 分析模型: prox for μ||Ψ†x||₁
        coeffs = self.Psi.conj().T.dot(v)
        coeffs_soft = np.sign(coeffs) * np.maximum(np.abs(coeffs) - self.mu, 0)
        return self.Psi.dot(coeffs_soft)

    def step(self, x, y):
        """单次迭代"""
        # 1. 提议步 (使用梯度)
        grad = self.grad_log_posterior(x, y)
        x_proposed = x + self.step_size * grad

        # 2. 近端投影
        x_proposed = self.proximal_prior(x_proposed)

        # 3. MH接受-拒绝步骤
        log_alpha = (self.log_posterior(x_proposed, y) -
                     self.log_posterior(x, y))

        if np.log(np.random.rand()) < log_alpha:
            return x_proposed  # 接受
        else:
            return x           # 拒绝

    def log_posterior(self, x, y):
        """计算对数后验"""
        # 数据项
        residual = y - self.Phi.dot(x)
        log_likelihood = -0.5 * np.sum(np.abs(residual)**2) / (self.sigma**2)

        # 先验项
        coeffs = self.Psi.conj().T.dot(x)
        log_prior = -self.mu * np.sum(np.abs(coeffs))

        return log_likelihood + log_prior

    def sample(self, y, n_samples, burn_in):
        """采样后验分布"""
        samples = []
        x = np.zeros(self.Phi.shape[1])  # 初始化

        for i in range(burn_in + n_samples):
            x = self.step(x, y)
            if i >= burn_in:
                samples.append(x.copy())

        return np.array(samples)
```

**MYULA算法：**

```python
class MYULA:
    """Moreau-Yosida ULA算法"""
    def __init__(self, Phi, Psi, mu, sigma, step_size, gamma):
        self.Phi = Phi
        self.Psi = Psi
        self.mu = mu
        self.sigma = sigma
        self.step_size = step_size
        self.gamma = gamma  # Moreau-Yosida参数

    def moreau_envelope_gradient(self, v):
        """Moreau-Yosida包络的梯度"""
        # 近端算子
        coeffs = self.Psi.conj().T.dot(v)
        coeffs_prox = np.sign(coeffs) * np.maximum(np.abs(coeffs) - self.gamma, 0)

        # 梯度近似
        return (v - self.Psi.dot(coeffs_prox)) / self.gamma

    def step(self, x, y):
        """MYULA单步 (无MH步骤)"""
        # 数据梯度
        residual = y - self.Phi.dot(x)
        grad_data = self.Phi.conj().T.dot(residual) / (self.sigma**2)

        # 近端先验梯度
        grad_prior_prox = self.moreau_envelope_gradient(x)

        # Langevin更新 + 噪声
        noise = np.sqrt(2 * self.step_size) * np.random.randn(len(x))
        x_new = x + self.step_size * (grad_data - grad_prior_prox) + noise

        return x_new

    def sample(self, y, n_samples, burn_in):
        """采样"""
        samples = []
        x = np.zeros(self.Phi.shape[1])

        for i in range(burn_in + n_samples):
            x = self.step(x, y)
            if i >= burn_in:
                samples.append(x.copy())

        return np.array(samples)
```

### 2.3 不确定量化实现

```python
class UncertaintyQuantification:
    """不确定量化"""

    @staticmethod
    def credible_intervals(samples, alpha=0.05):
        """
        方法1: 局部可信区间
        对每个像素计算α/2和1-α/2分位数
        """
        lower = np.percentile(samples, 100 * alpha / 2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha / 2), axis=0)
        median = np.median(samples, axis=0)

        return {
            'lower': lower,
            'upper': upper,
            'median': median,
            'width': upper - lower
        }

    @staticmethod
    def hpd_region(samples, alpha=0.05):
        """
        方法2: 最高后验密度(HPD)可信区域
        找最小区域使后验概率≥1-α
        """
        # 对每个像素找到最短的可信区间
        sorted_samples = np.sort(samples, axis=0)
        n_samples = samples.shape[0]

        # 计算需要包含的样本数
        n_included = int(n_samples * (1 - alpha))

        hpd_lower = np.zeros(samples.shape[1])
        hpd_upper = np.zeros(samples.shape[1])

        for j in range(samples.shape[1]):
            # 滑动窗口找最短区间
            min_width = np.inf
            for i in range(n_samples - n_included):
                width = sorted_samples[i + n_included, j] - sorted_samples[i, j]
                if width < min_width:
                    min_width = width
                    hpd_lower[j] = sorted_samples[i, j]
                    hpd_upper[j] = sorted_samples[i + n_included, j]

        return {'lower': hpd_lower, 'upper': hpd_upper}

    @staticmethod
    def hypothesis_testing(samples, region, threshold=0.5):
        """
        方法3: 假设检验
        H0: x在某个区域内无结构
        """
        # 计算区域内像素超过阈值的概率
        region_samples = samples[:, region]

        # 对每个样本，检查区域内的结构
        has_structure = np.any(region_samples > threshold, axis=1)

        # 后验概率
        prob_structure = np.mean(has_structure)

        return {
            'prob_structure': prob_structure,
            'prob_no_structure': 1 - prob_structure
        }
```

### 2.3 计算复杂度

| 项目 | 复杂度 | 说明 |
|------|--------|------|
| 单次Px-MALA迭代 | O(N log N) | Φ计算(FFT) + Ψ计算 |
| 单次MYULA迭代 | O(N log N) | 无MH开销 |
| 采样总数 | 1000-10000 | 收敛后开始记录 |
| **总复杂度** | O(iter·N log N) | 可扩展到大数据 |

### 2.4 实现建议

**推荐策略：**
1. 小规模：用Px-MALA获得精确结果
2. 大规模：用MYULA降低计算成本
3. 并行化：多链并行采样

---

## 💼 3. 应用专家Agent：价值分析

### 3.1 应用场景

**核心领域：**
- [✓] 射电天文学
- [✓] SKA (平方公里阵列)
- [✓] 逆问题求解
- [✓] 不确定量化

**具体应用：**

1. **射电望远镜成像**
   - SKA (平方公里阵列)
   - LOFAR
   - VLA

2. **科学问题**
   - 星系形成
   - 黑洞观测
   - 宇宙学参数估计

### 3.2 技术价值

**解决的问题：**

| 问题 | 现有方法 | 本文解决方案 |
|------|----------|-------------|
| 无不确定度信息 | CLEAN/MEM | 近端MCMC |
| 稀疏先验不可用 | Gibbs/HMC | 近端演算 |
| 大数据扩展 | MCMC慢 | MYULA可扩展 |

**核心贡献：**
1. 首次为RI成像提供稀疏先验的不确定量化
2. 两种近端MCMC算法(Px-MALA, MYULA)
3. 三种不确定量化方法

### 3.3 落地可行性

| 因素 | 评估 | 说明 |
|------|------|------|
| 数据需求 | 中 | 需要可见度数据 |
| 计算资源 | 高 | MCMC需要大量计算 |
| 部署难度 | 高 | 需要专业天文学知识 |

### 3.4 商业潜力

**目标市场：**
- 天文研究机构
- 射电望远镜项目
- 科学计算软件

---

## 🤨 4. 质疑者Agent：批判分析

### 4.1 方法论质疑

**理论假设：**
1. 假设噪声i.i.d高斯 → 实际可能更复杂
2. 假设Φ精确已知 → 校准误差

**数学严谨性：**
- MYULA引入偏差，虽可控但存在
- 收敛速度分析不足

### 4.2 实验评估批判

**数据集（PDF §6.1 实测设置，已核实）：**
- 四张真值天图：**M31 galaxy (256×256)**、**Cygnus A (256×512)**、**W28 supernova remnant (256×256)**、**3C288 (256×256)**。
- visibility 全部由 **simulated** 生成：variable-density sampling profile（Puy et al. 2011）在半 Fourier 平面随机取 **10% Fourier 系数**（Fig. 2 给覆盖示例）；加零均值复 Gaussian 噪声，$\sigma=\|f\|_\infty 10^{-\mathrm{SNR}/20}$，**SNR 固定 30 dB**。
- 字典 Ψ = **Daubechies 8 小波**（MATLAB `wavedec2`），故 analysis 与 synthesis 结果差异不大。
- 关键超参：$\ell_1$ 正则 $\mu=10^4$（visual cross-validation）；采样 $10^3$ samples、$10^5$ burn-in、thinning $10^3$（每链 $1.1\times10^6$ 次迭代）；credible interval $\alpha=0.05$、hypothesis test $\alpha=0.01$。硬件：24-core x86_64 + 256 GB，MATLAB R2015b。
- **批判仍成立**：全部 visibility 为合成、未用真实观测数据；这是 proof-of-concept 性质。

**评估指标（论文实际报告）：**
- **CPU 时间（Table 1，分钟）**：M31 MYULA 618(analysis)/581(synthesis)、Px-MALA 1307/944；Cygnus A 1056/942、2274/1762；W28 646/598、1122/879；3C288 607/538、1144/881。结论：**MYULA 约需 Px-MALA 一半时间**。
- **重建**：定性视觉对照（posterior mean 明显优于 dirty image，MYULA/Px-MALA 与 analysis/synthesis 一致）；**论文未给 PSNR/SNR 数值表**。
- **interval-length map（Fig. 5）**、**HPD isocontour $\gamma_\alpha$ 曲线（Fig. 6，量级 $\sim10^6$）**、**hypothesis test（Table 2/3，×10^6）**。
- **不确定性度量难以验证、缺 ground-truth UQ**：论文用 MYULA 与 Px-MALA、analysis 与 synthesis 之间的一致性来间接佐证可信度，而非与外部 UQ 真值对照。

### 4.3 局限性分析

**方法限制：**
1. 计算成本高
2. 参数选择敏感
3. 大规模应用困难

### 4.4 改进建议

1. 短期：GPU加速
2. 长期：变分推断近似
3. 补充：更多真实数据验证

---

## 🎯 5. 综合理解：核心创新与意义

### 5.1 核心创新点

| 维度 | 创新内容 | 评分 |
|------|----------|------|
| 理论 | 近端MCMC for稀疏先验 | ★★★★★ |
| 方法 | 三种不确定量化 | ★★★★★ |
| 应用 | RI成像不确定量化 | ★★★★★ |

### 5.2 研究意义

**学术贡献：**
- 首次将稀疏先验与MCMC结合用于RI成像
- 提供完整的不确定量化框架
- 连接压缩感知与贝叶斯推断

**实际价值：**
- SKA等大型项目的重要工具
- 科学解释的可信度

### 5.3 技术演进位置

```
CLEAN/MEM (1970s-2000s)
  ↓
压缩感知 (2006)
  ↓
稀疏正则化RI成像 (2009-2014)
  ↓
近端MCMC (2017) ← 本文
  ↓
大数据扩展 (论文II)
```

### 5.4 综合评分

| 维度 | 评分 |
|------|------|
| 理论深度 | ★★★★★ |
| 方法创新 | ★★★★★ |
| 实现难度 | ★★★★☆ |
| 应用价值 | ★★★★★ |
| 论文质量 | ★★★★★ |

**总分：★★★★★ (4.8/5.0)**

---

## 🎯 6. 三类 UQ 产品与假设检验（PDF §5，深化）

论文的"产出"不止重建，而是从后验样本 $\{x^{(j)}\}_{j=1}^K$ 派生三类**可解释的不确定性产品**（Figure 1 的流程图：observed visibilities → sample posterior → point estimator / pixel-wise interval / HPD region / hypothesis testing）。

### 6.1 Pixel-wise credible intervals（式 46–49）

对每个像素 $x_i$ 求边缘分位数：
$$(\hat\xi_{i-},\hat\xi_{i+})=\mathrm{quantile}\Big(\{x_i^{(j)}\}_{j=1}^K,\ \big\{\tfrac{\alpha}{2},1-\tfrac{\alpha}{2}\big\}\Big),$$
区间长度 $\xi_{i+}-\xi_{i-}$ 即每像素 error bar。synthesis 模型用 $(\Psi a)_i$ 投影得到（式 48）。$\alpha=0.05$ 对应 95% 区间。**直觉**：长 interval = 高不确定（论文观察物体边界与高频处 interval 更宽，因采样 profile 对高频覆盖不足）。

### 6.2 HPD credible region（式 50–52）

$C_\alpha=\{x:f(x)+g(x)\le\gamma_\alpha\}$，即 log-posterior 的一个 level-set（isocontour）。$\gamma_\alpha$ 由样本估计 $\hat\gamma_\alpha=\mathrm{quantile}(\{(\hat f+\hat g)(x^{(j)})\},1-\alpha)$。它是 decision-theoretically **minimum-volume** 的 $100(1-\alpha)\%$ 可信区域，作用在 image level（整图），用于分析较大结构与做后验检查。

### 6.3 Hypothesis testing of image structure（式 53，knock-out test）

这是论文最巧妙的一步，**把"某结构是真还是 artefact"变成一个可计算的判定**：
1. **构造 surrogate**：从点估计 $x^*$（posterior mean 或 median）出发，用 **segmentation-inpainting**（式 53，基于 Cai et al. 2008 的 recursive wavelet filter，迭代 $\mathbf{x}^{(m+1),\text{sgt}}=\mathbf{x}^*\mathbb{1}_{\Omega-\Omega_D}+\Lambda^\dagger\mathrm{soft}_{\lambda_{\text{th}}}(\Lambda\mathbf{x}^{(m),\text{sgt}})\mathbb{1}_{\Omega_D}$）把感兴趣结构"抠掉"并用背景填充，得 $x^{*,\text{sgt}}$。
2. **判定**：检查 $x^{*,\text{sgt}}\notin C_\alpha$，即 $(\bar f+\bar g)(x^{*,\text{sgt}})>\hat\gamma_\alpha$？
   - 若**是**（surrogate 落在 HPD 之外）→ likelihood 强烈反对"抠掉" → 数据强支持该结构 → **physical（✓）**。
   - 若**否**（surrogate 仍在 HPD 内）→ 数据对该修改不敏感 → 证据不足 → 可能 **artefact（✗）**。
3. **实测（Table 2/3，$\alpha=0.01$，×10^6）**：M31 / W28 / 3C288(test 1) 三个大物理结构正确判 physical；3C288 的人造 test area 2 正确判 artefact（MYULA 1.752 < $\gamma$ 2.032）；**Cygnus A 的极小亮结构**用 posterior mean 判 ✗、用 **posterior median** 判 ✓（1.597 vs 1.586）——论文据此推荐**小结构用 median**（median 更靠近 $C_\alpha$ 边界，对小结构更敏感）。

---

## 🔗 7. 与其它 14 篇的关系（深化）

- **RI UQ II（priority 13，companion article，Cai et al. 2017b）**：本篇是"完整后验采样"的基准（slow but exact-ish）；II 篇把 UQ 用 **MAP 近似**替代采样，宣称大规模加速，使其能 scale 到 big-data。读 II 时应把"II 的近似 UQ 与本篇的采样 UQ 一致到什么程度"作为验证锚点。本篇正是 II 的 ground-truth-by-sampling。
- **High-dimensional UQ（priority 11）**：把"用凸优化 MAP + Moreau-Yosida 给高维 credible region"的思想一般化，本篇的 HPD region（式 50–52）是其 RI 实例。
- **Proximal Nested Sampling（priority 14）**：proximal MCMC 背景的来源之一，把 proximal 思想接到 nested sampling 以算 marginal likelihood / model selection。
- **方法工具链**：Moreau-Yosida envelope、proximity operator、软阈值与本项目 framelet/tight-frame、SaT/SLaT 系列共享同一套凸优化/proximal 词汇；本篇是把它们用到**Bayesian 采样**而非点估计。

---

## ⚠️ 8. 阅读陷阱（careful reading）

1. **analysis vs synthesis 的变量不同**：analysis 后验在**图像域** $x$（先验 $\mu\|\Psi^\dagger x\|_1$），synthesis 后验在**系数域** $a$（先验 $\mu\|a\|_1$，且 $x=\Psi a$）。两者仅在 $\Psi$ 正交时等价；本文用 Daubechies 8（正交）故差异小，但**勿默认对 overcomplete frame 也等价**。
2. **MYULA 的 bias 是 feature 不是 bug**：它换来高维可采样与低方差；不要把"有 bias"误读为"错"。Px-MALA 用 MH 去 bias，代价是方差与成本上升。
3. **`prox` 的闭式只在 $\Psi^\dagger\Psi=I$ 成立**（式 29）；overcomplete 时 prox 需迭代近似（式 31、32、36），论文用一次 forward-backward 近似并用 MH 校正（Remark 4.1–4.3）。
4. **CPU 时间是"分钟级×百千"**：Table 1 单位是分钟，最贵的 Cygnus A Px-MALA 达 2274 分钟（≈38 小时）。这是本篇被 II 篇 MAP-UQ 取代的直接动因；**勿与本仓库 toy 的秒级 runtime 混为一谈**。
5. **hypothesis test 的判定方向易记反**：surrogate 跑出 HPD 之外（值 > $\gamma_\alpha$）才说明结构 physical；落在 HPD 之内反而是证据不足。
6. **论文不报告 PSNR/Dice 等单一数值指标**：重建是定性视觉对照，UQ 是 map/曲线/真伪判定。任何"该论文 PSNR=xx"的说法都需警惕（本仓库 toy 的 `map_psnr` 是玩具问题的内部量，与论文无关）。

---

## 复现判断

| 维度 | 判断 |
|------|------|
| 本仓库复现等级 | **toy**（`reproductionTruthLevel = toy-completed`） |
| runner | `reproduce/experiments/map_uq_toy.py`（priority 11/12/13 共享） |
| 实际做了什么 | 32×32 Fourier 欠采样玩具问题：梯度下降 + Gaussian smoothing 当 MAP/proxy；随机游走 + 平滑当"采样"，取经验分位差当 interval map |
| 当前 runMetrics | `map_psnr=18.7123`、`map_snr=9.6004`、`map_runtime_seconds=0.0017`、`mcmc_runtime_seconds=0.0041`、`gamma_alpha_toy=939.9229`、`mean_interval_length=0.1739` |
| 产物图 | `assets/repro/map_uq_reconstruction_uncertainty.png`（truth / MAP toy / HPD approx / MCMC interval 四联图） |
| 与论文的可比性 | **不可比**。无真正 RI 算子 Φ、无 Daubechies 8 字典、无 $\ell_1$ 后验、无 Moreau-Yosida/MYULA/Px-MALA、无 HPD isocontour、无 hypothesis test、无 MCMC 诊断 |
| 距 paper-like 的关键缺口 | (1) prox-Langevin 采样器（MYULA+软阈值 prox+$\nabla g$）；(2) 真实天图（M31 等 256²）+ Daubechies 8 + 10% Fourier 覆盖 + 30 dB；(3) UQ 产品（interval/HPD/hypothesis test）与 MCMC 诊断 |
| paper-level | **0/15**，本篇亦为 0；纪律：禁止把 toy 的秒级 runtime 或 `map_psnr` 当作论文级证据 |

> 诚实声明：runner 的 `notes` 已标注 "no RI operator or MCMC diagnostics" 且 "Toy runtime comparison is not comparable to the paper's large-scale 10^5 speedup claim"。本笔记中第 6 节引用的论文数值（Table 1 分钟数、Table 2/3 判定值）仅供对照，**当前实现不复现这些数值**。

---

## 完整复现流程

本篇的"完整复现流程 (Complete Reproduction Workflow)"规范文档详见
[`../reproduce/paper_like/workflows/ri-uq-i_reproduction_workflow.md`](../reproduce/paper_like/workflows/ri-uq-i_reproduction_workflow.md)。
其中含：第一作者核验、toy→paper-level 诚实分级、RI 模型与 MYULA/Px-MALA 的 step-by-step pipeline、四张真值图与 visibility 合成约定、Table 1/2/3 等论文报告结果、本仓库 toy 实现与差距清单、运行步骤与 proxy 风险说明。当前等级仍为 **toy**，paper-level 仍为 **0/15**。

---

*本笔记由5-Agent辩论分析系统生成*
