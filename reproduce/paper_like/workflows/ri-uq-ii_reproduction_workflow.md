# 无线电干涉 UQ II：MAP 快速版 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 13 篇 *Uncertainty Quantification for Radio Interferometric Imaging II: MAP Estimation* 的完整复现流程规范。

---

## 1. 论文身份与第一作者核验

| 项 | 内容 |
|----|------|
| **标题 (EN)** | Uncertainty quantification for radio interferometric imaging: II. MAP estimation |
| **标题 (CN)** | 无线电干涉成像的不确定性量化 II：MAP 估计 |
| **作者顺序** | **Xiaohao Cai**, Marcelo Pereyra, Jason D. McEwen |
| **第一作者核验** | 是。PDF 首页作者列表以 **Xiaohao Cai**$^{1\star}$ 开头，单位为 Mullard Space Science Laboratory, University College London (UCL)，邮箱 `x.cai@ucl.ac.uk`（标 XC）。确认 Xiaohao Cai 为第一作者。 |
| **年份 / 出处** | MNRAS（Monthly Notices of the Royal Astronomical Society），arXiv:1711.04819v2（11 Sep 2018，Preprint 12 September 2018，© 2017 The Authors） |
| **类型** | 方法论文（companion series 第二篇，与论文 I 的 MCMC 路线配套，给出 MAP 快速近似路线） |
| **PDF 路径** | `docs/00_papers_first_author_xiaohao_cai_deduped/无线电干涉不确定性II Radio Interferometric II.pdf` |
| **主题 (theme)** | `ri-uq`（radio interferometric imaging 的不确定性量化主线） |

本篇是 **companion series 的第二篇**。第一篇 (Cai, Pereyra, McEwen 2017a/2018, arXiv:1711.04818) 用 proximal MCMC（Px-MALA, MYULA）采样完整后验做 UQ；本篇换用 **MAP 点估计 + probability concentration 后处理**，把同样的三类 UQ 输出做成可扩展到 big-data（SKA 级）的快速版本。两篇共享同一套 UQ 输出定义，但计算路线相反：采样 vs 优化。

---

## 2. 复现目标与诚实分级

本项目对"复现"采用四级诚实分级（由弱到强）：

| 级别 | 含义 |
|------|------|
| **toy** | 合成小图 + 代理算子，演示直觉，不对齐论文任何具体数值 |
| **partial** | 实现论文核心步骤的一部分（如真实 forward-backward MAP 求解器 + 真正的 HPD 近似公式），在合成数据上验证趋势，但未对齐论文数据集与报告数值 |
| **paper-like** | 用论文同款或公开等价数据集，跑论文同款 pipeline，复现论文表格量级（不要求逐位一致） |
| **paper-level** | 严格复现论文报告数值（同数据、同基线、同指标、同表号） |

**本仓库当前等级（reproductionLevel）= `toy`；真实性（reproductionTruthLevel）= `toy-completed`。**

纪律红线：
- **paper-level 在 15 篇中仍为 0/15。** 本篇也不例外。
- 当前实现（`reproduce/experiments/map_uq_toy.py`）在 **32×32 合成 Fourier 欠采样反问题**上跑，用 **gradient step + Gaussian-filter** 作为 MAP 求解的轻量代理，并用一段 **Gaussian-perturbation 链**充当 "toy MCMC"。这**不是**论文的 forward-backward splitting（Algorithm 1/2），也**不是**论文的真实 HPD 近似公式。
- `runMetrics` 中 `map_runtime_seconds≈0.0017`、`mcmc_runtime_seconds≈0.0041` 的**时间比约 2.4 倍**，**绝不可**等同或外推为论文 Table 1 报告的 **≈10⁵ 倍** 加速。后者来自 256×256 真实 RI 反问题上 Px-MALA（数千分钟）对比 MAP（百分之几分钟），与 toy 完全不同量级。dashboard 的 `notes` 字段已明确写明此点，复现时务必沿用该警告口径。

---

## 3. 算法完整流程

论文核心是：**先用 convex optimisation 求 MAP 点估计 x_map，再用 probability concentration 的 HPD 阈值公式从 x_map 后处理出三类 UQ 输出**（HPD credible region / local credible intervals / hypothesis testing）。MAP 不是终点，而是 UQ 近似的入口。

### 3.1 RI 反问题与 Bayesian 设定（论文 §2）

- 离散设定下 `x ∈ ℝ^N` 为天空亮度图（sky brightness），可在字典 Ψ 下稀疏表示 `x = Ψa = Σ_i Ψ_i a_i`（Eq.1），`Ψ ∈ ℂ^{N×L}`（如 wavelet/over-complete dictionary）。
- 观测模型（Eq.2）：`y = Φx + n`，`y ∈ ℂ^M` 为可见度（visibilities），`Φ ∈ ℂ^{M×N}` 为线性测量算子（partial Fourier + degridding），`n` 为 i.i.d. Gaussian 噪声。恢复 x 是一个**病态线性反问题**。
- Bayesian 框架下后验 `p(x|y) ∝ exp{−f(x) − g(x)}`，其中 g 为似然项、f 为 sparsity-promoting prior 项。

### 3.2 MAP 估计模型（论文 §2.4，核心公式 Eq.3 / Eq.4）

- **Analysis 模型**（Eq.3）：

  ```
  x_map = argmin_x { μ‖Ψ† x‖₁ + ‖y − Φx‖₂² / (2σ²) }
  ```

- **Synthesis 模型**（Eq.4）：

  ```
  x_map = Ψ × argmin_a { μ‖a‖₁ + ‖y − ΦΨa‖₂² / (2σ²) }
  ```

  第一项是 ℓ₁ sparsity prior（正则化、降不确定性），第二项是与 Eq.2 对应的 Gaussian likelihood。两式均**凸**，可用 forward-backward / Douglas-Rachford / ADMM 高效求解（论文实验采用 forward-backward，附录 A）。

### 3.3 Forward-Backward MAP 求解（论文 §3，Algorithm 1 / 2）

记 analysis 设定下 `f̄(x)=μ‖Ψ†x‖₁`、`ḡ(x)=‖y−Φx‖₂²/(2σ²)`。

- **梯度（forward）步**（Eq.8）：`∇ḡ(x) = Φ†(Φx − y)/σ²`。
- **近端（backward）步**：用 soft-thresholding 算 `prox_{λ f̄}`。当 `Ψ†Ψ=I` 时有闭式（Eq.7）；一般 `Ψ†Ψ≠I` 时按 Eq.9-10 迭代。

**Algorithm 1（analysis）核心迭代**：

```
v^{(i+1)} = x^{(i)} − λ^{(i)} Φ†(Φ x^{(i)} − y)/σ²        ← 梯度步 (Eq.9)
u         = Ψ† v^{(i+1)}
x^{(i+1)} = v^{(i+1)} + Ψ ( soft_{λ^{(i)}μ}(u) − u )      ← 近端步 (Eq.10)
```

初始化 `x^{(0)} = Φ†y`（dirty image）。

**Algorithm 2（synthesis）核心迭代**（Eq.15）：

```
a^{(i+1)} = soft_{λ^{(i)}μ}( a^{(i)} − λ^{(i)} Ψ†Φ†(ΦΨ a^{(i)} − y)/σ² )
```

- **停机准则**：达到最大迭代数，或相邻解相对差 `‖x^{(i+1)}−x^{(i)}‖₂/‖x^{(i)}‖₂` 小于容差（论文实验取 max 500 iters，tol = 10⁻⁴）。
- **复杂度**：每步主要是 Φ/Φ† 的应用，实用上用非均匀 FFT（NUFFT），整体 `O(MJ + N log N)`，J 为 degridding 卷积核支持（见 Pratley et al. 2018）。

### 3.4 近似 HPD credible region（论文 §4.1，核心公式 Eq.18-20）

- **理论 HPD region**（Eq.17）：`C_α := { x : f(x)+g(x) ≤ γ_α }`，其中 γ_α 由 `∫_{C_α} p(x|y)dx = 1−α`（Eq.16）确定，是后验质量 100(1−α)% 的最小体积集合。直接算 γ_α 要做高维积分，big-data 不可行。
- **MAP 近似 HPD region**（Eq.18，本篇关键）：

  ```
  C'_α := { x : f(x)+g(x) ≤ γ'_α }
  ```

  **近似阈值**（Eq.19，来自 Pereyra 2017 的 concentration inequality）：

  ```
  γ'_α = f(x_map) + g(x_map) + τ_α √N + N,   其中  τ_α = √( 16 log(3/α) )
  ```

  N 为 x 的维度，100(1−α)% 为 credible level。**关键优势**：γ'_α 只需 x_map 即可算出，即使 N 极大也成立 → 这是整篇可扩展性的根源。
- **误差界与保守性**（Eq.20）：对 `α ∈ (4 exp(−N/3), 1)`，

  ```
  0 ≤ γ'_α − γ_α ≤ η_α √N + N,   其中  η_α = √(16 log(3/α)) + √(1/α)
  ```

  误差至多随 N **线性**增长；且 `γ'_α ≥ γ_α` ⇒ `C'_α ⊇ C_α`，近似在理论上**保守（conservative）**——宁可高估 credible region。
- analysis / synthesis 各自有对应版本（Eq.21-24）：`γ̄'_α = f̄(x_map)+ḡ(x_map)+τ_α√N+N`（analysis），`γ̂'_α = f̂(a_map)+ĝ(a_map)+τ_α√N+N`（synthesis）。

### 3.5 Local credible intervals（论文 §4.2，Eq.25-28）

把图像域 Ω 划分为 superpixels `Ω = ∪_i Ω_i`（不同尺度：单像素到块）。index 算子 `ζ_{Ω_i}`（Eq.25）：`(ζ_{Ω_i})_k = 1` 当 `k∈Ω_i`，否则 0。对每个 Ω_i 求饱和 HPD 边界的强度上下界（Eq.26-27）：

```
ξ_{−,Ω_i} = min_ξ { ξ | f(x_{i,ξ}) + g(x_{i,ξ}) ≤ γ'_α,  ∀ξ∈[0,+∞) }
ξ_{+,Ω_i} = max_ξ { ξ | f(x_{i,ξ}) + g(x_{i,ξ}) ≤ γ'_α,  ∀ξ∈[0,+∞) }
```

其中 `x_{i,ξ} = x_map ⊙ (I − ζ_{Ω_i}) + ξ ζ_{Ω_i}`——把 x_map 在 Ω_i 上的强度替换为常数 ξ、区域外保持不变。汇总成两张图（Eq.28）`ξ_− = Σ_i ξ_{−,Ω_i} ζ_{Ω_i}`、`ξ_+ = Σ_i ξ_{+,Ω_i} ζ_{Ω_i}`，**差图 `(ξ_+ − ξ_−)` 即 local credible interval 长度（error bars）**。论文用 10×10、20×20、30×30 三种 superpixel 尺度。

### 3.6 Hypothesis testing of image structure（论文 §4.3，Eq.29-31）

对感兴趣结构做 **knock-out test**：构造 surrogate 图 `x*_{sgt}`，把结构"抹掉"（用 segmentation-inpainting）。

- **Segmentation-inpainting**（Eq.29，迭代，max 200 iters）：

  ```
  x^{(m+1),sgt} = x*·1_{Ω−Ω_D} + Λ† soft_{λ_thd}(Λ x^{(m),sgt})·1_{Ω_D}
  ```

  Λ 为 wavelet 算子，Ω_D 为测试区域；区域外保留 x*，区域内反复软阈值平滑去除结构。
- **决策**（Eq.30）：若 `f(x*_{sgt}) + g(x*_{sgt}) ≤ γ'_α`，即 `x*_{sgt} ∈ C'_α`，则 likelihood 对该结构**不敏感** → 缺乏强证据（无法判定物理性，可能是 artefact）；若 `> γ'_α`，即 `x*_{sgt} ∉ C'_α`，则**强支持该结构是物理的**。
- 论文另给一种聚焦 sub-structure 的 surrogate（Eq.31）：`x*_{sgt} = x*·1_{Ω−Ω_D} + (Sx*)·1_{Ω_D}`，S 为平滑算子，用于评估区域内子结构。

### 3.7 可执行 step-by-step pipeline

1. 由可见度 y、测量算子 Φ、字典 Ψ、噪声 σ、正则化 μ 建立 analysis（或 synthesis）MAP 目标 Eq.3/Eq.4。
2. 用 forward-backward（Algorithm 1/2）迭代求 `x_map`（初始化 dirty image `Φ†y`）。
3. 算目标值 `f(x_map)+g(x_map)`，按 Eq.19 得 `γ'_α`（取 α=0.01 → 99% level）。
4. **HPD region**：得到隐式集合 `C'_α = {x: f+g ≤ γ'_α}`，作全局 UQ。
5. **Local intervals**：对各 superpixel 尺度，按 Eq.26-27 二分/线搜索求 `ξ_±`，输出差图。
6. **Hypothesis testing**：对每个待测结构做 segmentation-inpainting（Eq.29）得 `x*_{sgt}`，按 Eq.30 判定。
7. 与论文 I 的 Px-MALA 结果逐项对照（credible interval 长度、结构判定是否一致、CPU 时间）。

---

## 4. 完整复现所需数据集

论文实验在**模拟 RI 观测**上进行（§5.1）。下表给出论文实际使用的图与达 paper-like 的等价候选。

| 项 | 论文使用（PDF 实证） | 公开 / 等价候选 | 备注 |
|----|----------------------|-----------------|------|
| 测试 sky 图 | **M31 galaxy**（256×256）、**Cygnus A galaxy**（256×512）、**W28 supernova remnant**（256×256）、**3C288**（256×256） | 这些是 RI imaging 文献的标准 benchmark 图（多见于 Cai et al. 2017a、Pratley et al. 2018、Onose et al. 2016 配套代码 / SARA-CS 数据集） | 见 Fig.2（M31）、Fig.3（其余三幅） |
| 可见度数据 y | 在上述 ground truth 上**模拟** RI 观测（与论文 I 同方式），Φ 为 RI 测量算子（采样 uv 平面，主要覆盖低频，高频测点少） | 需自建/复用 RI measurement operator（partial Fourier + NUFFT/degridding），如 PURIFY / SOPT 工具链 | §5.1 明确"in a manner akin to Cai et al. 2017a" |
| 字典 Ψ | **Daubechies 8 wavelets**（§5.1） | pywt 可直接构造 | analysis 与 synthesis 同用 |

> 论文未使用私有医学数据；主要数据障碍在于**重建一个忠实的 RI measurement operator Φ（含真实 uv 采样与 degridding）**及对应的可见度模拟，而非数据获取本身。

---

## 5. 对照基线 (Baselines)

| 类别 | 论文中的对照 / 基线（PDF 实证） |
|------|--------------------------------|
| UQ 基准方法 | **Px-MALA**（proximal Metropolis-adjusted Langevin，论文 I 的 MCMC 方法），作为 (asymptotically) exact 基准，逐项对照 MAP-UQ（§5）。论文 I 还含 MYULA，但本篇取 Px-MALA 作 benchmark（更准）。 |
| 重建对照 | **Dirty map**（直接 inverse Fourier transform 可见度，Fig.2b/3b）作为未正则化参照；analysis vs synthesis 两套 MAP 模型互相对照（Fig.2c-f）。 |
| 上下文基线（Introduction 提及，未逐表对照） | CLEAN-based、MEM、compressed sensing（CS）等传统重建——它们**不提供** UQ，故仅作背景对比。 |
| 合理的最小对照（toy 层） | direct inverse-FFT（dirty image）vs MAP 重建；MAP-UQ uncertainty map vs 一段小采样得到的经验 interval（本仓库当前 toy 即此口径，但属代理）。 |

---

## 6. 评价指标与论文报告结果

### 6.1 指标定义
- **重建保真度**：与 ground truth 对比（论文 Fig.2/3 以 log₁₀ scale 定性展示 analysis/synthesis MAP 与 Px-MALA 一致；本仓库 toy 用 PSNR/SNR 作量化代理）。
- **HPD credible region 近似误差**：MAP 近似阈值 γ'_α 相对 Px-MALA 计算的 exact HPD 阈值的相对误差。
- **Local credible interval 长度**：各 superpixel 尺度上 `(ξ_+ − ξ_−)`，并与 Px-MALA 的 exact local intervals 对比。
- **效率**：CPU 时间（分钟），MAP vs Px-MALA。

### 6.2 论文报告的关键数值（PDF 能确认者，注明出处）
- **CPU 时间，Table 1（单位：分钟，analysis / synthesis）**：

  | 图像 | Px-MALA | MAP |
  |------|---------|-----|
  | M31 | 1307 / 944 | 0.03 / 0.02 |
  | Cygnus A | 2274 / 1762 | 0.07 / 0.04 |
  | W28 | 1122 / 879 | 0.06 / 0.04 |
  | 3C288 | 1144 / 881 | 0.03 / 0.02 |

  来源：**Table 1 图注与表体**。Table 1 图注明确 **"MAP estimation is approximately 10⁵ times faster than Px-MALA and can be scaled to big-data."**（abstract 也述 ≈10⁵×）。
- **HPD 近似误差**：§5.3 报告 MAP 近似 (Eq.22/24) 相对 Px-MALA 计算的 exact HPD 阈值，**在所有情形误差介于 1%–5%**，与 Pereyra (2017) 一致；并确认近似**保守**（高估 credible region）。
- **Local credible intervals（§5.4 定性结论）**：MAP 近似的 interval 长度**理论上保守**、略**高估** Px-MALA 的长度，故 trustworthy；coarser scale → interval 更短，object boundaries → interval 更长（高频信息少、采样 Φ 偏低频导致）。
- **实验参数（§5.1）**：μ = 10⁴；Ψ = Daubechies 8 wavelets；λ^{(i)} = 0.5；max 500 iters 或相对差 10⁻⁴；α ∈ [0.01, 0.99]，credible regions/intervals 取 α=0.01（99% level）；segmented-inpainting max 200 iters。
- **Px-MALA 运行环境**：在 high-performance workstation 上（论文 I），而 MAP 在 Macbook laptop（i7 / 16GB）上即可——进一步凸显效率差。

> 除上述外，本篇正文以图示（Fig.2-3 重建、Fig.4-9 系列 UQ 图）与 Table 1 时间为主。**禁止编造任何未在 PDF 出现的数字**（如 toy 的 PSNR、interval 长度均**不是**论文报告值）。

---

## 7. 本仓库当前复现实现

- **runner 文件**：`reproduce/experiments/map_uq_toy.py`（一个 runner 同时产出第 11 篇 high-dimensional-uq、第 12 篇 ri-uq-i、第 13 篇 ri-uq-ii 的 toy 结果，共用同一张图）。
- **它实际做了什么**：
  - 合成 **32×32** 二圆盘 ground truth（强度 0.85 / 0.55）。
  - 构造随机 Fourier 欠采样 mask（≈34% 采样），加复 Gaussian 噪声，得 toy 可见度 `y = mask·(FFT(x)+noise)`。
  - **"MAP" 代理**：35 步梯度下降 `recon ← clip(recon − 0.55·∇, 0, 1)`，每步后接 `gaussian_filter`（sigma=0.45）——这是 smoothing 代理，**非** forward-backward + ℓ₁ soft-thresholding。
  - **HPD/uncertainty 代理**：`uncertainty = gaussian_filter(σ·ones) + 0.15·|∇recon|`；`gamma_alpha = Σ residual² + √(N)`——**非** Eq.19 的 `f(x_map)+g(x_map)+τ_α√N+N`，缺 ℓ₁ 项、缺 τ_α。
  - **"MCMC" 代理**：120 步 Gaussian-perturbation + Gaussian-filter 链，丢弃前 40 步，取 95–5 百分位差作 interval——**非** Px-MALA / proximal MCMC，无 accept/reject、无收敛诊断。
  - 产图 `assets/repro/map_uq_reconstruction_uncertainty.png`（4 联：truth / MAP toy / HPD approx map / MCMC interval）。
- **本篇当前 runMetrics（来自 dashboard `reproStructured`）**：

  | 指标 | 数值 | 含义 |
  |------|------|------|
  | `map_psnr` | 18.7123 | toy MAP 重建对 ground truth 的 PSNR（dB，代理保真度） |
  | `map_snr` | 9.6004 | toy MAP 重建 SNR |
  | `map_runtime_seconds` | 0.0017 | toy MAP 代理运行时间 |
  | `mcmc_runtime_seconds` | 0.0041 | toy "MCMC" 代理运行时间 |
  | `gamma_alpha_toy` | 939.9229 | toy 的 γ 代理值（**非** Eq.19 口径） |
  | `mean_interval_length` | 0.1739 | toy interval 平均长度（代理 error bar） |

  > 上述均为 **32×32 合成 toy** 结果。`map/mcmc` 时间比约 **2.4×**，与论文 Table 1 的 **≈10⁵×** 完全不同量级，**不可外推**。
- **resultFiles**：`assets/repro/map_uq_reconstruction_uncertainty.png`。
- **fidelity 警告（dashboard `notes` 已记录）**：*"Toy MAP-UQ is faster than the toy sampler and gives a similar uncertainty pattern; not a paper-level SKA experiment. Toy runtime comparison is not comparable to the paper's large-scale 10⁵ speedup claim."*

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

**到 paper-like 的缺口清单：**

1. **MAP 求解器对齐**：当前用 gradient + Gaussian filter；应改为论文 Algorithm 1/2 的 **forward-backward splitting**，含真正的 ℓ₁ soft-thresholding（Eq.10/15）、Daubechies 8 wavelet Ψ、μ=10⁴、λ=0.5、max 500 iters / tol 10⁻⁴。
2. **HPD 近似公式对齐**：当前 `gamma_alpha` 缺 ℓ₁ 项与 τ_α；应实现 Eq.19 `γ'_α = f(x_map)+g(x_map)+τ_α√N+N`，τ_α=√(16 log(3/α))，并验证 Eq.20 的保守性。
3. **真实 RI measurement operator Φ**：当前是裸 partial FFT；应实现含 uv 采样 + degridding（NUFFT）的 RI 算子（PURIFY/SOPT 风格），偏低频采样以再现论文"高频不确定性更大"的结论。
4. **真实测试图**：当前 32×32 双圆盘；应接入 **M31 / Cygnus A / W28 / 3C288** 标准 RI benchmark 图（256×256 / 256×512）。
5. **真实基线 Px-MALA**：当前 "MCMC" 是 Gaussian 扰动链；应实现论文 I 的 **Px-MALA**（proximal MALA，含 MH accept/reject）作 exact 基准，并对照 1%–5% HPD 误差区间。
6. **Local credible intervals**：当前只产一张 uncertainty heatmap；应实现 Eq.26-27 的 superpixel 线搜索，复现 10×10 / 20×20 / 30×30 三尺度差图。
7. **Hypothesis testing**：完全未实现 Eq.29-31 的 segmentation-inpainting knock-out test。
8. **Analysis vs synthesis**：当前只一套代理；应同时跑 Eq.3 与 Eq.4 两模型并对照（Fig.2c-f）。

**到 paper-level 的额外缺口：**

9. 需严格对齐论文模拟 RI 观测的 uv 覆盖、噪声水平、随机种子，逐图复现 Table 1 的 CPU 时间量级（在可比硬件上），并复现 §5.3 的 1%–5% 误差与 §5.4 的 interval 尺度规律。
10. 需在真实/接近真实硬件上同时跑 Px-MALA（数千分钟级）与 MAP（分钟内），才能谈论 **≈10⁵×** 这一论文级结论——这是当前 toy 完全无法触及的。

---

## 9. 运行步骤

### 9.1 当前 toy 跑法

```bash
# 安装依赖（见下）
pip install -r requirements.txt

# 运行全部复现实验（含本篇 map_uq_toy）
cd reproduce && python run_all.py

# 或在仓库根校验 15 篇数据 / PDF / 笔记 / 静态复现资产
node docs/scripts/validate.mjs
```

- **依赖**（来自 `reproStructured.dependencies`）：`numpy`、`scipy`、`scikit-image`、`matplotlib`。
- **算力**：CPU 秒级（整段 runner `runtimeSeconds≈0.075`，MAP 代理本身 ≈0.0017 s）。
- **数据**：合成 32×32 Fourier 反问题，**无需下载真实数据**。
- 缺依赖时 runner 写 `skipped`（含 `skipped_reason`），**不伪造 completed**（遵守 CLAUDE.md 纪律）。

### 9.2 向 paper-like 扩展的步骤大纲

1. 把 MAP 代理换成 forward-backward（Algorithm 1/2）+ Daubechies 8 wavelet + ℓ₁ soft-thresholding，参数对齐 §5.1。
2. 实现 RI measurement operator Φ（partial Fourier + NUFFT degridding），并按论文 I 方式在 M31/Cygnus A/W28/3C288 上模拟可见度。
3. 实现 Eq.19 HPD 阈值与 Eq.20 误差界检查（保守性）。
4. 实现 Eq.26-27 local credible intervals（三尺度 superpixel）与 Eq.29-31 hypothesis testing。
5. 实现 Px-MALA 基准（或复用第 12 篇 ri-uq-i 的真实 sampler），对照 1%–5% HPD 误差、interval 长度、结构判定一致性。
6. 在 dashboard 中把本篇 `reproductionLevel` 从 `toy` 升级前，单独标注哪些子模块已达 partial，避免用单一 toy 数字代表整篇。

---

## 10. 风险与代理说明

- **MAP 代理的局限**：gradient + Gaussian filter 是**各向同性线性**平滑，不是凸 ℓ₁-regularised MAP 的 forward-backward 解，不保边、无 sparsity、无 Eq.3/4 的最优性。故 `map_psnr=18.71` 只能反映"欠采样反问题能粗略恢复结构"的**定性**事实，不能当作论文重建质量。
- **HPD 代理的局限**：`gamma_alpha_toy=939.92` 缺 ℓ₁ 项与 τ_α√N，**不是** Eq.19 的近似 HPD 阈值，更未验证 Eq.20 的保守性界。toy uncertainty map 是手工合成的启发式，不是任何 credible region。
- **"MCMC" 代理的局限**：Gaussian-perturbation 链无 Metropolis-Hastings 校正、无 burn-in 诊断、无 detailed balance，**不是** Px-MALA/MYULA，不能代表论文 I 的 exact 基准。
- **时间比不可外推**：toy 的 ≈2.4× 与论文 ≈10⁵× 是两个量级的概念。前者只是"小优化比小采样略快"；后者来自 256×256 真实 RI 上 Px-MALA（数千分钟）对 MAP（分钟内）。**严禁**把 toy 时间比表述为论文级加速。
- **不可外推的结论**：① 不能说本仓库"复现了" RI-UQ II 的任何论文结果；② toy PSNR/interval/时间均非论文报告值；③ 三类 UQ 输出（HPD region / local intervals / hypothesis testing）中，hypothesis testing 完全未实现，HPD 与 local intervals 仅有代理；④ paper-level 在 15 篇中仍为 0/15，本篇亦然。

---

## 11. 参考：精读笔记

完整中文精读笔记见：
[`../../../xiaohao_cai_ultimate_notes/Radio_Interferometric_Imaging_II_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Radio_Interferometric_Imaging_II_超精读笔记_已填充.md)
