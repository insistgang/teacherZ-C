# 高维逆问题不确定性量化入口 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 11 篇 *Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation* 的完整复现流程规范。

---

## 1. 论文身份与第一作者核验

| 项 | 内容 |
|----|------|
| **标题 (EN)** | Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation |
| **标题 (CN)** | 用凸优化量化高维逆问题中的不确定性 |
| **作者顺序** | **Xiaohao Cai** (1st), Marcelo Pereyra (2nd), Jason D. McEwen (3rd) |
| **第一作者核验** | 是。PDF 首页作者列表标注 "1st Xiaohao Cai"，单位为 Mullard Space Science Laboratory, University College London (UCL)，邮箱 `x.cai@ucl.ac.uk`。确认 Xiaohao Cai 为第一作者。 |
| **合作者单位** | Marcelo Pereyra：Maxwell Institute for Mathematical Sciences, Heriot-Watt University（注意：PDF 明确写 Heriot-Watt，而非旧笔记一度提到的 University of Geneva）；Jason D. McEwen：UCL MSSL。 |
| **年份 / 出处** | 2019，2019 27th European Signal Processing Conference (EUSIPCO)，IEEE，论文号 978-9-0827-9703-9/19。arXiv 预印本 1811.02514（v1: 2018-11；会议版 2019）。 |
| **资助** | EPSRC EP/M011089/1；STFC ST/M00113X/1；Leverhulme Trust。 |
| **类型** | 短会议论文（5 页正文 + 参考），方法演示型，非长篇理论论文。 |
| **PDF 路径** | `docs/00_papers_first_author_xiaohao_cai_deduped/高维逆问题不确定性量化 Uncertainty Quantification.pdf` |
| **主题 (theme)** | `ri-uq`（radio interferometric imaging 不确定性量化主线） |

定位：本篇是把作者在 radio interferometric (RI) imaging 中发展的 **MAP-based UQ** 思路（[5][6] = UQ for RI I & II）**抽象/一般化**到任意 image/signal processing 逆问题的"短入口"。它的核心方法承自 Pereyra [11]（"Maximum-a-posteriori estimation with Bayesian confidence regions", SIIMS 2016）的 HPD region approximation，并在其上**新增两个组件**：(i) 正则化参数 μ 的自动估计；(ii) over-complete dictionary（SARA）下 synthesis/analysis prior 的 UQ 对比。

---

## 2. 复现目标与诚实分级

本项目对"复现"采用四级诚实分级（由弱到强）：

| 级别 | 含义 |
|------|------|
| **toy** | 合成小图 + 代理算子，演示直觉，不对齐论文任何具体数值 |
| **partial** | 实现了论文核心步骤的一部分（如真实 forward-backward 求解器 + 真实 HPD 阈值公式），在合成数据上验证趋势，但未对齐论文数据集与报告数值 |
| **paper-like** | 用论文同款或公开等价数据集（M31、BrainWeb MRI），跑论文同款 pipeline（auto-μ + MAP + γ'_α + local credible interval），复现论文表格量级（不要求逐位一致） |
| **paper-level** | 严格复现论文报告数值（同数据、同基线 Px-MALA、同指标、同表号 Table I / Fig 3-5） |

**本仓库当前等级（reproductionLevel）= `toy`；真实性（reproductionTruthLevel）= `toy-completed`。**

纪律红线：
- **paper-level 在 15 篇中仍为 0/15。** 本篇也不例外。
- 当前实现（`reproduce/experiments/map_uq_toy.py`）已**从旧 Gaussian 平滑代理升级为论文真实方法的可运行实现**：用 **真实 ℓ1-wavelet（Daubechies-8）analysis-prior MAP via FISTA / forward-backward splitting**（闭式软阈值 prox）求解 `μ‖Ψ†x‖₁+‖y-Ax‖²/2σ²`；HPD 阈值用 **完整 Eq.(6)** `γ'_α=f(x*)+g(x*)+√(16log(3/α))√N+N`（concentration inequality，α=0.01）；局部可信区间用 **Eq.(7-9) superpixel 二分搜索**。但用的是 **64×64 标准 Shepp-Logan 测试图**（非论文 M31/BrainWeb），且无 SARA 字典、无自动 μ、无 Px-MALA/MCMC 采样器。
- `runMetrics` 中的 `map_psnr=22.2878`、`map_snr=9.1586`、`gamma_alpha_hpd=6180.6593`、`mean_interval_length=0.3709` 等均为 **64×64 Shepp-Logan 标准测试图上的真实算法结果**，**不得**表述为论文级精度或论文 Table I 数值。
- runner `fidelityWarning` 已明确写明指标与论文 **O(10^5) 相对 Px-MALA 的加速不可对照**；该加速只在大规模真实 RI pipeline 下成立，本仓库无 Px-MALA 基线、亦无 MAP-vs-MCMC 时间比，**不可外推**。wall-clock runtime（`map_runtime_seconds`/`lci_runtime_seconds` 等）为亚秒级且随运行波动，**不作复现指标**。

---

## 3. 算法完整流程

论文 pipeline 见 Fig. 1：Observations y → 形成 objective functional → 自动选 μ → MAP 估计 x*_μ → 同时输出 (Approximate HPD credible region C'_α) 与 (Approximate local credible intervals (ξ_-, ξ_+))。下面按可执行步骤拆解，公式编号沿用论文。

### 3.1 问题设定与贝叶斯后验

1. **线性观测模型**（Eq. 1）：
   `y = Φx + n` 或 `y = ΦΨa + n`
   其中 `y ∈ C^M` 为部分/受限观测，`Φ ∈ C^{M×N}` 为前向算子（如 Fourier transform + downsampling mask、blur、Radon），`x ∈ R^N` 为干净信号/图像，`n ∈ C^M` 为 i.i.d. 高斯噪声。`Ψ ∈ C^{N×L}` 为字典/frame，`x = Ψa`，`a` 为 synthesis 系数。complete: `L = N`；over-complete: `L > N`。

2. **贝叶斯后验**（Eq. 2）：
   `p(x|y) = p(y|x)p(x) / ∫_{R^N} p(y|x)p(x) dx`
   一般形式：似然 `p(y|x) ∝ exp(-g_y(x))`，先验 `p(x) ∝ exp(-μ f(x))`，μ 为正则化参数。经典选择：
   - 数据保真 `g_y(x) = ‖y - Φx‖_q^q / 2σ²`（q=2 时为高斯），σ 为噪声标准差；
   - 正则项 `f(x) = ‖Ψ† x‖_s`（常取 s=1，即 analysis ℓ1）。

### 3.2 自动正则化参数选择（论文新组件之一）

3. **联合 MAP 的层级贝叶斯迭代**（Eq. 4，承自 Pereyra et al. [25]）：
   ```
   x^(i)  = argmin_x { μ^(i-1) f(x) + g_y(x) }
   μ^(i)  = (N/k + γ - 1) / (f(x^(i)) + β)
   ```
   其中 γ, β 为固定超参（默认 = 1），k 与 f 的定义相关（f 为 ℓ1 范数时 k = 1）。论文实验用 **10 次迭代**，并取 γ = β = k = 1。
   （**勘误提示**：本仓库旧笔记一度把分子写成 `N/k + γ^{-1}`，与 PDF Eq.(4) 的 `N/k + γ - 1` 不一致，应以 PDF 为准。）

### 3.3 MAP 估计（凸优化）

4. **MAP estimator**（Eq. 3）：
   `x*_μ = argmin_{x∈R^N} { μ f(x) + g_y(x) }`
   论文假设 f 与 g_y 为 closed convex（不一定可微）。求解用 convex minimisation：**forward-backward splitting**、**Douglas-Rachford splitting**、**primal-dual**、**ADMM** 等（论文实验明确采用 [12] 中的 forward-backward splitting）。当 Φ 为 Fourier/RI 算子时用 FFT 加速 Φ、Φ†；ℓ1 的 proximal 为软阈值，闭式。

### 3.4 HPD credible region 的高维近似（论文核心继承自 [11]）

5. **精确 HPD region**（Eq. 5）：
   `C_α := { x : μ f(x) + g_y(x) ≤ γ_α }`，γ_α 满足 `∫_{x∈C_α} p(x|y)·1_{C_α} dx = 1 - α`。
   高维下精确求 γ_α 不可行。

6. **HPD 阈值近似**（Eq. 6）：
   `γ'_α = μ f(x*_μ) + g_y(x*_μ) + √(16 log(3/α)) · √N + N`
   该近似来自信息论中的 **probability concentration inequality**，对大 N 渐近精确（[11][12]）。用 C'_α（由 γ'_α 定义）替代 C_α。

### 3.5 局部可信区间（pixel / superpixel error bars）

7. **区域划分**：把 image domain Ω 分成不相交的 superpixels {Ω_i}，`Ω_i ∩ Ω_j = ∅ (i≠j)`，`Ω = ∪_i Ω_i`。论文实验用 grid scale **10×10 与 15×15**（Fig. 4）。

8. **每个 Ω_i 的局部上下界**（Eq. 7, 8）：
   ```
   ξ_{-,Ω_i} = min_ξ { ξ | μ f(x_{i,ξ}) + g_y(x_{i,ξ}) ≤ γ'_α, ∀ ξ ∈ [0,+∞) }
   ξ_{+,Ω_i} = max_ξ { ξ | μ f(x_{i,ξ}) + g_y(x_{i,ξ}) ≤ γ'_α, ∀ ξ ∈ [0,+∞) }
   ```
   其中 `x_{i,ξ} = x*_μ ⊙ ζ_{Ω\Ω_i} + ξ ζ_{Ω_i}`：在 Ω_i 内把像素值统一设为常数 ξ，其余像素保持 MAP 值（ζ_{Ω_i} 为 Ω_i 上为 1、其余为 0 的指示算子）。ξ_{-,Ω_i}, ξ_{+,Ω_i} 是使该区域饱和 HPD region C'_α 的最小/最大值，可用二分搜索求解。

9. **拼接成全局区间**（Eq. 9）：
   `ξ_- = Σ_i ξ_{-,Ω_i} ζ_{Ω_i}`，`ξ_+ = Σ_i ξ_{+,Ω_i} ζ_{Ω_i}`。
   局部区间长度图 `(ξ_+ - ξ_-)` 即逐像素/逐 superpixel 的 error bar，是论文主要的 UQ 可视化产物（Fig. 4）。

---

## 4. 完整复现所需数据集

| 数据 | 论文用途 | 公开/等价候选 | 备注 |
|------|----------|---------------|------|
| **M31 星系图** | RI imaging 测试图（Fig. 2 左，log10 尺度） | M31 是天文学常用 benchmark，PURIFY / RI 重建文献中广泛使用其干净图像作为 ground truth | 论文以其作为 RI 应用展示；为 paper-like 需取同一 256×256 M31 干净图并自建 `Φ = mask · FFT` |
| **MRI brain image** | medical imaging 测试图（Fig. 2 右） | **BrainWeb** 模拟脑数据库（ref [30]，http://brainweb.bic.mni.mcgill.ca/brainweb），公开 | 论文明确引用 BrainWeb 作为 MRI 来源，可直接下载等价图像 |
| **观测算子 Φ** | Fourier transform + downsampling mask，`M = N/10` | 可在代码中自建：随机/径向欠采样掩模 + FFT | 论文 RI/MRI 均用 Fourier + mask，采样率 10% |
| **图像尺寸** | 256 × 256（Fig. 4 标注） | — | 当前为 64×64 合成 Shepp-Logan 测试图 |
| **噪声** | i.i.d. 高斯，`σ = ‖x*‖_∞ · 10^{-SNR/20}`，SNR = 30 | — | 与 g_y 中 σ 一致 |

私有/受限数据说明：本篇**不依赖**私有医学或私有 RI 观测数据——M31 干净图与 BrainWeb 均为公开，观测是合成欠采样。因此 paper-like 在数据侧**可行**，主要门槛在求解器与 dictionary，而非数据获取。

---

## 5. 对照基线 (Baselines)

| 基线 | 角色 | 论文中的体现 |
|------|------|--------------|
| **Px-MALA** (proximal MALA, ref [6] = Pereyra 2016 "Proximal Markov chain Monte Carlo algorithms") | 主基线 / ground-truth 参照 | Fig. 5 中以 Px-MALA 计算的 local credible interval 长度作 benchmark，衡量 MAP 估计的相对误差；MAP 比 Px-MALA 快 O(10^5) |
| **Orthonormal basis (DB8 wavelet)** vs **over-complete SARA dictionary** | dictionary 对照 | Table I / Fig. 3 / Fig. 4 全程对比两类字典在 synthesis/analysis prior 下的 UQ 差异 |
| **Synthesis prior** vs **Analysis prior** | 先验形式对照 | Table I 报告两种 prior 的 SNR 与自动 μ；over-complete 字典下两者差异显著，orthonormal 下差异可忽略 |
| **完整 MCMC（隐含背景）** | 概念对照 | 文中反复强调相对完整后验采样（MCMC）的可扩展性优势 |

SARA 字典定义（ref [13]，Carrillo-McEwen-Wiaux）：nine bases 的拼接 = DB1–DB8 wavelets + Dirac basis。

---

## 6. 评价指标与论文报告结果

**指标定义：**
- **SNR**（signal-to-noise ratio）：point estimator x*_μ 相对干净图的重建质量（Table I）。
- **自动 μ**：算法 (4) 收敛得到的正则化参数（Table I）。
- **HPD 阈值 γ'_α**：随置信水平 (1-α) 变化的曲线（Fig. 3）。
- **Local credible interval length**：逐 superpixel 区间长度 `(ξ_+ - ξ_-)`，论文取 α = 0.01（99% credible level）（Fig. 4）。
- **Average relative error**：MAP 估计的区间长度相对 Px-MALA 的逐像素相对误差（Fig. 5）。

**论文报告的关键数值（可从 PDF Table I 直接核实）：**

| Image | Library/basis | SNR (Synthesis) | SNR (Analysis) | 自动 μ |
|-------|---------------|-----------------|----------------|--------|
| M31 | Orthonormal (DB8) | 25.04 | 25.04 | 196 |
| M31 | SARA | 23.66 | 31.09 | 65 |
| Brain | Orthonormal (DB8) | 19.06 | 19.06 | 33 |
| Brain | SARA | 19.89 | 23.63 | 11 |

**定性结论（PDF §IV / Fig. 3-5 可确认）：**
- orthonormal basis 下 synthesis 与 analysis prior 的 SNR 完全相同（25.04/25.04、19.06/19.06）；**over-complete SARA 下两者显著不同**（M31: 23.66 vs 31.09；Brain: 19.89 vs 23.63），这是 over-complete frame 的固有差异。
- Fig. 3 显示 orthonormal 下 synthesis/analysis 的 HPD 阈值 γ'_α 差异可忽略，SARA 下不可忽略，与上一行一致。
- Fig. 5：MAP 估计相对 Px-MALA 的区间长度相对误差**随 grid scale 增大单调下降**，当 grid scale 大于 10×10 时 **低于约 5%**，且 MAP 比 Px-MALA **快 O(10^5)** 量级。
- 实验环境：MacBook laptop，i7 Intel CPU，16 GB 内存，MATLAB R2015b。

**禁止外推**：本篇为 5 页短文，未给出完整覆盖率校准曲线、CRPS 等标准 UQ 评测；旧笔记中"可信区间覆盖率 ≥ 1-α"属于一般性主张而非本 PDF 报告的实验数字，复现时应标注为定性预期而非论文确证数值。

---

## 7. 本仓库当前复现实现

- **runnerFile**：`reproduce/experiments/map_uq_toy.py`（被第 11/12/13 三篇 RI-UQ 论文 runner 共用，统一 experiment_id = `map_uq_toy`）。`reproductionLevel = toy`，但**已从 Gaussian 平滑代理升级为论文真实方法的可运行实现**。
- **实际做了什么（真实算法，非论文数据）**：
  1. **真值图**：`skimage` 的 64×64 Shepp-Logan phantom（**标准测试图，非论文 M31/BrainWeb**），归一化到 [0,1]；
  2. **前向算子 A**：低频偏置的 variable-density Fourier 欠采样掩模（`prob = exp(-(r/0.18)²)`），实际采样率 ≈ **9.8%**（对齐论文 `M = N/10`）；`A(x) = mask·FFT(x)/√N`、`Aᴴ(y) = √N·Re ifft2(mask·y)`，构成 subsampled-unitary 算子（`‖A‖ ≤ 1`）；
  3. **噪声**：i.i.d. 复 Gaussian，按论文约定 **SNR = 30 dB** 设 `σ = sig_power·10^{-30/20}`；
  4. **真实 MAP（FISTA / forward-backward splitting）**：求 `x*_μ = argmin μ‖Ψ†x‖₁ + ‖y-Ax‖²/2σ²`。Ψ = **Daubechies-8 正交小波（pywt，level 2，periodization）**，`Ψ†Ψ=I` ⇒ analysis prox 闭式 `x + Ψ(soft_{tμ}(Ψ†x) − Ψ†x)`；梯度步 `∇g = Aᴴ(Ax−y)/σ²`，步长 `1/L = σ²`（`L = ‖AᴴA‖/σ² = 1/σ²`），FISTA 动量加速，250 次迭代，初值取 dirty image `Aᴴy`；`μ = 3σ`；
  5. **真实 HPD 阈值（完整 Eq.(6)）**：`γ'_α = f(x*) + g(x*) + √(16 log(3/α))·√N + N`，**α = 0.01（99% 可信度）**，`τ_α = √(16 log(3/α))`；
  6. **真实 local credible interval（Eq.(7-9) 二分搜索）**：把图划成 **8×8 像素的 superpixel**；对每个区域，固定区域外为 `x*`、把区域内统一设常数 ξ，二分搜索使 `f(x_{i,ξ})+g(x_{i,ξ}) ≤ γ'_α` 的最小/最大 ξ（各 30 次二分），得 `ξ_-, ξ_+`，区间宽度 `ξ_+ − ξ_-`；
  7. 输出四联图 `assets/repro/map_uq_reconstruction_uncertainty.png`（truth / dirty Aᴴy / ℓ1-wavelet MAP / LCI width）。
- **当前 runMetrics（真实算法，Shepp-Logan 标准测试图结果）：**

| 指标 | 值 | 说明 |
|------|-----|------|
| `map_psnr` | 22.2878 | 真实 ℓ1-wavelet MAP 重建 PSNR |
| `map_snr` | 9.1586 | MAP 重建 SNR |
| `dirty_snr` | 7.4821 | dirty image 基线 SNR |
| `snr_gain_over_dirty_db` | 1.6766 | **MAP 优于 dirty 基线 +1.68 dB**（验证真实先验有效） |
| `sampling_rate` | 0.0979 | Fourier 采样率（≈10%，对齐论文） |
| `noise_sigma` | 0.019635 | SNR=30 dB 对应噪声标准差 |
| `map_runtime_seconds` | 亚秒级 | FISTA MAP 求解时间（wall-clock，随运行波动，不作复现指标） |
| `lci_runtime_seconds` | 亚秒级 | local credible interval 二分搜索时间（wall-clock，随运行波动） |
| `gamma_alpha_hpd` | 6180.6593 | 真实 HPD 阈值 γ'_α（完整 Eq.(6)，α=0.01） |
| `mean_interval_length` | 0.3709 | 平均 superpixel 可信区间宽度 |
| `runtime_seconds`（整体） | 亚秒级 | runner 总耗时（wall-clock，随运行波动，不作复现指标） |

- **fidelityWarning（runner `extra` 字段已写）**：已实现真实 ℓ1-wavelet（db8）MAP + 真实局部可信区间方法，但用的是标准 Shepp-Logan 测试图而非论文 M31/BrainWeb 数据，且无真实射电干涉（NUFFT）算子、无 auto-μ、无 SARA 字典、无 Px-MALA/MCMC；指标与论文 Table I 及 O(10⁵) 加速**不可对照**。
- **resultFiles**：`assets/repro/map_uq_reconstruction_uncertainty.png`。
- **依赖**：numpy, scipy, scikit-image, matplotlib, **pywt（PyWavelets）**（缺失时 runner 写 skipped，不伪造 completed）。

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

| 维度 | 当前实现（升级后） | paper-like 目标 | 缺口 |
|------|----------|-----------------|------|
| **数据** | 64×64 Shepp-Logan 标准测试图 | 256×256 M31 + BrainWeb MRI | **仍缺**：需引入论文真实干净图（公开可得，门槛低）并放大到 256² |
| **观测算子** | low-freq-biased variable-density mask·FFT，采样率 ≈9.8% | Fourier + downsampling mask，`M = N/10`（10%） | 采样率已对齐；掩模 profile 仍为 toy 近似，非论文具体 mask |
| **MAP 求解器** | ✅ **真实 FISTA / forward-backward splitting**（ℓ1 软阈值 prox + FFT 算子） | 真正的 forward-backward splitting 解 `μ‖Ψ†x‖_1 + ‖y-Φx‖²/2σ²` | **已实现**（algorithm 已对齐，差别只在数据规模与字典） |
| **字典 Ψ** | ✅ **Daubechies-8 正交小波（pywt）** | DB8 orthonormal + SARA (DB1-DB8 + Dirac) | DB8 已接入；**仍缺 SARA over-complete 字典**与 synthesis-vs-analysis 对照 |
| **自动 μ** | 无（μ 固定 = 3σ） | 算法 (4)，10 次迭代，γ=β=k=1 | **仍缺**：需实现联合 MAP 不动点迭代估 μ |
| **HPD 阈值** | ✅ **完整 Eq.(6)** `f(x*)+g(x*)+√(16log(3/α))√N+N`，α=0.01 | 完整 Eq.(6) | **已实现** |
| **local credible interval** | ✅ **Eq.(7-9) superpixel 二分搜索**（8×8 grid） | Eq.(7-9) 二分搜索 superpixel 区间，grid 10×10 / 15×15 | **已实现**；grid 尺度可调到论文 10×10 / 15×15 |
| **基线** | dirty image（Aᴴy）作重建下界对照 | Px-MALA（ref [6]）作 ground-truth benchmark | **仍缺**：需实现/接入 proximal MALA 采样器对照 |
| **指标对照** | MAP SNR 9.16 / PSNR 22.29，比 dirty +1.68 dB | Table I 的 SNR & μ、Fig. 5 的 <5% 相对误差曲线 | **仍缺**：换论文数据后才能比 Table I 量级；缺误差-grid 曲线 |
| **置信水平** | ✅ α = 0.01（99%）显式用于阈值与区间 | α = 0.01（99%） | **已实现** |

paper-level 额外门槛：严格复现 Table I 四行 SNR/μ 的具体数值、Fig. 3 的 γ'_α 曲线、Fig. 4 的区间长度图、Fig. 5 相对 Px-MALA 的逐点相对误差曲线，且 synthesis vs analysis、orthonormal vs SARA 四组合全部跑通。

---

## 9. 运行步骤

**当前 toy（已可跑）：**
```bash
cd reproduce
python run_all.py            # 运行全部实验；本篇 experiment_id = map_uq_toy
```
依赖见 reproStructured.dependencies：`numpy, scipy, scikit-image, matplotlib`（`pip install -r requirements.txt`）。缺依赖时 runner 写 `skipped` 而非伪造 `completed`。产物：`docs/assets/repro/map_uq_reconstruction_uncertainty.png` 与 `runMetrics`。

**向 paper-like 扩展的步骤大纲（尚未实现，仅规划）：**
1. 下载/准备 256×256 M31 干净图与 BrainWeb MRI 切片，归一化。
2. 构建 `Φ = downsampling_mask · FFT`，采样率 `M = N/10`；按 `σ = ‖x‖_∞·10^{-30/20}` 加高斯噪声。
3. 实现 forward-backward splitting 求 `x*_μ = argmin μ‖Ψ†x‖_1 + ‖y-Φx‖²/2σ²`，Ψ 分别取 DB8 与 SARA；synthesis / analysis 两种先验各跑一遍。
4. 用算法 (4)（10 iters，γ=β=k=1）自动估 μ，记录到 Table I 对照。
5. 用完整 Eq.(6) 算 γ'_α（α=0.01），Eq.(7-9) 二分搜索算 10×10 / 15×15 的 local credible interval length map。
6. 实现 Px-MALA 采样器，计算其区间长度作 benchmark，画 Fig. 5 的相对误差-grid 曲线，核对 "grid > 10×10 时 <5%"。
7. 与论文 Table I 四行（SNR、μ）做量级对照，**不夸大为逐位一致**。

---

## 10. 风险与代理说明

**升级后已消除的代理**（不再适用）：~~Gaussian smoothing 代替 ℓ1 prox~~、~~极简 `Σres²+√N` 代替 HPD 阈值~~、~~随机游走代替区间搜索~~。现在 MAP 求解（FISTA + 软阈值 prox）、HPD 阈值（完整 Eq.(6)）、local credible interval（Eq.(7-9) 二分搜索）均为**论文真实方法的可运行实现**，且 MAP 比 dirty 基线 SNR 高 +1.68 dB，验证 ℓ1-wavelet 先验确实有效。

**仍存在的局限 / 代理（诚实标注）**：
- **非论文数据**：用的是 skimage Shepp-Logan 标准测试图（64×64），**不是**论文的 M31 星系图或 BrainWeb MRI。因此 `map_snr=9.16`、`map_psnr=22.29` 与论文 Table I（SNR 19–31 dB）**不可逐数对照**——量级差异主要来自数据与规模不同，而非算法不同。
- **掩模为 toy 近似**：low-freq-biased variable-density mask 是对论文 Fourier 欠采样的合理近似（采样率已对齐 ≈10%），但不是论文具体使用的掩模结构。
- **无 SARA / 无 synthesis-vs-analysis 对照**：仅实现 DB8 orthonormal analysis prior，**未**接入 over-complete SARA 字典，因此无法复现论文关于 SARA 下 synthesis/analysis 差异的核心发现（Table I 的 23.66 vs 31.09 等）。
- **无自动 μ**：μ 固定为 3σ，未实现论文算法 (4) 的联合 MAP 不动点迭代，因此不复现 Table I 的自动 μ 数值。
- **无 Px-MALA 基线**：用 dirty image 作重建下界对照，**未**实现 Px-MALA，因此无法复现 Fig. 5 相对 Px-MALA 的 <5% 区间相对误差曲线，也**不能**佐证论文 **O(10^5) 加速**——本 runner 不做任何加速比断言。
- **维度仍偏小**：64×64 < 论文 256×256，而 Eq.(6) 的近似精度依赖大 N 渐近，小图上近似本身偏保守。
- **结论外推红线**：本实现可诚实声称"已用论文真实方法（ℓ1-wavelet MAP + HPD 局部可信区间）在标准测试图上跑通并优于基线"，但**不得**声称复现了论文具体 SNR/μ 数值、SARA 差异、覆盖率或加速比；paper-level 仍为 **0/15**。

---

## 11. 参考：精读笔记

详见同篇精读笔记：[`../../../xiaohao_cai_ultimate_notes/High-Dimensional_Inverse_Problems_UQ_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/High-Dimensional_Inverse_Problems_UQ_超精读笔记_已填充.md)

笔记覆盖：论文元信息与第一作者核验、posterior/MAP/HPD/local credible interval 的数学框架、Eq.(1)-(9) 的逐项解释、自动 μ 算法 (4)、Table I 实验数据、与 RI UQ I/II 及 Proximal Nested Sampling 的关系，以及复现判断。
