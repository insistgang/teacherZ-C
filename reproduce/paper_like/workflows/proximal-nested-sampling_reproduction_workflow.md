# Proximal Nested Sampling 贝叶斯模型选择 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 15 篇 *Proximal Nested Sampling for High-Dimensional Bayesian Model Selection* 的完整复现流程规范。

---

## 1. 论文身份与第一作者核验

| 项 | 内容 |
|----|------|
| **标题 (EN)** | Proximal nested sampling for high-dimensional Bayesian model selection |
| **标题 (CN)** | 高维贝叶斯模型选择的近端嵌套采样 |
| **作者顺序** | **Xiaohao Cai**, Jason D. McEwen, Marcelo Pereyra |
| **第一作者核验** | 是。PDF 首页作者列表以 **Xiaohao Cai** 开头，单位标注为 MSSL/UCL 与 University of Southampton（School of Electronics and Computer Science），通讯邮箱 `x.cai@soton.ac.uk`。确认 Xiaohao Cai 为第一作者。 |
| **年份 / 出处** | 2022（PDF 页眉日期 9 Sep 2022），arXiv:2106.03646v3 [stat.ME]，stat.ME 预印本；后正式发表于 *Statistics and Computing* |
| **类型** | 方法论论文（新算法：proximal nested sampling），含理论框架 + 高维验证 + 成像应用 |
| **PDF 路径** | `docs/00_papers_first_author_xiaohao_cai_deduped/近端嵌套采样 Proximal Nested Sampling.pdf` |
| **主题 (theme)** | `bayes-model`（贝叶斯模型选择/证据计算线，本仓库 15 篇中问题层级最高的一篇） |

本篇是 RI/UQ/Bayesian inverse problem 线（第 11/12/13 篇）的**上层延伸**：从"给定模型下图像是什么、不确定性多大"上升到"哪个模型/prior/dictionary/measurement model 更可信"。合作者 McEwen、Pereyra 分别是 RI 成像与 proximal MCMC 方向的核心人物，本篇正是把两条线缝合在 nested sampling 框架内。

---

## 2. 复现目标与诚实分级

本项目对"复现"采用四级诚实分级（由弱到强）：

| 级别 | 含义 |
|------|------|
| **toy** | 合成小问题 + 代理算子，演示直觉，不对齐论文任何具体数值 |
| **partial** | 实现了论文核心步骤的一部分（如真实 proximal constrained sampler），在合成数据上验证趋势，但未对齐论文数据集与报告数值 |
| **paper-like** | 用论文同款或公开等价数据集，跑论文同款 pipeline，复现论文表格量级（不要求逐位一致） |
| **paper-level** | 严格复现论文报告数值（同数据、同基线、同指标、同表号） |

**本仓库当前等级（reproductionLevel）= `toy`；真实性（reproductionTruthLevel）= `toy-completed`。**

纪律红线：
- **paper-level 在 15 篇中仍为 0/15。** 本篇也不例外。
- 当前实现只复现了 **standard nested sampling 的证据估计骨架**，在 **2D Gaussian likelihood + uniform prior** 上演示"把 evidence 积分变成一维 prior-volume 积分并用 quadrature 求和"这一教学核心。它**完全没有**实现论文的关键创新——**proximal constrained sampler**（Algorithm 2/3/4，依赖 MYULA + Moreau-Yosida 近似 characteristic function + Metropolis-Hastings 校正）。
- 当前 toy 的 `absolute_log_error ≈ 2.4676`（log evidence 误差量级），**很大**，dashboard 已标注 `warning: large evidence error; toy only`。该数值**不得**被表述为论文级证据估计精度，也不对应论文任何报告值。
- 论文展示的是 d 从 2 一路到 **10^6** 的高维可扩展性（§6.2）与成像 model selection（§6.3 Tables 1-3）。当前 toy 只有 d=2，与论文核心卖点（高维 + 非光滑 prior + 真实图像 model selection）之间存在数量级差距。

---

## 3. 算法完整流程

论文把 **nested sampling**（Skilling 2006）与 **proximal MCMC**（Pereyra 2016；Durmus et al. 2018）缝合。前者负责把高维 evidence 积分降维成一维；后者负责在高维、log-concave、可能非光滑的 likelihood 约束区域内采样。下面按 PDF 章节拆成可执行 pipeline。

### 3.1 贝叶斯模型选择背景（论文 §2.1）

- 后验（Bayes 定理，Eq.(1)）：`p(x|y,M) = p(y|x,M)p(x|M) / p(y|M)`。
- **marginal likelihood / model evidence**（Eq.(2)）：
  `p(y|M) = ∫_Ω p(y,x|M) dx = ∫_Ω p(y|x,M) p(x|M) dx`
  它是似然在 prior 下的平均，自然包含 **Occam's razor**（所有 prior 同积分质量 1，复杂模型把质量摊薄，过拟合被惩罚）。
- **Bayes factor**（Eq.(3)(5)）：`ρ₁₂ = p(M₁|y)/p(M₂|y) · p(M₂)/p(M₁) = p(y|M₁)/p(y|M₂)`。likelihood ratio，对 prior 模型概率 invariant；`ρ₁₂ ≫ 1` 选 M₁，`≪ 1` 选 M₂，`≈ 1` 数据不足以区分。
- 困难根源（§2.1 末）：marginal likelihood 是 **doubly-intractable**——既要对 d 维解空间积分，prior 归一化常数往往也不可解析。posterior 本身可用 MCMC 采样而无需 evidence，但 model selection 绕不开 evidence。

### 3.2 proximal MCMC 背景（论文 §2.2）

针对 `π(x) ∝ exp{-f(x) - g(x)}`，其中 `f ∈ C¹` 有 Lipschitz 梯度（常数 `L_f`），`g` 凸、proper、l.s.c. 但可能非光滑（如 ℓ₁）。

- **proximal operator**（Eq.(6)）：`prox_h^λ(x) = argmin_u { h(u) + ‖u-x‖₂²/(2λ) }`。
- **Moreau-Yosida envelope**（Eq.(10)(11)）：`h^λ(x) = min_u { h(u) + ‖u-x‖₂²/(2λ) }`，连续可微，`∇h^λ(x) = (x - prox_h^λ(x))/λ`，`1/λ`-Lipschitz。λ 同时控制光滑度与近似误差。
- **MYULA**（Eq.(13)(14)）：过阻尼 Langevin SDE `dX_t = -[∇f(X_t) + ∇g^λ(X_t)]dt + √2 dW_t` 的 Euler-Maruyama 离散
  `X_{n+1} = X_n - (δ/2)∇f(X_n) - (δ/2λ)(X_n - prox_g^λ(X_n)) + √δ Z_{n+1}`，
  参数 `λ = 1/L_f`，`δ = 0.8/(L_f + 1/λ)`（论文采用 Durmus et al. 2018 推荐值）。

### 3.3 nested sampling 核心（论文 §3）

- evidence 重写为 prior volume 上的一维积分。定义 prior volume（Eq.(17)）`ξ(L*) = ∫_{Ω_{L*}} π(x)dx`，其中 `Ω_{L*} = {x | L(x) > L*}`。则（Eq.(18)）
  `Z = ∫₀¹ L†(ξ) dξ`，
  `L†` 是 `ξ(L*)` 的逆（tail quantile function）。
- quadrature 离散（Eq.(20)）：`Z ≈ Σ_{i=1}^N L_i w_i`，`w_i = ξ_{i-1} - ξ_i` 或梯形 `w_i = (ξ_{i-1} + ξ_{i+1})/2`。
- prior volume 用 **shrinkage ratio** 随机估计（Eq.(22)-(25)）：`ξ_{i+1} = t_{i+1} ξ_i`，`t_{i+1} ~ p(t) = N_live · t^{N_live-1}`，于是 `E(log t) = -1/N_live`，`σ(log t) = 1/N_live`，近似 `ξ_i = exp(-i/N_live)`。
- 误差估计（Eq.(26)(27)）：用 prior-volume 的负相对熵 `H` 给出 `log Z = log(Σ L_i w_i) ± √(H/N_live)`；Chopin & Robert (2010) 证明误差渐近高斯、以 `O(N^{-1/2})` 消失，且**误差随维度 d 近似线性增长**。

### 3.4 proximal nested sampling 框架（论文 §4，核心创新）

nested sampling 最难的一步是"从 prior 采样且满足硬似然约束 `L(x) > L*`"。论文用 proximal MCMC 解决：

1. **把硬约束转成 characteristic function**（Eq.(28)-(30)）：`L(x) > L*` 等价于 `g(x) < τ`，`τ = -log L*`。令 `B_τ := {x | g(x) < τ}`，约束写成 indicator `χ_{B_τ}`（属于集合为 0，否则 +∞）。
2. **约束 prior**（Eq.(31)(32)）：`π_{L*}(x) = π(x) ι_{L*}(x)`，`-log π_{L*}(x) = f(x) + χ_{B_τ}(x)`。
3. **用 Moreau-Yosida 近似 characteristic function**（Eq.(34)(35)）：`χ_{B_τ}^λ(x) = ‖x - x*‖₂²/(2λ)`，`x* = prox_{χ_{B_τ}}(x) = proj_{B_τ}(x)` 是 x 到约束集 B_τ 的投影；`∇χ_{B_τ}^λ(x) = (x - x*)/λ`。
4. **constrained ULA 迭代**（f 可微，Eq.(36)）：
   `x^{(k+1)} = x^{(k)} - (δ/2)∇f(x^{(k)}) - (δ/2λ)[x^{(k)} - prox_{χ_{B_τ}}^λ(x^{(k)})] + √δ w^{(k+1)}`。
   投影项的作用：若样本已在 B_τ 内则该项消失（只做带噪梯度下降）；若跑出 B_τ 则把它推回约束集方向。f 非光滑时再对 f 也用 Moreau-Yosida（Eq.(37)(38)）。
5. **Metropolis-Hastings 校正**（Eq.(39)(40)）：以 `min{1, q(x^{(k)}|x')π_{L*}(x') / (q(x'|x^{(k)})π_{L*}(x^{(k)}))}` 接受新样本；落在 B_τ 外的候选 `π_{L*}=0` 必被拒，**保证硬约束被严格满足**。
6. **ProxSampleDraw（Algorithm 2）**：给定起点和阈值 L*，迭代 Eq.(36)/(38) + MH，直到样本满足 `L(x_new) > L*` 且步数 `k ≥ K_gap`，返回 x_new。
7. **从无约束 prior 取初始 live 样本（Algorithm 3）**：用 Eq.(41) 迭代，burn-in `K_burn`，thinning `K_gap`，得 N_live 个无约束 prior 样本。
8. **proximal nested sampling 主循环（Algorithm 4）**：在标准 nested sampling（Algorithm 1）骨架里，把"约束区域采样"替换为 Algorithm 2，每步找最低似然样本、算 `w_i = (ξ_{i-1}-ξ_{i+1})/2`、累加 `Z += L_i w_i`、随机选一个 live 样本作起点用 ProxSampleDraw 生成替换样本；末尾加 live 样本剩余贡献 `Z += Σ L(x_n) w_{i+1}/N_live`，并算后验权重 `p_i = L_i w_i / Z`。

### 3.5 显式 prior/likelihood 形式（论文 §5）

针对成像常用 sparsity-promoting prior `f(x) = μ‖Ψ†x‖₁`（Ψ 正交 sparsifying transform）与 Gaussian likelihood `g(x) = ‖y - Φx‖₂²/(2σ²)`：

- **prior 的 prox**（Eq.(43)）：`prox_f^λ(x') = x' + Ψ(soft_{λμ}(Ψ†x') - Ψ†x')`，soft 为软阈值（Eq.(44)）。
- **likelihood 的 prox**（Eq.(45)-(46)）：`prox_{χ_{B_τ}}^λ(x') = proj_{B_τ}(x')`。当 `Φ=I`（去噪）是到半径 `√(2τσ²)` 的 ℓ₂ 球投影，有闭式解（Eq.(46)）。当 `Φ≠I`（重建），投影需解约束最小化（Eq.(47)），论文给出 **ADMM（Algorithm 5）** 与 **primal-dual（Algorithm 6）** 两套求解器，数值实验默认用 primal-dual（不必解线性系统，通常更快）。
- 各种 prior（uniform / Gaussian / Laplacian）的显式迭代式见 Eq.(66)-(71)。

---

## 4. 完整复现所需数据集

论文实验分两块：高维 Gaussian 验证 + 成像 model selection。下表给出**论文实际使用的数据**与**为达 paper-like 的公开/等价候选**。

| 实验 | 论文使用数据（PDF 实证） | 公开 / 等价候选 | 备注 |
|------|--------------------------|-----------------|------|
| 高维 Gaussian 验证（§6.2） | 合成：`y = x + w`，x 为 [0,1]^d 上均匀随机，w 为标准正态；prior `f(x)=μ‖Ψ†x‖₂²`（μ=1/2, Ψ=I），likelihood `g(x)=‖y-Φx‖₂²/(2σ²)`（σ=1, Φ=I）。维度 d 从 **2 到 10^6** | 可完全自行合成，无需下载 | 有闭式 evidence（Appendix A），是验证锚点 |
| 去噪 / dictionary 选择（§6.3.1, Table 1） | **Cameraman** 灰度图（256×256），加高斯噪声使 input SNR=20 | Cameraman 是标准测试图，公开 | Ψ ∈ {I, DB2, DB8}，Φ=I |
| 重建 / 正则参数选择（§6.3.2, Table 2） | **W28** supernova remnant 射电星系图（256×256，log10 scale）；30% incomplete Fourier 测量 `Φ=MF` | W28 射电图为天文公开图像；可用任意自然/天文图替代 | μ ∈ {10^6, 10^7, 10^8}，Ψ=DB8 |
| 重建 / measurement model 选择（§6.3.3, Table 3） | **M31** HI galaxy 射电图（256×256，log10 scale）；10% Fourier 测量；misspecified mask `M_γ`，γ ∈ {0(=truth), 0.03, 0.06, 0.09, 0.12} | M31 射电图为天文公开图像 | 模拟 radio interferometry 波长未标定导致的 mask 失真 |

> 数据壁垒相对低：本篇主实验数据要么可纯合成（Gaussian 验证），要么是标准测试图（Cameraman）或天文公开图（W28/M31）。真正的壁垒在**算法实现**（proximal constrained sampler）与**算力**（高维 + 大量约束采样），而非数据获取。论文实现已开源（MATLAB；Python 版 `proxnest` 见 `github.com/astro-informatics/proxnest`），可作对照与移植参考。

---

## 5. 对照基线 (Baselines)

论文中出现或与之对照的方法（PDF 实证）：

| 类别 | 基线 |
|------|------|
| 直接证据估计对照（§6.2） | **vanilla Monte Carlo integration**（uniform prior 下对 `f·g` 直接积分，10^5 samples）——论文 Fig.1 显示其只在 d≲20 可用，高维彻底失效 |
| 证据真值锚点（§6.2） | **closed-form Gaussian evidence**（Appendix A 给出解析 log evidence），作为 ground truth |
| 隐式质量对照（§6.3） | **RMSE**（posterior mean 与 ground truth 图像之差）——论文反复强调 RMSE 实际不可用（需 ground truth），仅用于验证 Bayes-factor 选模结论是否与"已知真值时的最优选择"一致 |
| 文献中相关 evidence 估计法（§2.3 综述提及，非本文实现） | thermodynamic integration、annealed importance sampling（Neal 2001）、Chib (1995) Rao-Blackwellized estimator、truncated estimator（Brosse et al. 2017）、harmonic mean / **learnt harmonic mean**（McEwen et al. 2022）、Laplace's method、Savage-Dickey、Reversible Jump MCMC、标准 nested sampling 的各 sampler（rejection / slice / MultiNest / PolyChord / diffusive） |

> 合理的最小对照（本仓库 toy 层）：**vanilla MC integration vs nested sampling** 在低维 Gaussian 上的 evidence 误差对比；以及 estimated vs 解析 reference log evidence 的差。

---

## 6. 评价指标与论文报告结果

### 6.1 指标定义
- **log Z（marginal likelihood / Bayesian evidence 的对数）**：核心量。论文强调高维下 |log Z| 很大（problem 极高维），且 Bayes factor 可能极大，故**不建议**套用 Jeffrey's scale 之类传统量表，而**直接比较 log Z 数值大小**（§6.3.1 末）。
- **absolute log error**：估计 log Z 与解析/参考 log Z 之差（仅 Gaussian 验证可算）。
- **RMSE**：posterior mean 图像与 ground truth 的均方根误差（仅 ground truth 已知的验证场景可算，实际 model selection 不可用）。
- **标准差**：高维多次运行的 log Z 波动（§6.2 给出 10 次运行统计）。

### 6.2 论文报告的关键数值（PDF 能确认者，注明出处）

**高维 Gaussian 验证（§6.2）：**
- 低维 d<200：N_live=2×10²，N=3×10³ dead samples，thinning 10；d=200 约 1 分钟（Fig.1）。
- 中维 d 到 10^5：N_live=10³，N=10⁴；d=10^5 约 10 分钟（Fig.2）。
- 高维 **d=10^6**：ground truth `log = 2.3850×10^5`；10 次 proximal nested sampling 运行均值 `2.3851×10^5`，标准差 `0.0002×10^5`；每次约 30 分钟。**与真值高度一致**（正文 p.29）。
- vanilla MC integration 仅在 d≲20 可接受，高维失效（Fig.1）。

**去噪 / dictionary 选择（§6.3.1, Table 1，Cameraman，μ=10^5，N_live=2×10³，N=4×10⁴，thinning 10²）：**

| Prior (Ψ) | log Z | RMSE |
|-----------|-------|------|
| Ψ = I | −6.54×10⁴ ± 0.08 | 41.07 |
| Ψ = DB2 | −3.06×10⁴ ± 0.09 | 14.29 |
| Ψ = DB8 | −3.09×10⁴ ± 0.09 | 14.51 |

结论：log Z 选 **DB2 最优**（其次 DB8，I 最差），与 RMSE 排序一致。计算时间：Ψ=I 约 10 分钟，DB2/DB8 约 60 分钟。

**重建 / 正则参数选择（§6.3.2, Table 2，W28，Ψ=DB8，30% Fourier，SNR=30）：**

| μ | log Z | RMSE |
|---|-------|------|
| 10⁶ | −2.61×10⁴ ± 0.09 | 1.82 |
| 10⁷ | −5.39×10⁴ ± 0.09 | 2.81 |
| 10⁸ | −2.90×10⁵ ± 0.09 | 6.70 |

结论：log Z 选 **μ=10⁶ 最优**，与 RMSE、肉眼一致。每个问题约 150 分钟。

**重建 / measurement model 选择（§6.3.3, Table 3，M31，10% Fourier，μ=10⁸，N_live=2×10³，N=3×10⁴，thinning 10²）：**

| Likelihood (Φ) | log Z | RMSE |
|----------------|-------|------|
| Φ = M_truth·F | −4.47×10³ ± 0.08 | 3.40 |
| Φ = M_0.03·F | −4.88×10³ ± 0.08 | 7.85 |
| Φ = M_0.06·F | −5.63×10³ ± 0.08 | 12.01 |
| Φ = M_0.09·F | −9.21×10³ ± 0.07 | 15.71 |
| Φ = M_0.12·F | −1.44×10⁴ ± 0.08 | 18.08 |

结论：misspecification γ 越大 log Z 单调下降，γ=0（真值 mask）最优，与 RMSE 一致。每个问题约 150 分钟。

> 以上数值均来自 PDF Fig.1/2、Table 1/2/3 与正文，已注明出处。**禁止编造任何未在 PDF 出现的数字。** 注意 log Z 绝对值很大是因为问题极高维（256×256 ≈ 6.5×10⁴ 维），属正常现象（Table 1 脚注 6）。

### 6.3 算力与实现（论文 §6.1）
- 低维实验：MacBook，i7 CPU，16GB 内存。
- 高维实验：24 核工作站，256GB 内存。
- 实现语言：MATLAB（论文）；后续有 Python `proxnest`。

---

## 7. 本仓库当前复现实现

- **runner 文件**：`reproduce/experiments/nested_sampling_toy.py`。
- **它实际做了什么**：
  - 在 **d=2** 上构造 Gaussian likelihood `loglike(x) = -½‖x‖²`（即标准二维高斯似然），prior 为宽度 `prior_width=12` 的 **uniform prior**。
  - 跑 **standard nested sampling 骨架**：`n_live=80` 个 live 点，`n_iter=180` 步；每步取最低似然样本、用 `ξ_i=exp(-i/n_live)` 估 prior volume、`w_i = exp(-i/n_live) - exp(-(i+1)/n_live)`、累加 `Z += w_i · L_i`。
  - **约束采样用 rejection sampling**（在 uniform prior 内反复均匀采样，最多 4000 次试到一个似然高于当前阈值的候选），**不是** proximal constrained MCMC。
  - 解析参考：`analytic = 2π / prior_width²`（uniform prior 下二维高斯归一化质量），`log_ref = log(analytic)`。
  - 产图：`assets/repro/nested_sampling_evidence_trace.png`（累积 log evidence 随迭代 vs 解析虚线）。
- **本篇当前 runMetrics（来自 dashboard `reproStructured`）**：

  | 指标 | 数值 | 含义 |
  |------|------|------|
  | `estimated_log_evidence` | −5.5996 | nested sampling 估计的 log evidence |
  | `reference_log_evidence` | −3.1319 | 解析参考 log evidence |
  | `absolute_log_error` | 2.4676 | 两者绝对差（**很大**） |
  | `live_points` | 80 | N_live |
  | `iterations` | 180 | 主循环步数 |
  | `runtimeSeconds` | 0.0419 | CPU 运行时间（秒级） |

  > 注意：`absolute_log_error=2.4676` 表明这只是**机制演示**而非精确估计。误差大的原因：N_live/迭代数都很小、rejection sampling 在 uniform prior 边缘效率低、未做误差校正——dashboard 已标 `resultQuality: rough illustrative`、`warning: large evidence error; toy only`。
- **resultFiles**：`assets/repro/nested_sampling_evidence_trace.png`。
- **fidelity 说明（dashboard `notes`）**：*"Toy nested sampling on a 2D Gaussian likelihood under a uniform prior; not proximal constrained MCMC. Completed with large error; use as nested sampling mechanism demo only."*

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

**到 paper-like 的缺口清单：**

1. **核心创新缺失**：当前用 rejection sampling 做约束采样，**完全没有** proximal constrained sampler。需实现 Algorithm 2（ProxSampleDraw）、Algorithm 3（取 live 样本）、Algorithm 4（主循环），含 MYULA 迭代 Eq.(36)/(38)、Moreau-Yosida 近似的 characteristic function 投影、Metropolis-Hastings 校正。
2. **prox 算子缺失**：需实现 §5 的 `prox_f`（软阈值，Eq.(43)）与 `prox_{χ_{B_τ}}`（去噪用 ℓ₂ 球投影 Eq.(46)；重建用 ADMM Algorithm 5 或 primal-dual Algorithm 6）。
3. **维度差距**：当前 d=2；论文核心是 d 到 10^6。需先在中维（10²–10³）跑通，再向高维扩展。
4. **prior/likelihood 形式**：当前 uniform prior + 纯 Gaussian；论文用 sparsity-promoting `μ‖Ψ†x‖₁`（Ψ=I/DB2/DB8）+ Gaussian likelihood，Φ ∈ {I, MF}。需接入 wavelet 变换与 Fourier sensing。
5. **误差估计缺失**：未实现 `log Z = log(Σ L_i w_i) ± √(H/N_live)`（Eq.(27)）的熵误差估计；论文每个 log Z 都带 ±标准差。
6. **数据/任务缺失**：未做 Cameraman 去噪 dictionary 选择、W28 正则参数选择、M31 measurement model 选择三组 model-selection 实验。
7. **基线缺失**：未实现 vanilla MC integration 对照（Fig.1）、未对齐解析 Gaussian evidence（Appendix A）作为高维真值锚点。
8. **结果对照缺失**：未尝试复现 Table 1/2/3 的 log Z 量级与排序结论（DB2 优于 DB8 优于 I；μ=10⁶ 最优；γ 越大 log Z 越低）。

**到 paper-level 的额外缺口：**

9. **算力**：高维（10^6 维 × 大量约束采样）需 24 核 / 256GB 级工作站；CPU 秒级 toy 无法触及。
10. **逐表数值对齐**：需严格对齐论文每个实验的 N_live、N、thinning、SNR、噪声种子、μ、Ψ、Φ、mask 失真 γ，逐表复现 log Z（含标准差）与 RMSE，且复现 d=10^6 的 `2.3851×10^5 ± 0.0002×10^5`。
11. **求解器一致性**：primal-dual vs ADMM 的 prox 子问题需与论文一致；MH 接受率与 Langevin 步长 δ、Moreau-Yosida λ 的调参需对齐 Durmus 推荐值。

---

## 9. 运行步骤

### 9.1 当前 toy 跑法

```bash
# 安装依赖（见下）
pip install -r requirements.txt

# 运行全部复现实验（含本篇 nested_sampling_toy）
cd reproduce && python run_all.py

# 或在仓库根校验 15 篇数据 / PDF / 笔记 / 静态复现资产
node docs/scripts/validate.mjs
```

- **依赖**（来自 `reproStructured.dependencies`）：`numpy`、`matplotlib`。
- **算力**：CPU，秒级（`runtimeSeconds≈0.0419`）。
- **数据**：纯合成 2D Gaussian，**无需下载任何数据**。
- 缺依赖时 runner 写 `skipped`，**不伪造 completed**（遵守 CLAUDE.md 纪律）。

### 9.2 向 paper-like 扩展的步骤大纲

1. 先实现 §5 的两个 prox 算子（软阈值 + ℓ₂ 球投影），并写 MYULA 单步（Eq.(36)）。
2. 实现 Algorithm 2（ProxSampleDraw，含 MH 校正）与 Algorithm 3（无约束 prior live 样本），把 runner 的 rejection sampling 替换掉。
3. 在**中维 Gaussian**（d=10²–10³，Φ=I, Ψ=I, μ=1/2, σ=1）上跑 Algorithm 4，对照 Appendix A 解析 log evidence，加 Eq.(27) 熵误差，目标把 absolute_log_error 压到小量级——这是从 toy 升 partial 的第一步。
4. 接入 wavelet 变换（DB2/DB8）与 Cameraman 去噪，复现 Table 1 的 log Z 排序（DB2 > DB8 > I）。
5. 接入 Fourier sensing（Φ=MF）+ primal-dual 投影（Algorithm 6），在 W28/M31 上复现 Table 2/3 的正则参数与 measurement model 选择结论。
6. 在 dashboard 中把每个子实验的 `reproductionLevel` 独立标注，避免用单一 d=2 toy 数字代表整篇高维方法。
7. 可参考开源 `proxnest`（Python）做实现校验与移植。

---

## 10. 风险与代理说明

- **rejection sampling 的局限**：当前约束采样用"在 uniform prior 内反复均匀采样直到命中似然阈值"。这在 d=2 勉强可用，但**维度一升就指数级失效**（命中概率随 prior volume 收缩急剧下降）——这恰恰是论文要用 proximal MCMC 解决的难题。因此当前实现**无法**扩展到论文的高维场景，也**未触及**论文的真正创新。
- **toy 误差大**：`absolute_log_error=2.4676`（log 域）说明当前连低维 evidence 都未精确估计，只能定性展示"evidence 积分如何被改写成 prior-volume 一维求和"。**不得**把它当作论文级 evidence 估计精度。
- **prior/likelihood 不匹配**：当前 uniform prior + 纯 Gaussian，**没有** sparsity-promoting ℓ₁ prior、没有 wavelet/Fourier 算子、没有非光滑性——而非光滑 log-concave prior（ℓ₁/TV）正是论文相对其它 nested sampling 的核心卖点。
- **不可外推的结论**：① 不能说本仓库"复现了"proximal nested sampling；② 不能把 d=2 toy 的 evidence 估计等同于论文 d=10^6 的 `2.3851×10^5`；③ 不能宣称验证了 Table 1/2/3 的任何 model-selection 结论（DB2 优选 / μ=10⁶ 优选 / γ=0 优选）；④ paper-level 在 15 篇中仍为 0/15，本篇亦然。
- **量纲提醒**：论文 log Z 绝对值大（如 −6.54×10⁴）是高维问题的正常表现（Table 1 脚注 6），不要误读为"误差大"；这与本仓库 toy 的小量级 log Z 不在同一尺度，不可直接比较。

---

## 11. 参考：精读笔记

完整中文精读笔记见：
[`../../../xiaohao_cai_ultimate_notes/Proximal_Nested_Sampling_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Proximal_Nested_Sampling_超精读笔记_已填充.md)
