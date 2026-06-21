# 无线电干涉 UQ I：Proximal MCMC 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 12 篇 *Uncertainty Quantification for Radio Interferometric Imaging I: Proximal MCMC Methods* 的完整复现流程规范。

---

## 1. 论文身份与第一作者核验

| 项 | 信息 |
|------|------|
| 标题 (EN) | Uncertainty quantification for radio interferometric imaging: I. proximal MCMC methods |
| 标题 (CN) | 无线电干涉成像不确定性量化 I：近端 MCMC 方法 |
| 作者顺序 | **Xiaohao Cai**, Marcelo Pereyra, Jason D. McEwen |
| 第一作者核验 | 是。PDF 首页标题下方作者行以 `Xiaohao Cai^{1*}` 开头，上标 1 指向 Mullard Space Science Laboratory (MSSL), University College London (UCL)；星号注脚邮箱 `x.cai@ucl.ac.uk (XC)`。Xiaohao Cai 为唯一第一作者。 |
| 年份 | 正式发表 2018（MNRAS Vol.480, Issue 3, pp.4154-4169；DOI 10.1093/mnras/sty2004）；arXiv 预印本 2017（arXiv:1711.04818v2，PDF 页眉 "Preprint 12 September 2018"） |
| 期刊 | Monthly Notices of the Royal Astronomical Society (MNRAS) |
| 主题 (theme) | ri-uq（无线电干涉成像的 Bayesian 不确定性量化） |
| PDF 路径 | `docs/00_papers_first_author_xiaohao_cai_deduped/无线电干涉不确定性I Radio Interferometric I.pdf` |
| 关键词 | techniques: image processing / interferometric; methods: data analysis / numerical / statistical |

核验依据：PDF 第 1 页标题、作者行上标与注脚邮箱（`x.cai@ucl.ac.uk`），以及页边竖排 `arXiv:1711.04818v2 [astro-ph.IM] 11 Sep 2018`。本文有一篇配套论文（companion article，Cai et al. 2017b，对应本项目 RI UQ II，priority 13），讨论如何把 UQ 用 MAP 近似扩展到 big-data；二者引用关系在第 8、11 节说明。

---

## 2. 复现目标与诚实分级

本项目对复现真实性采用四级口径，纪律是**绝不把 synthetic/proxy 结果夸大为论文级**：

| 等级 | 含义 | 本篇状态 |
|------|------|----------|
| toy | 用 synthetic 数据 + proxy 算子演示思路，量级/可视化对得上即可 | **本仓库当前等级 = `toy`** |
| partial | 用真实数据或正确算子复现局部流程，但不追求完整表格对照 | 未达到 |
| paper-like | 用论文同源/等价数据 + 正确求解器，复现主要定性结论与趋势 | 未达到 |
| paper-level | 严格复现论文表格/图的数值，方法与数据均一致 | **0/15，本篇亦为 0** |

- 本仓库 `reproductionLevel = toy`，`reproductionTruthLevel = toy-completed`。
- 当前实现（`map_uq_toy.py`）在一个 32×32 的 Fourier 欠采样逆问题上，用**梯度下降 + Gaussian smoothing 作为 MAP/proxy**，再用一个**带 Gaussian 抖动 + 平滑的随机游走**冒充 proximal-MCMC 采样器，从样本经验分位数算 interval map。它**不含**：真实 RI measurement operator Φ（NUFFT/visibility 算子）、Daubechies 小波字典 Ψ、$\ell_1$ analysis/synthesis 后验、Moreau-Yosida envelope、真正的 MYULA / Px-MALA（无 Metropolis-Hastings 校正、无 Langevin 噪声尺度 $\sqrt{2\delta}$ 的正确实现）、HPD isocontour $\gamma_\alpha$ 计算与 hypothesis testing。因此它只能算"思路演示 toy"，**不能**作为对论文 MYULA/Px-MALA 算法的复现。
- 纪律强调：本文档第 6 节引用的论文数值（Table 1 的 CPU 分钟数、Table 2/3 的 hypothesis test 值等）仅供对照，**当前实现不复现这些数值**。`notes` 字段已如实声明 "Toy runtime comparison is not comparable to the paper's large-scale 10^5 speedup claim"。任何把 toy 的 `mcmc_runtime_seconds=0.0041` 或 `map_psnr=18.7` 当作论文级证据的陈述都是错误的。

---

## 3. 算法完整流程

论文方法是把 **proximal calculus**（Moreau-Yosida envelope + proximity operator）接入高维 Langevin MCMC，从而让 MCMC 能采非光滑稀疏后验（$\ell_1$ priors），并提供三类 UQ 产品。忠于 PDF §2–§5（含 Algorithm 1、2）的 step-by-step pipeline 如下。

**记号与 RI 模型（PDF §2）**

1. **连续 RI 方程（式 1）**：$y(u)=\int A(l)\,x(l)\,e^{-2\pi i u\cdot l}\,d^2 l$，$x$ 为天空亮度，$A(l)$ 主波束，$u$ 基线坐标。论文不考虑 wide-field / DDE，narrow field 下 visibility 是天空的不完整 Fourier 测量。
2. **离散线性逆问题（式 2）**：$\boxed{y=\Phi x+n}$，其中 $x\in\mathbb{R}^N$（$N$ 像素天图），$y\in\mathbb{C}^M$（$M$ 个 visibility），$\Phi\in\mathbb{C}^{M\times N}$ 线性测量算子，$n\in\mathbb{C}^M$ 假设 i.i.d. 复 Gaussian 噪声。问题 ill-posed、ill-conditioned、高维（通常 $M\ll N$）。
3. **稀疏表示（式 3）**：$x=\Psi a=\sum_i \Psi_i a_i$，$\Psi\in\mathbb{C}^{N\times L}$ 字典（小波基或 overcomplete frame），$a$ 为合成系数；$\|a\|_0=K\ll N$ 即 $K$-稀疏。

**Bayesian 后验（PDF §2.3, §2.4）**

4. **似然（式 4）**：$p(y|x)\propto\exp(-\|y-\Phi x\|_2^2/2\sigma^2)$，$\sigma$ 噪声标准差。
5. **稀疏先验**：analysis 先验（式 5）$p(x)\propto\exp(-\mu\|\Psi^\dagger x\|_1)$；synthesis 先验（式 6）$p(a)\propto\exp(-\mu\|a\|_1)$。$\Psi^\dagger$ 是 $\Psi$ 的伴随，$\mu>0$ 为正则化参数。两者在 $\Psi$ 正交时等价，redundant dictionary 时性质不同。
6. **后验（式 9、10）**：
   $$p(x|y)\propto\exp\{-(\mu\|\Psi^\dagger x\|_1+\|y-\Phi x\|_2^2/2\sigma^2)\}$$ （analysis）
   $$p(a|y)\propto\exp\{-(\mu\|a\|_1+\|y-\Phi\Psi a\|_2^2/2\sigma^2)\}$$ （synthesis）
   写成通式 $\pi(x)\propto\exp\{-f(x)-g(x)\}$：$f$ 是非光滑 $\ell_1$ 先验项，$g$ 是光滑数据保真项。
7. **MAP（式 11、12）**：$\hat x_{\text{map}}=\arg\min_x\{\mu\|\Psi^\dagger x\|_1+\|y-\Phi x\|_2^2/2\sigma^2\}$，凸优化，可在高维高效求解；MAP 等价 $\ell_1$-regularised least-squares（CS 视角）。

**Proximal calculus 预备（PDF §3.1）**

8. **Moreau-Yosida envelope（式 17）**：对凸 l.s.c. $h$，$h^\lambda(z)=\min_u\{h(u)+\|u-z\|^2/2\lambda\}$；$h^\lambda\in C^1$，梯度 $\nabla h^\lambda(z)=(z-\mathrm{prox}_h^\lambda(z))/\lambda$（式 18）。
9. **Proximity operator（式 19）**：$\mathrm{prox}_h^\lambda(z)=\arg\min_u\{h(u)+\|u-z\|^2/2\lambda\}$。对 $\mu\|\cdot\|_1$ 即软阈值 $\mathrm{soft}_{\mu\lambda}$（式 27、28）；对凸集指示函数即投影 $\mathcal{P}_C$（式 20）。

**Langevin MCMC（PDF §3.2–§3.4）**

10. **Langevin diffusion（式 21）**：$d\mathcal{L}(t)=\tfrac12\nabla\log\pi(\mathcal{L}(t))\,dt+d\mathcal{W}(t)$，以 $\pi$ 为不变分布；离散得 ULA（式 22）$l^{(m+1)}=l^{(m)}+\tfrac{\delta}{2}\nabla\log\pi+\sqrt{\delta}\,w^{(m+1)}$。要求 $\log\pi$ 光滑 Lipschitz——这正是 $\ell_1$ 非光滑先验无法直接用的原因。
11. **MYULA（式 24）**：把非光滑 $f$ 换成 Moreau-Yosida envelope $f^\lambda$，得
    $$l^{(m+1)}=\Big(1-\tfrac{\delta}{\lambda}\Big)l^{(m)}+\tfrac{\delta}{\lambda}\mathrm{prox}_f^\lambda(l^{(m)})-\delta\nabla g(l^{(m)})+\sqrt{2\delta}\,w^{(m)}.$$
    取 $\lambda=2/\beta_{\text{Lip}}$、$\delta\in[1/5\beta_{\text{Lip}},\,1/2\beta_{\text{Lip}}]$（Durmus et al. 2016）。**无 MH 步**，bias 可任意小，scale 到高维。
12. **Px-MALA（式 25、26、43、44）**：用一次 MYULA 迭代当 proposal（式 43 $x^{(m+1)}=\mathrm{prox}_f^{\delta/2}(x^{(m)})+\sqrt\delta\,w^{(m)}$），再加 Metropolis-Hastings 接受概率 $\rho=\min\{1,\,q(l^{(m)}|l^*)\pi(l^*)/q(l^*|l^{(m)})\pi(l^{(m)})\}$（式 25）。**有 MH 校正**，渐近无偏，以 $\pi$ 为精确不变分布，但每步更贵、链相关性更高（估计方差更大）。论文调 $\delta$ 使接受率约 0.5。
13. **RI 专用 prox / grad（PDF §4）**：analysis 的 $\mathrm{prox}_f^\lambda(x)=x+\Psi(\mathrm{soft}_{\mu\lambda}(\Psi^\dagger x)-\Psi^\dagger x)$（式 29，闭式，假设 $\Psi^\dagger\Psi=I$）；$\nabla g(x)=\Phi^\dagger(\Phi x-y)/\sigma^2$（式 30）。synthesis 类似（式 37、38）。算法 1（MYULA）、算法 2（Px-MALA）给出含 analysis/synthesis 分支与 thinning（式 45：$m>K_{\text{burn}}$ 且 $\mathrm{mod}(m-K_{\text{burn}},K_{\text{gap}})=0$ 才存样本）的完整伪代码。

**三类 UQ 产品（PDF §5，Figure 1）**

14. **Pixel-wise credible intervals（式 46–49）**：对每像素 $x_i$ 由样本求分位数 $\hat\xi_{i-}=\mathrm{quantile}(\{x_i^{(j)}\},\alpha/2)$、$\hat\xi_{i+}=\mathrm{quantile}(\cdot,1-\alpha/2)$，区间长度 $\xi_{i+}-\xi_{i-}$ 给每像素 error bar；synthesis 时由 $(\Psi a)_i$ 投影得到。
15. **HPD credible region（式 50–52）**：$C_\alpha=\{x:f(x)+g(x)\le\gamma_\alpha\}$，$\gamma_\alpha$ 为 log-posterior 的 isocontour，由样本 $\hat\gamma_\alpha=\mathrm{quantile}(\{(\hat f+\hat g)(x^{(j)})\},1-\alpha)$ 估计；decision-theoretically minimum-volume。
16. **Hypothesis testing（PDF §5.3，式 53）**：knock-out test。先用 segmentation-inpainting（式 53，基于 Cai et al. 2008 的 recursive wavelet filter）构造去掉某结构的 surrogate $x^{*,\text{sgt}}$；若 $x^{*,\text{sgt}}\notin C_\alpha$（即 $(\bar f+\bar g)(x^{*,\text{sgt}})>\hat\gamma_\alpha$）则该结构受数据强支持（physical），否则证据不足（可能 artefact）。对小结构推荐用 posterior median 构造点估计。

---

## 4. 完整复现所需数据集

论文实验全部用 **simulated RI observations**（PDF §6.1）：真值天图来自四张公开射电天文图像，visibility 由人工 $u$-$v$ 覆盖 + 噪声合成。因此 paper-like 复现**无私有 visibility 数据障碍**，门槛在 RI 算子与采样器，而非数据获取。

| 数据（真值天图） | 尺寸 | 论文用途 | 来源 / 等价候选 |
|------|------|----------|------------------|
| **HI region of M31 galaxy** | 256×256 | 主实验（reconstruction + 全套 UQ，含 analysis/synthesis 对照，Fig. 3） | 经典 RI 测试图，astropy/radio imaging 教程与多篇 sparse RI 论文（McEwen & Wiaux 2011, Carrillo et al. 2012）共享同源 M31 图 |
| **Cygnus A radio galaxy** | 256×512 | reconstruction + UQ（Fig. 4、5、6、7） | VLA Cygnus A 经典图，公开射电天文图库 |
| **W28 supernova remnant** | 256×256 | reconstruction + UQ | 公开 SNR 射电图 |
| **3C288 radio galaxy** | 256×256 | reconstruction + UQ + 含一个 artefact 结构（hypothesis test 的反例，test area 2） | 公开射电星系图 |

**visibility 合成流程（PDF §6.1，可复现）**：
1. **$u$-$v$ 覆盖**：用 variable-density sampling profile（Puy et al. 2011）在半 Fourier 平面随机生成，取每张图 **10% 的 Fourier 系数**（Figure 2 给出 256×256 的覆盖示例）。
2. **加噪**：零均值复 Gaussian，$\sigma=\|f\|_\infty\,10^{-\mathrm{SNR}/20}$，$\|f\|_\infty$ 为 $f$ 分量最大绝对值，**SNR 固定 30 dB**。
3. **字典 Ψ**：analysis 与 synthesis 模型均用 **Daubechies 8 小波**（MATLAB `wavedec2`）；论文指出因此 analysis 与 synthesis 结果差异不大，overcomplete 基（如多尺度并）留作 better reconstruction 的方向。

paper-level 复现还需匹配：$\ell_1$ 正则参数 $\mu=10^4$（visual cross-validation 固定）、采样设置 $10^3$ samples、$10^5$ burn-in、thinning $10^3$（即每条链跑 $1.1\times10^6$ 次迭代产 $10^3$ 样本）、credible interval level $\alpha=0.05$（95%）、hypothesis test $\alpha=0.01$（99%），$\gamma_\alpha$ 在 $\alpha\in[0.01,0.99]$ 上扫。硬件：24-core x86_64 工作站 + 256 GB 内存，MATLAB R2015b。

---

## 5. 对照基线 (Baselines)

论文的对照是**方法内部互证 + 与经典 RI 重建的定性对比**，而非外部 UQ 方法的定量擂台（因为正是"现有方法都不给 UQ"才是本文动机）：

- **MYULA vs Px-MALA**：核心内部对照。Px-MALA 渐近无偏（MH 校正），MYULA 更快（约为 Px-MALA 一半 CPU 时间，Table 1）但有可控 bias、估计方差更小。论文反复用二者结果一致来佐证可信度。
- **Analysis model vs Synthesis model**：两种 formulation 互证；因用正交 Daubechies 8，结果近似一致（Fig. 3 e/f vs c/d）。
- **Dirty image（inverse Fourier transform of $y$）**：作为重建质量下界对照（Fig. 3b、Fig. 4 第 2 列），明显劣于 MYULA/Px-MALA 的 posterior mean。
- **背景方法（仅文献定位，未直接定量对比）**：CLEAN（式 13）及其多尺度变体（MS-CLEAN、ASP-CLEAN）、MEM（式中 entropic prior）、constrained CS（式 14、15）、Gibbs sampling（Sutter et al. 2014，唯一已有的 RI MCMC proof-of-concept，但限 Gaussian prior 且仅 idealised telescope）、RESOLVE（Junklewitz et al.，给近似 posterior covariance 但计算昂贵）。论文强调这些**都不能在 sparse prior 下给 scalable UQ**——这是本文的定位空白。

若要更强 baseline 体系，可补：与 RESOLVE 的 UQ 对照、与 II 篇 MAP-based UQ 的速度/精度权衡（这正是 companion paper 的内容）。

---

## 6. 评价指标与论文报告结果

**论文的"指标"以可视化 + 计算时间表 + hypothesis test 表为主**，重建质量主要用定性视觉对照（dirty image vs posterior mean），UQ 产品用 interval-length map、HPD isocontour 曲线、hypothesis test 的真/伪判定。以下数值已从 PDF 表格核实，引用注明表号；**不能确认的数字一律不编造**。

- **CPU 时间（Table 1，单位：分钟）**：核心可对照硬数值。
  - M31（256×256）：MYULA analysis **618** / synthesis **581**；Px-MALA analysis **1307** / synthesis **944**。
  - Cygnus A（256×512）：MYULA **1056 / 942**；Px-MALA **2274 / 1762**。
  - W28（256×256）：MYULA **646 / 598**；Px-MALA **1122 / 879**。
  - 3C288（256×256）：MYULA **607 / 538**；Px-MALA **1144 / 881**。
  - 趋势：**MYULA 比 Px-MALA 经济，约需其一半 CPU 时间**（因 Px-MALA 多一个 MH accept-reject step）。
- **重建（PDF §6.2）**：MYULA、Px-MALA 的 posterior mean 都明显优于 dirty image，且彼此、analysis/synthesis 之间高度一致；MYULA 重建质量略优（superior convergence，固定样本数下更好），但 Px-MALA 渐近无偏。**论文未给 RI 重建的 PSNR/SNR 数值表**——当前 toy 的 `map_psnr=18.7` 与论文无任何对应。
- **Pixel-wise credible intervals（PDF §6.3，Figure 5）**：MYULA 给出更宽、更平滑的 interval（low variance 但 overestimate uncertainty + 有 bias）；Px-MALA interval 更小但更 noisy（bias-variance tradeoff，由 MH 校正引起）。物体边界附近 interval 更宽（高频不确定）。
- **HPD isocontour $\gamma_\alpha$（PDF §6.4，Figure 6）**：$\gamma_\alpha$ 对 $\alpha\in[0.01,0.99]$ 单调上升曲线；MYULA 与 Px-MALA、analysis 与 synthesis 高度一致，量级 $\sim10^6$（图纵轴 ×10^6，如 M31 约 2.34×10^6）。
- **Hypothesis testing（PDF §5.3, Table 2/3，单位 ×10^6，$\alpha=0.01$）**：测试 Figure 7 的五个结构。Table 2（用 posterior mean 构造 surrogate）/ Table 3（用 posterior median）。判定规则：surrogate 目标函数值 $>$ isocontour $\hat\gamma_{0.01}$ 则 $\notin C_\alpha$ → 结构 physical（✓）。
  - 物理结构（M31、W28、3C288 test 1）：surrogate 值 $<\gamma_\alpha$，正确判为 physical（✓）。例 M31：MYULA 2.20 vs $\gamma$ 2.34（✓）。
  - 人造 artefact（3C288 test area 2）：MYULA 1.752 vs $\gamma$ 2.032，正确判为 artefact（✗ = not physical）。
  - 边界小结构（Cygnus A，仅几个亮像素）：用 mean 时判 ✗（Table 2），用 median 时 MYULA 能判 ✓（Table 3，1.597 vs 1.586）——论文据此**推荐用 posterior median 测小结构**（median 更靠近 $C_\alpha$ 边界）。
- **结论（PDF §6、§7）**：MYULA 与 Px-MALA、analysis 与 synthesis 给出 convincing & consistent 结果；三个大物理结构被正确分类为 physical，重建 artefact 被正确高亮为证据不足。**禁止把上述论文数值当作本仓库复现产物。**

---

## 7. 本仓库当前复现实现

- **runner 文件**：`reproduce/experiments/map_uq_toy.py`（`experiment_id = map_uq_toy`，被 priority 11/12/13 三篇共享：high-dimensional-uq、ri-uq-i、ri-uq-ii）。`reproductionLevel = toy`，但**已从 Gaussian 平滑 + 随机游走代理升级为论文优化/UQ 机制的真实可运行实现**。
- **它实际做了什么（真实算法）**：
  1. **真值图**：`skimage` 的 64×64 Shepp-Logan phantom（标准测试图，**非** M31/Cygnus A 等论文 RI 真值天图）。
  2. **前向算子 $A$**：low-freq-biased variable-density Fourier 欠采样掩模（`prob=exp(-(r/0.18)²)`），采样率 ≈ **9.8%**（对齐论文 10% Fourier 覆盖思路），$A(x)=\text{mask}\cdot\text{FFT}(x)/\sqrt N$、$A^\dagger$ 为其共轭，构成 subsampled-unitary（$\|A\|\le1$）。**作为 RI measurement 的合理 proxy**，但仍无真正 NUFFT/visibility 几何。
  3. **噪声**：i.i.d. 复 Gaussian，按论文约定 **SNR=30 dB** 设 $\sigma=\|f\|\cdot10^{-30/20}$。
  4. **真实 $\ell_1$-wavelet MAP（forward-backward / FISTA）**：解 $\hat x=\arg\min_x\mu\|\Psi^\dagger x\|_1+\|y-Ax\|_2^2/2\sigma^2$，$\Psi=$ **Daubechies-8 正交小波（pywt）**，软阈值 prox 闭式（式 29，$\Psi^\dagger\Psi=I$），梯度步 $\nabla g=A^\dagger(Ax-y)/\sigma^2$（式 30），FISTA 动量，250 iters，初值 dirty image $A^\dagger y$，$\mu=3\sigma$。
  5. **真实 HPD 阈值**：$\gamma'_\alpha=f(\hat x)+g(\hat x)+\sqrt{16\log(3/\alpha)}\sqrt N+N$（concentration inequality，$\alpha=0.01$）。
  6. **真实 local credible interval（superpixel 二分搜索）**：8×8 superpixel，对每区域固定区外为 MAP、区内统一设常数 $\xi$，二分搜索使 $f+g\le\gamma'_\alpha$ 的最小/最大 $\xi$，输出区间宽度图。
- **当前 runMetrics（真实算法，Shepp-Logan 测试图）**：`map_psnr = 22.2878`、`map_snr = 9.1586`、`dirty_snr = 7.4821`、`snr_gain_over_dirty_db = 1.6766`（**MAP 优于 dirty 基线 +1.68 dB**）、`sampling_rate = 0.0979`、`noise_sigma = 0.019635`、`map_runtime_seconds = 0.0588`、`lci_runtime_seconds = 0.3156`、`gamma_alpha_hpd = 6180.6593`、`mean_interval_length = 0.3709`；`status = completed`，`runtime_seconds ≈ 0.58`。
- **resultFiles**：`assets/repro/map_uq_reconstruction_uncertainty.png`（四联图：truth / dirty $A^\dagger y$ / $\ell_1$-wavelet MAP / LCI width）。
- **重要说明**：本 runner 实现的是论文的**优化 + MAP-UQ 机制**（与配套 II 篇路线一致），**仍不含**论文 I 的核心采样器 MYULA/Px-MALA（无 Langevin 漂移、$\sqrt{2\delta}$ 噪声、MH 校正），也无 hypothesis testing。`notes` 与 `fidelityWarning` 字段已如实标注："implements the optimisation/UQ machinery but NOT the paper's proximal-MCMC samplers (MYULA/Px-MALA), nor a true RI measurement operator, nor M31/Cygnus A/W28/3C288 data. Runtime is not comparable to the paper's large-scale results."

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

按缺口类型清单化（✅ = 升级后已实现，⛔ = 仍缺）：

1. ⛔ **测量算子（仍缺真实 RI 算子）**：已用 low-freq-biased variable-density mask·FFT（采样率 ≈10%，比旧 toy 的纯随机选频更接近 $u$-$v$ 覆盖思路），但仍**无** visibility 几何 / degridding / NUFFT（Puy et al. 2011 的精确 profile 也未实现）。
2. ✅ **稀疏先验 / 字典（已实现）**：已接入 **Daubechies 8 小波 Ψ** + $\ell_1$ analysis 后验 + **闭式软阈值 prox**（式 29）。这取代了旧 Gaussian smoothing 代理，是论文相对 Gaussian-prior 方法卖点的真实体现。**仍缺**：synthesis 模型与 over-complete frame 对照。
3. ⛔ **采样器（仍缺，核心）**：仍**无** Moreau-Yosida envelope、prox-gradient Langevin、MYULA 的 $(1-\delta/\lambda)$ 收缩 + $\sqrt{2\delta}$ 噪声、Px-MALA 的 MH accept-reject。当前走的是**优化（MAP）路线**（与配套 II 篇一致），不是论文 I 的采样路线。本篇要到 paper-like 必须补真正的 MYULA/Px-MALA。
4. **UQ 产品（部分已实现）**：✅ 真实 HPD 阈值 $\gamma'_\alpha=f+g+\sqrt{16\log(3/\alpha)}\sqrt N+N$（concentration-inequality 版，对应 MAP 路线）；✅ 真实 local credible interval（superpixel 二分搜索）。⛔ **仍缺**：基于后验样本的式 50–52 HPD isocontour（$\hat\gamma_\alpha$ 由样本分位数估）、式 46–49 的 sample-based pixel interval、§5.3 的 segmentation-inpainting knock-out hypothesis test（式 53）。
5. ⛔ **数据（仍缺）**：用 64×64 Shepp-Logan 标准测试图，未接入 M31 / Cygnus A / W28 / 3C288 真值天图，分辨率（64² vs 256²/256×512）与结构复杂度不可比。
6. ⛔ **参数 / 设置（仍缺）**：已对齐 SNR=30 dB；但 $\mu=3\sigma$ 非论文 $\mu=10^4$，且无 $10^3$ samples / $10^5$ burn-in / $10^3$ thinning（因走优化路线无采样），credible/test level 仅用了 $\alpha=0.01$。
7. ⛔ **诊断与表格对照（仍缺）**：无 MCMC 收敛诊断（trace、acceptance rate≈0.5、autocorrelation/ESS），无法产出 Table 1 的 MYULA/Px-MALA CPU 时间对照、Figure 6 的 $\gamma_\alpha$ 曲线、Table 2/3 的 hypothesis test 真伪判定。
8. ⛔ **量级不可比**：论文单次实验 CPU 时间为 $10^2$–$10^3$ 分钟级（24-core / 256 GB），本 runner 为 CPU 亚秒级；`fidelityWarning` 已声明 runtime 与论文大规模结果无关——**不得**用于任何速度/加速度结论。

**达到 paper-like 的最小充分条件**：实现真正的 prox-Langevin 采样器（MYULA + 闭式软阈值 prox + $\nabla g$）→ 接入一类真实天图（建议先 M31 256×256）+ Daubechies 8 字典 + variable-density 10% Fourier 覆盖 + SNR=30 dB → 复现"MYULA 与 Px-MALA 一致、posterior mean 优于 dirty image、边界处 interval 更宽、artefact 被 hypothesis test 判为证据不足"等定性趋势，并给出 MYULA 约为 Px-MALA 一半 CPU 时间的量级。paper-level 还需严格匹配 $\mu=10^4$、$10^3$ samples / $10^5$ burn-in / $10^3$ thinning、$\alpha$ 设置，并复现 Table 1 的分钟数、Figure 6 的 $\gamma_\alpha\sim10^6$ 曲线、Table 2/3 的逐结构判定（含 Cygnus A 小结构 median 才判 ✓ 这一细节）。

---

## 9. 运行步骤

**当前 toy 跑法**

```bash
# 安装依赖（见 reproStructured.dependencies）
pip install -r requirements.txt   # 关键：numpy, scipy, scikit-image, matplotlib

# 运行全部复现实验（含本篇 map_uq_toy，priority 11/12/13 共享）
cd reproduce && python run_all.py
```

- 依赖缺失时 runner 写入 `status=skipped`（见 `require_modules`），不会伪造 completed（符合项目纪律）。
- 产物：`assets/repro/map_uq_reconstruction_uncertainty.png` 与结果 JSON 中的 `runMetrics`。

**向 paper-like 扩展的步骤大纲（不在当前 toy 范围内）**

1. **算子**：实现/接入 RI measurement Φ（最简可用 masked FFT + variable-density profile 近似 visibility 覆盖；进一步用 NUFFT），与伴随 $\Phi^\dagger$，确保 $\nabla g(x)=\Phi^\dagger(\Phi x-y)/\sigma^2$（式 30）正确。
2. **先验**：用 `pywt` 实现 Daubechies 8 的 $\Psi^\dagger,\Psi$，写软阈值 prox（式 29），设 $\mu=10^4$。
3. **采样器**：实现 MYULA（式 24，$\lambda=2/\beta_{\text{Lip}}$、$\delta\in[1/5\beta_{\text{Lip}},1/2\beta_{\text{Lip}}]$）；再加 MH 步得 Px-MALA（式 25、43），调 $\delta$ 使接受率≈0.5。$\beta_{\text{Lip}}$ 取 $\|\Phi^\dagger\Phi\|/\sigma^2$。
4. **数据**：下载 M31（先做单图打通），合成 10% Fourier 覆盖 + 30 dB 噪声。
5. **UQ**：从样本算 pixel-wise interval（式 47）、HPD $\gamma_\alpha$（式 52）、knock-out hypothesis test（式 53 的 inpainting + §5.3 判定）。
6. **诊断与对照**：记录 acceptance rate、ESS、MYULA vs Px-MALA 的 CPU 时间比，与 Table 1 趋势比对。
7. 全程在图注 / JSON 中标注真实等级（partial / paper-like），严禁夸大为 paper-level。

---

## 10. 风险与代理说明

**升级后已消除的代理**（不再适用）：~~Gaussian smoothing 代替 $\ell_1$ 软阈值 prox~~、~~占位量 `gamma_alpha_toy` 代替 HPD 阈值~~、~~随机游走分位差代替 credible interval~~。现在 $\ell_1$-wavelet（db8）MAP 用真实 forward-backward/FISTA + 闭式软阈值 prox 求解，HPD 阈值与 local credible interval 均为论文公式的真实实现，且 MAP 比 dirty 基线 SNR 高 +1.68 dB。

**仍存在的局限 / 代理（诚实标注）**：
- **masked FFT ≠ RI measurement operator**：已改为 low-freq-biased variable-density mask（采样率 ≈10%，比纯随机更接近 $u$-$v$ 覆盖思路），但仍无 visibility 几何、无 degridding/NUFFT；不能据此推断对 SKA/VLA 等真实阵列的适用性。
- **走的是优化（MAP）路线，不是论文 I 的采样路线**：本 runner **未**实现 MYULA/Px-MALA（无 Langevin 漂移、$\sqrt{2\delta}$ 噪声、MH 校正）。论文 I 的核心贡献是"在非光滑稀疏后验上做 proximal MCMC 采样"，本实现只覆盖其 MAP/优化与 HPD-UQ 机制（与配套 II 篇路线一致），**不能**代表论文 I 的采样器结论与 bias-variance tradeoff。
- **HPD/interval 语义对齐 MAP 近似版**：`gamma_alpha_hpd`、`mean_interval_length` 是基于 MAP 的 concentration-inequality HPD 阈值与 superpixel 二分区间，**不是**论文 I 式 50–52 由后验样本估计的 $\hat\gamma_\alpha$（$\sim10^6$ 量级）与式 47 的 sample-based interval；二者数值不可直接对照。
- **非论文数据**：用 64×64 Shepp-Logan 测试图，**不是** M31/Cygnus A/W28/3C288；`map_snr=9.16`、`map_psnr=22.29` 非论文报告值。
- **无 hypothesis testing**：§5.3 的 segmentation-inpainting knock-out test（式 53）完全未实现。
- **runtime 不可比**：本 runner 为 CPU 亚秒级，论文为 $10^2$–$10^3$ 分钟级（24-core/256 GB）；**不得**用本 runner 的 runtime 佐证 "MYULA 约 Px-MALA 一半 CPU 时间" 或任何大规模加速结论。
- **可诚实外推的结论**：本实现可声称"用论文真实的 $\ell_1$-wavelet MAP + HPD 局部可信区间方法，在标准 Fourier 欠采样测试图上跑通并优于 dirty 基线"，但**不得**声称复现了论文 I 的 MYULA/Px-MALA 采样结果、Table 1 时间或 hypothesis test 判定；paper-level 仍为 **0/15**。

---

## 11. 参考：精读笔记

- 本篇精读笔记：[`../../../xiaohao_cai_ultimate_notes/Radio_Interferometric_Imaging_I_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Radio_Interferometric_Imaging_I_超精读笔记_已填充.md)
- 配套后续（MAP-based UQ，scalable）：RI UQ II（priority 13，Cai et al. 2017b，companion article）。
- 关联 UQ 谱系：High-dimensional UQ（priority 11）、Proximal Nested Sampling（priority 14）——见笔记"论文关系"小节。
