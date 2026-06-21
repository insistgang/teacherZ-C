# 在线无线电干涉成像 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 14 篇 *Online Radio Interferometric Imaging: Assimilating and Discarding Visibilities on Arrival* 的完整复现流程规范。

---

## 1. 论文身份与第一作者核验

| 项 | 信息 |
|------|------|
| 标题 (EN) | Online Radio Interferometric Imaging: Assimilating and Discarding Visibilities on Arrival |
| 标题 (CN) | 在线无线电干涉成像：到达时同化与丢弃可见度数据 |
| 作者顺序 | **Xiaohao Cai**, Luke Pratley, Jason D. McEwen |
| 第一作者核验 | 是。PDF 首页标题下方作者行以 `Xiaohao Cai^{1*}` 开头（星号脚注为通讯邮箱 `x.cai@ucl.ac.uk`），上标 1 指向唯一机构 *Mullard Space Science Laboratory (MSSL), University College London (UCL), Surrey RH5 6NT, UK*。Xiaohao Cai 为唯一第一作者。 |
| 年份 | PDF 本体为 arXiv:1712.04462v1，"Preprint 14 December 2017"，© 2017 The Authors（MNRAS 风格预印本）；正式刊出版本为 MNRAS 2019。dashboard 元数据标 `year=2019`（按刊出年），与 PDF 上的 2017（按 arXiv 投稿年）并不矛盾，但口径需统一，见第 8/factualIssues 说明。 |
| 主题 (theme) | `ri-uq`（big-data RI imaging：把 inverse-problem 求解器嵌入数据获取流程，online streaming reconstruction） |
| PDF 路径 | `docs/00_papers_first_author_xiaohao_cai_deduped/在线无线电干涉成像 Online Radio Imaging.pdf` |
| 关键词 | techniques: image processing — techniques: interferometric — methods: data analysis — methods: numerical |

核验依据：PDF 第 1 页标题、作者行 `Xiaohao Cai^{1*}, Luke Pratley^{1*} and Jason D. McEwen^{1*}`、机构脚注，以及页边竖排标识 `arXiv:1712.04462v1 [astro-ph.IM] 12 Dec 2017`。

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
- 当前实现（`reproduce/experiments/online_ri_toy.py`）用 64×64 synthetic 多结构图像（扩展圆盘 + 点源）+ variable-density Fourier 掩码（masked FFT 作为 sensing operator Φ_k=M_k F），并实现了**真实的 online forward-backward（Algorithm 2, analysis form）**：含 Daubechies-8 正交小波 Ψ（pywt）的软阈值 prox、按 B=8 块累加的部分梯度 Σ_{k≤b} Φ_k^H(Φ_k x − y_k)、丢弃机制（只保留累计 mask/dirty 摘要与当前估计），并与 standard offline FB 基线（每步全量 visibilities）对照。因此它**真正复现**了论文"online ≈ offline 重建质量、峰值存储 η_s≈1/B"这一核心机制——但仍是 toy 等级，缺口见下方算子/数据说明。
- 纪律强调：本文档第 6 节引用的论文数值（如 M31 SNR ≈ 14.2946 dB、Table 1 相对差 ~10⁻⁶）仍仅供对照；当前 toy 用 synthetic 圆盘 + masked-FFT，**不复现这些具体数值**（toy 的 SNR≈24.6 dB、rel.diff≈3e-3 是本 synthetic 设定下的量级，与论文 M31 的 14.29 dB / 10⁻⁶ 不可直接比较）。把本 toy 解读为"论文级 online RI 已复现"仍是错误陈述。

---

## 3. 算法完整流程

论文方法是一个**通用 online sparse regularisation 框架**，并以 forward-backward (FB) splitting 为具体实例。以下 step-by-step pipeline 忠于 PDF §2–§4（含 Algorithm 1/2/3 与 Figure 1 流程图）。

**记号与测量模型 (PDF §2.1)**
- 连续测量方程（式 1）：$y(u)=\int A(l)x(l)\,e^{-2\pi i u\cdot l}\,\mathrm d^2 l$，$A(l)$ 为 primary beam，$u=(u,v)$ 为 baseline 向量。
- 离散逆问题（式 3）：$y=\Phi x+n$，其中 $x\in\mathbb R^N$ 为 sky brightness，$y\in\mathbb C^M$ 为 visibilities，$\Phi\in\mathbb C^{M\times N}$ 为 measurement operator，$n$ 为 i.i.d. 高斯噪声。big-data 场景下 $M\gg N$。
- 稀疏表示（式 2）：$x=\Psi a$，$\Psi\in\mathbb C^{N\times L}$ 为 wavelet basis / over-complete frame，$a$ 在 $\Psi$ 下 $K$-sparse（$K\ll N$）。

**MAP 与正则化目标 (PDF §2.2)**
- 由 Bayes（式 4/5）得 MAP 估计 $x_{\mathrm{map}}=\arg\max_x p(x\mid y)$。
- 高斯似然（式 6）$p(y\mid x)\propto\exp(-\|y-\Phi x\|_2^2/2\sigma^2)$，稀疏先验（式 7）$p(x)\propto\exp(-\phi(\mathcal B x))$。
- analysis 形式（式 10）：$x_{\mathrm{map}}=\arg\min_x\{\mu\|\Psi^\dagger x\|_1+\|y-\Phi x\|_2^2/2\sigma^2\}$；synthesis 形式（式 11）：$x_{\mathrm{map}}=\Psi\arg\min_a\{\mu\|a\|_1+\|y-\Phi\Psi a\|_2^2/2\sigma^2\}$。

**标准（offline）forward-backward (PDF §2.3)**
- 一般问题（式 12）$\arg\min_x\{f(x)+g(x)\}$，$f$ 凸+下半连续，$g$ 凸+ $\beta_{\mathrm{Lip}}$-Lipschitz 可微（式 13）。
- proximity operator（式 14）$\mathrm{prox}_f^\lambda(z)=\arg\min_u\{f(u)+\|u-z\|^2/2\lambda\}$。
- FB 迭代（式 16）$x^{(i+1)}=\mathrm{prox}_{\lambda^{(i)}f}\!\big(x^{(i)}-\lambda^{(i)}\nabla g(x^{(i)})\big)$，$\lambda^{(i)}$ 为步长，收敛到式 12 的极小子。**关键：offline 每次迭代都要用全部 $M$ 个 visibilities。**

**online 核心：数据分块 (PDF §3.1, 式 20/21)**
- 把 $y$ 分成 $B$ 块：$y=[y_1^\top,\dots,y_B^\top]^\top$，$y_k\in\mathbb C^{M_k}$，$\sum_k M_k=M$；相应 $\Phi=[\Phi_1^\top,\dots,\Phi_B^\top]^\top$。各块在**不同时刻**到达。
- 数据保真项可分（式 23）：$g=g_1+\cdots+g_B$，$g_k$ 对应第 $k$ 块。完整目标（式 24）$F_y(x)=f(x)+\sum_{k=1}^B g_k(x)$。
- 在只收到前 $b$ 块时（式 25）只能形成部分目标 $\mathcal F_{y_1^b}(x)=f(x)+\sum_{k=1}^b g_k(x)$。

**online forward-backward (PDF §3.2, Algorithm 1, 式 26)**
1. **初始化** $i=0,\ b=0$，给定 $x^{(0)}\in\mathbb R^N$、$\sigma$、步长 $\lambda^{(b)}$。
2. **外层循环（数据块级）** $b\leftarrow b+1$；**load** 新数据块 $y_b$（telescope 此时刚观测到它）。
3. **内层循环（迭代级，同化 + 成像）**：用前 $b$ 块的部分梯度做一步 FB（式 26）
   $$x^{(i+1)}=\mathrm{prox}_{\lambda^{(i)}f}\!\Big(x^{(i)}-\lambda^{(i)}\nabla g_1^b(x^{(i)})\Big),\quad g_1^b=g_1+\cdots+g_b,$$
   $i\leftarrow i+1$；直到 type-II 停止准则（实践中**每块只做一次迭代**，最省算力）。
4. **discard** $y_b$：内层迭代用完该块后即可释放其存储（Fig 1 中"Delete data block"）。
5. 回到第 2 步直到 type-I 停止准则（最大数据块数 $b=B$，或反馈"无新块"）。输出 $x^*=x^{(i)}$。

**analysis 实例（PDF §4.1.1, Algorithm 2, 式 38/39/41/42）**
- 设 $\bar f(x)=\mu\|\Psi^\dagger x\|_1$、$\bar g_k(x)=\|y_k-\Phi_k x\|_2^2/2\sigma^2$。当 $\Psi^\dagger\Psi=\mathrm I$（正交基）时
  $$\mathrm{prox}_{\lambda f}(\bar z)=\bar z+\Psi\big(\mathrm{soft}_{\lambda\mu}(\Psi^\dagger\bar z)-\Psi^\dagger\bar z\big),$$
  部分梯度 $\nabla \bar g_1^b(x)=\sum_{k=1}^b\Phi_k^\dagger(\Phi_k x-y_k)/\sigma^2$。迭代展开为式 41/42：
  $$v^{(i)}=x^{(i)}-\lambda^{(i)}\sum_{k=1}^b\Phi_k^\dagger(\Phi_k x^{(i)}-y_k)/\sigma^2,\quad x^{(i+1)}=v^{(i)}+\Psi(\mathrm{soft}_{\lambda^{(i)}\mu}(\Psi^\dagger v^{(i)})-\Psi^\dagger v^{(i)}).$$
  软阈值（式 40）$\mathrm{soft}_\lambda(z_k)=z_k(|z_k|-\lambda)/|z_k|$ if $|z_k|>\lambda$ else 0。
- **重要工程优化**：$\Phi_k^\dagger y_k$（dirty map）以及 $\Phi_k^\dagger\Phi_k$ 可**预计算一次**后反复调用，使每块成本只取决于已同化的块数 $b$ 而非全量 $B$。
- synthesis 实例见 Algorithm 3（式 49），在正交基下两者性能差异可忽略；论文实验只报告 analysis 模型。

**收敛保证 (PDF §3.3, Theorem 3.2, 式 29)**
- 假设（式 29）：$\sum_{k=b+1}^B g_k(x^{(i)})\ge\sum_{k=b+1}^B g_k(x^{(i+1)})$——直觉是"用更多数据得到的中间重建，对尚未观测的块也拟合得更好"。
- **Theorem 3.2**：在式 29 下，online Algorithm 1 产生的 $\{\mathcal F_y(x^{(i)})\}_i$ **单调递减**至完整目标的极小值 $\mathcal F_y(x^*)$。证明把完整目标拆为"已用块的部分目标"+"未用块之和"（式 32–35），用 splitting 方法的标准收敛性 + 式 29 完成。

---

## 4. 完整复现所需数据集

论文实验是 **simulation**（不是真实 telescope 观测），数据获取流程如下（PDF §5.1）：

| 测试图 | 尺寸 | 类型 | 复现可获得性 |
|--------|------|------|--------------|
| M31 (HI region) | 256×256 | 星系氢区图 | 公开天文图常见；可用 SExtractor / NRAO 公开 fits 或经典 RI 测试图集 |
| Cygnus A | 256×512 | 射电星系 | 公开（VLA 经典源），可从射电天文图库获取 |
| W28 | 256×256 | 超新星遗迹 | 公开射电源 |
| 3C288 | 256×256 | 射电星系 | 公开射电源 |

- **从 ground-truth 生成 visibilities**：用 variable-density sampling profile（Puy et al. 2011）在**半个 Fourier 平面**采样，取每张 ground-truth 的 **10% 离散 Fourier 系数**；再加零均值高斯噪声，$\sigma=\|f\|_\infty 10^{-\mathrm{SNR}/20}$，输入 SNR 固定 **30 dB**。
- **sensing operator**：$\Phi_k=M_k F$，$F$ 为 FFT，$M_k$ 为 masking。论文简化为 on-grid，真实 off-grid 情形需把 $M_k$ 换成 degridding 矩阵 + 零填充 + degridding（见 Pratley et al. 2018 / PURIFY）。
- **达到 paper-like 的等价数据来源**：本仓库目前用 synthetic 圆盘，要走向 paper-like，需要 (a) 取上述四张公开射电图作 ground-truth，(b) 用 variable-density profile 生成半 Fourier 平面 10% 采样的 visibility blocks，(c) 加 30 dB 高斯噪声。若要做**真实** RI 数据（非 simulation），则需私有/公开的 telescope visibility（uv-track）数据，并接入 PURIFY 的 NUFFT 算子——这超出本仓库现状。

---

## 5. 对照基线 (Baselines)

- **主对照**：论文的 **standard (offline) forward-backward algorithm**（同一 analysis 模型 式 10，但每次迭代用全部 visibilities，$i_{\max}=50$）。这是论文唯一定量对照对象，目的是证明 online 与 offline **重建质量等价、存储/算力更省**。
- **方法学背景基线（论文综述提及，未逐一定量对照）**：CLEAN 及其变体（Högbom 1974 等）、最大熵法 MEM、compressive sensing 类方法（Wiaux et al. 2009a/b；McEwen & Wiaux 2011；Carrillo et al. 2012/2014；Onose et al. 2016/2017；Pratley et al. 2018）。论文强调这些都是 offline，需观测完成后才能开始重建。
- **合理可加的对照（复现可选）**：把同一 online 框架套到 synthesis 模型（Algorithm 3）做内部一致性对照；或与 PURIFY 的 offline 重建对齐。

---

## 6. 评价指标与论文报告结果

**指标定义（PDF §5.1, 式 53）**
$$\mathrm{SNR}=20\log_{10}\frac{\|x\|_2}{\|x-x^*\|_2}\ (\mathrm{dB}),$$
$x$ 为 ground-truth，$x^*$ 为重建。Table 1 用相对差（式 54）$\mathrm{rel.diff}=(\mathrm{SNR}_{\mathrm{standard}}-\mathrm{SNR}_{\mathrm{online}})/\mathrm{SNR}_{\mathrm{standard}}$。

**论文报告的关键数值（均可在 PDF 核实，引用时标注来源）**
- **M31**：online 与 standard 取得**相同** SNR **14.2946 dB**（PDF §5.2.1，analysis 模型，$B=50$）。用 alternative splitting（按距原点 vs 均匀随机）得 14.2943 dB，几乎相同。
- **Table 1（PDF p.13）**：四张图、$B\in\{50,100,200,300,500\}$ 下 online 与 standard 的 SNR 相对差量级——M31 约 **10⁻⁶~10⁻⁷**、3C288 约 **10⁻⁶~10⁻⁸**（极小），Cygnus A / W28 约 **10⁻²~10⁻³**；符号有正有负，说明两者互有胜负但**无实质差异**。
- **存储比（式 50）**：$\eta_s=\max_k\{M_k\}/M$，等块时 $\eta_s=1/B$；$B>100$ 时存储 $<1\%$。
- **算力比（式 51/52）**：当 $i_{\max}$ 足够大且两法迭代数相近时 $\eta_c\approx(B+1)/(2 i_{\max})\approx 1/2$，即 online 约省一半计算。
- **实验配置**：$\mu=10^4$（试错定），$i_{\max}=50$，$\Psi=$ Daubechies-8 wavelets（MATLAB `wavedec2`），硬件 MacBook 2.2 GHz i7 / 16 GB / MATLAB R2015b。

**禁止编造**：论文**未**报告 PSNR、Dice、SSIM 等指标；其定量主轴是 **SNR（dB）+ 存储比 + 算力比**。本仓库 toy 用 PSNR 仅为内部可视化，与论文指标体系不同，不可混为一谈。

---

## 7. 本仓库当前复现实现

- **runnerFile**：`reproduce/experiments/online_ri_toy.py`（experiment_id = `online_ri_toy`）。
- **现在用的是真实算法/求解器**：online forward-backward（论文 Algorithm 2，analysis form）+ Daubechies-8 正交小波软阈值 prox + standard offline FB 基线。不再是 inverse-FFT dirty image。
- **实际做了什么**：
  1. 造 64×64 synthetic ground-truth（3 个扩展圆盘 + 2 个点源），归一化到 [0,1]；
  2. 测量算子 $\Phi_k=M_k F$：$F$ 用 `np.fft.fft2(norm="ortho")`（正交 FFT，$\|\Phi^H\Phi\|=1$，梯度 1-Lipschitz），$M$ 为 **variable-density** 掩码（密度 $1/(1+(r/0.06)^2)$，uv-原点附近更密，约 30% 覆盖，强制保留 DC），呼应论文 Puy et al. 采样 profile；按 30 dB 输入 SNR 加复高斯噪声得 visibilities $y$；
  3. 稀疏基 $\Psi=$ Daubechies-8 小波（`pywt`，`periodization` 模式，level=2，$\Psi^H\Psi=\mathrm I$，Parseval 能量守恒，round-trip 误差 ~1e-15）；prox of $\mu\|\Psi^H x\|_1$ = 小波域软阈值 $\Psi\,\mathrm{soft}_{\lambda\mu}(\Psi^H z)$（论文式 38/40）；
  4. 把采样坐标打乱后 `array_split` 成 **B=8** 块；每块预计算 dirty map $\Phi_k^H y_k$（论文 Remark 4.2）后即可丢弃原始 visibilities；
  5. **online FB**：第 $i$ 步当 $i<B$ 时同化到达的第 $i$ 块（累加进 running mask/dirty 摘要后丢弃该块），梯度步 $v=x-\lambda\,(\Phi^H_{\le b}\Phi_{\le b}x-\sum_{k\le b}\Phi_k^H y_k)$，再 prox 步 + 非负投影；全部块到达后继续迭代到 $i_{\max}=50$（论文 Algorithm 1/2 的可选 extra iterations）；
  6. **offline FB 基线**：每步用全量 $M$ 个 visibilities 做同样的 FB，跑 $i_{\max}=50$ 步；
  7. 另算无正则的 dirty image 作参照，证明 FB 带来的增益；
  8. 输出五联图 `assets/repro/online_ri_storage_quality.png`（truth / dirty / offline FB / online FB / SNR-vs-FB-step 曲线）。
- **参数**：$\mu=0.005$（小波 L1 权重，相对 dirty-image 尺度调得；$\mu=0$ 也收敛，$\mu>0$ 给小幅 SNR 增益），$\lambda=1.0$（步长，正交 FFT 下 $L=1$），$i_{\max}=50$，$B=8$，input SNR=30 dB。
- **指标用论文口径 SNR（式 53）**：$\mathrm{SNR}=20\log_{10}(\|x\|_2/\|x-x^*\|_2)$ dB；并报存储比 $\eta_s=\max_k M_k/M$（式 50）。
- **当前 runMetrics（确定性，跨次运行完全一致，CPU ~0.2 s）**：

| 指标 | 值 | 含义 |
|------|----|------|
| offline_snr_db | 24.6404 | offline FB 重建 SNR |
| online_snr_db | 24.5580 | online FB 重建 SNR |
| dirty_snr_db | 15.1086 | 无正则 dirty image SNR（基线下界）|
| online_offline_rel_diff | 0.003347 | $(\mathrm{SNR}_{\mathrm{off}}-\mathrm{SNR}_{\mathrm{on}})/\mathrm{SNR}_{\mathrm{off}}$（式 54）|
| num_blocks_B | 8 | 数据块数 |
| total_measurements_M | 1229 | 总采样系数数 |
| peak_stored_measurements_online | 154 | 单块最大系数数 $\max_k M_k$ |
| storage_ratio_eta_s | 0.1253 | $\eta_s=\max_k M_k/M\approx 1/B=0.125$ |

- 解读：online (24.56 dB) ≈ offline (24.64 dB)，rel.diff≈3.3e-3——**真正复现**了论文"online 与 offline 重建质量等价"的结论（量级与论文 Cygnus A/W28 的 10⁻²~10⁻³ 同档）；两者都**显著优于** dirty image（15.11 dB），证明 ℓ1 正则化 FB 求解器在起作用。$\eta_s=0.1253\approx1/B$ 精确印证论文式 50 的核心存储量。SNR-vs-step 曲线显示 online 早期因块少而落后、随同化更多块逐步追平 offline——正是论文 Figure 5 的定性行为。
- **resultFiles**：`assets/repro/online_ri_storage_quality.png`。

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

**已经做到（本轮升级闭合的缺口）**：
- ✅ **求解器**：已实现论文 Algorithm 2（analysis online FB）——Daubechies-8 正交小波 prox（软阈值，式 38/40）、按块累加的部分梯度、$\Phi_k^H y_k$ 预计算缓存（Remark 4.2）、丢弃机制；offline FB 基线并行运行。
- ✅ **指标**：已换成论文 SNR（式 53）+ rel.diff（式 54）+ 存储比 $\eta_s=\max_k M_k/M$（式 50），并复现了 online≈offline（rel.diff≈3e-3）与 $\eta_s\approx1/B$。
- ✅ **算子（部分）**：on-grid 正确的 $\Phi_k=M_k F$（正交 FFT），variable-density 采样 profile。

**仍缺（到 paper-like / paper-level）**：

1. **真实数据缺口**：当前 64×64 synthetic 多结构图 → 需换成 M31 / Cygnus A / W28 / 3C288 公开射电图作 ground-truth（256×256 / 256×512），并用 Puy et al. variable-density 在**半个 Fourier 平面**取 **10%** 系数 + 30 dB 噪声生成 visibility blocks。当前是全平面 ~30% 覆盖的 synthetic 圆盘。
2. **算子保真缺口**：masked-FFT 是 on-grid 理想化；paper-level 真实 off-grid RI 需 NUFFT/degridding（PURIFY 风格）、baseline 坐标与权重、w-projection / non-coplanar 修正、interpolation kernel 存储（论文指出 kernel 可达 measurement 存储 16+ 倍）。当前 $\eta_s$ 只刻画 visibility 部分。
3. **数据项尺度缺口**：当前把论文数据项的 $1/(2\sigma^2)$ 折进 $\mu$（masked-FFT 逆问题的标准重参数化），$\mu=0.005$ 而非论文 MATLAB 尺度的 $\mu=10^4$；paper-level 对齐需用论文同源数据 + 同尺度算子重定 $\mu$。
4. **规模缺口**：当前 $B=8$、$i_{\max}=50$、64×64；论文 $B\in\{50,100,200,300,500\}$（每块约 2% 系数）、256×256+，需放大规模并扫 $B$ 复现 Figure 2/5 的 $\eta_s$、$\eta_c\approx(B+1)/(2i_{\max})$ 曲线。
5. **表格对照缺口**：paper-level 需复现 Table 1（四图 × $B$ 的相对差量级 10⁻⁶~10⁻³）与 M31 SNR≈14.2946 dB；当前 synthetic 设定下的 24.6 dB / 3e-3 与论文具体数值不可直接比较。
6. **算力比缺口**：当前只报存储比 $\eta_s$；可加报算力比 $\eta_c\approx(B+1)/(2i_{\max})$（论文式 51/52）作为另一核心量。
7. **块到达时序缺口**：当前一次性采样后再分块（按 FB step 顺序逐块同化已模拟"用完即丢"），但未建模真实 streaming 的 wall-clock block-arrival 时间线与 peak-memory 实时监控；paper-like 应显式建模到达时序。

---

## 9. 运行步骤

**当前 toy 跑法**

```bash
# 安装依赖（项目统一依赖见 requirements.txt）
pip install -r requirements.txt

# 运行全部复现实验（含本篇 online_ri_toy）
cd reproduce && python run_all.py
```

- 依赖：本 runner `online_ri_toy.py` 的 `require_modules` 现检查 `numpy`、`matplotlib`、`scikit-image`、`pywt`（PyWavelets，提供 Daubechies-8 小波 Ψ）；缺任一所需模块时 runner 返回 `status=skipped`（不伪造 completed），与项目纪律一致。
- 产物：`docs/assets/repro/online_ri_storage_quality.png`（五联图：truth / dirty / offline FB / online FB / SNR-vs-FB-step），以及写回 dashboard 的 `runMetrics`。
- 单篇调试：可在 `reproduce/experiments/online_ri_toy.py` 中调 `run()`（CPU 秒级，~0.2 s）。

**向 paper-like 扩展的步骤大纲**（已实现的标 ✅，仅设计未实现的不标）

1. 载入四张公开射电 ground-truth（256×256 / 256×512）。*（仍缺，当前用 synthetic）*
2. 实现 variable-density **半** Fourier 平面 10% 采样 + 30 dB 高斯噪声 → visibility blocks。*（当前为全平面 ~30%；噪声口径 30 dB 已对齐 ✅）*
3. ✅ 实现 analysis online FB（Algorithm 2）：Daubechies-8 `prox` + 部分梯度 + 丢弃机制。*（参数当前 $\mu=0.005$、$i_{\max}=50$、$B=8$；paper-level 需 $\mu=10^4$ 同尺度、$B\in\{50,\dots,500\}$）*
4. ✅ 实现 standard offline FB 作基线，并行运行。
5. ✅ 计算 SNR（式 53）、存储比 $\eta_s=\max_k M_k/M$（式 50）。*（算力比 $\eta_c$ 与扫 $B$ 曲线、Figure 2/5、Table 1 仍缺）*
6. 输出 SNR-vs-step 曲线 ✅；storage/compute 扫 $B$ 曲线与论文表格对照仍缺。

---

## 10. 风险与代理说明

- **求解器已是真算法（不再是 proxy）**：当前重建是真实的 ℓ1-Daubechies-8-正则化 online forward-backward MAP（Algorithm 2），不是 dirty image。online (24.56 dB) 与 offline (24.64 dB) 同档、都优于 dirty (15.11 dB)，rel.diff≈3e-3——这是对论文 Theorem 3.2"online 收敛到与 offline 等价的极小值"的**数值印证**（在本 synthetic 设定下）。注意：这印证的是机制与趋势，**不是**论文 M31 的具体 14.2946 dB 数值。
- **仍存在的代理（核心）**：
  - **算子代理**：masked-FFT 是 on-grid 理想化，未含 degridding/NUFFT、w-projection/non-coplanar、baseline 坐标与权重、interpolation kernel 存储（论文指出 kernel 可达 measurement 存储的 16+ 倍）。因此 toy 的 $\eta_s$ 数字只刻画 visibility 部分，不代表真实 RI 的完整存储画像。
  - **数据代理**：64×64 synthetic 多结构图代替 M31/Cygnus A/W28/3C288 公开射电图；全平面 ~30% 覆盖代替半 Fourier 平面 10% 采样。
  - **尺度代理**：数据项 $1/(2\sigma^2)$ 折进 $\mu$（$\mu=0.005$），与论文 MATLAB 尺度的 $\mu=10^4$ 不同口径；$B=8$、64×64 远小于论文 $B\in\{50..500\}$、256×256+。
- **不可外推的结论**：toy 已能定性印证"online≈offline 且峰值存储 $\eta_s\approx1/B$"，但**不**支持任何关于论文 M31 具体 SNR（14.2946 dB）、Table 1 量级（10⁻⁶~10⁻³）、"SKA 级存储节省 99%"或"算力恰省一半"的**逐数值**主张；这些必须由 paper-like 实现 + 论文同源数据 + 真实 NUFFT 算子才能谈。
- **年份口径风险**：dashboard 标 `year=2019`（刊出年），PDF 本体为 2017（arXiv v1）。引用时需说明二者各自含义，避免"PDF 年份与卡片年份不符"的误读。

---

## 11. 参考：精读笔记

- 精读笔记：[`../../../xiaohao_cai_ultimate_notes/Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md)
- 复现代码：`reproduce/experiments/online_ri_toy.py`
- 论文 PDF：`docs/00_papers_first_author_xiaohao_cai_deduped/在线无线电干涉成像 Online Radio Imaging.pdf`
