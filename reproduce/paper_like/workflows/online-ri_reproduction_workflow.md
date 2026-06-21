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
- 当前实现（`reproduce/experiments/online_ri_toy.py`）用 40×40 synthetic 双圆盘图像 + 随机 Fourier 掩码（masked FFT 作为 sensing operator），online 部分把已采样的 Fourier 系数按块逐步注入，再用**纯 inverse-FFT 的 dirty-image 反投影**作为"重建"，**完全没有** ℓ1 稀疏正则化、没有 Daubechies-8 小波 Ψ、没有 forward-backward proximal 迭代（Algorithm 2/3）、没有 PURIFY 的 NUFFT/degridding 算子。因此它只是把"分块同化 + 丢弃 + peak storage 下降"这一**工程直觉**做成可视化，**不能**当作对论文 online forward-backward MAP 算法的复现。
- 纪律强调：本文档第 6 节引用的论文数值（如 M31 SNR ≈ 14.2946 dB、Table 1 相对差 ~10⁻⁶）仅供对照；当前 toy 实现**不复现这些数值**。把 toy 的 `online_psnr=12.3359`、`peak storage 585→98` 解读为"论文级 online RI 已复现"是错误陈述。

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
- **实际做了什么**：
  1. 造 40×40 synthetic 双圆盘 ground-truth；
  2. 取其 FFT，用随机布尔掩码（约 38% 系数，强制保留 DC）+ 小高斯噪声构成"已采样 visibilities" $y$；
  3. offline = 对全 $y$ 做 `ifft2` 取实部并 clip 到 [0,1]（dirty image，**无正则化**）；
  4. online = 把已采样坐标打乱后 `array_split` 成 6 块，逐块把对应 Fourier 系数写入 `online_y`，每写完一块就重算一次 `ifft2` dirty image，记录 PSNR trace 与 peak block size；
  5. 输出四联图 `assets/repro/online_ri_storage_quality.png`（truth / offline / online final / online PSNR-vs-block）。
- **用的 proxy**：masked-FFT 代替真实 NUFFT/degridding 算子；inverse-FFT dirty image 代替 ℓ1-regularised forward-backward MAP；"逐块注入 Fourier 系数"代替真正的 online proximal 同化；synthetic 圆盘代替 M31/Cygnus A/W28/3C288。
- **当前 runMetrics（来自 `reproStructured.runMetrics`）**：

| 指标 | 值 |
|------|----|
| offline_psnr | 12.3359 |
| online_psnr | 12.3359 |
| offline_snr | 2.6069 |
| online_snr | 2.6069 |
| peak_stored_measurements_offline | 585 |
| peak_stored_measurements_online | 98 |
| runtimeSeconds | 0.07 |

- 解读：offline==online 的 PSNR/SNR 完全相等是因为 online 末块把**全部**已采样系数都注入，最终 dirty image 与 offline 逐字节相同——这恰好把论文"online 与 offline 等价"的直觉演示出来，但 quality 本身是 dirty-image 量级（SNR 仅 2.6 dB），**不是**论文那种正则化重建质量。peak storage 585→98 演示了"峰值只需存单块"的工程要点（注：585=全掩码系数数，98=单块最大系数数）。
- **resultFiles**：`assets/repro/online_ri_storage_quality.png`。

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

清单（按缺口类型）：

1. **数据缺口**：当前 synthetic 圆盘 → 需换成 M31 / Cygnus A / W28 / 3C288 公开射电图作 ground-truth，并用 variable-density 半 Fourier 平面 10% 采样 + 30 dB 高斯噪声生成 visibility blocks。
2. **求解器缺口（最关键）**：当前是 inverse-FFT dirty image，**完全没有正则化重建**。需实现论文 Algorithm 2（analysis online FB）：含 $\Psi=$ Daubechies-8 的 `prox`（软阈值，式 38/40）、部分梯度 $\sum_{k=1}^b\Phi_k^\dagger(\Phi_k x-y_k)/\sigma^2$、步长 $\lambda$ 与 $\mu=10^4$、$i_{\max}=50$。
3. **算子缺口**：masked-FFT → 至少 on-grid 正确的 $\Phi_k=M_k F$ 实现，并补上 $\Phi_k^\dagger y_k$ / $\Phi_k^\dagger\Phi_k$ 预计算缓存（论文强调的算力优化点）；paper-level 还需 NUFFT/degridding（PURIFY 风格）。
4. **基线缺口**：需并行实现 standard offline FB（同模型、全数据、$i_{\max}=50$）作为定量对照，才能复现"online≈offline"的结论。
5. **指标缺口**：把内部 PSNR 换成论文 SNR（式 53）；补存储比 $\eta_s=\max_k\{M_k\}/M$ 与算力比 $\eta_c\approx(B+1)/(2 i_{\max})$ 曲线（对应论文 Figure 2/5、Table 1）。
6. **表格对照缺口**：paper-level 需复现 Table 1（$B\in\{50,100,200,300,500\}$ 的相对差量级）与 M31 SNR≈14.2946 dB；当前实现不具备产生这些数值的能力。
7. **块到达时序缺口**：当前一次性持有全采样后再分块，并未模拟"块在不同时刻到达、用完即释放"的真实 streaming 时序与内存监控；paper-like 应显式建模 block-arrival 时间线与 peak-memory 追踪。

---

## 9. 运行步骤

**当前 toy 跑法**

```bash
# 安装依赖（项目统一依赖见 requirements.txt）
pip install -r requirements.txt

# 运行全部复现实验（含本篇 online_ri_toy）
cd reproduce && python run_all.py
```

- 依赖：本 runner `online_ri_toy.py` 的 `require_modules` 实际只检查 `numpy`、`scikit-image`、`matplotlib`（未导入 `scipy`）；项目统一 `requirements.txt` 仍含 `scipy`，供其他实验使用。缺任一所需模块时 runner 返回 `status=skipped`（不伪造 completed），与项目纪律一致。
- 产物：`docs/assets/repro/online_ri_storage_quality.png`，以及写回 dashboard 的 `runMetrics`。
- 单篇调试：可在 `reproduce/experiments/online_ri_toy.py` 中调 `run()`（CPU 秒级，~0.07 s）。

**向 paper-like 扩展的步骤大纲**（仅设计，不在本 toy 内实现）

1. 载入四张公开射电 ground-truth（256×256 / 256×512）。
2. 实现 variable-density 半 Fourier 平面 10% 采样 + 30 dB 高斯噪声 → visibility blocks。
3. 实现 analysis online FB（Algorithm 2）：Daubechies-8 `prox` + 部分梯度 + $\mu=10^4$、$i_{\max}=50$、$B\in\{50,\dots,500\}$。
4. 实现 standard offline FB 作基线，并行运行。
5. 计算 SNR（式 53）、存储比 $\eta_s=1/B$、算力比 $\eta_c$，复现 Figure 2/5、Table 1。
6. 输出 SNR-vs-iteration、storage/compute 曲线，与论文对照。

---

## 10. 风险与代理说明

- **proxy 局限（核心）**：inverse-FFT dirty image ≠ ℓ1-regularised MAP 重建。当前 online_psnr/online_snr 只是 dirty-image 量级（SNR≈2.6 dB），**不能**外推到论文的正则化重建质量（M31 SNR≈14.29 dB）。
- **"online==offline" 的来源**：toy 中两者数值逐字节相等，是因为末块注入全量系数后 dirty image 相同——这只能演示"信息无损同化"的直觉，**不能**当作论文 Theorem 3.2（目标函数单调收敛到等价极小值）的数值验证。
- **算子失真**：masked-FFT 是 on-grid 理想化，未含 degridding/NUFFT、baseline 坐标与权重、interpolation kernel 存储（论文指出这部分可达 measurement 存储的 16+ 倍）。因此 toy 的 peak storage 数字不代表真实 RI 的存储画像。
- **不可外推的结论**：toy 不支持任何关于"SKA 级存储节省 99%""算力省一半""不同 $B$ 下质量稳定"的**定量**主张；这些必须由 paper-like 实现 + 论文同源数据才能谈。
- **年份口径风险**：dashboard 标 `year=2019`（刊出年），PDF 本体为 2017（arXiv v1）。引用时需说明二者各自含义，避免"PDF 年份与卡片年份不符"的误读。

---

## 11. 参考：精读笔记

- 精读笔记：[`../../../xiaohao_cai_ultimate_notes/Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md)
- 复现代码：`reproduce/experiments/online_ri_toy.py`
- 论文 PDF：`docs/00_papers_first_author_xiaohao_cai_deduped/在线无线电干涉成像 Online Radio Imaging.pdf`
