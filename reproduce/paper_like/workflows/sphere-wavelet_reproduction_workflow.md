# 球面小波图像分割 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 8 篇 *Wavelet-Based Segmentation on the Sphere* 的完整复现流程规范。

---

## 1. 论文身份与第一作者核验

| 项 | 信息 |
|------|------|
| 标题 (EN) | Wavelet-Based Segmentation on the Sphere |
| 标题 (CN) | 球面小波图像分割 |
| 作者顺序 | **Xiaohao Cai**, Christopher G. R. Wallis, Jennifer Y. H. Chan, Jason D. McEwen |
| 第一作者核验 | 是。PDF 首页作者列表以 `XIAOHAO CAI*` 开头，星号注脚指向 Mullard Space Science Laboratory (MSSL), University College London (UCL)。Xiaohao Cai 为唯一第一作者。 |
| 年份 | 2016（arXiv:1609.06500，本仓库 PDF 为 v2，2019-11-10 修订版） |
| 主题 (theme) | extension（把 tight-frame 分割从 Euclidean 域扩展到 sphere $\mathbb{S}^2$） |
| PDF 路径 | `docs/00_papers_first_author_xiaohao_cai_deduped/球面小波分割 Wavelet Sphere.pdf` |
| 关键词 | Image segmentation, Wavelets, Curvelets, Tight frame, Sphere |

核验依据：PDF 第 1 页标题下方作者行、注脚机构信息，以及 arXiv 标识 `arXiv:1609.06500v2 [cs.CV] 10 Nov 2019`（页边竖排）。

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
- 当前实现已**从 Gaussian 代理升级为真实算法**：平面 lat-lon 网格上用 pywt undecimated SWT tight-frame 软阈值（真实 $\mathcal{A}^\top\mathcal{T}_\lambda\mathcal{A}$）去噪 + 真实离散球面梯度（式 2.5）+ 真实 WSSA 区间收缩迭代（式 3.3–3.10），并复现 $|\Lambda^{(i)}|\to 0$ 的 few-iteration 收敛。但**仍不含** S2LET / SSHT / SO3 的真正球面（球谐）小波栈与方向/curvelet 小波，平面 SWT 与 lat-lon 网格只是球面采样的近似。因此它仍是"toy"，**不能**作为对论文 WSSA 算法的方向性结论或真实数据结果的复现。
- 纪律强调：本文档中第 6 节引用的论文数值仅供对照，**当前实现不复现这些数值**；任何把 toy 的 `Dice 0.9529` / `kmeans_dice 1.0` 当作论文级指标的陈述都是错误的（论文本身根本未报告 Dice，见第 6、8 节）。

---

## 3. 算法完整流程

论文方法称为 **WSSA (Wavelet-based Spherical Segmentation Algorithm)**，是 tight-frame vessel segmentation（[9,10]，对应本项目 Framelet/Tight-frame Vessel 论文）从 Euclidean 域到 $\mathbb{S}^2$ 的直接扩展。忠于 PDF 第 2、3 节（含 Algorithm 1）的 step-by-step pipeline 如下。

**记号与基础（PDF §2）**
- 球面图像 $f \in L^2(\mathbb{S}^2)$，不失一般性假设 $f \in [0,1]$；球面坐标 $\omega=(\theta,\phi)$，$\theta\in[0,\pi]$ 余纬，$\phi\in[0,2\pi)$ 经度。
- **等角采样 (equiangular sampling, [38])**：$\theta_t=\frac{\pi(2t+1)}{2L-1},\ \phi_p=\frac{2\pi p}{2L-1}$，$t\in\{0,\dots,L-1\}$，$p\in\{0,\dots,2L-2\}$。当 $L=512$ 时离散球面有 $512\times1023=523776$ 个样本。
- **球面小波 (scale-discretised wavelets)**：方向小波系数定义在旋转群 $SO(3)$ 上
  $$W^{\Psi^{(j)}}(\rho)\equiv(f\circledast\Psi^{(j)})(\rho)=\langle f,\mathcal{R}_\rho\Psi^{(j)}\rangle,$$
  尺度系数 $W^\Phi(\omega)=(f\odot\Phi)(\omega)=\langle f,\mathcal{R}_\omega\Phi\rangle$（式 2.1、2.2）。
- **精确重构 (synthesis, 式 2.3)**：
  $$f(\omega)=\int_{\mathbb{S}^2}\!d\Omega(\omega')W^\Phi(\omega')(\mathcal{R}_{\omega'}\Phi)(\omega)+\sum_{j=J_{\min}}^{J_{\max}}\int_{SO(3)}\!d\varrho(\rho)\,W^{\Psi^{(j)}}(\rho)(\mathcal{R}_\rho\Psi^{(j)})(\omega).$$
  小波在调和空间因式化构造 $\Psi^{(j)}_{\ell m}=\sqrt{\frac{2\ell+1}{8\pi^2}}\,\kappa^{(j)}(\ell)\,\zeta_{\ell m}$（式 2.4），其中 $\kappa^{(j)}$ 控制角向定位、$\zeta$ 控制方向定位。**curvelets** 具有 parabolic scaling，对 curvilinear 结构最敏感。代码在公开包 **S2LET**（依赖 **SSHT** 与 **SO3**）中。
- **离散球面梯度 (式 2.5)**：$\|\nabla f\|\equiv\sqrt{(\delta_\theta f)^2+\frac{1}{\sin^2\theta_t}(\delta_\phi f)^2}$，$\delta_\theta,\delta_\phi$ 为有限差分。
- **tight-frame 通式 (式 2.6–2.8)**：$f^{(i+1/2)}=\mathcal{U}(f^{(i)})$，$f^{(i+1)}=\mathcal{A}^\top\mathcal{T}_\lambda(\mathcal{A}f^{(i+1/2)})$，软阈值 $t_\lambda(v_k)=\mathrm{sign}(v_k)(|v_k|-\lambda)$ if $|v_k|>\lambda$ else $0$。$\mathcal{A},\mathcal{A}^\top$ 在本文是球面小波正/逆变换。

**WSSA 主流程（PDF §3, Algorithm 1）**

1. **预处理去噪 (式 3.1)**：若 $f$ 含显著噪声，做一步 tight-frame 软阈值
   $$\bar f=\mathcal{A}^\top\mathcal{T}_{\bar\lambda}(\mathcal{A}f),\qquad \bar\lambda=\sigma/4.$$
2. **初始化边界候选集 (式 3.2)**：$\Lambda^{(0)}\equiv\{k\in\mathbb{S}^2\mid\|[\nabla\bar f]_k\|_1>\epsilon\}$，即球面梯度大于阈值 $\epsilon$ 的像素。置 $f^{(0)}=\bar f$，$i=0$。
3. **Step 1 — 计算灰度区间 $[a_i,b_i]$ (式 3.3–3.5)**：在未分类集 $\Lambda^{(i)}$ 上
   $$\mu^{(i)}=\tfrac{1}{|\Lambda^{(i)}|}\sum_{k\in\Lambda^{(i)}}f^{(i)}_k,\quad
   a_i=\max\!\big(\tfrac{\mu^{(i)}+\mu^{(i)}_-}{2},0\big),\quad
   b_i=\min\!\big(\tfrac{\mu^{(i)}+\mu^{(i)}_+}{2},1\big),$$
   其中 $\mu^{(i)}_-,\mu^{(i)}_+$ 是以 $\mu^{(i)}$ 为界的下/上子集均值。关键性质：$[a_i,b_i]$ 长度约为上一区间的一半（shrinkage）。
4. **Step 2 — 三段阈值 (式 3.6–3.8)**：
   $$f^{(i+1/2)}_k=\begin{cases}0,&f^{(i)}_k\le a_i\\ \frac{f^{(i)}_k-m_i}{M_i-m_i},&a_i\le f^{(i)}_k\le b_i\\ 1,&b_i\le f^{(i)}_k\end{cases}$$
   $M_i,m_i$ 为区间内最大/最小值；更新剩余未分类集 $\Lambda^{(i+1)}=\{k\mid 0<f^{(i+1/2)}_k<1\}$。若 $\Lambda^{(i+1)}=\emptyset$ 则 $f^{(i+1/2)}$ 已二值，停止。
   - **可选简化分支 (式 3.9)**：对各向同性结构，可用单阈值 $f^{(i+2/3)}_k=\mathbb{1}[f^{(i+1/2)}_k\ge\mu]$ 直接终止，跳过球面小波迭代以节省时间（论文指出对 mostly isotropic 结构这样更经济）。
5. **Step 3 — 球面小波迭代 (式 3.10)**：令 $\mathcal{P}^{(i+1)}$ 为在 $\Lambda^{(i+1)}$ 处取 1 的掩码算子，
   $$f^{(i+1)}=(\mathcal{I}-\mathcal{P}^{(i+1)})f^{(i+1/2)}+\mathcal{P}^{(i+1)}\mathcal{A}^\top\mathcal{T}_\lambda(\mathcal{A}f^{(i+1/2)}),\qquad \lambda=\sigma/100.$$
   仅在未分类区附近做球面小波正/逆变换可显著降本（论文留作 future work）。
6. **停止准则**：当 $\Lambda^{(i)}=\emptyset$（所有像素已是 0/1）时终止；值为 1 的像素构成 object of interest，0 为 background。论文报告通常**约 10 次迭代内收敛**，且第 3 次迭代后未分类像素数已远小于 $|\mathbb{S}^2|$。

**变体命名**：装备 axisymmetric / directional / hybrid 小波的 WSSA 分别记为 **WSSA-A / WSSA-D / WSSA-H**。Hybrid 在调和空间以 transition band-limit $L_{\text{trans}}$ 切分：$\ell\lesssim L_{\text{trans}}$ 用 curvelets，其余用 directional wavelets，兼顾 curvelet 的方向定位与 directional 的计算效率。

---

## 4. 完整复现所需数据集

论文在四类真实球面图像上测试（PDF §4），均为公开来源或可由公开工具构造：

| 数据 | 论文用途 | 来源 / 等价候选 |
|------|----------|------------------|
| **Earth topographic map** | 陆海分割 | Earth Gravitational Model **EGM2008**（U.S. NGA EGM Dev. Team 公开发布）；论文用 Frederik Simons 网页工具下载提取，band-limit 到 $L=512$。来源注脚：`http://www.frederik.net`。 |
| **Light probe image (Uffizi Gallery)** | 天空/窗框等亮部分割 | Paul Debevec light probe 库，`http://www.pauldebevec.com/Probes/`；由两张相隔 90° 的镜面球照片拼成完整球面。 |
| **Solar data-set 1** | 太阳耀斑特征分割 | SDO/AIA + STEREO-A/SECCHI + STEREO-B/SECCHI 三航天器 2012-07-08 在 30.4 nm 的拼接全日面图。来源：`http://sdo.gsfc.nasa.gov/`、`http://www.stereo.rl.ac.uk/`。 |
| **Solar data-set 2** | 活动磁区分割 | 太阳径向磁场 synoptic 图（Carrington Rotation 1974，2001-03-13 至 2001-04-09）。来源：JSOC HMI synoptic，`http://jsoc.stanford.edu`。 |
| **Retina images (×2)** | 强各向异性血管网络分割 | **DRIVE** 数据集（荷兰糖网筛查，Canon CR5 3CCD，45° FOV，$768\times584$，8-bit）。来源：`http://www.isi.uu.nl/Research/Databases/DRIVE/`。论文把 2D 视网膜图投影到球面构造球面测试数据。 |

**视网膜→球面构造流程（PDF §4.2，可复现）**：(1) 取彩色图绿通道；(2) 用 MATLAB `medfilt2` 估背景并相减得 tidy background；(3) 加 Gaussian 噪声得 noisy image；(4) 投影到球面坐标得球面测试数据。

**噪声约定**：对测试数据加 SNR=30 dB、零均值 Gaussian 噪声，$\sigma=\|f\|_\infty\,10^{-\mathrm{SNR}/20}$，$\|\cdot\|_\infty$ 为最大值。

paper-like 复现**无私有/受限数据障碍**：四类数据均公开可得，主要门槛在 S2LET 球面小波栈与采样约定，而非数据获取。

---

## 5. 对照基线 (Baselines)

- **K-means**（论文唯一定量/定性对照）：MATLAB 内置 `kmeans`，按球面像素强度聚类。论文结论是 WSSA 在保留方向/曲线结构上优于 K-means，尤其在 solar 与 retina 这类强方向数据上 K-means 失败明显（retina 中 K-means 丢失大量血管）。
- **WSSA 内部三变体互为对照**：WSSA-A（axisymmetric）/ WSSA-D（directional, $N=5,6$）/ WSSA-H（hybrid）。论文观察：WSSA-D、WSSA-H 在方向特征上略优于 WSSA-A，但 WSSA-A 最快。
- 论文背景中提到的相关方法（Mumford-Shah、Chan-Vese、graph-cut、deformable models、SaT/SLaT、spherical CNN 等）作为文献定位，**未**与本文方法直接定量对比；若要更强 baseline 体系可补这些方法。

---

## 6. 评价指标与论文报告结果

**论文实际报告的"指标"**：本文是以**可视化分割图 + 收敛性/计算时间表**为主的定性论文，**未报告 IoU / Dice / 精度召回等数值分割指标**（这一点在本项目精读笔记 §4.2 中也被如实指出）。可对照的硬数值来自 Table 4.1–4.4，记录每次迭代未分类像素数 $|\Lambda^{(i)}|$ 与计算时间（单位秒，机器为 2.2 GHz Intel Core i7 + 16 GB RAM）。以下数值已从 PDF 表格核实，引用注明表号：

- **收敛速度**：四张表均显示约 10 次迭代内 $|\Lambda^{(i)}|\to 0$；第 3 次迭代后未分类像素已远小于 $|\mathbb{S}^2|=523776$。例：Earth map（Table 4.1）WSSA-A 的 $|\Lambda^{(i)}|$ 序列 $111371\to106977\to25880\to6352\to\dots\to0$（第 10 步）。
- **计算时间（Table 4.1，Earth map）**：K-means `<1 s`；WSSA-A `51.9 s`；WSSA-D ($N=5$) `200.5 s`；WSSA-D ($N=6$) `217.2 s`；WSSA-H `883.5 s`。趋势：axisymmetric ≪ directional ≪ hybrid（含 curvelet）。
- **计算时间（Table 4.2，Uffizi light probe）**：K-means `<1 s`；WSSA-A `41.9 s`；WSSA-D ($N=5$) `145.7 s`；WSSA-D ($N=6$) `152.2 s`；WSSA-H `702.7 s`。
- **计算时间（Table 4.3，Solar map 1）**：K-means `<1 s`；WSSA-A `34.1 s`；WSSA-D ($N=5$) `124.3 s`；WSSA-D ($N=6$) `151.8 s`；WSSA-H `682.2 s`。
- **计算时间（Table 4.4，Retina）**：K-means `<1 s`；WSSA-A `50.66 s`；WSSA-D ($N=5$) `160.54 s`；WSSA-D ($N=6$) `197.0 s`；WSSA-H ($L_{\text{trans}}=32$) `789.6 s`；WSSA-H ($L_{\text{trans}}=64$) `4538.9 s`。
- **复杂度（PDF §2.2）**：axisymmetric $\mathcal{O}(L^3)$；directional $\mathcal{O}(NL^3)$；curvelet $\mathcal{O}(L^3\log_2 L)$。$L=512$ 一次正反 round-trip：axisymmetric 几秒、directional 几分钟、curvelet 几小时。

**定性结论（PDF §4）**：所有方法在 Earth map 上都能合理分出陆海；WSSA 优于 K-means；WSSA-D/H 在方向结构上略优于 WSSA-A；在 retina 上 K-means 大量丢失血管而 WSSA 检出大部分血管，WSSA-D/WSSA-H 优于 WSSA-A。**禁止编造任何 Dice/IoU 数字冒充论文结果。**

---

## 7. 本仓库当前复现实现

- **runner 文件**：`reproduce/experiments/sphere_wavelet_toy.py`（`experiment_id = sphere_wavelet_toy`，priority 8）。**已从 Gaussian 代理升级为真实小波 tight-frame 算法**（见下）。
- **它现在实际做了什么（真实算法栈）**：
  1. 在 $72\times144$ 的等角风格 lat-lon 网格上合成一张 sphere-like 图（两条由 `sin(2*lon)` 调制的弯曲 band），按论文噪声约定 SNR$=30$ dB 加噪：$\sigma=\|f\|_\infty 10^{-30/20}$。网格尺寸均可被 $2^{\text{level}}=4$ 整除，满足无下采样 SWT tight frame 的要求。
  2. **真实 tight-frame 去噪算子（替代 Gaussian）**：用 **pywt `swt2`/`iswt2`**（undecimated stationary wavelet transform，`db2`，level 2，`norm=True`）构造 $\mathcal{A}^\top\mathcal{T}_\lambda\mathcal{A}$，对细节子带做软阈值 $\bar\lambda=\sigma/4$（式 3.1）。SWT 无下采样、满足紧框架 $\mathcal{A}^\top\mathcal{A}=\mathcal{I}$（实测 round-trip 误差 $\sim 3\times10^{-16}$），这正是论文"tight frame"的要求，**Gaussian blur 不具备此性质**。可测效果：去噪后到 clean 信号的 RMS 比原噪声图降低 `denoise_gain_db ≈ 3.33` dB。
  3. **真实离散球面梯度（式 2.5，替代 `max(cos,0.2)` 截断）**：用余纬 $\theta\in(0,\pi)$ 等角内点的有限差分，$\|\nabla f\|=\sqrt{(\delta_\theta f)^2+\frac{1}{\sin^2\theta}(\delta_\phi f)^2}$，$\phi$ 方向周期 wrap；不再用 `max(cos(lat),0.2)` 人为压制高纬奇异。
  4. **真实 WSSA 区间收缩迭代（式 3.3–3.10，替代单步阈值）**：初始化 $\Lambda^{(0)}=\{\|\nabla\bar f\|>\epsilon\}$（$\epsilon=0.02$，论文 Earth-map 取值）；每步计算 $[a_i,b_i]$（式 3.3–3.5）→ 三段阈值得 $f^{(i+1/2)}$（式 3.6–3.8）→ 在未分类区做掩码 tight-frame 小波步 $\lambda=\sigma/100$（式 3.10）；记录 $|\Lambda^{(i)}|$ 收缩序列。
  5. 终值二值化得分割，与 truth 算 Dice；并跑一个 **K-means 强度聚类基线**（论文唯一定量对照）做透明对照。
- **当前 runMetrics**（确定性，派生自 `common.SEED`）：`dice = 0.9529`，`kmeans_dice = 1.0`，`denoise_gain_db = 3.3291`，`lambda_initial = 8226`，`lambda_final = 0`，`lambda_shrink_ratio = 0.0`，`interval_halving_ratio ≈ 0.8132`，`wssa_iterations = 5`，`snr_db = 30.0`，`epsilon = 0.02`，`runtime_seconds ≈ 0.2`，`status = completed`。
  - **可验证的论文现象**：$|\Lambda^{(i)}|$ 从 8226 在 **5 次迭代内收缩到 0**，复现了论文"约 10 次迭代内收敛"的 few-iteration 收敛性质（式 3.3 区间逐步缩小）。`interval_halving_ratio ≈ 0.81` 是区间长度每步平均缩小比例的实测见证（论文称约缩一半；本 toy 因强度分布近双峰而偏松）。
  - **诚实对照说明**：在论文 SNR$=30$ dB 约定下，本强度 toy 近似线性可分，K-means 反而拿满分（`kmeans_dice=1.0`）。`kmeans_dice` 仅作透明记录，**不**声称 WSSA 在此 toy 上"赢" K-means——论文 WSSA 优于 K-means 是**方向性**现象，本平面 proxy 无方向小波，genuinely 无法展示。
- **resultFiles**：`assets/repro/sphere_wavelet_toy.png`（五联图：noisy equirect / truth bands / SWT tight-frame denoise / spherical grad (eq2.5) / WSSA seg dice）。
- **诚实标注**：`notes` 与 `extra.fidelityWarning` 明确：用平面 pywt SWT 近似球面采样；lat-lon 网格非真正 $\mathbb{S}^2$ 等角采样；仍无 S2LET/SSHT/SO3 的 axisymmetric/directional/curvelet/hybrid 栈；Dice 是 toy 内部度量，论文不报告。

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

本轮升级后，**已落实为真实算法**的部分与**仍为缺口**的部分如下。

**✅ 升级已落实（真实算法，非代理）**：
- **tight-frame 软阈值去噪算子**：pywt undecimated SWT（`swt2`/`iswt2`，`db2`，level 2，`norm=True`）实现 $\mathcal{A}^\top\mathcal{T}_{\sigma/4}\mathcal{A}$（式 3.1），紧框架性质成立（round-trip $\sim 10^{-16}$），实测去噪增益 $\sim 3.33$ dB。**已替换 Gaussian 代理。**
- **离散球面梯度**：式 2.5 的 $\sqrt{(\delta_\theta f)^2+\frac{1}{\sin^2\theta}(\delta_\phi f)^2}$，余纬有限差分 + $\phi$ 周期 wrap。**已替换 `max(cos,0.2)` 截断。**
- **WSSA 区间收缩迭代**：式 3.3–3.10 的 $[a_i,b_i]$ 计算 + 三段阈值 + 掩码 tight-frame 小波步（$\lambda=\sigma/100$）。**已替换单步固定阈值**；并复现 $|\Lambda^{(i)}|\to 0$ 的 few-iteration 收敛（本 toy 5 步）。
- **K-means 基线**：已接入强度聚类对照（透明记录，见 §7 诚实说明）。

**❌ 仍为缺口（到 paper-like / paper-level）**：
1. **球面表示**：仍用平面 lat-lon 网格近似 $\mathbb{S}^2$ 等角采样（论文 $L=512$，$512\times1023$ 样本）。SWT 是平面无下采样小波，**不是**真正的球面（球谐）小波；网格也未严格遵循论文等角采样约定 $\theta_t,\phi_p$。
2. **球面小波栈（核心缺口）**：没有 S2LET/SSHT/SO3，没有 axisymmetric / directional / curvelet / hybrid 小波。平面 SWT 虽是真实 tight frame，但**不具备**论文方向/曲线小波（在 $SO(3)$ 上定义、parabolic scaling 的 curvelets）的方向选择性——这正是论文 WSSA-D/H 优于 WSSA-A 与 K-means 的关键卖点，本实现 genuinely 无法体现。
3. **数据缺失**：用 synthetic band 图，未接入 EGM2008 / Uffizi light probe / SDO+STEREO solar / DRIVE retina 真实数据，也未做论文的 retina→sphere 构造流程。
4. **变体缺失**：只有单一（平面）tight frame，无 WSSA-A/D/H 三变体及其计算时间对比。
5. **表格对照缺失**：虽已产出 $|\Lambda^{(i)}|$ 收敛序列（趋势对得上 Table 4.1–4.4 的"约 10 次收敛"），但未在真实 $512\times1023$ 球面 + 真实小波栈上复现表中具体数值与各变体秒级时间。
6. **指标口径**：论文不报告 Dice，本 toy 的 Dice/kmeans_dice 与论文不可比；在 SNR=30 dB 干净强度 toy 上 K-means 反而满分，**不能**外推为"WSSA 不如/优于 K-means"。

**达到 paper-like 的最小充分条件（更新）**：把现已实现的 WSSA 区间迭代骨架的 $\mathcal{A}/\mathcal{A}^\top$ 从平面 SWT 换成 **S2LET 球面小波栈**（至少 axisymmetric + directional）→ 在真正 $\mathbb{S}^2$ 等角采样上 → 用真实 Earth/solar/retina 之一 → 复现"约 10 次迭代收敛 + WSSA-D/H 在方向结构上优于 K-means"的定性趋势与秒级时间量级。paper-level 还需严格匹配采样约定、$\epsilon/\lambda$ 参数、hybrid 的 $L_{\text{trans}}\in\{32,64\}$ 设置，并复现表格中的 $|\Lambda^{(i)}|$ 数值。**注意：现有迭代/梯度/软阈值逻辑可直接复用，主要换的是 $\mathcal{A}$ 这一块。**

---

## 9. 运行步骤

**当前 toy 跑法**

```bash
# 安装依赖（见 reproStructured.dependencies）
pip install -r requirements.txt   # 关键：numpy, scipy, matplotlib, pywavelets(pywt)

# 运行全部复现实验（含本篇 sphere_wavelet_toy）
cd reproduce && python run_all.py
```

- 依赖缺失时 runner 写入 `status=skipped`（含 `pywt`），不会伪造 completed（符合项目纪律）。
- 产物：`assets/repro/sphere_wavelet_toy.png`（五联图）与结果 JSON 中的 `runMetrics`（含 `dice/kmeans_dice/denoise_gain_db/lambda_*/interval_halving_ratio/wssa_iterations` 等）。

**向 paper-like 扩展的步骤大纲（不在当前 toy 范围内；现有迭代/梯度/软阈值骨架可直接复用，主要换 $\mathcal{A}$）**

1. 安装球面栈：S2LET（依赖 SSHT、SO3），确认 `pyssht` 等角采样约定与 $L=512$；把现 runner 中的平面 `swt2/iswt2` 替换为球面小波正/逆变换。
2. 下载一类真实数据（建议先 Earth EGM2008 或 DRIVE retina），按论文方式 band-limit / 投影到真实 $\mathbb{S}^2$。
3. 现 runner 已实现式 2.5 离散球面梯度与式 3.1–3.10 的 WSSA 迭代（$\bar\lambda=\sigma/4$、$\lambda=\sigma/100$、$\epsilon$ 当前取 Earth 的 0.02）；扩展时按数据调 $\epsilon$（论文 Earth 0.02、light probe 0.05、Solar map 1 太阳耀斑 0.04、Solar map 2 活动磁区 0.05、retina Fig 4.5/4.6 均 0.04），并把网格换成真实等角采样。
4. 现已有 K-means 强度聚类对照与 $|\Lambda^{(i)}|$ 序列；扩展时记录各 WSSA-A/D/H 变体的计算时间，与 Table 4.1–4.4 趋势比对。
5. 全程在图注/JSON 中标注真实等级（partial / paper-like），严禁夸大。

---

## 10. 风险与代理说明

> 升级后，**已移除**的旧代理（不再适用）：Gaussian smoothing 代替小波去噪、`max(cos,0.2)` 截断代替 $1/\sin\theta$、单步固定阈值代替区间迭代。以下是**升级后仍然存在**的真实局限。

- **平面 SWT ≠ 球面（球谐）小波**（核心仍存代理）：pywt undecimated SWT 是真实的平面 tight frame（满足 $\mathcal{A}^\top\mathcal{A}=\mathcal{I}$，比 Gaussian 更接近论文"wavelet on sphere sampling"精神），但**不是**论文在 $SO(3)$ 上定义、具 parabolic scaling 的方向/curvelet 小波。因此 toy 仍**无法**体现 WSSA-D/WSSA-H 相对 WSSA-A 与 K-means 的方向性优势——这些结论**不可**从 toy 外推。
- **lat-lon 网格 ≈ $\mathbb{S}^2$ 等角采样（近似）**：现用余纬 $\theta\in(0,\pi)$ 等角内点 + 真实 $1/\sin\theta$ 几何加权（式 2.5），已去掉高纬人为截断；但仍非论文严格等角采样约定 $\theta_t=\frac{\pi(2t+1)}{2L-1}$、$\phi_p=\frac{2\pi p}{2L-1}$，真实球面拓扑（无平面平移不变性、极点奇异、$SO(3)$ 卷积）未被完整建模。
- **WSSA 迭代是真实的，但作用域是平面**：$[a_i,b_i]$ shrinkage + 三段阈值 + 掩码小波步（式 3.3–3.10）已按论文实现，并复现 $|\Lambda^{(i)}|\to 0$ 的 few-iteration 收敛；但 $\mathcal{A}$ 仍是平面 SWT，故收敛序列的**趋势**对得上 Table 4.1–4.4，**具体数值**不可比。
- **Dice 0.9529 / kmeans_dice 1.0 的语义**：仅是 toy 在自造 truth 上的内部度量，与论文无任何可比性（论文不报告 Dice）。在 SNR=30 dB 干净强度 toy 上 K-means 反而满分，**不得**据此声称"WSSA 优于/不如 K-means"，更**不得**将 Dice 表述为"球面小波分割论文级指标"。
- **可外推的有限结论**：toy 现可支撑"真实 tight-frame 小波去噪 + 真实球面梯度 + 真实 WSSA 区间迭代能在合成 band 图上收敛并给出合理分割，且去噪有可测增益（$\sim3.3$ dB）"这一命题——比旧 toy 强，但仍远弱于论文的方向性/真实数据结论。

---

## 11. 参考：精读笔记

- 本篇精读笔记：[`../../../xiaohao_cai_ultimate_notes/Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md)
- 关联前作（tight-frame 思想来源）：Framelet/Tight-frame Vessel Segmentation、SaT/SLaT 系列（见笔记"论文关系"小节）。
