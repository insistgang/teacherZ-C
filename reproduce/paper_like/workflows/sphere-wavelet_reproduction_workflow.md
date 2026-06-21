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
- 当前实现用 `equirectangular` 平面网格 + Gaussian smoothing + 近似 spherical-gradient，**不含** S2LET / SSHT / SO3 的真正球面小波栈，也不含论文的 boundary-interval 迭代收缩。因此它只能算"思路演示 toy"，**不能**作为对论文 WSSA 算法的复现。
- 纪律强调：本文档中第 6 节引用的论文数值仅供对照，**当前实现不复现这些数值**；任何把 toy 的 `Dice 0.8418` 当作论文级指标的陈述都是错误的（论文本身根本未报告 Dice，见第 6、8 节）。

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

- **runner 文件**：`reproduce/experiments/sphere_wavelet_toy.py`（`experiment_id = sphere_wavelet_toy`，priority 8）。
- **它实际做了什么**：
  1. 在 $72\times144$ 的 equirectangular 平面网格上合成一张 sphere-like 图（含两条由 `sin(2*lon)` 调制的弯曲 band），加 $\sigma=0.12$ 的 Gaussian 噪声。
  2. 用 `scipy.ndimage.gaussian_filter`（`sigma=(1.0,1.6)`，$\phi$ 方向 `wrap` 周期边界）平滑——**作为 wavelet 去噪的 proxy**。
  3. 用 `np.gradient` 算近似球面梯度，$\phi$ 分量除以 $\max(\cos(\mathrm{lat}),0.2)$ 模拟 $1/\sin\theta$ 几何加权。
  4. 用固定阈值 `smooth>0.47` 并集上 `spherical_grad > quantile(0.93)` 得到分割，与 truth 算 Dice。
- **当前 runMetrics（取自 `reproStructured`）**：`dice = 0.8418`，`gradient_threshold_quantile = 0.93`，`runtime_seconds ≈ 0.0723`，`status = completed`。
- **resultFiles**：`assets/repro/sphere_wavelet_toy.png`（四联图：equirectangular toy / truth bands / approx sphere grad / segmentation）。
- **proxy 说明**：`notes` 字段如实标注 "Approximate sphere toy: equirectangular smoothing plus spherical-gradient correction; no S2LET/SSHT/SO3 stack."

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

按缺口类型清单化：

1. **球面表示缺失**：当前用 equirectangular 平面网格替代真正的 $\mathbb{S}^2$ 等角采样（$L=512$，$512\times1023$ 样本）。`np.gradient` + `1/\cos(lat)` 只是几何近似，未实现式 2.5 的离散球面梯度算子。
2. **小波栈缺失（核心）**：没有 S2LET/SSHT/SO3，没有 axisymmetric / directional / curvelet / hybrid 小波；Gaussian smoothing 完全无法替代方向/曲线敏感的小波框架（这正是论文的关键卖点）。`implementationRisk` 字段已指出此点。
3. **算法缺失**：未实现 WSSA 的 boundary-interval 迭代收缩（式 3.3–3.10），当前是单步固定阈值，而非论文的 $[a_i,b_i]$ shrinkage + 三段阈值 + 球面小波迭代。
4. **数据缺失**：用 synthetic band 图，未接入 EGM2008 / Uffizi light probe / SDO+STEREO solar / DRIVE retina 真实数据，也未做论文的 retina→sphere 构造流程。
5. **基线缺失**：未实现 K-means 对照，也无 WSSA-A/D/H 三变体对比。
6. **表格对照缺失**：未产出 $|\Lambda^{(i)}|$ 收敛序列与各变体计算时间表，无法与 Table 4.1–4.4 趋势对照。
7. **指标口径**：论文本身不报告 Dice，因此当前 toy 的 Dice 与论文不可比；paper-like 复现应改为对照"约 10 次迭代收敛 + 计算时间量级 + 定性可视化优于 K-means"，而非凑 Dice。

**达到 paper-like 的最小充分条件**：接入 S2LET 球面小波栈（至少 axisymmetric + directional）→ 用真实 Earth/solar/retina 之一 → 实现 WSSA 区间迭代 → 复现"约 10 次迭代收敛 + WSSA 优于 K-means"的定性趋势与时间量级。paper-level 还需严格匹配采样约定、$\epsilon/\lambda$ 参数、hybrid 的 $L_{\text{trans}}\in\{32,64\}$ 设置，并复现表格中的 $|\Lambda^{(i)}|$ 数值。

---

## 9. 运行步骤

**当前 toy 跑法**

```bash
# 安装依赖（见 reproStructured.dependencies）
pip install -r requirements.txt   # 关键：numpy, scipy, matplotlib

# 运行全部复现实验（含本篇 sphere_wavelet_toy）
cd reproduce && python run_all.py
```

- 依赖缺失时 runner 写入 `status=skipped`，不会伪造 completed（符合项目纪律）。
- 产物：`assets/repro/sphere_wavelet_toy.png` 与结果 JSON 中的 `runMetrics`。

**向 paper-like 扩展的步骤大纲（不在当前 toy 范围内）**

1. 安装球面栈：S2LET（依赖 SSHT、SO3），确认 `pyssht` 等角采样约定与 $L=512$。
2. 下载一类真实数据（建议先 Earth EGM2008 或 DRIVE retina），按论文方式 band-limit / 投影。
3. 实现式 2.5 离散球面梯度与式 3.1–3.10 的 WSSA 迭代，$\bar\lambda=\sigma/4$、$\lambda=\sigma/100$，$\epsilon$ 按数据取（论文 Earth 用 0.02、light probe 0.05、Solar map 1 太阳耀斑 0.04、Solar map 2 活动磁区 0.05、retina 两张图 Fig 4.5/4.6 均 0.04）。
4. 接入 K-means 对照，记录 $|\Lambda^{(i)}|$ 序列与计算时间，与 Table 4.1–4.4 趋势比对。
5. 全程在图注/JSON 中标注真实等级（partial / paper-like），严禁夸大。

---

## 10. 风险与代理说明

- **Gaussian smoothing ≠ spherical wavelet denoising**：proxy 抹平噪声但**不具备**方向/曲线选择性，因此 toy 完全无法体现论文 WSSA-D/WSSA-H 相对 WSSA-A 与 K-means 的核心优势——这些结论**不可**从 toy 外推。
- **equirectangular 近似 ≠ $\mathbb{S}^2$ 几何**：极点附近 $1/\sin\theta$ 加权被 `max(cos(lat),0.2)` 截断，几何失真在高纬被人为压制；真实球面拓扑（无平面平移不变性、极点奇异）未被建模。
- **单步阈值 ≠ 区间收缩迭代**：toy 没有 $[a_i,b_i]$ shrinkage，无法复现"约 10 次迭代收敛"这一论文核心可验证现象。
- **Dice 0.8418 的语义**：仅是 toy 在自造 truth 上的内部一致性度量，与论文无任何可比性（论文不报告 Dice）。**不得**将其表述为"球面小波分割论文级指标"。
- **可外推的有限结论**：toy 仅能说明"球面几何加权的梯度 + 平滑能在合成 band 图上给出合理分割"这一最弱命题。

---

## 11. 参考：精读笔记

- 本篇精读笔记：[`../../../xiaohao_cai_ultimate_notes/Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md)
- 关联前作（tight-frame 思想来源）：Framelet/Tight-frame Vessel Segmentation、SaT/SLaT 系列（见笔记"论文关系"小节）。
