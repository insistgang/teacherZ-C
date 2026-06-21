# SaT 分割方法论总览 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 1 篇 *An Overview of SaT Segmentation Methodology and Its Applications in Image Processing* 的完整复现流程规范。

---

## 1. 论文身份与第一作者核验

| 项 | 内容 |
|----|------|
| **标题 (EN)** | An Overview of SaT Segmentation Methodology and Its Applications in Image Processing |
| **标题 (CN)** | SaT 分割方法论总览及其在图像处理中的应用 |
| **作者顺序** | **Xiaohao Cai**, Raymond Chan, Tieyong Zeng |
| **第一作者核验** | 是。PDF 首页（p.1385）作者列表以 **Xiaohao Cai** 开头，单位为 University of Southampton (School of Electronics and Computer Science)，邮箱 `x.cai@soton.ac.uk`。确认 Xiaohao Cai 为第一作者。 |
| **年份 / 出处** | 2023，Springer *Handbook of Mathematical Models and Algorithms in Computer Vision and Imaging*，Chapter 40，pp. 1385-1409，DOI 10.1007/978-3-030-98661-2_75 |
| **类型** | 综述 / 方法论章节 (overview / handbook chapter)，非单一新算法论文 |
| **PDF 路径** | `docs/00_papers_first_author_xiaohao_cai_deduped/分割方法论总览 SaT Overview.pdf` |
| **主题 (theme)** | `sat-rof`（SaT 方法论与 ROF 平滑主线，本仓库 15 篇的入口论文） |

这是整组 15 篇的**入口/索引论文**：它把 smoothing and thresholding (SaT) 作为统一方法论框架，组织 T-ROF、SLaT、Poisson/Gamma 噪声、hyperspectral、tight-frame vascular、spherical wavelet、intensity inhomogeneity 等多条分支。它本身不证明所有定理，证明细节散落在被引用的原始论文中。

---

## 2. 复现目标与诚实分级

本项目对"复现"采用四级诚实分级（由弱到强）：

| 级别 | 含义 |
|------|------|
| **toy** | 合成小图 + 代理算子，演示直觉，不对齐论文任何具体数值 |
| **partial** | 实现了论文核心步骤的一部分（如真实 ROF 求解器 + 阈值迭代），在合成数据上验证趋势，但未对齐论文数据集与报告数值 |
| **paper-like** | 用论文同款或公开等价数据集，跑论文同款 pipeline，复现论文表格量级（不要求逐位一致） |
| **paper-level** | 严格复现论文报告数值（同数据、同基线、同指标、同表号） |

**本仓库当前等级（reproductionLevel）= `partial`；真实性（reproductionTruthLevel）= `partial-completed`。**

纪律红线：
- **paper-level 在 15 篇中仍为 0/15。** 本篇也不例外。
- 当前实现的 SaT 主路径已切换为**真实凸 ROF 求解器（Chambolle-Pock 原始-对偶投影，`rof_chambolle_pock`）**：先解 ROF 得 g，再对 g 做 K-means 阈值（见 §7、§10）。Gaussian smoothing 仅保留为对照 baseline（`gaussian_baseline_accuracy`），不再是 headline 指标的来源。仍仅在合成 multiphase 图上跑：`runMetrics` 中的 `sat_accuracy=0.9961`（真实 ROF）等数值是 toy 合成图上的结果，**不得**被表述为论文级或论文报告的精度。
- 本篇是综述，paper-level 复现意味着覆盖其下属**多个分支**（T-ROF / SLaT / 高光谱 / vascular / spherical / intensity inhomogeneity）。当前仅触及 SaT 两段式骨架 + T-ROF 阈值迭代（与第 2、3 篇 runner 共用 `sat_rof_trof.py`）。

---

## 3. 算法完整流程

SaT 方法论的核心是**两段式分解**：先 smoothing（解凸目标得稳定中间图像 g），再 thresholding（把 g 分成 K 相）。关键在于**类别数 K 只进入第二段**，改 K 不必重解第一段。

### 3.1 经典模型背景（论文 Introduction）

- **Mumford-Shah (MS)** 能量，Eq.(1)：
  `E_MS(u, Γ; Ω) = H¹(Γ) + λ' ∫_{Ω\Γ} |∇u|² dx + λ ∫_Ω (u-f)² dx`
  三项分别是边界长度惩罚、`Ω\Γ` 上的 H¹ 光滑项、数据保真项。**非凸、非光滑**。

- **Piecewise Constant MS (PCMS)**，Eq.(2)(3)（限制 `∇u=0` 于 `Ω\Γ`）：
  `E_PCMS(Ω, m) = ½ Σ_{i} Per(Ω_i; Ω) + λ Σ_i ∫_{Ω_i} (m_i - f)² dx`
  当 K=2 时退化为 **Chan-Vese** 模型 Eq.(4)。仍非凸，易陷局部极小。

- **ROF**，Eq.(7)：
  `min_{u∈BV(Ω)} { TV(u) + (μ/2) ∫_Ω (u-f)² dx }, μ>0`
  **凸**，可高效求解。SaT 的关键洞察：把 ROF/TV 恢复当作分割的第一段。

### 3.2 SaT 第一段：Smoothing（论文 Eq.(8)，核心公式）

```
inf_{g ∈ W^{1,2}(Ω)}  { (μ/2) ∫_Ω (f - Ag)² dx           ← 数据保真项 (A 为模糊算子或恒等)
                       + (λ/2) ∫_Ω |∇g|²  dx              ← H¹ 半范光滑项
                       +        ∫_Ω |∇g|   dx }            ← TV 正则项 (保证水平集正则性)
```

- `A`：观测算子；有模糊则为卷积模糊算子，无模糊则为恒等。
- 论文强调 Eq.(8) **严格凸**，可用 **split-Bregman (Goldstein & Osher 2009)** 或 **Chambolle-Pock (Chambolle & Pock 2011)** 快速求解。
- **Theorem 1**（论文 p.1389）：Ω 为有 Lipschitz 边界的有界连通开集，`f∈L²(Ω)`，`Ker(A)∩Ker(∇)={0}`，则 Eq.(8) 在 `W^{1,2}(Ω)` 中**存在唯一最小解 g**。

### 3.3 SaT 第二段：Thresholding（论文 p.1389）

给定阈值 `min{g}=ρ₀<ρ₁<…<ρ_{K-1}<ρ_K=max{g}`，令 `x∈Ω_i` 当且仅当 `ρ_{i-1} ≤ g(x) < ρ_i`。阈值 `{ρ_i}` 可由：
1. **K-means** 对 g 的灰度做聚类（论文默认）；或
2. 人工试错 / 交互调节得到更精细分割。

**K 只在此步出现 → 改 K 不重解 Eq.(8)，这是 SaT 相对 PCMS 的核心效率优势。**

### 3.4 可执行 step-by-step pipeline（两相 / 多相通用）

1. 读入退化图像 f（噪声 / 模糊 / 信息缺失）。
2. 选保真项：高斯噪声用 L² 保真 Eq.(8)；Poisson 噪声用 Eq.(10) `∫(g - f log g) + β∫|∇g|`；Gamma 乘性噪声用 Eq.(12)(13)（对数变量 `w=log g`）。
3. 用 split-Bregman 或 Chambolle-Pock 求解凸 smoothing 得 g。
4. 归一化 g。
5. 用 K-means / 阈值得到 K 相分割。
6. 若需改 K，只重做步骤 5。

### 3.5 T-ROF 分支（论文 p.1392, Theorem 2）

T-ROF (thresholded-ROF, Cai & Steidl 2013; Cai et al. 2019) 把阈值自动选取**理论化**：

- **Theorem 2 (K=2, ROF–PCMS/Chan-Vese 关系)**：设 K=2，`u*∈BV(Ω)` 是 ROF Eq.(7) 的解。给定 `0<m₀<m₁≤1`，令 `Σ̃ := {x∈Ω : u*(x) > (m₁+m₀)/2}` 满足 `0<|Σ̃|<|Ω|`。则 **Σ̃ 是 PCMS/Chan-Vese 模型 Eq.(4) 在 λ := μ/(2(m₁-m₀))、固定 m₀,m₁ 下的最小化子**；进一步若 `m₀=mean_f(Ω\Σ̃)` 且 `m₁=mean_f(Σ̃)`，则 `(Σ̃,m₀,m₁)` 是 Eq.(4) 的 **partial minimizer**。
- 意义：ROF 解 + 阈值 ⇒ 分割，且阈值选取有 PCMS 最优性支撑；ROF 与 PCMS 都**只需解一次**。K>2 在特定假设下仍成立（细节回 Cai et al. 2019）。

### 3.6 其它分支（论文 SaT-Based Methods and Applications，仅入口）

- **Poisson/Gamma 噪声**（Chan et al. 2014）：换保真项 Eq.(10)-(13)；Theorem 3/4 保证 Eq.(13) 唯一解（恒等或模糊算子）。
- **SLaT 彩色**（Cai et al. 2017）：① 对 RGB 三通道分别解 Eq.(8) 得 (g₁,g₂,g₃)；② lifting 到 Lab 得 (ḡ₁,ḡ₂,ḡ₃) 以降通道相关性；③ 对 6 维 (g₁,g₂,g₃,ḡ₁,ḡ₂,ḡ₃) 做 K-means。
- **Hyperspectral**（Chan et al. 2020）：① SVM 得每类概率图 f_k；② 对每个概率图解带训练像素约束的 Eq.(14) `s.t. g_k|_{Ω_train}=f_k|_{Ω_train}`；③ `Label(x)=argmax_k g_k(x)`。
- **Tight-frame vascular**（Cai et al. 2011, 2013a）：迭代 Eq.(15)-(18)，`f^{(i+1/2)}=U(f^{(i)})`、`f^{(i+1)}=A^T T_λ(A f^{(i+1/2)})`，软阈值 Eq.(17)，有限步收敛到二值血管分割。
- **Spherical wavelet**（Cai et al. 2020；正文 p.1403 "In Cai et al. (2020) ... on the sphere"，参考文献 Cai, Wallis, Chan, McEwen, Pattern Recogn. 100, 2020，第一作者为 Cai，勿与上一行高光谱分支的 Chan et al. 2020 混淆）：tight-frame 法在球面上的推广，可用 axisymmetric / directional / curvelet 球面小波。
- **Intensity inhomogeneity 三段法**（Li et al. 2020）：① lifting 加入不均匀图作为额外通道；② 对各通道用 sPADMM 解 SaT 平滑（Q-linear 收敛）；③ 阈值化。

---

## 4. 完整复现所需数据集

论文按分支使用不同数据。下表给出**论文实际命名的数据**与**为达 paper-like 的公开/等价候选**。

| 分支 | 论文使用数据（PDF 实证） | 公开 / 等价候选 | 备注 |
|------|--------------------------|-----------------|------|
| T-ROF retina | **DRIVE** 视网膜数据集（PDF 给出链接 `isi.uu.nl/Research/Databases/DRIVE`），单张人工分割图改三相（背景0、左血管1→右血管0.3），加高斯噪声 mean 0 / variance 0.1 | DRIVE 公开可下载 | 三相、极细血管，挑战性高 |
| 4-phase 合成 | 论文 Fig.3 合成 256×256 四相图，高斯噪声 variance 0.03 | 可自行合成（本仓库已合成 96×96 四相） | 阈值 ρ₁=0.1652, ρ₂=0.4978, ρ₃=0.8319 |
| SLaT 彩色 | **Berkeley Segmentation Dataset (BSDS)**（PDF 链接 `eecs.berkeley.edu/.../bsds`） | BSDS300/BSDS500 公开 | 退化彩色图（噪声/模糊/60% 信息缺失） |
| Hyperspectral | **Indian Pines** 高光谱数据集 | Indian Pines 公开（Purdue / EHU GIC） | 10% 训练像素 |
| Spherical | **Uffizi Gallery** light probe 图（Fig.10） | 公开 light probe 库 | 球面 + Mollweide 投影 |
| Vascular tight-frame | kidney / brain volume MRA / CTA（Fig.8/9） | **私有/受限医学数据**，需自备等价 MRA/CTA | 标注医学数据通常不公开 |
| Intensity inhomogeneity | **Alpert's dataset**（300×225，Fig.11） | Alpert 公开 | 单通道 + 不均匀图 |

> 私有/受限医学数据（vascular MRA/CTA）是 paper-level 的主要数据障碍，需在仓库内显式标注无法获取并用公开等价数据替代。

---

## 5. 对照基线 (Baselines)

论文各分支对照方法（PDF 图注实证），可作复现基线：

| 分支 | 论文对照基线 |
|------|--------------|
| 两相 / 四相 | Chan-Vese (Chan & Vese 2001)，Li et al. 2010，Sandberg et al. 2010，Yuan et al. 2010b |
| Retina T-ROF | Li et al. 2010，Pock et al. 2009a，Yuan et al. 2010b，He et al. 2012，Cai et al. 2013b（SaT）vs Cai et al. 2019（T-ROF） |
| Gamma 噪声 | Yuan et al. 2010a，Dong et al. 2011 |
| SLaT 彩色 | Pock et al. 2009a |
| Hyperspectral | 纯 SVM（无空间信息）vs SaT 两段法 |
| Vascular | CURVES (Lorigo et al. 2001)，ADA (Franchini et al. 2009/2010) |
| Spherical | K-means vs WSSA-A / WSSA-D / WSSA-H（三种球面小波） |
| Intensity inhomogeneity | Cai et al. 2017，Li et al. 2010/2020，Zhi & Shen 2018，Wang et al. 2009，**U-net (Ronneberger et al. 2015)**（深度学习对照） |

合理的最小对照（本仓库 toy 层）：**direct K-means（不平滑）** vs **SaT（先平滑后阈值）**，以及 Chambolle-Pock ROF vs Gaussian proxy 的差异图。

---

## 6. 评价指标与论文报告结果

### 6.1 指标定义
- **Overall accuracy / pixel accuracy**：分类正确像素占比（高光谱、合成多相用）。
- **Dice**：分割重叠度（两相、血管用）。
- **定性**：分割边界是否贴合真实边界（Fig.1/5/8/9 用黄/红线对比）。
- **效率**：CPU 时间（论文反复强调 SaT/T-ROF 速度快、与 K 无关）。

### 6.2 论文报告的关键数值（PDF 能确认者，注明出处）
- **Indian Pines 高光谱分类，overall accuracy = 98.83%**（10% 训练像素），来源 **Fig.7 图注**（Chan et al. 2020）。这是 PDF 中**唯一明确给出的量化精度数字**。
- **Fig.3 四相阈值**：ρ₁=0.1652, ρ₂=0.4978, ρ₃=0.8319；噪声 variance 0.03（Fig.3 图注）。
- **DRIVE retina**：噪声 mean 0 / variance 0.1；三相强度 0、1→0.3（正文 p.1393）。

> 除上述外，本综述其余分支（SLaT、vascular、spherical、intensity inhomogeneity）在本章中**以定性图示为主，未在本章正文给出可引用的逐表数值**；具体数表需回各原始论文核对。**禁止编造任何未在 PDF 出现的数字。**

---

## 7. 本仓库当前复现实现

- **runner 文件**：`reproduce/experiments/sat_rof_trof.py`（与第 2 篇 pcms-rof-linkage、第 3 篇 iterated-rof 共用同一 runner）。
- **它实际做了什么**：
  - 合成两相圆形图、两套四相图（含"close gray"近强度四相，levels≈[0.28,0.32,0.36,0.40]，更难）。
  - **SaT 主路径 = 真实凸 ROF**：本篇四相图先用 **Chambolle-Pock ROF**（`rof_chambolle_pock`，μ=8.0，240 步，tol 2e-5）求解 g，再对 g 做 K-means（仓库内 `simple_kmeans`）阈值得四相分割。`sat_accuracy` 即此真实-ROF 路径的像素精度。
  - **Gaussian smoothing 仅作对照 baseline**（`scipy.ndimage.gaussian_filter`，sigma=1.0），其 K-means 精度记为 `gaussian_baseline_accuracy`，不再是 headline 指标。
  - 同 runner 另含 **Split-Bregman ROF**（`rof_split_bregman`）交叉验证、**T-ROF 阈值迭代**（`run_trof_thresholds`，按 Eq.(15) 用 `mean_f(Ω_i)` 更新阈值并检查 Lemma 2/3、Assumption A、漂移收敛）、**K=2 Proposition/Theorem 2 代理检查**（`run_k2_proposition_demo`，导出 `λ = μ/(2(m₁-m₀))`），及 **Multi-Otsu** 基线（`skimage.filters.threshold_multiotsu`）——这些主要服务第 3 篇。
  - 产图：`assets/repro/sat_demo.png`（本篇 resultFiles），及 `trof_thresholds.png`、`iterated_rof_convergence.png`、`iterated_rof_chanvese.png`（第 3 篇用）。
- **本篇当前 runMetrics（来自 runner 实测，确定性可复现）**：

  | 指标 | 数值 | 含义 |
  |------|------|------|
  | `direct_accuracy` | 0.6590 | direct K-means（不平滑）在四相 toy 图的像素精度 |
  | `gaussian_baseline_accuracy` | 0.9799 | Gaussian smoothing + K-means（**对照 baseline**）的像素精度 |
  | `sat_accuracy` | 0.9961 | **真实 ROF（Chambolle-Pock）+ K-means** 主路径的像素精度 |
  | `accuracy_gain` | 0.3371 | sat_accuracy 相对 direct 的提升（真实 ROF 路径） |
  | `runtimeSeconds` | ≈0.77 | CPU 运行时间（约 1 秒内） |

  > 注意：上述是**合成 toy 图**结果，演示"先解真实 ROF 再阈值比直接阈值（及比 Gaussian 平滑）更稳"的趋势，**不是**论文报告值。真实 ROF（0.9961）> Gaussian 对照（0.9799）> direct（0.6590）。
- **resultFiles**：`assets/repro/sat_demo.png`。
- **fidelity 警告（runner `extra` 已记录）**：`Real ROF (Chambolle-Pock) on a synthetic toy four-phase image; no blur operator A, no H1 term, and no paper dataset/baseline. Covers only the SaT skeleton (one of many SaT branches).`

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

**到 paper-like 的缺口清单：**

1. **求解器对齐（已部分完成）**：headline `sat_accuracy` 已切换为真实 Chambolle-Pock ROF 主路径（Gaussian 仅作对照 baseline）。**仍缺**：完整实现 Eq.(8) 的**模糊算子 `A`** 与 **H¹ 半范光滑项**（当前仅纯 ROF = TV + 数据项，无 A、无 H¹），并对齐论文用的 split-Bregman/ADMM 求解器实现细节与参数。
2. **真实数据**：当前仅合成图。需接入 DRIVE（retina）、BSDS（SLaT）、Indian Pines（高光谱）、Alpert（intensity inhomogeneity）等公开数据。
3. **分支覆盖**：当前只覆盖 SaT 骨架 + T-ROF 两相/四相。SLaT、Poisson/Gamma、hyperspectral SVM+SaT、tight-frame vascular、spherical wavelet、intensity inhomogeneity sPADMM **均未实现**。
4. **基线缺失**：未实现 Chan-Vese、Pock 2009a、Yuan 2010b、CURVES/ADA、U-net 等论文对照方法。
5. **指标/表格对照**：未对齐论文 Fig.3 阈值（0.1652/0.4978/0.8319）、未尝试复现 Indian Pines 98.83% 这一可对照数字。
6. **退化算子**：未实现 motion blur（Fig.1 filter size 15）、信息缺失 60%（Fig.6）等论文退化设置。

**到 paper-level 的额外缺口：**

7. 私有/受限医学 MRA/CTA 数据无法获取，需显式标注并用公开等价数据。
8. 需严格对齐论文每个分支的数据划分、参数、随机种子，逐表复现——本综述涉及多篇原始论文，工作量等于复现一组论文而非一篇。

---

## 9. 运行步骤

### 9.1 当前 toy/partial 跑法

```bash
# 安装依赖（见下）
pip install -r requirements.txt

# 运行全部复现实验（含本篇 sat_rof_trof）
cd reproduce && python run_all.py

# 或在仓库根校验 15 篇数据 / PDF / 笔记 / 静态复现资产
node docs/scripts/validate.mjs
```

- **依赖**（来自 `reproStructured.dependencies`）：`numpy`、`scipy`、`matplotlib`（runner 中 K-means 用仓库内 `simple_kmeans`，未强依赖 sklearn）。
- **算力**：CPU，约 1 秒内（`runtimeSeconds≈0.71`）。
- **数据**：合成 4 相退化图，**无需下载真实数据**。
- 缺依赖时 runner 写 `skipped`，**不伪造 completed**（遵守 CLAUDE.md 纪律）。

### 9.2 向 paper-like 扩展的步骤大纲

1. （已完成）本篇 headline metrics 的 smoothing 已从 Gaussian proxy 切换为真实 ROF（`rof_chambolle_pock`）；**下一步**加入模糊算子 A 与 H¹ 项以补全 Eq.(8)。
2. 新增 `reproduce/data/` 数据接入脚本：下载并预处理 DRIVE / BSDS / Indian Pines / Alpert。
3. 逐分支补实现：先 retina T-ROF（对齐 Fig.4 设置），再 SLaT（6 维 Lab K-means），再 hyperspectral（SVM 概率图 + Eq.(14) 约束 SaT，目标对照 98.83%）。
4. 加入论文基线（至少 Chan-Vese、direct K-means、纯 SVM）做并排对照表。
5. 在 dashboard 中把每个分支的 `reproductionLevel` 独立标注，避免用单一 toy 数字代表整篇综述。

---

## 10. 风险与代理说明

- **真实 ROF 已上主路径，但仍非完整 Eq.(8)**：headline `sat_accuracy=0.9961` 来自真实 Chambolle-Pock ROF（TV + L² 数据项），已不再用 Gaussian 代理；但它**仍缺** Eq.(8) 的模糊算子 `A` 与 H¹ 半范项，故只能说明"先解 ROF 再阈值更稳"的**定性**趋势，不能外推为 SaT 的论文级精度或 Theorem 1/2 的数值验证。Gaussian baseline（0.9799）仅作对照，**不**等价于凸 minimizer。
- **合成数据的局限**：toy 四相图不含真实退化（motion blur、Poisson/Gamma、信息缺失、强度不均匀、彩色通道相关性、球面几何），故无法反映论文在真实数据上的鲁棒性优势。
- **共享 runner 的口径**：本篇与第 2、3 篇共用 `sat_rof_trof.py`；本篇 metrics（`sat_accuracy` 等）现已是真实 ROF 口径，与第 3 篇 (iterated-rof) 的 T-ROF 阈值迭代 metrics 各自独立，阅读时按 id 区分。
- **不可外推的结论**：① 不能说本仓库"复现了"任一分支的论文结果；② 不能把 toy 精度等同于论文报告精度；③ paper-level 在 15 篇中仍为 0/15，本篇亦然。

---

## 11. 参考：精读笔记

完整中文精读笔记见：
[`../../../xiaohao_cai_ultimate_notes/分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md)
