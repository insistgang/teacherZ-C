# Tight-frame 医学血管分割长版 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 6 篇 *Vessel Segmentation in Medical Imaging Using a Tight-Frame Based Algorithm* 的完整复现流程规范。

## 1. 论文身份与第一作者核验

| 字段 | 内容 |
| --- | --- |
| 英文标题 | Vessel Segmentation in Medical Imaging Using a Tight-Frame Based Algorithm |
| 作者顺序 | **Xiaohao Cai**, Raymond Chan, Serena Morigi, Fiorella Sgallari |
| 第一作者核验 | 是。PDF 首页作者列表第一位即 Xiaohao Cai（蔡晓昊），与 dashboard 字段一致。 |
| 年份 | arXiv 预印本 2011（arXiv:1109.0217v1 [math.NA]，2011-08-13 提交，extended version，本仓库 PDF）/ 正式期刊版 2013（SIAM J. Imaging Sci. 6(1):464-486, 2013，即 dashboard 与 sat-overview/sphere-wavelet 笔记引用的 "Cai et al. 2013a"） |
| 出处 | extended 版正式发表于 SIAM Journal on Imaging Sciences (SIIMS) 6(1):464-486, 2013；本仓库 PDF 为其 arXiv:1109.0217v1 预印本（2011）；会议精简版为 X.H. Cai, R.H. Chan, S. Morigi, F. Sgallari, "Framelet-Based Algorithm for Segmentation of Tubular Structures", SSVM 2011, LNCS6667, Springer（PDF 参考文献 [9]） |
| PDF 路径 | `docs/00_papers_first_author_xiaohao_cai_deduped/框架分割管状结构 Framelet Tubular.pdf` |
| 主题 | medical（医学影像血管/管状结构分割） |
| 单位 | Department of Mathematics, CUHK（Cai, Chan）；Department of Mathematics-CIRAM, University of Bologna（Morigi, Sgallari） |

核验依据：PDF 第 1 页标题下作者行明确为 "Xiaohao Cai, Raymond Chan, Serena Morigi, Fiorella Sgallari"；脚注给出 CUHK 邮箱 `xhcai@math.cuhk.edu.hk`。第一作者身份确凿。

## 2. 复现目标与诚实分级

本项目对每篇论文区分四个等级：

- **toy**：合成数据上验证核心机制（如 Λ 候选集收缩、有限收敛现象），不追求论文数值。
- **partial**：用接近论文的真实/公开数据跑通主要 pipeline 的一部分，但简化求解器或基线。
- **paper-like**：用论文同源或等价的公开数据 + 论文级算法实现（此处即真正的 tight-frame / DℂWT 变换），复现论文报告的定性结论与可比指标。
- **paper-level**：严格复现论文全部实验设置（同数据、同基线、同参数、同表格），数值与论文一致。

**本仓库当前等级**：`reproductionLevel = toy`，`reproductionTruthLevel = toy-completed`。

**纪律声明**：截至当前，本项目 15 篇论文 paper-level 复现仍为 **0/15**。本篇的 toy 复现仅在合成 2D 血管网络上演示 Λ 候选集合的有限收缩与最终二值化，**禁止**把 toy 的 Dice/IoU（见 §7）解释为论文在真实 2D/3D MRA 上的性能。本篇 toy 实现用 **Gaussian smoothing 代替了论文真正的 tight-frame / DℂWT 迭代**（见 §7、§10），这一代理决定了它最多只能说明"候选集收缩 + 局部平滑"的骨架逻辑，而非论文的方向选择性细节提取能力。

## 3. 算法完整流程

论文方法（PDF Section III, Algorithm 1）不基于任何变分泛函最小化，而是迭代地把可能边界像素集合 Λ 收缩、并把图像逐步推向二值。设图像 `f` 动态范围已归一到 `[0,1]`，`Ω` 为全部像素索引集。

**核心思想**：MRA 图像中血管为高强度区域；血管边界附近的像素值落在某个区间内，而非边界像素则远离该区间。算法迭代地用 tight-frame 去噪平滑去精确逼近这个区间，并把确定属于背景/血管的像素固定为 0/1。

**Step 0 — 初始化 (PDF eq.6)**

1. 令 `f^(0) = f`（原图）。
2. 用梯度阈值确定初始可能边界集合：
   `Λ^(0) ≡ { j ∈ Ω : ‖[∇f]_j‖_1 ≥ ε }`，
   其中 `[∇f]_j` 为第 `j` 像素处离散梯度，`ε` 为阈值（论文取 2D 时 ε=0.003，3D 时 ε=0.06）。

**Step (i) — 计算区间 [α_i, β_i] (PDF eq.7–10)**

3. 在当前 `Λ^(i)` 上算均值：`μ^(i) = (1/|Λ^(i)|) Σ_{j∈Λ^(i)} f_j^(i)` (eq.7)。
4. 以 `μ^(i)` 把 `Λ^(i)` 分两半，分别算偏背景侧与偏血管侧均值：
   `μ_-^(i) = mean{ f_j^(i) : j∈Λ^(i), f_j^(i) ≤ μ^(i) }` (eq.8)，
   `μ_+^(i) = mean{ f_j^(i) : j∈Λ^(i), f_j^(i) ≥ μ^(i) }` (eq.9)。
5. 形成区间端点 (eq.10)：
   `α_i = max{ (μ_-^(i)+μ_+^(i))/2 , 0 }`，`β_i = min{ (μ_-^(i)+μ_+^(i))/2 , 1 }`。
   （注：论文 eq.10 两式右端均为 `(μ_-+μ_+)/2` 再分别与 0、1 取 max/min，使 `[α_i,β_i]⊆[0,1]`。）

**Step (ii) — 三段阈值 + 线性拉伸 (PDF eq.11–13)**

6. 在区间内像素的极值：
   `M_i = max{ f_j^(i) : α_i ≤ f_j^(i) ≤ β_i, j∈Λ^(i) }`，
   `m_i = min{ f_j^(i) : α_i ≤ f_j^(i) ≤ β_i, j∈Λ^(i) }` (eq.11)。
7. 三段映射得 `f^(i+1/2)` (eq.12)，对所有 `j∈Ω`：
   - 若 `f_j^(i) ≤ α_i` → 0（背景）；
   - 若 `α_i ≤ f_j^(i) ≤ β_i` → `(f_j^(i) − m_i)/(M_i − m_i)`（线性 contrast stretch 到 [0,1]）；
   - 若 `β_i ≤ f_j^(i)` → 1（血管）。
8. 更新候选集 (eq.13)：`Λ^(i+1) = { j : 0 < f_j^(i+1/2) < 1, j∈Ω }`。被映射到 0/1 的像素离开候选集，不再处理。

**Step (iii) — tight-frame 迭代去噪平滑 (PDF eq.14)**

9. 只在 `Λ^(i+1)` 上做 tight-frame 去噪平滑。设 `P^(i+1)` 为对角矩阵（`Λ^(i+1)` 内对角元为 1，否则 0），`A` 为 tight-frame 前向变换（满足 perfect reconstruction `Aᵀ A = I`），`T_λ` 为软阈值算子：
   `f^(i+1) ≡ (I − P^(i+1)) f^(i+1/2) + P^(i+1) Aᵀ T_λ( A f^(i+1/2) )` (eq.14)。
   即候选集外的像素保持 0/1 不动，候选集内像素用 tight-frame 去噪平滑更新。

**Step (iv) — 停机判据**

10. 当 `f^(i+1/2)` 全为 0/1（等价 `Λ^(i) = ∅`）时停机，输出二值图像 `f^(i+1/2)`：值 1 的像素构成管状结构，值 0 为背景。

**软阈值与 tight-frame (PDF Section II)**：通用 tight-frame 算法形如 `f^(i+1/2)=U(f^(i))`，`f^(i+1)=Aᵀ T_λ(A f^(i+1/2))` (eq.3–4)；软阈值 (eq.5) `t_λ(v)=sgn(v)(|v|−λ)` 当 `|v|>λ`，否则 0。论文用两类 tight-frame：分段线性 B-spline tight-frame（滤波器 `h_0=¼[1,2,1], h_1=(√2/4)[1,0,−1], h_2=¼[−1,2,−1]`，eq.1）和 dual-tree complex wavelet transform（DℂWT，具备 ±15°/±45°/±75° 方向选择性）。实验中采用 DℂWT（见 §6）。

## 4. 完整复现所需数据集

论文实验数据（PDF Section IV）：

1. **Example 1**：182×182 的 2D carotid（颈动脉）vascular system MRA 图像，`|Ω|=11284`（注：表 I 标注该尺寸下 |Ω|，PDF 表 I 列出每轮 |Λ^(i)|）。来源关联 [23][24]（Franchini, Morigi, Sgallari 的 PDE 工作）。
2. **Example 2**：256×256 的 2D kidney（肾）vascular system MRA 图像，`|Ω|=66049 (=257²，论文标 66049)`。
3. **Example 3**：3D，从 436×436×540 的 CTA（Computed Tomographic Angiography）肾血管图像中抽取 201×201×201 体数据，`|Ω|=8120601`。
4. **Example 4**：3D，从 120×448×540 的 brain-neck MRA 中抽取 120×250×200 脑动脉瘤体数据，`|Ω|=6000000`。

**达到 paper-like 的公开/等价候选数据**（论文原始 MRA/CTA 数据未随论文公开，需替代）：

- 2D/3D 脑血管 MRA：**TubeTK**（UNC，Bullitt et al. 提供的脑 MRA + 血管标注）。
- 视网膜血管（2D 等价管状结构，便于验证细节提取）：**DRIVE**、**STARE**、**CHASE_DB1**、**HRF**（带专家标注，可直接算 Dice/IoU/敏感度，作为 2D paper-like 替代）。
- 冠脉/肝脏血管 3D：**Rotterdam Coronary CTA**、**3D-IRCADb**（肝血管 CT，含标注）、**VascuSynth**（合成可控血管树，便于参数与拓扑研究）。
- 若坚持复现论文 carotid/kidney/brain MRA-CTA，需向原作者或对应医院数据库申请**私有医学影像**，并通过伦理/数据使用协议。本仓库不内置任何真实患者数据。

**注意**：论文图像无逐像素 ground truth 标注（其评价以定性视觉对比为主，见 §6），因此用 DRIVE/STARE 等带标注公开集替代时，可获得论文未给出的定量指标，但这是"等价任务"而非"同一数据"，须如实标注。

## 5. 对照基线 (Baselines)

论文 Section IV 明确对照三类代表方法（程序由各自作者提供）：

- **PDE-based anisotropic diffusion**：Franchini, Morigi, Sgallari，[23]（SSVM/MMCS 2008/2010 的 PDE 各向异性扩散管状结构模型）；3D 例还对照 [24]（composed segmentation by anisotropic PDE model, SSVM 2009）。
- **Frame based segmentation for medical images**：Dong, Chien, Shen，[20]（UCLA Technical Report 2010，tight-frame + TV 的变分分割）。
- **Active contour without edges**：Chan-Vese，[16]（IEEE TIP 2001）。

paper-like 复现建议至少纳入：(a) Chan-Vese active contour（成熟开源实现众多），(b) 一种 PDE/各向异性扩散或 frangi vesselness + 阈值，(c) 一个现代深度基线（如 U-Net / nnU-Net，用于 DRIVE/STARE 等带标注集）作为上界参照。论文未用深度方法，深度基线仅作语境对照，不应替换论文原对照口径。

## 6. 评价指标与论文报告结果

**论文的评价方式**：以**定性视觉对比**为主（图 2–5 中把分割边界叠加在原图上比较细节、连通性、伪影去除），辅以**收敛性定量**（迭代次数、运行时间、每轮 |Λ^(i)| 收缩，PDF Table I）。论文**未报告** Dice/IoU/敏感度等逐像素指标。

可从 PDF 直接确认的关键数值（来源标注于括号）：

| 项目 | 数值 | 来源 |
| --- | --- | --- |
| 收敛迭代上限（实测） | 通常 ≤10 次迭代（2D 与 3D 均如此） | Abstract；Section III；Section IV |
| Example 1 收敛 | 5 次迭代，0.64 秒 | Section IV, Example 1 |
| Example 2 收敛 | 6 次迭代，0.78 秒 | Section IV, Example 2 |
| Example 3 (3D) 收敛 | 9 次迭代 | Section IV, Example 3 |
| Example 4 (3D) 收敛 | 9 次迭代 | Section IV, Example 4 |
| 软阈值参数 | λ_k ≡ 0.1 | Section IV |
| 梯度阈值 ε | 2D: 0.003；3D: 0.06 | Section IV |
| tight-frame | DℂWT（[14] 下载），小波层数 4，默认参数 | Section IV |
| 复杂度 | 每轮 O(n)，n 为像素/体素数 | Section III（μ、μ_±、α/β 均 O(n)，tri-diagonal 滤波 O(n)） |
| Table I 收缩示例 | 如 Example 1：i=0:1721 → i=1:354 → i=2:82 → i=3:26 → i=4:4 → i=5:0 | PDF Table I |
| Table I (Example 3, 3D) | i=0:137330 → i=1:32760 → ... → i=8:3 → i=9:0 | PDF Table I |
| 算力 | 2D：2.4GHz/4GB MacBook；3D：120GB RAM 集群节点 | Section IV |

**禁止编造**：论文未给 Dice 数字，因此任何 Dice/IoU 数值（包括本仓库 toy 的 0.9981）都不能写成"论文报告值"。paper-like 阶段若在 DRIVE/STARE 上计算指标，须注明是"等价任务上的复现结果"。

## 7. 本仓库当前复现实现

- **runner 文件**：`reproduce/experiments/tubular_tight_frame.py`（该 runner 同时产出 priority 5 framelet-tubular 与 priority 6 tight-frame-vessel 两条 toy 记录，共用同一合成实验与图像）。
- **它实际做了什么**：
  1. 用 `skimage.draw.line` + `skimage.morphology.dilation(disk(3))` 构造一个合成 2D 血管网络 mask（112×112，4 条线段血管），加 Gaussian 噪声得 `image`。
  2. 用固定区间 `[alpha, beta]=[0.38, 0.62]` 近似论文 `[α_i, β_i]`，迭代 12 次：取 `uncertain = (current>alpha)&(current<beta)` 作为 Λ；记录 |Λ|。
  3. **代理**：在 uncertain 区域用 `scipy.ndimage.gaussian_filter(sigma=1.0)` 代替论文真正的 tight-frame/DℂWT 软阈值去噪 (eq.14)。
  4. 把 `>=beta` 且预测为前景的像素固定为 1，`<=alpha` 且预测为背景的固定为 0；每轮把 `alpha += 0.008`、`beta -= 0.008` 收紧区间，模拟 Λ 收缩。
  5. 阈值 0.5 得二值预测，算 Dice / IoU；画四联图（noisy tube / truth / toy output / Lambda size 曲线）。
- **当前 runMetrics（来自 reproStructured，非论文值）**：

| 指标 | 数值 |
| --- | --- |
| dice | 0.9981 |
| iou | 0.9962 |
| lambda_initial | 651 |
| lambda_final | 2 |
| iterations | 12 |
| runtimeSeconds | 0.076 |

- **结果图**：`assets/repro/tubular_lambda_shrinkage.png`。
- **诚实说明（runner 内 notes 原文）**：toy 复现，Λ 收缩与有限收敛模式在合成 2D 血管网络上演示；Dice 仅在简单合成 2D toy 上测量，**不代表真实 2D/3D MRA 论文级性能**。

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

清单（按缺口类别）：

1. **求解器（最关键）**：当前用 Gaussian smoothing 代替 tight-frame。需实现真正的 tight-frame 前向/逆变换：分段线性 B-spline tight-frame（滤波器 eq.1）或 DℂWT（论文实测用 DℂWT，4 层），并在 Λ 上做软阈值 (eq.5) 去噪平滑 (eq.14)。这是从 toy 到 paper-like 的核心鸿沟。
2. **区间计算**：当前用固定 `[0.38,0.62]` + 线性收紧，未实现论文的 `μ/μ_-/μ_+ → [α_i,β_i]` 自适应计算 (eq.7–10) 与 `m_i/M_i` contrast stretch (eq.11–12)。需替换为论文公式。
3. **初始化**：当前 Λ^(0) 由固定区间隐式给出，未用梯度阈值 `‖∇f‖_1 ≥ ε` (eq.6)。需按 eq.6 实现，并暴露 ε 参数（2D 0.003 / 3D 0.06）。
4. **数据**：当前为单张 112×112 合成图。需接入真实/公开数据（DRIVE/STARE/TubeTK/IRCADb 或论文私有 MRA-CTA），并支持 2D 与 3D 体数据。
5. **3D 支持**：论文核心贡献含 3D MRA/CTA（Example 3/4，百万级体素）。当前完全无 3D。需实现 3D DℂWT、3D 梯度、`isosurface` 可视化与内存优化（论文 3D 用 120GB RAM 节点）。
6. **基线对照**：当前无任何 baseline。需接入 Chan-Vese active contour [16]、PDE 各向异性扩散 [23][24]、frame-based 分割 [20] 中至少一两个，复现"提取更多 tubular details"的定性结论。
7. **指标与表格**：需复现 Table I 的逐轮 |Λ^(i)| 收缩表、迭代次数、运行时间；若用带标注公开集，再补 Dice/IoU/敏感度（论文未给，须标注为等价任务）。
8. **参数对照**：λ_k=0.1、ε、DℂWT 层数=4 等需可配置并与论文一致。
9. **稀疏加速**：论文指出可只在 Λ 周围计算（O(n) 但常数更小）。paper-level 还需复现这一工程优化以匹配运行时间量级。

## 9. 运行步骤

**当前 toy/partial 运行**：

```bash
# 安装依赖
pip install -r requirements.txt   # numpy / scipy / scikit-image / matplotlib

# 运行全部复现（含本篇 tubular_tight_frame）
cd reproduce && python run_all.py
```

依赖（reproStructured.dependencies）：`numpy`、`scipy`、`scikit-image`、`matplotlib`。缺依赖时 runner 写入 `skipped` 而非伪造 `completed`。

**向 paper-like 扩展的步骤大纲**（仅规划，本文档不执行、不重跑结果）：

1. 实现 tight-frame 模块：分段线性 B-spline tight-frame（eq.1–2 的 Toeplitz/tri-diagonal 滤波 + 张量积到 2D/3D）；或封装 DℂWT（参考论文 [14] 的 Matlab WaveletSoftware 或等价 Python 实现）。验证 perfect reconstruction `Aᵀ A = I`。
2. 实现自适应区间 (eq.6–13)：梯度阈值初始化、`μ/μ_±`、`[α_i,β_i]`、三段映射 + contrast stretch、Λ 更新。
3. 组装 Algorithm 1 主循环 + 停机判据（Λ=∅），记录每轮 |Λ^(i)|、迭代数、时间，对齐 Table I 形式。
4. 接入公开数据（先 2D：DRIVE/STARE；再 3D：TubeTK/IRCADb），跑通 2D 后再扩 3D（注意内存）。
5. 接入基线（Chan-Vese 等），做定性叠加对比图；带标注集上补定量指标，明确标注"等价任务"。
6. 全程遵守纪律：实现确为论文级前，等级保持 toy/partial，禁止把指标标为 paper-level。

## 10. 风险与代理说明

- **Gaussian smoothing ≠ tight-frame**：当前 toy 用各向同性高斯平滑代替论文的 tight-frame / DℂWT 软阈值。高斯平滑无方向选择性、无 perfect reconstruction、无 1-norm 稀疏正则语义，因此**无法复现论文"方向选择性提取更多细血管/分叉细节"的核心优势**。toy 的高 Dice 来自合成图过于简单（清晰直线血管 + 适中噪声），不可外推。
- **固定区间 ≠ 自适应区间**：固定 `[0.38,0.62]` + 线性收紧只是模拟 Λ 收缩的"现象"，未实现论文 `μ/μ_±` 自适应逼近边界值域，故对真实低对比、强度不均的 MRA 不具代表性。
- **2D 合成 ≠ 真实 2D/3D MRA**：合成血管无真实 MRA 的 speckle 噪声、partial occlusion、intensity inhomogeneity、血管交叉/分叉复杂拓扑；3D 完全缺失。
- **指标不可外推**：toy Dice 0.9981 / IoU 0.9962 仅描述合成 toy；论文本身**未报告 Dice**。不得将该数字呈现为论文性能或 paper-level 复现。
- **有限收敛的可信部分**：Λ 单调收缩、有限步到二值这一**定性现象**与论文 Theorem 1 一致，是 toy 中相对可信的部分；但"几次迭代收敛 / O(n) 复杂度"的论文级量化仍依赖真实 tight-frame 实现与真实数据，当前未验证。

## 11. 参考：精读笔记

本篇精读笔记：[`../../../xiaohao_cai_ultimate_notes/Tight_Frame_Vessel_Segmentation_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Tight_Frame_Vessel_Segmentation_超精读笔记_已填充.md)
