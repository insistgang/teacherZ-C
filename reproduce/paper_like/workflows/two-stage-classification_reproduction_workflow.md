# 高维数据与点云两阶段分类 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 9 篇 *A Two-Stage Classification Method for High-Dimensional Data and Point Clouds* 的完整复现流程规范。

---

## 1. 论文身份与第一作者核验

| 项 | 内容 |
|----|------|
| **标题 (EN)** | A Two-Stage Classification Method for High-Dimensional Data and Point Clouds |
| **标题 (CN)** | 高维数据与点云两阶段分类方法 |
| **作者顺序** | **Xiaohao Cai**, Raymond Chan, Xiaoyu Xie, Tieyong Zeng |
| **第一作者核验** | 是。PDF 首页（p.1）作者列表 `XIAOHAO CAI*, RAYMOND CHAN†, XIAOYU XIE‡, AND TIEYONG ZENG‡` 以 **Xiaohao Cai** 开头，第一单位为 Mullard Space Science Laboratory (MSSL), University College London，邮箱 `x.cai@ucl.ac.uk`。确认 Xiaohao Cai 为第一作者。 |
| **年份 / 出处** | 2019，arXiv:1905.08538v1 [math.NA] 21 May 2019 |
| **类型** | 单一新方法论文（variational 半监督分类，非综述） |
| **PDF 路径** | `docs/00_papers_first_author_xiaohao_cai_deduped/两阶段分类 Two-Stage.pdf` |
| **主题 (theme)** | `classification`（把 SaT 两段式从像素分割迁移到图分类） |

本篇是 SaT（smoothing and thresholding）方法论从**图像像素分割**向**图分类（high-dimensional data + point clouds）**迁移的入口论文，是第 10 篇 *Efficient Variational Classification*（期刊版）的早期版本。核心思想：先 warm initialization（SVM 或随机标签），再在图上解**无约束凸**变分 smoothing 模型，最后投影（argmax）成 binary partition，并可迭代 refinement。

---

## 2. 复现目标与诚实分级

本项目对"复现"采用四级诚实分级（由弱到强）：

| 级别 | 含义 |
|------|------|
| **toy** | 合成小数据 + 代理算子，演示直觉，不对齐论文任何具体数值 |
| **partial** | 实现了论文核心步骤的一部分（如真实 warm init + 图平滑 + 投影），在合成数据上验证趋势，但未对齐论文数据集与报告数值 |
| **paper-like** | 用论文同款或公开等价数据集，跑论文同款 pipeline（含 primal-dual 凸求解器），复现论文表格量级（不要求逐位一致） |
| **paper-level** | 严格复现论文报告数值（同数据、同基线、同指标、同表号） |

**本仓库当前等级（reproductionLevel）= `partial`；真实性（reproductionTruthLevel）= `partial-completed`。**

纪律红线：
- **paper-level 在 15 篇中仍为 0/15。** 本篇也不例外。
- 当前实现使用 **kNN 图标签传播 / Laplacian-style 平滑**（凸组合 `probs = 0.72·probs + 0.28·neighbor_avg`）作为 Eq.(3.5) convex smoothing model + Algorithm 2 primal-dual 求解器的轻量代理（见 §7、§10），且仅在合成 **two-moon** 二类数据上跑。`runMetrics` 中的 `smoothed_accuracy=0.8139`、`accuracy_gain=0.0139` 是 toy 合成数据上的结果，**不得**被表述为论文级或论文报告的精度（论文 Three Moon=99.4%、COIL=94.0%、MNIST=97.5%、Opt-Digits 最高 98.6%，见 §6）。
- 当前 runner 与第 10 篇 efficient-variational-classification 共用 `graph_classification.py`，但只是把同一组 toy metrics 复用两份，**没有**实现论文真正区分两篇的 graph TV primal-dual solver。

---

## 3. 算法完整流程

论文方法的核心是把 SaT 两段式（smoothing → thresholding）映射到图上：**第一段**解无约束凸模型得 fuzzy partition `U`，**第二段**用 argmax 投影成 binary partition；两段可迭代。

### 3.1 问题设定（论文 §3.1）

给定点云 `V`（含 `N` 个 `R^M` 中的点），目标是划分为 `K` 类 `V_1,…,V_K`，满足 **no vacuum and overlap constraint** Eq.(3.1)：
```
V = ∪_{j=1}^K V_j,   V_i ∩ V_j = ∅  (∀ i≠j).
```
训练集 `T = {T_j}_{j=1}^K ⊂ V`，`|T|=N_T`，标签已知；待标注集 `S = V \ T`（test set）。partition matrix `U=(u_1,…,u_K)∈R^{N×K}`，`u_j(x)∈{0,1}` 是 indicator（Eq.(3.2)），满足 `Σ_j u_j(x)=1`。这是 **unit simplex 顶点约束**，二值化导致 NP-hard。凸松弛 Eq.(3.3) 把 `u_j(x)∈[0,1]`，但整模型一般仍非凸且耗时。

### 3.2 图与算子定义（论文 §2，Eq.(2.1)-(2.10)）

- **权重函数**（任选其一）：RBF Eq.(2.1) `w(x,y)=exp(−d(x,y)²/(2ξ))`；Zelnic-Manor & Perona Eq.(2.2) `exp(−d(x,y)²/(var(x)var(y)))`；cosine similarity Eq.(2.3)。
- **affinity matrix** `W=(w(x,y))`，**度矩阵** `D=diag(Σ_z w(x,z))`（Eq.(2.4)）。
- **graph Laplacian** `L = D − W`。
- **graph gradient** Eq.(2.5)/(2.8)：`∇u(x) = (w(x,y)(u(x)−u(y)))_{y∈N(x)}`。
- **graph TV (ℓ₁-norm)** Eq.(2.6)/(2.9)：`||∇u||₁ = Σ_x Σ_{y∈N(x)} |w(x,y)(u(x)−u(y))|`。
- **Dirichlet energy (ℓ₂-norm)** Eq.(2.7)/(2.10)：`||∇u||² = ½ uᵀLu = ½ Σ w(x,y)(u(x)−u(y))²`。
- **k-NN 图**：用 `N(x)`（k 近邻）代替全连接边集 E，既省算力又能捕捉流形局部结构（Eq.(2.8)-(2.10)）。

### 3.3 第一段 Stage one — 凸 smoothing 模型（论文 §3.2，**核心公式** Eq.(3.5)）

```
argmin_U  Σ_{j=1}^K {  (β/2) ||u_j − û_j||²₂        ← 数据保真项（贴近 warm init û_j）
                      + (α/2) u_jᵀ L u_j              ← graph Laplacian / Dirichlet ℓ₂ 平滑项
                      +        ||∇u_j||₁ }            ← graph total variation（ℓ₁）项
```
其中 `α, β > 0` 是正则参数。**三项作用**：
1. fidelity `(β/2)||u_j−û_j||²`：保持 fuzzy partition 不偏离 warm init；
2. `(α/2)u_jᵀLu_j`（即 `α·Dirichlet energy`）：在图上平滑标签，使相邻点标签接近；
3. `||∇u_j||₁`（graph TV）：把"相似信息"的点逼到一起，鼓励分块常数（piecewise-constant）划分，同时缓解纯 ℓ₁ 的 staircase 伪影（这正是论文相对仅含 TV 的方法多加 ℓ₂ 项的理由，见 §5.6 讨论）。

**关键结构性质（效率来源）**：Eq.(3.5) 对 `K` 个标签函数 `u_j` **无耦合约束**（去掉了 simplex 约束 Eq.(3.3)），因此可拆成 `K` 个**独立**凸子问题，天然适合 parallelism。这是相对 CVM/GL/TVRF 等带 simplex 约束方法的核心计算优势。

**训练标签固定**：`û_j(x)=ū_j(x), ∀x∈T`（Eq.(3.6)）；只对 test set `S` 求解。按 Eq.(3.7)/(3.8) 把 `u_j` 拆成 `(u_{S_j}, ū_j)`，`L` 按训练/测试块分解为 Eq.(4.13) `L=[[L_S+L_1, L_3],[L_3ᵀ, L̄+L_2]]`。

**Theorem 3.1（唯一解）**：给定 `Û∈R^{N×K}` 与 `α,β>0`，Eq.(3.5) 有**唯一解** `U`。证明：Eq.(3.5) 强凸（strongly convex），强凸函数有唯一极小（依据 [7] Chapter 9）。

### 3.4 第二段 Stage two — 投影成 binary partition（论文 §3.2，Eq.(3.4)）

对 stage-one 得到的 fuzzy `U`，逐点取最大分量投影到 unit simplex 顶点：
```
(u_1(x),…,u_K(x)) ↦ e_i,   i = argmax_j {u_j(x)}_{j=1}^K,  ∀x∈V.    (Eq.(3.4))
```
该 binary partition 自动满足 no vacuum and overlap constraint Eq.(3.1)。stage two 相对 stage one **计算可忽略**。

### 3.5 两段迭代 refinement（论文 §3.2，Algorithm 1）

```
Algorithm 1 (SaT for high-dimensional data classification)
Initialization: SVM（或随机标签）生成 Û。
For l = 0,1,…  until ||U^(l) − U^(l+1)|| = 0（标签不再变化）:
    Stage one: 解 Eq.(3.5) 得 fuzzy U。
    Stage two: 用 Eq.(3.4) 投影得 binary U^(l+1)。
    Set Û = U^(l+1)，并 β ← 2β（每次迭代 β 翻倍以加速收敛）。
Endfor
Output: U* = U^(l+1)。
```
论文经验：**≈10 次迭代**通常足够；用 SVM 初始化时一般 ≤15 次，poor / 随机初始化时 ≤20 次（§6 Conclusions）。

### 3.6 第一段求解器 — primal-dual algorithm（论文 §4，Algorithm 2）

Eq.(3.5) 无约束凸，可用 split-Bregman、ADMM 或 primal-dual 求解；论文采用 Chambolle-Pock primal-dual（[23]）：

- 通用 saddle-point Eq.(4.1) `min_x max_x̄ {⟨Kx, x̄⟩ + G(x) − F*(x̄)}`，迭代 Eq.(4.2)-(4.4)。
- 把 `L` 分解 Eq.(4.13)、`∇u_j = A_S(u_{S_j}) + H_j`（Eq.(4.15)），代入 Eq.(3.5) 得逐 `j` 子问题 Eq.(4.16)：
  ```
  argmin_{u_{S_j}} (β/2)||û_{S_j}−u_{S_j}||² + (α/2)u_{S_j}ᵀL_S u_{S_j} + α u_{S_j}ᵀL_3 ū_j + ||A_S(u_{S_j})+H_j||₁
  ```
- 定义 `G_j` Eq.(4.17)、`F_j(x̃)=||x̃+H_j||₁` Eq.(4.18)，共轭 `F_j*` Eq.(4.19)。
- **proximal 算子**：`(I+σ∂F_j*)⁻¹` 是逐点投影 `ι_P(x̃+σH_j)` Eq.(4.21)-(4.22)（把分量截断到 |·|≤1）；`(I+τ∂G_j)⁻¹` 归结为解**正定线性系统** Eq.(4.24) `(αL_S + βI + (1/τ)I)u_{S_j} = β û_{S_j} + (1/τ)x − αL_3 ū_j`，可用 **conjugate gradient** 高效求解。
- **Lemma 4.1**：`G_j` 关于 `β` 强凸，可据此**自适应**调整 `σ,τ`（Eq. 中 `θ=1/√(1+βτ)`、`τ←θτ`、`σ←σ/θ`）以加速。
- **Theorem 4.2（收敛性）**：取 `τ⁰σ⁰ < 1/(N²(k−1))`（由 `||A_S||₂ ≤ N√(k−1)` 上界得到），Algorithm 2 收敛。

> 完整 pipeline：构 k-NN 图 → SVM warm init → (Algorithm 2 解 Eq.(3.5) 得 fuzzy U) → argmax 投影 Eq.(3.4) → 检查标签是否变化，否则 β←2β 重复。

---

## 4. 完整复现所需数据集

论文 §5 在**四个 benchmark 数据集**上评测（Table 5.1）。下表给出论文实证使用的数据与公开/等价候选。

| 数据集 | 类数 K | 维度 | 点数 N | 论文设置（PDF 实证） | 公开 / 获取 |
|--------|--------|------|--------|----------------------|-------------|
| **Three Moon**（合成） | 3 | 100 | 1500 | 三个半圆（两上半径1、一下半径1.5，圆心 (0,0)/(3,0)/(1.5,0.4)），各采 500 点嵌入 R^100，逐维加 i.i.d. 高斯噪声 std=0.14；75 训练点 | 可按 §5.2 程序**自行合成**（[1,56] 同款） |
| **COIL** | 6 | 241 | 1500 | Columbia object image library；红通道 16×16 下采，取 24 物体分 6 类（每类 4 物体×288），最终每类 250 张；10% 训练 | [28] supplementary material |
| **Opt-Digits** | 10 | 64 | 5620 | 手写数字 0-9，32×32 bitmap 按 4×4 块计"on"像素 → 8×8 整数矩阵 [0,16]；训练 50/100/150 | **UCI ML repository** `archive.ics.uci.edu/ml/datasets.html` |
| **MNIST** | 10 | 784 | 70000 | 28×28 手写数字 0-9；训练 2500（3.57%） | **yann.lecun.com/exdb/mnist** |

**图构建参数（Table 5.1 周边正文，逐数据集）**：
- Three Moon：k-NN `k=10`，RBF 权重 `σ=3`，欧氏距离；75 训练点（uniform）。
- COIL：`k=4`，RBF `σ=250`，欧氏；10% 训练。
- Opt-Digits：`k=8`，RBF `σ=30`，欧氏；训练 50/100/150。
- MNIST：`k=8`，**Zelnic-Manor & Perona 权重 Eq.(2.2)**（8 closed neighbors），欧氏；训练 2500。
- 全数据集用 **randomized kd-tree [53]** 求近邻。

**正则参数（论文 §5 正文）**：`β`：MNIST=1e-4、COIL=1e-5、Three Moon/Opt-Digits=1e-2；`α`：Three Moon/Opt-Digits=1、MNIST=0.4、COIL=1e-2。运行环境：MacBook 2.8 GHz / 16 GB RAM / Matlab 2017a。

> 本篇数据**全部公开或可合成**，无私有医学/RI 数据障碍（与第 1、5、7 篇不同）。这是 paper-like 复现门槛相对低的一篇。

---

## 5. 对照基线 (Baselines)

论文 §5.1 与多种 state-of-the-art variational / graph 分类方法对照（数值取自 [1,56] 或原文，PDF 表格实证）：

| 简称 | 全称 / 出处 | 出现表 |
|------|-------------|--------|
| **CVM** | Convex (constrained) variational model [1] | Table 5.2, 5.4, 5.5 |
| **GL** | Ginzburg-Landau [33] | Table 5.2, 5.4, 5.5 |
| **MBO** | Merriman-Bence-Osher scheme [33] | Table 5.2, 5.4, 5.5 |
| **TVRF** | Total-variation-based / region-force [56] | Table 5.2-5.7（含计时） |
| **LapRF** | Laplacian region-force [56] | Table 5.2, 5.4-5.6 |
| **LapRLS** | Laplacian regularized least squares [54] | Table 5.6 |
| **MP** | Measure propagation [54] | Table 5.6 |
| **SQ-Loss-I** | Squared-loss [28] | Table 5.6 |
| **SGT** | Spectral graph transducer | Table 5.6 |
| **k-NN** | 最近邻基线 | Table 5.6 |

本方法的 warm init 用 **SVM（linear kernel）[29]**；数据集过大或不适用时退化为**随机标签**初始化。

合理的最小对照（本仓库 toy 层）：**warm init（centroid / SVM）直接分类** vs **本方法（warm init + graph smoothing + projection 迭代）**，对比 accuracy 增量；以及 direct（不平滑）vs smoothing 后的 decision 着色图。

---

## 6. 评价指标与论文报告结果

### 6.1 指标定义
- **Classification accuracy**：正确标注点占比（论文定义于 §5：percentage of correctly labeled data points）。所有数据集 10 次随机训练集试验取平均。
- **Number of iterations**：两段迭代收敛所需次数（论文报告平均值）。
- **Computation time (s)**：CPU 时间，论文强调本方法因无约束、K 子问题独立而具速度优势（Table 5.7）。

### 6.2 论文报告的关键数值（PDF 表格确认，注明出处）

**Three Moon（uniform 训练，Table 5.2）**：CVM 98.7、GL 98.4、MBO 99.1、TVRF 98.6、LapRF 98.4、**Proposed 99.4**（%）。平均迭代 3.8（正文 §5.2，另一处给 3.3）。
**Three Moon（non-uniform 训练，Table 5.3）**：TVRF 97.8、**Proposed 99.3**；体现对训练点分布的鲁棒性（TVRF 掉 0.8%，本方法几乎不掉），non-uniform 时平均 12.0 次迭代。
**COIL（Table 5.4）**：CVM 93.3、TVRF 92.5、LapRF 87.7、GL 91.2、MBO 91.5、**Proposed 94.0**；平均迭代 12.2。
**MNIST（Table 5.5）**：**CVM 97.7**、TVRF 96.9、LapRF 96.9、GL 96.8、MBO 96.9、Proposed 97.5（此处 CVM 略高于本方法，论文称"comparable to or better"）。
**Opt-Digits（Table 5.6，样本率 0.89%(50)/1.78%(100)/2.67%(150)）**：k-NN 85.5/92.0/93.8、SGT 91.4/97.4/97.4、LapRLS 92.3/97.6/97.3、SQ-Loss-I 95.9/97.3/97.7、MP 94.7/97.0/97.1、TVRF 95.9/98.3/98.2、LapRF 94.1/97.7/98.1、**Proposed 96.6/98.5/98.6**。
**Computation time（Table 5.7，秒，本方法括号内为平均迭代）**：

| Method | Three Moon | COIL | MNIST | Opt-Digits |
|--------|-----------|------|-------|------------|
| TVRF | 0.71 | 0.65 | 66.00 | 3.42 |
| Proposed | **0.30 (3.3)** | 0.76 (11.7) | 82.04 (9.4) | 4.45 (9.3) |

> 注：本方法在 Three Moon 上**又快又准**；在 COIL/MNIST/Opt-Digits 上时间与 TVRF 同量级，论文论点是"**考虑 K 子问题并行后**可再快约 K 倍"（§5.6）。**禁止编造任何未在 PDF 出现的数字。**

---

## 7. 本仓库当前复现实现

- **runner 文件**：`reproduce/experiments/graph_classification.py`（与第 10 篇 efficient-variational-classification 共用，返回两份相同 metrics）。
- **它实际做了什么**：
  - 合成 **two-moon** 二类数据（`n=360`，`rng.normal(0,0.18)` 噪声），每类随机选 10 个 labeled 点。
  - **centroid warm initialization**：用 labeled 点的类质心，对全体点按最近质心赋初始标签（代替论文的 **SVM warm init**）。
  - 构 **k-NN 图**（`cKDTree`，`k=13`），对称化得邻接 `graph`。
  - **标签传播 / Laplacian-style 平滑**：迭代 18 次 `probs = 0.72·probs + 0.28·neighbor_avg`，每步把 labeled 点钳回真值，`argmax` 得标签（代替论文的 Eq.(3.5) 凸模型 + Algorithm 2 primal-dual 求解）。
  - 产图：`assets/repro/graph_classification_before_after.png`（warm init / graph smoothing / accuracy trace 三联图）。
- **本篇当前 runMetrics（来自 dashboard `reproStructured`）**：

  | 指标 | 数值 | 含义 |
  |------|------|------|
  | `initial_accuracy` | 0.8000 | centroid warm init 在 two-moon toy 的精度 |
  | `smoothed_accuracy` | 0.8139 | 图平滑 + argmax 投影后的精度 |
  | `accuracy_gain` | 0.0139 | smoothing 相对 warm init 的提升 |
  | `iterations` | 18 | 平滑迭代次数（固定 18，非论文收敛判据） |
  | `runtimeSeconds` | ≈0.0904 | CPU 运行时间 |

  > 注意：上述是**合成 two-moon toy 二类**结果，演示"图平滑后再投影比纯 warm init 略好"的趋势，**不是**论文报告值（论文是 3/6/10 类、benchmark 数据、99.4%/94.0%/97.5% 量级）。
- **resultFiles**：`assets/repro/graph_classification_before_after.png`。
- **implementationRisk（dashboard 已记录）**：`toy 使用 Laplacian smoothing，不是完整 graph TV convex model。`（注：本篇 dashboard 条目记录在 `implementationRisk` 字段，无单独 `fidelityWarning` 字段。）

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

**到 paper-like 的缺口清单：**

1. **求解器对齐**：当前是固定 18 步的凸组合标签传播；需实现 Eq.(3.5) 的真正 **convex smoothing model**（含 fidelity + `(α/2)uᵀLu` graph Laplacian + `||∇u||₁` **graph TV**），并用 **Algorithm 2 primal-dual**（Eq.(4.21)-(4.24)，含 CG 解线性系统 + 逐点投影 + 自适应 σ,τ）求解。当前实现**缺 graph TV ℓ₁ 项**，只有 Laplacian 平滑。
2. **warm init 对齐**：把 centroid 初始化换成论文的 **SVM（linear kernel）**，并支持随机标签初始化作为大数据集 fallback。
3. **两段迭代 + β 翻倍**：实现 Algorithm 1 的外层迭代（argmax 投影 → 设为新 Û → β←2β → 重复直到标签不变），而非固定单轮 18 步平滑。
4. **真实数据接入**：当前仅 two-moon 二类合成。需接入 **Three Moon（按 §5.2 合成）/ COIL / Opt-Digits（UCI）/ MNIST**，并按论文 Table 5.1 周边参数建图（k、σ、权重函数、训练比例）。
5. **多类（K>2）**：当前只跑二类；论文是 3/6/10 类，需验证 K 个独立子问题的并行结构。
6. **基线缺失**：未实现 CVM / GL / MBO / TVRF / LapRF 等论文对照方法，无法生成并排对照表。
7. **指标/表格对照**：未对齐 Table 5.2-5.7 的 accuracy / 平均迭代 / CPU time；未区分 uniform vs non-uniform 训练点（Table 5.2 vs 5.3 的鲁棒性实验）。

**到 paper-level 的额外缺口：**

8. 需严格对齐论文每个数据集的 10 次随机划分、`α/β` 取值（如 MNIST β=1e-4,α=0.4）、随机种子、收敛判据 `||U^(l)−U^(l+1)||=0`，逐表复现 99.4 / 94.0 / 97.5 / 98.6 等数字。
9. 需复现 Table 5.7 的计时口径（同机型量级），并实现 §5.6 论述的 **K-way parallelism** 才能体现论文宣称的速度优势。

---

## 9. 运行步骤

### 9.1 当前 toy/partial 跑法

```bash
# 安装依赖（见下）
pip install -r requirements.txt

# 运行全部复现实验（含本篇 graph_classification）
cd reproduce && python run_all.py

# 或在仓库根校验 15 篇数据 / PDF / 笔记 / 静态复现资产
node docs/scripts/validate.mjs
```

- **依赖**（来自 `reproStructured.dependencies`）：`numpy`、`scipy`、`matplotlib`。
- **算力**：CPU，约 1 秒内（`runtimeSeconds≈0.09`）。
- **数据**：合成 two-moon 二类，**无需下载真实数据**。
- 缺依赖时 runner 返回 `skipped`（见 `require_modules`），**不伪造 completed**（遵守 CLAUDE.md 纪律）。

### 9.2 向 paper-like 扩展的步骤大纲

1. 把标签传播替换为 Eq.(3.5) 凸模型 + Algorithm 2 primal-dual：实现 graph TV ℓ₁ 项、`L` 的训练/测试块分解 Eq.(4.13)、CG 解 Eq.(4.24)、逐点投影 Eq.(4.22)、自适应 σ,τ（Lemma 4.1）。
2. 实现 Algorithm 1 外层：argmax 投影 Eq.(3.4) → 设新 Û → β←2β → 迭代到标签稳定。
3. 新增 `reproduce/data/` 接入脚本：按 §5.2 合成 Three Moon；下载 Opt-Digits（UCI）、MNIST、COIL（[28] supplementary），并按论文参数建 k-NN 图（randomized kd-tree）。
4. 把 warm init 改为 SVM（linear kernel），并加随机标签 fallback；支持 K>2 多类与 K-way 并行。
5. 加入论文基线（至少 TVRF、CVM、direct warm-init）做 Table 5.2-5.6 风格并排对照，并记录平均迭代与 CPU time（Table 5.7 风格）。
6. 在 dashboard 中把每个数据集的 `reproductionLevel` 独立标注，避免用单一 two-moon toy 数字代表整篇。

---

## 10. 风险与代理说明

- **Laplacian 标签传播 ≠ Eq.(3.5) 凸模型**：当前 `probs=0.72·probs+0.28·neighbor_avg` 是邻域平均的凸组合迭代，相当于一种 graph Laplacian 平滑/扩散，**缺少 graph TV (ℓ₁) 项**，也**不是** Algorithm 2 的 primal-dual 凸优化求解，更未验证 Theorem 3.1（唯一解）或 Theorem 4.2（收敛性）。因此 `smoothed_accuracy=0.8139` 只能说明"图平滑后再投影略优于纯 warm init"的**定性**趋势。
- **合成数据 + 二类的局限**：toy 是 two-moon 二类，不含论文的高维（100/241/64/784 维）、多类（3/6/10）、真实噪声分布、不同权重函数（RBF vs Zelnic-Manor）与训练点分布（uniform/non-uniform）等设置，无法反映论文在 benchmark 上的鲁棒性与速度优势。
- **共享 runner 的口径**：本篇与第 10 篇 efficient-variational-classification 共用 `graph_classification.py` 且返回**相同** metrics；当前实现**没有**体现两篇的真实差异（如期刊版的扩展/不同 solver），阅读 dashboard 时勿把两份相同数字当作两次独立复现。
- **warm init 差异**：论文用 SVM，本仓库用 centroid，初始精度口径不同（本 toy initial=0.80），不可与论文"SVM 初始化 + ≤15 次迭代"的设定对应。
- **不可外推的结论**：① 不能说本仓库"复现了"论文任一数据集结果；② 不能把 toy 二类精度（0.8139）等同于论文报告精度（99.4%/94.0%/97.5% 等）；③ paper-level 在 15 篇中仍为 0/15，本篇亦然。

---

## 11. 参考：精读笔记

完整中文精读笔记见：
[`../../../xiaohao_cai_ultimate_notes/Two-Stage_Classification_Point_Clouds_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Two-Stage_Classification_Point_Clouds_超精读笔记_已填充.md)
