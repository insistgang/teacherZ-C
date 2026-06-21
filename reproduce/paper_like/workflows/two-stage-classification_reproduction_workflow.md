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
- 当前实现（2026-06 升级后）已是 Eq.(3.5) 真正的 **graph TV（ℓ₁）凸模型 + Chambolle-Pock primal-dual 求解器** + **SVM warm init** + 论文式 **Three-Moon（3 类）**（见 §7、§10），不再是 Laplacian 标签传播代理。`runMetrics` 中的 `tv_accuracy=0.9957` 是 **Three-Moon（R²,3 类，N=750）** 合成数据上的结果，**不得**被表述为论文级或论文报告的精度（论文 Three Moon=99.4%、COIL=94.0%、MNIST=97.5%、Opt-Digits 最高 98.6%，见 §6）；本仓库仍是单一数据集、规模缩小、无论文对照基线。
- 当前 runner 与第 10 篇 efficient-variational-classification 共用 `graph_classification.py`，返回同一组 metrics 两份；它实现的是论文共有的 graph-TV primal-dual 骨架，**未**额外区分期刊版（#10）相对会议版（#9）的扩展。

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

> **2026-06 升级**：runner 已从"centroid 初始化 + Laplacian 标签传播代理"升级为**真实算法**——在论文式 **Three-Moon**（3 个半月形簇）数据上实现**真正的 graph total variation（ℓ₁）凸模型 + Chambolle-Pock primal-dual 求解器**，并提供三方法对照。

- **runner 文件**：`reproduce/experiments/graph_classification.py`（与第 10 篇 efficient-variational-classification 共用，返回两份相同 metrics）。
- **它实际做了什么**：
  - **论文式 Three-Moon 数据集**（Cai et al. §5.2 的构造法）：两个上半单位圆（圆心 (0,0)/(3,0)）+ 一个半径 1.5 的下半圆（圆心 (1.5,0.4)），每弧 250 点，加 i.i.d. 高斯噪声 `std=0.14`（论文 std=0.14）。共 `N=750` 点、`K=3` 类。**几何保留在 R²**（brief 允许 R² 或加噪高维；在纯噪声高维下 98 维噪声会劣化 k-NN 图并使 TV 过度平滑，故取 R² 以让 ℓ₁-vs-ℓ₂ 优势真实可见）。每类随机选 15 个 labeled 点（论文 75/1500）。
  - **RBF 加权 k-NN 图**：`cKDTree` 取 `k=10`，权重 `w(x,y)=exp(−d²/(2ξ))`，`ξ=0.5·median(d²)`（self-tuning 带宽），对称化得稀疏 `W`；度矩阵 `D`、graph Laplacian `L=D−W`、graph gradient 算子 `K`（逐边加权差）与其伴随 `K*`（散度），`||K||₂` 用 `K*K` 上的幂迭代估计。
  - **SVM warm initialization**：`sklearn.svm.LinearSVC`（linear kernel，论文口径）对全体点给初始标签。
  - **三方法对照**（共用同一 warm init）：
    1. **raw K-means**（`common.simple_kmeans`，无图无标签）——参照基线。
    2. **graph-Laplacian（ℓ₂，旧方法）**：精确解 `(αL+βI)u_j=β û_j`（训练行钳为 Dirichlet 边界），`argmin (β/2)||u−û||² + (α/2)uᵀLu`，仅 Dirichlet energy、**无 graph TV**。
    3. **graph-TV（ℓ₁，新）**：解论文 Eq.(3.5)/Eq.(15) 凸模型 `argmin_U Σ_j (β/2)||u_j−û_j||² + (α/2)u_jᵀLu_j + ||∇u_j||₁`，用**真正的 Chambolle-Pock primal-dual**：对偶步 `p←clip_{[-1,1]}(p+σKz)`（ℓ₁ 共轭的逐点 box 投影），原始步解 `(I+τ(αL+βI))u=u−τK*p+τβû`（用 **conjugate gradient**），训练行每步钳回已知标签（Eq.(3.6)/(3.7) 只对 test set S 求解）。`K` 个标签函数**解耦**逐类独立求解（论文核心结构）。stage two = argmax 投影 Eq.(3.4)；外层 **Algorithm 1** 循环（argmax → 新 Û → `β←2β` → 直到标签不变）。
  - **参数**：`α=1`、初始 `β=1e-2`（论文 Three Moon 缺省 α=1、β=1e-2）；primal-dual 步长 `τ=σ=0.9/||K||₂`（满足 `στ||K||²<1`）。
  - 产图：`assets/repro/graph_classification_before_after.png`（warm init / graph-Laplacian / graph-TV 三幅决策着色图 + 四方法 accuracy 柱状图）。
- **本篇当前 runMetrics（来自 runner 实际 `common.completed(...)` 返回）**：

  | 指标 | 数值 | 含义 |
  |------|------|------|
  | `kmeans_accuracy` | 0.8170 | raw K-means（无标签）在 test set 的聚类精度 |
  | `initial_accuracy` | 0.8950 | SVM warm init 的 test 精度 |
  | `laplacian_accuracy` | 0.9220 | graph-Laplacian（ℓ₂，旧方法）的 test 精度 |
  | `tv_accuracy` | 0.9957 | **graph-TV（ℓ₁，新）的 test 精度（最高）** |
  | `tv_gain_over_laplacian` | 0.0738 | TV 相对 Laplacian 的提升（体现 ℓ₁ 优于 ℓ₂） |
  | `tv_gain_over_init` | 0.1007 | TV 相对 SVM warm init 的提升 |
  | `outer_iterations` | 2 | Algorithm 1 外层收敛所需迭代数（标签不再变化即停） |
  | `runtimeSeconds` | ≈0.45 | CPU 运行时间（确定性，<8 秒预算内） |

  > 注意：上述是**合成 Three-Moon（R²，3 类）**结果，体现"**graph-TV（ℓ₁）> graph-Laplacian（ℓ₂）> raw**"的真实算法趋势；但仍是单一合成数据集、规模缩小（N=750），**不是**论文 benchmark 报告值（论文 Three Moon 99.4% / COIL 94.0% / MNIST 97.5% / Opt-Digits 最高 98.6%）。
- **resultFiles**：`assets/repro/graph_classification_before_after.png`。
- **fidelityWarning（runner `extra` 实际暴露）**：`Real graph-TV (l1) Chambolle-Pock primal-dual on a paper-like Three-Moon (R^2, RBF k-NN graph, SVM warm init) -- NOT paper-level: Three Moon kept in R^2 not R^100, only this one dataset (no COIL/Opt-Digits/MNIST), reduced N=750, no CVM/GL/MBO/TVRF comparison baselines, no 10-run averaging. paper-level remains 0/15.`

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

> **已落地（2026-06 升级，不再是缺口）**：① Eq.(3.5) 真正的 convex smoothing model（fidelity + `(α/2)uᵀLu` + `||∇u||₁` **graph TV**）+ **Chambolle-Pock primal-dual**（逐点 box 投影 + CG 解正定线性系统）；② **SVM（linear kernel）** warm init；③ Algorithm 1 外层迭代（argmax 投影 → 新 Û → `β←2β` → 停在标签不变）；④ 论文式 **Three-Moon**（3 类）数据 + **RBF 加权 k-NN 图**；⑤ K=3 个**独立**子问题逐类求解（论文结构）；⑥ 三方法对照基线（raw K-means / graph-Laplacian ℓ₂ / graph-TV ℓ₁），并已实证 **TV > Laplacian > raw**。

**到 paper-like 的剩余缺口清单：**

1. **数据维度与规模**：当前 Three-Moon 保留在 **R²**、`N=750`、每类 15 标注点；论文是 **R¹⁰⁰**、`N=1500`、75 标注点。需把嵌入升到 R¹⁰⁰ 并解决高维噪声下 TV 过度平滑的问题（如改权重函数 / 调 ξ / 加归一化 Laplacian），同时放大 N 与标注预算到论文量级。
2. **更多数据集**：仍只有 Three Moon。需接入 **COIL / Opt-Digits（UCI）/ MNIST**，并按论文 Table 5.1 周边参数建图（k、σ、权重函数、训练比例、randomized kd-tree）。
3. **更高类数（K=6/10）**：当前 K=3；论文 COIL=6、Opt-Digits/MNIST=10，需验证更高类数下 K 个独立子问题的并行结构与精度。
4. **论文对照基线缺失**：当前对照是 raw/ℓ₂/ℓ₁ 的**内部**消融，**未**实现论文的 **CVM / GL / MBO / TVRF / LapRF** 等 state-of-the-art 方法，无法生成 Table 5.2-5.6 风格的并排对照表。
5. **指标/表格对照**：未对齐 Table 5.2-5.7 的 accuracy / 平均迭代 / CPU time；未区分 uniform vs non-uniform 训练点（Table 5.2 vs 5.3 的鲁棒性实验）。
6. **自适应步长**：当前 primal-dual 用固定 `τ=σ=0.9/||K||₂`；论文 Lemma 4.1 用强凸性自适应更新 `θ=1/√(1+βτ)`、`τ←θτ`、`σ←σ/θ` 加速，尚未实现。

**到 paper-level 的额外缺口：**

7. 需严格对齐论文每个数据集的 **10 次随机划分取平均 + std**、`α/β` 取值（如 MNIST β=1e-4,α=0.4）、随机种子、收敛判据 `||U^(l)−U^(l+1)||=0`，逐表复现 99.4 / 94.0 / 97.5 / 98.6 等数字。
8. 需复现 Table 5.7 的计时口径（同机型量级），并实现 §5.6 论述的 **K-way parallelism** 才能体现论文宣称的速度优势。

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

> 2026-06 升级后，原"Laplacian 标签传播代理 / centroid 初始化 / two-moon 二类"代理**已被真实算法取代**（见 §7）：现在是 Eq.(3.5) graph-TV ℓ₁ 凸模型 + Chambolle-Pock primal-dual + SVM warm init + Three-Moon 3 类。下列为**仍然存在**的局限：

- **数据规模与维度仍非论文级**：Three-Moon 保留在 **R²**、`N=750`、每类 15 标注点；论文是 R¹⁰⁰、N=1500、75 标注点。**之所以用 R² 而非加噪 R¹⁰⁰**：在 98 维纯噪声下 k-NN 图质量退化，graph-TV 会过度平滑甚至塌缩到多数类（实测 R¹⁰⁰ 下 TV 反而劣于 ℓ₂）；R² 是让 ℓ₁-vs-ℓ₂ 优势真实、稳定、可视化的最小设置。这本身就是"未到 paper-level"的一个具体原因。
- **理论保证仅部分对应**：实现确为强凸模型 + primal-dual（对应 Theorem 3.1 唯一解、Theorem 4.2 收敛），但**未做收敛性数值验证**（未监控 `τ⁰σ⁰<1/(N²(k−1))` 边界、未画 energy 下降曲线），且 stage-two argmax 投影后的 hard labels 本就无全局最优保证。
- **单一数据集 + 内部对照**：只有 Three Moon 一套；三方法对照（raw / ℓ₂ / ℓ₁）是**内部消融**，**未**实现论文的 CVM/GL/MBO/TVRF/LapRF 等真实对照方法，故不能生成 Table 5.2-5.6 风格的论文并排表。无 10 次随机划分平均/std，无 CPU-time（Table 5.7）口径。
- **共享 runner 的口径**：本篇与第 10 篇 efficient-variational-classification 共用 `graph_classification.py` 且返回**相同** metrics；当前实现是同一 graph-TV 求解器，**没有**额外体现期刊版（#10）相对会议版（#9）的扩展（如一类分类、更多 benchmark）。阅读 dashboard 时勿把两份相同数字当作两次独立复现。
- **不可外推的结论**：① 不能说本仓库"复现了"论文任一 benchmark 数据集结果；② 不能把 Three-Moon（R²,3 类）的 `tv_accuracy=0.9957` 等同于论文 Three Moon 报告的 99.4%（数据维度/规模/采样/平均口径都不同）；③ paper-level 在 15 篇中仍为 0/15，本篇亦然。

---

## 11. 参考：精读笔记

完整中文精读笔记见：
[`../../../xiaohao_cai_ultimate_notes/Two-Stage_Classification_Point_Clouds_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Two-Stage_Classification_Point_Clouds_超精读笔记_已填充.md)
