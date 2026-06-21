# 高维数据与点云两阶段分类

> 当前 15 篇口径内第 9 篇。本文档按 PDF 首页作者顺序和 dashboard 结构化精读字段重写，避免旧论文笔记混入。

## 论文元信息

| 字段 | 内容 |
| --- | --- |
| 英文标题 | A Two-Stage Classification Method for High-Dimensional Data and Point Clouds |
| 作者顺序 | Xiaohao Cai, Raymond Chan, Xiaoyu Xie, Tieyong Zeng |
| 第一作者核验 | 是，PDF 首页作者列表以 Xiaohao Cai 开头 |
| 年份 | 2019 |
| 类型 | arXiv |
| PDF | docs/00_papers_first_author_xiaohao_cai_deduped/两阶段分类 Two-Stage.pdf |
| 阅读顺序 | 9 / 15 |
| 主题 | classification |
| 难度 | 中等偏难 |

## 一句话贡献

把 SaT 迁移到图分类。

## 核心问题

高维数据和点云没有规则图像网格；传统 graph-based variational classification 常受 unit simplex 约束、非凸或 NP-hard 形式影响，速度和可扩展性受限。

## 为什么难

点云分类要在 k-NN 图上处理标签传播，既需要保留少量标签或 warm initialization 的信息，又要在图结构上平滑；多类问题中 K 个类别互相耦合会拖慢求解。

## 方法抓手

论文先用 support vector machine (SVM) 或随机标签生成 fuzzy warm initialization，再在图上解无约束凸变分 smoothing 模型。该模型包含保真项、graph Laplacian 平滑项和 graph Total Variation，最后把 smoothed partition 投影到 binary partition。

## 关键模型或公式

Stage one 的核心凸模型（论文 Eq.(3.5)）：

```
argmin_U  Σ_{j=1}^K {  (β/2) ||u_j − û_j||²₂   +   (α/2) u_jᵀ L u_j   +   ||∇u_j||₁  }
```

逐项解释（对每个标签函数 `u_j`，`j=1,…,K`）：

| 项 | 名称 | 作用 |
| --- | --- | --- |
| `(β/2)‖u_j − û_j‖²₂` | data fidelity（保真项） | 让 fuzzy partition 不偏离 warm initialization `û_j`；训练点上 `û_j=ū_j` 固定不动（Eq.(3.6)），只对 test set `S` 求解 |
| `(α/2) u_jᵀ L u_j` | graph Laplacian / Dirichlet energy（ℓ₂ 平滑项） | `= (α/2)·½Σ w(x,y)(u(x)−u(y))²`（Eq.(2.7)/(2.10)），鼓励相邻点标签接近，是"smooth"来源；并缓解纯 TV 的 staircase 伪影 |
| `‖∇u_j‖₁` | graph total variation（ℓ₁ 项） | `= Σ_x Σ_{y∈N(x)}|w(x,y)(u(x)−u(y))|`（Eq.(2.6)/(2.9)），鼓励 piecewise-constant 划分，把"相似信息"的点逼到一起 |

图算子定义（论文 §2）：affinity `W=(w(x,y))`、度矩阵 `D=diag(Σ_z w(x,z))`、**graph Laplacian `L=D−W`**；graph gradient `∇u(x)=(w(x,y)(u(x)−u(y)))_{y∈N(x)}`。权重 `w` 可取 RBF Eq.(2.1) `exp(−d²/(2ξ))`、Zelnic-Manor & Perona Eq.(2.2) 或 cosine similarity Eq.(2.3)。用 **k-NN 图** `N(x)` 代替全连接边集，既省算力又能捕捉流形局部结构。

**关键结构**：Eq.(3.5) 去掉了 unit simplex 约束（Eq.(3.3) `Σ_j u_j(x)=1, u_j∈[0,1]`），对 `K` 个 `u_j` **无耦合**，故可拆成 `K` 个**独立**凸子问题——这是相对 CVM/GL/TVRF 等带约束方法的核心效率与并行优势。`α,β>0` 为正则参数（论文按数据集取值，如 MNIST `β=1e-4,α=0.4`；Three Moon/Opt-Digits `β=1e-2,α=1`；COIL `β=1e-5,α=1e-2`）。

Stage two 投影（论文 Eq.(3.4)）：`(u_1(x),…,u_K(x)) ↦ e_i, i=argmax_j u_j(x)`，逐点取最大分量到 simplex 顶点，自动满足 no vacuum and overlap constraint Eq.(3.1)。

## 算法流程

**外层 Algorithm 1（SaT for high-dimensional data classification）**：

1. 构建 k-NN 图和图权重（randomized kd-tree 求近邻，权重用 RBF / Zelnic-Manor）。
2. 用 SVM（linear kernel）或随机标签得到 warm initialization `Û`（论文强调初始化精度不关键，方法会显著提升它）。
3. **Stage one**：对每个类别 `j` **独立**求解凸 smoothing 子问题 Eq.(3.5) 得 fuzzy partition `U`（用 Algorithm 2 primal-dual）。
4. **Stage two**：把平滑标签函数按 Eq.(3.4) `argmax_j u_j(x)` 投影到 simplex 顶点 / binary partition `U^(l+1)`。
5. 用结果作为新初始化（`Û←U^(l+1)`）并 **`β←2β`** 加速，重复 stage one/two，直到标签不再变化 `‖U^(l)−U^(l+1)‖=0`，输出 `U*`。

> 收敛实践：论文经验 **≈10 次迭代**足够；SVM 初始化时一般 ≤15 次，随机/poor 初始化时 ≤20 次（§6 Conclusions）。non-uniform 训练点时 SVM 初始化更差，需更多迭代（Three Moon 从 uniform 的 3.3 升到 non-uniform 的 12.0 次）。

**内层 Algorithm 2（解 Eq.(3.5) 的 primal-dual 求解器，论文 §4）**：

- 把 `L` 按训练/测试块分解 Eq.(4.13) `L=[[L_S+L_1, L_3],[L_3ᵀ, L̄+L_2]]`，`∇u_j=A_S(u_{S_j})+H_j`（Eq.(4.15)），得逐 `j` 子问题 Eq.(4.16)。
- 定义 `G_j`（fidelity+Laplacian，Eq.(4.17)）、`F_j(x̃)=‖x̃+H_j‖₁`（Eq.(4.18)），用 Chambolle-Pock 迭代 Eq.(4.2)-(4.4)。
- 两个 proximal：`(I+σ∂F_j*)⁻¹` 是逐点投影 `ι_P`（把分量截断到 |·|≤1，Eq.(4.21)-(4.22)）；`(I+τ∂G_j)⁻¹` 归结为解**正定线性系统** Eq.(4.24) `(αL_S+βI+(1/τ)I)u_{S_j}=β û_{S_j}+(1/τ)x−αL_3 ū_j`，用 **conjugate gradient** 求解。
- 由 Lemma 4.1（`G_j` 关于 `β` 强凸）**自适应**更新 `σ,τ`（`θ=1/√(1+βτ)`）加速。

## 理论保证

论文给出两条主要理论结果，外加一条引理：

- **Theorem 3.1（唯一解）**：给定 `Û∈R^{N×K}` 与 `α,β>0`，凸模型 Eq.(3.5) 有**唯一解** `U`。直觉：Eq.(3.5) 是**强凸（strongly convex）**函数（fidelity 项 `(β/2)‖u−û‖²` 提供强凸性，graph Laplacian 与 TV 项均为凸），而强凸函数有唯一全局极小（依据 [7] Chapter 9）。注意这保证的是**第一段 smoothing 子问题**的唯一全局解，**不是**整体分类全局最优——分类全局最优在原 simplex 约束下仍是 NP-hard，论文用"凸子问题 + 投影 + 迭代"绕开它。这与 CVM/GL/TVRF 等只能找局部极小或松弛解的方法形成对比。
- **Lemma 4.1（强凸参数）**：`G_j`（fidelity+Laplacian 部分）对所有 `j` 关于参数 `β` 强凸。证明：`L_S` 半正定 ⇒ `(α/2)uᵀL_S u+α uᵀL_3 ū` 凸；剩余 `(β/2)‖u−û‖²` 关于 `β` 强凸。该引理用于自适应调 `σ,τ` 加速 primal-dual。
- **Theorem 4.2（收敛性）**：取 `τ⁰σ⁰ < 1/(N²(k−1))`，Algorithm 2 收敛。证明思路：由 Chambolle-Pock [23] Theorem 2，收敛条件是 `‖A_S‖²₂ < 1/(τ⁰σ⁰)`；论文给出上界 `‖A_S‖₂ ≤ √(‖A_S‖₁‖A_S‖_∞) ≤ N√(k−1)`（因权重函数取值在 [−1,1]，`A_S` 各项有界），从而得到充分条件。`k` 是 k-NN 的近邻数。

## 实验重点

实验覆盖 4 个 benchmark 数据集（论文 Table 5.1），重点看 warm init 后迭代 refinement 对 accuracy 与 computation speed 的提升。

| 数据集 | 类数 K | 维度 | 点数 N | 建图参数（论文正文） |
| --- | --- | --- | --- | --- |
| Three Moon（合成） | 3 | 100 | 1500 | k=10，RBF σ=3，75 训练点 |
| COIL | 6 | 241 | 1500 | k=4，RBF σ=250，10% 训练 |
| Opt-Digits | 10 | 64 | 5620 | k=8，RBF σ=30，训练 50/100/150 |
| MNIST | 10 | 784 | 70000 | k=8，Zelnic-Manor 权重 Eq.(2.2)，训练 2500（3.57%） |

论文报告的关键 accuracy（%，本方法 = Proposed；均为 10 次随机试验平均）：

- **Three Moon（Table 5.2，uniform）**：CVM 98.7 / GL 98.4 / MBO 99.1 / TVRF 98.6 / LapRF 98.4 / **Proposed 99.4**；平均迭代 3.3–3.8。
- **Three Moon（Table 5.3，non-uniform）**：TVRF 97.8 / **Proposed 99.3**——TVRF 掉 0.8%，本方法几乎不掉，体现对训练点分布的鲁棒性（non-uniform 时平均 12.0 次迭代）。
- **COIL（Table 5.4）**：CVM 93.3 / TVRF 92.5 / LapRF 87.7 / GL 91.2 / MBO 91.5 / **Proposed 94.0**；平均迭代 12.2。
- **MNIST（Table 5.5）**：**CVM 97.7**（此处略高）/ TVRF 96.9 / LapRF 96.9 / GL 96.8 / MBO 96.9 / Proposed 97.5（论文措辞"comparable to or better"）。
- **Opt-Digits（Table 5.6，样本率 50/100/150）**：k-NN 85.5/92.0/93.8 … TVRF 95.9/98.3/98.2 / **Proposed 96.6/98.5/98.6**（全面领先）。
- **计时（Table 5.7，秒，括号内平均迭代）**：Three Moon Proposed **0.30 (3.3)** vs TVRF 0.71；COIL 0.76 (11.7) vs 0.65；MNIST 82.04 (9.4) vs 66.00；Opt-Digits 4.45 (9.3) vs 3.42。论文论点：Three Moon 又快又准；大数据集时间同量级，但**考虑 K 路并行后理论可再快约 K 倍**（§5.6）。

读表要点：记录初始化方式（SVM/random）、迭代次数、accuracy 与 CPU time，并区分 uniform vs non-uniform 训练点（Table 5.2 vs 5.3 的鲁棒性实验），而不是只看最终分类率。

## 精读方式

先读 Abstract 和 Section 3 模型；重点看 graph Laplacian L、graph TV 和无约束凸模型；再读 Section 4 primal-dual；最后看 point cloud benchmark。

## 论文证据点

- Abstract
- warm initialization
- graph Laplacian
- graph TV
- unconstrained convex model
- projection to binary partition
- point cloud experiments

## 与其他 14 篇的关系

它把 SaT 从 pixel segmentation 推到 graph classification，是期刊版（第 10 篇 Efficient Variational Classification）的早期版本。

关联论文：#1 SaT 分割方法论总览; #10 高维数据高效变分分类期刊版; #3 多类 ROF 阈值迭代分割

更具体的论述：

- **与 #1（SaT 总览）**：本篇是 SaT "smoothing → thresholding" 两段式范式的**跨域迁移实例**。#1 中 smoothing 解的是图像域凸 ROF/TV 能量（Eq.(8)），thresholding 用 K-means；本篇 smoothing 解的是**图域**凸能量 Eq.(3.5)（fidelity+graph Laplacian+graph TV），thresholding 换成 argmax 投影 Eq.(3.4)。关系是**思想同构、算子换域**：像素网格 → 一般 k-NN 图，∇ → graph gradient，TV → graph TV。
- **与 #10（Efficient Variational Classification 期刊版）**：本篇（arXiv 2019）是其**早期/会议版**，模型骨架与 two-stage 思想一致；期刊版通常扩展实验、加固理论或换 solver。本仓库 runner 两篇共用 `graph_classification.py`，但当前实现**未体现**两篇的真实差异（见复现判断）。
- **与 #3（Iterated/Multiclass ROF 阈值迭代）**：共享"凸模型 + 阈值/投影 + 迭代 refinement"的循环结构。#3 在图像域用 ROF 解 + 阈值迭代；本篇在图域用凸 smoothing + argmax 投影迭代，且每轮 `β←2β` 加速——都是"解一个凸子问题再硬投影，把投影结果当新初始化"的同一类策略。
- **差异提醒**：与 SaT/ROF 的关系是**思想迁移**，模型公式并非一一相同（本篇多了 graph Laplacian ℓ₂ 项与 graph TV ℓ₁ 项的组合，且去掉了 simplex 约束以解耦 K 个子问题）。

## 阅读陷阱

1. **"convex" 的范围**：Eq.(3.5) 的凸性与 Theorem 3.1 的唯一解只针对**第一段 smoothing 子问题**，不代表整体分类问题全局最优；分类全局最优在 simplex 约束下仍 NP-hard。
2. **K 个子问题"独立"的前提**：是因为去掉了 simplex 约束 Eq.(3.3)（`Σ_j u_j=1`）。一旦保留该约束（如 CVM），K 类就耦合、不能独立并行——这正是本方法效率优势的来源，别误以为任何图分类都能解耦。
3. **投影不丢 smoothing 信息**：stage two 的 argmax 投影虽是硬判决，但 smoothing 已把信息编码进 fuzzy 概率的相对大小；且两段**迭代**进行（投影结果回灌为新 Û），逐步纠正初值误差，所以单次投影的"信息损失"会被后续迭代补偿。
4. **β 翻倍的角色**：`β←2β` 是**加速收敛**的工程技巧（增强 fidelity 使相邻迭代结果更接近），不是模型的一部分，别当成理论必需。
5. **MNIST 上 CVM 略高**：Table 5.5 中 CVM 97.7 > Proposed 97.5，论文如实写"comparable to or better"，本方法卖点是**速度 + 鲁棒性 + 可并行**，不是在每个数据集都最高精度。

## 报告扩展字段

- context: 这篇是从 image segmentation 到 graph classification 的迁移入口。它把像素区域分割抽象成图上标签函数分类，目标对象变成 high-dimensional data 和 point clouds。
- technicalReading: 技术路线是 two-stage：先用 SVM 或随机标签做 warm initialization，再在图上解无约束凸变分模型，包含 fidelity、graph Laplacian 和 graph Total Variation (graph TV)，最后投影到 binary partition 或 simplex 顶点。
- theoremReading: 理论阅读要关注为什么去掉 simplex constraint 后 K 个类别子问题可以独立求解，以及 convex smoothing 子问题如何保证可计算性。这里的重点不是证明分类全局最优，而是用凸模型替代 NP-hard 或强约束图分割。
- experimentReading: 实验覆盖 benchmark high-dimensional data sets 和 unstructured point clouds。读表时要记录初始化方式、迭代 refinement、accuracy 和 CPU time，而不是只看最终分类率。
- relationReading: 它把 SaT 的 smoothing + thresholding 迁移成 graph smoothing + projection，是 Efficient Variational Classification 的早期版本。与 SaT/ROF 的关系是思想迁移，不是模型公式一一相同。
- researchValue: 这篇提供了可解释图分类路线，适合发展到医学点云、遥感点云和少标签半监督分类。研究入口在于图构建、graph TV 权重、初始化质量和 projection 误差。

## 阅读问题

1. 为什么 K 个类别子问题可以独立求解？
2. graph Laplacian 与 graph TV 在模型中分别起什么作用？
3. projection 到 binary partition 会不会丢失 smoothing 得到的信息？

## 读后产出

写出 graph variational classification 的三步：warm initialization、convex smoothing、binary projection。

## 复现判断

| 字段 | 内容 |
| --- | --- |
| 复现等级 | partial |
| 真实性等级 | partial-completed |
| 难度 | 高 |
| 效果 | 很明显 |
| 最小实验 | synthetic moons/blobs，warm initialization，kNN graph，graph Laplacian smoothing，argmax projection。 |
| 预期产出 | smoothing + projection improves warm initialization；toy accuracy 从 0.8000 提升到 0.8139。 |
| 依赖 | numpy / scipy / matplotlib |
| 数据需求 | synthetic 2D classification data；不下载 benchmark。 |
| 算力需求 | CPU，约 1 秒内。 |
| 实现风险 | toy 使用 Laplacian smoothing，不是完整 graph TV convex model。 |

### 复现指标

- initial_accuracy
- smoothed_accuracy
- accuracy_gain
- iterations

### 验证计划

比较 warm init 与 smoothing 后 accuracy，并保存 before/after decision colors。

### 当前运行结果

- initial_accuracy: 0.8
- smoothed_accuracy: 0.8139
- accuracy_gain: 0.0139
- iterations: 18

### 结果说明

Toy graph classification: centroid warm initialization, kNN graph smoothing, argmax projection.

## 完整复现流程

本篇的完整复现流程规范（从论文身份核验、Eq.(3.5) 凸模型 + Algorithm 1/2 primal-dual 拆解、四个 benchmark 数据集与公开获取、CVM/GL/MBO/TVRF 等基线、Table 5.2-5.7 报告数值，到本仓库 toy 实现的差距分析与诚实分级）见独立文档：

[`../reproduce/paper_like/workflows/two-stage-classification_reproduction_workflow.md`](../reproduce/paper_like/workflows/two-stage-classification_reproduction_workflow.md)

诚实提醒：当前等级为 **partial**，runner（`graph_classification.py`）用的是 two-moon 二类合成数据 + Laplacian-style 标签传播（缺 graph TV ℓ₁ 项与 primal-dual 求解器），`smoothed_accuracy=0.8139` 是 toy 结果，**不是**论文报告值（论文 Three Moon 99.4% / COIL 94.0% / MNIST 97.5% / Opt-Digits 最高 98.6%）。paper-level 在 15 篇中仍为 0/15。
