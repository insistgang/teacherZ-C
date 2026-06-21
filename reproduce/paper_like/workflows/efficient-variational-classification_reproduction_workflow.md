# 高维数据高效变分分类期刊版 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 10 篇 *An Efficient and Versatile Variational Method for High-Dimensional Data Classification* 的完整复现流程规范。

## 1. 论文身份与第一作者核验

| 项目 | 内容 |
|------|------|
| 标题 (EN) | An Efficient and Versatile Variational Method for High-Dimensional Data Classification |
| 标题 (CN) | 高维数据高效变分分类方法（期刊版） |
| 作者顺序 | **Xiaohao Cai**, Raymond H. Chan, Xiaoyu Xie, Tieyong Zeng |
| 第一作者核验 | 是。PDF 首页作者列表以 Xiaohao Cai (School of Electronics and Computer Science, University of Southampton) 开头，确认为本项目 15 篇口径内第一作者论文。 |
| 年份 / 出处 | 2024，Journal of Scientific Computing (2024) 100:81，DOI 10.1007/s10915-024-02644-9 |
| 投稿/接收 | Received 16 Oct 2023 / Revised 6 Jun 2024 / Accepted 24 Jul 2024 / Published 1 Aug 2024 |
| PDF 路径 | `docs/00_papers_first_author_xiaohao_cai_deduped/高效变分分类 Efficient Variational.pdf` |
| 主题 (theme) | classification（半监督多类分类 / 点云分类） |
| 关键词 | Semi-supervised clustering · Point cloud classification · Variational methods · Graph Laplacian |

本篇是高维分类线的"期刊成熟版"，把早期 two-stage / SaT 思想补成更完整的无约束凸模型、唯一解定理、specifically designed primal-dual 算法与系统 benchmark（4 个数据集 + 一类分类扩展）。

## 2. 复现目标与诚实分级

本项目对"复现"采用四级诚实分级，禁止把 synthetic/proxy 结果夸大为论文级复现：

- **toy**：用合成数据演示"机制方向"，不追求论文数值。
- **partial**：实现部分核心机制（如重复 smoothing + projection 的迭代精炼思想），但缺少完整 graph TV primal-dual 求解器与论文 benchmark 数据。
- **paper-like**：用论文同源或等价公开数据 + 论文算法骨架，复现趋势与量级（接近但不要求逐位对齐）。
- **paper-level**：完全复现论文 Table 2/4/5/6/7 的数据集、求解器、基线与数值。

**本仓库当前等级**：`reproductionLevel = partial`，真实性 `reproductionTruthLevel = partial-completed`。
**纪律声明**：本项目 paper-level 复现仍为 **0/15**。本篇当前实现只演示"K 个标签函数独立更新 + argmax projection + 重复迭代精炼"的机制，**没有**实现严格的 graph total variation (TV) primal-dual 求解器，也**没有**使用论文 benchmark（Three Moon / COIL / Opt-Digits / MNIST）。当前指标只能在 toy two-moons 数据上解读，不可外推到论文级。

## 3. 算法完整流程

论文方法受 SaT (Smoothing-and-Thresholding) 分割方法学启发，整体是"warm initialization → 凸 smoothing → binary projection → 迭代精炼"。以下为忠于 PDF 的 step-by-step pipeline。

**符号**（Section 2）：点云 $V$ 含 $N$ 个点（每个点 $\in \mathbb{R}^M$），类别数 $K$，训练集 $T=\{T_j\}_{j=1}^K$，$|T|=N_T$，测试集 $S=V\setminus T$。partition matrix $U=(u_1,\dots,u_K)\in\mathbb{R}^{N\times K}$，$u_j(x)\in\{0,1\}$ 指示 $x$ 是否属于第 $j$ 类（Eq. 12）。凸松弛为 unit simplex 约束 $\sum_j u_j(x)=1,\ u_j(x)\in[0,1]$（Eq. 13），但论文**最终模型不带这个约束**。

**Step 0 — 图与算子构建（Section 2）**
1. 用 $k$-NN（randomized kd-tree, Euclidean 距离）构建稀疏图，替代全连接图以省算力（Eq. 8）。
2. 权重函数三选一：radial basis function $w(x,y)=\exp(-d(x,y)^2/(2\xi))$（Eq. 1）、Zelnik-Manor & Perona $w(x,y)=\exp(-d(x,y)^2/(\mathrm{var}(x)\mathrm{var}(y)))$（Eq. 2）、cosine similarity（Eq. 3）。
3. 度矩阵 $D$（Eq. 4），graph Laplacian $L=D-W$，梯度算子 $\nabla u(x)=(w(x,y)[u(x)-u(y)])_{(x,y)\in E}$（Eq. 5/8）。
4. $\ell_1$ 范数 $\|\nabla u\|_1=\sum_{(x,y)\in E}|w(x,y)[u(x)-u(y)]|$（Eq. 6/9）；$\ell_2$（Dirichlet energy）$\|\nabla u\|_2^2=\tfrac12 u^\top L u$（Eq. 7/10）。

**Step 1 — Warm initialization（Section 3.2 Initialization）**
5. 用 standard SVM（linear kernel）做一次快速分类，得到 partition matrix $\hat U=(\hat u_1,\dots,\hat u_K)$。若数据太大或无合适方法，可随机赋标签作为初始化。其精度**不关键**，后续步骤会显著改善。

**Step 2 — 凸 smoothing（Step one，核心，Eq. 15）**
6. 求解无约束凸模型：
$$\arg\min_U \sum_{j=1}^K \left\{ \frac{\beta}{2}\|u_j-\hat u_j\|_2^2 + \frac{\alpha}{2} u_j^\top L u_j + \|\nabla u_j\|_1 \right\}.$$
三项分别为：数据保真项（约束 fuzzy partition 不偏离 warm init）、graph Laplacian 平滑项（$\ell_2$，使标签平滑）、graph TV 项（$\ell_1$，使相似点聚成段）。
7. **关键性质**：模型含 $K$ 个**相互独立**的子问题（每个 $u_j$ 一个），无 simplex 约束，因此天然可并行（Section 3.2、3.7 强调），并避免 NP-hard。训练点标签固定 $\hat u_j(x)=\bar u_j(x),\ \forall x\in T$（Eq. 16），只对测试集 $S$ 求解（Eq. 17/18）。

**Step 3 — Binary projection（Step two，Eq. 14）**
8. 把 fuzzy $U$ 投影到 unit simplex 的顶点得到硬分类：$(u_1(x),\dots,u_K(x))\mapsto e_i,\ i=\arg\max_j u_j(x)$。该步开销可忽略，且结果自动满足 no-vacuum-and-overlap 约束（Eq. 11）。

**Step 4 — 迭代精炼（Algorithm 1）**
9. 把 step two 的 binary $U^{(l+1)}$ 当作新的初始化 $\hat U=U^{(l+1)}$，并令 $\beta=2\beta$（每轮翻倍以加速收敛、强化与上一轮一致性）。
10. 重复 Step 2–3，直到标签不再变化（$\|U^{(l)}-U^{(l+1)}\|=0$）。论文报告一般 $l\le 10$（实际平均 3.3–12.2 次，依数据集与采样方式而定）。
11. 输出 $U^*=U^{(l+1)}$。

**Step 2 的内部求解 — Primal-dual（Section 4）**
- saddle-point 形式（Eq. 19）：$\min_x\max_{\tilde x}\{\langle\mathcal Kx,\tilde x\rangle+\mathcal G(x)-\mathcal F^*(\tilde x)\}$。
- 迭代（Eqs. 20–22）：
$$\tilde x^{(l+1)}=(I+\sigma\partial\mathcal F^*)^{-1}(\tilde x^{(l)}+\sigma\mathcal Kz^{(l)}),$$
$$x^{(l+1)}=(I+\tau\partial\mathcal G)^{-1}(x^{(l)}-\tau\mathcal K^*\tilde x^{(l+1)}),$$
$$z^{(l+1)}=x^{(l+1)}+\theta(x^{(l+1)}-x^{(l)}).$$
- 论文把 $L$ 按训练/测试边分解（Eqs. 24–31，$L=\begin{psmallmatrix}L_S+L_1 & L_3\\ L_3^\top & \bar L+L_2\end{psmallmatrix}$），把梯度算子拆成测试集算子 $\mathcal A_S$ 与固定训练部分 $H_j$（Eqs. 32–33），得到只对 $u_{S_j}$ 求解的模型（Eq. 34/35）。
- $\mathcal F_j^*$ 的 proximal 退化为对集合 $P=\{p:\|p\|_\infty\le1\}$ 的逐点投影 $\iota_P$（Eqs. 37–40）；$\mathcal G_j$ 的 proximal 退化为解线性系统 $(\alpha L_S+\beta I+\tfrac1\tau I)u_{S_j}=\beta\hat u_{S_j}+\tfrac1\tau x-\alpha L_3\bar u_j$（Eq. 42），系数正定，可用 conjugate gradient 高效求解。
- 整体求解模型即 **Algorithm 2**（$K$ 个子问题可并行，自适应更新 $\theta^{(l)}=1/\sqrt{1+\beta\tau^{(l)}}$、$\tau,\sigma$）。

**理论保证**
- **Theorem 1**：给定 $\hat U$ 与 $\alpha,\beta>0$，模型 (15) 是 strongly convex 的，故有**唯一解**。
- **Lemma 2**：$\mathcal G_j$ 以参数 $\beta$ strongly convex。
- **Theorem 3**：若 $\tau^{(0)}\sigma^{(0)}<\tfrac{1}{N^2(k-1)}$，Algorithm 2 收敛（基于 $\|\mathcal A_S\|_2\le N\sqrt{k-1}$ 的上界）。
- 注意：唯一解/收敛覆盖的是 **step one 的连续/松弛 fuzzy 标签优化**；step two 投影后的 hard labels 没有全局最优保证（这是阅读陷阱，见笔记）。

## 4. 完整复现所需数据集

论文在 4 个 benchmark 上测试（Table 1）。要达到 paper-like 需要等价公开数据：

| 数据集 | 类数 K | 维度 dim | 点数 N | 公开/等价来源 | 备注 |
|--------|--------|----------|--------|----------------|------|
| Three Moon | 3 | 100 | 1500 | **可合成**：两个上半单位圆 + 一个半径 1.5 的下半圆，中心 (0,0)/(3,0)/(1.5,0.4)，每弧采 500 点嵌入 $\mathbb{R}^{100}$（其余维补零），加 i.i.d. 高斯噪声 std=0.14。$k=10,\ \xi=3$，RBF 权重，Euclidean。 | 与 [1,2] 完全一致的构造法，**无需私有数据**。 |
| COIL | 6 | 241 | 1500 | Columbia Object Image Library（[58] supplementary）。红通道下采样到 16×16，随机 mask 15 像素得 241 维。$k=4,\ \xi=250$，RBF。 | 公开。 |
| Opt-Digits | 10 | 64 | 5620 | UCI Optical Recognition of Handwritten Digits（archive.ics.uci.edu）。32×32 bitmap 分 4×4 block，得 8×8 整数矩阵。$k=8,\ \xi=30$，RBF。 | 公开。 |
| MNIST | 10 | 784 | 70000 | MNIST（yann.lecun.com/exdb/mnist）。28×28。$k=8$，Zelnik-Manor & Perona 权重 (Eq. 2)。训练取 2500 (3.57%)。 | 公开。 |

- 训练集 $T$：从每个数据集**随机**选少量带标签点（可少于 1%），并显著类别不平衡。Three Moon 用 75 点；Opt-Digits 测 50/100/150 三档；COIL 取 10%；MNIST 取 2500。
- 一类分类扩展（Section 5.6）：每数据集构造 true:outlier = 2:1 与 1:1 两种比例（Table 8 给出具体样本数）。

## 5. 对照基线 (Baselines)

论文与下列 state-of-the-art 半监督/变分/图方法对比（深度学习方法因需大量带特征样本、不在本文比较范围）：

- **CVM** [1]：constrained variational method（带 simplex 约束）。
- **GL** [7]：Ginzburg-Landau / diffuse-interface。
- **MBO** [7]：Merriman-Bence-Osher 阈值动力学。
- **TVRF** [2]：TV + region-force 变分法（作者提供代码、trial-and-error 调参，是主要对照）。
- **LapRF** [2]、**LapRLS** [60]：Laplacian regularized 方法。
- **MP** [60]、**SQ-Loss-I** [58]：图半监督方法。
- **k-NN**、**SGT** [Table 6]：Opt-Digits 上的额外对照。

## 6. 评价指标与论文报告结果

**指标定义**：classification accuracy = 正确分类点占比（test set 上）；computation time（秒）；average number of iterations。所有方法跑 10 次取平均（Three Moon/COIL/Opt-Digits/MNIST），并报告 std。

**论文报告的关键数值**（均来自 PDF，标注表号；proposed = 本文方法）：

- **Table 2（Three Moon，均匀采样训练点）**：CVM 98.7 / GL 98.4 / MBO 99.1 / TVRF 98.6 / LapRF 98.4 / **Proposed 99.4**（最高，加粗）。平均迭代 3.8。
- **Table 3（Three Moon，非均匀采样训练点）**：TVRF 97.8 / **Proposed 99.3**（std 0.11%）。TVRF 较均匀采样下降 0.8%，本文仅降 0.1%，显示鲁棒性；非均匀时平均 12.0 次迭代 vs 均匀 3.3 次。
- **Table 4（COIL）**：CVM 93.3 / TVRF 92.5 / LapRF 87.7 / GL 91.2 / MBO 91.5 / **Proposed 94.0**（最高，std 0.84%）。平均迭代 12.2。
- **Table 5（MNIST）**：**CVM 97.7（最高）** / TVRF 96.9 / LapRF 96.9 / GL 96.8 / MBO 96.9 / Proposed 97.4。注意：MNIST 上 proposed 是 comparable，**不是**最高，CVM 略高。std 0.03%。
- **Table 6（Opt-Digits，三档采样率 0.89%/1.78%/2.67%）**：Proposed **97.0 / 98.4 / 98.5**，三档均为最高（对照含 k-NN 85.5/92.0/93.8、SGT、LapRLS、SQ-Loss-I、MP、TVRF、LapRF）。std 1.25%/0.53%/0.28%。
- **Table 7（计算时间，秒；括号为平均迭代次数）**：Three Moon TVRF 0.71 / **Proposed 0.30 (3.3)**；COIL TVRF 0.65 / Proposed 0.76 (11.7)；MNIST TVRF 66.00 / Proposed 82.04 (9.4)；Opt-Digits TVRF 3.42 / Proposed 4.45 (9.3)。论文称其方法在多类问题上至少比 multi-class MBO 快 10 倍。
- **Table 8（一类分类）**：Three Moon 99.58 (2:1) / 99.62 (1:1)；COIL 91.30 / 94.42；MNIST 99.80 / 99.81；Opt-Digits 99.97 / 99.97。

**参数缺省**：$\alpha=1$，初始 $\beta=0.01$（COIL 需更小初始 $\beta$），对 $\alpha\in[0.5,2]$、$\beta\in[0.001,0.1]$ 鲁棒。环境：MATLAB 2017a，MacBook 2.8 GHz，16 GB RAM。

**禁止编造**：以上数值全部可在 PDF 对应表号核实；任何未在表中出现的数值不得写入复现产出。

## 7. 本仓库当前复现实现

- **runnerFile**：`reproduce/experiments/graph_classification.py`（与第 9 篇 two-stage-classification 共用同一 runner，返回两条 completed 记录）。
- **实际做了什么**：
  1. 用 `numpy` 生成 **two-moons**（注意：是 two-moons，**不是论文的 Three Moon**）合成数据，360 点，加噪 std=0.18，每类各 10 个带标签点。
  2. warm init 用**类质心最近邻**（centroid nearest，**不是论文的 SVM**）。
  3. 用 `scipy.spatial.cKDTree` 建 $k=13$ 的 kNN 图（对称化，0/1 权重，**非 RBF/ZMP 权重**）。
  4. 迭代 18 次"邻居平均 + 阻尼" `probs = 0.72*probs + 0.28*neighbor_avg`，固定带标签点，再 `argmax` 投影。这是**线性 label-propagation / Laplacian 平滑代理**，**不是** graph TV primal-dual。
  5. 画 warm init / smoothing / accuracy trace 三联图。
- **proxy 说明**：以邻居平均 (graph Laplacian-like diffusion) 代替严格 $\ell_2+\ell_1$ 变分模型；以 0/1 邻接代替加权图；以质心初始化代替 SVM；缺 $\beta$ 翻倍的外层迭代精炼与唯一解/收敛理论。
- **当前 runMetrics**（来自 `reproStructured`，toy two-moons）：

| 指标 | 数值 |
|------|------|
| initial_accuracy | 0.8 |
| smoothed_accuracy | 0.8139 |
| accuracy_gain | 0.0139 |
| iterations | 18 |
| runtime_seconds | 0.0904 |

- **resultFiles**：`assets/repro/graph_classification_before_after.png`。
- **notes**（代理说明，dashboard 原文）："Toy repeated graph smoothing: demonstrates independent label-function update idea without full graph TV primal-dual solver."

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

| 维度 | 当前 | paper-like 需要 | paper-level 需要 |
|------|------|------------------|-------------------|
| 数据 | toy two-moons (360 pts, $\mathbb{R}^2$) | 至少 Three Moon (合成, 1500 pts, $\mathbb{R}^{100}$) + Opt-Digits/COIL 之一 | Three Moon + COIL + Opt-Digits + MNIST 四套 |
| 权重函数 | 0/1 邻接 | RBF (Eq. 1) 带正确 $\xi$ | 按数据切换 RBF / ZMP (Eq. 2) |
| 初始化 | 类质心最近邻 | linear SVM（论文做法） | SVM + 随机初始化两种 |
| smoothing 模型 | 邻居平均阻尼（纯 $\ell_2$ 扩散代理） | 含 $\ell_1$ graph TV 的凸模型 (Eq. 15) | 完整 Eq. 15 + Eq. 34 分解 |
| 求解器 | 无（迭代平均） | Chambolle-Pock primal-dual (Eqs. 20–22, 39, 42) | Algorithm 2 + CG 解 Eq. 42 + 收敛监控 |
| 外层迭代精炼 | 单层 18 次平均 | Algorithm 1：projection→新 init→$\beta=2\beta$，停在 $\|U^{(l)}-U^{(l+1)}\|=0$ | 同 + 记录平均迭代次数对照 Table 7 |
| 并行 | 无 | $K$ 子问题独立求解 | 并行计时以验证速度优势 |
| 基线 | 无 | TVRF（作者提供代码） | CVM/GL/MBO/TVRF/LapRF/LapRLS/MP/SQ-Loss-I/k-NN/SGT |
| 指标对照 | accuracy gain (toy) | accuracy + iterations 趋势对齐 | 复现 Table 2/3/4/5/6/7/8 数值与 std |
| 评价口径 | toy 0.8→0.8139 | Three Moon 接近 99% 量级 | 逐表逐档数值对齐 |
| 一类分类 | 无 | 跳过 | Table 8 全部 (2:1 / 1:1) |

## 9. 运行步骤

**当前 toy/partial 跑法**：

```bash
# 安装依赖（reproStructured.dependencies）
pip install -r requirements.txt   # numpy, scipy, matplotlib（本实验额外用到 scipy.spatial）

# 运行全部复现实验（含本篇 graph_classification）
cd reproduce && python run_all.py

# 校验 15 篇数据、PDF、笔记与静态复现资产
node docs/scripts/validate.mjs
```

产物：`docs/assets/repro/graph_classification_before_after.png`，指标写入 repro 结果 JSON（initial/smoothed accuracy、gain、iterations）。

**向 paper-like 扩展的步骤大纲**（不在本次写作范围，仅规划）：
1. 新增独立 runner（不要污染现有 toy），实现 Three Moon 合成（$\mathbb{R}^{100}$, std=0.14, $k=10$, $\xi=3$, RBF）。
2. 用 `sklearn.svm.LinearSVC` 做 warm init（替代质心）。
3. 实现 Eq. 15 的 Chambolle-Pock primal-dual（Eqs. 20–22；$\mathcal F^*$ 投影 Eq. 39/40，$\mathcal G$ 线性系统 Eq. 42 用 `scipy.sparse.linalg.cg`），稀疏 $L$ 用 `scipy.sparse`。
4. 实现 Algorithm 1 外层：argmax projection（Eq. 14）→ 新 init → $\beta=2\beta$ → 停在标签不变。
5. 记录 accuracy、平均迭代、运行时间，与论文 Table 2/7 **趋势**对照（标注 paper-like，不声称 paper-level）。
6. 逐步加入 Opt-Digits（UCI）、COIL、MNIST 与 TVRF 基线，向 paper-level 推进。

## 10. 风险与代理说明

- **proxy 的局限**：当前实现是 graph Laplacian-like 扩散 + argmax，**缺 $\ell_1$ graph TV 项**，因此没有 TV 带来的 piecewise-constant / 抗 staircase 效果；论文明确指出额外 $\ell_2$ 项正是为减少纯 $\ell_1$ 的 staircase artifact，这一对照在 toy 中无法体现。
- **数据不可比**：toy two-moons 在 $\mathbb{R}^2$、360 点，与论文 $\mathbb{R}^{64\sim784}$、1500–70000 点的高维高相似 benchmark 完全不同；toy 的 0.8→0.8139 gain **不可**与论文 99% 级 accuracy 或 Table 7 计时类比。
- **理论未实现**：Theorem 1（唯一解）、Theorem 3（收敛条件 $\tau^{(0)}\sigma^{(0)}<1/(N^2(k-1))$）依赖真正的凸模型与 primal-dual，当前代理未涉及，故收敛/唯一性结论不能由本实现验证。
- **初始化差异**：论文用 SVM 且强调"init 精度不关键、迭代会改善"；toy 用质心初始化，且只单层迭代，无法复现 Algorithm 1 的 $\beta$ 翻倍精炼带来的稳定提升。
- **不能外推的结论**：不得据 toy 结果声称"复现了论文的 accuracy / 速度优势 / 一类分类鲁棒性"；这些均属 paper-level，本仓库为 0/15。

## 11. 参考：精读笔记

- 精读笔记：[../../../xiaohao_cai_ultimate_notes/高效变分分类方法_Efficient_Variational_Classification_超精读笔记_已填充.md](../../../xiaohao_cai_ultimate_notes/高效变分分类方法_Efficient_Variational_Classification_超精读笔记_已填充.md)
- 论文 PDF：`docs/00_papers_first_author_xiaohao_cai_deduped/高效变分分类 Efficient Variational.pdf`
- 复现代码：`reproduce/experiments/graph_classification.py`
</content>
</invoke>
