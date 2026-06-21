# SLaT 彩色图像三阶段分割 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 7 篇 *SLaT: Smoothing, Lifting and Thresholding* 的完整复现流程规范。

---

## 1. 论文身份与第一作者核验

| 项 | 内容 |
|----|------|
| **标题 (英)** | A Three-stage Approach for Segmenting Degraded Color Images: Smoothing, Lifting and Thresholding (SLaT) |
| **标题 (中)** | SLaT 彩色图像三阶段分割 |
| **作者顺序** | Xiaohao Cai, Raymond Chan, Mila Nikolova, Tieyong Zeng |
| **第一作者核验** | 是。PDF 首页作者列表第一位即 Xiaohao Cai（X. Cai, Department of Plant Sciences & DAMTP, University of Cambridge）。本仓库 15 篇口径要求所有论文 `authors` 均以 `Xiaohao Cai` 开头，本篇满足。 |
| **年份** | 2015（arXiv:1506.00060v1, 30 May 2015） |
| **PDF 路径** | `docs/00_papers_first_author_xiaohao_cai_deduped/SLaT三阶段分割 SLaT Segmentation.pdf` |
| **主题 (theme)** | extension（SaT 灰度两阶段方法到退化彩色图像的扩展） |
| **关联期刊** | Index Terms 标注关键词 Mumford-Shah model, convex variational models, multiphase color image segmentation, color spaces；正式版发表于 *Journal of Scientific Computing*（笔记中原标注 IEEE TIP 系"相关"推测，非确证，详见第 8 节差距与事实性说明）。 |

---

## 2. 复现目标与诚实分级

本项目对"复现"按真实性分四级，纪律为：**禁止把 synthetic / proxy 结果夸大为论文级复现**。

| 分级 | 含义 | 本篇当前状态 |
|------|------|--------------|
| **toy** | 仅示意流程，合成数据、玩具规模、用 proxy 算子代替严格求解 | — |
| **partial** | 部分复现：搭出真实 pipeline 骨架，但关键环节用近似（如 Gaussian smoothing 代替严格凸 Mumford-Shah 求解） | **← 本篇当前等级 = `partial`** |
| **paper-like** | 复现论文数据集与表格量级结果，求解器、基线、指标对齐论文 | 未达 |
| **paper-level** | 与论文逐表逐图对齐、数值落在论文报告区间，可被独立验证 | **未达；全仓库 paper-level 仍为 0/15** |

- 本篇 `reproductionLevel = partial`，`reproductionTruthLevel = partial-completed`（实验确实运行完成并产出图与指标，但用了 proxy）。
- **纪律重申**：当前 partial 结果（升级后 `accuracy_gain = 0.1533`，RGB-only 0.5418 → RGB+Lab 0.6951）只能说明"真实 CIELab lifting 在合成退化图上带来清晰、定性可见的稳健性提升"，**不能**外推为论文 Table I 的 99.21% 平均准确率，也**不能**声称已复现论文级性能。paper-level 复现在全仓库 15 篇中仍为 **0/15**。

---

## 3. 算法完整流程

SLaT 的核心目标（论文 Sec. III）：分割被 **噪声 (noise) / 信息丢失 (information loss) / 模糊 (blur)** 退化的彩色图像 $f=(f_1,\dots,f_d)$，$f:\Omega\to\mathbb{R}^d$，RGB 时 $d=3$，取值约束在 $[0,1]^3$。论文设定 $\mathcal{V}_1=$ RGB，$\mathcal{V}_2=$ Lab。

下面拆为可执行 step-by-step pipeline，忠于 PDF Sec. III-A/B/C 与 Algorithm 1。

### Stage 1 — Smoothing（逐通道凸变分恢复）

对每个通道 $f_i$ 独立求解（论文式 (4)）：

$$
E(g_i)=\frac{\lambda}{2}\int_\Omega \omega_i\cdot\Phi(f_i,g_i)\,dx+\frac{\mu}{2}\int_\Omega|\nabla g_i|^2dx+\int_\Omega|\nabla g_i|\,dx,\quad i=1,\dots,d.
$$

要点：

1. **特征函数 $\omega_i$**（论文式 (5)）：$\omega_i(x)=1$ 当 $x\in\Omega_0^i$（该通道已知像素子集），否则 $=0$。用于处理 **information loss**（缺失像素不进数据项）。
2. **数据保真项 $\Phi$ 两种选择**（论文 Sec. III-A）：
   - i) $\Phi(f,g)=(f-\mathcal{A}g)^2$ —— 高斯噪声 / 一般情形，$\mathcal{A}$ 为模糊算子（处理 **blur**）；
   - ii) $\Phi(f,g)=\mathcal{A}g-f\log(\mathcal{A}g)$ —— 泊松噪声。
3. **三项结构**：数据项（保真）+ $\frac{\mu}{2}\|\nabla g\|_F^2$（$H^1$ 半范，强平滑）+ $\|\nabla g\|_{2,1}$（TV 半范，保边）。这是 Mumford-Shah 模型 (1) 的一个 **凸（convex）非紧松弛**，因此有全局唯一解（见第 3 节末与 Theorem III.1）。
4. **离散模型**（论文 Sec. III-A 离散段）：
   $$E(g_i)=\frac{\lambda}{2}\Psi(f_i,g_i)+\frac{\mu}{2}\|\nabla g_i\|_F^2+\|\nabla g_i\|_{2,1},$$
   其中 $\|\nabla g_i\|_F^2=\sum_{j\in\Omega}\big((\nabla_x g_i)_j^2+(\nabla_y g_i)_j^2\big)$，$\|\nabla g_i\|_{2,1}=\sum_{j\in\Omega}\sqrt{(\nabla_x g_i)_j^2+(\nabla_y g_i)_j^2}$；$\nabla=(\nabla_x,\nabla_y)$ 用后向差分 + Neumann 边界条件离散。
5. **求解器**：论文用 **primal-dual (Chambolle-Pock)**（$\Phi=(f-\mathcal{A}g)^2$ 时）或 **split-Bregman**（$\Phi=\mathcal{A}g-f\log(\mathcal{A}g)$ 时）；亦可用 ADMM。终止条件 $\frac{\|g_i^{(k)}-g_i^{(k+1)}\|_2}{\|g_i^{(k+1)}\|_2}<10^{-4}$ 或达到 200 次迭代。
6. 把每个 $\bar g_i$ rescale 到 $[0,1]$，得 $\bar g=(\bar g_1,\bar g_2,\bar g_3)\in[0,1]^3$。三通道可 **并行 (parallelism)**。

### Stage 2 — Lifting（维度提升到次级颜色空间）

1. 把 Stage 1 的平滑 $\bar g$（RGB, $\mathcal{V}_1$）变换到 **Lab ($\mathcal{V}_2$)** 得 $\bar g'$。论文实现用 MATLAB `makecform('srgb2lab')`。
2. 把 $\bar g'$ 各通道 rescale 到 $[0,1]$ 得 $\bar g^t=(\bar g_1^t,\bar g_2^t,\bar g_3^t)$。
3. **堆叠**得 $2d=6$ 维向量值图像（论文 Sec. III-B）：
   $$\bar g^*:=(\bar g_1,\bar g_2,\bar g_3,\bar g_1^t,\bar g_2^t,\bar g_3^t)\in[0,1]^6.$$
   直觉：Lab 的 $L$ 通道近似感知亮度 (perceived lightness)，$a,b$ 近似 red-green / yellow-blue 色度。当 RGB 三通道高度相关 (highly correlated) 时（论文 Fig. 2 (a)-(c) 显示 Stage 1 后 R/G/B 三通道几乎无法单独支撑分割），Lab 提供 **互补信息 (complementary information)**，这是 SLaT 区别于"只在单一颜色空间分割"的核心。

### Stage 3 — Thresholding（多通道 K-means 阈值化）

1. 在 6 维特征 $\{\bar g^*(x):x\in\Omega\}\subset\mathbb{R}^6$ 上跑 **K-means**（论文式 (7)-(8)）。
2. 聚类中心（每段均值，论文式 (7)）：
   $$c_k=\frac{\int_{\Sigma_k}\bar g^*\,dx}{\int_{\Sigma_k}dx},\quad k=1,\dots,K,\ c_k\in\mathbb{R}^6.$$
3. 用 $\ell_2$ 距离做最终分配（论文式 (8)）：
   $$\Omega_k=\Big\{x\in\Omega:\ \|\bar g^*(x)-c_k\|_2=\min_{1\le j\le K}\|\bar g^*(x)-c_j\|_2\Big\}.$$
4. **关键工程优势**：相位数 $K$ 只在 Stage 3 进入。改变 $K$ 无需重算 Stage 1/2（$\bar g^*$ 预先算好），用户可自由试不同 $K$（论文 Sec. III-D / Algorithm 1）。

### Algorithm 1（论文原文综述）
> 输入：彩色图 $f\in\mathcal{V}_1$、次级空间 $\mathcal{V}_2$。
> 1. Stage one：解式 (4) 得 $\bar g_i$，rescale 到 $[0,1]$，$i=1,2,3$，置 $\bar g=(\bar g_1,\bar g_2,\bar g_3)$。
> 2-4. Stage two：算 $\bar g'\in\mathcal{V}_2$ 得 $\bar g^t$，拼 $\bar g^*=(\bar g_1,\bar g_2,\bar g_3,\bar g_1^t,\bar g_2^t,\bar g_3^t)$。
> 5-6. Stage three：选 $K$，对 $\bar g^*$ 跑 K-means 得 $\{c_k\}$（式 (7)），按式 (8) 得 $\{\Omega_k\}$。

---

## 4. 完整复现所需数据集

论文 Sec. IV（Fig. 3）用 **2 张合成图 + 7 张真实彩色图**，全部 RGB：

| 类别 | 图像 (论文 Fig. 3 编号) | 备注 |
|------|------------------------|------|
| 合成 (synthetic) | (i) 6-phase（5 个重叠彩色圆，size 100×100）、(ii) 4-quadrant（4 矩形 + 变光照, 256×256） | 可程序化生成，**复现优先做这两张**（有 ground-truth，可算准确率）|
| 真实 (real-world) | (iii) Rose、(iv) Sunflower、(v) Pyramid、(vi) Kangaroo、(vii) Vase、(viii) Elephant、(ix) Man | 自然图像；无逐像素 GT，论文主要看视觉与 CPU time |

**退化设定**（论文 Sec. IV）：
- 高斯噪声：mean 0, variance 0.001 或 0.1（MATLAB `imnoise`）；
- 泊松噪声：先线性拉伸到 [1,255]，泊松均值 10，再拉回 [0,1]（`imnoise`）；
- 信息丢失：随机删除 **60%** 像素值；
- 模糊：**vertical motion-blur, 10 pixels length**。

**为达 paper-like 的公开 / 等价候选数据来源**：
- 合成 6-phase / 4-quadrant：直接按论文描述用 numpy 生成（彩色圆叠加 / 彩色矩形 + 线性光照梯度），并自带 ground-truth 标签——**这是最干净的对齐路径**。
- 真实自然图：Rose / Sunflower / Pyramid / Kangaroo / Elephant 等为常见自然图，可用公开自然图像（如 BSDS500、Berkeley、或同名公开素材）作等价替代；论文未公开原始素材包，复现时应**标注为等价替代而非原图**。
- 退化算子可在本地用 numpy/scipy 实现（高斯/泊松噪声、随机掩膜、motion-blur 卷积核），无需私有数据。
- **无私有医学 / RI 数据需求**：本篇全部为公开可生成的合成图 + 自然图，不涉及医学/雷达干涉私有数据。

---

## 5. 对照基线 (Baselines)

论文 Sec. IV 明确对照 **三个 state-of-the-art 变分彩色分割方法**：

| 论文引用 | 方法 | 简述 |
|----------|------|------|
| [31] (Li et al.) | fuzzy membership functions | 近似分段常数 Mumford-Shah 模型 (2) |
| [39] (Pock et al.) | primal-dual + fixed codebook | 解凸松弛模型 (2) |
| [44] (Storath et al.) | ADMM + Potts priors | 解模型 (2)（不含相位数 $K$），convergent minimization |

复现对照时建议：
- **最小可行对照**：把 SLaT 与 "RGB-only（去掉 Lifting 阶段，$\mathcal{V}_2=\varnothing$）" 对比——这正是论文 Fig. 1(b) vs (c) 的核心消融，最能体现 Lifting 价值，且无需实现外部方法。
- **完整对照**：复现 [31]/[39]/[44] 需各自实现（Chan-Vese 类凸松弛、Potts/ADMM 等），工程量大；可优先引用其作者公开代码（论文称 "codes used are provided by the authors"）。

---

## 6. 评价指标与论文报告结果

**指标定义**：
- **Segmentation accuracy（像素准确率）**：正确分割像素数 / 总像素数（论文 Table I 用此，仅合成图有 GT）。
- **Iteration numbers & CPU time（秒）**：迭代数与 CPU 时间（论文 Table II），测试平台为 MacBook 2.4 GHz / 4GB RAM / MATLAB R2014a。

**论文报告的关键数值**（已从 PDF Table I / Table II 核实，注明来源）：

| 来源 | 设定 | Method [31] | Method [39] | Method [44] | **SLaT (Ours)** |
|------|------|------------|------------|------------|-----------------|
| Table I, Fig.4 (A) 去噪 | 6-phase Gaussian noise | 70.11% | **99.53%** | 82.55% | 99.51% |
| Table I, Fig.4 (B) 信息丢失 | + 60% loss | 13.90% | 16.92% | 85.04% | **99.25%** |
| Table I, Fig.4 (C) 模糊+噪声 | blur + noise | 28.08% | 98.58% | 74.77% | **98.88%** |
| Table I, **Average** | 三种退化平均 | 37.36% | 71.68% | 80.79% | **99.21%** |

> 解读（忠于论文 Sec. IV-A）：去噪情形 [39] 以 0.02% 微弱领先；但在信息丢失与模糊+噪声两种更难退化下 SLaT 明显领先，平均准确率 99.21% 远高于所有基线。这说明 SLaT 的优势在 **退化鲁棒性**，而非单一干净去噪。

**CPU time（论文 Table II，单位秒）**：SLaT 平均 **17.67s**，对照 [31] 22.17s / [39] 25.25s / [44] 41.69s；论文强调三通道并行后时间可再约缩 3 倍。

> 纪律提醒：以上数字**全部来自论文 Table I/II**，仅用于"复现目标值"参照，**不是本仓库产出的结果**。本仓库当前 toy 不得报告或暗示已达到这些量级（见第 7 节）。

---

## 7. 本仓库当前复现实现

- **runner 文件**：`reproduce/experiments/slat_color.py`
- **依赖**：`numpy`, `matplotlib`, `scipy`, **`skimage`（scikit-image，用于真实 CIELab 变换）**。缺任一时 runner 走 `skipped`，不伪造 completed。
- **实际做了什么**（partial，已升级为真实 CIELab + 强退化基准）：
  1. **合成场景（贴近论文 motivating case）**：numpy 程序化生成 100×100 的 4-quadrant 合成彩色图，四个区域的 RGB 三元组**高度相关**（base=0.55，沿 red/green/blue 色相微扰 eps=0.012），即四区在 RGB 空间几乎不可分——这正是论文 Fig. 1 / Fig. 2"RGB 通道高度相关导致单一颜色空间分割失败"的设定。带逐像素 ground-truth。
  2. **强退化基准（贴近论文 degraded color 设定）**：依次施加 **高斯模糊**（`gaussian_filter` σ=1.0，作 vertical/isotropic blur 代理）+ **加性高斯噪声**（σ=0.08）+ **60% 随机信息丢失**（`loss_mask`，即 known-pixel 子集对应论文 $\omega_i$ 支撑）。丢失像素用 known-pixel 加权高斯平均（normalized-convolution inpainting）填补，作为 Stage-1 TV/H¹ 外推到缺失像素的代理。
  3. **Stage 1 proxy**：用 `scipy.ndimage.gaussian_filter`（σ=1.1）逐通道平滑——仍是对论文式 (4) 严格凸 Mumford-Shah / TV 求解的 **proxy**（无 TV 保边、无 $\mathcal{A}$ 反卷积、无泊松分支）。
  4. **Stage 2（真实 CIELab，已去代理）**：用 **`skimage.color.rgb2lab` 做真实 sRGB→CIELab 变换**（取代旧版 luminance/rg/yb 的 "Lab-like" 代理），再按 **CIELab 名义范围**（L∈[0,100]、a,b∈[−128,127]）把每个 Lab 通道 rescale 到 [0,1]（数据无关、确定性，避免逐图 min-max 放大噪声），与 RGB 拼成 6 维向量值图像 $\bar g^*$（论文 Sec. III-B）。
  5. **Stage 3**：用 `common.simple_kmeans` 在 3 维 (RGB-only) 与 6 维 (RGB+真实 Lab) 特征上分别 K-means（K=4），用 `clustering_accuracy`（匈牙利匹配像素准确率）评分。
  6. 产图对比 degraded / truth / RGB-only / RGB+Lab。
- **当前 runMetrics**（runner 实测，确定性可复现）：

  | 指标 | 值 |
  |------|----|
  | `rgb_only_accuracy` | 0.5418 |
  | `rgb_lab_accuracy` | 0.6951 |
  | `accuracy_gain` | **0.1533** |
  | `runtime_seconds` | 亚秒级（wall-clock，随运行波动，不作复现指标）|

- **关键改进**：因 RGB 通道高度相关 + 强退化，RGB-only K-means 准确率仅 **0.5418**（接近随机崩溃），而真实 CIELab lifting 把感知色差展开为可分距离，RGB+Lab 升到 **0.6951**，带来 **+0.1533** 清晰且可复现的增益（旧 toy 仅 0.0053）。这定性复现了论文"Lifting 在 RGB 高相关/退化下显著优于单一颜色空间"的核心结论。
- **当前 resultFiles**：`assets/repro/slat_rgb_vs_rgblab.png`
- **runner 自带 `fidelityWarning`**：Stage 1 仍为 Gaussian 代理（无 TV 保边 / 无 $\mathcal{A}$ / 无泊松分支）；信息丢失用 normalized-convolution 填补而非论文 TV/H¹ 外推；Lab 用固定名义范围 rescale（非逐图 min-max）；单张合成图，无真实图数据集，无 [31]/[39]/[44] 外部基线，无 Table I/II 量级对齐——**非 paper-level**。

---

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

| 缺口类别 | 当前 (partial) | paper-like / paper-level 需补 |
|----------|----------------|-------------------------------|
| **Stage 1 求解器** | Gaussian filter proxy（仍是代理） | 实现式 (4) 的凸 Mumford-Shah/TV 求解：primal-dual (Chambolle-Pock) 与 split-Bregman；含 $\mathcal{A}$ 模糊算子与泊松 $\Phi$ 分支；停止准则 ‖g^{(k)}−g^{(k+1)}‖₂/‖g^{(k+1)}‖₂<10⁻⁴ 或 200 迭代 |
| **退化建模** | **已加强**：高斯模糊 + 高斯噪声 + 60% 随机信息丢失（$\omega_i$ 掩膜 + normalized-convolution 填补） | 仍缺：泊松噪声分支、严格 10-px vertical motion-blur 卷积核（当前用各向同性高斯 blur 代理）、把信息丢失外推交给真正的 Stage-1 TV/H¹ 求解而非独立 inpainting |
| **颜色空间 (Stage 2)** | **已升级为真实 CIELab**：`skimage.color.rgb2lab` 做 sRGB→CIELab，按名义范围 rescale 到 [0,1] | 与论文 `makecform('srgb2lab')` 已等价；仅 rescale 策略差异（固定名义范围 vs 论文逐通道 [0,1]）——影响极小，可视作已对齐 |
| **数据集** | 单张 100×100 合成 4-quadrant 图（带 GT） | 论文 6-phase (100×100) + 4-quadrant (256×256) 合成图（带 GT）+ 7 张真实图 |
| **类别数 $K$** | 固定 K=4 | 复现"改 $K$ 不重算 Stage 1/2"的工程特性；6-phase 用 $K=6$ |
| **基线对照** | 仅 RGB-only vs RGB+Lab 内部消融 | 实现/引入 [31] Li、[39] Pock、[44] Storath 三方法 |
| **指标与表格** | 单一 pixel accuracy（rgb_only 0.5418 / rgb_lab 0.6951 / gain 0.1533） | 复现 Table I（三退化 × 4 方法准确率）与 Table II（迭代数 + CPU time），并对齐量级 |
| **元信息核实** | 笔记标"IEEE TIP（相关）" | **事实性待修**：本文正式发表于 *Journal of Scientific Computing*（2017），arXiv 版未标期刊；笔记"IEEE TIP" 应改为"待核实/JSC"以免误导 |

> 升级后到 paper-level 的**剩余关键步骤**（按优先级）：(1) 实现 Stage 1 真正的凸 Mumford-Shah/TV primal-dual + split-Bregman 求解器（含 $\mathcal{A}$ 反卷积、$\omega_i$ 掩膜进数据项、泊松 $\Phi$ 分支），取代当前 Gaussian 代理；(2) 用严格 10-px vertical motion-blur 核 + 泊松噪声补全退化类型；(3) 扩到论文 6-phase + 4-quadrant + 7 张真实图，跑 $K\in\{6,4\}$；(4) 引入 [31]/[39]/[44] 三个外部基线；(5) 复现 Table I/II 的准确率与 CPU time 量级并逐项比对。在 (1)(3)(4)(5) 完成前，本篇维持 partial，全仓库 paper-level 仍为 0/15。

---

## 9. 运行步骤

**当前 toy/partial 运行方式**：

```bash
# 安装依赖（见 reproStructured.dependencies: numpy, scipy, matplotlib）
pip install -r requirements.txt

# 运行全部复现实验（含本篇 slat_color）
cd reproduce && python run_all.py
```

- 缺依赖时 runner 写入 `skipped`（不伪造 completed）。本篇依赖：`numpy`, `scipy`, `matplotlib`。
- 产物：指标写入复现结果 JSON，图写入 `docs/assets/repro/slat_rgb_vs_rgblab.png`。
- 计算需求：CPU，亚秒级（wall-clock 随运行波动，不作复现指标）。

**向 paper-like 扩展的步骤大纲**（不在当前 runner 内，仅规划）：
1. 实现 Stage 1 严格求解器（primal-dual + split-Bregman），加 $\mathcal{A}$ 与 $\omega_i$。
2. 程序化生成 6-phase / 4-quadrant 合成图 + GT，并按论文施加三类退化。
3. Stage 2 换成严格 sRGB→Lab。
4. 跑 $K\in\{6,4\}$，按式 (7)(8) 阈值化。
5. 引入 [31]/[39]/[44] 基线，复现 Table I / Table II 量级，逐项比对。

---

## 10. 风险与代理说明

- **Stage 1 proxy 的局限（仍存在）**：Gaussian filter 是各向同性线性平滑，**会模糊边缘**，且不含 $\mathcal{A}$ 反卷积与泊松数据项；论文式 (4) 的 TV 项专门 **保边 (edge-preserving)**。因此本仓库平滑结果在边界稳健性上**弱于**论文，不能据此评判 SLaT 真实分割质量。这是当前**最主要的剩余代理**。
- **信息丢失填补为代理（仍存在）**：60% 丢失像素由 known-pixel 加权高斯平均（normalized-convolution inpainting）填补，是对论文 Stage-1 TV/H¹ 把缺失像素当 $\omega_i=0$ 外推的**近似**，不是同一求解过程。
- **Lab rescale 策略差异（轻微）**：Stage 2 颜色变换本身已是**真实 CIELab**（`skimage.color.rgb2lab`，等价论文 `srgb2lab`，**不再是 Lab-like 代理**）；唯一与论文的差异是把每个 Lab 通道按**固定名义范围**（L/100、(a+128)/255、(b+128)/255）rescale 到 [0,1]，而非论文的逐图 min-max——这一选择是确定性的、避免噪声放大，对结论影响极小。
- **退化覆盖（已显著改善，但未全）**：当前已覆盖 **高斯模糊 + 高斯噪声 + 60% information loss** 三类退化（含 $\omega_i$ 已知像素掩膜），这是 SLaT 最能体现优势的设定；**仍缺** 泊松噪声与严格 10-px vertical motion-blur 核（当前 blur 为各向同性高斯代理）。
- **结果解读（升级后）**：当前 `accuracy_gain=0.1533`（RGB-only 0.5418 → RGB+Lab 0.6951）在合成退化图上**清晰且可复现地**展示了"真实 CIELab lifting 显著优于单一 RGB 颜色空间"这一论文核心结论；但其**绝对数值仍是合成图上的内部消融**，**不得**被表述为"复现了论文 Table I/II"或"达到论文级性能"。
- **不可外推的结论**：本仓库**任何数值都不得**被外推为论文级 Table I/II（如 Average 99.21%）。paper-level 在全仓库 15 篇中仍为 0/15。

---

## 11. 参考：精读笔记

- 精读笔记：[`../../../xiaohao_cai_ultimate_notes/SLaT_Three-stage_Segmentation_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/SLaT_Three-stage_Segmentation_超精读笔记_已填充.md)
- 论文 PDF：`docs/00_papers_first_author_xiaohao_cai_deduped/SLaT三阶段分割 SLaT Segmentation.pdf`
- 复现 runner：`reproduce/experiments/slat_color.py`
