# 多类 ROF 阈值迭代分割

> 当前 15 篇口径内第 3 篇。本文档按 PDF 首页作者顺序和 dashboard 结构化精读字段重写，避免旧论文笔记混入。

## 论文元信息

| 字段 | 内容 |
| --- | --- |
| 英文标题 | Multiclass Segmentation by Iterated ROF Thresholding |
| 作者顺序 | Xiaohao Cai, Gabriele Steidl |
| 第一作者核验 | 是，PDF 首页作者列表以 Xiaohao Cai 开头 |
| 年份 | 2013 |
| 类型 | LNCS / EMMCVPR |
| PDF | docs/00_papers_first_author_xiaohao_cai_deduped/多类ROF分割 Iterated ROF.pdf |
| 阅读顺序 | 3 / 15 |
| 主题 | sat-rof |
| 难度 | 中等 |

## 一句话贡献

T-ROF 的早期算法雏形。

## 核心问题

多相 PCMS/Chan-Vese 直接优化困难，且灰度值接近的类别容易被合并；论文要用 ROF 阈值化避免直接求解非凸多相分割。

## 为什么难

多类分割要同时估计多个区域和多个均值，类别间灰度差很小时，固定阈值或单次聚类容易失败；非凸模型反复优化又会放大初始化和参数敏感性。

## 方法抓手

T-ROF 先解一次 Rudin-Osher-Fatemi (ROF) 恢复问题，再对同一个 ROF 解做多阈值分割。算法反复根据当前分割区域均值 m_i 更新阈值 τ_i = 1/2(m_{i-1}+m_i)，使阈值自动适配相邻类别；projection 只在收敛证明的 modified algorithm 中出现。

## 关键模型或公式

ROF: min_u TV(u)+μ/2∫(u-f)^2dx; T-ROF: E(Σ,τ)=Σ_i[Per(Σ_i;Ω)+μ∫_{Σ_i}(τ_i-f)dx]; nested segments Ω_i=Σ_i\Σ_{i+1}; threshold update τ_i=1/2(m_{i-1}+m_i).

## 算法流程

1. 初始化有序阈值 τ_i。
2. 先求解一次 ROF 模型得到 u。
3. 按当前阈值令 Σ_i={x:u(x)>τ_i} 并由差集得到 Ω_i。
4. 计算每个 Ω_i 上的均值 m_i。
5. 用 τ_i=1/2(m_{i-1}+m_i) 更新阈值并重复，直到阈值序列收敛。

## 理论保证

论文给出 projected T-ROF algorithm 在 assumption (A) 下阈值序列收敛的定理；K=2 时模型与 Chan-Vese 之间有等价/联系，并带有调整后的正则参数解释。这里的 projection 是收敛证明里的 slight modification，不是数值 Algorithm T-ROF 的核心步骤。

## 实验重点

实验对象包括 cartoon、texture 和 medical images；重点看灰度值相近类别是否被正确分开，以及算法速度相对其他 variational segmentation 方法的差异。

## 精读方式

先读 Abstract 和 Section 2，理解为什么 ROF 与 Chan-Vese 能接上；再读 Algorithm T-ROF 和阈值更新规则；最后看 texture/medical 实验中的失败与成功样例。

## 论文证据点

- Abstract
- Algorithm T-ROF
- threshold update τ_i = 1/2(m_{i-1}+m_i)
- convergence discussion
- Experiments

## 与其他 14 篇的关系

它是 Linkage 论文的算法前身，也是 SaT 方法论中 T-ROF 分支的核心实例。

关联论文：#1 SaT 分割方法论总览; #2 PCMS 与 ROF 的理论连接; #4 分割与恢复耦合模型

## 报告扩展字段

- context: 这篇可以看作 T-ROF 的算法原型，位置在 Linkage 之后最合适。Linkage 告诉你为什么 ROF thresholding 有理论意义，这篇告诉你多类分割时阈值如何自动更新、如何落成可运行算法。
- technicalReading: 技术阅读的抓手是 solve ROF once 和 iterative threshold update 的配合。先用 ROF 平滑输入图像，再根据当前分割计算区域均值 m_i，用 τ_i = 1/2(m_{i-1}+m_i) 更新相邻类阈值。这样多类分割不必直接求解完整非凸 PCMS，也不必在每轮阈值更新时重解 ROF。
- theoremReading: 理论部分应关注 assumption (A) 与 projected T-ROF 的收敛条件，以及 K=2 时与 Chan-Vese 之间的等价或对应关系。要注意 projection 是证明里的 slight modification，不是数值 Algorithm T-ROF 的核心步骤；收敛也不是任意图像任意 K 的全局保证。
- experimentReading: 实验阅读重点是 cartoon、texture、medical images 中灰度接近类别的分割表现。应记录哪些例子是单次阈值化失败而迭代阈值成功，以及速度优势是否来自只求解一次 ROF。
- relationReading: 它是 SaT Overview 中 T-ROF 分支的原始算法来源，也是 Linkage 后续理论化的前身。与 Segmentation Restoration 相比，它保留 two-stage 结构；与 SLaT 相比，它处理灰度/多类阈值，而不是彩色特征 lifting。
- researchValue: 这篇适合提炼可复现算法：输入、ROF 解、区域均值、阈值更新、停止条件都很清楚。读完后可以直接把它改成伪代码或小实验，用来观察 K、噪声、灰度间隔对分割稳定性的影响。

## 阅读问题

1. 为什么 τ_i 要用相邻区域均值的一半和更新？
2. 只解一次 ROF 与迭代更新阈值之间如何配合？
3. T-ROF 在灰度值相近类别上的优势来自模型还是阈值更新？

## 读后产出

画出 T-ROF 阈值更新流程图，并标出 m_i、τ_i 和分割区域的循环关系。

## 复现判断

| 字段 | 内容 |
| --- | --- |
| 复现等级 | partial |
| 真实性等级 | partial-completed |
| 难度 | 中 |
| 效果 | 明显 |
| 最小实验 | close-gray-value (差 0.04) 4-phase synthetic image + K=2 synthetic case；先解一次 Chambolle-Pock ROF，再迭代更新 tau_i = 1/2(m_{i-1}+m_i)，其中 m_i := mean_f(Omega_i) on raw f。 |
| 预期产出 | partial 复现显示 ROF T-ROF 可从 raw K-means 0.5650 提升到 0.9463；Gaussian proxy baseline 为 0.9438，Split-Bregman 对照为 0.9510。 |
| 依赖 | numpy / scipy / matplotlib |
| 数据需求 | synthetic close-gray-value 4-phase image + K=2 synthetic two-phase image；不需要下载真实数据。 |
| 算力需求 | CPU，约 1 秒内。 |
| 实现风险 | partial 升级已用 ROF solver 替代 Gaussian proxy 主路径，并按 Eq. (15) 用 raw f 计算 mean_f(Omega_i)；仍不等同于 Theorem 1 完整证明或论文真实数据实验。 |

### 复现指标

- raw_kmeans_accuracy
- gaussian_proxy_trof_accuracy
- rof_trof_accuracy
- split_bregman_trof_accuracy
- threshold_iterations
- max_threshold_drift
- monotonicity_violated
- sign_changes_final
- sign_changes_nonincreasing
- assumption_a_violations
- rof_iterations_chambolle_pock
- rof_iterations_split_bregman
- k2_lambda_derived
- k2_rof_threshold_dice
- k2_chanvese_proxy_dice
- k2_segmentation_disagreement
- runtime_seconds

### 验证计划

对比 raw K-means / Gaussian proxy T-ROF / Chambolle-Pock ROF T-ROF / Split-Bregman T-ROF；检查 Lemma 2 单调性、Lemma 3 sign changes、K=2 lambda 关系，并保存 threshold history、convergence 与 Chan-Vese proxy 图。

### 当前运行结果

- raw_kmeans_accuracy: 0.565
- gaussian_proxy_trof_accuracy: 0.9438
- rof_trof_accuracy: 0.9463
- split_bregman_trof_accuracy: 0.951
- threshold_iterations: 3
- monotonicity_violated: false
- sign_changes_final: 0
- assumption_a_violations: 0
- k2_rof_threshold_dice: 0.9976
- k2_chanvese_proxy_dice: 0.8994

### 结果说明

Partial reproduction: Chambolle-Pock ROF + iterative threshold update with raw-image mean_f(Omega_i) per Eq. (15), Lemma 2/3 checks, and a K=2 Proposition 2 proxy check. A Gaussian proxy T-ROF baseline is preserved for comparison; still toy/partial, not paper-level.
