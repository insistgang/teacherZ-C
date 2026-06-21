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

### 公式逐项拆解（忠于 PDF Eq. 编号）

| 公式 | PDF 出处 | 含义与逐项解释 |
| --- | --- | --- |
| E_MS(Γ,u)=H¹(Γ)+μ∫_{Ω\Γ}\|∇u\|²+λ∫_Ω(u-f)² | 引言 | Mumford-Shah：长度项 + 区域内平滑项 + 数据项；非凸难解。 |
| E_PCMS(Γ,u)=H¹(Γ)+λ∫_Ω(u-f)² | Eq. (1) | 限制 ∇u=0（piecewise constant），即 Potts/PCMS。 |
| E_PCMS(Ω,m)=½Σ Per(Ω_i;Ω)+λΣ∫_{Ω_i}(m_i-f)² | Eq. (2) | 分块常值改写：周长项 + 各区域到均值 m_i 的方差项。 |
| E_CV(Ω₁,m₀,m₁)=Per(Ω₁;Ω)+λ(∫_{Ω₁}(m₁-f)²+∫_{Ω\Ω₁}(m₀-f)²) | Eq. (3) | K=2 即 Chan-Vese（active contours without edges）。 |
| E(Σ,τ)=Per(Σ;Ω)+μ∫_Σ(τ-f)dx | Eq. (7) | **本文 K=2 能量**：注意数据项是线性的 (τ-f)，不是平方差；τ∈(0,1)。 |
| τ*=½(mean_f(Σ*)+mean_f(Ω\Σ*)) | Eq. (8) | 最优阈值落在两侧均值的正中间——阈值更新规则的根。 |
| min_{u,0≤u≤1} TV(u)+μ∫_Ω(τ-f)u dx | Eq. (10) | Proposition 1：固定 τ 时 (7) 的**凸松弛**，解后阈值 {u>ρ} 即得 Σ。 |
| λ=μ/(2(m₁*-m₀*)) | Proposition 2 | K=2 时本文模型 ↔ Chan-Vese 的参数桥；m₁-m₀ 越小 λ 越大。 |
| {x:u(x)>τ} 解 (7) ⇔ u 解 ROF (11) | Proposition 3 | **"只解一次 ROF"的理论依据**：阈值化 ROF 解即得分割。 |
| E(Σ,τ)=Σ_{i=1}^{K-1}(Per(Σ_i;Ω)+μ∫_{Σ_i}(τ_i-f)) | Eq. (12) | 多类能量：K-1 个嵌套水平集叠加。 |
| Ω⊇Σ_{τ₁}⊇…⊇Σ_{τ_{K-1}}⊇∅ | Eq. (13) | 由 Lemma 1：阈值递增 ⇒ 水平集嵌套递缩。 |
| Ω_i=Σ_i\Σ_{i+1}, Σ₀=Ω, Σ_K=∅ | Eq. (14) | wanted segments：相邻两层水平集的差集即第 i 类区域。 |
| τ_i*=½(m_{i-1}*+m_i*), m_i*=mean_f(Ω_i*) | Eq. (15) | **核心阈值更新**：均值在 raw image f 上取，而非平滑后的 u。 |

要点（容易读错的地方）：
- **数据项是线性的**。Eq. (7) 用 ∫_Σ(τ-f)dx 而非 ∫(u-τ)²，这使得固定 τ 后能量对 Σ 的依赖可凸松弛为 Eq. (10)，并与 ROF 接上（Proposition 3）。
- **mean_f 在 raw f 上算**。Eq. (15) 的 m_i=mean_f(Ω_i)，区域 Ω_i 由 ROF 解 u 的水平集划定，但**均值取自原图 f**。本仓库复现严格按此实现（见 runner 的 `run_trof_thresholds(raw_image=image4)`）。
- **τ 在两均值正中间**。τ_i=½(m_{i-1}+m_i) 让阈值随当前分割自适应漂移到相邻两类均值的中点，这是处理"相近灰度类别"的关键，而非来自更复杂的模型。

## 算法流程

1. 初始化有序阈值 τ_i。
2. 先求解一次 ROF 模型得到 u。
3. 按当前阈值令 Σ_i={x:u(x)>τ_i} 并由差集得到 Ω_i。
4. 计算每个 Ω_i 上的均值 m_i。
5. 用 τ_i=1/2(m_{i-1}+m_i) 更新阈值并重复，直到阈值序列收敛。

### Algorithm (T-ROF) 逐步展开（忠于 PDF Section 3）

```
Initialization: τ^(0) = (τ_i^(0))_{i=1}^{K-1}, 0 < τ_1^(0) < ... < τ_{K-1}^(0) < 1.
                论文用 fuzzy C-means [7]（100 迭代步）计算初始 τ^(0)。

1. Compute u = argmin ROF(u) 一次（Eq. 11），全过程不再重解。
2. For k = 0,1,2,...:
   2.1 Σ_i^(k) = { x : u(x) > τ_i^(k) },  i=1,...,K-1.           # K-1 次阈值化
   2.2 Ω_i^(k) = Σ_i^(k) \ Σ_{i+1}^(k),  Σ_0=Ω, Σ_K=∅.          # 取差集得到各类区域
   2.3 m_i^(k) = mean_f(Ω_i^(k)),  i=0,...,K-1.                  # 在 raw f 上取均值
   2.4 τ_i^(k+1) = ½(m_{i-1}^(k) + m_i^(k)),  i=1,...,K-1.       # 阈值更新
   直到 ||u^(i)-u^(i-1)||_2/||u^(i)||_2 ≤ ε_u 且 ||τ^(k)-τ^(k-1)||_2 ≤ ε_τ。
```

论文实现细节（Section 4 首段，PDF p.244-246）：
- 离散 ROF 用 [13]；论文数值实验实际用 **ADMM**（fixed inner parameter 2）求 ROF 极小元，并说明用更精巧方法加速留作 future work。
- 停止阈值 **ε_u = 10⁻⁴**（u 的相对变化），**ε_τ = 10⁻⁵**（τ 的变化）。
- 初始阈值由 fuzzy C-means（100 步）给出，不是随机或等分。

为什么"只解一次 ROF"成立：Proposition 3 证明 {x:u(x)>τ} 解 K=2 能量 (7) 当且仅当 u 解 ROF (11)。ROF 极小元 u 与阈值 τ 无关，因此整个外层 τ 迭代只是在**同一个 u** 上反复移动 K-1 条阈值线——这正是论文强调"efficient"的来源（Table 1 中 T-ROF 的 ROF 内迭代固定为 84 步，外层 τ 更新仅 4-5 次）。

## 理论保证

论文给出 projected T-ROF algorithm 在 assumption (A) 下阈值序列收敛的定理；K=2 时模型与 Chan-Vese 之间有等价/联系，并带有调整后的正则参数解释。这里的 projection 是收敛证明里的 slight modification，不是数值 Algorithm T-ROF 的核心步骤。

### 收敛证明链条（直觉版，忠于 PDF p.242-244）

- **Assumption (A)（Eq. 16）**：若 Σ_τ、Σ_τ̄ 分别是 E(·,τ)、E(·,τ̄) 的极小集（0<τ<τ̄<1），则 τ ≤ mean_f(Σ_τ\Σ_τ̄) ≤ τ̄。直觉：相邻两层水平集之间那条"环带"的灰度均值，被夹在两条阈值之间。论文指出右不等式在"Σ_τ̄ 也是 Per(Σ;Σ_τ)+μ∫_Σ(τ̄-f) 的极小集"时成立，左不等式在对称条件下成立——这是合理但**非自动**的假设。
- **Lemma 2（单调性/交错性）**：在 (A) 下，T-ROF 产生 0≤m₀^(k)≤τ₁^(k)≤m₁^(k)≤…≤τ_{K-1}^(k)≤m_{K-1}^(k)（i 部分）；ii 部分给出"阈值升 ⇒ 对应均值升"的传递关系（设 τ₀=0、τ_K=1）。这保证序列始终"有序交错"，不会塌缩成同一类。
- **Lemma 3（sign changes 单调不增）**：定义符号向量 ζ^(k)（第 i 条阈值相对上一步升取 +1、降取 -1），s_k 为 ζ^(k) 中相邻符号翻转的次数。Lemma 3 证 s_k 关于 k 单调不增，且当 ζ₁^(k+1)≠ζ₁^(k) 时严格下降 s_{k+1}<s_k。直觉：阈值的"震荡复杂度"只会越来越简单。
- **Theorem 1（收敛）**：把 [0,1) 等分为 n 个子区间，定义投影 P_n（machine precision 级），对 s_k 做归纳。s_k=0 时每条阈值单调且有界 ⇒ 收敛；s_k=N 时利用 Lemma 3 让 N 逐步下降，最终每条阈值依次收敛到 τ_i*。

### 关键纪律点（避免误读）

- **projection 只是证明用的 slight modification**：数值 Algorithm T-ROF 本身不含 P_n；P_n 只是为了让归纳论证收敛而引入的人工离散化（取 n 足够大近似机器精度）。本仓库 runner 里的 `projection_bins=4096` 即对应这个证明用投影，不是论文算法主体。
- **收敛不是全局/无条件的**：定理依赖 Assumption (A)。对任意图像、任意 K，(A) 不必然成立；因此"阈值序列收敛"是有条件结论，不能外推成"任意输入都保证收敛到最优分割"。
- **K=2 ↔ Chan-Vese（Proposition 2）**：解 (8) 得到的 (Σ*,m₀*,m₁*) 是带参数 **λ=μ/(2(m₁*-m₀*))** 的 Chan-Vese partial minimizer。因 0<m₁*-m₀*≤1，严格的界是 **λ=μ/(2(m₁*-m₀*))≥μ/2**（在 m₁*-m₀*=1 时取等）；论文 p.241 进一步措辞为 "larger than μ"，但这只在 m₁*-m₀*<1/2 时才成立（本仓库 k2_lambda_derived=7.7416<μ=8 即一个 λ 未超过 μ 的例子，对应 m₁*-m₀*≈0.517>0.5）。无论如何，两类均值越接近，m₁*-m₀* 越小，λ 越大、数据项越被加重——这是 T-ROF 能分开相近灰度类别的**理论根因**，也是它相对固定参数方法的优势来源。

## 实验重点

实验对象包括 cartoon、texture 和 medical images；重点看灰度值相近类别是否被正确分开，以及算法速度相对其他 variational segmentation 方法的差异。

### 论文 6 个 Example 与规模（忠于 PDF Section 4）

| Example | 数据 | 规模 | K | 要点 |
| --- | --- | --- | --- | --- |
| 1 | 2-class cartoon（含 missing pixel） | 256×256 | 2 | 需 codebook 更新；只有 [12][20] 和本文给出好结果（Fig. 1） |
| 2 | 2-class close-intensity | 128×128 | 2 | 常图 0.5 + Gaussian noise(var 1e-8)，白部保留/黑部乘 2×10⁻⁴（Fig. 2） |
| 3 | brain MRI gray/white matter | 319×256 | 4 | 来自 [25]；本文 11 次 τ 更新仍较快，约 3× 快于 Pock[25]（Fig. 3） |
| 4 | stripe image（30 stripes） | 140×240 | 5/10/15 | Gaussian noise var 1e-3；对应 **Table 1** |
| 5 | 3-class close-intensity | 256×256 | 3 | Gaussian noise var 1e-2，黑/白标量 0.1/0.6（Fig. 5） |
| 6 | 4-class close-gray-value | 256×256 | 4 | Gaussian noise var 3×10⁻²（Fig. 6） |

### 论文报告的 SA（均直接引自 PDF，注明出处，禁止编造）

指标定义：**SA = #correctly classified pixels / #all pixels**（PDF Section 4），并用 SA 来选各方法的 μ。

| 来源 | 设定 | T-ROF SA | 同图最强对照 |
| --- | --- | --- | --- |
| Fig. 1 | 2-class cartoon (missing) | 0.9913 (Ite. 6) | He 0.9888 / Cai 0.9878 |
| Fig. 2 | 2-class close-intensity | 0.9845 | Cai 0.9816 / He 0.9663 |
| Fig. 5 | 3-class close-intensity | 0.9550 | He 0.9637 / Yuan 0.9557 |
| Fig. 6 | 4-class close-gray-value | 0.9798 | Cai 0.9688 |

**Table 1（Example 4 stripe，μ / Ite. / Time(s) / SA，直接引自 PDF p.247）**：

| Phases | T-ROF SA | Cai[12] SA | 对比 |
| --- | --- | --- | --- |
| Five | 0.9986（μ=8, 84(4), 1.39s） | 0.9770（41, 1.33s） | T-ROF 最高 |
| Ten | 0.9967（84(5), 2.33s） | 0.8900（41, 2.11s） | 差距拉开 |
| Fifteen | 0.9933（84(5), 3.74s） | 0.5280（41, 3.06s） | Cai 崩溃，T-ROF 稳健 |

其中 "84 (4)" = ROF 内迭代 84 步、外层 τ 更新 4 次。**最值得记的趋势**：相位数 5→15 时，前身方法 Cai[12]（single thresholding）的 SA 从 0.977 跌到 0.528，而 T-ROF（iterative thresholding）维持在 0.99 量级。这直接回答了"迭代更新阈值到底有没有用"——在多相、相近灰度场景下，**单次阈值化会塌缩，迭代更新才稳**。

### 速度优势的来源

不是更快的求解器，而是**只解一次 ROF**（Proposition 3）：ROF 极小元与 τ 无关，外层只在同一个 u 上移动阈值线，故 T-ROF 的 ROF 内迭代固定（84 步），τ 更新仅 4-5 次，总时间与 Cai[12] 同量级但 SA 明显更高。

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

### 更具体的脉络

- **承上 — Cai, Chan, Zeng (2013) two-stage [12]**：本篇的直接前身。[12] 先解凸 Mumford-Shah 变体再做**一次**阈值化（single thresholding）。本篇把它升级为**迭代**阈值更新（Eq. 15），并在论文中明确 *outperforms* [12]——Table 1 的 Fifteen-phase 上 Cai[12] 跌到 0.528、本文仍 0.9933 就是直接证据。读这篇时务必把它和 #4（segmentation-restoration / two-stage）对照：两者都是 two-stage 结构，区别在 thresholding 是一次还是迭代。
- **理论侧 — #2 PCMS-ROF Linkage**：#2 给出"为什么阈值化 ROF 解有理论意义"（凸松弛、ROF 与 PCMS 的桥梁），本篇则把该理论落成可运行的多类算法，并补上 K=2 与 Chan-Vese 的精确参数关系 λ=μ/(2(m₁-m₀))（Proposition 2）。
- **总览侧 — #1 SaT Overview**：本篇是 SaT/T-ROF 分支的原型实例；Overview 里 T-ROF 这一支的算法来源就是它。
- **与 SLaT 的区别**：本篇处理灰度/多类阈值（标量场上移动 K-1 条阈值线）；SLaT 处理彩色图，靠 Lab + 特征 lifting 再分割。两者共享"smoothing then thresholding"骨架，但本篇不做颜色空间提升。
- **方法谱系定位**：Mumford-Shah(非凸) → PCMS/Potts(Eq.1-2) → Chan-Vese(K=2, Eq.3) → 本文凸松弛 + ROF 等价(Eq.7,10,11) → 多类迭代阈值(Eq.12-15)。本篇是把"凸化 + 阈值化"思想从 K=2 推到任意 K 的关键一跳。

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

### 阅读陷阱（容易踩的坑）

- **陷阱一：以为每轮都要重解 ROF**。算法只在 step 1 解一次 ROF，外层 k 循环只移动阈值；Proposition 3 保证这样合法。误读成"每轮重解 ROF"会高估复杂度，也错失论文 efficiency 的核心论点。
- **陷阱二：把均值算在平滑后的 u 上**。Eq. (15) 的 m_i=mean_f(Ω_i) 是在 **raw image f** 上取均值（区域 Ω_i 由 u 的水平集划定）。算在 u 上会改变阈值动力学，偏离论文。本仓库 runner 已按 raw f 实现，是正确的。
- **陷阱三：把 projection 当算法步骤**。P_n 投影只出现在 Theorem 1 的证明里（slight modification），数值 Algorithm T-ROF 不含它。把它当成必须的实现步骤是误读。
- **陷阱四：数据项当成平方差**。Eq. (7) 的数据项是线性的 ∫_Σ(τ-f)dx，不是 ∫(u-τ)²；正是线性才能凸松弛为 Eq. (10) 并与 ROF 接上。
- **陷阱五：把收敛当无条件**。Theorem 1 依赖 Assumption (A)；对任意图像/任意 K 不必然成立，"阈值序列收敛"是有条件结论。
- **陷阱六：把 partial 复现的合成图数值当论文级**。本仓库 0.9463 accuracy、Dice 0.9976 来自 96×96 合成图，**不可**与论文 Table 1 / Fig. 的 SA 等同。

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

## 完整复现流程

本篇的"完整复现流程 (Complete Reproduction Workflow)"规范文档已单独成文，覆盖论文身份核验、算法 step-by-step、完整数据集（stripe / cartoon / brain MRI 等 6 个 Example）、五个 baseline、SA 指标与论文报告数值、当前 partial 实现与到 paper-like/paper-level 的差距分析。当前仍为 partial（paper-level 0/15），文档严格区分合成 proxy 结果与论文级结论。

详见：[../reproduce/paper_like/workflows/iterated-rof_reproduction_workflow.md](../reproduce/paper_like/workflows/iterated-rof_reproduction_workflow.md)
