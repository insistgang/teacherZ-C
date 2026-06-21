# Framelet 管状结构分割短版

> 当前 15 篇口径内第 5 篇。本文档按 PDF 首页作者顺序和 dashboard 结构化精读字段重写，避免旧论文笔记混入。

## 论文元信息

| 字段 | 内容 |
| --- | --- |
| 英文标题 | Framelet-Based Algorithm for Segmentation of Tubular Structures |
| 作者顺序 | Xiaohao Cai, Raymond H. Chan, Serena Morigi, Fiorella Sgallari |
| 第一作者核验 | 是，PDF 首页作者列表以 Xiaohao Cai 开头 |
| 年份 | 2011 / 2012 |
| 类型 | SSVM / LNCS |
| PDF | docs/00_papers_first_author_xiaohao_cai_deduped/框架管状结构分割 Framelet.pdf |
| 阅读顺序 | 5 / 15 |
| 主题 | medical |
| 难度 | 中等 |

## 一句话贡献

只平滑管状边界候选区。

## 核心问题

MRA 血管、道路和其他 tube-like structures 有细长、弱边缘、分叉、遮挡等特点；全图平滑会抹掉细节，传统 PDE/active contour 又容易被噪声和初始化影响。

## 为什么难

管状结构的内部、外部和边界灰度不是单点可分；真正难的是灰度落在边界候选区间内的像素，既不能粗暴归类，也不能对整幅图无差别平滑。

## 方法抓手

算法估计边界灰度区间 [α_i, β_i]，每轮把图像分成 below、inside、above 三部分，只对 inside，即可能边界区域，做 framelet denoising / smoothing 和 soft-thresholding，再收缩候选区间直到得到二值图像。

## 关键模型或公式

candidate boundary Λ_i = {x : α_i < f_i(x) < β_i}; framelet denoising on Λ_i; stop when all pixels map to 0 or 1.

### 公式逐项解释（忠于 PDF Section 2–4）

**(1) Framelet 滤波器（Eq. 1）**：采用 piecewise linear B-spline framelet，1D 滤波器为
- h₀ = ¼[1, 2, 1]（低通 / 平滑），
- h₁ = (√2/4)[1, 0, -1]（一阶差分 ≈ 梯度），
- h₂ = ¼[-1, 2, -1]（二阶差分 ≈ Laplacian）。

每个 h_i 生成一个 Toeplitz 滤波矩阵 H_i（如 H₀ = ¼ tridiag[1,2,1]），1D framelet 前向变换 A = [H₀; H₁; H₂]（Eq. 2）。对图像做 framelet 分析即算 Af，把图像表示成低通 + 两个差分通道的 redundant 系数。

**(2) 张量积扩展到 d 维**：2D 有 9 个 framelet（h_ij = h_iᵀ ⊗ h_j），3D 有 27 个；A 是若干 block-Toeplitz-Toeplitz-block 矩阵的堆叠。**关键性质：AᵀA = I（perfect reconstruction）**，但一般 AAᵀ ≠ I —— 这正是 tight frame 区别于正交小波之处，redundancy（冗余）带来更强的去噪与方向表示能力。

**(3) Generic framelet 迭代（Eq. 3–5）**：
- f^(i+½) = U(f^(i))：先用问题相关算子 U 把图像做一次处理（这里 U 就是下文 Eq. 6 的三分阈值-拉伸）。
- f^(i+1) = Aᵀ T_λ(A f^(i+½))：把图像变到 framelet 域，soft-threshold，再变回——一次"分析-收缩-合成"去噪。
- soft-thresholding（Eq. 5）：t_{λk}(vk) = sgn(vk)(|vk| - λk) 若 |vk| > λk，否则 0。这是 Donoho 软阈值 [15]，对小系数（噪声）置零、对大系数（结构）收缩。

**(4) 三分阈值-拉伸 U（Eq. 6–7）**：给定区间 [αᵢ, βᵢ]，把像素分三类：
- f_j ≤ αᵢ → 0（below，背景，绿色）；
- βᵢ ≤ f_j → 1（above，血管，黄色）；
- αᵢ ≤ f_j ≤ βᵢ → 线性拉伸 (f_j - mᵢ)/(Mᵢ - mᵢ)（inside，候选边界，红色），其中 mᵢ/Mᵢ 是区间内的最小/最大灰度。
候选集合 Λ^(i) = {j | mᵢ < f_j^(i) < Mᵢ}（Eq. 7），即拉伸后仍非 0 非 1 的像素。**核心思想：only the "inside" pixels are uncertain，only they deserve framelet smoothing。**

**(5) 受限 framelet 更新（Eq. 8）**：令 P^(i) 为对角投影矩阵（Λ^(i) 内为 1）。则
- f^(i+1) = (I - P^(i)) f^(i+½) + P^(i) Aᵀ T_λ(A f^(i+½))。
含义：Λ^(i) 外像素（已是 0/1）原样保留，**只对 Λ^(i) 内像素做一次 framelet 去噪**。因为 Λ 外都是 0/1，Af^(i+½) 可用 sparse 结构加速，整体 O(n) per iteration。

**(6) 区间初始化与细化（Eq. 9–12）**：
- 初始化用梯度 g_j = [Σ_ℓ (∂_{xℓ} f^(0))²]^{1/2}，取 g_j > ε 的像素集 Γ（ε ∈ [10⁻³,10⁻¹]），按均值 μ_Γ 把 Γ 分成靠近血管的 Γ₊ 与靠近背景的 Γ₋，得 μ₊/μ₋，进而 Λ^(-1)、μ^(0)（Eq. 9）。
- 每轮更新 μ^(i+1)（Eq. 10），再用对比函数 c(α)（上侧均值减下侧均值）的值域长度 ℓ 和参数 γ ∈ (0,1/2) 反解 αᵢ（Eq. 11）、βᵢ（Eq. 12）。论文取 **γ = 1/5**：γ 越小区间越窄、收敛越快。

> 阅读要点：把"区间 [αᵢ,βᵢ] 持续收缩"和"候选集合 Λ^(i) 严格变小"分开理解——前者是参数动力学（Eq. 11–12），后者是它带来的结果（Eq. 7），二者共同保证 Theorem 1 的有限步收敛。

## 算法流程

1. 估计当前边界灰度区间 [α_i, β_i]。
2. 把像素分成背景、边界候选和血管三类。
3. 只在候选区域 Λ_i 做 framelet soft-thresholding。
4. 更新图像并收缩候选区域。
5. 候选区域为空时输出二值管状结构。

### 论文 Algorithm 原文骨架（Section 4）

> 1. Initialize: f^(0) = f, μ^(0) by (9), [α₀, β₀] by (11)(12).
> 2. Do i = 0,1,…, until stopped:
>    - (a) Compute f^(i+½) = U(f^(i)) by (6)  —— 三分阈值-拉伸。
>    - (b) Stop if f^(i+½) is a binary image  —— 即 |Λ^(i)| = 0。
>    - (c) Update f^(i+½) to f^(i+1) by (8)  —— 仅在 Λ^(i) 上 framelet 去噪。
>    - (d) Update μ^(i+1) by (10), then [α_{i+1}, β_{i+1}] by (11)(12)  —— 收缩区间。
> 3. Extract the boundary from the binary image f^(i+½)（2D 用 `contour`，3D 用 `isosurface`）。

### isotropic vs anisotropic thresholding

- **isotropic**（各向同性）：对全部 framelet 系数 Af^(i+½) 统一软阈值，λk = 2^(-1/2)。
- **anisotropic**（各向异性，引 [7][26]）：h₁ 对应中心差分 ≈ 梯度方向，把这些系数旋转到切向/法向，**只在切向阈值化**（λk = 0.1×2^(-1/2)），其余系数仍按 isotropic（λk = 2^(-1/2)）。论文实验发现 anisotropic 能更紧地贴合边界、在 tips 处恢复更多像素、更好地连接沿 coherence direction 的小遮挡。这是 isotropic/anisotropic 的核心差异，也是论文一大卖点。

> 阅读陷阱：这套流程**不是变分极小化**。它没有能量泛函要 minimize，而是"有限步分类 + 局部去噪"的组合——每个像素最终被钉到 0 或 1，候选集合单调收缩。不要套用 ROF / Mumford-Shah / Chan-Vese 的"求极小"框架去理解收敛性。

## 理论保证

论文给出 convergence statement：framelet-based algorithm 会在有限步收敛到二值图像；关键是候选边界区域 Λ_i 持续收缩。

### Theorem 1 与证明直觉

**Theorem 1**：Our framelet-based algorithm will converge to a binary image.

证明思路（PDF Section 4）只需证某有限步 i 有 |Λ^(i)| = 0：
1. 若初始 f^(0) 已是二值图像，直接结束。
2. 否则，对 i ≥ 1，由 Eq. 8 知 Λ^(i-1) 外的像素已是 0/1，其值不被改变（保持稳定）。
3. 由 Eq. 11–12 得 [αᵢ, βᵢ]，且 [αᵢ, βᵢ] ⊆ [α_i^L, β_i^H] ⊊ [0,1]（只要图像非二值，此区间严格小于 [0,1]）。
4. 由 mᵢ ≥ αᵢ、Mᵢ ≤ βᵢ（即 [mᵢ, Mᵢ] ⊆ [αᵢ, βᵢ]），满足 f^(i) ≤ mᵢ 或 f^(i) ≥ Mᵢ 的像素会被 Eq. 6 钉到 0 或 1。
5. 因此每轮至少有一个像素离开候选集合：**|Λ^(i)| < |Λ^(i-1)|（严格递减）**。
6. 由于 |Λ^(0)| 有限，必存在某 i 使 |Λ^(i)| = 0，算法在有限步停止。∎

**直觉**：这是一个"单调收缩的分类过程"。每次迭代区间 [αᵢ,βᵢ] 收紧 → 更多像素被判定为确定的背景(0)或血管(1) → 候选集合严格变小 → 有限步内清空。Table 1 的 |Λ^(i)| 列正是这一严格递减的经验证据（如 Example 1：2374 → 307 → 83 → 23 → 7 → 1 → 0）。

**与变分方法的区别**：传统 ROF/Chan-Vese 靠最小化能量泛函收敛（不动点/梯度流），收敛速度依赖步长与条件数；本方法靠"候选集合基数严格递减 + 集合有限"这一组合论证，**收敛是有限步且可数（论文实验 6–9 iters）**，性质更接近"逐步消去不确定像素"。复杂度 O(n) per iteration。

## 实验重点

实验为 real 2D/3D images，并在文本中明确指向 Magnetic Resonance Angiography (MRA) 血管场景；重点看细血管、分叉和弱边界是否保留。

### 三个实验的具体设定（PDF Section 5，Table 1）

| 实验 | 数据 | 尺寸 | 参数 | 收敛迭代数 | 对照 |
| --- | --- | --- | --- | --- | --- |
| Example 1（Fig. 2） | 2D MRA carotid 颈动脉 | 182 × 62（\|Ω\|=11,284） | γ=1/5, ε=1.6×10⁻² | 6（iso 与 aniso 均 6） | [10] Chan-Vese, [16] PDE |
| Example 2（Fig. 3） | 2D MRA kidney 肾血管（即 Fig.1(a)） | 257 × 257（\|Ω\|=66,049） | γ=1/5, ε=5×10⁻³ | 7（iso 与 aniso 均 7） | [10], [16] |
| Example 3（Fig. 4） | 3D CTA kidney 肾脏体数据 | 201×201×201（从 436×436×540 提取，\|Ω\|=8,120,601） | γ=1/5, ε=6×10⁻² | iso=9, aniso=8 | [17] PDE |

**Table 1（候选集合基数 \|Λ^(i)\| per iteration）可确认数值**：
- Fig. 2(d) iso：2374, 307, 83, 23, 7, 1, 0
- Fig. 2(e) aniso：2374, 233, 48, 13, 5, 1, 0
- Fig. 3(c) iso：8314, 1834, 565, 137, 29, 18, 4, 0
- Fig. 3(d) aniso：8314, 1557, 406, 95, 19, 5, 1, 0
- Fig. 4(d) iso：104329, 21333, 5460, 1430, 326, 70, 9, 3, 1, 0
- Fig. 4(e) aniso：104329, 20020, 4984, 1260, 299, 72, 19, 6, 0

可见 3 次迭代后只剩极少像素需分类（如 Example 1 仅剩 23/11284）。

**阈值参数**：只用 piecewise linear filters 的第一层（no downsampling）；isotropic λk = 2^(-1/2)；anisotropic 切向 λk = 0.1×2^(-1/2)、其余 = 2^(-1/2)。

### 定性结论（论文文字，不是数字指标）

- 比 [10] Chan-Vese：本方法不断裂（Chan-Vese 在小遮挡处断开），结构更连贯。
- 比 [16][17] PDE anisotropic diffusion：本方法提取更多细节、去噪更干净（Fig. 3(e)–(g) vs (i)–(k) 放大对比）。
- anisotropic vs isotropic：anisotropic 边界更紧、tips 处恢复更多像素、更好连接 coherence direction 上的小遮挡（Fig. 3(h)(l) superimposed boundaries 显示 anisotropic 更贴 tips）。

> **重要诚实说明**：论文 **未报告 Dice / IoU / sensitivity 等重叠指标**，其量化证据集中在 Table 1 的 \|Λ^(i)\| 收缩与迭代数。本仓库 partial 用的 Dice 0.9863 是合成数据上的自定义内部评估，**不可与论文结果对等**。阅读时不要把论文说成"达到某 Dice"。

## 精读方式

先读 Section 2 的 tight frame / framelet 基础；重点读 Section 3 算法步骤；再读 Theorem 1 的 finite convergence 证明；最后看 2D/3D 图像实验。

## 论文证据点

- Abstract
- Section 3 algorithm
- boundary interval [α_i, β_i]
- finite convergence theorem
- 2D/3D tubular experiments

## 与其他 14 篇的关系

这是 vessel tight-frame 长版的短版基础，也与 SaT 共享“平滑不确定区域 + 阈值化”的思想。

关联论文：#6 Tight-frame 医学血管分割长版; #1 SaT 分割方法论总览; #8 球面小波图像分割

### 更具体的论述

- **与 #6 Tight-frame 医学血管分割长版**：本篇（SSVM 2011 短版）是长版的 conference 前身/基础。两者共享同一核心机制（候选边界区间 [αᵢ,βᵢ] + 仅在 Λ 上 framelet 去噪 + 有限步收缩）。本仓库 runner `tubular_tight_frame.py` **同时产出第 5、6 篇两条记录**，正是因为它们方法同源——这点在复现工程上要特别留意：两篇共用一个 toy 实验、一张图。

- **与 #1 SaT（Smoothing-and-Thresholding）方法论**：共享"先稳定不确定部分，再阈值化得结构"的两段式骨架。但本篇用 **framelet 表示 + 候选灰度区间**，而 SaT 系列（如 SaT/SLaT）以 **凸松弛 ROF / Mumford-Shah / PCMS** 为理论核心。换言之：相同的"smooth-then-threshold"哲学，不同的数学工具（framelet 局部去噪 vs 变分凸优化）。

- **与 #8 球面小波/小波类分割**：同属 framelet/wavelet 表示在分割中的应用谱系，体现作者用 redundant tight frame 做图像分析的一贯路线。

- **作者方法谱系**：参考文献里大量自引（[3][4][5] Cai-Chan-Shen 的 framelet inpainting / split Bregman frame restoration），说明本篇是把作者在 **inpainting / restoration** 上成熟的 framelet 工具迁移到 **segmentation** 的一次延伸。理解本篇有助于把握作者"framelet 工具箱 → 多任务"的研究主线。

## 报告扩展字段

- context: 这篇是管状结构分割线的短版入口，适合先读来理解思想。目标对象不是普通区域分割，而是 MRA 血管、道路等细长结构，它们的边界弱、分叉多、噪声下容易断裂。
- technicalReading: 技术抓手是 possible boundary gray interval [α_i, β_i]。算法每轮把像素分成 below、inside、above，只对 inside 的候选边界区域做 framelet denoising / soft-thresholding，而不是对整幅图做统一平滑。
- theoremReading: 理论阅读重点是 finite convergence：候选边界集合在迭代中持续收缩，已经确定为 0 或 1 的像素离开候选区。要理解这个保证与传统 variational minimization 不同，它更像有限步分类和局部平滑的组合。
- experimentReading: 实验要看 2D/3D tubular structures，尤其是细血管、弱边界、分叉处是否保留。不要只看最终二值图，还要看候选区域收缩是否可能导致漏检或断裂。
- relationReading: 它是 Tight-frame Vessel 长版的基础，也与 SaT 有相似结构：先稳定不确定部分，再阈值化得到结构。但它更强调候选边界区间和 framelet 表示，而不是 ROF/PCMS 理论。
- researchValue: 这篇的价值是给出一种局部处理策略：复杂图像中不是所有像素都同等困难，真正值得用 framelet 平滑的是边界候选集合。这种思想可迁移到医学点云、血管中心线和遥感线状目标。

## 阅读问题

1. 为什么只对 inside 候选区域做 framelet 平滑？
2. 候选区间 [α_i, β_i] 如何影响收敛速度和漏检？
3. 这个算法为什么不是标准 variational minimization？

### 参考解答（结合 PDF）

1. **为什么只对 inside 做 framelet 平滑**：below(0)/above(1) 的像素已被高/低对比确定为背景/血管，对它们再平滑只会模糊已确定结构、抹掉细血管；真正不确定的只有灰度落在 [αᵢ,βᵢ] 的边界候选像素。把去噪算力集中在 Λ^(i)（Eq. 8 的 P^(i) 投影）既保细节又省算力（Λ 外全 0/1，可 sparse 加速，O(n)）。

2. **区间如何影响收敛与漏检**：区间长度由 γ 控制（Eq. 11–12，γ∈(0,1/2)，论文取 1/5）。**γ 越小 → 区间越窄 → 每轮判定为确定的像素越多 → Λ 收缩越快、迭代数越少**；但区间过窄可能把弱边界的真血管过早钉成背景，造成**漏检/断裂**。γ 越大则更谨慎、保留更多候选，收敛慢但漏检风险低。这是 speed–recall 的权衡。

3. **为什么不是 variational minimization**：它没有定义任何能量泛函去 minimize（论文引言明确区别于 [13] 的"minimizing variational model"）。收敛来自"候选集合 \|Λ^(i)\| 严格递减且有限"的组合论证（Theorem 1），而非泛函极小的不动点。它更像"有限步分类 + 局部去噪"的迭代消去过程。

## 读后产出

画出 below / inside / above 三分图像和候选区域收缩过程。

## 复现判断

| 字段 | 内容 |
| --- | --- |
| 复现等级 | partial |
| 真实性等级 | partial-completed |
| 难度 | 高 |
| 效果 | 很明显 |
| 最小实验 | synthetic tube/vessel mask with noise；梯度初始化 Λ^(0) (Eq.6)，每轮在 Λ 上用真实 tight-frame（pywt 无下采样 SWT，本身即 tight frame）软阈值去噪 (Eq.14)，自适应区间 μ/μ± (Eq.7–10) + 三段阈值-拉伸 (Eq.11–13) 收缩 Λ。 |
| 预期产出 | uncertainty region shrinks and binary tube mask emerges；partial Dice 0.9863（优于 raw 0.5 阈值基线 0.9823），Lambda 从 12416 单调收缩到 **0**（5 轮收敛）。升级后 Λ 真正达到 |Λ|=0，最终二值掩膜由收敛准则触发而非硬阈值强制，论文 Theorem 1 的"有限步 |Λ|=0 收敛"现已在真实 tight-frame 算子下被演示。 |
| 依赖 | numpy / scipy / scikit-image / matplotlib / pywavelets |
| 数据需求 | synthetic 2D tube network；full reproduction 需要论文使用的 2D/3D tubular structures（私有 MRA/CTA 或 DRIVE/STARE/TubeTK/VascuSynth 公开等价）。 |
| 算力需求 | CPU，约 0.2 秒。 |
| 实现风险 | tight-frame 用 Haar 无下采样小波（真实 tight frame）而非论文精确的分段线性 B-spline framelet / anisotropic DℂWT；仍为合成 2D 数据（无 3D、无 anisotropic、无论文基线）；Dice/IoU 是论文未报告的 toy 内部量。 |

### 复现指标

- dice
- iou
- raw_dice
- raw_iou
- lambda_initial
- lambda_final
- iterations
- converged_empty_lambda

### 验证计划

记录 Lambda size per iteration（应单调非增并收敛到 0）、toy Dice/IoU 与 raw 基线对照（真实算法应优于裸阈值），并检查候选区域是否随迭代收缩。

### 当前运行结果

- dice: 0.9863
- iou: 0.9729
- raw_dice: 0.9823
- raw_iou: 0.9653
- lambda_initial: 12416
- lambda_final: 0
- iterations: 5
- converged_empty_lambda: 1

（Λ 收缩序列 `[12416, 206, 42, 8, 0]`，单调非增并收敛到 0。）

### 结果说明

Partial reproduction with a REAL tight-frame denoiser: the paper's A^T T_lambda(A f) framelet step is implemented via an undecimated stationary wavelet transform (pywt.swt2/iswt2, Haar, level 2, lambda=0.08) — itself a tight frame with perfect reconstruction — applied as soft thresholding only on the candidate boundary set Lambda. Adaptive interval [alpha_i,beta_i] from mu/mu_-/mu_+ (Eq.7–10), three-segment threshold + linear contrast stretch (Eq.11–12), gradient init (Eq.6); |Lambda| now converges to 0. dice/iou are TOY internal overlap on synthetic 2D vessels (it beats the raw-threshold baseline but) the paper reports neither Dice nor real 2D/3D MRA/CTA performance.

## 完整复现流程

本篇的"完整复现流程 (Complete Reproduction Workflow)"规范文档已单独成文，覆盖论文身份核验、算法 step-by-step pipeline（Eq. 1–12）、所需数据集（MRA/CTA 原始 + DRIVE/STARE/TubeTK/VascuSynth 公开等价）、对照基线（[10] Chan-Vese、[16][17] PDE）、评价指标与论文报告结果（Table 1 的 \|Λ^(i)\| 收缩、6–9 iters，论文未报 Dice）、本仓库 toy 实现与 proxy 局限、以及到 paper-like / paper-level 的差距清单。

详见：[`../reproduce/paper_like/workflows/framelet-tubular_reproduction_workflow.md`](../reproduce/paper_like/workflows/framelet-tubular_reproduction_workflow.md)
