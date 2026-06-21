# Framelet 管状结构分割短版 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 5 篇 *Framelet-Based Algorithm for Segmentation of Tubular Structures* 的完整复现流程规范。

## 1. 论文身份与第一作者核验

| 字段 | 内容 |
| --- | --- |
| 英文标题 | Framelet-Based Algorithm for Segmentation of Tubular Structures |
| 中文短标 | Framelet 管状结构分割短版 |
| 作者顺序 | **Xiaohao Cai**, Raymond H. Chan, Serena Morigi, Fiorella Sgallari |
| 第一作者核验 | 是。PDF 首页作者列表以 Xiaohao Cai 开头。第 1、2 作者（Cai、Chan）属 CUHK Department of Mathematics，第 3、4 作者（Morigi、Sgallari）属 University of Bologna Department of Mathematics-CIRAM。 |
| 会议 / 出版 | SSVM 2011（Scale Space and Variational Methods），LNCS 6667，pp. 411–422，© Springer-Verlag 2012 |
| 年份 | 2011（会议）/ 2012（LNCS 出版） |
| PDF 路径 | `docs/00_papers_first_author_xiaohao_cai_deduped/框架管状结构分割 Framelet.pdf` |
| 主题 | medical（MRA / CTA 血管分割） |
| 阅读顺序 | 5 / 15 |

核验依据：PDF p.411 标题页作者署名 "Xiaohao Cai¹, Raymond H. Chan¹, Serena Morigi², and Fiorella Sgallari²"，与 dashboard `authors` 字段一致。

## 2. 复现目标与诚实分级

本项目对"复现"分四级，纪律如下：

- **toy**：合成数据 + proxy 算子，只验证算法骨架与定性行为（如候选集合是否收缩）。
- **partial**：真实数据子集 + 部分忠实算子，复现部分定量趋势但不对齐论文数表。
- **paper-like**：忠实算子 + 公开/等价数据，能在可比设定下逼近论文报告的量级与趋势。
- **paper-level**：论文同源数据 + 完整方法 + 完整对照与数表对齐。

| 维度 | 当前状态 |
| --- | --- |
| 仓库复现等级 `reproductionLevel` | **toy** |
| 真实性等级 `reproductionTruthLevel` | **toy-completed** |
| paper-level 全局状态 | **0 / 15**（本项目尚无任何一篇达到 paper-level） |

**纪律声明**：本仓库当前对本篇的实现使用 Gaussian smoothing 代替严格的 framelet（tight frame）变换，数据是合成 2D 管网，因此 toy Dice 0.9981 **只能解读为骨架自检**，**禁止**将其表述为论文级 MRA / CTA 分割性能或与论文 Table 1 / Fig. 2–4 做对等比较。

## 3. 算法完整流程

论文方法**不是**变分极小化（作者在引言中明确区别于 [13] 等基于最小化变分能量的方法），而是"**迭代收缩可能边界灰度区间 + 仅在候选集合上做 framelet 去噪/平滑 + 三分阈值化**"的有限步分类过程。逐步拆解如下（所有公式忠于 PDF Section 2–4）。

**符号约定**：图像 dynamic range 归一到 [0,1]；Ω 为全部像素索引集；f^(i) 为第 i 轮近似图像；[αᵢ, βᵢ] ⊆ [0,1] 为"可能边界像素灰度区间"。

### Step 0 — Framelet（tight frame）基础（Section 2）

1. 采用 piecewise linear B-spline framelet，对应 1D 滤波器（Eq. 1）：
   - h₀ = ¼[1, 2, 1]，h₁ = (√2/4)[1, 0, -1]，h₂ = ¼[-1, 2, -1]。
2. 每个滤波器构成 Toeplitz 滤波矩阵 H_i（如 H₀ = ¼ tridiag[1,2,1]）；1D framelet 前向变换 A = [H₀; H₁; H₂]（Eq. 2）。对向量 v 的 framelet 系数即 Av，H_i v 给出滤波器 h_i 的系数。
3. d 维 framelet 由张量积构造（Section 2，引 [14]）：2D 有 9 个 framelet（h_ij ≡ h_iᵀ ⊗ h_j），3D 有 27 个。2D 时 A 为 9 个 block-Toeplitz-Toeplitz-block 矩阵的堆叠，A·vec(f) 给出全部 framelet 系数。
4. **Perfect reconstruction property**：AᵀA = I（恒等）。注意一般 AAᵀ ≠ I（与正交小波不同，这是 tight frame 的 redundancy）。

### Step 1 — 通用 framelet 迭代算子（Eq. 3–5）

generic framelet 算法形式：
- f^(i+½) = U(f^(i))（Eq. 3，U 为问题相关算子，这里即 Step 2 的三分阈值-拉伸）。
- f^(i+1) = AᵀT_λ(Af^(i+½))，i = 1, 2, …（Eq. 4）。
- soft-thresholding（Eq. 5）：t_{λk}(vk) = sgn(vk)(|vk| - λk) 当 |vk| > λk，否则 0。λk 选取见 [15]（Donoho soft-thresholding）。

### Step 2 — 三分阈值化与对比拉伸（Step (i)，Eq. 6–7）

给定当前区间 [αᵢ, βᵢ]，把 f^(i) 分成 below / inside / above 三部分（Fig. 1(b)：绿=below，红=inside，黄=above）：
1. 令 Mᵢ = max{f_j^(i) | αᵢ ≤ f_j^(i) ≤ βᵢ}，mᵢ = min{f_j^(i) | αᵢ ≤ f_j^(i) ≤ βᵢ}。
2. 定义 f^(i+½)（Eq. 6）：
   - f_j^(i) ≤ αᵢ 时设为 0；
   - αᵢ ≤ f_j^(i) ≤ βᵢ 时线性拉伸为 (f_j^(i) - mᵢ)/(Mᵢ - mᵢ)（simple linear contrast stretch，引 [19] Gonzalez-Woods）；
   - βᵢ ≤ f_j^(i) 时设为 1。
   记 f^(i+½) = U(f^(i))。
3. 候选边界集合（Eq. 7）：Λ^(i) = {j | mᵢ < f_j^(i) < Mᵢ, j ∈ Ω}，即 f^(i+½) 中值既非 0 也非 1 的像素索引集。

### Step 3 — 仅在 Λ^(i) 上做 framelet 迭代（Step (ii)，Eq. 8）

令 P^(i) 为对角矩阵：若索引在 Λ^(i) 则对角元为 1，否则为 0。f^(i+½) 向量化为 f**（粗体），则：

   f^(i+1) = (I - P^(i))f^(i+½) + P^(i) Aᵀ T_λ(A f^(i+½))    （Eq. 8）

即：候选集合 Λ^(i) 外的像素保持其 0/1 取值不变；只对 Λ^(i) 内像素做一次 framelet soft-thresholding 去噪/平滑。论文指出 Λ^(i) 外像素已是 0 或 1，可用 sparse 结构显著降低 Af^(i+½) 的计算成本。

### Step 4 — 区间初始化与细化（Section 4，Eq. 9–12）

**初始化（i = 0）**：f^(0) 取给定图像。用梯度找初始边界：
- g_j = [Σ_{ℓ=1}^d (∂_{xℓ} f_j^(0))²]^{1/2}（forward difference，d=2 或 3）。
- Γ = {j | g_j > ε}（ε ∈ [10⁻³, 10⁻¹]）；μ_Γ = (1/|Γ|) Σ_{j∈Γ} f_j^(0)。
- 按 μ_Γ 把 Γ 分成靠近管状的 Γ₊ = {j | f_j > μ_Γ} 与靠近背景的 Γ₋；μ₊、μ₋ 为对应均值。
- Λ^(-1) = {j | μ₋ < f_j^(0) < μ₊}；μ^(0) = (1/|Λ^(-1)|) Σ_{j∈Λ^(-1)} f_j^(0)（Eq. 9）。

**迭代均值（i ≥ 0）**：μ^(i+1) = (1/|Λ^(i)|) Σ_{j∈Λ^(i)} f_j^(i+1)（Eq. 10）。

**区间选取（Eq. 11–12）**：先算包含 μ^(i) 的粗区间 [α_i^L, β_i^H]（按 ≤ μ^(i) 与 ≥ μ^(i) 的像素均值）；再对 α ∈ [α_i^L, β_i^H] 定义对比函数 c(α)（上侧均值减下侧均值），令其值域 [c_m, c_M]，ℓ = c_M - c_m：
- αᵢ = min{α | c(α) = c(μ^(i)) - γℓ}（Eq. 11）；
- βᵢ = max{α | c(α) = c(μ^(i)) + γℓ}（Eq. 12）。
- γ ∈ (0, 1/2) 控制区间长度；论文取 **γ = 1/5**。γ 越小区间越窄，收敛越快。

### Step 5 — 停止准则与完整算法（Section 4，Algorithm）

> Algorithm: Framelet-based algorithm for segmentation
> 1. Initialize: f^(0) = f, μ^(0) by (9), [α₀, β₀] by (11)(12).
> 2. Do i = 0,1,…, until stopped:
>    (a) Compute f^(i+½) = U(f^(i)) by (6).
>    (b) Stop if f^(i+½) is a binary image.
>    (c) Update f^(i+½) to f^(i+1) by (8).
>    (d) Update μ^(i+1) by (10), then [α_{i+1}, β_{i+1}] by (11)(12).
> 3. Extract the boundary from the binary image f^(i+½).

停止条件：所有像素取 0 或 1，等价于 |Λ^(i)| = 0。最终用 MATLAB `contour`（2D）/ `isosurface`（3D）提取边界。

### isotropic vs anisotropic thresholding

- isotropic：对全部 framelet 系数 Af^(i+½) 统一软阈值。
- anisotropic（引 [7]）：h₁ 对应中心差分 ≈ 梯度方向，应把这些系数旋转到切向/法向，只在切向阈值化，其余系数仍按 isotropic 处理。论文实验发现 anisotropic 能更紧地贴合边界、连接小遮挡。

## 4. 完整复现所需数据集

论文实验全部为真实医学影像（Section 5）：

| 实验 | 数据 | 尺寸 | 来源 |
| --- | --- | --- | --- |
| Example 1 | 2D MRA carotid vascular system（颈动脉） | 182 × 62 | 引 [16]（与 Fig.2 对照同源） |
| Example 2 | 2D MRA kidney vascular system（肾血管，即 Fig.1(a)） | 257 × 257 | 引 [16]（与 Fig.3(b) 对照同源；PDF 仅整体声明数据来自 [16][17]，未逐例标注） |
| Example 3 | 3D CTA kidney vasculature（肾脏 CTA 体数据） | 201 × 201 × 201（从 436 × 436 × 540 CTA 提取） | 引 [17]（与 Fig.4(b) 对照同源） |

**达到 paper-like 的数据候选**（论文原数据多为私有医学影像，需等价公开替代）：
- 2D 视网膜血管：**DRIVE / STARE / CHASE_DB1**（有血管 ground-truth，适合 Dice/IoU 评估，是 MRA carotid/kidney 的公开等价物）。
- 3D 脑血管 MRA：**TubeTK / Bullitt MRA**、**IXI**（含 MRA 序列）、**MIDAS** 公开 MRA 数据。
- 3D CTA：公开 CTA 血管数据较稀缺，可用 **VascuSynth** 合成血管树作为 controlled 3D 替代。
- 道路 / 线状目标（论文引言提到 aerial photography 道路提取）：**Massachusetts Roads** 等遥感线状数据可作 cross-domain 验证。

**注意**：PDF Section 5 仅整体声明 "All the data are obtained from [16] and [17]"，并未逐例标注每个 Example 的数据归属；上表"来源"列是结合各例对照方法（Example 1/2 对照 [16]、Example 3 对照 [17]）的同源推断，不应被读作论文逐例的明确事实。论文原始 MRA carotid（Example 1）、MRA kidney（Example 2）、3D CTA（Example 3）均为私有/合作机构数据，无法直接获取，paper-level 复现需向原作者或 [16][17] 团队申请，否则只能用上述公开等价数据达到 paper-like。

## 5. 对照基线 (Baselines)

论文显式对照的方法（Fig. 2–4）：

| 基线 | 出处 | 说明 |
| --- | --- | --- |
| Chan-Vese active contours without edges | [10] Chan & Vese 2001 | Example 1（Fig. 2(b)）；论文指出其结果断裂、不令人满意 |
| PDE-based anisotropic diffusion model（Franchini 等） | [16] Franchini, Morigi, Sgallari 2010 | Example 1/2（Fig. 2(c), 3(b)）；论文同源数据来源 |
| Composed segmentation by anisotropic PDE | [17] Franchini, Morigi, Sgallari 2009 | Example 3（Fig. 4(b)）3D CTA 对照 |
| 本方法 isotropic vs anisotropic | 本文 | 自身两种阈值方案对照（Fig. 2(d)(e), 3(c)(d), 4(d)(e)） |

合理的现代附加基线（用于公开数据 paper-like 验证）：Frangi vesselness filter、region growing、U-Net / 监督血管分割（仅作上界参照，非论文同类）。

## 6. 评价指标与论文报告结果

**论文的定量呈现以收敛性为主，而非 Dice 类重叠指标**（这是关键事实，不能凭空给论文安上 Dice 数）：

- **Table 1**：报告三个例子在每次迭代的候选集合基数 |Λ^(i)|（i=0,…,9）。可从 PDF Table 1 确认的数值：
  - Fig. 2(d)（Example 1，isotropic）：|Ω| = 182×62 = 11,284；|Λ^(0)| = 2374, |Λ^(1)| = 307, |Λ^(2)| = 83, |Λ^(3)| = 23, |Λ^(4)| = 7, |Λ^(5)| = 1, |Λ^(6)| = 0 → **6 iterations**。
  - Fig. 2(e)（Example 1，anisotropic）：|Λ^(0)| = 2374, 233, 48, 13, 5, 1, 0。
  - Fig. 3(c)（Example 2，isotropic）：|Ω| = 66,049（=257×257）；|Λ^(0)| = 8314, 1834, 565, 137, 29, 18, 4, 0。
  - Fig. 3(d)（Example 2，anisotropic）：8314, 1557, 406, 95, 19, 5, 1, 0。
  - Fig. 4(d)（Example 3，isotropic）：|Ω| = 8,120,601（=201³）；|Λ^(0)| = 104329, 21333, 5460, 1430, 326, 70, 9, 3, 1, 0 → **9 iterations**。
  - Fig. 4(e)（Example 3，anisotropic）：104329, 20020, 4984, 1260, 299, 72, 19, 6, 0 → **8 iterations**。
- **收敛迭代数**：Example 1 = 6（两方案），Example 2 = 7，Example 3 = 9（iso）/ 8（aniso）。
- **参数**：γ = 1/5（全部例子）；ε：Example 1 = 1.6×10⁻²，Example 2 = 5×10⁻³，Example 3 = 6×10⁻²。阈值 λk：isotropic = 2^(-1/2)；anisotropic 切向 = 0.1×2^(-1/2)，其余 = 2^(-1/2)。只用 piecewise linear filters 的第一层（no downsampling）。
- **复杂度**：每轮 O(n)（n 为像素数），μ^(i) 与 [αᵢ,βᵢ] 计算 O(n)，步骤 (a)(c) 也 O(n)。

**定性结论（论文文字，非数字）**：本方法比 [10] 更连贯（不断裂）、比 [16] 能提取更多细节并更好去噪；anisotropic 比 isotropic 边界更紧、tips 处恢复更多像素、更好连接小遮挡（沿 coherence direction）。

> 诚实说明：论文 **未报告** Dice / IoU / sensitivity 等标准重叠指标；其量化证据集中在 |Λ^(i)| 收缩表与迭代数。复现时若要给 Dice，需自备 ground-truth 公开数据并自定义评估，**不能声称等同论文结果**。

## 7. 本仓库当前复现实现

- **runner**：`reproduce/experiments/tubular_tight_frame.py`（同一 runner 同时产出 priority 5 framelet-tubular 与 priority 6 tight-frame-vessel 两条记录）。
- **实际做了什么**：
  1. 用 `skimage.draw.line` + `skimage.morphology.dilation(disk(3))` 构造一个 112×112 合成管网 mask；加 Gaussian 噪声（σ=0.13）得到 noisy 图像。
  2. 设固定区间 alpha=0.38, beta=0.62，循环 12 次：取 uncertain = (alpha < current < beta) 作为候选集合，记录 |Λ| 大小；
  3. 对整图做 `gaussian_filter(sigma=1.0)`，**仅把候选集合内像素**替换为平滑值（对应 Eq. 8 的"仅在 Λ 上更新"思想）；
  4. 阈值 0.5 判前景，把确定像素钉到 0/1；每轮 alpha += 0.008、beta -= 0.008（对应区间收缩）；
  5. 输出二值图，算 Dice / IoU，画 `tubular_lambda_shrinkage.png`（noisy / truth / output / Λ size 曲线）。
- **使用的 proxy**：Gaussian smoothing **代替**严格 framelet（tight frame）soft-thresholding；固定线性收缩区间 **代替** Eq. 9–12 的梯度初始化 + c(α) 自适应区间；合成管网 **代替** 真实 MRA/CTA。
- **当前 runMetrics**（来自 `reproStructured`）：

  | 指标 | 值 |
  | --- | --- |
  | dice | 0.9981 |
  | iou | 0.9962 |
  | lambda_initial | 651 |
  | lambda_final | 2 |
  | iterations | 12 |
  | runtimeSeconds | 0.076 |

- **结果图**：`docs/assets/repro/tubular_lambda_shrinkage.png`（dashboard `resultFiles`: `assets/repro/tubular_lambda_shrinkage.png`）。
- **结果说明（保持诚实表述）**：Approximate toy reproduction；Gaussian smoothing stands in for framelet smoothing inside uncertain boundary interval；Dice 仅在合成 2D 玩具上测得，不代表真实 2D/3D MRA paper-level 性能。

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

| 缺口 | 现状 | paper-like / paper-level 需要 |
| --- | --- | --- |
| **平滑算子** | Gaussian filter | 真实 framelet：piecewise linear B-spline 滤波器（Eq. 1），构造 A（Eq. 2），AᵀT_λ(A·) soft-thresholding（Eq. 4–5），2D 9 个 / 3D 27 个 framelet，AᵀA=I |
| **区间机制** | 固定线性收缩 alpha±0.008 | Eq. 9–12 完整实现：梯度初始化 Γ/Γ±/μ±，c(α) 对比函数，γ=1/5 自适应 [αᵢ,βᵢ] |
| **三分阈值-拉伸** | 简单 0.5 二分 + 钉 0/1 | Eq. 6 的 below/inside/above 三分 + linear contrast stretch，Eq. 7 的 Λ^(i) 定义 |
| **anisotropic 选项** | 无 | 把 h₁ 系数旋转到切向/法向，只切向阈值化（引 [7][26]）；论文核心增益来源之一 |
| **数据** | 合成 112×112 管网 | 真实 MRA carotid 182×62、MRA kidney 257×257、3D CTA 201³（私有，需申请）；或 DRIVE/STARE/TubeTK/VascuSynth 公开等价 |
| **对照基线** | 无 | 复现 [10] Chan-Vese、[16][17] PDE anisotropic diffusion 作对照（Fig. 2–4） |
| **评估** | toy Dice/IoU | 对齐论文：报告 |Λ^(i)| 收缩表（Table 1 形式）、迭代数；在公开数据上另报 Dice/sensitivity 但与论文区分 |
| **3D 支持** | 仅 2D | 27-framelet 3D 变换 + isosurface 边界提取（Example 3） |
| **参数对齐** | sigma=1.0, 固定区间 | γ=1/5；ε 按例子（1.6e-2 / 5e-3 / 6e-2）；λk=2^(-1/2)（iso）/ 0.1×2^(-1/2)（aniso 切向） |

## 9. 运行步骤

**当前 toy/partial 跑法**：

```bash
# 安装依赖（见 reproStructured.dependencies）
pip install -r requirements.txt   # numpy, scipy, scikit-image, matplotlib

# 运行全部复现实验（含本篇 tubular_tight_frame）
cd reproduce && python run_all.py

# 校验数据/PDF/笔记/静态资产
node docs/scripts/validate.mjs
```

依赖缺失时 runner 会写入 `skipped`（见 `require_modules`），不会伪造 completed。

**向 paper-like 扩展的步骤大纲**（不在本次写代码范围内，仅规划）：
1. 实现真实 framelet 变换模块（1D 滤波器 Eq.1 → 2D/3D 张量积 → A，AᵀT_λ(A·)），单测验证 AᵀA=I（perfect reconstruction）。
2. 实现 Eq. 6 三分阈值-拉伸 U(·) 与 Eq. 7 的 Λ^(i)。
3. 实现 Eq. 9–12 区间初始化/细化（梯度、c(α)、γ=1/5）。
4. 实现 Eq. 8 的 P^(i) restricted 更新，并加 sparse 加速。
5. 加 isotropic / anisotropic 两种阈值开关。
6. 接入公开数据（DRIVE/STARE 2D，TubeTK/VascuSynth 3D），按论文参数跑，输出 |Λ^(i)| 收缩表与迭代数，复现 Table 1 形式。
7. 实现并对照 [10] Chan-Vese、[16][17] PDE 基线（Fig. 2–4 对应）。

## 10. 风险与代理说明

- **Gaussian ≠ framelet**：高斯平滑是各向同性低通，缺少 framelet 的 redundant multi-orientation 表示与 soft-thresholding 的稀疏去噪特性；无法体现论文 anisotropic thresholding 在切向贴边、连接小遮挡上的核心增益。
- **合成管网 ≠ 真实 MRA/CTA**：合成数据无 speckle 噪声、partial occlusion、intersection、可变直径等真实难点；toy Dice 0.9981 主要反映"任务太简单"，不可外推。
- **固定区间 ≠ 自适应区间**：固定线性收缩绕过了 Eq. 9–12 的自适应机制，因此无法复现论文"几次迭代内 |Λ| 急剧收缩"的精确动力学（论文 6–9 iters，且 |Λ| 的衰减形态由 c(α)/γ 决定）。
- **不可外推的结论**：任何关于"优于 Chan-Vese [10] / PDE [16][17]"、"3D CTA 连接性更好"、"anisotropic 优于 isotropic"的结论 **均属论文结论，本仓库 toy 未验证**，不得据此声称复现成功。
- **Theorem 1 未在代码层验证**：有限步收敛到二值图像是论文证明（基于 |Λ^(i)| < |Λ^(i-1)| 严格递减），toy 中虽观察到 Λ 收缩，但未在忠实算子下验证该定理前提。更具体地：toy 循环到 12 次迭代上限即停，Λ 仅收缩到 2（lambda_final=2，**始终未达到 |Λ|=0**），最终二值掩膜是由 0.5 硬阈值强制得到，而非靠 |Λ|=0 收敛准则触发；因此论文 Theorem 1 的"有限步 |Λ|=0 收敛"在本 toy 中并未被真正演示，Λ 收缩叙述与未收敛事实需如此并置理解。

## 11. 参考：精读笔记

- 精读笔记：[`../../../xiaohao_cai_ultimate_notes/Framelet_Based_Tubular_Structures_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Framelet_Based_Tubular_Structures_超精读笔记_已填充.md)
- 论文 PDF：`docs/00_papers_first_author_xiaohao_cai_deduped/框架管状结构分割 Framelet.pdf`
- runner：`reproduce/experiments/tubular_tight_frame.py`
