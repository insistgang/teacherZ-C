# 分割与恢复耦合模型 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 4 篇 *Variational Image Segmentation Model Coupled with Image Restoration Achievements* 的完整复现流程规范。

## 1. 论文身份与第一作者核验

| 字段 | 内容 |
| --- | --- |
| 英文标题 | Variational Image Segmentation Model Coupled with Image Restoration Achievements |
| 中文标题 | 分割与恢复耦合模型 |
| 作者顺序 | Xiaohao Cai（单一作者；PDF 首页作者列表只列出 Xiaohao Cai） |
| 第一作者核验 | 是。PDF 首页署名为 "Xiaohao Cai"，单位为 University of Cambridge（Department of Plant Sciences, and Department of Applied Mathematics and Theoretical Physics）。Acknowledgement 致谢 University of Kaiserslautern 的 Gabriele Steidl 教授，说明主要工作在 Kaiserslautern 完成。 |
| 年份 | 2014（arXiv:1405.2128v1, 9 May 2014；preprint submitted to Elsevier） |
| PDF 路径 | `docs/00_papers_first_author_xiaohao_cai_deduped/分割恢复联合模型 Segmentation Restoration.pdf` |
| 主题 | sat-rof（分割—恢复耦合 / PCMS 扩展线） |

核验结论：本篇确为 Xiaohao Cai 第一（且唯一）作者论文，符合 15 篇口径纪律。

## 2. 复现目标与诚实分级

本项目对每篇论文区分四个递进等级：

- **toy**：自造极小 synthetic 图，用 proxy 算子替代论文严格求解器，只演示"恢复参与分割能改善结果"这一定性方向。
- **partial**：实现论文 pipeline 的核心子集（例如真实的 g 闭式更新 + c_i 更新），但数据/基线/参数仍不全。
- **paper-like**：复刻论文 Algorithm 1 完整三变量 AM、在论文同款或等价公开图像上跑、对比同类 baseline、报告 SA 指标且数量级与论文一致。
- **paper-level**：在论文原始数据与原始对照方法代码上复现，数值可逐表对齐。

**本仓库当前等级（reproductionLevel）= partial；真实性（reproductionTruthLevel）= partial-completed。**

纪律声明：截至本文档，本项目 **paper-level 复现仍为 0/15**。本篇当前实现已经把旧的 proxy 全部换成**真实算法**：真实 Gaussian 模糊算子 A（15×15 PSF，FFT 圆周卷积）、Eq.(13) 的**精确频域 Tikhonov g 子问题求解**、Eq.(14) 的 ω 加权区域均值、Eq.(15-16) 的**真实 Chambolle-Pock 多相 TV 区域拟合 u 子问题**（替代旧的 1D K-means），外层 AM 用 `‖c^{k+1}−c^k‖≤ε` 收敛判据。但数据仍为单张 96×96 合成图、无论文同款数据、无 [43]/[23]/[6] 基线、只实现 Gaussian fidelity（Poisson/impulsive 未做）、未做收敛定理的数值核验。因此当前 SA（direct ≈75.9 / joint ≈91.3）**只能解读为"恢复参与分割能在退化图上显著改善分割"的真实但有限佐证**，与论文报告的 SA（如 99.29、95.66 等，见第 6 节，论文同款数据 + 完整基线）**不在同一口径，不可混用，也不代表论文级精度**。

## 3. 算法完整流程

论文从 piecewise constant Mumford-Shah (PCMS) 模型出发，将图像恢复的 data fidelity 项耦合进多相分割能量。下面是忠于 PDF 的 step-by-step pipeline。

**符号约定（PDF Section 1-2）**
- f：观测（退化）图像，f : Ω → [0,1]。
- g：待恢复的 clean 图像，g ∈ L²(Ω)。
- A（论文记 𝒜）：problem-related linear operator——噪声情形取 identity，模糊情形取 blurring operator。
- u_i：第 i 相的 indicator / label 函数，约束 Σ_i u_i(x)=1, u_i(x)∈{0,1}（松弛后 ≥0）。
- c_i：第 i 相区域常数（codebook / class mean）。
- K：相数（phase 数）。
- μ, λ：分别平衡 restoration fidelity 与 segmentation fidelity 的正参数。

**核心能量（PDF Eq. 7）**

```
E(u_i, c_i, g) = μ Φ(f, A g) + λ Σ_{i=1}^K ∫_Ω (g - c_i)^2 u_i dx + Σ_{i=1}^K ∫_Ω |∇u_i| dx,
s.t.  Σ_i u_i(x) = 1,  u_i(x) ∈ {0,1}.
```

其中 Ψ(u_i,c_i,g) = Σ_i ∫(g-c_i)² u_i dx 是来自分割模型的 region-fitting 项，最后一项是 TV 正则（控制各相边界长度）。第一项 Φ(f,Ag) 来自图像恢复模型（PDF Eq. 1 的一般 restoration 形式）。

**三种噪声对应的 fidelity Φ（PDF Section 2）**
1. Gaussian noise：Φ(f,Ag) = ∫_Ω (f - Ag)² dx。
2. Poisson noise (I-divergence)：Φ(f,Ag) = ∫_Ω (Ag - f log(Ag)) dx。
3. Impulsive noise：Φ(f,Ag) = ∫_Ω |f - Ag| dx。

**缺失像素扩展（PDF Eq. 8-9）**：引入指示权 ω(x)=1 若 x∈Ω\Ω′（像素已知），否则 0（Ω′ 为缺失像素集），把两个 fidelity 都乘上 ω：

```
E = μ ∫_Ω (f - Ag)^2 ω dx + λ Σ_i ∫_Ω (g - c_i)^2 ω u_i dx + Σ_i ∫_Ω |∇u_i| dx.
```

**向量值（彩色）图像扩展（PDF Eq. 10）**：对 f=(f_1,…,f_N)、g=(g_1,…,g_N)、c_i=(c_{i,1},…,c_{i,N})，把恢复与 region 项对通道求和：

```
E = μ Σ_{j=1}^N ∫(f_j - A_j g_j)^2 ω dx + λ Σ_i Σ_j ∫(g_j - c_{i,j})^2 ω u_i dx + Σ_i ∫|∇u_i| dx.
```

**Algorithm 1 — 三变量 Alternating Minimization（PDF Section 3, Eq. 11-17）**

先把 u_i 松弛到凸集（Eq. 11-12）：约束 Σ_i u_i=1, u_i≥0。然后循环：

1. **初始化**：给定 f、相数 K、初始 codebook c^(0)（论文对 baseline [23][43] 用 fuzzy C-means 100 步初始化）、初始 u^(0)。
2. **更新 g（固定 u_i, c_i）**：g 只出现在前两项。对 Gaussian fidelity Φ=∫(f-Ag)²dx，子问题是二次的，闭式解（PDF Eq. 13）：
   ```
   g = (μ AᵀA + λ)⁻¹ (μ Aᵀf + λ Σ_i c_i u_i) ω.
   ```
   即一个 Tikhonov 型线性系统；A=identity 时退化为对每像素加权平均，A=blur 时需解去卷积线性系统（频域或共轭梯度）。Poisson / impulsive 情形论文留作 future work。
3. **更新 c_i（固定 u_i, g）**：c_i 只与 region 项相关，闭式（PDF Eq. 14）：
   ```
   c_i = ∫_Ω g ω u_i dx / ∫_Ω ω u_i dx.
   ```
   即第 i 相在已知像素上的加权均值。
4. **更新 u_i（固定 g, c_i）**：g 固定后第一项为常数，u 子问题退化为带 ω 的多相 model (6)。设 s=((g-c_i)²ω)_i，问题写成（PDF Eq. 15）：
   ```
   min_{v,u,d}  λ⟨v,s⟩ + ‖d‖_1 + ι_S(u),  s.t. ∇v=d, v=u,
   ```
   其中 ι_S 是单纯形 S={y∈ℝ^K | Σ y_i=1, y≥0} 的 indicator。用 ADMM / split-Bregman 迭代（PDF Eq. 16）更新 v、d、u 及两个 Bregman 变量 b_d, b_u。也可用 primal-dual 或 max-flow 求解。
5. **标签硬化**：解出松弛 u 后，每相 Ω_i = {x | u_i(x) = max(u_1,…,u_K)}（PDF Eq. 17）。
6. **收敛判据**：`while ‖c^(k+1) - c^k‖ > ε do` 循环 2-4 步，ε 取 10⁻⁴。

## 4. 完整复现所需数据集

论文实验数据（PDF Section 5），向 paper-like 复现需准备以下或其公开等价物：

| 论文使用 | 类型 | 公开 / 等价候选 |
| --- | --- | --- |
| Two-phase synthetic shapes（128×128） | 灰度合成 | 自造（圆+三角+方块），代码可复刻 |
| 2D barcode 图（195×195） | 灰度合成 | 任意 QR/barcode 生成器输出，或论文图 |
| Four-phase / five-phase synthetic（256×256, 91×91） | 灰度合成 | 自造多相图（不同形状 / 不同强度的 star） |
| Cameraman | 真实灰度 | 经典图，scikit-image `data.camera()` 等价可用 |
| MRI brain（319×256） | 真实医学灰度 | 公开脑 MRI 切片（如 BrainWeb），论文未给具体来源 |
| Rose（303×250×3） | 彩色 | 通用彩色花卉图 |
| Crown（225×300×3）、Flowers（188×250×3） | 彩色 | 通用彩色自然图 |

退化设置（PDF Section 5）：Gaussian noise 由 MATLAB `imnoise` 加（不同方差，如 0.2、0.05、0.01；blurry 图加 10⁻⁴ 方差）；默认 Gaussian blur 核 15×15、std=15；motion blur 核 15px、角度 90°；但 Fig.3 A3（five-phase）按 PDF 第11页特例用 10×10、std=10；信息丢失用 `rand` 随机移除，默认丢失 40%（部分实验 20%）。

私有 / 需注明：MRI brain 切片论文未给明确数据集出处，复现时需用公开脑 MRI 替代并注明"非论文原图"；彩色自然图同理（rose/crown/flowers 为示例图，非标准 benchmark）。

## 5. 对照基线 (Baselines)

论文（PDF Section 5）对比三种 state-of-the-art 多相分割方法（均为作者提供代码、参数试错调优）：

| 论文引用 | 方法 | 特点 |
| --- | --- | --- |
| [43] | max-flow 方法（Yuan 等） | 最小化 model (6)，**固定 c_i** 只优化 u_i |
| [23] | ADMM 方法（Pock-style 凸松弛多相） | 最小化 model (6)，**c_i 与 u_i 都优化**；彩色用 Eq.(10) 扩展后比较 |
| [6] | two-stage 分割（SaT 路线） | 先解凸变体 Mumford-Shah model (3)，再 thresholding；对一般含模糊图很有效 |

注：方法 [6][23][43] 三个基线均只能分割灰度图（PDF 第9页）；彩色对比时仅把 [23] 用 Eq.(10) 策略扩展为 "extended [23]"。论文方法（model (11)）是唯一同时带 restoration fidelity + segmentation fidelity 的。

## 6. 评价指标与论文报告结果

**指标定义（PDF Section 5）**：Segmentation Accuracy

```
SA = (#correctly classified pixels / #all pixels) × 100.
```

**论文报告的关键 SA（均为图注 / 括号内数值，已逐图核对 PDF）**

- Fig.1 two-phase shapes，noisy（A1）：[43] 99.50 / [23] 99.64 / [6] 99.48 / **Our 99.65**——噪声下各法都好，本法最高。
- Fig.1 two-phase shapes，**40% 信息丢失**（A2）：[43] 64.23 / [23] 98.13 / [6] 97.15 / **Our 99.29**——信息丢失下只有本法稳。
- Fig.1 barcode，noisy（A3）：[43] 97.91 / [23] 98.37 / [6] 98.08 / **Our 98.43**。
- Fig.1 barcode，**信息丢失**（A4）：[43] 68.27 / [23] 74.28 / [6] 86.11 / **Our 95.66**——差距最大的一组。
- Fig.2 four-phase，noisy / 20% 丢失：Our 99.65 / 99.48；[43] 99.64 / 75.41，[23] 99.63 / 86.89，[6] 97.96 / 95.88。
- Fig.2 five-phase（star），noisy / 丢失：Our 98.72 / 97.45；丢失行 [43] 85.61、[23] 84.17、[6] 86.11。
- Fig.3 four-phase **blurry**（Gaussian/motion）：Our 99.44 / 99.92；[43] 86.05 / 90.42，[23] 86.31 / 90.44，[6] 95.61 / 97.24。
- Fig.3 five-phase **blurry**：Our 96.38 / 96.96；[43] 72.91 / 71.05，[23] 72.66 / 71.25，[6] 92.66 / 92.53——模糊下 [43][23] 显著退化，本法最好。

定性结论（PDF Section 5-6）：在 noisy 图上各法接近；在 **信息丢失** 与 **模糊** 图上，本法（耦合 restoration）显著优于纯分割的 [43][23]，并优于 two-stage 的 [6]；彩色图上 extended [23] 对模糊给 over-smoothed 边界，本法更细。

**禁止编造**：以上数字逐一来自 PDF 图 1-3 的括号标注，可直接引用并注明图号。MRI brain、cameraman、彩色图（Fig.4-8）论文**只给视觉对比，未给 SA 数值**，复现时只能定性描述，不得杜撰精度。

## 7. 本仓库当前复现实现

- runnerFile：`reproduce/experiments/segmentation_restoration.py`
- 求解器/算法（已全部为真实实现，无 proxy）：
  1. **数据**：自造 96×96 三相 synthetic 图（两圆 + 背景，灰度 levels=[0.18,0.50,0.84]，三相分离良好）。
  2. **真实退化算子 A**：15×15 Gaussian PSF（std=2.2），以 FFT 圆周卷积实现 `A(img)=ℱ⁻¹(ℱ(img)·H)`；`Aᵀ` 用共轭特征值 `Hc=conj(H)`。退化图 `f = A(clean) + 高斯噪声(σ=0.07)`，并保留 ω 加权（当前 headline run ω≡1，缺失分量已实现但默认关闭以保证确定性/稳定）。
  3. **g 子问题（Eq.13，精确闭式）**：`(μAᵀA+λ)g = μAᵀf + λΣ_i c_i u_i`。因 A 为圆周卷积（周期 BC），`AᵀA` 在频域对角化为 `H2=|H|²`，故 g 用**逐频率除法** `g=ℱ⁻¹(ℱ(rhs)/(μH2+λ))` 精确求解（不是平滑近似，是真正解 Tikhonov 线性系统 / 去卷积）。codebook 目标 `Σ c_i u_i` 用 argmax 硬化后的标签，避免 codebook 退化塌缩。
  4. **c 子问题（Eq.14）**：`c_i = Σ(g·ω·1[lab=i]) / Σ(ω·1[lab=i])`，即恢复图 g 上第 i 相的 ω 加权均值。
  5. **u 子问题（Eq.15-16，真实多相 TV）**：`min_{u∈simplex} λ⟨u, ((g−c_i)²ω)_i⟩ + Σ_i TV(u_i)`，用 **Chambolle-Pock primal-dual** 迭代（梯度/散度算子 + 对偶变量投到单位球做各向同性 TV + 原始变量逐像素**投影到单纯形**），最后 argmax 硬化标签（Eq.17）。这替代了旧版的 1D K-means。
  6. **外层 AM**：`for outer in 15: g→c→u`，用 `‖c^{k+1}−c^k‖≤ε`（ε=1e-4）做收敛判据。
  7. **基线 `direct`**：用**同一个** Chambolle-Pock 多相 TV 求解器直接在退化图 f 上跑（即 model (6)，无恢复变量 g），c_i 同样用 ω 加权均值。这样 joint vs direct 的差异**只来自恢复耦合**，对照公平。
  8. **指标**：用论文 SA 定义（百分制，匈牙利匹配对齐相标签）算 direct/joint 的 Segmentation Accuracy，并报恢复图 g 相对 clean 的 PSNR。保存 5 联图（observed f / ground truth / direct segm. / restored g / joint segm.，标题内嵌 SA/PSNR）。
- 关键参数：K=3，μ=8.0，λ=1.0，ε=1e-4，blur std=2.2，noise σ=0.07，rng=`np.random.default_rng(SEED+4)`（确定性，禁用 wall-clock）。
- 当前 runMetrics（确定性，runtime≈1.2s，CPU）：
  - direct_SA_percent = 75.9
  - joint_SA_percent = 91.34
  - SA_gain_percent = 15.44
  - restoration_psnr_db = 22.84
  - am_outer_iterations = 15
  - 恢复出的 codebook ≈[0.20,0.50,0.87]，与真值 [0.18,0.50,0.84] 吻合，佐证 g 子问题确实恢复了对比度。
- 当前 resultFiles：`assets/repro/segmentation_restoration_partial.png`（observed f / truth / direct TV segm. / restored g / joint segm. 五联，标题含 SA/PSNR）。
- fidelityWarning（runner 用 extra 暴露）：合成单图、无论文同款数据、无 [43]/[23]/[6] 基线、仅 Gaussian fidelity、未核验收敛定理；SA 与论文 99.29/95.66 不同源不可比。

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

**已完成（本次升级，proxy → 真实算法）**：
- ✅ 恢复算子 A：已实现真实 15×15 Gaussian PSF + FFT 圆周卷积，`Aᵀ` 用共轭特征值。
- ✅ g 子问题（Eq.13）：已用**精确频域 Tikhonov 求解**（`g=ℱ⁻¹(ℱ(rhs)/(μ|H|²+λ))`），是真正的去卷积，不是平滑近似。
- ✅ c_i 更新（Eq.14）：已显式实现 ω 加权区域均值（在恢复图 g 上、按 argmax 硬标签估计）。
- ✅ u_i 子问题（Eq.15-16）：已用**真实 Chambolle-Pock 多相 TV primal-dual**（对偶投单位球 + 原始投单纯形），替代 K-means。
- ✅ 指标：已切换到论文 **SA（百分制）** 口径（匈牙利匹配对齐相标签），并报恢复 PSNR。
- ✅ 公平对照：direct 基线用同一 TV 多相求解器跑退化图 f（model (6)，无 g），joint vs direct 差异只来自恢复耦合。

**仍缺（到 paper-like / paper-level）**：

1. **u_i 求解器形式**：论文用 ADMM/split-Bregman（Eq.16，带两个 Bregman 变量 b_d,b_u）；本仓库用等价目标的 **Chambolle-Pock primal-dual**。数学等价、收敛同类，但**不是论文同款 split-Bregman 实现**，迭代变量/步长口径不同。
2. **运动模糊与更大 blur**：当前只用 Gaussian blur std=2.2；缺论文默认的 15×15 std=15、motion blur 15px@90°、以及 five-phase 特例 10×10 std=10 等档位。当前选 std=2.2 是为了去卷积良态 + 确定性稳定。
3. **缺失像素 ω 的 headline 演示**：ω 加权已**完整实现**（fidelity 与 region 项都乘 ω），但 headline run 设 ω≡1（noise+blur 档），因为在高缺失比例下 codebook 易塌缩、跨 seed 不稳；要做 paper 的 40%/20% 缺失档需更稳健的初始化（如 fuzzy C-means 100 步，论文做法）。
4. **Poisson / impulsive fidelity**：仅实现 Gaussian Φ=‖f−Ag‖²；Poisson（I-divergence）与 impulsive（L1）的 g 求解未做（论文亦把这两者的 g 留作 future work，但 u/c 仍可补）。
5. **向量值 / 彩色**：完全未做（Eq.10 多通道、共享 u_i）。
6. **基线**：缺 [43] max-flow、[23] ADMM（凸松弛多相）、[6] two-stage（SaT）三个论文对照实现；当前只有"同一 TV 求解器直接跑 f"这一内部对照。
7. **数据**：缺 barcode（195×195）、cameraman、MRI brain、rose/crown/flowers 等论文同款 / 公开等价图；当前只有一张自造 96×96。
8. **表格逐项对齐**：当前 SA（75.9 / 91.3）是自造图 + 内部基线，**无法逐表对齐**论文 Fig.1-3 的 99.29/95.66/99.92 等（需论文同款数据 + 三个原始基线代码）。
9. **收敛性验证**：缺能量单调下降（Theorem 2）/ partial minimizer 收敛（Theorem 4）的数值核验；当前只记录 `‖c^{k+1}−c^k‖` 收敛与外层迭代数。
10. **初始化**：论文对 baseline 用 fuzzy C-means 100 步初始化 codebook；当前用观测像素分位数 + direct 标签初始化（足够稳，但非论文初始化）。

## 9. 运行步骤

**当前 toy/partial 怎么跑**

```bash
# 安装依赖（reproStructured.dependencies）
pip install -r requirements.txt   # 含 numpy, scipy, matplotlib

# 运行全部复现（含本篇 toy）
cd reproduce && python run_all.py
```

依赖：numpy、scipy、matplotlib（见 reproStructured.dependencies）。若缺依赖，runner 写入 `skipped` 而非伪造 `completed`（项目纪律）。算力：CPU，约 1.2 秒（真实去卷积 + 多相 TV primal-dual）。产物图：`docs/assets/repro/segmentation_restoration_partial.png`（五联：observed f / truth / direct TV segm. / restored g / joint segm.）。

单实验自测（venv 解释器）：

```bash
MPLCONFIGDIR=/private/tmp/teacherZ-mplconfig \
  /Users/insistgang/Desktop/zx/teacherZ-C/.venv/bin/python -c \
  "import sys; sys.path.insert(0,'reproduce/experiments'); import segmentation_restoration as m; print(m.run())"
```

**已落地（本次升级完成）**：真实 blur 算子 A + Eq.(13) 频域去卷积 g 求解；Eq.(14) ω 加权均值；Eq.(8-9) 的 ω 权重（实现，headline 默认 ω≡1）；Eq.(15-16) 真实多相 TV u 子问题（Chambolle-Pock，替代 K-means）；`while ‖c^{k+1}-c^k‖>ε` 收敛判据（ε=1e-4）；SA 百分制指标。

**向 paper-like 进一步扩展的步骤大纲（仍未做，仅规划）**

1. 把 u 子问题换成论文同款 split-Bregman/ADMM（Eq.16，带 b_d,b_u），与 Chambolle-Pock 结果交叉验证。
2. 引入论文档位的强 Gaussian/motion blur（15×15 std=15 / 15px@90°）并用更稳健的去卷积（Wiener/CG + 边界处理）。
3. 用 fuzzy C-means 100 步初始化 codebook，打开 ω 缺失像素档（20%/40%）做 paper 的缺失实验且保持稳定。
4. 实现 Poisson（I-divergence）/ impulsive（L1）fidelity 的 u/c 更新；向量值彩色（Eq.10）。
5. 准备论文同款 / 公开等价数据（barcode、cameraman、公开 MRI、彩色图）。
6. 加 [43] max-flow、[23] ADMM、[6] two-stage 三个原始基线做定量对照，与论文图 1-3 数值对齐（标注非原始数据）。

## 10. 风险与代理说明

**已消除的旧代理**（不再适用）：
- ~~Gaussian smoothing 代替去卷积~~ → 现为**真实频域 Tikhonov 去卷积**（Eq.13）。
- ~~K-means 代替多相分割~~ → 现为**真实 Chambolle-Pock 多相 TV**（Eq.15-16），含边界长度正则与空间一致性。
- ~~clustering accuracy 口径~~ → 现报**论文 SA 百分制**。

**仍存在的局限/风险**：
- **求解器非论文同款**：u_i 用 Chambolle-Pock primal-dual 而非论文的 split-Bregman/ADMM（Eq.16）。两者解同一凸目标、收敛同类，但实现与中间变量口径不同，不能声称"逐步复刻论文 Algorithm 1 的 u 子问题实现"。
- **blur 档位受限**：为保证去卷积良态与确定性，用 std=2.2 的 Gaussian blur；论文默认的强模糊（15×15 std=15、motion blur）下去卷积更病态，本仓库未覆盖，故不能代表论文 Fig.3 的强去模糊能力。
- **缺失像素未进 headline**：ω 加权已实现，但 headline run 关闭缺失（ω≡1）；论文 40%/20% 缺失档需更稳健初始化才不塌缩，当前未做。
- **SA 口径相同但数据/基线不同**：本仓库 SA（75.9/91.3）来自自造图 + 内部基线，与论文 99.29/95.66（论文同款数据 + 三个原始基线）**不可逐表比较**，也不能声称"复现了论文精度"。
- **未验证收敛定理**：只记录 codebook 收敛 `‖c^{k+1}−c^k‖`，未数值核验能量单调下降（Theorem 2）/ partial minimizer 收敛（Theorem 4）。
- **可外推到何处**：当前结果**真实地**支持"把已知退化算子 A 的恢复并入分割能量，能在 blur+noise 退化图上显著提升 SA（≈+15 分），并恢复出与真值吻合的 codebook"；但 Poisson/impulsive fidelity、向量值彩色、强去模糊、与 [43]/[23]/[6] 三基线的定量优势，本仓库仍未复现，paper-level 仍为 0/15。

## 11. 参考：精读笔记

精读笔记：[../../../xiaohao_cai_ultimate_notes/Variational_Segmentation-Restoration_超精读笔记_已填充.md](../../../xiaohao_cai_ultimate_notes/Variational_Segmentation-Restoration_超精读笔记_已填充.md)
