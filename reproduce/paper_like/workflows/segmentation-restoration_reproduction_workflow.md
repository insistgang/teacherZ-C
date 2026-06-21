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

**本仓库当前等级（reproductionLevel）= toy；真实性（reproductionTruthLevel）= toy-completed。**

纪律声明：截至本文档，本项目 **paper-level 复现仍为 0/15**。本篇的 toy 实现用 `scipy.ndimage.gaussian_filter` 作为恢复 proxy，用 1D K-means 作为分割 proxy，**没有**实现论文真正的去模糊算子 A、ADMM/split-Bregman 的 u_i 子问题、收敛证明所依赖的能量单调性。因此 toy 的数值（如 0.5332 → 0.9604）**只能解读为"耦合方向有效"的玩具佐证，禁止外推为论文级精度**。论文报告的 SA（如 99.29、95.66 等，见第 6 节）与本仓库 toy accuracy 不在同一口径，不可混用。

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
- 它实际做了什么：
  1. 自造 96×96 三相 synthetic 图（圆+圆，灰度 levels=[0.22,0.52,0.78]）。
  2. 退化：`gaussian_filter(sigma=2.0)` + 高斯噪声(σ=0.11) + 随机 12% 缺失像素置 0。
  3. `direct`：直接在退化图上做 1D K-means（K=3）→ direct segmentation（proxy of model (6)）。
  4. 耦合 toy：循环 8 次，用 `gaussian_filter(sigma=1.1)` 平滑作为恢复 proxy，对缺失像素回填，凸组合 `g = 0.55*g + 0.45*restored`，再对 g 做 K-means → joint labels。
  5. 用 `clustering_accuracy` 对齐 ground truth 计算精度，保存 4 联图。
- 使用的 proxy（明确）：
  - 恢复算子 A 与去卷积 → 用 **Gaussian smoothing 近似**，没有解 PDF Eq.(13) 的 Tikhonov 系统，也没有真正的去模糊。
  - u_i 多相 ADMM/split-Bregman（PDF Eq.16）→ 用 **1D K-means** 近似。
  - c_i 加权均值（Eq.14）、ω 指示权 → 隐含在 K-means / 回填中，未显式实现。
  - 收敛证明（Theorem 1-4）→ 未涉及，只跑固定 8 次迭代。
- 当前 runMetrics（取自 reproStructured，runtime≈0.111s）：
  - direct_accuracy = 0.5332
  - joint_toy_accuracy = 0.9604
  - accuracy_gain = 0.4272
  - alternating_iterations = 8
- 当前 resultFiles：`assets/repro/segmentation_restoration_toy.png`（degraded / truth / direct K-means / joint toy 四联）。

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

清单（按子问题列出与 PDF 的缺口）：

1. **恢复算子 A**：缺真实 blurring operator 与去卷积。需实现 Gaussian/motion blur 核（15×15 std15 / 15px 90°）及 Eq.(13) 的 (μAᵀA+λ)⁻¹ 线性求解（频域或 CG）。当前只有 Gaussian 平滑。
2. **g 子问题**：缺 Tikhonov 闭式 / 线性系统求解；当前用凸组合平滑硬替。
3. **u_i 子问题**：缺 ADMM/split-Bregman（Eq.15-16）或 primal-dual / max-flow 的真正凸多相求解；当前用 K-means。
4. **c_i 更新**：缺 Eq.(14) 的 ω 加权均值显式实现。
5. **缺失像素 ω**：当前把缺失像素置 0 后平滑回填，未把 ω 作为权重进 fidelity（Eq.8-9）。
6. **向量值 / 彩色**：完全未做（Eq.10 多通道）。
7. **基线**：缺 [43] max-flow、[23] ADMM、[6] two-stage 三个对照实现；当前只有"退化图直接 K-means"这一弱对照。
8. **数据**：缺 barcode、cameraman、MRI、彩色等论文同款图；当前只有一张自造 96×96。
9. **指标与表格对照**：当前 accuracy 是 K-means clustering accuracy，与论文 SA 口径不同，无法逐表对齐论文 99.29/95.66 等。
10. **收敛性验证**：缺能量单调性 / partial minimizer 的实验性核验（Theorem 2-4）。

## 9. 运行步骤

**当前 toy/partial 怎么跑**

```bash
# 安装依赖（reproStructured.dependencies）
pip install -r requirements.txt   # 含 numpy, scipy, matplotlib

# 运行全部复现（含本篇 toy）
cd reproduce && python run_all.py
```

依赖：numpy、scipy、matplotlib（见 reproStructured.dependencies）。若缺依赖，runner 写入 `skipped` 而非伪造 `completed`（项目纪律）。算力：CPU，约 1 秒内（runtime≈0.111s）。产物图：`docs/assets/repro/segmentation_restoration_toy.png`。

**向 paper-like 扩展的步骤大纲（不在本次改动范围，仅规划）**

1. 实现真实 blur 算子 A 与 Eq.(13) 的 g 求解（频域去卷积或 CG）。
2. 实现 Eq.(14) c_i 加权均值、Eq.(8-9) 的 ω 权重。
3. 实现 Eq.(15-16) 的 ADMM/split-Bregman u_i 子问题，替换 K-means。
4. 用 Algorithm 1 的 `while ‖c^{k+1}-c^k‖>ε` 控制循环（ε=10⁻⁴）。
5. 准备论文同款 / 公开等价数据（barcode、cameraman、公开 MRI、彩色图）。
6. 接入 SA 指标（按 Eq. SA 定义），与论文图 1-3 数值对照（仅作趋势对齐，标注非原始数据）。
7. 可选：加 [6] two-stage 与一个凸多相 baseline 做对照。

## 10. 风险与代理说明

- **Gaussian smoothing ≠ 去卷积**：toy 用平滑做"恢复"，对真正的 Gaussian/motion blur 无法反演，所以 toy 在"模糊"维度的优势不可代表论文 Fig.3 的去模糊能力。
- **K-means ≠ TV 正则多相分割**：K-means 无边界长度正则，缺 PCMS 的空间一致性；论文 u_i 子问题的 TV 项是其抗噪/抗缺失关键，被 proxy 抹掉。
- **accuracy 口径不一致**：toy 的 clustering accuracy（0.5332→0.9604）与论文 SA（百分制，如 99.29）不同源，**两者不能直接比较，也不能声称"复现了论文精度"**。
- **未验证收敛**：toy 固定 8 次迭代，未检验能量单调下降（Theorem 2）或 partial minimizer 收敛（Theorem 4）。
- **不可外推结论**：toy 仅支持"把恢复信息引入分割能让退化图分割更稳"这一**定性方向**；论文级的 Poisson/impulsive fidelity、向量值彩色、去模糊、与三个 baseline 的定量优势，本仓库均未复现。

## 11. 参考：精读笔记

精读笔记：[../../../xiaohao_cai_ultimate_notes/Variational_Segmentation-Restoration_超精读笔记_已填充.md](../../../xiaohao_cai_ultimate_notes/Variational_Segmentation-Restoration_超精读笔记_已填充.md)
