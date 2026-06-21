# PCMS 与 ROF 的理论连接 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 2 篇 Linkage Between Piecewise Constant Mumford-Shah Model and ROF Model and Its Virtue in Image Segmentation 的完整复现流程规范。

## 1. 论文身份与第一作者核验

| 字段 | 内容 |
| --- | --- |
| 英文标题 | Linkage Between Piecewise Constant Mumford-Shah Model and ROF Model and Its Virtue in Image Segmentation |
| 中文标题 | PCMS 与 ROF 的理论连接 |
| 作者顺序 | Xiaohao Cai, Raymond Chan, Carola-Bibiane Schönlieb, Gabriele Steidl, Tieyong Zeng |
| 第一作者核验 | 是。PDF 首页作者列表以 `XIAOHAO CAI∗` 开头（标记 ∗ 对应 DAMTP, University of Cambridge / MSSL, UCL，邮箱 x.cai@ucl.ac.uk），其余四位为合作者。 |
| 年份 | arXiv:1807.10194，v1 于 2018 年提交，v2 于 2019-10-15 更新（PDF 首页标注 `arXiv:1807.10194v2 [math.NA] 15 Oct 2019`） |
| arXiv ID | 1807.10194 |
| PDF 路径 | docs/00_papers_first_author_xiaohao_cai_deduped/变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf |
| 主题 | sat-rof（Smoothing-and-Thresholding / ROF 阈值化分支，理论核心篇） |

核验结论：本篇满足 15 篇口径内"Xiaohao Cai 为第一作者"的硬约束。它是 SaT/T-ROF 路线中**理论合法性论文**：第 1 篇 SaT Overview 给出"恢复+阈值化"方法论，第 3 篇 Iterated ROF 给出多类阈值更新算法原型，而本篇负责回答**为什么恢复解经过阈值化可以解释为分割解**——把启发式步骤放回变分能量与 minimizer 关系中严格分析。

## 2. 复现目标与诚实分级

本项目对复现真实性采用四级口径，必须如实标注，禁止夸大：

- **toy**：只用合成图、用 Gaussian smoothing 等代理替换真实 ROF/TV 求解，仅演示"先平滑后阈值"的现象。
- **partial**：使用真实凸 ROF 求解器（Chambolle-Pock 对偶投影 / Split-Bregman），实现论文核心阈值更新公式与 K=2 Proposition / Theorem 的代理校验，但**不用论文真实数据、不复刻 Table 5.1-5.4 的完整 baseline 对照、不证明定理**。
- **paper-like**：使用论文同源或公开等价数据（cartoon、close-intensity、Gaussian noisy multiphase、brain MRI、stripe、retina vessel），实现完整 T-ROF Algorithm 1 + 论文 baseline 对照，复现 SA / DICE 的数量级与趋势。
- **paper-level**：在论文真实数据、真实参数（λ/μ）、真实 baseline 实现下，复现 Table 5.1-5.4 的具体数值与 Fig. 5.1-5.9 的可视化。

| 维度 | 当前状态 |
| --- | --- |
| 本仓库复现等级 reproductionLevel | **partial** |
| 真实性等级 reproductionTruthLevel | **partial-completed** |

**纪律声明**：截至本文档，15 篇论文 paper-level 复现仍为 **0/15**。本篇当前为 partial：headline `rof_threshold_dice` 已由真实凸 ROF（Chambolle-Pock）阈值化产出，Gaussian smoothing 仅作对照 baseline。

特别提醒（针对本篇的核心诚实边界）：本论文的核心贡献是**定理**（Theorem 3.4 / 3.6 / 3.7 与收敛性 Theorem 4.6），不是某个可被"跑分超过"的数据集。代码只能在合成图上**演示** ROF minimizer 阈值化的现象、并对 Theorem 3.6 的 K=2 结论做一个 proxy 检查；**它不构成、也不能替代定理证明**。任何"代码复现了 Theorem 3.6"的表述都是错误的，本仓库当前的 `pcms_like_energy`、`rof_threshold_dice` 等数值只能解读为合成 toy 现象，绝不能与论文 Table 中的 SA/DICE 直接等同。

## 3. 算法完整流程

论文总思路：把"二相/多相非凸 PCMS（Chan-Vese）直接优化"替换为"**解一次凸 ROF，再对同一个 ROF 解做阈值化，并按区域均值迭代更新阈值**"，并证明这一替换在 K=2 时给出 PCMS 的 partial minimizer。关键在于 ROF 极小元在整个阈值循环中保持不变（Proposition 3.3），因此 ROF 只需求解一次。

### 3.1 关键模型与公式（忠于 PDF）

记 Ω⊂R² 为有界开集，f:Ω→[0,1] 为给定（退化）图像，meanf(A):=(1/|A|)∫_A f dx（|A|>0 时，否则为 0）。

1. **Mumford-Shah 出发点（Eq. 1.1）**：E_MS(u,Γ;Ω)=H¹(Γ)+λ'∫_{Ω\Γ}|∇u|²dx+λ∫_Ω(u-f)²dx（λ',λ>0）；H¹ 为一维 Hausdorff 测度。

2. **PCMS 模型（Eq. 1.2-1.3）**：限制 ∇u=0（即 u 分块为常数）得 E_PCMS(u,Γ;Ω)=H¹(Γ)+λ∫_Ω(u-f)²dx；在 Ω=⋃_{i=0}^{K-1}Ω_i、u≡m_i 的假设下写成
   E_PCMS(Ω,m) = ½ Σ_{i=0}^{K-1} Per(Ω_i;Ω) + λ Σ_{i=0}^{K-1} ∫_{Ω_i}(m_i - f)² dx。

3. **K=2 即 Chan-Vese（Eq. 1.4）**：E_CV(Ω₁,m₀,m₁) = Per(Ω₁;Ω) + λ[∫_{Ω₁}(m₁-f)²dx + ∫_{Ω\Ω₁}(m₀-f)²dx]。直接最小化易陷局部极小（Chan-Vese 原文用 level set）。

4. **Chan-Esedoglu-Nikolova 紧凸松弛（Eq. 1.5）**：固定 m₀,m₁，可由解凸问题 ū=argmin_{u∈BV} TV(u)+λ∫_Ω[(m₀-f)²-(m₁-f)²]u dx，再令 Ω₁:={x:ū(x)>ρ}（任意 ρ∈[0,1)）得全局极小。这是"凸恢复 + 阈值化"思路的前身，但它**只对 fixed m₀,m₁ 成立**，且其阈值集是否对应某个能量的解、对一般 TV 替换（weighted TV、framelet）并无理论保证——本篇正是补上这块理论。

5. **ROF 模型（Eq. 1.7）**：min_{u∈BV(Ω)} TV(u) + (μ/2)∫_Ω (u-f)² dx，μ>0。凸、有全局极小，是图像恢复经典模型。

6. **T-ROF 单阈值能量（Eq. 3.1）**：固定 τ∈(0,1)，E(Σ,τ) := Per(Σ;Ω) + μ∫_Σ (τ - f) dx。脚注说明：因 f∈[0,1]，τ≤0 时全局极小为 Ω，τ≥1 时为 ∅，故只关心 τ∈(0,1)。

7. **凸化求解（Proposition 3.1）**：E(·,τ) 的极小集 Σ_τ 可由解凸问题 ū=argmin_{u∈BV} TV(u)+μ∫_Ω(τ-f)u dx，再令 Σ_τ={x:ū(x)>ρ}（任意 ρ∈[0,1)）得到。证明引用综述 [20] Prop. 2.1，思想同 [7,34]。

8. **嵌套集引理（Lemma 3.2）**：固定 0<τ₁<τ₂<1，对应极小集满足 |Σ₂\Σ₁|=0，即 Σ₁⊇Σ₂（up to 零测集）。

9. **ROF 等价（Proposition 3.3，引 [20] Prop. 2.6）**：Σ_τ:={x:u(x)>τ} 对每个 τ∈(0,1) 解 E(·,τ) **当且仅当** u 解 ROF 模型 (1.7)。这是"只解一次 ROF、用不同阈值切多类"的核心依据。

10. **多类 T-ROF 能量与模型（Eq. 3.5-3.9）**：Σ={Σ_i}_{i=1}^{K-1}，τ={τ_i}_{i=1}^{K-1}，τ_i<τ_j (i<j)，
    E(Σ,τ) = Σ_{i=1}^{K-1}[Per(Σ_i;Ω) + μ∫_{Σ_i}(τ_i - f)dx]。
    T-ROF 模型寻找 (Σ*,τ*)，使 (i) E(Σ*,τ*)≤E(Σ,τ*) 对所有 Σ⊂Ω^{K-1}（Eq. 3.7），且 (ii) 阈值满足 τ_i* = ½(m*_{i-1}+m*_i)，其中 m*_i=meanf(Ω*_i)、Ω*_i=Σ*_i\Σ*_{i+1}、Σ*_0=Ω、Σ*_K=∅（Eq. 3.8-3.9）。注意这与"对所有可行 τ 也最小化"的 (3.10) 不同——T-ROF 不在 τ 上做无约束最小化，而是用均值中点条件锚定 τ。

11. **partial minimizer 定义（Eq. 3.11）**：(Σ*,m*) 是 partial minimizer 当 E(Σ*,m*)≤E(Σ*,m)（对所有可行 m）且 E(Σ*,m*)≤E(Σ,m*)（对所有可行 Σ）。关键陷阱：**partial minimizer 不必是 local minimizer，反之亦然**（PDF Fig. 3.1 用 (x,y)↦Re((x+iy)⁴)=x⁴-6(xy)²+y⁴ 在原点的例子说明）。

### 3.2 三个核心定理（本篇的真正贡献）

- **Theorem 3.4（T-ROF 与 PCMS，K=2）**：设 (Σ*₁,τ*₁) 满足 T-ROF 模型 (3.7)-(3.8) 且 0<|Σ*₁|<|Ω|，则 (Σ*₁,m*₀,m*₁) 是 PCMS/Chan-Vese 模型 (1.4) 的 partial minimizer，参数 **λ = μ / (2(m*₁-m*₀))**。证明骨架：由 E(Σ*₁,τ*₁)≤E(∅,τ*₁)=0 推出 τ*₁<m*₁；由 E(Σ*₁,τ*₁)≤E(Ω,τ*₁) 推出 m*₀≤τ*₁；代入 τ*₁=(m*₁+m*₀)/2 并配方，把单阈值能量加常数化成 E_CV(Σ,m₀,m₁)（Eq. 3.13）。

- **Remark 3.5（λ 的自适应机制）**：因 f∈[0,1]，有 0<m*₁-m*₀≤1，故 λ=μ/(2(m*₁-m*₀))≥μ，且当 m*₁-m*₀ 变小时 λ 急剧增大——即两类灰度越接近、数据项被加权越重。这解释了 T-ROF 为何能分开"相近灰度类别"：Chan-Vese 需要事先盲选 λ，而 T-ROF 只需自动调阈值 τ*₁ 即可隐式给出合适 λ。

- **Theorem 3.6（ROF 与 PCMS，K=2，本篇标题级结论）**：设 u* 解 ROF 模型 (1.7)，给定 0<m₀<m₁≤1，令 Σ̃:={x∈Ω:u*(x)>(m₁+m₀)/2} 且 0<|Σ̃|<|Ω|，则 Σ̃ 是固定 m₀,m₁、λ=μ/(2(m₁-m₀)) 时 PCMS/Chan-Vese 模型 (1.4) 的 minimizer。特别地，若 m₀=meanf(Ω\Σ̃) 且 m₁=meanf(Σ̃)，则 (Σ̃,m₀,m₁) 是 PCMS 的 partial minimizer。证明依赖 (3.14)：E(Σ,(m₁+m₀)/2)+λ∫_Ω(m₀-f)²dx = E_CV(Σ,m₀,m₁)，再用 Proposition 3.3。

- **Theorem 3.7（T-ROF 与 PCMS-V，K>2）**：对 K>2，若 T-ROF 解满足 m*_i<m*_{i+1}，则 {Ω*_i,m*_i} 是**变体 PCMS-V 模型 (3.16)** 的 partial minimizer，其正则参数逐相不同（Eq. 3.17）：边界相 μ̃₀=μ/(2(m*₁-m*₀))、μ̃_{K-1}=μ/(2(m*_{K-1}-m*_{K-2}))，内部相为相邻两项之和。**关键限制**：T-ROF 与标准 PCMS 的等价性在 K>2 时仅当 ∂Σ_i∩∂Σ_{i+1}=∅（即 ROF 解相邻跳变不重叠）时成立；否则只能得到对 PCMS 的**近似**。Remark 3.8 指出 PCMS-V 的逐相自适应 μ̃_i 反而对"相近灰度多相图"更有利。

### 3.3 T-ROF Algorithm 1（可执行 pipeline）

输入：退化图 f:Ω→[0,1]，相位数 K≥2，参数 μ>0，初始阈值 0≤τ₁⁽⁰⁾<…<τ_{K-1}⁽⁰⁾≤1。

1. **一次 ROF 求解**：计算 u = argmin TV(u)+(μ/2)∫(u-f)²dx（论文数值实验用 ADMM；亦可用 primal-dual / split-Bregman）。此后 u 固定不变。
2. **主循环（k=0,1,…，直到收敛）**：
   a. **阈值化**：Σ_i⁽ᵏ⁾ = {x∈Ω : u(x) > τ_i⁽ᵏ⁾}，i=1,…,K-1；由嵌套性得 Ω_i⁽ᵏ⁾=Σ_i⁽ᵏ⁾\Σ_{i+1}⁽ᵏ⁾（Σ_0=Ω,Σ_K=∅）。
   b. **清理无效相 / 不必要分割**：应用准则 (4.2)（去零测度相）与 (4.5)（Lemma 4.2：若朴素分割优于继续二分则保留，记为算子 C(·)，见步骤 i)-iii)）。
   c. **更新均值**：m_i⁽ᵏ⁾ = meanf(Ω_i⁽ᵏ⁾)。
   d. **更新阈值（Eq. 4.1）**：τ_i⁽ᵏ⁺¹⁾ = ½(m_{i-1}⁽ᵏ⁾ + m_i⁽ᵏ⁾)，记为 τ⁽ᵏ⁺¹⁾=Φ(Σ⁽ᵏ⁾,τ⁽ᵏ⁾)。
   e. **收敛判据**：‖τ⁽ᵏ⁺¹⁾-τ⁽ᵏ⁾‖₂ < ε 则停止。
3. 输出分割 {Ω_i}_{i=0}^{K-1}。

### 3.4 收敛性（Theorem 4.6 直觉）

论文构造符号序列 ζ⁽ᵏ⁾（记录每个 τ_i 相邻迭代的升/降，Eq. 4.10-4.12），令 s_k 为 ζ⁽ᵏ⁾ 中的"符号变号数"。**Lemma 4.5**：s_k 关于 k 单调不增；若 ζ₁⁽ᵏ⁺¹⁾≠ζ₁⁽ᵏ⁾ 则严格下降 s_{k+1}<s_k。由于 τ 有界于 [0,1] 且变号数有限并单调下降，归纳可得 **Theorem 4.6**：T-ROF Algorithm 1 产生的阈值序列 (τ⁽ᵏ⁾)_k 收敛到某 τ*，且 (Σ*,τ*) 是 T-ROF 模型 (3.7) 的解。K=2 时 s_k≡0，收敛尤其直接。实验上一般约十步内收敛（Fig. 5.8）。

## 4. 完整复现所需数据集

论文实验（Section 5）使用的数据，可达 paper-like 的公开/等价候选：

| 论文示例 | 数据描述（PDF） | 公开/等价候选 |
| --- | --- | --- |
| Example 1 | 两相 cartoon 图（256×256），随机移除 80% 像素（missing pixels） | 自合成：常数两相图 + 随机掩膜（可复现，无版权问题） |
| Example 2 | 两相 close-intensity 图（128×128）：在常数 0.5 图上加 mean 0、方差 10⁻⁵ 的 Gaussian 噪声 + mask | 自合成（论文给了完整生成方式，可严格复刻） |
| Example 3 | 五相（five-phase）noisy cartoon 图（91×96）：clean 图 + mean 0、方差 10⁻² Gaussian 噪声 | 自合成 |
| Example 4 | 四相 brain MRI（gray/white matter，319×256），取自 Pock [35] | 公开脑 MRI（BrainWeb 仿真 MRI、IBSR 等）作等价替代；论文原图需向 [35] 出处获取 |
| Example 5 | stripe 图（140×240，30 条纹），加 mean 0、方差 10⁻³ 噪声 | 自合成（条纹+噪声，论文给了参数） |
| Example 6-7 | close-intensity 合成图：Exa.6 三相（three-phase），Exa.7 四相（four-phase，噪声方差 3×10⁻²） | 自合成 |
| Fig. 5.9 retina | DRIVE 数据集的人工分割眼底血管图，改成三相（把右侧血管灰度由 1 降到 0.3） | **DRIVE 公开数据集**（http://www.isi.uu.nl/Research/Databases/DRIVE/，需注册下载；论文脚注 2 即此来源） |

说明：本篇绝大多数实验是**可严格复刻的合成图**（生成方式在 PDF 中写明），只有 brain MRI 取自他人工作、retina 来自 DRIVE 公开集。**无私有医学/RI 数据依赖**，这使 paper-like 比依赖私有数据的论文更可达成。

## 5. 对照基线 (Baselines)

论文在 Table 5.1-5.4 中固定对照如下五个 PCMS/分割方法（编号即 PDF 参考文献号）：

| 基线 | 方法类型 | 角色 |
| --- | --- | --- |
| Li [32] | hybrid level set（用边界特征图 + 固定阈值数据项） | 经典 level set 对照 |
| Pock [35] | PCMS 凸（非紧）松弛 | SOTA PCMS 凸松弛 |
| Yuan [39] | continuous max-flow / 凸优化分割 | 凸优化对照 |
| He [30] | PCMS 相关凸方法 | 多相对照 |
| Cai [15] | SaT（Smoothing-and-Thresholding，本组前作） | 最近邻基线，与 T-ROF 同源 |
| **T-ROF（本文）** | 一次 ROF + 迭代阈值更新 (4.1) | 提出方法 |

合理的现代补充基线（非论文原文，仅供 paper-like 拓展参考）：标准 Chambolle-Pock/ADMM 求解的 Chan-Vese 凸松弛、K-means/FCM 直接阈值、以及（若做监督对照）U-Net 类分割网络——但与论文"无训练、变分"定位不同，需明确标注是拓展而非原文对照。

## 6. 评价指标与论文报告结果

**指标定义（PDF）**：
- **SA（Segmentation Accuracy，Eq. 5.2）**：正确分类像素占比（整体准确率）。
- **DICE（Eq. 在 retina 节给出）**：DICE(Ω_i,Ω'_i)=2|Ω_i∩Ω'_i|/(|Ω_i|+|Ω'_i|)，逐相计算，既反映正确区域也惩罚错误区域，适合不平衡的细血管相。
- 另报告 λ/μ（数据项系数）、iteration steps（u 与 τ 分别迭代次数，记法如"418 (6)"=求 u 用 418 步、求 τ 用 6 步）、CPU time(s)。

**论文报告的关键数值（可从 PDF 直接确认，注明出处）**：

- **Table 5.1（Examples 1-4 的 SA）**，T-ROF（Our method）vs SaT [15] 的 SA：
  - Exa.1 missing pixels：T-ROF **0.9913**（time 8.34s）vs Cai[15] 0.9878（5.42s）vs He[30] 0.9888；
  - Exa.2 close-intensity：T-ROF **0.9845**（0.38s，最快）vs Cai[15] 0.9816；
  - Exa.3 Gaussian noisy：T-ROF **0.9831**（0.32s，最快）vs Cai[15] 0.9827；
  - Exa.4 MRI：T-ROF time **1.96s**（最快），SA 与 He[30] 同档（详见 PDF Table 5.1 第四块）。
- **Table 5.2（Examples 含 5/10/15 相多相，SA）**：T-ROF 在 5/10/15 相分别约 **0.9986 / 0.9967 / 0.9933**（time 1.39 / 2.33 / 3.74s），且其 time **基本与相数 K 无关**（论文强调点）；对照 Cai[15] 在 15 相退化到 SA 0.5280。
- **Table 5.3（Examples 6-7，三相/四相 close-intensity）**：Exa.6（三相）T-ROF SA **0.9550**（2.07s，68(6) 步）；Exa.7（四相）T-ROF SA **0.9798**（3.13s，111(5) 步）；均优于 SaT[15] 并最快或近最快。
- **Table 5.4（retina，Fig. 5.9，三相 DICE）**，T-ROF（Our method）：SA **0.9929**、DICE_Ω0（背景）**0.9962**、DICE_Ω1（右侧低强度血管）**0.7749**、DICE_Ω2（左侧血管）**0.9991**、time **2.09s**（35(15) 步）。对照 SaT[15]：SA 0.9803、DICE_Ω1 **0.5673**、time 3.51s。论文反复强调的关键提升即 **右侧低强度血管 DICE 0.7749 vs 0.5673**（PDF 正文与 Table 5.4 双重确认）。

注：现有精读笔记中"应用专家"小节给出的 retina 表（SA 0.9929 / DICE_Ω0 0.9962 / DICE_Ω1 0.7749 / time 2.09）与 PDF Table 5.4 **完全一致**，可信。笔记中"SA 提升 1.3%"等百分比是笔记作者基于这些数对算得的派生量，非 PDF 原文表格列。

## 7. 本仓库当前复现实现

- **runnerFile**：`reproduce/experiments/sat_rof_trof.py`（一个 runner 同时产出 priority 1/2/3 三篇的结果；本篇 id=`pcms-rof-linkage`）。
- **实际做了什么**：
  - 提供两个**真实凸 ROF 求解器**：`rof_chambolle_pock`（Chambolle-Pock 对偶投影，目标 TV(u)+(μ/2)‖u-f‖²）与 `rof_split_bregman`（交叉验证）。
  - `run_trof_thresholds` 实现 Eq. 4.1 的阈值更新（用 raw image 的 meanf 取均值、相邻均值中点更新阈值），并记录 drift、Lemma 3.2 单调性、Lemma 4.5 符号变号数等诊断量。
  - `run_k2_proposition_demo` 做 **Theorem 3.6 的 K=2 proxy 检查**：在合成两相图上求 ROF 解 u*，按 (m₀+m₁)/2 阈值化得 Σ̃，对比 ROF-threshold 与"直接对原图阈值"的 Dice，并按 λ=μ/(2(m₁-m₀)) 计算派生 λ。
- **本篇（pcms-rof-linkage）当前 runMetrics**（来自 runner 实测，合成两相图，确定性可复现）：
  - `direct_dice` = **0.8989**（对噪声原图直接阈值的 Dice）；
  - `gaussian_baseline_dice` = **0.9962**（Gaussian smoothing + 阈值的 Dice，仅作**对照 baseline**）；
  - `rof_threshold_dice` = **0.9983**（**真实 Chambolle-Pock ROF** 解阈值化的 Dice，是本篇 headline，高于 Gaussian 对照与 direct）；
  - `pcms_like_energy` = **205**（仅各向异性 perimeter/TV 形式的能量代理：`Σ|∂x sat2| + Σ|∂y sat2|`，**不含数据保真项**；此处 `sat2` 现为真实 ROF 阈值结果）。
  - `runtimeSeconds` ≈ **0.77**，CPU 约 1 秒内。
- **当前 resultFiles**：`assets/repro/sat_demo.png`（本篇绑定图；同 runner 另产出 `trof_thresholds.png`、`iterated_rof_convergence.png`、`iterated_rof_chanvese.png` 服务 priority 3）。
- **fidelity 警示（runner `extra` 内置）**：`"Real ROF on a synthetic toy two-phase image; pcms_like_energy is an anisotropic perimeter/TV proxy without the data-fidelity term and is not a paper-reported number. Code cannot substitute for the Theorem 3.4/3.6/3.7 proofs."`

诚实说明：本篇在 dashboard 上挂在 priority 2 的 `direct_dice / gaussian_baseline_dice / rof_threshold_dice / pcms_like_energy` 四项。headline `rof_threshold_dice` 现由 runner 主路径的**真实 Chambolle-Pock ROF**（`rof2 = rof_chambolle_pock(image2,...)`，`sat2 = rof2 > 0.48`）产出，Gaussian smoothing 已降级为对照（`gaussian_baseline_dice`）；`run_k2_proposition_demo` 也用真实 ROF 做 Theorem 3.6 的现象级检查。因此本篇升级为 **partial**。但它仍是合成图现象演示，**不是**论文级复现，更**不证明**任何定理。

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

到 **paper-like** 还缺：
1. **求解器（已部分完成）**：本篇 dashboard headline 路径已统一切换到真实 ROF（Chambolle-Pock），Gaussian 仅作对照。**仍缺**：论文用 **ADMM**（inner param 2）解 ROF，本仓库用 Chambolle-Pock / Split-Bregman，数值现象一致但非同一实现，需对齐求解器与参数。
2. **完整 Algorithm 1**：缺准则 (4.2)/(4.5) 的完整清理算子 C(·)（去零测度相 + 忽略不必要分割）、缺 FCM(100 迭代) 初始化（论文用 fuzzy C-means 给初始阈值）。
3. **数据**：缺论文 7 类示例的严格复刻（missing/close-intensity/Gaussian/MRI/stripe/multiphase/retina）。retina 需接入 **DRIVE** 公开集并按论文方式改三相。
4. **基线**：缺 Li[32]/Pock[35]/Yuan[39]/He[30]/Cai[15] 五个对照的可运行实现。
5. **指标与表格对照**：缺按 Eq. 5.2 的 SA、按 retina 节定义的逐相 DICE，以及 Table 5.1-5.4 的列对齐复现。

到 **paper-level** 还缺：
6. **真实参数**：复刻论文每例的 λ/μ（如 Exa.1 用 1、Exa.2-3 用 8、Exa.4（MRI）用 40、retina 用 25 等 Table 中具体值）与迭代步数记法（如 retina 35(15)）。
7. **数值复现**：在上述真实数据 + 真实 baseline 实现下复现 Table 5.4 的 T-ROF SA=0.9929、DICE_Ω1=0.7749 等具体数值。
8. **定理层面**：本篇的"full reproduction"在数学侧意味着**重读并复核 Theorem 3.4/3.6/3.7 与 Theorem 4.6 的证明**，代码无法替代这一步——只能提供与定理一致的实验佐证。

## 9. 运行步骤

**当前 toy/partial 跑法**：

```bash
pip install -r requirements.txt   # 依赖：numpy, scipy, matplotlib（见 reproStructured.dependencies）
cd reproduce && python run_all.py # 运行全部复现实验，产出 assets/repro/*.png 与结果 JSON
```

依赖缺失时，runner 会写入 `skipped` 而非伪造 `completed`（符合项目纪律）。本篇属于 `sat_rof_trof` 实验，与 priority 1/3 共用一次运行。

**向 paper-like 扩展的步骤大纲**（设计指引，当前不执行）：
1. （已完成）dashboard headline 路径已从 Gaussian proxy 切到真实 Chambolle-Pock ROF；**下一步**对齐论文的 ADMM(inner param 2) 实现。
2. 实现完整 Algorithm 1：FCM 初始化 + 阈值循环 + 清理算子 C(·)（准则 4.2/4.5）+ 收敛判据。
3. 接入 DRIVE retina（三相改造）与公开 brain MRI 等价数据，严格复刻 7 类合成图生成方式。
4. 实现/接入五个 baseline，按 Eq. 5.2 SA 与逐相 DICE 对齐 Table 5.1-5.4。
5. 全程保持诚实标签：未达数值复现前，结果一律标 partial/paper-like，禁止标 paper-level。

## 10. 风险与代理说明

- **真实 ROF 已上 headline，但仍是合成现象**：runner headline 路径（`rof2 = rof_chambolle_pock(image2,...)`，`sat2 = rof2 > 0.48`）现用**真实凸 ROF**，已不再用 Gaussian 代理；Gaussian smoothing（`gaussian_baseline_dice`）仅保留为对照，它**不是** ROF/TV 求解（无 TV 边缘保持与 staircasing），不能承载 Theorem 3.6 的精确结论。`run_k2_proposition_demo` 同样用真实 Chambolle-Pock，但仍是合成图上的**现象级**检查。
- **不能外推的结论**：
  1. 不能因 toy 上 `rof_threshold_dice=0.9983` 就声称"复现了 retina"——两者数值无对应，前者是合成两相 toy 的真实-ROF 阈值 Dice，后者是 Table 5.4 retina 逐相 DICE。
  2. 不能声称"代码证明了 Theorem 3.4/3.6/3.7"——定理是数学证明，代码只能给一致性佐证。
  3. K>2 的等价性在论文中本就有 ∂Σ_i∩∂Σ_{i+1}=∅ 的前提，toy 演示更不足以支撑多相等价结论。
  4. `pcms_like_energy=204` 仅是各向异性 perimeter/TV（`Σ|∂x sat2|+Σ|∂y sat2|`）的**代理量、不含数据保真项**，非论文任何报告数值，不可与论文能量横向比较。
- **参数敏感**：μ（及派生 λ）影响分割与"恢复"强度，合成图上调好的 μ 不能迁移到真实数据结论。

## 11. 参考：精读笔记

本流程的精读笔记见同仓库：
[`../../../xiaohao_cai_ultimate_notes/Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md`](../../../xiaohao_cai_ultimate_notes/Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md)

该笔记包含论文元信息、第一作者核验、五 Agent 辩论式分析（数学/工程/应用/质疑/综合）、Theorem 3.4/3.6 解读、Algorithm 1 流程、Table 5.4 retina 数值与复现判断小节。
