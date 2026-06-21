# 多类 ROF 阈值迭代分割 完整复现流程 (Complete Reproduction Workflow)

> 本文档为 15 篇口径内第 3 篇 Multiclass Segmentation by Iterated ROF Thresholding 的完整复现流程规范。

## 1. 论文身份与第一作者核验

| 字段 | 内容 |
| --- | --- |
| 英文标题 | Multiclass Segmentation by Iterated ROF Thresholding |
| 中文标题 | 多类 ROF 阈值迭代分割 |
| 作者顺序 | Xiaohao Cai, Gabriele Steidl |
| 第一作者核验 | 是。PDF 首页作者列表以 Xiaohao Cai 开头，第二作者 Gabriele Steidl；二人同属 University of Kaiserslautern, Department of Mathematics, Germany（邮箱 {cai,steidl}@mathematik.uni-kl.de）。 |
| 年份 | 2013 |
| 出处 | A. Heyden et al. (Eds.): EMMCVPR 2013, LNCS 8081, pp. 237-250, Springer-Verlag Berlin Heidelberg 2013 |
| PDF 路径 | docs/00_papers_first_author_xiaohao_cai_deduped/多类ROF分割 Iterated ROF.pdf |
| 主题 | sat-rof（Smoothing-and-Thresholding / ROF 阈值化分支） |

核验结论：本篇满足 15 篇口径内"Xiaohao Cai 为第一作者"的硬约束。它是 SaT/T-ROF 方法论中**多类阈值更新算法的原型论文**，处于第 1 篇 SaT Overview 与第 2 篇 PCMS-ROF Linkage 之后的"算法落地"位置。

## 2. 复现目标与诚实分级

本项目对复现真实性采用四级口径，必须如实标注，禁止夸大：

- **toy**：只用合成图、用 Gaussian smoothing 等代理替换真实变分求解，仅演示"先平滑后阈值"的思想。
- **partial**：使用真实凸 ROF 求解器（如 Chambolle-Pock 对偶投影 / Split-Bregman），实现论文核心阈值更新公式 (15)，并在合成图上验证若干引理性质，但**不用论文真实数据，不复刻 Table 1 / Fig. 1-6 的完整 baseline 对照**。
- **paper-like**：使用论文同源或公开等价数据（stripe 图、cartoon、brain MRI 等），实现完整 T-ROF Algorithm + 论文全部 baseline 对照，复现 SA 数量级与趋势。
- **paper-level**：在论文真实数据、真实参数、真实 baseline 实现下复现 Table 1 / Fig. 1-6 的具体数值。

| 维度 | 当前状态 |
| --- | --- |
| 本仓库复现等级 reproductionLevel | **partial** |
| 真实性等级 reproductionTruthLevel | **partial-completed** |

**纪律声明**：截至本文档，15 篇论文 paper-level 复现仍为 **0/15**。本篇当前为 partial，已用真实 ROF 求解器替换 Gaussian proxy 主路径，但**不得**把 partial 结果（合成 4-phase 图上的 0.9463 accuracy、K=2 Dice 0.9976 等）解读为论文级复现，也不得与论文 Table 1 / Fig. 中的 SA 数值直接等同比较。

## 3. 算法完整流程

论文的总思路：把"多类非凸 PCMS 直接优化"替换为"**解一次凸 ROF，再对同一个 ROF 解做 K-1 次阈值化，并迭代更新阈值**"。关键在于 ROF 极小元在整个阈值过程中保持不变（Proposition 3），因此只需一次 ROF 求解。

### 3.1 关键模型与公式（忠于 PDF）

1. **Mumford-Shah / PCMS 出发点**：E_MS(Γ,u) = H¹(Γ) + μ∫_{Ω\Γ}|∇u|²dx + λ∫_Ω(u-f)²dx（μ,λ>0）；限制 ∇u=0 得到 piecewise constant 模型 E_PCMS(Γ,u)=H¹(Γ)+λ∫_Ω(u-f)²dx（Eq. 1）。在 Ω=⋃Ω_i、u=m_i 的分块常数假设下写成 E_PCMS(Ω,m)=½Σ Per(Ω_i;Ω)+λΣ∫_{Ω_i}(m_i-f)²dx（Eq. 2）。

2. **K=2（与 Chan-Vese 接口）**：E_CV(Ω₁,m₀,m₁)=Per(Ω₁;Ω)+λ(∫_{Ω₁}(m₁-f)²dx+∫_{Ω\Ω₁}(m₀-f)²dx)（Eq. 3）。

3. **本文 K=2 能量（Eq. 7）**：E(Σ,τ) := Per(Σ;Ω) + μ∫_Σ (τ - f) dx，μ>0，τ∈(0,1)。注意 mean_f(A) := (1/|A|)∫_A f dx（|A|>0 时），否则为 0。

4. **K=2 阈值条件（Eq. 8）**：寻找 (Σ*,τ*) 使 E(Σ*,τ*) ≤ E(Σ,τ*) 对所有 Σ⊂Ω 成立，且 τ* = ½(mean_f(Σ*) + mean_f(Ω\Σ*))。

5. **凸松弛（Proposition 1, Eq. 10）**：固定 τ∈(0,1)，E(·,τ) 的全局极小集可由解凸问题 min_{u∈BV(Ω),0≤u≤1} TV(u) + μ∫_Ω (τ-f) u dx 得到，再令 Σ := {x∈Ω: u(x)>ρ}（任意 ρ∈[0,1)）。

6. **嵌套集引理（Lemma 1）**：固定 0<τ₁<τ₂<1，对应极小集满足 Σ₁ ⊇ Σ₂（up to negligible set）。

7. **与 Chan-Vese 的关系（Proposition 2）**：若 (Σ*,τ*) 是 (8) 的解、Σ*∉{∅,Ω}，记 m₀*=mean_f(Σ*)、m₁*=mean_f(Ω\Σ*)，则 (Σ*,m₀*,m₁*) 是带参数 **λ = μ / (2(m₁* - m₀*))** 的 Chan-Vese 模型 (3) 的 partial minimizer。直觉：当两类灰度差 m₁*-m₀* 很小时，λ 变大，数据项被加重——这正是 T-ROF 能分开"相近灰度类别"的机制来源。

8. **ROF 等价（Proposition 3, Eq. 11）**：{x: u(x)>τ} 解 (7) 当且仅当 u 解 ROF 模型 min_{u∈BV(Ω)} TV(u) + (μ/2)∫_Ω (u-f)² dx。这是"只解一次 ROF"的理论依据。

9. **多类推广（Eq. 12-15）**：Σ={Σ_i}_{i=1}^{K-1}，τ={τ_i}_{i=1}^{K-1}，0<τ₁≤τ₂≤…≤τ_{K-1}<1，
   E(Σ,τ)=Σ_{i=1}^{K-1}(Per(Σ_i;Ω) + μ∫_{Σ_i}(τ_i - f)dx)（Eq. 12）。
   由 Lemma 1 得嵌套 Ω⊇Σ_{τ₁}⊇…⊇Σ_{τ_{K-1}}⊇∅（Eq. 13）；
   wanted segments Ω_i := Σ_i\Σ_{i+1}，i=0,…,K-1，Σ₀:=Ω，Σ_K:=∅（Eq. 14）；
   目标阈值 **τ_i* = ½(m_{i-1}* + m_i*)，m_i* := mean_f(Ω_i*)**，i=1,…,K-1（Eq. 15）。

### 3.2 Algorithm (T-ROF) —— 可执行 step-by-step（忠于 PDF Section 3）

```
Initialization: τ^(0) = (τ_i^(0))_{i=1}^{K-1}, 0 < τ_1^(0) < ... < τ_{K-1}^(0) < 1.
                论文用 fuzzy C-means [7]（100 步）初始化 τ^(0)。

1. Compute the solution u of the ROF model (Eq. 11).   # 只解一次，全程复用
2. For k = 0, 1, 2, ... repeat:
   2.1 Σ^(k) = (Σ_i^(k))_{i=1}^{K-1},  Σ_i^(k) := { x ∈ Ω : u(x) > τ_i^(k) }.
   2.2 Ω_i^(k) := Σ_i^(k) \ Σ_{i+1}^(k),  i=0,...,K-1,  with Σ_0^(k):=Ω, Σ_K^(k):=∅.
   2.3 m_i^(k) := mean_f(Ω_i^(k)),  i=0,...,K-1.   # 注意：在 raw image f 上取均值
   2.4 Update τ_i^(k+1) := ½(m_{i-1}^(k) + m_i^(k)),  i=1,...,K-1.
   直到 ||u^(i)-u^(i-1)||_2/||u^(i)||_2 ≤ ε_u 且 ||τ^(k)-τ^(k-1)||_2 ≤ ε_τ。
```

论文实现细节（Section 4 首段）：ROF 用 [13] 的离散 ROF；论文实际数值实验用 **ADMM**（fixed inner parameter 2）求 ROF 极小元；停止阈值 **ε_u = 10⁻⁴，ε_τ = 10⁻⁵**。

### 3.3 收敛证明结构（理论直觉）

- **Assumption (A)**：若 Σ_τ, Σ_τ̄ 是 E(·,τ)、E(·,τ̄) 的极小集（0<τ<τ̄<1），则 τ ≤ mean_f(Σ_τ\Σ_τ̄) ≤ τ̄（Eq. 16）。这是一种"分割层之间均值有序"的合理性假设，并非对所有图像任意 K 都自动成立。
- **Lemma 2（单调性）**：在 (A) 下，T-ROF 产生的 (τ^(k)) 与 (m^(k)) 满足 0≤m₀^(k)≤τ₁^(k)≤m₁^(k)≤…≤τ_{K-1}^(k)≤m_{K-1}^(k)（i 部分）；并给出阈值升降与均值升降之间的传递关系（ii 部分）。
- **Lemma 3（sign changes 单调不增）**：定义符号序列 ζ^(k)（τ_i 相对上一步升/降取 ±1），s_k 为 ζ^(k) 中的 sign change 数；则 s_k 关于 k 单调不增，且若 ζ₁^(k+1)≠ζ₁^(k) 则严格下降 s_{k+1}<s_k。
- **Theorem 1（收敛）**：把 [0,1) 划分为 n 个子区间并定义投影算子 P_n（"projected T-ROF"），对 s_k 做归纳，证明投影后的阈值序列 (τ^(k)) 收敛到 τ*。

**关键纪律点**：projection 只是收敛证明中的 *slight modification*，不是数值 Algorithm T-ROF 的核心步骤；收敛也**不是**对任意图像、任意 K 的全局保证（需 (A) 成立）。

## 4. 完整复现所需数据集

论文 Section 4 共用 6 个 Example（数据规模均来自 PDF）：

| Example | 数据 | 规模 | 类别数 K | 备注 |
| --- | --- | --- | --- | --- |
| 1 | 两类 cartoon 图（含 missing pixel） | 256×256 | 2 | 需 codebook 更新；只有 [12][20] 与本文给出好结果 |
| 2 | 两类 close-intensity 图 | 128×128 | 2 | 常图灰度 0.5 加 Gaussian noise(var 1e-8)，白部保留、黑部乘 2×10⁻⁴ |
| 3 | brain MRI gray/white matter | 319×256 | 4 | 四相脑 MRI，来自 [25]；T-ROF 11 次 τ 更新仍较快 |
| 4 | stripe image（30 stripes） | 140×240 | 5/10/15 | Gaussian noise var 1e-3；**对应 Table 1 的 SA/time** |
| 5 | 三类 close-intensity 图 | 256×256 | 3 | Gaussian noise var 1e-2，黑/白部标量 0.1/0.6 |
| 6 | 四类 close-gray-value 图 | 256×256 | 4 | Gaussian noise var 3×10⁻² |

**为达 paper-like 的公开/等价候选数据来源**：

- stripe / cartoon / 多相合成图：可按论文描述用脚本程序化重建（30 条灰度条纹、close-gray 常值块 + 指定 Gaussian noise），属于"等价合成数据"，可逼近 Table 1 / Fig. 5-6 的设定。
- brain MRI：论文取自参考 [25] 的脑 MRI；公开等价可用 **BrainWeb (McGill)** 模拟脑 MRI 体数据或 **IBSR** 切片（注意它们与论文原图不完全相同，只能做 paper-like，不能宣称复现 Fig. 3 的具体像素）。
- baseline 方法 [22][25][29][20][12] 的原始实现多需各自作者代码或重写；若无法获得，应在文档中标注"baseline 用复现版/近似版"，避免误导。

私有/受限提示：论文未公布数据下载链接，brain MRI 来源属第三方医学影像。若不能取得与论文一致的原图，**任何 SA 数值都只能定性对照，禁止与 Table 1 / Fig. 数字逐一对齐宣称复现**。

## 5. 对照基线 (Baselines)

论文在 Section 4 / Table 1 / Fig. 1-6 中与以下方法对照（均为 PDF 中明确出现的引用）：

- **Cai [12]**：Cai, Chan, Zeng (2013) two-stage segmentation（SaT 的直接前身，one thresholding）。这是最关键的对照——本文宣称 *outperforms* [12]，因为本文做 *iterative* 阈值更新。
- **Li [22]**：Li, Ng, Zeng, Shen (2010) fuzzy region competition 多相分割。
- **Pock [25]**：Pock, Chambolle, Cremers, Bischof (2009) convex relaxation minimal partition。
- **Yuan [29]**：Yuan, Bae, Tai, Boykov (2010) continuous max-flow Potts。
- **He [20]**：He, Hussaini, Ma, Shafei, Steidl (2012) fuzzy c-means + TV。

合理的额外对照（非论文必需，可在 partial→paper-like 中加入）：raw K-means / fuzzy C-means（无 TV）、直接阈值化（不平滑）、Chan-Vese level set 原版。本仓库当前用 raw K-means 与 Gaussian proxy T-ROF 作为轻量对照。

## 6. 评价指标与论文报告结果

**指标定义（PDF, Section 4）**：Segmentation Accuracy
> SA = #correctly classified pixels / #all pixels。

论文用 SA 来选择各方法的正则参数 μ（"by judging the SA"）。

**论文报告的关键数值（均直接来自 PDF，注明出处）**：

| 来源 | 设定 | T-ROF SA | 同图最强对照 | 备注 |
| --- | --- | --- | --- | --- |
| Fig. 1 | 2-class cartoon, missing pixels, 256×256 | 0.9913 (Ite. 6) | He 0.9888 / Cai 0.9878 | T-ROF 最高 |
| Fig. 2 | 2-class close-intensity, 128×128 | 0.9845 | Cai 0.9816 / He 0.9663 | T-ROF 最高 |
| Fig. 5 | 3-class close-intensity, 256×256 | 0.9550 | He 0.9637 / Yuan 0.9557 | He 略高于 T-ROF |
| Fig. 6 | 4-class close-gray-value, 256×256 | 0.9798 | Cai 0.9688 | T-ROF 最高 |

**Table 1（Example 4，stripe 图 140×240，参数 μ / Ite. / Time(s) / SA）——直接引用 PDF Table 1**：

| Phases | 方法 | μ | Ite. | Time(s) | SA |
| --- | --- | --- | --- | --- | --- |
| Five | Li[22] | 80 | 100 | 3.87 | 0.9946 |
| Five | Pock[25] | 100 | 100 | 6.25 | 0.9965 |
| Five | Yuan[29] | 10 | 87 | 4.33 | 0.9867 |
| Five | He[20] | 50 | 100 | 16.75 | 0.9968 |
| Five | Cai[12] | 10 | 41 | 1.33 | 0.9770 |
| Five | **T-ROF** | 8 | 84 (4) | 1.39 | **0.9986** |
| Ten | T-ROF | 8 | 84 (5) | 2.33 | **0.9967** |
| Ten | Cai[12] | 10 | 41 | 2.11 | 0.8900 |
| Ten | Li[22] | 80 | 100 | 7.71 | 0.8545 |
| Fifteen | T-ROF | 8 | 84 (5) | 3.74 | **0.9933** |
| Fifteen | Cai[12] | 10 | 41 | 3.06 | 0.5280 |
| Fifteen | Li[22] | 80 | 100 | 11.56 | 0.7715 |

（Table 1 中 "84 (4)" 表示 ROF 内迭代步数 84、外层 τ 更新 4 次。完整三块表见 PDF p.247。要点：相位数增大时，Cai[12] 的 SA 从 0.977 跌到 0.528，而 T-ROF 仍维持 0.99 量级，凸显 *iterative threshold update* 的价值。）

**禁止编造**：以上为 PDF 可确认的数值；任何未在表/图中出现的 SA/时间，文档与笔记一律以定性描述呈现，不补造数字。

## 7. 本仓库当前复现实现

- **runnerFile**：reproduce/experiments/sat_rof_trof.py（同一文件同时服务第 1/2/3 篇；id=iterated-rof 对应其中 `completed(3, "iterated-rof", ...)` 分支）。
- **实际做了什么**：
  1. `generate_close_gray_multiphase()` 生成 96×96、4 相、灰度 [0.28,0.32,0.36,0.40]（相邻差 0.04）的 close-gray 合成图 + Gaussian noise，对应论文"相近灰度"难点。
  2. `rof_chambolle_pock()`：用 **Chambolle-Pock 原始-对偶投影**解离散 ROF（μ=8.0，240 步，tol 2e-5）——这是 partial 的主求解路径，替换了原来的 Gaussian proxy。
  3. `rof_split_bregman()`：**Split-Bregman** ROF 求解器作为 solver 交叉校验（非论文 baseline）。
  4. `run_trof_thresholds()`：实现 Eq. (15) 阈值迭代 τ_i=½(m_{i-1}+m_i)，**均值在 raw image f 上计算 mean_f(Ω_i)**（忠于 Eq. 14-15 的 wanted segment 均值定义），并记录 Lemma 2 单调性、Lemma 3 sign changes、drift。
  5. `run_k2_proposition_demo()`：K=2 合成圆盘图，验证 Proposition 2 的 λ = μ/(2(m₁-m₀)) 关系，并与"在 raw f 上同阈值"得到的 Chan-Vese proxy 对比 Dice。
- **代理 (proxy) 用法**：Gaussian smoothing 保留为 *轻量对照 baseline*（`gaussian_trof`），不再是主路径；Chan-Vese 用"raw f 上同阈值"做 proxy，**不是**真实 Chan-Vese level-set 求解。
- **当前 runMetrics（取自 reproStructured，合成图，非论文数据）**：

| 指标 | 数值 |
| --- | --- |
| raw_kmeans_accuracy | 0.5650 |
| gaussian_proxy_trof_accuracy | 0.9438 |
| rof_trof_accuracy | 0.9463 |
| split_bregman_trof_accuracy | 0.9510 |
| threshold_iterations | 3 |
| max_threshold_drift | 0 |
| monotonicity_violated | false |
| sign_changes_final | 0 |
| sign_changes_nonincreasing | true |
| assumption_a_violations | 0 |
| rof_iterations_chambolle_pock | 240 |
| rof_iterations_split_bregman | 70 |
| k2_lambda_derived | 7.7416 |
| k2_rof_threshold_dice | 0.9976 |
| k2_chanvese_proxy_dice | 0.8994 |
| k2_segmentation_disagreement | 0.0476 |
| runtime_seconds_total | 0.7057 |

- **resultFiles（图）**：assets/repro/sat_demo.png、assets/repro/trof_thresholds.png、assets/repro/iterated_rof_convergence.png、assets/repro/iterated_rof_chanvese.png。

## 8. 差距分析（到 paper-like / paper-level 还缺什么）

到 **paper-like** 还缺：

- **数据**：缺 stripe 30-条图（5/10/15 相）、2-class cartoon(含 missing pixel)、3-class、4-class 的论文规模合成图（256×256 / 140×240）。当前只有 96×96 单张 4-phase。
- **初始化**：论文用 fuzzy C-means(100 步) 初始化 τ^(0)；当前用 quantile / 手设阈值，应替换为 FCM 初始化以贴近论文。
- **ROF 求解器**：论文用 ADMM(inner param 2)；当前用 Chambolle-Pock / Split-Bregman。求解器不同会影响 ROF 解与最终阈值，需对齐或至少标注差异。
- **停止准则**：论文 ε_u=1e-4、ε_τ=1e-5；当前 drift tol=1e-4，需对齐 u 与 τ 的双停止准则。
- **指标**：当前 4-phase 用 clustering_accuracy（含标签对齐），应统一为论文 SA 定义并报告每类相位的 SA。

到 **paper-level** 还缺（在 paper-like 之上）：

- **真实 baseline 实现**：Li[22]/Pock[25]/Yuan[29]/He[20]/Cai[12] 五个对照的原版或忠实复现，且各自按 SA 调 μ（论文给了每个方法的 μ）。
- **Table 1 / Fig. 1-6 对照**：复刻 Table 1 的 μ / Ite. / Time / SA 三块表，以及 6 个 Example 的 SA 数值与可视化。
- **真实 brain MRI**：Example 3 的脑 MRI 原图（来自 [25]）；公开等价数据只能近似。
- **Theorem 1 数值层面**：当前只检查 Lemma 2/3 的性质，未在数值上完整复刻 projected-T-ROF 的收敛定理证明链。

## 9. 运行步骤

**当前 toy/partial 跑法**：

```bash
# 安装依赖（见 reproStructured.dependencies）
pip install -r requirements.txt   # numpy, scipy, matplotlib（本实验最小集）

# 运行全部复现（含本篇 id=iterated-rof）
cd reproduce && python run_all.py
```

依赖：numpy / scipy / matplotlib。算力：CPU，约 1 秒内（Chambolle-Pock 240 步 + Split-Bregman 70 步 + T-ROF ≤20 步）。缺依赖时 runner 写入 skipped，不伪造 completed。

**向 paper-like 扩展的步骤大纲（规范，不在本任务内执行）**：

1. 程序化重建论文数据：stripe(30 条, 140×240, var 1e-3)、2/3/4-class 与 cartoon(256×256)。
2. 接入 fuzzy C-means(100 步) 做 τ^(0) 初始化。
3. 把 ROF 求解器切到 ADMM(inner param 2)，并加 ε_u=1e-4、ε_τ=1e-5 双停止准则。
4. 统一 SA 指标，输出每个 Example 的 SA 与 Table 1 风格 μ/Ite./Time/SA 表。
5. 实现/接入 5 个 baseline，按各自 μ 复现对照；与 PDF 数值并列展示但**明确标注是否同源数据**。

## 10. 风险与代理说明

- **proxy 局限**：Gaussian smoothing 只是 TV/ROF 的极粗代理，不满足 TV 的边缘保持性质；它在本仓库已降级为对照 baseline，不可当作 ROF。Chan-Vese proxy（raw f 上同阈值）不解 level-set 能量，只用于 Proposition 2 的 λ 直觉演示。
- **数据局限**：96×96 单张 close-gray 合成图无法代表论文 6 个 Example 的难度分布；其 accuracy(0.9463) / Dice(0.9976) **不可外推**为论文 SA。
- **理论局限**：Lemma 2/3 的数值检查（monotonicity_violated=false、sign_changes 不增）只是性质自洽性验证，**不等于** Theorem 1 的完整证明复现，也未验证 Assumption (A) 在真实图上的成立性。
- **不可外推的结论清单**：① 不得宣称复现 Table 1 / Fig. 1-6 任一 SA；② 不得宣称在论文真实 brain MRI / cartoon 上验证；③ 不得宣称 baseline 对照成立（当前无 Li/Pock/Yuan/He/Cai 原版实现）；④ paper-level 仍为 0/15。

## 11. 参考：精读笔记

详见同篇精读笔记：[../../../xiaohao_cai_ultimate_notes/Multiclass_Segmentation_Iterated_ROF_超精读笔记_已填充.md](../../../xiaohao_cai_ultimate_notes/Multiclass_Segmentation_Iterated_ROF_超精读笔记_已填充.md)
