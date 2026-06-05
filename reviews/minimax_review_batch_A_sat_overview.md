# Minimax Review Batch A — Paper #1 sat-overview

> 审查日期：2026-06-05
> 审查员：minimax (Mavis)
> 审查对象：Xiaohao Cai, Raymond Chan, Tieyong Zeng. "An Overview of SaT Segmentation Methodology and Its Applications in Image Processing." Chapter 40, pp. 1385–1409, in *Handbook of Mathematical Models and Algorithms in Computer Vision and Imaging*, Springer 2023.
> 严格按 `reviews/minimax_paper_review_orchestration.md` 的三轮 + 输出 schema。

---

## Executive Result

- must-fix: 3
- should-fix: 2
- note: 4
- batch verdict: **fail** (paper #1 静态声明和实际跑结果不一致)
- post-remediation status: **resolved in Codex pass on 2026-06-05**. M1/M2/S1/S2/N1 have been patched in the SaT note; M3 has been resolved by rerunning all 15 reproduction toys to `completed`, updating dashboard metrics/assets, and adding `reproduce/sync_to_dashboard.mjs` plus `docs/scripts/validate.mjs` repro-sync validation.

---

## Must Fix

### M1. 笔记"定理1"中 Ker 条件符号错误（∩ 应为 ⊇）
- severity: must-fix
- target: #1 sat-overview
- file: `xiaohao_cai_ultimate_notes/分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md` 行 93
- claim: "且 $\text{Ker}(A) \cap \text{Ker}(\nabla) = \{0\}$，则上述问题在 $W^{1,2}(\Omega)$ 中存在唯一最小解。"
- evidence: PDF p.5（book 1389）Theorem 1 原文："Let Ω be a bounded connected open subset of R² with a Lipschitz boundary. Let f ∈ L²(Ω) and **Ker(A) ⊇ Ker(∇) = {0}**, where A is a bounded linear operator from L²(Ω) to itself and Ker(A) is the kernel of A. Then (8) has a unique minimizer g ∈ W^{1,2}(Ω)."
- judgment: **错误**。PDF 用的是 ⊇（包含），笔记写成了 ∩（交集），两者数学上不等价。⊇ Ker(∇) = {0} 等价于 Ker(∇) = {0}，是一个比 ∩ = {0} 更严格的条件，因为 Ker(A) 还要包含整个 Ker(∇)。
- fix: 把行 93 的 "$\text{Ker}(A) \cap \text{Ker}(\nabla) = \{0\}$" 改成 "$\text{Ker}(A) \supseteq \text{Ker}(\nabla) = \{0\}$"。
- confidence: high

### M2. 笔记"定理2"遗漏 partial minimizer 成立的关键条件
- severity: must-fix
- target: #1 sat-overview
- file: `xiaohao_cai_ultimate_notes/分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md` 行 97
- claim: "则 $\tilde{\Sigma}$ 是PCMS模型对于 $\lambda := \frac{\mu}{2(m_1-m_0)}$ 和固定 $m_0, m_1$ 的最小化子。"
- evidence: PDF p.8（book 1392）Theorem 2 完整陈述末尾："**In particular, (˜Σ, m0, m1) is a partial minimizer of (4) if m0 = meanf(Ω\˜Σ) and m1 = meanf(˜Σ).**"
- judgment: **部分正确 / 不完整**。笔记把 Theorem 2 的前半段（"Σ 是 minimizer"）抄了，但漏了"partial minimizer"成立的条件。这会误导读者以为 Σ 直接就是 minimizer 而不是 partial minimizer。这是 SaT 综述中关于 ROF/PCMS 关系的核心定理，partial minimizer 条件不能省。
- fix: 在 "$\tilde{\Sigma}$ 是PCMS模型对于 $\lambda := \frac{\mu}{2(m_1-m_0)}$ 和固定 $m_0, m_1$ 的最小化子" 后追加 "。特别地，若 $m_0 = \text{mean}_f(\Omega \setminus \tilde{\Sigma})$ 且 $m_1 = \text{mean}_f(\tilde{\Sigma})$，则 $(\tilde{\Sigma}, m_0, m_1)$ 进一步构成 PCMS 模型 (4) 的 partial minimizer。"
- confidence: high

### M3. reproDetails[1] 静态声明与实际跑结果不一致（resultStatus="completed" 但实际 skipped）
- severity: must-fix
- target: #1 sat-overview
- file: `docs/js/reading-data.js` 行 1055–1062
- claim: `resultStatus: "completed"`, `runtimeSeconds: 0.5428`, `runMetrics: { direct_accuracy: 0.659, sat_accuracy: 0.9799, accuracy_gain: 0.321 }`, `resultFiles: ["assets/repro/sat_demo.png"]`
- evidence:
  - 实际跑（`python3 reproduce/run_all.py`） 写入 `reproduce/results/repro_results.json` 和 `docs/assets/repro/repro_results.json` 的 sat-overview 项：`{ "status": "skipped", "runtime_seconds": 0.0, "metrics": {}, "skipped_reason": "Missing modules: numpy, matplotlib, scipy" }`
  - `run_all.py` 的设计意图（看 `experiments/common.py`）：依赖缺失时统一写 skipped，不伪造结果
  - 但是 `run_all.py` **不重写 `docs/js/reading-data.js` 的 `reproDetails` 段**，dashboard 静态声明是硬编码的"完成态"+ 假 metrics
- judgment: **错误**。当前环境下 15 项复现评估**全部**实际跑出来是 skipped（缺 numpy/scipy/matplotlib/scikit-image），但 `reading-data.js` 静态声明"completed"+ 假 metrics 0.9799。这违反"不伪造结果"的项目基线，也违反 validate.mjs 的核心约束（虽然 validate.mjs 没主动 cross-check 这一字段，但本质上是文档级不一致）。
- fix 方案（两条择一）：
  - 方案 A（推荐）：`pip install -r requirements.txt` 装好依赖 → 重跑 `python3 reproduce/run_all.py` → 然后**手动**修改 `docs/js/reading-data.js` 的 reproDetails[1]（以及 2, 3, …, 15），把 resultStatus 改成实际跑出来的状态，runMetrics 改成实际数字。这一步需要**单独的工作流**（见 Suggest Patch Plan 中的 R-workflow 改造建议）。
  - 方案 B（临时）：如果暂时不装依赖，把 `reproDetails[1]` 改成 `resultStatus: "skipped"`, `skipped_reason: "Missing modules: numpy, matplotlib, scipy"`, `runtimeSeconds: 0`, `runMetrics: {}`, `resultFiles: []`。但这会同时影响 dashboard 上 15 张精读卡片的复现状态展示 — 不推荐。
- confidence: high

---

## Should Fix

### S1. 笔记"3.6 SaT 变体方法 / 高光谱图像分类"漏写约束条件
- severity: should-fix
- target: #1 sat-overview
- file: 笔记 行 167–170
- claim: "第一阶段：SVM分类器生成概率图；第二阶段：SaT模型优化概率图" + 公式 `inf_{g_k} { ... }`（无约束）
- evidence: PDF p.15（book 1399）model (14) 完整陈述是 `inf_{g_k} { ... } s.t. g_k|_{Ω_train} = f_k|_{Ω_train}`，约束是**训练像素上 g_k 必须等于 f_k**。
- judgment: **公式不完整**。约束条件是 hyperspectral 分类能复现的关键 — 没有这个约束，模型退化为普通 SaT 恢复，分类精度会显著下降。
- fix: 在公式后追加 "约束条件：$g_k|_{\Omega_{train}} = f_k|_{\Omega_{train}}$（训练像素上 g_k 必须等于 SVM 给出的概率 f_k）"
- confidence: high

### S2. 笔记"3.4 Split-Bregman 算法求解"小节不在 PDF 原文
- severity: should-fix
- target: #1 sat-overview
- file: 笔记 行 103–118
- claim: 包含完整的 Split-Bregman 迭代格式：g-子问题 FFT 求解、d-子问题 shrink、Bregman 更新
- evidence: PDF p.5（book 1389）原文只说："model (8) can be minimized quickly by using currently available efﬁcient algorithms such as the split-Bregman algorithm (Goldstein and Osher 2009) or the Chambolle-Pock method (Chambolle and Pock 2011)"。没有给出具体迭代公式。
- judgment: 这是**助教/笔记作者补充的标准算法内容**，不在 SaT 论文原文中。如果不注明来源，会被误以为属于论文核心算法。这不算错（标准算法本身正确），但混在"论文精读"里会模糊论文实际贡献。
- fix: 在该小节开头加一行注："本节内容来自标准 Split-Bregman 教材 (Goldstein and Osher 2009)，不属于 SaT 论文原文。SaT 论文本身只把 split-Bregman / Chambolle-Pock 列为可用的求解器。"
- confidence: high

---

## Notes

### N1. 笔记"3.6 SaT 变体方法 / Tight-Frame算法"通用形式公式未在 PDF 找到出处
- target: #1 sat-overview
- file: 笔记 行 142–145
- claim: "$f^{(i+1/2)} = \mathcal{U}(f^{(i)})$；$f^{(i+1)} = A^T \mathcal{T}_\lambda(A f^{(i+1/2)})$"
- judgment: **未证实**。本批未深读 PDF 的 Tight-Frame 段（PDF p.16–17，book 1400–1401），通用形式的具体公式在 SaT Overview 中可能没展开（指向 Cai et al. 2011, 2013a 的 tight-frame 原文）。建议在 Batch B (`framelet-tubular` 和 `tight-frame-vessel`) 中核对。笔记先标 "未证实"。
- confidence: medium

### N2. 笔记"3.5 不同噪声模型的保真项"公式与 PDF 完全一致（keep）
- target: #1 sat-overview
- file: 笔记 行 122–134
- claim: Poisson `∫(g − f log g)dx + β ∫|∇g|dx`；Gamma（w = log g 变换后）`∫(f e^{-w} + w)dx + β ∫|∇w|dx`
- evidence: PDF p.10（book 1394）equations (10) 和 (12) 文字完全一致。
- judgment: ✅ 与 PDF 一致。**保留**。

### N3. 笔记"3.6 SLaT 方法"6 维 K-means 描述与 PDF 一致（keep）
- target: #1 sat-overview
- file: 笔记 行 156–162
- claim: smoothing → Lab lifting → 6 维 $(g_1, g_2, g_3, \bar{g}_1, \bar{g}_2, \bar{g}_3)$ K-means
- evidence: PDF p.13（book 1397）原文："K-means to threshold the lifted image with 6 channels (g1, g2, g3, ¯g1, ¯g2, ¯g3)"。
- judgment: ✅ 与 PDF 一致。**保留**。

### N4. effectScore=4 配合 toy accuracy_gain=0.321 的"很明显"标签略过强
- target: #1 sat-overview
- file: `docs/js/reading-data.js` 行 1044–1045
- claim: `effectScore: 4`, `effectLabel: "很明显"`, `accuracy_gain: 0.321`
- judgment: toy synthetic 4-phase 上 0.659 → 0.9799 提升确实很大（+0.32），但 effectLabel="很明显"对 toy result 略过强。toy 数据集本身不具代表性，4 类灰度差较大是设计上让 SaT 容易赢的设定。**这是 note**，不强制降级，但用户回填时可以考虑降到 3（"明显"）。
- confidence: low

---

## Per Paper Table

| # | id | metadata | content | reproduction | verdict |
|---|---|---|---|---|---|
| 1 | sat-overview | ✅ pass | ❌ 2 处 must-fix (M1, M2) + 2 处 should-fix (S1, S2) + 1 处未证实 (N1) | ❌ must-fix (M3) | fail |

**metadata 核验**：
- PDF 标题 "An Overview of SaT Segmentation Methodology and Its Applications in Image Processing" — 与 `papers[0].title` 和 `paperNotesV2[0].titleEn` 一致 ✅
- 作者 "Xiaohao Cai, Raymond Chan, and Tieyong Zeng" — 与 `papers[0].authors` ("Xiaohao Cai, Raymond Chan, Tieyong Zeng") 一致 ✅
- 章节 Chapter 40, pp. 1385–1409 — 与 `paperNotesV2[0].titleCn` 注释 / papers[0].pages=27 推导一致（1409-1385+1=25 实际页 + 章节首页 + 参考文献尾页 ≈ 27） ✅
- 年份 2023 — `paperNotesV2[0].year: 2023` + PDF metadata `creationDate: 2023-01-30` 一致 ✅
- Springer Handbook of Mathematical Models and Algorithms in Computer Vision and Imaging — 与 `paperNotesV2[0].titleEn` 注释 / PDF 版权页一致 ✅
- PDF 文件名 `分割方法论总览 SaT Overview.pdf` — 实际存在于 `docs/00_papers_first_author_xiaohao_cai_deduped/` ✅
- 第一作者核验通过：PDF 首页作者行 "Xiaohao Cai, Raymond Chan, and Tieyong Zeng" ✅

---

## Suggested Patch Plan

| priority | project file | field/section | action | replacement summary |
|---|---|---|---|---|
| high | `xiaohao_cai_ultimate_notes/分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md` 行 93 | 笔记"3.2 定理1" | edit | 改 "$\text{Ker}(A) \cap \text{Ker}(\nabla) = \{0\}$" → "$\text{Ker}(A) \supseteq \text{Ker}(\nabla) = \{0\}$" |
| high | 同上 行 97 | 笔记"3.3 定理2" | edit | 在 "的最小化子" 后追加 "。特别地，若 $m_0 = \text{mean}_f(\Omega \setminus \tilde{\Sigma})$ 且 $m_1 = \text{mean}_f(\tilde{\Sigma})$，则 $(\tilde{\Sigma}, m_0, m_1)$ 进一步构成 PCMS 模型 (4) 的 partial minimizer" |
| high | `docs/js/reading-data.js` 行 1055–1062 | reproDetails[1] | work | 选 M3 fix 方案 A：装依赖 + 重跑 reproduce + 手动把 runMetrics / resultStatus 改成实际跑结果；或方案 B：临时改成 skipped |
| medium | `xiaohao_cai_ultimate_notes/分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md` 行 167–170 | 笔记"3.6 SaT 变体方法 / 高光谱图像分类" | edit | 在 inf_{g_k} 公式后追加约束 "s.t. $g_k\|_{\Omega_{train}} = f_k\|_{\Omega_{train}}$" |
| medium | 同上 行 103–118 | 笔记"3.4 Split-Bregman算法求解" | edit | 在小节开头加一行注："本节内容来自标准 Split-Bregman 教材 (Goldstein and Osher 2009)，不属于 SaT 论文原文" |
| low | `xiaohao_cai_ultimate_notes/分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md` 行 142–145 | 笔记"3.6 SaT 变体方法 / Tight-Frame算法" | note | 标 "未证实 — 待 Batch B 核对"，暂不动 |
| low | `docs/js/reading-data.js` 行 1044–1045 | reproDetails[1] effectScore / effectLabel | optional | 考虑把 effectScore 从 4 降到 3，把 effectLabel 从"很明显"改成"明显"；视 M3 修复方向决定 |

**R-workflow 改造建议（M3 衍生）**：
- `reproduce/run_all.py` 只写 `reproduce/results/` 和 `docs/assets/repro/repro_results.json`，**不重写 `docs/js/reading-data.js` 的 reproDetails 段**。这是历史遗留 — reproDetails 是手工维护的快照。
- 建议增加一个 `reproduce/sync_to_dashboard.py` 脚本（或者扩展 run_all.py）：从 `reproduce/results/repro_results.json` 读实际结果，**生成 patch diff** 给 `docs/js/reading-data.js` 的 reproDetails 段（不能直接重写，因为还有 difficultyScore / effectScore / fidelityWarning 等人工评注）。这样手工维护和实际跑结果就有了明确的 reconcile 流程。
- 这一条超出本批 paper 范围，记为工作流改造提案，下一批（#2 pcms-rof-linkage）报告里可以继续追。

---

## Claims To Keep

以下项目表述经本批核验确认准确，无需修改：

1. **元信息**：PDF 标题、作者顺序（Xiaohao Cai 第一）、年份 2023、Springer Handbook Chapter 40、pp. 1385–1409 — 全部与项目文件一致
2. **`paperNotesV2[0]` 字段**：
   - `coreProblem`（PCMS / Chan-Vese 非凸）— 与 PDF Introduction 一致
   - `methodHandle`（SaT = smoothing + thresholding）— 与 PDF SaT Methodology 一致
   - `keyModelOrFormula`（K 在 thresholding，不在 smoothing）— 与 PDF 原文 "the thresholding step is independent of the smoothing step" 一致
   - `algorithmFlow`（5 步）— 与 PDF SaT Methodology 段对应
   - `relation.links: [2, 3, 7, 8]`（pcms-rof-linkage / iterated-rof / slat-color / sphere-wavelet）— 全部对应 SaT 综述里点名的下游分支论文
   - `experimentFocus`（synthetic retina / degraded color / vascular / spherical / hyperspectral）— 与 PDF 目录一致
3. **`noteEnhancements["sat-overview"]`**：
   - `evidence` 字段（Abstract / Introduction / SaT Methodology / Theorem 1 / T-ROF / SLaT / vascular / sphere sections）— 全部能在 PDF 找到对应位置
   - `reportExpansion` 5 个子段（context / technicalReading / theoremReading / experimentReading / relationReading / researchValue）— 准确标出"综述是地图，证明在 Linkage 和 Multiclass T-ROF"
4. **笔记中以下三段与 PDF 完全一致**：
   - 3.5 Poisson / Gamma 保真项公式（行 122–134）
   - 3.6 SLaT 6 维 K-means 描述（行 156–162）
   - 3.1 SaT 核心模型公式（行 84）— 与 PDF equation (8) 一致；A 是 "blurring operator or identity" 笔记（行 89）的"退化算子"措辞合理
5. **`reproDetails[1]` 的 `fidelityWarning` 字段** — "Uses Gaussian proxy smoothing, not an exact convex ROF/TV minimizer." 这是项目自陈，**准确且必要**（toy 用 Gaussian 平滑代替严格 ROF/TV minimizer 是 partial repro 的合理做法）。validate.mjs 也会检查这条 warning 是否存在（行 152–154）。

---

*本批 paper #1 审查结束。等用户决定是否继续 paper #2 pcms-rof-linkage。*
