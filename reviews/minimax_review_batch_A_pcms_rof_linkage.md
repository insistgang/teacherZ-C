# Minimax Review Batch A — Paper #2 pcms-rof-linkage

> 审查日期：2026-06-05
> 审查对象：Xiaohao Cai, Raymond Chan, Carola-Bibiane Schoenlieb, Gabriele Steidl, Tieyong Zeng. "Linkage Between Piecewise Constant Mumford-Shah Model and ROF Model and Its Virtue in Image Segmentation." arXiv:1807.10194v2, 2019.
> 严格按 `reviews/minimax_paper_review_orchestration.md` 的三轮 + 输出 schema。

---

## Executive Result

- must-fix: 2
- should-fix: 4
- note: 3
- batch verdict: **pass-with-fixes**
- post-remediation status: **resolved in Codex pass on 2026-06-05**. The note now separates Theorem 3.4 from Theorem 3.6, fixes the λ factor, moves ROF solving outside the threshold loop, softens complexity/global-optimality/application claims, and the dashboard title/experiment wording now matches the PDF more closely.

---

## Must Fix

### M1. 笔记把核心定理编号/参数写错
- severity: must-fix
- target: #2 pcms-rof-linkage
- file: `xiaohao_cai_ultimate_notes/Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md`
- claim: "核心定理 (Theorem 3.4): PCMS与ROF的联系"；"参数 λ = 2μ/(m*_1 - m*_0)"。
- evidence: PDF p.7 Theorem 3.4 是 "Relation between T-ROF and PCMS models for K=2"，参数为 `λ := μ / (2(m*_1-m*_0))`。PDF p.8 Theorem 3.6 才是 "Relation between ROF and PCMS models for K=2"，并同样使用 `λ := μ / (2(m1-m0))`。
- judgment: **错误**。定理 3.4 和 3.6 被混写，且 λ 比例差了 4 倍。
- fix: 把 Theorem 3.4 写成 T-ROF-PCMS 关系；另列 Theorem 3.6 为 ROF-PCMS 关系；所有 λ 改为 `μ/[2(m1-m0)]`。
- confidence: high

### M2. 笔记的 T-ROF 算法流程错误：ROF 只解一次，不在阈值循环内反复求解
- severity: must-fix
- target: #2 pcms-rof-linkage
- file: `xiaohao_cai_ultimate_notes/Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md`
- claim: 主循环每轮 Step 1 "求解ROF模型"，伪代码也把 `u* = argmin...` 放在 MAIN LOOP 内，并写 "Primal-Dual algorithm (35 iterations recommended)"。
- evidence: PDF p.9 Section 4 明确说 ROF minimizer "just need to be solved once"；PDF p.11 Algorithm 1 在 Initialization 后先 "Compute the solution u of the ROF model (1.7)"，循环内只 threshold、apply criteria、update τ。PDF p.11 还说本文数值实验用 ADMM 求 ROF；"35 (15)" 是 retina 示例中 u/τ 的实际迭代步数，不是统一推荐。
- judgment: **错误**。这会让读者误以为 T-ROF 的代价随阈值迭代重复 ROF 求解，扭曲论文的效率主张。
- fix: 算法图、代码和伪代码改成：初始化阈值后先解一次 ROF 得到 u；主循环只做阈值化、criterion C、均值/阈值更新。把 "35 recommended" 改成示例默认值或删掉，并注明论文实验使用 ADMM。
- confidence: high

---

## Should Fix

### S1. dashboard 英文题名使用缩写，未完全匹配 PDF 首页
- severity: should-fix
- target: #2 pcms-rof-linkage
- file: `docs/js/reading-data.js`
- claim: `"Linkage Between PCMS and ROF Model and Its Virtue in Image Segmentation"`。
- evidence: PDF p.1 标题是 "LINKAGE BETWEEN PIECEWISE CONSTANT MUMFORD-SHAH MODEL AND ROF MODEL AND ITS VIRTUE IN IMAGE SEGMENTATION"。
- judgment: **部分正确**。PCMS 是论文里的标准缩写，但 metadata title 应优先与 PDF 首页全题名一致。
- fix: `papers[1].title` 和 `paperNotesV2[1].titleEn` 改成 PDF 全题名。
- confidence: high

### S2. 笔记把复杂度写成严格 O(N) 与 K 无关，证据不足
- severity: should-fix
- target: #2 pcms-rof-linkage
- file: `xiaohao_cai_ultimate_notes/Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md`
- claim: "ROF求解：O(N)"、"总复杂度：O(N) 与K无关"。
- evidence: PDF p.4 说 T-ROF 继承 SaT 的 "computational cost independent of K"；PDF p.18-28 给的是 CPU time 和 u/τ iteration steps。PDF 没给出一般性 O(N) 复杂度定理。
- judgment: **过度形式化**。可以说主要优势是 ROF 只解一次、阈值更新快、实测随 K 更稳；不能把它写成已证明 O(N)。
- fix: 改成 "主导成本为一次 ROF 数值求解 + 若干阈值更新；论文强调相对 PCMS 方法对 K 更不敏感，但未证明统一 O(N) 上界。"
- confidence: high

### S3. dashboard 的实验对象写得略泛，"blurry" 不应作为本篇数值实验主线
- severity: should-fix
- target: #2 pcms-rof-linkage
- file: `docs/js/reading-data.js`
- claim: "实验部分展示 T-ROF 在 noisy、blurry、information loss 等退化图像上的分割效率和质量"。
- evidence: PDF Section 5 具体实验包括 missing pixel values、close intensities、Gaussian noisy multiphase、MRI、stripe、retina manual segmentation；"blurry" 主要出现在 Abstract/Introduction 对 ROF/SaT 灵活性的概括中。
- judgment: **部分正确 / 不够精确**。
- fix: 改成 "missing pixels/noisy/close-intensity/MRI/stripe/retina" 这类 PDF Section 5 实验对象。
- confidence: medium

### S4. 笔记混入过强的落地/商业化和全局最优表述
- severity: should-fix
- target: #2 pcms-rof-linkage
- file: `xiaohao_cai_ultimate_notes/Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md`
- claim: "全球市场规模约$100B"、"FDA认证"、"凸优化，全局最优"、"复杂度O(N)与K无关"。
- evidence: PDF 只讨论数学模型、T-ROF 算法、合成/MRI/retina 实验，没有商业市场或 FDA 路线；理论结论是 fixed parameters 下的 minimizer / partial minimizer，不是整套 segmentation 全局最优。
- judgment: **过度解释**。这些可以作为个人延展，但不能放成论文事实。
- fix: 删除或标为 "非论文原文延展"；把 "全局最优" 改成 "ROF 子问题是凸恢复；PCMS 关系是 partial minimizer / fixed-parameter minimizer"。
- confidence: high

---

## Notes

### N1. 复现评估当前诚实且已同步
- target: #2 pcms-rof-linkage
- evidence: `docs/assets/repro/repro_results.json` 中 #2 为 `completed`，`reproductionLevel: toy-to-partial`，`direct_dice: 0.8989`、`rof_threshold_dice: 0.996`，且 `notes` 明确 "does not solve the exact ROF model or prove Theorem 3.6"。
- judgment: ✅ 保留。`fidelityWarning: Uses proxy smoothing; does not solve the exact ROF model.` 是必要克制说明。

### N2. retina Table 5.4 数值在笔记中基本准确
- target: #2 pcms-rof-linkage
- evidence: PDF p.28 Table 5.4 给出 Li/Pock/Yuan/He/Cai/T-ROF 的 time、SA、DICE_Ω0/Ω1/Ω2；笔记中的 SA、DICE_Ω0、DICE_Ω1、time 与表格一致。
- judgment: ✅ 保留。但表格标题应注明这是 retina test example，不代表 paper-level generic benchmark。

### N3. K>2 dashboard 口径基本克制
- target: #2 pcms-rof-linkage
- evidence: PDF p.8-9 Theorem 3.7 / summary 说明 K>2 进入 PCMS-V，并依赖 `∂Σ_i ∩ ∂Σ_{i+1} = ∅` 等条件；dashboard 已写 "依赖更具体的假设和阈值结构"。
- judgment: ✅ 保留，可在后续精读中补 "PCMS-V" 名称。

---

## Suggested Patch Plan

| priority | project file | action |
|---|---|---|
| high | `xiaohao_cai_ultimate_notes/Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md` | 修正 Theorem 3.4/3.6 定理定位和 λ 参数 |
| high | 同上 | 修正 T-ROF 算法：ROF 解一次，循环只更新 threshold；删除 "35 recommended" |
| medium | `docs/js/reading-data.js` | 英文题名改成 PDF 全题名；实验描述改成 Section 5 实验对象 |
| medium | 笔记 | 降级 O(N)、全局最优、商业化/FDA 等过强表述 |

---

## Claims To Keep

1. 作者顺序以 Xiaohao Cai 开头，PDF 首页和项目作者字段一致。
2. PDF 页数 31 与项目 `pages: 31` 一致。
3. `reproductionTruthLevel: partial-completed` 和 `toy-to-partial` 口径合理。
4. `relation.links: [1, 3, 4]` 合理：连接 SaT Overview、Iterated ROF 和 Segmentation Restoration。
