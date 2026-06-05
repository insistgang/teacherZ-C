# Minimax Review Batch A — Paper #3 iterated-rof

> 审查日期：2026-06-05
> 审查对象：Xiaohao Cai, Gabriele Steidl. "Multiclass Segmentation by Iterated ROF Thresholding." EMMCVPR 2013, LNCS 8081, pp. 237-250.
> 严格按 `reviews/minimax_paper_review_orchestration.md` 的三轮 + 输出 schema。

---

## Executive Result

- must-fix: 0
- should-fix: 2
- note: 4
- batch verdict: **pass-with-fixes**
- post-remediation status: **resolved in Codex pass on 2026-06-05**. The dashboard and note now separate the practical T-ROF algorithm from the projection used in the convergence proof, and the key formula field now states the actual ROF/T-ROF energies instead of a generic linear/quadratic description.

---

## Must Fix

None.

---

## Should Fix

### S1. 普通算法流程不应把 projection 写成必经步骤
- severity: should-fix
- target: #3 iterated-rof
- file: `docs/js/reading-data.js`; `xiaohao_cai_ultimate_notes/Multiclass_Segmentation_Iterated_ROF_超精读笔记_已填充.md`
- claim: "用投影后的阈值序列继续迭代"；"投影阈值并重复，直到阈值序列收敛。"
- evidence: PDF p.6 Algorithm (T-ROF) 的循环只有 threshold ROF minimizer、构造 segments、计算 means、更新 thresholds。PDF p.8 为证明 Theorem 1 才引入 projector `P_n`，并称这是 "a slight modification"。
- judgment: **部分正确但易误导**。projected T-ROF 是收敛证明对象，不是 Section 3 算法主体或数值实验必须步骤。
- fix: 算法流程改成普通 T-ROF 的四步循环；理论保证里再说明收敛定理针对 projected T-ROF algorithm under assumption (A)。
- confidence: high

### S2. keyModelOrFormula 过于泛化，应该直接列出论文公式
- severity: should-fix
- target: #3 iterated-rof
- file: `docs/js/reading-data.js`; `xiaohao_cai_ultimate_notes/Multiclass_Segmentation_Iterated_ROF_超精读笔记_已填充.md`
- claim: "ROF/T-ROF energy uses Total Variation plus linear or quadratic data term."
- evidence: PDF p.5 gives ROF model (11): `TV(u) + μ/2 ∫(u-f)^2 dx`; PDF p.5 gives multiclass T-ROF energy (12): `Σ_i [Per(Σ_i;Ω) + μ ∫_{Σ_i}(τ_i-f)dx]`; PDF p.6 gives threshold update (15): `τ_i = 1/2(m_{i-1}+m_i)`.
- judgment: **部分正确但不够精读**。原句不会造成严重错误，但会丢失这篇最关键的模型结构。
- fix: 在 dashboard 和 Markdown 笔记中直接写 ROF、T-ROF energy、nested sets 和 threshold update。
- confidence: high

---

## Notes

### N1. 身份信息通过
- target: #3 iterated-rof
- evidence: PDF p.1 标题 "Multiclass Segmentation by Iterated ROF Thresholding"，作者 "Xiaohao Cai and Gabriele Steidl"；项目 metadata 与之一致。
- judgment: pass; keep.

### N2. "solve ROF once" 口径准确
- target: #3 iterated-rof
- evidence: PDF p.6 says the ROF minimizer "remains the same during the whole thresholding process"；Algorithm Step 1 computes the ROF solution once before the loop。
- judgment: pass; dashboard/note 已正确强调这一点。

### N3. 实验描述基本准确
- target: #3 iterated-rof
- evidence: PDF abstract says cartoon, texture and medical images；Section 4 includes missing-pixel cartoon, close-intensity synthetic images, brain MRI and stripe examples。
- judgment: pass; 可继续用 "cartoon / texture / medical" 作为高层概括。

### N4. 复现评估诚实且已同步
- target: #3 iterated-rof
- evidence: `docs/assets/repro/repro_results.json` 中 #3 为 `completed`，metrics 为 `raw_kmeans_accuracy: 0.659`、`trof_accuracy: 0.9799`、`threshold_iterations: 3`；dashboard notes 明确 "proxy smoothing" 且 "strict T-ROF should solve ROF once"。
- judgment: pass; keep.

---

## Suggested Patch Plan

| priority | project file | action |
|---|---|---|
| medium | `docs/js/reading-data.js` | 改 keyModelOrFormula；算法 flow 去掉 projection 必经表述；theorem/report 字段保留 projected proof 口径 |
| medium | `xiaohao_cai_ultimate_notes/Multiclass_Segmentation_Iterated_ROF_超精读笔记_已填充.md` | 同步上述公式和算法文字 |

---

## Claims To Keep

1. 作者顺序：Xiaohao Cai 第一，Gabriele Steidl 第二。
2. `reproductionTruthLevel: partial-completed` / `toy-to-partial` 合理。
3. `relation.links: [1, 2, 4]` 合理：对应 SaT Overview、PCMS-ROF Linkage、Segmentation Restoration。
4. toy reproduction warning 充分，没有把 proxy smoothing 写成 exact ROF/ADMM reproduction。
