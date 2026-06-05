# Minimax 论文审查编排

本文档用于把 15 篇 Xiaohao Cai 第一作者论文交给 Minimax 做外部审查。目标不是让 Minimax 写泛泛论文综述，而是审查本项目的 PDF、dashboard、精读笔记和复现评估是否准确、克制、可回填。

## 审查目标

Minimax 要完成三件事：

1. 核验论文身份：标题、作者顺序、年份、PDF 文件和当前 15 篇口径是否一致。
2. 核验精读内容：项目中的核心问题、方法抓手、关键公式、算法流程、理论保证、实验重点和论文关系是否忠实于原论文。
3. 核验复现评估：当前 toy/partial/paper-level 口径是否诚实，是否有过度声称、指标误导或应该降级/补充说明的地方。

当前项目基线：

- 15 篇 PDF：`docs/00_papers_first_author_xiaohao_cai_deduped/`
- dashboard 数据源：`docs/js/reading-data.js`
- 15 份独立 Markdown 精读笔记：`xiaohao_cai_ultimate_notes/`
- 复现评估说明：`reproduce/README.md`
- 静态复现结果：`docs/assets/repro/repro_results.json`
- 项目校验：`node docs/scripts/validate.mjs`

重要口径：

- `completed` 只表示 toy/partial 脚本跑通，不表示论文级完整复现。
- 当前 `paper-level-completed = 0 / 15`。
- Minimax 若无法在 PDF 中确认某条项目表述，必须标为“未证实”，不要用常识补全。

## 给 Minimax 的总提示

复制下面这段作为每次会话的系统/任务提示：

```text
你是严谨的论文审查员。请只基于我提供的 PDF、项目数据和笔记进行审查，不要擅自补充外部知识。如果某个说法无法从 PDF 或项目文件中确认，请标记为“未证实”。

审查对象是 Xiaohao Cai 15 篇第一作者论文项目。你要审查的不是论文好不好，而是项目内容是否准确：
1. PDF 标题、作者顺序、年份、论文对象是否正确。
2. dashboard / reading-data.js 的精读字段是否忠实于论文。
3. 独立 Markdown 笔记是否混入旧论文、错作者、错公式、错实验或过度解释。
4. 复现评估是否诚实区分 toy、partial、paper-level；若只是 toy，不得写成论文级复现。

输出必须按“问题优先”组织。每个问题给出：
- severity: must-fix / should-fix / note
- target: 影响的论文 priority + id + 文件路径
- claim: 项目中的原说法
- evidence: PDF 中可定位的章节、页码、标题、作者行或实验段落
- judgment: 正确 / 部分正确 / 错误 / 未证实
- fix: 建议如何改写
- confidence: high / medium / low

不要输出长篇论文摘要。只输出能用于修项目的审查结果。
```

## 三轮审查

### 第一轮：身份与元信息

对每篇论文确认：

- PDF 首页标题是否与 `reading-data.js` 的 `papers[].title` 一致。
- 作者顺序是否以 `Xiaohao Cai` 开头。
- 合作者是否写错、漏写或错拼。
- 年份和版本描述是否合理，尤其是 arXiv preprint / journal / conference 的差别。
- 当前 PDF 文件是否确实对应这篇论文，没有错命名或重复论文。

输出重点：

- 错作者、错标题、错论文对象是 `must-fix`。
- 年份口径不精确但不影响理解是 `should-fix`。
- 版本差异说明不足是 `note` 或 `should-fix`。

### 第二轮：精读内容

对照 PDF 审查 `docs/js/reading-data.js` 和对应 Markdown 笔记：

- `coreProblem` 是否准确描述论文要解决的问题。
- `methodHandle` 是否抓住论文方法，而不是把其他论文方法套过来。
- `keyModelOrFormula` 是否与论文公式一致，是否过度简化到会误导。
- `algorithmFlow` 是否符合论文算法顺序。
- `theoremOrGuarantee` 是否把唯一解、收敛、partial minimizer、复杂度或 HPD 近似混淆。
- `experimentFocus` 是否对应论文实验对象、数据集、指标和对照方法。
- `relation.text` 和 `reportExpansion.relationReading` 是否把论文关系讲反、讲重或夸大。

输出重点：

- 公式错、算法错、定理错是 `must-fix`。
- 术语不严谨、实验对象不完整是 `should-fix`。
- 可以增强但不影响事实的是 `note`。

### 第三轮：复现评估

对照 `reproduce/README.md`、`docs/assets/repro/repro_results.json` 和 `reading-data.js` 的复现字段：

- `reproductionTruthLevel` 是否合理。
- `resultStatus=completed` 是否有明确说明只是 toy/partial。
- `minimalExperiment` 是否真的对应论文核心思想。
- `expectedOutcome` 是否和实际 `runMetrics` 匹配。
- `resultQuality`、`warning`、`fidelityWarning` 是否足够克制。
- 是否有任何地方暗示 paper-level completed。

输出重点：

- toy 被写成论文级复现是 `must-fix`。
- 指标提升很小但效果评分过高是 `should-fix`。
- full reproduction 依赖缺失但 warning 不足是 `should-fix`。

## 分批安排

不要一次塞 15 篇给 Minimax。按下面 5 个批次跑，每批跑完输出一个报告。

### Batch A：SaT / ROF 理论核心

| # | id | PDF | Markdown | 复现口径 |
|---|---|---|---|---|
| 1 | `sat-overview` | `docs/00_papers_first_author_xiaohao_cai_deduped/分割方法论总览 SaT Overview.pdf` | `xiaohao_cai_ultimate_notes/分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md` | `partial-completed` |
| 2 | `pcms-rof-linkage` | `docs/00_papers_first_author_xiaohao_cai_deduped/变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf` | `xiaohao_cai_ultimate_notes/Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md` | `partial-completed` |
| 3 | `iterated-rof` | `docs/00_papers_first_author_xiaohao_cai_deduped/多类ROF分割 Iterated ROF.pdf` | `xiaohao_cai_ultimate_notes/Multiclass_Segmentation_Iterated_ROF_超精读笔记_已填充.md` | `partial-completed` |
| 4 | `segmentation-restoration` | `docs/00_papers_first_author_xiaohao_cai_deduped/分割恢复联合模型 Segmentation Restoration.pdf` | `xiaohao_cai_ultimate_notes/Variational_Segmentation-Restoration_超精读笔记_已填充.md` | `toy-completed` |

重点问题：

- ROF / PCMS / Chan-Vese / T-ROF 的关系是否讲准。
- Gaussian proxy smoothing 是否被明确区分于真正 ROF/TV minimizer。
- Segmentation Restoration 是否被误写成其他 joint restoration 论文。

### Batch B：Framelet / SLaT / Sphere 扩展

| # | id | PDF | Markdown | 复现口径 |
|---|---|---|---|---|
| 5 | `framelet-tubular` | `docs/00_papers_first_author_xiaohao_cai_deduped/框架管状结构分割 Framelet.pdf` | `xiaohao_cai_ultimate_notes/Framelet_Based_Tubular_Structures_超精读笔记_已填充.md` | `toy-completed` |
| 6 | `tight-frame-vessel` | `docs/00_papers_first_author_xiaohao_cai_deduped/框架分割管状结构 Framelet Tubular.pdf` | `xiaohao_cai_ultimate_notes/Tight_Frame_Vessel_Segmentation_超精读笔记_已填充.md` | `toy-completed` |
| 7 | `slat-color` | `docs/00_papers_first_author_xiaohao_cai_deduped/SLaT三阶段分割 SLaT Segmentation.pdf` | `xiaohao_cai_ultimate_notes/SLaT_Three-stage_Segmentation_超精读笔记_已填充.md` | `partial-completed` |
| 8 | `sphere-wavelet` | `docs/00_papers_first_author_xiaohao_cai_deduped/球面小波分割 Wavelet Sphere.pdf` | `xiaohao_cai_ultimate_notes/Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md` | `toy-completed` |

重点问题：

- Framelet 短版和 tight-frame 长版是否混淆。
- SLaT 的 smoothing / lifting / thresholding 三阶段是否讲准。
- sphere 论文是否被错误归入 RI/UQ，或是否缺少球面小波依赖说明。
- SLaT toy gain 只有 `0.0053`，效果评分是否需要更克制。

### Batch C：高维分类线

| # | id | PDF | Markdown | 复现口径 |
|---|---|---|---|---|
| 9 | `two-stage-classification` | `docs/00_papers_first_author_xiaohao_cai_deduped/两阶段分类 Two-Stage.pdf` | `xiaohao_cai_ultimate_notes/Two-Stage_Classification_Point_Clouds_超精读笔记_已填充.md` | `partial-completed` |
| 10 | `efficient-variational-classification` | `docs/00_papers_first_author_xiaohao_cai_deduped/高效变分分类 Efficient Variational.pdf` | `xiaohao_cai_ultimate_notes/高效变分分类方法_Efficient_Variational_Classification_超精读笔记_已填充.md` | `partial-completed` |

重点问题：

- Two-Stage Classification 不能再被写成 Two-Stage Segmentation。
- graph Laplacian、graph TV、projection、simplex constraint 的表述是否准确。
- 当前 graph smoothing toy 是否明显低于完整 graph TV primal-dual solver。

### Batch D：RI / UQ / Online

| # | id | PDF | Markdown | 复现口径 |
|---|---|---|---|---|
| 11 | `high-dimensional-uq` | `docs/00_papers_first_author_xiaohao_cai_deduped/高维逆问题不确定性量化 Uncertainty Quantification.pdf` | `xiaohao_cai_ultimate_notes/High-Dimensional_Inverse_Problems_UQ_超精读笔记_已填充.md` | `toy-completed` |
| 12 | `ri-uq-i` | `docs/00_papers_first_author_xiaohao_cai_deduped/无线电干涉不确定性I Radio Interferometric I.pdf` | `xiaohao_cai_ultimate_notes/Radio_Interferometric_Imaging_I_超精读笔记_已填充.md` | `toy-completed` |
| 13 | `ri-uq-ii` | `docs/00_papers_first_author_xiaohao_cai_deduped/无线电干涉不确定性II Radio Interferometric II.pdf` | `xiaohao_cai_ultimate_notes/Radio_Interferometric_Imaging_II_超精读笔记_已填充.md` | `toy-completed` |
| 14 | `online-ri` | `docs/00_papers_first_author_xiaohao_cai_deduped/在线无线电干涉成像 Online Radio Imaging.pdf` | `xiaohao_cai_ultimate_notes/Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md` | `toy-completed` |

重点问题：

- MAP-UQ、HPD credible region、local credible interval 是否讲准。
- RI UQ I 的 proximal MCMC 和 RI UQ II 的 MAP estimation 是否被清楚区分。
- toy Fourier inverse problem 是否被误认为真实 RI operator。
- Online RI 的 storage claim 是否只在 toy block splitting 上成立。

### Batch E：Bayesian model selection

| # | id | PDF | Markdown | 复现口径 |
|---|---|---|---|---|
| 15 | `proximal-nested-sampling` | `docs/00_papers_first_author_xiaohao_cai_deduped/近端嵌套采样 Proximal Nested Sampling.pdf` | `xiaohao_cai_ultimate_notes/Proximal_Nested_Sampling_超精读笔记_已填充.md` | `toy-completed` |

重点问题：

- nested sampling、model evidence、Bayes factor、proximal MCMC 的关系是否讲准。
- 当前 toy 的 absolute log evidence error `2.4676` 是否已足够标注为 rough illustrative。
- 是否需要把任何效果评分降级，或增强 warning。

## 每批复制给 Minimax 的任务模板

```text
请审查 Batch <A/B/C/D/E>。

我会提供：
1. 本批 PDF 文件。
2. 本批对应 Markdown 精读笔记。
3. docs/js/reading-data.js 中本批 paperNotesV2 / reproAssessments 相关字段。
4. docs/assets/repro/repro_results.json 中本批运行结果。

请按三轮审查：
第一轮：标题、作者顺序、年份、PDF 对象核验。
第二轮：精读内容核验，包括核心问题、方法、公式、算法、理论、实验、关系。
第三轮：复现评估核验，包括 toy/partial/paper-level 口径、指标解释、warning 是否足够。

输出不要写论文摘要。输出问题清单、逐篇核验表和建议补丁。
```

## Minimax 输出格式

每批报告必须使用下面格式：

```markdown
# Minimax Review Batch <A/B/C/D/E>

## Executive Result

- must-fix: <number>
- should-fix: <number>
- note: <number>
- batch verdict: pass / pass-with-fixes / fail

## Must Fix

### M1. <short title>
- severity: must-fix
- target: #<priority> <id>
- file: <project file path>
- claim: <project claim>
- evidence: <PDF page/section/title/author line>
- judgment: <why wrong>
- fix: <specific rewrite>
- confidence: high / medium / low

## Should Fix

同上。

## Notes

同上。

## Per Paper Table

| # | id | metadata | content | reproduction | verdict |
|---|---|---|---|---|---|

## Suggested Patch Plan

| priority | project file | field/section | action | replacement summary |
|---|---|---|---|---|

## Claims To Keep

列出 Minimax 明确认为准确、无需修改的关键项目表述。
```

## 回填规则

收到 Minimax 报告后，按这个优先级回填：

1. `must-fix`：立刻改 `docs/js/reading-data.js`、Markdown 笔记或 PDF 目录 README。
2. `should-fix`：确认是否影响 dashboard 展示；影响用户理解的就改。
3. `note`：只在 README 或笔记中补充，不改核心数据。

复现相关改动要特别谨慎：

- 不把 toy 改成 paper-level，除非 Minimax 明确指出我们真的实现了论文级实验，且能用代码和数据证明。
- 指标差或误差大的结果，优先加 `warning` / `resultQuality`，而不是删除。
- 如果 Minimax 建议降级效果评分，要同步改 `effectScore`、`effectLabel`、`expectedOutcome` 和 Markdown 笔记中的复现判断。

## 最终验收

Minimax 全部批次审完、项目回填后，必须跑：

```bash
node docs/scripts/validate.mjs
git diff --check
```

最终 dashboard 应满足：

- 15/15 第一作者核验仍成立。
- 15/15 精读卡片仍存在。
- 15/15 独立 Markdown 笔记仍存在。
- 所有复现仍明确区分 toy / partial / paper-level。
- `paper-level-completed` 未经真实证据不得大于 0。
