# Xiaohao Cai 15 篇论文复现评估

本目录不是论文完整复现实验仓库，而是 dashboard 使用的最小可复现评估系统。目标是诚实地区分 toy reproduction、partial reproduction 和 paper-level reproduction，并把能在普通笔记本上运行的 toy/partial 结果写入静态页面。

## 如何运行

```bash
python reproduce/run_all.py
```

脚本会生成：

- `reproduce/results/repro_results.json`
- `reproduce/results/repro_results.csv`
- `reproduce/results/figures/*.png`
- `docs/assets/repro/*.png`
- `docs/assets/repro/repro_results.json`

`docs/assets/repro/` 中的图用于 GitHub Pages 静态展示。

## 可选依赖

脚本会先检测依赖，缺少依赖时对应实验写入 `skipped`，不会伪造结果。

- `numpy`
- `scipy`
- `matplotlib`
- `scikit-image`
- `scikit-learn`
- `pywavelets`

当前脚本会检测这些包，但为了避免本地 `scikit-learn` ABI 或版本问题，核心实验已尽量使用 `numpy/scipy/matplotlib` 和少量 `scikit-image`。`scikit-learn` 与 `pywavelets` 保留为后续 paper-level 扩展依赖。

## 复现等级定义

- `toy`：使用 synthetic/toy 数据，只验证论文核心思想的一个小型可运行片段。
- `toy-to-partial`：实现了论文算法路线的一部分，但关键求解器使用轻量 proxy，例如 Gaussian smoothing 代替严格 ROF/TV minimization。
- `partial`：复现论文核心算法路线的一部分，例如 iterated ROF 的 Chambolle-Pock ROF + threshold update、SLaT RGB+Lab、graph smoothing。
- `paper-level`：接近论文实验设置。当前没有把任何重依赖论文标成 paper-level。
- `assessment-only`：只做难度评估，不运行实验。当前 15 篇都至少有 toy 或 partial 实验。

`resultStatus=completed` 只表示脚本跑通，不表示论文级完整复现。Dashboard 有两个不同层次的字段：
`reproductionLevel` 是展示/声明等级，可以在严格数据-backed gate 通过后显示 `paper-like`；
`reproductionTruthLevel` 是更保守的真实性归类，paper-like 但非 paper-level 的 Iterated ROF 仍保持
`partial-completed`。

- `toy-completed`：synthetic minimal demo only
- `partial-completed`：partial algorithmic route demo
- `paper-level-completed`：close to paper experiments
- `assessment-only`：只做评估

当前 `paper-like = 0 / 15`，`paper-level-completed = 0 / 15`。`docs/scripts/validate.mjs`
默认同时检查 dashboard 和 run-result JSON，拒绝任何 `paper-like` 或 `paper-level` 晋升；
只有在数据-backed gate 已通过并准备进入 promotion review 时，才用 `ALLOW_PAPER_LIKE=1`
或 `ALLOW_PAPER_LEVEL=1` 显式放行。

Iterated ROF 的 paper-like 复现还要求所有本地数据 `source_id` 来自
`reproduce/paper_like/iterated_rof_dataset_sources.json`。Dashboard promotion
必须由 runner 生成并通过 source registry、canonical manifest、summary artifact
和当前本地文件证据的交叉校验；不能靠手填 summary、manifest 或浅层 gate 字段绕过。
晋升后的 `resultFiles` 也必须是 `docs/assets/repro/iterated_rof_paper_like/...`
下的静态 PNG，并且 SHA-256 要绑定到 source summary 中的 runner figure evidence；
旧 toy 图或路径穿越不会被接受。
普通 sync / validate 路径也会校验 canonical source registry 的 schema；测试用
`ITERATED_ROF_SOURCE_REGISTRY_PATH` 只会增加 override 检查，不能替代 canonical registry。

## 完整复现流程文档（per-paper workflows）

`reproduce/paper_like/workflows/<id>_reproduction_workflow.md` 为 15 篇论文每篇提供一份完整复现流程规范，统一 11 节结构：论文身份与第一作者核验、复现目标与诚实分级、完整算法管线、所需数据集与公开等价来源、对照基线、评价指标与论文报告数值、本仓库当前实现、差距分析、运行步骤、风险与代理说明、回链精读笔记。

这些文档描述的是 **paper-level 目标流程**，不代表已完成复现；它们与 `docs/js/reading-data.js` 中的 toy/partial `reproAssessments` 共存，dashboard 复现报告页每张卡片都链接到对应文档。`docs/scripts/validate.mjs` 会校验每个 reproAssessment id 都存在同名 workflow 文档。

## 哪些不是 full reproduction

以下方向的 full reproduction 需要真实数据、专门库或长时间运行，本仓库只提供 toy/partial 演示：

- Tight-frame vessel：需要真实 2D/3D MRA 数据和严格 tight-frame/DCWT 实现。
- Iterated ROF：当前 partial 已实现 Chambolle-Pock ROF、Split-Bregman 对照、raw `mean_f(Omega_i)` threshold update 与 K=2 proxy 检查，但 full reproduction 仍需要 cartoon/texture/medical 数据和论文 Table 1-2 baseline 对照。
- Wavelet sphere：需要 S2LET/SSHT/SO3 等球面小波栈和球面数据。
- RI UQ I：需要 radio interferometric operators、大规模 MCMC、诊断与真实 RI 数据。
- Online RI：需要大规模 visibility streams 才能接近论文实验。
- Proximal Nested Sampling：需要 constrained proximal MCMC、high-dimensional imaging benchmarks 和 evidence validation。

## 结果同步

`run_all.py` 负责写出 JSON/CSV 和图像。Dashboard 的 `reproAssessments` 位于 `docs/js/reading-data.js`，其中 `resultStatus` 与 `resultFiles` 指向这些已生成的静态图。更新实验后，需要重新运行：

```bash
python3 reproduce/run_all.py
node reproduce/sync_to_dashboard.mjs --check
node docs/scripts/validate.mjs
```

`run_all.py` 不生成 paper-like dashboard candidate。只有在 Iterated ROF 的真实本地数据、manifest、source audit、mask 和 runner 输出都通过后，才进入候选晋升验证：

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --review-data-drop /path/to/iterated_rof_drop --data-drop-review-output /tmp/iterated_rof_data_drop_review.json
python3 reproduce/experiments/iterated_rof_paper_like.py --ingest-data-drop /path/to/iterated_rof_drop
python3 reproduce/experiments/iterated_rof_paper_like.py --data-package-review-output /tmp/iterated_rof_data_package_review.json
python3 reproduce/experiments/iterated_rof_paper_like.py --strict-data-ready
python3 reproduce/experiments/iterated_rof_paper_like.py --run --dashboard-candidate-output /tmp/iterated_rof_dashboard_candidate.json --dashboard-static-assets-output /tmp/iterated_rof_dashboard_static_assets.json --copy-dashboard-static-assets --strict-paper-like
ALLOW_PAPER_LIKE=1 node reproduce/sync_to_dashboard.mjs --candidate /tmp/iterated_rof_dashboard_candidate.json --check
```

`--review-data-drop` is a no-write dry run for a user-prepared local drop. It
reports would-copy/current/conflict files, unsupported image extensions, and
path escapes before the canonical data root is touched.

`--ingest-data-drop` is optional and only copies a user-prepared local
`{family}/{images,masks,audit}` drop into the canonical data root; it refuses
conflicting overwrites and still leaves source/license/provenance review fields
for manual completion.

`--data-package-review-output` is a no-download handoff report for that local
package. It recomputes family image/mask status, manifest file-claim freshness,
source-audit artifact freshness, and the manual manifest fields still missing;
it does not make a paper-like claim.

`--dashboard-static-assets-output` writes a manifest that maps each promotable
runner figure to `docs/assets/repro/iterated_rof_paper_like/...`; adding
`--copy-dashboard-static-assets` copies only after the same source summary
artifact and dashboard candidate are promotable. It does not raise the
dashboard level by itself.

Do not set `REPRO_SYNC_REPO_ROOT` for real promotion review. That environment
override is only for isolated tests that build a temporary repo fixture; the
CLI rejects it unless `REPRO_SYNC_ALLOW_REPO_ROOT_OVERRIDE=1` is also set.
Normal sync and promotion checks must run against the actual repository root.

`sync_to_dashboard.mjs` 会对比 `reproduce/results/repro_results.json` 与 `docs/js/reading-data.js` 中的 `reproDetails` 手工字段，严格检查 `id` 唯一性、`priority` 顺序、派生出的 `reproductionTruthLevel`、`resultStatus`、非运行耗时类 `runMetrics`、`resultFiles`、`notes`、`warning`、`fidelityWarning` 等是否一致；如果 `docs/assets/repro/repro_results.json` 与最新 run JSON 同时存在，也会要求两者 id 唯一、顺序一致且 JSON 内容完全一致，避免站点静态资产展示旧复现状态。`--candidate <path> --check` 会先检查 candidate 顶层 metadata 与 `runResultPatch` / `dashboardDetailPatch` 一致，再把 patch 无写入地嵌入当前 dashboard、run JSON 和静态 asset 快照中验证。默认情况下，这个 overlay 也保持 `paper-like = 0 / 15` 与 `paper-level = 0 / 15` 的晋升守卫；只有 generated data-backed candidate 已审查时，才用 `ALLOW_PAPER_LIKE=1` 或 `ALLOW_PAPER_LEVEL=1` 显式进入 promotion review。run JSON 不能用显式 `reproductionTruthLevel` 覆盖派生结果；如果结果声明 `paper-like`，还要求完整的 `paper_like_gate` 结构，包括 `passed=true`、空 `reasons`、`paper-like` dashboard level、非空 `checked_requirements`、全部通过的 `checklist`，以及包含 dataset fingerprint、三类 family、source claim 数量和 figure-evidence 数量的 `evidence_summary`；同时必须有 runner 生成的 `paper_like_verification`，确认 gate 是 dashboard candidate 生成器重算、`can_promote=true`、promotion shape blocker 为 0，且 dataset fingerprint 与 gate evidence summary 一致。该 verification 还必须指向 repo 内 `reproduce/results/` 下可读取的 source summary artifact，并提供匹配的 SHA-256；sync 会读取该 summary，确认其中 `paper_like_gate` 与 run JSON gate 一致，并要求 summary 包含三类 completed quantitative image rows、matching masks、family summaries、run protocol、canonical `reproduce/data/iterated_rof/dataset_manifest.json`、与当前 manifest 一致的 citation/license/provenance 文本、matching structured source audit、`source_id` 存在于 `reproduce/paper_like/iterated_rof_dataset_sources.json`、image/mask/figure/sidecar file evidence、supported image file signatures、image/mask minimum dimensions、image/mask shape match、source audit artifact minimum evidence content，以及由 image/mask evidence 重算得到的 dataset fingerprint，避免只靠 JSON 自称通过。浅层 `{passed:true}`、只有 checklist、没有 promotion verification、没有 summary artifact/SHA、summary artifact 在 `/tmp` 等 repo 外路径、没有 canonical manifest、手填 manifest/summary 无法被当前文件和 source registry 校验、manifest 文本缺失或与 summary 不一致、source audit 缺失或不完整、非图片/过小图片文件证据、或没有 runner image evidence 的伪 gate 不算；如果声明 `paper-level`，则必须有独立完整的 `paper_level_gate`、paper-level evidence summary、`paper_level_verification` 和位于 `reproduce/results/` 下的独立 verification artifact/SHA，artifact 还要包含非空 table comparisons、baseline comparisons、parameter records 和 data source audits，并且每条 row 都要有可读取、SHA 匹配、内容足够且非 placeholder/fixture 文本的 audited artifact，而不是只填一个 id 或薄文件。`runtimeSeconds`/`runtime_seconds` 只检查两边字段存在且为数值；`runMetrics`/`metrics` 中 key 名含 `runtime_seconds` 或以 `_runtime_seconds` 结尾的运行耗时字段会从严格比较中剔除，避免重跑 `run_all.py` 后因 wall-clock 漂移导致同步检查失败。它不改写人工评注字段，例如 `difficultyScore`、`effectScore`。
