# Full Reproduction Team Plan

## Scope

This project currently provides toy and partial reproductions for 15 Xiaohao Cai first-author papers. The next objective is to move from teaching demos to data-backed paper-like reproductions, without claiming paper-level completion until original or equivalent data, baselines, parameter records, and paper-table comparisons exist.

`paper-level-completed` must remain `0 / 15` until the dashboard has generated evidence for a specific paper-level claim.

Per-paper complete-reproduction workflow specs live in `reproduce/paper_like/workflows/<id>_reproduction_workflow.md` (one per paper). Each defines the paper-level target pipeline, required datasets / public-equivalent sources, baselines, metrics with paper-reported numbers, the current toy/partial implementation, the gap to paper-like / paper-level, and explicit proxy disclosures. They are linked from each dashboard reproduction card and validated by `docs/scripts/validate.mjs`; they state target flows only and do not by themselves raise any reproduction level.

## Team Roles

| Agent | Role | Ownership | Deliverable |
|---|---|---|---|
| A | Reproduction route auditor | Read-only audit across docs and reproduce code | Feasibility ranking, data/baseline/library requirements |
| B | Iterated ROF paper-like runner | `reproduce/experiments/iterated_rof_paper_like.py`, related tests/spec | Local-data runner for paper #3 |
| C | Result sync hardening | `reproduce/sync_to_dashboard.mjs`, `docs/scripts/validate.mjs`, docs | Runtime-stable sync checks and complete paper-like gate shape checks |
| Lead | Integration | Review, verification, coordination | Clean working tree, validated next step |

## Priority Order

1. `iterated-rof`
   - Current level: `partial`.
   - Why first: existing Chambolle-Pock ROF, Split-Bregman cross-check, threshold update, and paper-like readiness scaffold.
   - Paper-like target: run cartoon, texture, and medical image families with direct threshold/clustering and at least one classical baseline.

2. `slat-color`
   - Current level: `partial`.
   - Why second: RGB plus Lab-like lifting is implemented but the current synthetic case shows weak metric gain.
   - Paper-like target: degraded color image set, real Lab conversion, RGB-only and RGB+Lab comparisons.

3. `two-stage-classification` and `efficient-variational-classification`
   - Current level: `partial`.
   - Why third: current graph smoothing demo is simple and stable, but missing graph TV / primal-dual solver and benchmarks.
   - Paper-like target: public high-dimensional or point-cloud benchmark, graph TV solver, accuracy/runtime tables.

4. `segmentation-restoration`
   - Current level: `toy`.
   - Why fourth: toy effect is clear, but the joint variational model is not faithfully implemented.
   - Paper-like target: implement a closer alternating minimization with documented fidelity terms.

5. `framelet-tubular`, `tight-frame-vessel`, `sphere-wavelet`
   - Current level: `toy`.
   - Why later: require framelet/tight-frame or spherical wavelet stacks and domain data.
   - Paper-like target: replace Gaussian proxy with actual transform stack or explicitly keep proxy status.

6. `high-dimensional-uq`, `ri-uq-i`, `ri-uq-ii`, `online-ri`, `proximal-nested-sampling`
   - Current level: `toy`.
   - Why later: real reproduction needs RI operators, posterior diagnostics, online operators, nested-sampling validation, and long-running experiments.
   - Paper-like target: first build honest small inverse-problem benchmarks, then move to RI-specific operators/data.

## Phase 1: Foundations

- [x] Harden result synchronization so rerunning experiments does not fail solely because wall-clock runtime changed.
- [x] Add a local-data runner for `iterated-rof` that remains blocked when data is absent.
- [x] Keep synthetic-only outputs labeled `toy` or `partial`.
- [x] Add tests for dataset scanning, image loading, mask loading, and blocked states.
- [x] Add local dataset manifest/provenance checks so paper-like promotion is blocked without source and license-review evidence.
- [x] Add a gated dashboard static asset manifest/copy command that only copies runner figures after a source-summary-backed dashboard candidate is promotable.

## Phase 2: Data-Backed Iterated ROF

- [x] Add a no-download data layout preparation command for local directories and manifest template.
- [x] Add a no-download local data-drop ingest command that copies user-prepared images/masks/audit files into the canonical layout without overwriting conflicts.
- [x] Add a no-download data-drop dry-run review command that detects copy plans, conflicts, unsupported files, and path escapes before touching canonical data.
- [x] Add a no-download data gap checklist that tells reviewers which family paths, masks, manifest claims, and source records are still missing.
- [x] Add a no-download data package review report that surfaces stale file/source-audit claims and manual manifest fields still blocking the runner.
- [x] Harden source-audit artifact content checks so matched SHA-256 files still fail without the manifest source URL, a valid date, and non-empty reviewer-note / conversion-or-mapping fields in the artifact text.
- [ ] Populate local data under `reproduce/data/iterated_rof/{cartoon,texture,medical}/images`.
- [ ] Use nontrivial local images; tiny, blank, or near-binary smoke-test fixtures remain blocked below paper-like.
- [ ] Add matching same-relative-path masks for all three families; strict paper-like promotion requires completed quantitative output for `cartoon`, `texture`, and `medical`.
- [ ] Run `--refresh-manifest-file-claims` after placing local files.
- [ ] Fill `reproduce/data/iterated_rof/dataset_manifest.json` with source, citation, license/provenance notes that are not fixture/tempfile placeholders, structured `source_audit` records with source URL/date/source artifact and license snapshot paths under the matching `<family>/audit/` directory, source/license artifact text containing the manifest source URL, a valid date, and non-empty reviewer-note / conversion-or-mapping fields, matching artifact hashes, conversion notes, local mapping review, `license_reviewed=true`, `provenance_reviewed=true`, explicit `synthetic_fixture=false`, and file-level `files[]` entries with matching image/mask SHA-256 values.
- [ ] Run T-ROF on all available image families.
- [ ] Report quantitative metrics for every required family; qualitative-only status is allowed for exploratory local runs but cannot promote.
- [ ] Add the gate-accepted baseline comparisons: raw K-means on grayscale input plus Multi-Otsu, or the runner-labeled quantile fallback when Multi-Otsu is unavailable or degenerate. Chan-Vese-style comparisons are future optional analysis, not current promotion evidence.
- [ ] Copy reviewed runner figure PNGs to `docs/assets/repro/iterated_rof_paper_like/...` before dashboard promotion validation; the sync guard checks those static files against source-summary figure SHA-256 values.

## Phase 3: Dashboard Promotion

- [ ] Define a `paper-like` display level only after generated data-backed results exist.
- [x] Keep static validation at `paper-like = 0 / 15` by default; future promotion must explicitly set `ALLOW_PAPER_LIKE=1` after a generated data-backed candidate passes review.
- [x] Harden dashboard sync so duplicate ids, shallow `{passed:true}` paper-like gates, missing source summary artifacts, missing priority/order fields, and paper-level claims without independent verification are rejected; paper-level verification rows must also carry audited artifact path/SHA evidence with substantive non-placeholder content instead of only non-empty ids or thin files.
- [ ] Sync dashboard fields from generated outputs after a real data-backed candidate passes the gate.
- [ ] Keep `paper-level-completed = 0 / 15` unless original or equivalent protocol is implemented.
- [ ] Run current-state validation:

```bash
python3 -m unittest discover -s reproduce/tests -p 'test_*.py'
env MPLCONFIGDIR=/private/tmp/teacherZ-mplconfig python3 reproduce/run_all.py
node reproduce/sync_to_dashboard.mjs --check
node docs/scripts/validate.mjs
```

- [ ] After a generated data-backed Iterated ROF candidate passes reviewer inspection, run promotion-review validation:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --review-data-drop /path/to/iterated_rof_drop --data-drop-review-output /tmp/iterated_rof_data_drop_review.json
python3 reproduce/experiments/iterated_rof_paper_like.py --data-package-review-output /tmp/iterated_rof_data_package_review.json
python3 reproduce/experiments/iterated_rof_paper_like.py --strict-data-ready
python3 reproduce/experiments/iterated_rof_paper_like.py --run --dashboard-candidate-output /tmp/iterated_rof_dashboard_candidate.json --dashboard-static-assets-output /tmp/iterated_rof_dashboard_static_assets.json --copy-dashboard-static-assets --strict-paper-like
ALLOW_PAPER_LIKE=1 node reproduce/sync_to_dashboard.mjs --candidate /tmp/iterated_rof_dashboard_candidate.json --check
```

## Phase 4: Next Papers

- [ ] Rework `slat-color` with real Lab conversion and a stronger degraded color benchmark.
- [ ] Add graph TV / primal-dual solver for classification papers.
- [ ] Replace UQ random-interval demos with calibrated small posterior checks.
- [ ] Revisit RI and nested sampling only after operators/data and diagnostics are defined.

## Claim Rules

- `completed`: script ran successfully.
- `toy-completed`: synthetic mechanism demo only.
- `partial-completed`: partial algorithmic route implemented.
- `paper-like`: public or local data-backed reproduction with baselines and metrics; not necessarily the original paper protocol.
- `paper-level-completed`: original/equivalent paper protocol, baselines, parameter records, and figures/tables reproduced.

Never promote a claim based on synthetic data alone.
