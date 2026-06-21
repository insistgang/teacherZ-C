# Spec: Iterated ROF Paper-Like Reproduction

## Assumptions

1. Start with paper #3, `iterated-rof`, because its partial reproduction already has a real ROF solver and T-ROF threshold loop.
2. Target `paper-like` first, not `paper-level`. Paper-level requires original cartoon / texture / medical images or an explicitly equivalent experimental protocol.
3. No dashboard level should be raised until the experiment has real data, baselines, metrics, figures, and synced run output.
4. Network downloads are not automatic. The source manifest records official URLs and download policy; large or license-restricted archives should be downloaded manually after terms review.

## Objective

Build a reproducible path from the current `partial` T-ROF implementation to a paper-like reproduction of Xiaohao Cai and Gabriele Steidl, "Multiclass Segmentation by Iterated ROF Thresholding".

Success means:

- Real or public-replacement cartoon, texture, and medical images are run through the same T-ROF pipeline.
- The accepted local-runner baselines are reported for every quantitative image: raw K-means on the grayscale input plus Multi-Otsu thresholds when `scikit-image` is available, or the runner-labeled quantile fallback when Multi-Otsu is unavailable or degenerate. Chan-Vese-style baselines are future optional analysis, not part of the current strict gate.
- Metrics and figures are produced for each dataset family.
- Local source provenance, citation, and license-review status are recorded for every family with local images.
- The dashboard remains honest: `partial` until data-backed paper-like results exist.

## Tech Stack

- Python 3.9+
- Hard dependencies for the local runner: `numpy`, `scipy`, `matplotlib`
- Optional enhancement: `scikit-image` is used for Multi-Otsu thresholds when available; if it is missing or Otsu fails on a degenerate image, the runner labels the fallback as quantile-based rather than presenting it as Otsu.
- Existing project validation: `node docs/scripts/validate.mjs`

## Commands

Current partial reproduction:

```bash
python3 -m unittest reproduce.tests.test_iterated_rof
env MPLCONFIGDIR=/private/tmp/teacherZ-mplconfig python3 reproduce/run_all.py
node reproduce/sync_to_dashboard.mjs --check
node docs/scripts/validate.mjs
```

Paper-like readiness audit:

```bash
python3 -m unittest reproduce.tests.test_iterated_rof_paper_like_scaffold
python3 reproduce/experiments/iterated_rof_paper_like.py --prepare-data-layout
python3 reproduce/experiments/iterated_rof_paper_like.py --review-data-drop /path/to/iterated_rof_drop --data-drop-review-output /tmp/iterated_rof_data_drop_review.json
python3 reproduce/experiments/iterated_rof_paper_like.py --ingest-data-drop /path/to/iterated_rof_drop
python3 reproduce/experiments/iterated_rof_paper_like.py --refresh-manifest-file-claims
python3 reproduce/experiments/iterated_rof_paper_like.py --check-manifest-file-claims
python3 reproduce/experiments/iterated_rof_paper_like.py --data-package-review-output /tmp/iterated_rof_data_package_review.json
python3 reproduce/experiments/iterated_rof_paper_like.py
python3 reproduce/experiments/iterated_rof_paper_like.py --sources
```

Local runner scaffold:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --run
```

`--run` scans local files under `reproduce/data/iterated_rof/{cartoon,texture,medical}/images`,
matches masks with the same relative path and extension under `masks`, and writes a JSON summary to
`reproduce/results/iterated_rof_paper_like_summary.json` by default. If no local
images exist it remains blocked; if masks are missing it runs ROF + T-ROF and
marks those rows `qualitative_only`.

When local images exist, copy `reproduce/data/iterated_rof/dataset_manifest.template.json`
to `reproduce/data/iterated_rof/dataset_manifest.json`, replace every template
source note with the reviewed source/citation/license/provenance details plus a
structured `source_audit`, and record one `files[]` entry per local image. Each file entry records the image
relative path, image SHA-256, optional mask relative path, and optional mask
SHA-256. Missing, copied-template, or incomplete local manifests are reported as
`claim_blockers`: they do not stop exploratory local runs, but they block
paper-like dashboard promotion.
Use `--prepare-data-layout` to create the family directories and copy the
manifest template without downloading data or overwriting an existing manifest.
Use `--review-data-drop <path>` before ingesting reviewed files to dry-run the
temporary `{family}/{images,masks,audit}` layout against the canonical data
root. It reports would-copy/current/conflict files, unsupported image
extensions, and staging path escapes without creating directories or copying
files. Add `--data-drop-review-output <path>` to write the dry-run JSON; the
CLI exits non-zero when conflicts are present.
Use `--ingest-data-drop <path>` when reviewed files have been downloaded and
converted outside the repository into a temporary
`{family}/{images,masks,audit}` layout. It copies supported image/mask files and
audit artifacts into `reproduce/data/iterated_rof/`, refuses to overwrite
conflicting canonical files, creates the manifest template when needed, and
refreshes `files[]` image/mask SHA-256 claims. It does not set source review
flags, citation, license, provenance, or audit approval fields.
Use `--refresh-manifest-file-claims` after placing files to populate `files[]`
image/mask paths and SHA-256 values from local data. It does not set source,
license, or citation fields, and it does not silently delete stale `files[]`
claims for removed local images.
Every manifest `source_id` must match an entry in
`reproduce/paper_like/iterated_rof_dataset_sources.json`; local-only or
hand-written source names do not count as reviewed provenance for paper-like
promotion.
Use `--check-manifest-file-claims` in CI or review workflows to verify those
file claims without writing; it exits non-zero if the manifest is missing,
invalid, stale, or would change after refresh. Its JSON report lists stale
claims as concrete `{family, image}` records for manual cleanup.
Use `--refresh-source-audit-artifact-claims` after placing source artifacts and
license snapshots to populate `source_artifact_sha256` and
`license_snapshot_sha256` from local audit files, including file-level
`files[].source_audit` overrides. It does not set review flags, dates,
citation, license text, or provenance notes. Use
`--check-source-audit-artifact-claims` to verify the same artifact SHA fields
without writing. The check is stale when a manifest family entry has no
structured `source_audit`; a no-path hash check is not accepted as source
review.

Both readiness and run summaries include `paper_like_gate`. This is the
machine-readable promotion gate for dashboard work: `status=completed_local_runner`
is not enough. Promotion remains blocked unless `paper_like_gate.passed` is
true. Quantitative image rows must include numeric T-ROF `clustering_accuracy`
and numeric raw K-means / Multi-Otsu baseline `clustering_accuracy`; empty
metric objects are not paper-like evidence. Quantitative masks must be
decodable, shape-compatible, and contain at least two labels; a constant mask
does not count as quantitative evidence. Completed quantitative rows must also
record the expected solver string, nonempty threshold list matching
`n_classes - 1`, threshold iteration count, ROF iteration count, finite ROF
residual, and runner parameters (`mu`, `rof_n_iter`, `trof_max_iter`). Baseline
rows must record the raw K-means method and the Multi-Otsu or quantile-fallback
method plus thresholds, not just metrics. File-level source claims must also
carry `source_id`, `license_reviewed=true`, `provenance_reviewed=true`,
non-empty `citation` / `license_note` / `provenance_note`,
`synthetic_fixture=false`, a structured `source_audit` with reviewed source URL
matching the selected registry entry, download date, local raw source artifact
path plus hash, local license snapshot path plus hash, conversion notes, and
local-file mapping review, no obvious
fixture/tempfile text or copied template placeholder text, and
image/mask SHA-256 values that match the local file evidence.
The raw source artifact and license snapshot paths must resolve under the
matching `reproduce/data/iterated_rof/<family>/audit/` directory, so one
family's source review evidence cannot be borrowed from another family or from
an arbitrary local path. Relative source-audit paths in the manifest are
resolved from the local data root, so `cartoon/audit/source-artifact.ext` points
to `reproduce/data/iterated_rof/cartoon/audit/source-artifact.ext` for the
canonical data root. Those artifact files must be substantive review evidence,
not tiny placeholders or local test-fixture stubs; matching SHA-256 alone is not
enough for data-ready or promotion checks. The artifact text must also carry
structured review evidence in the file itself: the manifest `source_url` or
matching `source_url=...`, a valid `YYYY-MM-DD` review/download date, a
non-empty `reviewer_note=` or equivalent source-review field, and a non-empty
`conversion_note=` / `local_file_mapping=` or equivalent mapping field.
Otherwise narrative-only mentions of review or mapping remain invalid even if
the artifact path and SHA-256 match the manifest.
The gate must also re-check local image, mask, and generated figure paths
against disk so stale or fabricated hashes cannot promote a saved report.
Dashboard candidate and saved-summary verification additionally rescan the
current canonical data root; promotion is blocked if current image/mask files
change outside the report, if a current file lacks a manifest `files[]` claim,
or if saved per-image source claims differ from the effective current manifest
claim.
Completed image rows must point to files under
`reproduce/data/iterated_rof/<family>/images/`; masks must point to the matching
relative path under `masks/`; figures must be PNG files under the configured
figure output directory; and every recorded file-evidence `path` must resolve to
the same path as the report row. Source IDs are checked against
`reproduce/paper_like/iterated_rof_dataset_sources.json`, whose entries must
also pass schema checks for matching `target_family`, unique `source_id`, integer
`priority`, and non-empty URL/license/local-layout fields, so arbitrary source
names do not count as reviewed provenance.
Every generated figure also has a sibling `*.png.evidence.json` sidecar. The
gate re-reads that sidecar and requires it to match the current report row's
image/mask/figure SHA-256 values, panel list, solver parameters, thresholds,
metrics, and baseline method metadata.
Promotion checks also re-open image and mask files with the local image reader,
and reject generated figures that are too small or visually blank. Matching
hashes and sidecars alone are not enough for dashboard promotion.
They also reject tiny, visually blank, or near-binary input images as paper-like
evidence, which keeps synthetic smoke-test fixtures below the promotion gate.
The gate also includes a structured `checklist` so reviewers can see which
requirement group failed instead of parsing only flat reason strings.
The gate includes `evidence_summary` with the gate id, dataset fingerprint,
family coverage, source-claim count, and figure-evidence count; dashboard sync
requires this summary for any future `paper-like` result.

Run summaries also include `family_summaries`, a three-row table candidate for
paper-like reporting. Each row aggregates completed/quantitative image counts,
mean T-ROF metrics, mean baseline metrics, figure paths, source claims, and
errors for one data family. When `--run` is used, the runner also writes these
rows as CSV to `reproduce/results/iterated_rof_paper_like_family_summary.csv`
by default; pass `--family-summary-output <path>` to choose a different table
path.
Run summaries also include `images`, the per-image evidence source. When
`--run` is used, the runner writes it to
`reproduce/results/iterated_rof_paper_like_image_evidence.csv` by default; pass
`--image-evidence-output <path>` to choose a different table path. Each row
records the report/gate/fingerprint context, image and mask paths, qualitative
status, figure path/panels/hash/size, figure-evidence sidecar path/hash, file
hashes, manifest source claim, source claim hashes, solver parameters,
thresholds, T-ROF metrics, baseline methods/metrics, and runner errors.
Formula-like text values from manifests or errors are prefixed with `'` in CSV
outputs so reviewer spreadsheets do not execute them as formulas.
Run summaries and dashboard candidates also include `dataset_fingerprint`, a
SHA-256 digest over the sorted local image/mask file hashes, so reviewers can
tie metrics and figures to one data snapshot. Promotion checks re-compute the
fingerprint from per-image evidence rows and block stale or arbitrary
fingerprint values.

Pass `--dashboard-candidate-output <path>` to write a gated dashboard promotion
candidate. This file is advisory only: it remains `can_promote=false` while
`paper_like_gate.passed` is false, and it does not modify `docs/js/reading-data.js`.
The candidate builder recomputes `paper_like_gate` from the current summary
evidence instead of trusting a saved `paper_like_gate.passed` boolean. It also
recomputes family summaries, top-level image counts, result files, and run
metrics from per-image evidence rows; re-loads the current on-disk
`dataset_manifest.json`; and requires the canonical run protocol and figure
directory before it can promote. Promotable candidates include
`candidateDetails.paper_like_verification`; dashboard sync requires that
runner-generated verification metadata for any future `paper-like` entry, so a
complete-looking but hand-written gate is still rejected. That verification
must include a readable source summary path under repo `reproduce/results/`
and matching SHA-256; dashboard sync reloads that summary and checks that its
`paper_like_gate` and dataset
fingerprint match the run result gate. It also checks that the summary contains
the three completed quantitative image rows, matching masks, family summaries,
run protocol, the canonical
`reproduce/data/iterated_rof/dataset_manifest.json`, image/mask/figure/sidecar
file evidence, reviewed citation/license/provenance text that matches the
canonical manifest, matching structured source-audit records, `source_id` values that exist in
`reproduce/paper_like/iterated_rof_dataset_sources.json`, supported image file
signatures, minimum 32px image/mask dimensions, exact source-claim /
figure-evidence counts, `family_summaries` figure paths and source claims that
match same-family image rows, and a dataset fingerprint
recomputed from image/mask evidence. The
source summary and canonical manifest are evidence inputs, not promotion
authority by themselves: self-filled summary rows, manifest source claims, or
gate booleans cannot bypass runner-generated verification, current-file rescan,
the source registry check, and dashboard overlay validation.
The same candidate also emits
`runResultPatch` for `reproduce/results/repro_results.json` and
`dashboardDetailPatch` for `docs/js/reading-data.js`, keeping metrics, result
files, gate evidence, and verification metadata in one reviewable artifact.
Pass `--dashboard-static-assets-output <path>` to write the corresponding
static-figure manifest. The manifest maps each completed quantitative runner
figure to the derived `assets/repro/iterated_rof_paper_like/...` result file,
records source and static SHA-256 values, and marks every asset as `missing`,
`stale`, `current`, or blocked. Add `--copy-dashboard-static-assets` only in a
promotion workflow; copying is refused unless the same source summary artifact
and recomputed dashboard candidate are promotable. This copy step only places
verified figure PNGs under `docs/assets/repro`; it does not edit dashboard JSON
or claim `paper-like` by itself.
Programmatic shape-only candidate checks are not promotion evidence; promotion
review must validate the candidate as an overlay on the current dashboard,
run-result JSON, and static asset snapshots.
Before applying those patches, validate the candidate with
`ALLOW_PAPER_LIKE=1 node reproduce/sync_to_dashboard.mjs --candidate <candidate.json> --check`
after the generated data-backed candidate has been reviewed. Without that
environment flag, candidate overlay validation keeps the project-wide
`paper-like = 0 / 15` invariant and intentionally fails promotion overlays.
Do not set `REPRO_SYNC_REPO_ROOT` during real promotion review; that override is
for isolated test fixtures and the CLI requires
`REPRO_SYNC_ALLOW_REPO_ROOT_OVERRIDE=1` before it will honor it.
Never set the flag for ordinary dashboard validation. The candidate check also
fails for blocked candidates, for metadata drift between the candidate and its
generated patches, and for promotable-looking candidates whose patches would
not pass after being embedded into the current dashboard, run-result JSON, and
static asset snapshots.
Pass `--promotion-audit-output <path>` to write a compact reviewer/CI summary of
the same promotion state, including gate checklist status, blocker counts,
family status counts, manifest status, dataset fingerprint, and the same
promotion shape blockers used by the dashboard candidate. A promotion audit can
recommend `paper-like` only when it is tied to the same readable source summary
artifact under `reproduce/results/` that the dashboard candidate would use; a
gate-passing in-memory report without that artifact remains advisory and
blocked. It also copies
`ready_for_local_runner`, `data_ready_blockers`, and the per-family
`source_audit` artifact/path/hash status matrix from a freshly recomputed data
gap checklist. It is advisory only and does not modify dashboard data; it
recomputes both the gate and the data-gap/source-audit matrix from current
summary evidence plus current canonical data/manifest state before writing.
Pass `--data-gap-output <path>` to write a compact local-data acquisition
checklist. This checklist records the current family paths, primary source
candidates, a per-family `acquisition_plan` with target local paths, required
manifest/source-audit fields, and post-download commands, missing image/mask
requirements, manifest claim blockers, gate reason counts, preflight content
issues, and next actions for a reviewer. The preflight checks image/mask
decodability, mask shape compatibility, and whether input images are large and
nontrivial enough for paper-like evidence. It does not download data or change
dashboard claims; each write recomputes the checklist instead of trusting a
saved `data_gap_checklist` embedded in an older summary. Each family row also
includes a structured `source_audit` status
showing whether source artifacts and license-snapshot artifacts live under the
expected family `audit/` root and whether their SHA-256 values match.
File-level `files[].source_audit` overrides are summarized under the same
family row, so a complete family-level claim cannot mask a broken per-image
source review.
When used without `--output`, `--data-gap-output` writes only the compact
checklist and leaves the default readiness JSON under `reproduce/results`
untouched.
Pass `--data-package-review-output <path>` after ingest or manual file placement
to write a no-download operator review of the current package. It includes the
same family image/mask and source-audit matrix as the data-gap checklist, plus
the current manifest `files[]` claim check, source-audit artifact claim check,
and a per-family list of manual manifest fields still missing (`citation`,
license/provenance review flags, source-audit date, artifact paths/SHA-256
values, and local mapping review). When used without `--output`, it writes only
the package-review JSON and leaves the default readiness report untouched.
Pass `--strict-data-ready` to make that checklist a cheap pre-run CI gate. It
exits non-zero unless `ready_for_local_runner=true`, meaning all required local
data, masks, manifest claims, and preflight checks are ready for the local
runner. This gate does not claim paper-like completion because ROF/T-ROF,
baselines, figures, and promotion evidence still have to run afterward.
Pass `--verify-summary <summary.json>` to verify a saved `--run` summary without
rerunning ROF/T-ROF. This rewrites a verification JSON with a recomputed
`paper_like_gate` and promotion-shape blockers, can also emit candidate/audit
files, and exits non-zero unless the recomputed candidate is promotable.
Pass `--strict-paper-like` in CI or promotion workflows to exit non-zero unless
the recomputed gate passes and the generated dashboard candidate is promotable.
Normal audits keep writing reports even when blocked.

Run summaries include `run_protocol`, which records the protocol id, schema
version, solver path, threshold-update formula, stopping tolerances, iteration
limits, seed, figure directory, and dashboard promotion rule. Dashboard
candidates copy this protocol into `candidateDetails.run_protocol` only after
the promotion checks verify the static protocol fields, canonical figure
directory, positive-integer iteration fields, and agreement between
`run_protocol.parameters` and every completed quantitative image row. Dashboard
`resultFiles` are static `docs/assets/repro/iterated_rof_paper_like/...` PNG
paths derived from runner figures; promotion validation rejects stale toy assets
or any static file whose SHA-256 does not match the source summary figure
evidence.

## Project Structure

```text
reproduce/experiments/sat_rof_trof.py
  Current partial runner shared by #1, #2, #3.

reproduce/experiments/iterated_rof_paper_like.py
  Paper-like readiness audit and future dataset-backed runner for #3 only.

reproduce/data/iterated_rof/
  Local data drop zone. Data is not committed unless it is small and license-safe.
  `images/`, `masks/`, and `audit/` are ignored by git; use `dataset_manifest.json` for source provenance.

reproduce/paper_like/iterated_rof_spec.md
  This spec and task plan.

reproduce/paper_like/iterated_rof_dataset_sources.json
  Candidate source registry with URLs, local target family, download policy, and license notes.

reproduce/tests/test_iterated_rof.py
  Existing partial-level tests.

reproduce/tests/test_iterated_rof_paper_like_scaffold.py
  Readiness-audit tests.
```

## Data Protocol

The paper-like reproduction needs three data families:

| Family | Minimum local requirement | Paper-like purpose | Paper-level gap |
|---|---|---|---|
| `cartoon` | At least 1 degraded or clean cartoon-style image, preferably with mask/labels | Missing-pixel or smooth-region T-ROF behavior | Original paper figure and exact degradation protocol |
| `texture` | At least 1 close-intensity texture image, preferably with mask/labels | T-ROF behavior on texture / stripe-like regions | Original texture image and baseline table |
| `medical` | At least 1 medical image, preferably MRI-like with mask/labels | Medical-style grayscale segmentation | Original paper medical image and reported labels |

Recommended first sources:

| Family | Source | Fit | URL |
|---|---|---|---|
| `texture` | Prague Texture Segmentation Benchmark | Texture mosaics with corresponding ground truth and mask images; best match for paper texture experiments | https://mosaic.utia.cas.cz/index.php?act=bench_form |
| `medical` | BrainWeb Simulated Brain MRI Database | MRI-like volumes with anatomical labels; stable first medical source before large clinical archives | https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1 |
| `cartoon` | Berkeley Segmentation Dataset and Benchmark | Natural-image public substitute with human segmentations; useful for smooth-region/cartoon-like checks but not original paper cartoon data | https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/ |

Backup sources:

- TCIA Pretreat-MetsToBrain-Masks: real brain MRI segmentations, but larger and tumor/metastasis-specific.
- Weizmann Segmentation Evaluation Database: small natural-image segmentation backup with stricter research-use terms.

Expected local layout:

```text
reproduce/data/iterated_rof/
  cartoon/
    images/
    masks/
  texture/
    images/
    masks/
  medical/
    images/
    masks/
```

Mask files are optional for the first data audit but required before claiming quantitative paper-like reproduction. A mask must share the image's relative path and extension under `masks/`; same-stem files with a different extension or in a different subdirectory are reported as warnings and are not used for quantitative metrics. Without masks, only qualitative figures and runtime can be reported.

## Code Style

Use small pure functions for scan / metric / runner logic, and keep dashboard sync separate.

```python
def scan_dataset(root, family):
    image_count = count_images(root / family / "images")
    mask_count = count_images(root / family / "masks")
    return {
        "family": family,
        "image_count": image_count,
        "mask_count": mask_count,
        "status": "ready" if image_count and mask_count else "missing_or_incomplete",
    }
```

## Testing Strategy

- Unit tests for readiness scanning must use temporary directories, not committed data.
- Unit tests for image loading and optional mask loading use synthetic tempfile PNGs.
- Tests must prove missing data is reported as a blocker, not silently converted into a completed experiment.
- Existing partial tests remain the guard for T-ROF math behavior.
- Full project validation must pass before dashboard changes.

## Boundaries

- Always: Keep `paper-level-completed = 0 / 15` unless original/equivalent data and baselines exist, and every independent paper-level table/baseline/parameter/data-source row has a readable audited artifact with matching SHA-256, substantive content, and no placeholder/fixture text.
- Always: Preserve `fidelityWarning` when data or baselines are incomplete.
- Always: Keep generated JSON and dashboard fields synchronized when a run is promoted into `run_all.py`.
- Ask first: Download external datasets, add new Python dependencies, or commit nontrivial image datasets.
- Never: Use synthetic-only output to claim paper-like or paper-level reproduction.
- Never: Raise dashboard level without generated run evidence.

## Success Criteria

First slice:

- [x] A spec exists for #3 paper-like reproduction.
- [x] A local data README documents expected data layout.
- [x] A readiness-audit script reports missing data explicitly.
- [x] Tests cover the readiness audit.
- [x] Existing partial reproduction and dashboard validation still pass.
- [x] Candidate source manifest exists and records download policy / license notes.

Paper-like milestone:

- [ ] `cartoon`, `texture`, and `medical` each have at least one local image.
- [ ] Every family with local images has a source entry in `dataset_manifest.json` with known `source_id` from `reproduce/paper_like/iterated_rof_dataset_sources.json`, citation, license/provenance notes that are not fixture/tempfile placeholders, `license_reviewed=true`, `provenance_reviewed=true`, explicit `synthetic_fixture=false`, and a structured `source_audit` with reviewed source URL matching the selected registry entry, download date, source artifact path/hash and license snapshot path/hash under that family's `audit/` directory, conversion notes, and local-file mapping review.
- [ ] Source audit artifacts are substantive local review evidence, not tiny placeholder/test-fixture files, include the manifest source URL, a valid date, non-empty reviewer-note and conversion-or-mapping fields in the artifact text, and `--check-source-audit-artifact-claims` reports `content_status=review_evidence_present`; if not, the report exposes `content_size_bytes`, `min_content_size_bytes`, `content_issue_codes`, `content_issues`, and `placeholder_pattern_hits` for review.
- [ ] Every local image has a file-level `files[]` manifest claim with matching `sha256`; masks are bound to the same relative path and extension with matching `mask_sha256` when present.
- [ ] Every family has masks or labels for quantitative Dice / accuracy.
- [ ] `--refresh-manifest-file-claims` has refreshed file paths and hashes from local data.
- [ ] `--check-manifest-file-claims` passes without rewriting `dataset_manifest.json`.
- [ ] T-ROF runs on real/local images with parameter records.
- [ ] Generated summaries include `dataset_fingerprint` for the exact local image/mask snapshot, and promotion checks verify it against per-image evidence.
- [ ] Input images are large enough and nontrivial enough for paper-like evidence, not tiny binary smoke-test fixtures.
- [ ] Completed image rows include image, mask, figure, and figure-evidence sidecar SHA-256 / size evidence that still matches local disk.
- [ ] Source summary verification points at the canonical local manifest; dashboard sync rejects missing/mismatched canonical citation/license/provenance text, non-image evidence, or too-small image/mask evidence even if hashes are self-consistent.
- [ ] Dashboard promotion cannot be produced by self-filled summary/manifest fields alone; candidate verification rechecks source IDs against `reproduce/paper_like/iterated_rof_dataset_sources.json`, reloads the canonical manifest, and rescans current image/mask evidence.
- [ ] Baselines include direct threshold / clustering and Otsu or Chan-Vese proxy.
- [ ] Figures include the required evidence panels: input, ROF solution, T-ROF labels, raw K-means, Multi-Otsu or quantile fallback, T-ROF error/difference, and T-ROF-vs-Otsu/quantile difference.
- [ ] `paper_like_gate.passed` is true in the generated local runner summary.
- [ ] `--strict-paper-like` passes, proving both the gate and dashboard promotion candidate are promotable.
- [ ] Static validation is intentionally run with `ALLOW_PAPER_LIKE=1` for the promotion review; without that flag it must keep `paper-like = 0 / 15`.
- [ ] Dashboard remains truthful and says `paper-like` only if the project defines and validates that level.

Paper-level milestone:

- [ ] Original paper images or author-approved/equivalent data are available.
- [ ] Main paper figures and tables are reproduced or explicitly marked unavailable.
- [ ] Parameter settings and stopping criteria are documented.
- [ ] Full baseline comparison is implemented.

## Task List

### Phase 1: Foundation

- [x] Task 1: Add paper-like spec and data layout README.
  - Acceptance: Documentation states data families, layout, commands, and claim boundaries.
  - Verify: `rg -n "paper-like|iterated_rof" reproduce/paper_like reproduce/data`.
  - Files: `reproduce/paper_like/iterated_rof_spec.md`, `reproduce/data/iterated_rof/README.md`

- [x] Task 2: Add readiness audit script.
  - Acceptance: Script writes a JSON report and exits successfully even when data is missing.
  - Verify: `python3 reproduce/experiments/iterated_rof_paper_like.py`
  - Files: `reproduce/experiments/iterated_rof_paper_like.py`

- [x] Task 3: Add readiness audit tests.
  - Acceptance: Tests cover missing data and ready temporary data.
  - Verify: `python3 -m unittest reproduce.tests.test_iterated_rof_paper_like_scaffold`
  - Files: `reproduce/tests/test_iterated_rof_paper_like_scaffold.py`

- [x] Task 4: Add candidate source manifest.
  - Acceptance: Manifest includes Prague, BrainWeb, BSDS500, and backup sources with URL, local target, and redistribution warnings.
  - Verify: `python3 reproduce/experiments/iterated_rof_paper_like.py --sources`
  - Files: `reproduce/paper_like/iterated_rof_dataset_sources.json`, `reproduce/experiments/iterated_rof_paper_like.py`

### Phase 2: Real Data Runner

- [x] Task 5: Add local image loading and grayscale normalization.
  - Acceptance: Local images can be loaded from each family without changing `run_all.py`.
  - Verify: `python3 -m unittest reproduce.tests.test_iterated_rof_paper_like_scaffold`

- [ ] Task 6: Run T-ROF and baselines on each available family.
  - Acceptance: Produces per-family metrics and figures.
  - Current slice: ROF + T-ROF JSON summary exists for local images; rows with masks report clustering accuracy and binary Dice when applicable, rows without masks are marked `qualitative_only`, raw K-means and Multi-Otsu baselines are recorded, and comparison figures include difference panels.
  - Provenance slice: Readiness and run summaries include local dataset manifest status, `claim_blockers`, per-image file size/SHA-256 evidence, and file-level source claims whose manifest hashes match local files.
  - Gate slice: `paper_like_gate` requires completed quantitative outputs for all three families, canonical local data-root evidence, no readiness/runner/source blockers, a schema-valid source registry, dataset fingerprint evidence, numeric T-ROF and baseline clustering-accuracy metrics, and per-image baselines/generated-figure file evidence/input file evidence/source claims with reviewed license/citation details, explicit `synthetic_fixture=false`, no obvious fixture/tempfile/scaffold placeholder text, matching hashes, decodable nontrivial image/mask files, nonblank figures, and a dashboard-checkable `evidence_summary`; it also emits a structured checklist grouped by requirement.
  - Summary slice: `family_summaries` aggregates per-family counts, mean metrics, mean baseline metrics, figure paths, source claims, and runner errors for paper-like tables; dashboard promotion recomputes this table from per-image evidence rather than trusting saved summary rows.
  - Fingerprint slice: `dataset_fingerprint` records a SHA-256 digest over sorted local image/mask file hashes so a report can be tied to one data snapshot; gate, candidate, and audit outputs verify it against per-image evidence.
  - CSV slice: `--run` writes `family_summaries` to `iterated_rof_paper_like_family_summary.csv`, with flattened metric and baseline columns.
  - Evidence CSV slice: `--run` writes one `images` row per local runner image to `iterated_rof_paper_like_image_evidence.csv`, including gate/fingerprint context, paths, metrics, baselines, generated figure hash/size and sidecar evidence, input/mask hashes, source claims, solver parameters, thresholds, and errors.
  - Dashboard-candidate slice: `--dashboard-candidate-output` writes a gated JSON candidate for future dashboard edits; it stays blocked unless `paper_like_gate.passed=true`, family summaries/counts match per-image evidence, the current on-disk manifest still matches, and protocol/figure-root checks pass.
  - Promotion-audit slice: `--promotion-audit-output` writes a compact reviewer/CI summary of checklist status, blocker counts, promotion shape blockers, family states, manifest status, source-audit artifact/path/hash status, data fingerprint, and the same source-summary artifact requirement used by dashboard candidates.
  - Saved-summary verification slice: `--verify-summary <summary.json>` reloads an existing run summary, recomputes `paper_like_gate` and promotion blockers, can write candidate/audit outputs, and exits non-zero when the saved summary is not promotable.
  - Protocol slice: `run_protocol` records solver, threshold formula, tolerances, iteration limits, seed, and promotion rule in the run summary and dashboard candidate; promotion rejects forged static protocol fields and noncanonical figure directories.
  - Fixture E2E slice: tests exercise the real CLI runner, strict gate, saved-summary verification, dashboard candidate, and promotion audit on temporary three-family PNG fixtures. This proves the infrastructure path without treating fixture output as dashboard evidence.
  - Manifest helper slice: `--refresh-manifest-file-claims` updates local `files[]` image/mask paths and hashes without claiming license review or silently deleting stale claims; `--check-manifest-file-claims` verifies the same claims without writing, exits non-zero when they are stale, and reports concrete stale `{family, image}` records.
  - Remaining: Populate real local cartoon/texture/medical data and keep dashboard promotion blocked until data-backed outputs exist.
  - Verify: JSON includes solver, mu, thresholds, runtime, baselines, figure paths and figure file evidence, input/mask file evidence, source claims, metric fields where masks exist, `run_protocol`, `family_summaries`, and `paper_like_gate`; family CSV includes one row per family; image evidence CSV includes one row per local runner image; dashboard candidate reports `can_promote` from the gate and includes `run_protocol`; promotion audit summarizes the same candidate-gated state without dashboard edits.

### Phase 3: Dashboard Promotion

- [ ] Task 7: Promote paper-like results into dashboard only after data-backed run exists.
  - Acceptance: `sync_to_dashboard.mjs --check` and `validate.mjs` pass.
  - Verify: Full validation commands pass.

## Open Questions

- Do we have rights to commit any real image data, or should all data stay local under `reproduce/data/` and be gitignored?
- Should paper-like results use public replacements only, or should we try to locate original paper figures/data first?
- Which baseline should be first: Otsu multi-threshold, Chan-Vese proxy, or graph cut?
