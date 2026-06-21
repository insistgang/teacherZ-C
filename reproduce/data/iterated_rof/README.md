# Iterated ROF Data Drop Zone

This directory is for local data used to move paper #3, `iterated-rof`, from `partial` toward paper-like reproduction.

Do not claim paper-like or paper-level reproduction from synthetic-only data.

Expected layout:

```text
reproduce/data/iterated_rof/
  cartoon/
    audit/
    images/
    masks/
  texture/
    audit/
    images/
    masks/
  medical/
    audit/
    images/
    masks/
```

Rules:

- `images/` contains input images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`).
- `masks/` contains segmentation masks or labels when available.
- `audit/` contains the local source artifact and license/terms snapshot evidence for that family.
- Mask files must use the same relative path and extension as their matching image, for example `images/a/sample.png` pairs with `masks/a/sample.png`.
- Masks are required for quantitative Dice / accuracy.
- Images without masks can only support qualitative figures and runtime checks.
- Large or license-restricted datasets should stay local and should not be committed.
- `images/`, `masks/`, and `audit/` are ignored by git to avoid accidental dataset, source-artifact, or license-snapshot commits.
- Copy `dataset_manifest.template.json` to `dataset_manifest.json` after adding local data, then replace every template note with the actual source, citation, license note, provenance note, and structured `source_audit`, set `license_reviewed` / `provenance_reviewed` only after terms and file origin are checked, keep `synthetic_fixture=false` for real data, and add one `files[]` claim for every local image.

Local source manifest:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --prepare-data-layout
```

This creates the `images/` / `masks/` / `audit/` directories and copies
`dataset_manifest.template.json` to `dataset_manifest.json` if no local
manifest exists. It does not download data and does not overwrite an existing
manifest.

If you have already downloaded and converted files outside the repository,
place them in a temporary drop directory with the same family layout and ingest
them without overwriting existing canonical files:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --review-data-drop /path/to/iterated_rof_drop --data-drop-review-output /tmp/iterated_rof_data_drop_review.json
python3 reproduce/experiments/iterated_rof_paper_like.py --ingest-data-drop /path/to/iterated_rof_drop
```

The drop layout is:

```text
/path/to/iterated_rof_drop/
  cartoon/{images,masks,audit}/...
  texture/{images,masks,audit}/...
  medical/{images,masks,audit}/...
```

The review command is a no-write dry run. It reports which files would copy,
which canonical files are already current, unsupported image extensions, staging
symlink/path escapes, and overwrite conflicts. It exits non-zero on conflicts so
reviewers can fix the drop before touching the canonical data root.

The ingest command copies supported image/mask files and audit artifacts into
`reproduce/data/iterated_rof/`, creates the manifest template when needed, and
refreshes `files[]` image/mask SHA-256 claims. It does not download data, does
not overwrite conflicting target files, and does not set
`license_reviewed=true`, `provenance_reviewed=true`, citation, license text, or
audit review fields. Those remain manual review gates before any paper-like
promotion.

After placing local files, refresh the file-level hash claims:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --refresh-manifest-file-claims
```

This updates `files[]` image/mask paths and SHA-256 values from local files. It
does not set `license_reviewed`, `provenance_reviewed`, citation, license text,
or provenance notes; those still require manual source review. It also does not
delete stale `files[]` claims for local images that were removed; those are
reported for manual review so provenance records are not silently discarded.

For CI or review workflows, check that file-level claims are current without
rewriting the manifest:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --check-manifest-file-claims
```

This exits non-zero if `dataset_manifest.json` is missing, invalid, has stale
`files[]` entries, or would change after `--refresh-manifest-file-claims`.
When stale file claims exist, the JSON report includes concrete
`stale_file_claims` entries with `family` and `image` keys.

After placing source audit artifacts and license snapshots, refresh their
SHA-256 claims:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --refresh-source-audit-artifact-claims
```

This updates `source_artifact_sha256` and `license_snapshot_sha256` from local
files referenced by `source_audit`, including file-level `files[].source_audit`
overrides. It does not set `license_reviewed`, `provenance_reviewed`,
`downloaded_at`, citation, license text, or provenance notes. Check the same
artifact hashes without writing via:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --check-source-audit-artifact-claims
```

The data-gap checklist and promotion audit also include a per-family
`source_audit` matrix. Use it to see whether each family's `audit/` directory
has the reviewed source artifact and license snapshot files under the expected
root, and whether their SHA-256 values match the manifest. If a `files[]` entry
overrides `source_audit`, the matrix reports that file-level audit status under
the same family so a complete family-level claim cannot hide a broken file-level
override. `--check-source-audit-artifact-claims` also reports stale when a
family has a manifest entry but no structured `source_audit`; a hash check that
finds no artifact paths is not treated as source review evidence.

`dataset_manifest.json` is included in readiness and run summaries. If local images exist but this manifest is missing, invalid, uses an unknown `source_id`, lacks `license_reviewed=true` or `provenance_reviewed=true`, omits `citation` / `license_note` / `provenance_note`, lacks a structured `source_audit`, does not explicitly set `synthetic_fixture=false`, contains obvious fixture/tempfile text or copied template placeholder text in source notes, lacks a matching `files[]` entry for any local image/mask pair, or has a `sha256` / `mask_sha256` that does not match the local file, the runner still executes but reports `claim_blockers` so the result cannot be promoted as paper-like.

`source_audit` must include a reviewed `source_url` matching the selected
`source_id` entry's registry `url` or `download_url`, `downloaded_at` in
`YYYY-MM-DD` format, `source_artifact_path` plus
`source_artifact_sha256`, `license_snapshot_path` plus
`license_snapshot_sha256`, non-empty `conversion_notes`, and
`local_file_mapping_reviewed=true`. The artifact paths must point under that
family's `reproduce/data/iterated_rof/<family>/audit/` directory, for example
`cartoon/audit/source-artifact.ext` and
`cartoon/audit/license-snapshot.html`. Relative paths in `dataset_manifest.json`
are resolved from the local data root, not from the repository root. Use those files for the raw
download/archive or source-file bundle evidence and a saved snapshot of the
license or terms page used for review. The runner and dashboard promotion sync
re-read those local files and reject paths outside the family audit root or
mismatched SHA-256 values. Audit artifacts must also contain enough concrete
review evidence to be useful: tiny placeholder files, test-fixture text, or
copied "reviewed source artifact" / "reviewed license snapshot" stubs are
reported as incomplete even when their SHA-256 values match. A substantive
artifact also needs structured review content in the file itself: the manifest
`source_url` or matching `source_url=...`, a valid `YYYY-MM-DD`
review/download date, a non-empty `reviewer_note=` or equivalent source-review
field, and a non-empty `conversion_note=` / `local_file_mapping=` or equivalent
mapping field.
The readiness, data-gap, and `--check-source-audit-artifact-claims` reports
include `content_status`, `content_size_bytes`, `min_content_size_bytes`,
`content_issue_codes`, `content_issues`, and `placeholder_pattern_hits` so a
reviewer can see why a matched artifact is still not acceptable.

File entries use this shape:

```json
{"image": "sample.png", "sha256": "...", "mask": "sample.png", "mask_sha256": "..."}
```

Readiness and run summaries also include `data_ready_status`,
`data_ready_blocker_count`, and `paper_like_gate`. The legacy top-level
`status` records the broad runner stage, while `data_ready_status` is the
machine-readable data package gate: it is `ready_for_local_runner` only when the
data-gap checklist has no blockers. In data-gap JSON,
`ready_for_paper_like_runner_outputs` means the recomputed local runner gate
passes; `ready_for_dashboard_promotion` remains false there because dashboard
promotion also requires a bound source summary artifact and candidate overlay
validation. Treat `paper_like_gate` as necessary but not sufficient for dashboard
promotion: a local run can be `completed_local_runner` while
`paper_like_gate.passed` remains false because data families, masks, source
claims, source/license/provenance review details, source claim hashes, dataset
fingerprint, quantitative T-ROF/baseline metrics, generated figure file
evidence, or baselines are incomplete.
For completed image rows, the gate re-reads local image, mask, and generated
figure paths and verifies that recorded SHA-256 / size evidence still matches
disk before allowing dashboard promotion.
Rows must stay inside the canonical layout: images under
`reproduce/data/iterated_rof/<family>/images/`, masks under the matching
relative path in `<family>/masks/`, and generated figures as PNG files under the
configured figure directory. The recorded file-evidence `path` must resolve to
the same path as the report row, and each `source_id` must exist in the source
registry. The registry itself is validated before promotion: entries must use
the matching `target_family`, non-empty source/URL/license/local-layout fields,
unique `source_id` values, and integer `priority` values.
Each generated figure also writes a sibling `*.png.evidence.json` sidecar. The
gate checks that sidecar against the report row's image/mask/figure hashes,
panel list, solver parameters, thresholds, metrics, and baseline method
metadata.
Promotion checks also re-open image and mask files with the local image reader
and reject generated figures that are too small or visually blank. File hashes
and matching sidecars are necessary evidence, but not sufficient by themselves.
The gate also rejects input images that are tiny, visually blank, or have too
few gray levels, so small binary smoke-test fixtures cannot be promoted as
paper-like evidence even if their manifest claims look reviewed.
The gate includes both flat `reasons` and a structured `checklist` grouped by
canonical data root, readiness/provenance, runner outputs, and output evidence.
It also includes `evidence_summary`, which records the gate id, dataset
fingerprint, completed/quantitative image counts, required family coverage,
source-claim count, and figure-evidence count for dashboard sync checks.
Dashboard sync recomputes those counts from the source summary image rows and
also checks that each `family_summaries` row's figure paths and source claims
match the same-family image evidence rows exactly.

Run summaries include `family_summaries`, which is the intended table source for
paper-like reporting. It aggregates each family into counts, mean T-ROF metrics,
mean baseline metrics, figure paths, source claims, and errors.
Run summaries also include `images`, the per-image evidence source for reviewer
audits. Use the image table to inspect each local file, mask, metric, baseline,
figure path/hash/size, figure-evidence sidecar hash, source claim, and runner
error before any promotion decision.
They also include `dataset_fingerprint`, a SHA-256 digest over the sorted local
image/mask file hashes. Use it to tie a report or dashboard candidate to one
specific local data snapshot; paper-like promotion re-computes this value from
per-image evidence rows and blocks stale or arbitrary fingerprints. It does not
replace source, citation, or license review.

With `--run`, the family table and image-evidence table are written as CSV:

```text
reproduce/results/iterated_rof_paper_like_family_summary.csv
reproduce/results/iterated_rof_paper_like_image_evidence.csv
```

Use `--family-summary-output <path>` to write the CSV elsewhere.
Use `--image-evidence-output <path>` to write the image evidence CSV elsewhere.
Formula-like text values from manifests or errors are prefixed with `'` in CSV
outputs so reviewer spreadsheets do not execute them as formulas.

To prepare a gated dashboard-edit draft without modifying the dashboard, pass:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --run --dashboard-candidate-output /tmp/iterated_rof_dashboard_candidate.json --dashboard-static-assets-output /tmp/iterated_rof_dashboard_static_assets.json
```

The candidate remains `can_promote=false` until `paper_like_gate.passed=true`
and the report has the expected local-runner shape (`completed_local_runner`,
`ready_for_paper_like_runner`, present local manifest, and completed
quantitative image evidence rows). Candidate generation recomputes family
summaries, image counts, result files, and run metrics from per-image evidence;
re-loads the current on-disk `dataset_manifest.json`; and blocks forged
protocol fields or noncanonical figure directories. Even when promotable, its
`reproductionTruthLevel` remains `partial-completed`; `paper-like` is not
paper-level evidence.
Candidate and audit generation recompute `paper_like_gate` from the current
summary evidence instead of trusting a saved `paper_like_gate.passed` value.
They also rescan the current canonical data root and block promotion when the
on-disk image/mask fingerprint differs from the report, when any current local
image lacks a matching manifest `files[]` claim, or when a saved per-image
`source_claim` no longer matches the effective current manifest claim.
Promotable candidates include `candidateDetails.paper_like_verification`, and
dashboard sync requires that runner-generated verification metadata before any
future `paper-like` result can be accepted. That verification includes the
source summary path plus SHA-256 hash; dashboard sync re-reads the summary,
checks that its `paper_like_gate` and dataset fingerprint match the run result,
checks that completed quantitative image rows include matching masks and
image/mask/figure/sidecar file evidence, requires the canonical local
`dataset_manifest.json`, rejects missing or mismatched canonical
citation/license/provenance text, rejects non-image file signatures or
too-small image/mask evidence even when SHA-256 values are self-consistent,
requires `resultFiles` to live under
`docs/assets/repro/iterated_rof_paper_like/...` and match the SHA-256 of the
source summary figure evidence, and rejects candidates that only contain a
self-claimed gate. They also include
`runResultPatch` for `reproduce/results/repro_results.json` and
`dashboardDetailPatch` for `docs/js/reading-data.js`, so reviewers can apply the
same gate, metrics, result files, and verification metadata consistently.
The optional static asset manifest records the source runner figure, derived
`assets/repro/iterated_rof_paper_like/...` result file, source SHA-256, static
asset SHA-256 if present, and `missing` / `stale` / `current` status for every
paper-like figure. Add `--copy-dashboard-static-assets` only after the candidate
is promotable; blocked candidates write a blocked manifest and do not copy
figures into `docs/assets/repro`.
Before applying those patches, validate the candidate against the current
dashboard, run-result JSON, and static asset snapshots:

```bash
ALLOW_PAPER_LIKE=1 node reproduce/sync_to_dashboard.mjs --candidate /tmp/iterated_rof_dashboard_candidate.json --check
```

This exits non-zero for blocked candidates, for candidate metadata that drifts
from its generated patches, and for promotable-looking candidates whose patches
would fail after being embedded into the current dashboard, run-result JSON, and
static asset snapshots.
Programmatic shape-only checks are useful for unit tests, but they are not
promotion evidence without this current-snapshot overlay validation.
Use `ALLOW_PAPER_LIKE=1` only for this reviewed data-backed candidate overlay
check; ordinary dashboard validation intentionally keeps `paper-like = 0 / 15`
until the promotion patches are accepted.
To verify a saved run summary without rerunning ROF/T-ROF, pass:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --verify-summary reproduce/results/iterated_rof_paper_like_summary.json --promotion-audit-output /tmp/iterated_rof_promotion_audit.json
```

This writes a verification JSON with a recomputed gate plus promotion-shape
blockers and exits non-zero unless the recomputed dashboard candidate is
promotable.
For a compact reviewer/CI summary of the same promotion state, pass:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --run --promotion-audit-output /tmp/iterated_rof_promotion_audit.json
```

The audit records gate checklist status, blocker counts, family status counts,
manifest status, dataset fingerprint, and the same promotion shape blockers
used by the dashboard candidate. It can recommend `paper-like` only when the
same source summary artifact under `reproduce/results/` is present, readable,
and bound to the current report; a gate-passing in-memory report without that
artifact remains blocked. It also includes `ready_for_local_runner` and
`data_ready_blockers` from a freshly recomputed data gap checklist, so reviewers
can distinguish missing/invalid local data from failed promotion evidence
without trusting stale saved checklist fields. It does not modify the dashboard
or relax any gate.
To write a compact local-data acquisition checklist for the next manual data
drop, pass:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --data-gap-output /tmp/iterated_rof_data_gap.json
```

The checklist records each family path, primary source candidate, an
`acquisition_plan` with target local paths, required manifest/source-audit
fields, and post-download commands, missing image or mask requirements,
manifest claim blockers, paper-like gate reason counts, preflight content
issues, and next actions. The content preflight checks whether local
images/masks are decodable, whether masks match image shape, and whether input
images are large and nontrivial enough for paper-like evidence. It is an audit
artifact only; each write recomputes the checklist from current data and
manifest state instead of copying any saved checklist in a summary. It does not
download data, modify the manifest, or change dashboard status.
When used without `--output`, `--data-gap-output` writes only the compact
checklist and does not refresh the default readiness JSON in `reproduce/results`.
For an operator-facing review of a local data package after ingest, pass:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --data-package-review-output /tmp/iterated_rof_data_package_review.json
```

This no-download report recomputes the current family image/mask status,
manifest `files[]` claim check, source-audit artifact claim check, preflight
content issues, and the manual manifest fields still missing per family
(`license_reviewed`, citation, provenance notes, `source_audit.downloaded_at`,
artifact paths/SHA-256 values, and related review flags). When used without
`--output`, it writes only the package-review JSON and does not refresh the
default readiness report. A clean package review only means the local data is
ready to enter the runner; dashboard promotion still requires a generated
source-summary-backed candidate.
For a cheap CI gate before running ROF/T-ROF, add `--strict-data-ready`. It
exits non-zero unless the data gap checklist's `ready_for_local_runner` flag is
true: all three families have nontrivial images, matching masks, a present
reviewed manifest, valid file-level claims, and clean preflight content checks.
This is still not a paper-like claim; it only proves the local data package is
ready to enter the runner.
For CI or dashboard-promotion checks, add `--strict-paper-like`; it exits
non-zero unless the recomputed gate passes and the generated dashboard
candidate is promotable. A run can have `paper_like_gate.passed=true` and still
fail this strict check if promotion-shape requirements such as canonical figure
paths or source summary evidence are not satisfied.

Readiness audit:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py
python3 reproduce/experiments/iterated_rof_paper_like.py --sources
```

The audit writes:

```text
reproduce/results/iterated_rof_paper_like_readiness.json
```

Recommended first sources:

| Family | Source | URL | Local target |
|---|---|---|---|
| texture | Prague Texture Segmentation Benchmark | https://mosaic.utia.cas.cz/index.php?act=bench_form | `texture/images`, `texture/masks` |
| medical | BrainWeb Simulated Brain MRI Database | https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1 | `medical/images`, `medical/masks` |
| cartoon | Berkeley Segmentation Dataset and Benchmark (public substitute) | https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/ | `cartoon/images`, `cartoon/masks` |

Backup sources:

- TCIA Pretreat-MetsToBrain-Masks for larger real brain MRI segmentation data.
- Weizmann Segmentation Evaluation Database for small natural-image segmentation, with stricter research-use terms.

Source-specific landing checklist:

| Source id | Image selection | Mask mapping | Final naming example |
|---|---|---|---|
| `prague-texture` | Choose reviewed texture mosaics that include ground-truth labels. | Convert each paired ground-truth label image to an integer PNG with identical shape. | `texture/images/<source_id>/<case>.png` and `texture/masks/<source_id>/<case>.png` |
| `brainweb` | Choose one simulated MRI volume plus its matching anatomical/label volume, then select representative 2D slices. | Export the same slice from the anatomical labels as an integer PNG. | `medical/images/brainweb/<volume>_slice_<index>.png` and `medical/masks/brainweb/<volume>_slice_<index>.png` |
| `tcia-pretreat-metstobrain-masks` | Choose a matched image/segmentation case after TCIA terms and citation review. | Convert the matched segmentation slice to the same relative PNG path as the image slice. | `medical/images/tcia/<case>_slice_<index>.png` and `medical/masks/tcia/<case>_slice_<index>.png` |
| `bsds500` | Choose BSDS images with smooth/cartoon-like regions and available human segmentation ground truth. | Convert one reviewed human segmentation to an integer label PNG. | `cartoon/images/bsds500/<image_id>.png` and `cartoon/masks/bsds500/<image_id>.png` |
| `weizmann-segmentation-db` | Choose reviewed image/ground-truth pairs only after research-use terms review. | Convert each reviewed segmentation to a label PNG with identical dimensions. | `cartoon/images/weizmann/<case>.png` and `cartoon/masks/weizmann/<case>.png` |

The same rules are also available in machine-readable form through
`--data-gap-output`: each family row includes `acquisition_plan.image_selection_rule`,
`mask_mapping_rule`, `conversion_checklist`, and `final_naming_example`.
For strict paper-like, all three families need matching masks and completed
quantitative outputs; qualitative-only images are useful for local inspection but
cannot promote the dashboard.

The machine-readable source manifest is:

```text
reproduce/paper_like/iterated_rof_dataset_sources.json
```
