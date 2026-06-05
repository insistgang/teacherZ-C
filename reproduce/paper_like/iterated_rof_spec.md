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
- At least two baselines are reported: direct threshold / clustering and one classical segmentation baseline such as Otsu or Chan-Vese-style proxy.
- Metrics and figures are produced for each dataset family.
- The dashboard remains honest: `partial` until data-backed paper-like results exist.

## Tech Stack

- Python 3.9+
- Existing dependencies: `numpy`, `scipy`, `matplotlib`
- Optional later dependency: `scikit-image` for public sample images, Otsu thresholds, and image IO if local image loading needs more formats
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
python3 reproduce/experiments/iterated_rof_paper_like.py
python3 reproduce/experiments/iterated_rof_paper_like.py --sources
```

Future paper-like run, after data exists:

```bash
python3 reproduce/experiments/iterated_rof_paper_like.py --run
```

## Project Structure

```text
reproduce/experiments/sat_rof_trof.py
  Current partial runner shared by #1, #2, #3.

reproduce/experiments/iterated_rof_paper_like.py
  Paper-like readiness audit and future dataset-backed runner for #3 only.

reproduce/data/iterated_rof/
  Local data drop zone. Data is not committed unless it is small and license-safe.

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

Mask files are optional for the first data audit but required before claiming quantitative paper-like reproduction. Without masks, only qualitative figures and runtime can be reported.

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
- Tests must prove missing data is reported as a blocker, not silently converted into a completed experiment.
- Existing partial tests remain the guard for T-ROF math behavior.
- Full project validation must pass before dashboard changes.

## Boundaries

- Always: Keep `paper-level-completed = 0 / 15` unless original/equivalent data and baselines exist.
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
- [ ] At least one family has masks or labels for quantitative Dice / accuracy.
- [ ] T-ROF runs on real/local images with parameter records.
- [ ] Baselines include direct threshold / clustering and Otsu or Chan-Vese proxy.
- [ ] Figures compare input, ROF solution, T-ROF labels, baselines, and differences.
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

- [ ] Task 5: Add local image loading and grayscale normalization.
  - Acceptance: Local images can be loaded from each family without changing `run_all.py`.
  - Verify: Manual run on one local image.

- [ ] Task 6: Run T-ROF and baselines on each available family.
  - Acceptance: Produces per-family metrics and figures.
  - Verify: JSON includes solver, mu, thresholds, runtime, and result files.

### Phase 3: Dashboard Promotion

- [ ] Task 7: Promote paper-like results into dashboard only after data-backed run exists.
  - Acceptance: `sync_to_dashboard.mjs --check` and `validate.mjs` pass.
  - Verify: Full validation commands pass.

## Open Questions

- Do we have rights to commit any real image data, or should all data stay local under `reproduce/data/` and be gitignored?
- Should paper-like results use public replacements only, or should we try to locate original paper figures/data first?
- Which baseline should be first: Otsu multi-threshold, Chan-Vese proxy, or graph cut?
