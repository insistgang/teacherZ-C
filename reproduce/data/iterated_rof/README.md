# Iterated ROF Data Drop Zone

This directory is for local data used to move paper #3, `iterated-rof`, from `partial` toward paper-like reproduction.

Do not claim paper-like or paper-level reproduction from synthetic-only data.

Expected layout:

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

Rules:

- `images/` contains input images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`).
- `masks/` contains segmentation masks or labels when available.
- Masks are required for quantitative Dice / accuracy.
- Images without masks can only support qualitative figures and runtime checks.
- Large or license-restricted datasets should stay local and should not be committed.

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

The machine-readable source manifest is:

```text
reproduce/paper_like/iterated_rof_dataset_sources.json
```
