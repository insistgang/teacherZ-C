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
```

The audit writes:

```text
reproduce/results/iterated_rof_paper_like_readiness.json
```
