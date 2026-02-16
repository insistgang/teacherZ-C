# Xiaohao Cai Algorithms

Image analysis algorithms based on Xiaohao Cai research.

## Installation

```bash
pip install -e .
```

## Quick Start

### ROF Denoising

```python
from xcai.denoising import ROFDenoiser

denoiser = ROFDenoiser(lambda_param=0.1, method='chambolle')
clean_image = denoiser.denoise(noisy_image)
```

### SLaT Segmentation

```python
from xcai.segmentation import SLatSegmenter

segmenter = SLatSegmenter(n_classes=3)
labels = segmenter.segment(image)
```

### Tucker Decomposition

```python
from xcai.tensor import TuckerDecomposer

decomposer = TuckerDecomposer(rank=[10, 10, 10])
core, factors = decomposer.decompose(tensor)
```

## Modules

- `xciai.denoising`: ROF-based image denoising algorithms
- `xciai.segmentation`: SLaT and GraphCut segmentation methods
- `xciai.tensor`: Tensor decomposition methods (Tucker, TT, CUR)
- `xciai.pointcloud`: Varifold-based point cloud analysis
- `xciai.peft`: Tensor CUR LoRA for parameter-efficient fine-tuning
- `xciai.utils`: Common utilities for image processing and metrics

## License

MIT License
