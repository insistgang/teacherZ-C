# 论文核心算法实现

基于 D:\Documents\zx 项目中精读的核心论文，实现可运行的代码。

## 模块列表

| 模块 | 论文 | 描述 |
|:---|:---|:---|
| `slat_segmentation.py` | SLaT三阶段分割 (JSC 2017) | Smoothing-Lifting-Thresholding |
| `rof_iterative_segmentation.py` | 迭代ROF分割 (TIP 2013) | 多类分割 + 标签树 |
| `tucker_decomposition.py` | Sketching Tucker (SIAM JSC) | 随机Sketching + HOOI |
| `neural_varifold.py` | Neural Varifolds (TPAMI 2022) | 点云神经表示 |

## 快速开始

```python
# 1. SLaT 三阶段分割
from slat_segmentation import SLATSegmentation
segmenter = SLATSegmentation(lambda_param=1.0)
result = segmenter.segment(image, K=4)

# 2. ROF 迭代分割
from rof_iterative_segmentation import IterativeROFSegmentation
segmenter = IterativeROFSegmentation(lambda_param=0.1)
seg = segmenter.segment(gray_image, K=3)

# 3. Tucker 分解
from tucker_decomposition import SketchingTucker, reconstruct_tucker
sketchy = SketchingTucker(ranks=[5, 4, 3])
core, factors = sketchy.fit(tensor)
reconstructed = reconstruct_tucker(core, factors)

# 4. Neural Varifold
from neural_varifold import NeuralVarifoldNet
net = NeuralVarifoldNet(num_classes=10)
logits = net(positions, normals)
```

## 依赖

```
numpy
opencv-python
scipy
scikit-learn
torch
```

## 运行演示

```bash
cd implementations
python usage_examples.py
```
