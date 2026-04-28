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
- `partial`：复现论文核心算法路线的一部分，例如 SaT smoothing + thresholding、SLaT RGB+Lab、graph smoothing。
- `paper-level`：接近论文实验设置。当前没有把任何重依赖论文标成 paper-level。
- `assessment-only`：只做难度评估，不运行实验。当前 15 篇都至少有 toy 或 partial 实验。

## 哪些不是 full reproduction

以下方向的 full reproduction 需要真实数据、专门库或长时间运行，本仓库只提供 toy/partial 演示：

- Tight-frame vessel：需要真实 2D/3D MRA 数据和严格 tight-frame/DCWT 实现。
- Wavelet sphere：需要 S2LET/SSHT/SO3 等球面小波栈和球面数据。
- RI UQ I：需要 radio interferometric operators、大规模 MCMC、诊断与真实 RI 数据。
- Online RI：需要大规模 visibility streams 才能接近论文实验。
- Proximal Nested Sampling：需要 constrained proximal MCMC、high-dimensional imaging benchmarks 和 evidence validation。

## 结果同步

`run_all.py` 负责写出 JSON/CSV 和图像。Dashboard 的 `reproAssessments` 位于 `docs/js/reading-data.js`，其中 `resultStatus` 与 `resultFiles` 指向这些已生成的静态图。更新实验后，需要重新运行：

```bash
python reproduce/run_all.py
node docs/scripts/validate.mjs
```
