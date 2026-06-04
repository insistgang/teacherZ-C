# Xiaohao Cai 第一作者论文去重阅读集

本目录是当前项目使用的 15 篇 Xiaohao Cai 第一作者论文 PDF 集，也是 `docs/js/reading-data.js` 的 PDF 权威目录。

- 初始第一作者 PDF：16
- 去重后保留 PDF：15
- 内容级重复移除：1

## 去重规则

- 先抽取 PDF 全文文本，计算规范化文本哈希、词表 Jaccard 和全文序列相似度。
- 完全相同哈希，或词表 Jaccard > 0.95 且文本序列相似度 > 0.93 的文件视作内容级重复。
- 对重复组保留文件名更准确、内容更适合作为阅读入口的一份。

## 移除的重复项

- `分布式无线电优化 Distributed Radio Optimization.pdf`
  - 标题：Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation
  - 原因：与 `高维逆问题不确定性量化 Uncertainty Quantification.pdf` 基本为同一论文不同版本/错命名文件。

## 保留文件

1. `SLaT三阶段分割 SLaT Segmentation.pdf`
   - 主题组：`variational_segmentation`
   - 标题：A Three-stage Approach for Segmenting Degraded Color Images: Smoothing, Lifting and Thresholding (SLaT)
2. `两阶段分类 Two-Stage.pdf`
   - 主题组：`variational_segmentation`
   - 标题：A TWO-STAGE CLASSIFICATION METHOD FOR HIGH-DIMENSIONAL DATA AND POINT CLOUDS
3. `分割恢复联合模型 Segmentation Restoration.pdf`
   - 主题组：`variational_segmentation`
   - 标题：Variational Image Segmentation Model Coupled with Image Restoration Achievements
4. `分割方法论总览 SaT Overview.pdf`
   - 主题组：`variational_segmentation`
   - 标题：An Overview of SaT Segmentation Methodology and Its Applications in Image Processing
5. `变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf`
   - 主题组：`variational_segmentation`
   - 标题：LINKAGE BETWEEN PIECEWISE CONSTANT MUMFORD-SHAH MODEL AND ROF MODEL AND ITS VIRTUE IN IMAGE SEGMENTATION
6. `在线无线电干涉成像 Online Radio Imaging.pdf`
   - 主题组：`inverse_uq_radio`
   - 标题：MNRAS 000, 1–14 (2017) Online radio interferometric imaging: assimilating and discarding visibilities on arrival
7. `多类ROF分割 Iterated ROF.pdf`
   - 主题组：`variational_segmentation`
   - 标题：Multiclass Segmentation by Iterated ROF Thresholding
8. `无线电干涉不确定性I Radio Interferometric I.pdf`
   - 主题组：`inverse_uq_radio`
   - 标题：MNRAS 000, 1–16 (2017) Uncertainty quantiﬁcation for radio interferometric imaging: I. proximal MCMC methods
9. `无线电干涉不确定性II Radio Interferometric II.pdf`
   - 主题组：`inverse_uq_radio`
   - 标题：MNRAS 000, 1–13 (2017) Uncertainty quantiﬁcation for radio interferometric imaging: II. MAP estimation
10. `框架分割管状结构 Framelet Tubular.pdf`
   - 主题组：`variational_segmentation`
   - 标题：Vessel Segmentation in Medical Imaging Using a Tight-Frame Based Algorithm
11. `框架管状结构分割 Framelet.pdf`
   - 主题组：`variational_segmentation`
   - 标题：Framelet-Based Algorithm for Segmentation of Tubular Structures
12. `球面小波分割 Wavelet Sphere.pdf`
   - 主题组：`inverse_uq_radio`
   - 标题：WAVELET-BASED SEGMENTATION ON THE SPHERE
13. `近端嵌套采样 Proximal Nested Sampling.pdf`
   - 主题组：`inverse_uq_radio`
   - 标题：Proximal nested sampling for high-dimensional Bayesian model selection
14. `高效变分分类 Efficient Variational.pdf`
   - 主题组：`classification_methods`
   - 标题：An Efﬁcient and Versatile Variational Method for High-Dimensional Data Classiﬁcation
15. `高维逆问题不确定性量化 Uncertainty Quantification.pdf`
   - 主题组：`inverse_uq_radio`
   - 标题：Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation
