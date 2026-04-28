# Xiaohao Cai 第一作者论文去重与 Agent 阅读报告

## 处理范围

来源目录：`D:/Documents/zx/docs/00_papers_first_author_xiaohao_cai`

去重后阅读目录：`D:/Documents/zx/docs/00_papers_first_author_xiaohao_cai_deduped`

- 原始第一作者 PDF：16 篇
- 内容去重后保留：15 篇
- 内容级重复移除：1 篇
- 阅读 agent：3 个

## 去重结论

移除的重复文件：

- `分布式无线电优化 Distributed Radio Optimization.pdf`

原因：该 PDF 内容实际为 `Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation`，与 `高维逆问题不确定性量化 Uncertainty Quantification.pdf` 基本是同一论文不同版本/错命名文件。内容相似度核查中，两者词表 Jaccard 约 `0.9876`，文本序列相似度约 `0.977`。

去重后保留文件分为三组：

- `variational_segmentation`：8 篇
- `inverse_uq_radio`：6 篇
- `classification_methods`：1 篇

详表见 `dedup_manifest.tsv` 和 `similarity_pairs.tsv`。

## Agent 分工

Agent A 阅读变分分割/图像分割组：

- `SLaT三阶段分割 SLaT Segmentation.pdf`
- `两阶段分类 Two-Stage.pdf`
- `分割恢复联合模型 Segmentation Restoration.pdf`
- `分割方法论总览 SaT Overview.pdf`
- `变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf`
- `多类ROF分割 Iterated ROF.pdf`
- `框架分割管状结构 Framelet Tubular.pdf`
- `框架管状结构分割 Framelet.pdf`

Agent B 阅读无线电干涉/不确定性量化/采样/球面组：

- `在线无线电干涉成像 Online Radio Imaging.pdf`
- `无线电干涉不确定性I Radio Interferometric I.pdf`
- `无线电干涉不确定性II Radio Interferometric II.pdf`
- `球面小波分割 Wavelet Sphere.pdf`
- `近端嵌套采样 Proximal Nested Sampling.pdf`
- `高维逆问题不确定性量化 Uncertainty Quantification.pdf`

Agent C 阅读高维分类方法组：

- `高效变分分类 Efficient Variational.pdf`

## 总体脉络

这 15 篇第一作者论文可以归纳为三条主线。

第一条是变分分割与 SaT 方法论。核心思想是用图像恢复、凸优化、稀疏/框架表示或图正则化替代直接求解非凸分割模型，然后通过阈值、投影或聚类得到最终分割/分类。关键关键词是 `ROF`、`Mumford-Shah`、`TV`、`thresholding`、`SaT`、`framelet`。

第二条是无线电干涉成像、高维逆问题和不确定性量化。主线是从 scalable reconstruction 出发，进入 proximal MCMC 的完整后验 UQ，再用 MAP + probability concentration 做可扩展 UQ，最后推进到 proximal nested sampling 做 Bayesian evidence 和模型选择。

第三条是高维图分类。`高效变分分类 Efficient Variational.pdf` 把 SaT/两阶段分割思想迁移到高维数据和点云的半监督分类：先求图上的无约束凸变分模型，再用 `argmax` 投影得到硬分类。

## 变分分割组结论

代表论文排序：

1. `变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf`
2. `SLaT三阶段分割 SLaT Segmentation.pdf`
3. `多类ROF分割 Iterated ROF.pdf`
4. `分割方法论总览 SaT Overview.pdf`
5. `两阶段分类 Two-Stage.pdf`
6. `分割恢复联合模型 Segmentation Restoration.pdf`
7. `框架分割管状结构 Framelet Tubular.pdf`
8. `框架管状结构分割 Framelet.pdf`

建议精读顺序：

1. 先快速读 `分割方法论总览 SaT Overview.pdf` 建立地图。
2. 精读 `变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf`，这是理论核心。
3. 读 `多类ROF分割 Iterated ROF.pdf`，理解 T-ROF 的历史源头。
4. 读 `SLaT三阶段分割 SLaT Segmentation.pdf`，关注 smoothing-lifting-thresholding。
5. 读 `分割恢复联合模型 Segmentation Restoration.pdf`，对比联合优化路线。
6. 读 `两阶段分类 Two-Stage.pdf`，看 SaT 如何迁移到图分类。
7. 合并读两个 framelet/tight-frame 管状结构分割文件。

注意：

- `分割方法论总览 SaT Overview.pdf` 是综述章节，不是原始方法论文。
- `框架管状结构分割 Framelet.pdf` 是早期会议版，`框架分割管状结构 Framelet Tubular.pdf` 是扩展版。
- `多类ROF分割 Iterated ROF.pdf` 与 `变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf` 是“会议前身 + 后续理论扩展”，不是简单重复。
- `两阶段分类 Two-Stage.pdf` 严格说是高维数据/点云半监督分类论文，归入本组是因为它继承 SaT 的 smoothing-thresholding 方法论。

## 无线电/UQ/采样组结论

代表论文排序：

1. `无线电干涉不确定性II Radio Interferometric II.pdf`
2. `近端嵌套采样 Proximal Nested Sampling.pdf`
3. `无线电干涉不确定性I Radio Interferometric I.pdf`
4. `在线无线电干涉成像 Online Radio Imaging.pdf`
5. `高维逆问题不确定性量化 Uncertainty Quantification.pdf`
6. `球面小波分割 Wavelet Sphere.pdf`

建议精读顺序：

1. `在线无线电干涉成像 Online Radio Imaging.pdf`
2. `无线电干涉不确定性I Radio Interferometric I.pdf`
3. `无线电干涉不确定性II Radio Interferometric II.pdf`
4. `高维逆问题不确定性量化 Uncertainty Quantification.pdf`
5. `近端嵌套采样 Proximal Nested Sampling.pdf`
6. `球面小波分割 Wavelet Sphere.pdf`

技术承接关系：

- `Online Radio Imaging` 解决 RI 大数据下的流式重建和存储压力。
- `Radio Interferometric I` 用 proximal MCMC 采样完整后验，给出 credible intervals、HPD regions 和结构假设检验。
- `Radio Interferometric II` 用 MAP + probability concentration 替代 MCMC 做可扩展 UQ，是最关键的落地论文。
- `Uncertainty Quantification` 将 RI MAP-UQ 推广到一般高维逆问题，并加入自动正则参数估计和 over-complete dictionary。
- `Proximal Nested Sampling` 进一步从“给定模型下的不确定性”推进到“模型本身如何选择”。

## 高维分类组结论

`高效变分分类 Efficient Variational.pdf` 是 2024 年 Journal of Scientific Computing 论文，题目为 `An Efficient and Versatile Variational Method for High-Dimensional Data Classification`。

它的核心问题是多类半监督分类：少量训练点有标签，目标是根据图结构推断其余点类别。

方法主干：

- 用 `k`-NN 图表示数据点相似性。
- 用 SVM 或随机标签做 warm initialization。
- 求解无 simplex 约束的凸变分模型，包含保真项、图 Laplacian 平滑项和图 TV 项。
- 每个类别标签函数独立求解，天然可并行。
- 用 `argmax` 投影把 fuzzy partition 变成硬分类。
- 可迭代 refinement，并逐步增大 `beta`。

它在 15 篇去重集中的位置：不是纯粹延续早期图像分割，也不是彻底转向机器学习，而是把 SaT/ROF/Mumford-Shah/TV 的变分分割工具箱综合到高维图分类问题上。

## 总体优先阅读路线

如果目标是最快理解蔡老师第一作者论文的核心贡献，建议按以下顺序：

1. `分割方法论总览 SaT Overview.pdf`
2. `变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf`
3. `多类ROF分割 Iterated ROF.pdf`
4. `SLaT三阶段分割 SLaT Segmentation.pdf`
5. `两阶段分类 Two-Stage.pdf`
6. `高效变分分类 Efficient Variational.pdf`
7. `在线无线电干涉成像 Online Radio Imaging.pdf`
8. `无线电干涉不确定性I Radio Interferometric I.pdf`
9. `无线电干涉不确定性II Radio Interferometric II.pdf`
10. `高维逆问题不确定性量化 Uncertainty Quantification.pdf`
11. `近端嵌套采样 Proximal Nested Sampling.pdf`
12. `球面小波分割 Wavelet Sphere.pdf`
13. `分割恢复联合模型 Segmentation Restoration.pdf`
14. `框架管状结构分割 Framelet.pdf`
15. `框架分割管状结构 Framelet Tubular.pdf`

## 后续工作建议

下一步可以按两种方式继续：

1. 做“逐篇精读笔记”：每篇拆成问题定义、模型、算法、定理、实验、局限和可复现点。
2. 做“研究脉络图”：按 SaT/ROF、framelet/tight-frame、graph classification、RI-UQ、proximal Bayesian computation 五个节点画技术演化关系。
