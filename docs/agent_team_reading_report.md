# Xiaohao Cai 第一作者论文研究脉络完整报告

网页版完整报告请访问：

[reading_report.html](reading_report.html)

这份 Markdown 文件保留为兼容入口，不再作为 dashboard 的主要报告页面。新版 dashboard 的“完整研究报告”按钮已经统一指向 `reading_report.html`，避免用户看到裸 Markdown 或旧版 agent 粗读日志。

## 新版报告内容

网页版报告包含以下部分：

1. 报告定位
2. 总体研究方向判断
3. 论文发表时间线
4. 推荐阅读顺序
5. 三层研究脉络图
6. 15 篇逐篇精读报告
7. 15 篇之间的依赖关系
8. 四周阅读计划
9. 可选研究选题入口
10. 最终总结

## 研究主线

新版报告围绕 5 条主线组织 15 篇 Xiaohao Cai 第一作者论文：

- SaT / ROF / PCMS 变分分割主线
- framelet / tight-frame 管状结构与血管分割主线
- SLaT / spherical wavelet 彩色与几何扩展主线
- graph variational / high-dimensional classification 高维分类主线
- RI imaging / Bayesian UQ / proximal sampling 贝叶斯逆问题主线

## 推荐阅读顺序

1. SaT Overview
2. Linkage Between PCMS and ROF
3. Multiclass Segmentation by Iterated ROF Thresholding
4. Segmentation Restoration
5. Framelet-Based Tubular Segmentation
6. Tight-Frame Vessel Segmentation
7. SLaT
8. Wavelet-Based Segmentation on the Sphere
9. Two-Stage Classification
10. Efficient Variational Classification
11. Quantifying Uncertainty in High-Dimensional Inverse Problems
12. RI UQ I: Proximal MCMC
13. RI UQ II: MAP Estimation
14. Online RI Imaging
15. Proximal Nested Sampling

## 说明

旧版 `agent_team_reading_report.md` 的 agent 分工式摘要已经不再作为主要内容。完整报告现在与 dashboard 的 `paperNotesV2` 数据对齐，报告页通过 `app.js` 和 `report.js` 渲染同一批论文标题、年份、主题、阅读顺序和 PDF 路径。
