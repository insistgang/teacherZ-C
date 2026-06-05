# Framelet 管状结构分割短版

> 当前 15 篇口径内第 5 篇。本文档按 PDF 首页作者顺序和 dashboard 结构化精读字段重写，避免旧论文笔记混入。

## 论文元信息

| 字段 | 内容 |
| --- | --- |
| 英文标题 | Framelet-Based Algorithm for Segmentation of Tubular Structures |
| 作者顺序 | Xiaohao Cai, Raymond H. Chan, Serena Morigi, Fiorella Sgallari |
| 第一作者核验 | 是，PDF 首页作者列表以 Xiaohao Cai 开头 |
| 年份 | 2011 / 2012 |
| 类型 | SSVM / LNCS |
| PDF | docs/00_papers_first_author_xiaohao_cai_deduped/框架管状结构分割 Framelet.pdf |
| 阅读顺序 | 5 / 15 |
| 主题 | medical |
| 难度 | 中等 |

## 一句话贡献

只平滑管状边界候选区。

## 核心问题

MRA 血管、道路和其他 tube-like structures 有细长、弱边缘、分叉、遮挡等特点；全图平滑会抹掉细节，传统 PDE/active contour 又容易被噪声和初始化影响。

## 为什么难

管状结构的内部、外部和边界灰度不是单点可分；真正难的是灰度落在边界候选区间内的像素，既不能粗暴归类，也不能对整幅图无差别平滑。

## 方法抓手

算法估计边界灰度区间 [α_i, β_i]，每轮把图像分成 below、inside、above 三部分，只对 inside，即可能边界区域，做 framelet denoising / smoothing 和 soft-thresholding，再收缩候选区间直到得到二值图像。

## 关键模型或公式

candidate boundary Λ_i = {x : α_i < f_i(x) < β_i}; framelet denoising on Λ_i; stop when all pixels map to 0 or 1.

## 算法流程

1. 估计当前边界灰度区间 [α_i, β_i]。
2. 把像素分成背景、边界候选和血管三类。
3. 只在候选区域 Λ_i 做 framelet soft-thresholding。
4. 更新图像并收缩候选区域。
5. 候选区域为空时输出二值管状结构。

## 理论保证

论文给出 convergence statement：framelet-based algorithm 会在有限步收敛到二值图像；关键是候选边界区域 Λ_i 持续收缩。

## 实验重点

实验为 real 2D/3D images，并在文本中明确指向 Magnetic Resonance Angiography (MRA) 血管场景；重点看细血管、分叉和弱边界是否保留。

## 精读方式

先读 Section 2 的 tight frame / framelet 基础；重点读 Section 3 算法步骤；再读 Theorem 1 的 finite convergence 证明；最后看 2D/3D 图像实验。

## 论文证据点

- Abstract
- Section 3 algorithm
- boundary interval [α_i, β_i]
- finite convergence theorem
- 2D/3D tubular experiments

## 与其他 14 篇的关系

这是 vessel tight-frame 长版的短版基础，也与 SaT 共享“平滑不确定区域 + 阈值化”的思想。

关联论文：#6 Tight-frame 医学血管分割长版; #1 SaT 分割方法论总览; #8 球面小波图像分割

## 报告扩展字段

- context: 这篇是管状结构分割线的短版入口，适合先读来理解思想。目标对象不是普通区域分割，而是 MRA 血管、道路等细长结构，它们的边界弱、分叉多、噪声下容易断裂。
- technicalReading: 技术抓手是 possible boundary gray interval [α_i, β_i]。算法每轮把像素分成 below、inside、above，只对 inside 的候选边界区域做 framelet denoising / soft-thresholding，而不是对整幅图做统一平滑。
- theoremReading: 理论阅读重点是 finite convergence：候选边界集合在迭代中持续收缩，已经确定为 0 或 1 的像素离开候选区。要理解这个保证与传统 variational minimization 不同，它更像有限步分类和局部平滑的组合。
- experimentReading: 实验要看 2D/3D tubular structures，尤其是细血管、弱边界、分叉处是否保留。不要只看最终二值图，还要看候选区域收缩是否可能导致漏检或断裂。
- relationReading: 它是 Tight-frame Vessel 长版的基础，也与 SaT 有相似结构：先稳定不确定部分，再阈值化得到结构。但它更强调候选边界区间和 framelet 表示，而不是 ROF/PCMS 理论。
- researchValue: 这篇的价值是给出一种局部处理策略：复杂图像中不是所有像素都同等困难，真正值得用 framelet 平滑的是边界候选集合。这种思想可迁移到医学点云、血管中心线和遥感线状目标。

## 阅读问题

1. 为什么只对 inside 候选区域做 framelet 平滑？
2. 候选区间 [α_i, β_i] 如何影响收敛速度和漏检？
3. 这个算法为什么不是标准 variational minimization？

## 读后产出

画出 below / inside / above 三分图像和候选区域收缩过程。

## 复现判断

| 字段 | 内容 |
| --- | --- |
| 复现等级 | toy |
| 真实性等级 | toy-completed |
| 难度 | 高 |
| 效果 | 很明显 |
| 最小实验 | synthetic tube/vessel mask with noise，构造 boundary interval [alpha,beta]，只在 uncertain region 做 wavelet/TV-like smoothing。 |
| 预期产出 | uncertainty region shrinks and binary tube mask emerges；toy Dice 0.9981，Lambda 从 651 收缩到 2。 |
| 依赖 | numpy / scipy / scikit-image / matplotlib |
| 数据需求 | synthetic 2D tube network；full reproduction 需要论文使用的 2D/3D tubular structures。 |
| 算力需求 | CPU，约 1 秒内。 |
| 实现风险 | 这里用 Gaussian smoothing 近似 framelet smoothing，不是严格 framelet implementation。 |

### 复现指标

- dice
- iou
- lambda_initial
- lambda_final
- iterations

### 验证计划

记录 Lambda size per iteration、Dice/IoU，并检查候选区域是否随迭代收缩。

### 当前运行结果

- dice: 0.9981
- iou: 0.9962
- lambda_initial: 651
- lambda_final: 2
- iterations: 12

### 结果说明

Approximate toy reproduction: Gaussian smoothing stands in for framelet smoothing inside uncertain boundary interval. Dice is measured on a simple synthetic 2D vessel toy; it does not represent real 2D/3D MRA paper-level performance.
