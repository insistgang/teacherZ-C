# 项目进展与路线图 (Project Status & Roadmap)

> 口径：蔡晓昊（Xiaohao Cai）15 篇第一作者论文的精读 + 复现评估项目。
> 本文件是**当前状态与后续工作的单一入口**；项目说明见 [README.md](README.md)，逐篇完整复现规范见 [`reproduce/paper_like/workflows/`](reproduce/paper_like/workflows/)。
> 最后更新：2026-06-21。

---

## 一、已完成的工作

### 1. 精读笔记（15 篇，`xiaohao_cai_ultimate_notes/`）
- 15 篇独立 Markdown 精读笔记全部以各篇 PDF 为依据深化，覆盖论文元信息、第一作者核验、核心问题、关键公式、算法流程、理论保证、实验重点、论文关系、阅读问题、读后产出、复现判断。
- 经 **3 轮 multi-agent 对抗式审查**（逐篇 PDF 数值核验 → 跨篇一致性 → 终检）修正约 60 处事实/数值/年份/编号问题。
- 校验脚本要求 15 篇作者顺序均以 `Xiaohao Cai` 开头、每篇含「第一作者核验」与「复现判断」标记。

### 2. 完整复现流程文档（15 篇，`reproduce/paper_like/workflows/<id>_reproduction_workflow.md`）
- 每篇一份，统一 11 节：论文身份与第一作者核验、复现目标与诚实分级、完整算法管线、所需数据集与公开等价来源、对照基线、评价指标与论文报告数值、本仓库当前实现、**差距分析（到 paper-level 还缺哪些步骤）**、运行步骤、风险与代理说明、回链精读笔记。
- `docs/scripts/validate.mjs` 校验每个 reproAssessment id 都存在同名 workflow 文档。

### 3. Web 展示 / Dashboard（`docs/`）
- 精读 Dashboard、阅读报告页、复现报告页，数据统一来自 `docs/js/reading-data.js`。
- 复现报告页每张卡片链接到对应 workflow 文档；版本/日期渲染已加固。

### 4. 复现实现：真实算法替换 toy 代理（`reproduce/experiments/`）
9 个 runner 从「假动作」（Gaussian / Lab-like / rejection 等代理）升级为真实算法：

| runner | 真实算法 |
| --- | --- |
| `sat_rof_trof.py` (#1/#2/#3) | 真实 Chambolle-Pock ROF 凸求解器 + Eq.(15) 阈值迭代 + Multi-Otsu/K-means 基线 |
| `segmentation_restoration.py` (#4) | 真实联合恢复+分割交替最小化（真实模糊算子 A + Fourier Tikhonov + Chambolle-Pock 多相 TV） |
| `tubular_tight_frame.py` (#5/#6) | 真实 pywt 无下采样小波(SWT) tight-frame 软阈值，\|Λ\| 真正收缩到 0 |
| `slat_color.py` (#7) | 真实 sRGB→CIELab + RGB+Lab 六维 K-means |
| `sphere_wavelet_toy.py` (#8) | 真实 SWT tight-frame + 离散球面梯度（平面 SWT 近似） |
| `graph_classification.py` (#9/#10) | 真实 graph-TV(ℓ1) Chambolle-Pock primal-dual + SVM warm init + Three-Moon |
| `map_uq_toy.py` (#11/#12/#13) | 真实 ℓ1-wavelet(db8) MAP via FISTA + HPD 阈值 + superpixel 二分局部可信区间 |
| `online_ri_toy.py` (#14) | 真实 online forward-backward（分块累加梯度 + 丢弃），存储比 η_s≈1/B |
| `nested_sampling_toy.py` (#15) | 真实 MYULA proximal-Langevin 约束采样 + MH 校正（d=10） |

### 5. 校验与测试（全绿）
```bash
node docs/scripts/validate.mjs            # 15 篇数据/PDF/笔记/复现资产一致性
node reproduce/sync_to_dashboard.mjs --check
python -m unittest discover -s reproduce/tests -p 'test_*.py'   # 165 通过
node reproduce/tests/test_sync_to_dashboard.mjs                  # 75 通过
node reproduce/tests/test_repro_promotion_guards.mjs             # 4 通过
```

---

## 二、当前复现真实性快照（诚实分级）

四级口径：`toy`（合成图 + 代理）< `partial`（真实算法路线一部分，合成/标准数据）< `paper-like`（公开等价数据 + baseline + 论文表格量级）< `paper-level`（论文真实数据/参数/baseline 复现具体数值）。

| # | id | 复现等级 | 真实性 |
| --- | --- | --- | --- |
| 1 | sat-overview | partial | partial-completed |
| 2 | pcms-rof-linkage | partial | partial-completed |
| 3 | iterated-rof | partial | partial-completed |
| 4 | segmentation-restoration | partial | partial-completed |
| 5 | framelet-tubular | partial | partial-completed |
| 6 | tight-frame-vessel | partial | partial-completed |
| 7 | slat-color | partial | partial-completed |
| 8 | sphere-wavelet | toy | toy-completed |
| 9 | two-stage-classification | partial | partial-completed |
| 10 | efficient-variational-classification | partial | partial-completed |
| 11 | high-dimensional-uq | toy | toy-completed |
| 12 | ri-uq-i | toy | toy-completed |
| 13 | ri-uq-ii | toy | toy-completed |
| 14 | online-ri | toy | toy-completed |
| 15 | proximal-nested-sampling | toy | toy-completed |

**汇总：partial-completed 9 / toy-completed 6 / `paper-like 0 / 15` / `paper-level 0 / 15`。**
真实算法已就位，但仍是合成/标准测试数据、缺论文真实数据与对照基线，因此严守 paper-level = 0/15。

---

## 三、后续需要做的工作（Roadmap）

总原则：**paper-level 必须有论文真实/等价数据 + 论文 baseline + 表格数值对照才能晋升**，禁止用合成数据自称 paper-like/paper-level。每篇的精确缺步见各自 workflow 文档第 8 节「差距分析」。

### A 类：缺「数据 + 基线」即可推进到 paper-like（门槛较低）
适用 #1–#7、#9、#10（分割 / 分类线，数据多可公开或可严格合成）。共同待办：
1. 取得论文同款或公开等价数据（如 DRIVE 视网膜、BrainWeb MRI、COIL/MNIST/Opt-Digits、按论文参数严格合成的 stripe/cartoon/close-gray 图）。
2. 实现/接入论文对照基线（如 #2/#3 的 Li/Pock/Yuan/He/Cai，#9/#10 的 CVM/GL/MBO/TVRF）。
3. 对齐论文参数与停止准则，输出论文表格量级的指标（SA/DICE/accuracy）。
4. 把求解器/初始化对齐论文（如 #3 的 ADMM + fuzzy C-means 初始化）。

### B 类：缺「专用库 / 专用算子」，短期只能保持 toy（门槛高）
- **#8 sphere-wavelet**：缺真正的球谐球面小波栈（S2LET / SSHT / SO3）与真实球面数据；当前用平面 SWT 近似。
- **#11–#13 UQ / RI**：缺真实射电干涉测量算子（NUFFT / w-projection）、论文真实数据（M31 等）、以及 #12 的 proximal-MCMC 采样器（MYULA / Px-MALA）；当前用标准测试图 + 掩膜 FFT。
- **#14 online-ri**：缺真实 visibility 流式算子与真实观测、论文规模（B∈{50..500}、256² 以上）。
- **#15 proximal-nested-sampling**：缺 ℓ1/小波稀疏先验下的高维（10³–10⁶）成像模型选择实验与熵误差棒；当前 d=10 解析对照。

### C 类：理论类（无法用代码"复现"，只能演示）
- **#2 pcms-rof-linkage**：核心贡献是 Theorem 3.4/3.6/3.7 与收敛性证明；代码只能给一致性佐证，不能替代证明。

### 旗舰路径：iterated-rof 数据-backed 晋升
`reproduce/paper_like/iterated_rof_paper_like.py` 已搭好 paper-like 数据门禁（source registry / manifest / 审计 / mask / runner 交叉校验）。一旦放入真实/等价数据并通过门禁，可按 `reproduce/README.md` 的 promotion 流程用 `ALLOW_PAPER_LIKE=1` 进入晋升审查——这是把第一篇推到 paper-like 的最短路径。

---

## 四、文档导航

| 想看什么 | 去哪 |
| --- | --- |
| 项目总览、快速开始 | [README.md](README.md) |
| Claude Code 工作指南 | [CLAUDE.md](CLAUDE.md) |
| 15 篇精读笔记 | [`xiaohao_cai_ultimate_notes/`](xiaohao_cai_ultimate_notes/) |
| 逐篇完整复现流程 + 差距分析 | [`reproduce/paper_like/workflows/`](reproduce/paper_like/workflows/) |
| 复现评估系统说明、晋升流程 | [`reproduce/README.md`](reproduce/README.md) |
| 全量复现团队计划 | [`reproduce/paper_like/full_reproduction_team_plan.md`](reproduce/paper_like/full_reproduction_team_plan.md) |
| iterated-rof paper-like 详规 | [`reproduce/paper_like/iterated_rof_spec.md`](reproduce/paper_like/iterated_rof_spec.md) |
