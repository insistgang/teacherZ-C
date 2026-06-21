# Xiaohao Cai 学术研究精读与复现项目

> **目标**: 系统精读 15 篇 Xiaohao Cai 第一作者论文，并提供 toy/partial 复现评估
> **状态**: 15 个结构化精读卡片已完成，15 个独立 Markdown 笔记文件已整理

---

## 📁 项目结构

```
teacherZ-C/
│
├── 📄 README.md                          # 本文件
├── 📄 CLAUDE.md                          # Claude Code 工作指南
├── 📄 start-server.sh                    # 启动本地静态服务器脚本
├── 📄 requirements.txt                   # Python依赖
├── 📄 .gitignore                         # Git忽略文件
│
├── 📁 docs/                              # 15篇论文 Web 展示系统
│   ├── index.html                        # 主页面（精读Dashboard）
│   ├── reading_report.html               # 阅读报告页
│   ├── reproduction_report.html          # 复现报告页
│   ├── style.css / js/                   # 前端资源
│   ├── scripts/validate.mjs              # 数据验证脚本
│   ├── assets/repro/                     # 复现实验图片
│   └── 00_papers_first_author_xiaohao_cai_deduped/  # 15篇PDF
│
├── 📁 xiaohao_cai_ultimate_notes/        # 15个独立 Markdown 精读笔记文件
│   ├── Framelet_Based_Tubular_Structures_超精读笔记_已填充.md
│   ├── High-Dimensional_Inverse_Problems_UQ_超精读笔记_已填充.md
│   ├── Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md
│   ├── Multiclass_Segmentation_Iterated_ROF_超精读笔记_已填充.md
│   ├── Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md
│   ├── Proximal_Nested_Sampling_超精读笔记_已填充.md
│   ├── Radio_Interferometric_Imaging_I_超精读笔记_已填充.md
│   ├── Radio_Interferometric_Imaging_II_超精读笔记_已填充.md
│   ├── SLaT_Three-stage_Segmentation_超精读笔记_已填充.md
│   ├── Tight_Frame_Vessel_Segmentation_超精读笔记_已填充.md
│   ├── Two-Stage_Classification_Point_Clouds_超精读笔记_已填充.md
│   ├── Variational_Segmentation-Restoration_超精读笔记_已填充.md
│   ├── Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md
│   ├── 分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md
│   └── 高效变分分类方法_Efficient_Variational_Classification_超精读笔记_已填充.md
│
└── 📁 reproduce/                         # 15项复现评估的 toy/partial 实验代码
    ├── run_all.py
    ├── experiments/
    ├── results/
    └── paper_like/workflows/             # 15篇 per-paper 完整复现流程文档（<id>_reproduction_workflow.md）
```

---

## 📊 数据统计

| 指标 | 数值 |
|:---|:---:|
| **第一作者论文** | 15篇 |
| **PDF论文** | 15篇 |
| **结构化精读卡片** | 15个 |
| **独立 Markdown 笔记** | 15个文件（一篇 PDF 对应一份笔记） |
| **复现实验** | 9个实验脚本，生成15项复现评估 |

---

## 📖 15篇论文列表

| # | 论文 | 年份 | 笔记状态 | 复现状态 |
|:---:|:---|:---:|:---:|:---:|
| 1 | 分割方法论总览 | 2023 | ✅ | ✅ |
| 2 | Mumford-Shah与ROF联系 | 2019 | ✅ | ✅ |
| 3 | T-ROF迭代阈值分割 | 2013 | ✅ | ✅ |
| 4 | 分割恢复联合模型 | 2014 | ✅ | ✅ |
| 5 | Framelet管状结构短版 | 2012 | ✅ | ✅ |
| 6 | Tight-frame血管分割扩展版 | 2011 | ✅ | ✅ |
| 7 | SLaT三阶段分割 | 2015 | ✅ | ✅ |
| 8 | 球面小波分割 | 2016 | ✅ | ✅ |
| 9 | 高维数据与点云两阶段分类 | 2019 | ✅ | ✅ |
| 10 | 高效变分分类期刊版 | 2024 | ✅ | ✅ |
| 11 | 高维逆问题不确定性量化 | 2019 | ✅ | ✅ |
| 12 | 无线电干涉成像UQ I | 2018 | ✅ | ✅ |
| 13 | 无线电干涉成像UQ II | 2018 | ✅ | ✅ |
| 14 | 在线无线电干涉成像 | 2019 | ✅ | ✅ |
| 15 | 近端嵌套采样 | 2022 | ✅ | ✅ |

---

## 🚀 快速开始

### 启动Web展示系统

```bash
bash start-server.sh
```

访问 http://localhost:8080/docs/ （精读Dashboard）

### 查看精读笔记

独立 Markdown 笔记位于 `xiaohao_cai_ultimate_notes/`。Dashboard 中另有 15 个结构化精读卡片，统一来自 `docs/js/reading-data.js`。每篇论文的作者顺序已从 PDF 首页抽取到 `papers[].authors`，校验脚本会要求 15 篇均以 `Xiaohao Cai` 开头。

笔记重点覆盖：
- 论文元信息与第一作者核验
- 核心问题、难点和方法抓手
- 关键模型或公式、算法流程、理论保证和实验重点
- 与其他 14 篇论文的关系、阅读问题和读后产出
- toy/partial 复现等级、指标、风险和当前运行结果

### 运行复现实验

```bash
cd reproduce
python run_all.py
```

运行项目一致性校验：

```bash
node docs/scripts/validate.mjs
```

---

## 🎓 研究领域

- **变分分割**: SLaT, Mumford-Shah, ROF, T-ROF, 多类分割
- **射电天文**: 无线电干涉成像, 不确定性量化, MAP-UQ
- **医学影像**: 管状结构分割, 球面小波
- **贝叶斯推断**: 近端嵌套采样, 模型选择

---

## 📈 复现效果

| 论文 | 复现等级 | 难度 | 效果 |
|:---|:---:|:---:|:---:|
| SaT总览 / Mumford-Shah / T-ROF | toy-to-partial | 3/5 | 优秀 |
| 分割恢复 | toy | 4/5 | 良好 |
| Framelet / Tight-frame管状结构 | toy | 4/5 | 良好 |
| SLaT | partial | 3/5 | 明显 |
| 球面小波 | toy | 5/5 | 一般 |
| 两阶段 / 高效变分分类 | partial | 4/5 | 良好 |
| UQ / RI / Online RI / Nested Sampling | toy | 4-5/5 | 一般到良好 |

`completed` 只表示 toy/partial 脚本跑通，不表示论文级完整复现。当前 `paper-level-completed = 0 / 15`。

每篇论文还配有一份**完整复现流程规范**：`reproduce/paper_like/workflows/<id>_reproduction_workflow.md`（论文身份与第一作者核验、完整算法管线、所需数据集与公开等价来源、对照基线、评价指标与论文报告数值、本仓库当前实现、差距分析、运行步骤、风险与代理说明）。这些文档描述的是 paper-level 目标流程而非已完成复现，复现报告页每张卡片均链接到对应文档。

---

## 📄 许可证

本项目仅供学习和研究使用。论文版权归原作者所有。

---

**最后更新**: 2026年6月21日
