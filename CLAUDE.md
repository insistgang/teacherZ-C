# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在本项目工作时提供指导。

## 项目概述

这是一个学术研究精读与复现项目，口径固定为蔡晓昊（Xiaohao Cai）的 15 篇第一作者论文。项目包含 15 个结构化精读卡片、15 个独立 Markdown 笔记文件、Web 展示系统和 toy/partial 复现实验代码。

## 常用命令

```bash
# 运行复现实验
cd reproduce && python run_all.py

# 启动Web展示系统
python -m http.server 8080

# 校验15篇数据、PDF、笔记和静态复现资产
node docs/scripts/validate.mjs
```

## 依赖安装

通过以下命令安装：`pip install -r requirements.txt`
- 核心库：numpy, scipy, matplotlib, scikit-image
- 缺少依赖时，复现实验 runner 应写入 skipped，而不是伪造 completed 结果。

## 代码架构

### Web展示系统 (`docs/`)
- `index.html` - 主页面（精读Dashboard）
- `reading_report.html` - 阅读报告页
- `reproduction_report.html` - 复现报告页
- `style.css` / `js/` - 前端资源
- `scripts/validate.mjs` - 数据验证脚本
- `assets/repro/` - 复现实验图片
- `00_papers_first_author_xiaohao_cai_deduped/` - 15篇PDF

### 复现实验代码 (`reproduce/`)
- `run_all.py` - 运行所有复现实验
- `experiments/` - 实验代码
- `results/` - 实验结果
- `paper_like/workflows/<id>_reproduction_workflow.md` - 15 篇 per-paper 完整复现流程规范（paper-level 目标流程，非已完成复现；dashboard 复现报告页每张卡片链接此处）。新增 reproAssessment id 时需同步新增同名文档，否则 `validate.mjs` 报错。

### 精读笔记 (`xiaohao_cai_ultimate_notes/`)
- 15 个独立 Markdown 精读笔记文件；一篇 PDF 对应一份笔记。
- `docs/js/reading-data.js` 中维护 15 个结构化精读卡片和 `papers[].authors` 作者顺序。
- 校验脚本会要求 15 篇论文作者顺序均以 `Xiaohao Cai` 开头，并要求独立 Markdown 文件集合严格等于 15 篇口径。
- 笔记覆盖论文元信息、第一作者核验、核心问题、关键模型或公式、算法流程、理论保证、实验重点、论文关系、阅读问题、读后产出和复现判断。

## 目录结构

```
├── README.md                          # 项目说明文件
├── CLAUDE.md                          # Claude Code 工作指南
├── requirements.txt                   # Python依赖
├── .gitignore                         # Git忽略文件
├── start-server.sh                    # 启动服务器脚本
├── docs/                              # Web展示系统
│   ├── index.html                     # 主页面（精读Dashboard）
│   ├── reading_report.html            # 阅读报告页
│   ├── reproduction_report.html       # 复现报告页
│   ├── style.css / js/                # 前端资源
│   ├── scripts/validate.mjs           # 数据验证脚本
│   ├── assets/repro/                  # 复现实验图片
│   └── 00_papers_first_author_xiaohao_cai_deduped/  # 15篇PDF
├── xiaohao_cai_ultimate_notes/        # 15个独立 Markdown 精读笔记文件
└── reproduce/                         # 复现实验代码
    ├── run_all.py
    ├── experiments/
    ├── results/
    └── paper_like/workflows/         # 15篇 per-paper 完整复现流程文档
```
