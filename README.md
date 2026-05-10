# Xiaohao Cai 学术研究精读与复现项目

> **目标**: 系统精读80+篇论文，复现核心方法
> **状态**: 54篇精读笔记已填充（约80篇PDF）

---

## 📁 项目结构

```
D:\Documents\zx\
│
├── 📄 README.md                          # 本文件
├── 📄 CLAUDE.md                          # Claude Code 工作指南
├── 📄 start-server.bat                   # 启动本地服务器脚本
│
├── 📁 docs/                              # Web展示系统
│   ├── index.html                        # 主页面
│   ├── reading_report.html               # 阅读报告页
│   ├── reproduction_report.html          # 复现报告页
│   ├── style.css / js/                   # 前端资源
│   ├── assets/repro/                     # 复现实验图片
│   └── 00_papers_first_author_xiaohao_cai_deduped/  # 15篇去重PDF元数据
│
├── 📁 visualizer_complete/               # 完整可视化系统
│   ├── 00_papers/                        # 80篇论文PDF（346 MB）
│   ├── app.js / data.js / index.html    # 前端代码
│   └── style.css / check_files.py
│
├── 📁 xiaohao_cai_ultimate_notes/        # 论文精读笔记（108个文件）
│   ├── 00_分析报告汇总.md
│   ├── *_超精读笔记_已填充.md            # 54篇完整笔记
│   ├── *_超精读笔记_完整.md / 深度版.md   # 精简版本
│   └── README.md
│
└── 📁 reproduce/                         # 复现实验代码
    ├── run_all.py
    ├── experiments/
    └── results/
```

---

## 📊 数据统计

| 指标 | 数值 |
|:---|:---:|
| **PDF论文** | 80篇 |
| **精读笔记文件** | 108个 |
| **已填充完整笔记** | 54篇 |
| **复现实验** | 9个实验（reproduce/） |

---

## 🚀 快速开始

### 启动Web展示系统

```bash
start-server.bat
```

访问 http://localhost:8080/docs/ （精读Dashboard）
或 http://localhost:8080/visualizer_complete/ （完整可视化系统）

### 查看精读笔记

所有笔记位于 `xiaohao_cai_ultimate_notes/` 目录，命名格式：

```
{主题}_超精读笔记_已填充.md     # 完整版（54篇）
{主题}_超精读笔记_完整.md       # 精简完整版
{主题}_超精读笔记_深度版.md     # 深度分析版
{主题}_超精读笔记.md            # 普通版
```

### 查看PDF

```
visualizer_complete/00_papers/          # 全部80篇PDF（346 MB）
```

---

## 📝 精读笔记内容

每篇完整笔记包含5-Agent辩论分析：

1. 📄 论文元信息（标题、作者、年份、arXiv）
2. 🔢 数学家Agent：理论分析
3. 🔧 工程师Agent：实现细节
4. 💼 应用专家Agent：应用价值
5. 🤨 质疑者Agent：批判性审查
6. 🎯 综合者Agent：共识总结

---

## 🎓 研究领域

- **变分分割**: SLaT, Mumford-Shah, ROF, 多类分割
- **射电天文**: 无线电干涉成像, 不确定性量化
- **3D视觉**: 树木分割, 点云分析, LiDAR
- **医学影像**: MRI重建, 图像分类, 报告生成
- **张量分解**: Tucker分解, Tensor Train
- **深度学习**: 多模态, 可解释AI, 目标检测

---

## 📈 当前进度

```
PDF收集          ████████████████████░░ 80篇
精读笔记         ██████████████░░░░░░░░ 54/80篇已填充
复现实验         ████████████░░░░░░░░░░ 进行中
```

---

## 📄 许可证

本项目仅供学习和研究使用。论文版权归原作者所有。

---

**最后更新**: 2026年5月10日
