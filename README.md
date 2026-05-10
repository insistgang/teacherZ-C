# Xiaohao Cai 学术研究精读与复现项目

> **目标**: 系统精读15篇第一作者论文，复现核心方法
> **状态**: 15篇精读笔记已全部完成

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
│   ├── index.html                        # 主页面（精读Dashboard）
│   ├── reading_report.html               # 阅读报告页
│   ├── reproduction_report.html          # 复现报告页
│   ├── style.css / js/                   # 前端资源
│   ├── assets/repro/                     # 复现实验图片（10张）
│   └── 00_papers_first_author_xiaohao_cai_deduped/  # 15篇PDF
│
├── 📁 visualizer_complete/               # 完整可视化系统
│   ├── 00_papers/                        # 15篇论文PDF（35 MB）
│   ├── app.js / data.js / index.html    # 前端代码
│   └── style.css
│
├── 📁 xiaohao_cai_ultimate_notes/        # 15篇精读笔记
│   ├── SLaT_Three-stage_Segmentation_超精读笔记_已填充.md
│   ├── Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md
│   ├── Two-Stage_Segmentation_2013_超精读笔记_已填充.md
│   ├── Variational_Segmentation-Restoration_超精读笔记_已填充.md
│   ├── High-Dimensional_Inverse_Problems_UQ_超精读笔记_已填充.md
│   ├── 高效变分分类方法_超精读笔记_已填充.md
│   ├── 框架分割管状结构_超精读笔记_已填充.md
│   ├── 多类分割迭代ROF_超精读笔记_已填充.md
│   ├── 分割方法论总览_超精读笔记_已填充.md
│   ├── Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md
│   ├── Radio_Interferometric_Imaging_I_超精读笔记_已填充.md
│   ├── Radio_Interferometric_Imaging_II_超精读笔记_已填充.md
│   ├── Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md
│   └── Proximal_Nested_Sampling_超精读笔记_已填充.md
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
| **第一作者论文** | 15篇 |
| **PDF论文** | 14篇（+1篇去重版） |
| **精读笔记** | 16个文件（15篇+1篇UQ补充） |
| **复现实验** | 15个实验全部完成 |

---

## 📖 15篇论文列表

| # | 论文 | 年份 | 笔记状态 | 复现状态 |
|:---:|:---|:---:|:---:|:---:|
| 1 | SLaT三阶段分割 | 2015 | ✅ | ✅ |
| 2 | Mumford-Shah与ROF联系 | 2018 | ✅ | ✅ |
| 3 | T-ROF迭代阈值分割 | 2013 | ✅ | ✅ |
| 4 | 分割恢复联合模型 | 2013 | ✅ | ✅ |
| 5 | 高维逆问题不确定性量化 | 2018 | ✅ | ✅ |
| 6 | 高效变分分类 | 2019 | ✅ | ✅ |
| 7 | 框架管状结构分割 | 2016 | ✅ | ✅ |
| 8 | 迭代ROF多类分割 | 2014 | ✅ | ✅ |
| 9 | 分割方法论总览 | 2017 | ✅ | ✅ |
| 10 | 球面小波分割 | 2016 | ✅ | ✅ |
| 11 | 无线电干涉成像I | 2017 | ✅ | ✅ |
| 12 | 无线电干涉成像II | 2017 | ✅ | ✅ |
| 13 | 在线无线电干涉成像 | 2017 | ✅ | ✅ |
| 14 | 近端嵌套采样 | 2021 | ✅ | ✅ |
| 15 | 两阶段图像分割 | 2013 | ✅ | ✅ |

---

## 🚀 快速开始

### 启动Web展示系统

```bash
start-server.bat
```

访问 http://localhost:9090/docs/ （精读Dashboard）

### 查看精读笔记

所有笔记位于 `xiaohao_cai_ultimate_notes/` 目录，包含5-Agent辩论分析：
- 数学家Agent：理论分析
- 工程师Agent：实现细节
- 应用专家Agent：应用价值
- 质疑者Agent：批判性审查
- 综合者Agent：共识总结

### 运行复现实验

```bash
cd reproduce
python run_all.py
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
| SLaT | partial | 3/5 | 优秀 |
| Mumford-Shah | toy-to-partial | 3/5 | 优秀 |
| T-ROF | toy-to-partial | 3/5 | 优秀 |
| 分割恢复 | toy | 4/5 | 良好 |
| UQ不确定性 | toy | 4/5 | 良好 |
| 高效变分 | toy | 4/5 | 良好 |
| 框架管状 | toy | 4/5 | 良好 |
| 迭代ROF | toy | 3/5 | 良好 |
| 分割总览 | toy | 3/5 | 良好 |
| 球面小波 | toy | 5/5 | 一般 |
| RI成像I | toy | 4/5 | 良好 |
| RI成像II | toy | 5/5 | 一般 |
| 在线RI | toy | 4/5 | 良好 |
| 近端采样 | toy | 5/5 | 一般 |
| 两阶段 | partial | 4/5 | 良好 |

**平均难度**: 3.9/5 | **完成率**: 100%

---

## 📄 许可证

本项目仅供学习和研究使用。论文版权归原作者所有。

---

**最后更新**: 2026年5月10日
