# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在本项目工作时提供指导。

## 项目概述

这是一个学术研究仓库，用于分析和复现蔡晓昊（Xiaohao Cai）的67篇以上论文，涵盖变分方法、张量分解、3D视觉、医学影像和深度学习等领域。项目核心特性是一个多智能体辩论系统，用于协作分析学术论文。

## 常用命令

```bash
# 运行多智能体辩论系统
python debate_system.py

# 运行交互式辩论模式
python debate_interactive.py

# 运行高级辩论功能
python debate_advanced.py

# 运行批量论文处理
python batch_process_papers.py

# 运行算法实现示例
cd implementations
python usage_examples.py
```

## 依赖安装

通过以下命令安装：`pip install -r requirements.txt`
- 核心库：pyyaml, pydantic
- LLM客户端：anthropic, openai
- PDF解析：pdfplumber, PyPDF2

## 代码架构

### 多智能体辩论系统 (`debate_system.py`)
核心系统使用5个专业智能体：
- **数学家（Mathematician）** - 理论分析
- **工程师（Engineer）** - 实现评估
- **应用专家（Application Expert）** - 应用价值评估
- **质疑者（Skeptic）** - 批判性审查
- **综合者（Synthesizer）** - 共识构建

核心类：
- `AgentRole` - 智能体类型枚举
- `Paper` - 论文数据结构，包含标题、作者、摘要、正文
- `AgentMessage` - 智能体消息格式
- `DebateState` - 跟踪辩论进度和消息

### 算法实现 (`implementations/`)
- `slat_segmentation.py` - SLaT三阶段分割
- `rof_iterative_segmentation.py` - 迭代ROF多类分割
- `tucker_decomposition.py` - 随机Sketching Tucker分解
- `neural_varifold.py` - 神经变分点云分析

### 知识图谱 (`knowledge_graph/`)
- `queries.py` - 论文关系和方法演进的查询接口
- JSON文件中定义了实体和关系

### 输出结构
- `outputs/debate_log_*.md` - 辩论会话日志
- `outputs/report_*.md` - 分析报告

## 目录结构

```
├── debate_system.py          # 多智能体系统主入口
├── debate_interactive.py    # 交互式模式
├── debate_advanced.py       # 高级功能
├── implementations/         # 算法代码实现
├── knowledge_graph/         # 知识图谱查询
├── outputs/                 # 生成的分析报告
├── benchmark/               # 性能基准测试
└── visualizations/         # 数据可视化
```
