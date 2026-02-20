# 蔡晓昊论文研究可视化系统

这是一个交互式论文可视化系统，用于展示蔡晓昊（Xiaohao Cai）的85篇学术论文。

## 功能特性

- **研究概览**: 论文总数、分类统计、年度趋势图
- **论文列表**: 可筛选、排序、搜索的论文卡片网格
- **研究时间线**: 基于ECharts的时间线散点图
- **方法演进**: 展示论文之间的引用关系和方法演进
- **研究领域**: 按分类组织的论文展示
- **引用网络**: 交互式力导向图展示论文引用关系
- **PDF原文**: 快速访问论文PDF文件

## 使用方法

### 本地访问

直接用浏览器打开 `index.html` 文件即可：

```bash
# Windows
start index.html

# macOS/Linux
open index.html
```

### 通过HTTP服务器访问（推荐）

```bash
# Python 3
cd visualizer
python -m http.server 8000

# Node.js (需要安装 http-server)
npx http-server -p 8000
```

然后在浏览器访问 `http://localhost:8000`

## 文件结构

```
visualizer/
├── index.html      # 主HTML文件
├── style.css       # 样式文件
├── app.js          # 应用逻辑
├── data.js         # 论文数据
└── README.md       # 说明文档
```

## 数据分类

系统将论文分为7大研究领域：

1. **基础理论** (12篇): 图像处理与机器学习的理论基础
2. **变分分割** (18篇): 基于变分方法的图像分割技术
3. **深度学习** (14篇): 深度学习与大语言模型应用
4. **雷达与无线电** (10篇): 射电干涉成像与雷达成像
5. **医学图像** (12篇): 医学影像处理与分析
6. **张量分解** (8篇): 高维张量分解与近似
7. **3D视觉与点云** (11篇): 3D重建、点云处理与LiDAR

## 技术栈

- HTML5 + CSS3 + JavaScript (ES6+)
- ECharts 5.4.3 (图表)
- Chart.js 4.4.0 (饼图、折线图)

## 更新日志

### 2026-02-20
- 更新论文数据至85篇
- 新增张量分解和3D视觉分类
- 改进交互体验和视觉效果
- 添加论文详情模态框

## PDF和笔记路径

- PDF文件路径: `../web-viewer/00_papers/[filename]`
- 笔记文件路径: `../xiaohao_cai_ultimate_notes/[filename]`

## 浏览器兼容性

支持所有现代浏览器：
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
