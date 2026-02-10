# 分片处理索引 (CHUNK INDEX)

> **创建日期**: 2026年2月9日
> **分片策略**: 按研究阶段和领域分片，每chunk约50K token
> **总计**: 6 chunks

---

## 分片概览

| Chunk ID | 论文数量 | Token估算 | 核心内容 | Chunk文件 | 状态 |
|----------|----------|-----------|----------|-----------|------|
| **#1** | 7篇 | ~48K | 第一阶段：建立基础 | CHUNK_01_第一阶段建立基础.md | ✅ 已创建 |
| **#2** | 14篇 | ~49K | 第二阶段：图像分割与变分法 | CHUNK_02_第二阶段图像分割与变分法.md | ✅ 已创建 |
| **#3** | 17篇 | ~47K | 第二阶段：3D视觉与医学图像 | CHUNK_03_第二阶段3D视觉与医学图像.md | ✅ 已创建 |
| **#4** | 13篇 | ~48K | 第三阶段：前沿探索 | CHUNK_04_第三阶段前沿探索.md | ✅ 已创建 |
| **#5** | 19篇 | ~49K | 第四阶段：雷达与遥感 | CHUNK_05_第四阶段雷达与遥感.md | ✅ 已创建 |
| **#6** | 14篇 | ~48K | 第四阶段：其他重要工作 | CHUNK_06_第四阶段其他重要工作.md | ✅ 已创建 |

**总计**: 83篇论文 → 6个chunks → 全部创建完成 ✅

---

## Chunk #1: 第一阶段 - 建立基础

**Chunk ID**: #1/6
**Token数**: ~48K
**包含论文**: 7篇

### 论文列表

| 论文ID | 中文标题 | 英文关键词 | 重要性 |
|--------|----------|------------|--------|
| [1-01] | 深度学习架构综述 | CNNs RNNs Transformers | ⭐⭐⭐⭐⭐ |
| [1-02] | 分割方法论总览 | SaT Overview | ⭐⭐⭐⭐⭐ |
| [1-03] | 数据增强基础 | Data Augmentation | ⭐⭐⭐⭐ |
| [1-04] | 变分法基础 | Mumford-Shah与ROF | ⭐⭐⭐⭐ |
| [1-05] | 高维数据分类 | Two-Stage Classification | ⭐⭐⭐⭐ |
| [1-06] | 可解释AI综述 | XAI Advancements | ⭐⭐⭐ |
| [1-07] | 动作识别架构综述补充 | Action Recognition | ⭐⭐⭐ |

### 核心内容摘要

**研究主题**: 深度学习基础架构与图像分割方法论

**关键知识点**:
- CNN/RNN/Transformer三大架构对比
- 图像分割SaT框架
- 变分法数学基础 (Mumford-Shah/ROF)
- 数据增强技术
- XAI基础概念

**方法论关联**:
- 变分法能量泛函设计
- 深度学习架构选择原则

---

## Chunk #2: 第二阶段 - 图像分割与变分法

**Chunk ID**: #2/6
**Token数**: ~49K
**包含论文**: 14篇

### 论文列表

| 论文ID | 中文标题 | 英文关键词 | 核心贡献 |
|--------|----------|------------|----------|
| [2-01] | 凸优化分割 | Convex Mumford-Shah | 凸松弛技术 |
| [2-02] | 多类分割迭代ROF | Iterated ROF | 迭代阈值 |
| [2-03] | SLaT三阶段分割 | SLaT Segmentation | ⭐ 三阶段框架 |
| [2-04] | 分割与恢复联合 | Segmentation Restoration | 联合优化 |
| [2-05] | 语义比例分割 | Semantic Proportions | 语义融入 |
| [2-06] | 可见表面检测 | Detect Closer Surfaces | 表面检测 |
| [2-07] | 光流分割 | Potts Priors | 扩展Potts |
| [2-08] | 小波框架血管分割 | Vessel Segmentation | 紧框架小波 |
| [2-09] | 框架分割管状结构 | Framelet Tubular | 框架算法 |
| [2-10] | 生物孔隙变分分割 | Bio-Pores Segmentation | 断层图像 |
| [2-11] | 3D检测新范式 | CornerPoint3D | ⭐ 角点预测 |
| [2-12] | 点云神经表示 | Neural Varifolds | ⭐⭐ 神经Varifolds |
| [2-13] | 跨域3D目标检测 | Cross-Domain 3D | 跨域泛化 |
| [2-14] | 3D生长轨迹重建 | 3D Growth Trajectory | 轨迹重建 |

### 核心内容摘要

**研究主题**: 变分法图像分割 + 3D视觉起步

**关键知识点**:
- 凸优化与松弛技术
- SLaT三阶段处理模式 (Smoothing-Lifting-Thresholding)
- CornerPoint3D: 角点替代中心的检测范式
- Neural Varifolds: 点云神经表示

**方法论关联**:
- Split Bregman迭代算法
- 三阶段处理模板
- Varifolds几何度量

---

## Chunk #3: 第二阶段 - 3D视觉与医学图像

**Chunk ID**: #3/6
**Token数**: ~47K
**包含论文**: 11篇

### 论文列表

| 论文ID | 中文标题 | 英文关键词 | 核心贡献 |
|--------|----------|------------|----------|
| [2-15] | 3D树木分割图割 | 3D Tree Segmentation | 图割3D |
| [2-16] | 3D树木描绘图割 | 3D Tree Delineation | 精细描绘 |
| [2-17] | 形状签名Krawtchouk矩 | 3DKMI | Krawtchouk矩 |
| [2-18] | 3D方向场变换 | 3D Orientation Field | 方向场 |
| [2-19] | 多传感器树木映射 | Tree Mapping | 多传感器 |
| [2-20] | 放疗直肠分割 | Deep Rectum Segmentation | 放疗应用 |
| [2-21] | 扩散模型脑MRI病变 | Diffusion Brain MRI | ⭐ 扩散模型 |
| [2-22] | 前列腺放疗器官勾画 | Prostate Radiotherapy | 器官勾画 |
| [2-23] | 直肠轮廓精度分析 | Rectal Contours Accuracy | 精度对比 |
| [2-24] | VoxTox研究计划 | VoxTox Programme | 临床计划 |
| [2-25] | 医学图像小样本学习 | Medical Few-Shot | ⭐⭐ 小样本 |
| [2-26] | 非负子空间小样本 | Non-negative Subspace | ⭐⭐⭐ 非负子空间 |
| [2-27] | 临床变量医学分类 | Medical Classification | 临床融合 |
| [2-28] | 医学报告生成IIHT | Medical Report Generation | 报告生成 |
| [2-29] | 中心体分割网络 | CenSegNet | ⭐ 细小结构 |
| [2-30] | 高效变分分类方法 | Efficient Variational | 变分分类 |
| [2-31] | 点云神经表示补充 | Neural Varifolds Supplement | 补充论文 |

### 核心内容摘要

**研究主题**: 3D树木分析 + 医学图像处理

**关键知识点**:
- 3D图割分割方法
- 医学图像小样本学习
- 非负子空间特征学习
- 扩散模型在脑MRI中的应用

**方法论关联**:
- 小样本学习协议 (N-way K-shot)
- 非负约束优化策略
- 元学习框架

---

## Chunk #4: 第三阶段 - 前沿探索

**Chunk ID**: #4/6
**Token数**: ~48K
**包含论文**: 13篇

### 论文列表

| 论文ID | 中文标题 | 英文关键词 | 核心贡献 |
|--------|----------|------------|----------|
| [3-01] | 大模型高效微调 | LLM Fine-tuning | PEFT概述 |
| [3-02] | 张量CUR分解LoRA | tCURLoRA | ⭐⭐⭐⭐⭐ ICML |
| [3-03] | 自监督图神经网络 | LL4G Graph | 自监督GNN |
| [3-04] | 低秩Tucker近似 | sketching Tucker | ⭐⭐⭐ Tucker |
| [3-05] | 大规模张量分解 | Two-Sided Sketching | ⭐⭐⭐ Sketching |
| [3-06] | 雷达语言多模态 | Talk2Radar | ⭐⭐⭐⭐⭐ Oral |
| [3-07] | 多模态虚假新闻检测GAMED | GAMED Fake News | ⭐ 多专家 |
| [3-08] | 3D人体运动生成Mogo | Mogo Motion | ⭐⭐ ICLR |
| [3-09] | 迁移学习动作识别 | TransNet | 迁移学习 |
| [3-10] | CNN与Transformer动作识别 | CNN-ViT Action | 架构融合 |
| [3-11] | 概念级XAI指标 | Concept-based XAI | ⭐⭐⭐⭐⭐ TPAMI |
| [3-12] | 多层次XAI解释 | Multilevel XAI | 多模态XAI |
| [3-13] | GAMED多专家解耦 | GAMED Decoupling | 补充 |

### 核心内容摘要

**研究主题**: 2023-2025年最新前沿研究

**关键知识点**:
- tCURLoRA: 张量CUR分解用于LoRA
- Talk2Radar: 语言-雷达多模态交互
- GAMED: 多专家解耦框架
- 概念级XAI评估指标
- Mogo: 3D运动生成

**方法论关联**:
- 张量分解方法体系 (CP/Tucker/CUR)
- 跨模态对比学习
- 多专家学习框架
- 概念保真度评估

---

## Chunk #5: 第四阶段 - 雷达与遥感

**Chunk ID**: #5/6
**Token数**: ~49K
**包含论文**: 19篇

### 论文列表

**雷达信号处理** (9篇):
| [4-01] 雷达工作模式识别 | [4-02] DNCNet去噪 | [4-03] ISAR卫星 |
| [4-04] 无线电干涉I | [4-05] 在线重建 | [4-06] 分布式优化 |
| [4-07] 高维不确定性 | [4-08] 嵌套采样 | [4-09] 数据驱动先验 |
| [4-28] 离线与在线重建 |

**遥感与植被** (6篇):
| [4-10] 树种分类 | [4-11] 非参数配准 | [4-12] 球面小波 |
| [4-13] 贝壳识别 | [4-14] 基因形态学 | [4-15] 颜色空间 |

**个性检测** (3篇):
| [4-16] 情感感知 | [4-17] 脑启发 | [4-18] 错误标记 |

### 核心内容摘要

**研究主题**: 雷达信号处理 + 遥感应用

**关键知识点**:
- 贝叶斯不确定性量化
- 嵌套采样方法
- 深度学习雷达信号处理
- 多传感器融合

---

## Chunk #6: 第四阶段 - 其他重要工作

**Chunk ID**: #6/6
**Token数**: ~48K
**包含论文**: 14篇

### 论文列表

| 论文ID | 中文标题 | 英文关键词 | 核心贡献 |
|--------|----------|------------|----------|
| [4-19] | 神经架构搜索NAS | Balanced NAS | ⭐ TPAMI |
| [4-20] | NAS在SEI应用 | NAS for SEI | SEI应用 |
| [4-21] | 多目标跟踪GRASPTrack | GRASPTrack MOT | 几何推理 |
| [4-22] | 跨域LiDAR检测 | Cross-Domain LiDAR | 跨域检测 |
| [4-23] | 双层优化形式化 | Bilevel Formalism | 双层优化 |
| [4-24] | 生物启发迭代学习 | Biologically-Inspired ILC | 迭代学习 |
| [4-25] | 分割分类社论 | Editorial Segmentation | 社论 |
| [4-26] | 电子断层分析前质体 | Electron Tomography Prolamellar | 细胞器 |
| [4-27] | 电子断层分析类囊体 | Electron Tomography Thylakoid | 叶绿体 |
| [4-29] | 遥感图像舰船匹配 | Remote Sensing Ship | 线特征 |
| [4-30] | 稀疏贝叶斯质量映射假设检验 | Sparse Bayesian Hypothesis | 假设检验 |
| [4-31] | 稀疏贝叶斯质量映射可信区间 | Sparse Bayesian Credible | 可信区间 |
| [4-32] | 稀疏贝叶斯质量映射峰值统计 | Sparse Bayesian Peak | 峰值统计 |

### 核心内容摘要

**研究主题**: NAS、多目标跟踪、贝叶斯应用

**关键知识点**:
- 神经架构搜索
- 多目标跟踪
- 双层优化
- 稀疏贝叶斯质量映射

---

## 处理优先级

基于当前任务需求，推荐处理顺序：

1. **Chunk #2** (图像分割与变分法) - 包含SLaT三阶段框架
2. **Chunk #4** (前沿探索) - 包含tCURLoRA、Talk2Radar等高优先级论文
3. **Chunk #3** (医学图像) - 包含小样本学习、非负子空间

---

## 分片处理状态

| Chunk | 分片完成 | 摘要完成 | 方法提取 | 状态 |
|-------|----------|----------|----------|------|
| #1 | ✅ | ✅ | ✅ | 已完成 ✅ |
| #2 | ✅ | ✅ | ✅ | 已完成 ✅ |
| #3 | ✅ | ✅ | ✅ | 已完成 ✅ |
| #4 | ✅ | ✅ | ✅ | 已完成 ✅ |
| #5 | ✅ | ✅ | ✅ | 已完成 ✅ |
| #6 | ✅ | ✅ | ✅ | 已完成 ✅ |

**更新时间**: 2026年2月10日
**完成状态**: 全部完成 ✅

---

*索引创建时间: 2026年2月9日*
*分片策略: knowledge-path*
*处理目标: method-summarizer*
