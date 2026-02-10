# Researcher 紧急分片索引 (EMERGENCY CHUNK INDEX)

> **创建原因**: researcher 遇到 context overflow (185,491 tokens)
> **分片策略**: 按论文ID分组，每组约5-10篇，避免超载
> **使用说明**: 请每次只处理一个chunk，完成后报告给team-lead再处理下一个

---

## 快速分片总览

| Chunk | 论文范围 | 论文数 | 核心主题 | 优先级 |
|-------|----------|--------|----------|--------|
| **R-1** | [1-01] ~ [1-07] | 7篇 | 基础架构 | 基础 |
| **R-2** | [2-01] ~ [2-10] | 10篇 | 图像分割核心 | ⭐⭐⭐⭐⭐ |
| **R-3** | [2-11] ~ [2-20] | 10篇 | 3D视觉+医学 | ⭐⭐⭐⭐ |
| **R-4** | [2-21] ~ [2-31] | 11篇 | 医学前沿+补充 | ⭐⭐⭐⭐ |
| **R-5** | [3-01] ~ [3-07] | 7篇 | 大模型+多模态 | ⭐⭐⭐⭐⭐ |
| **R-6** | [3-08] ~ [3-13] | 6篇 | 3D生成+XAI | ⭐⭐⭐⭐ |
| **R-7** | [4-01] ~ [4-10] | 10篇 | 雷达+遥感 | ⭐⭐⭐ |
| **R-8** | [4-11] ~ [4-32] | 22篇 | 其他重要工作 | ⭐⭐⭐ |

---

## R-1: 基础架构 (7篇)

**论文ID**: [1-01] ~ [1-07]

### 快速列表

| ID | 中文标题 | 关键词 | 重要性 |
|----|----------|--------|--------|
| [1-01] | 深度学习架构综述 | CNNs RNNs Transformers | ⭐⭐⭐⭐⭐ |
| [1-02] | 分割方法论总览 | SaT Overview | ⭐⭐⭐⭐⭐ |
| [1-03] | 数据增强基础 | Data Augmentation | ⭐⭐⭐ |
| [1-04] | 变分法基础 | Mumford-Shah ROF | ⭐⭐⭐⭐ |
| [1-05] | 高维数据分类 | Two-Stage Classification | ⭐⭐⭐ |
| [1-06] | 可解释AI综述 | XAI Advancements | ⭐⭐⭐ |
| [1-07] | 动作识别架构综述 | Action Recognition | ⭐⭐⭐ |

### 关键词

```
深度学习架构, CNN, RNN, Transformer, 图像分割, 变分法,
Mumford-Shah, ROF, 数据增强, 可解释AI, XAI
```

---

## R-2: 图像分割核心 (10篇) ⭐⭐⭐⭐⭐

**论文ID**: [2-01] ~ [2-10]

### 快速列表

| ID | 中文标题 | 关键词 | 重要性 |
|----|----------|--------|--------|
| [2-01] | 凸优化分割 | Convex Mumford-Shah | ⭐⭐⭐⭐⭐ |
| [2-02] | 多类分割迭代ROF | Iterated ROF | ⭐⭐⭐⭐ |
| [2-03] | SLaT三阶段分割 | SLaT Segmentation | ⭐⭐⭐⭐⭐ |
| [2-04] | 分割与恢复联合 | Segmentation Restoration | ⭐⭐⭐ |
| [2-05] | 语义比例分割 | Semantic Proportions | ⭐⭐⭐⭐ |
| [2-06] | 可见表面检测 | Detect Closer Surfaces | ⭐⭐⭐ |
| [2-07] | 光流分割 | Potts Priors | ⭐⭐⭐ |
| [2-08] | 小波框架血管分割 | Vessel Segmentation | ⭐⭐⭐⭐ |
| [2-09] | 框架分割管状结构 | Framelet Tubular | ⭐⭐⭐⭐ |
| [2-10] | 生物孔隙变分分割 | Bio-Pores Segmentation | ⭐⭐⭐ |

### 关键词

```
凸优化, Split Bregman, SLaT三阶段, Smoothing-Lifting-Thresholding,
小波框架, 紧框架, 管状结构分割, 语义比例, Potts先验
```

### 方法论亮点

- [2-01]: 凸松弛技术，全局最优
- [2-03]: SLaT三阶段框架（已有883行笔记）
- [2-05]: 语义融入分割

---

## R-3: 3D视觉+医学 (10篇) ⭐⭐⭐⭐

**论文ID**: [2-11] ~ [2-20]

### 快速列表

| ID | 中文标题 | 关键词 | 重要性 |
|----|----------|--------|--------|
| [2-11] | 3D检测新范式 | CornerPoint3D | ⭐⭐⭐⭐⭐ |
| [2-12] | 点云神经表示 | Neural Varifolds | ⭐⭐⭐⭐⭐ |
| [2-13] | 跨域3D目标检测 | Cross-Domain 3D | ⭐⭐⭐⭐ |
| [2-14] | 3D生长轨迹重建 | 3D Growth Trajectory | ⭐⭐⭐ |
| [2-15] | 3D树木分割图割 | 3D Tree Segmentation | ⭐⭐⭐⭐ |
| [2-16] | 3D树木描绘图割 | 3D Tree Delineation | ⭐⭐⭐⭐ |
| [2-17] | 形状签名Krawtchouk矩 | 3DKMI | ⭐⭐⭐ |
| [2-18] | 3D方向场变换 | 3D Orientation Field | ⭐⭐⭐⭐ |
| [2-19] | 多传感器树木映射 | Tree Mapping | ⭐⭐⭐⭐ |
| [2-20] | 放疗直肠分割 | Deep Rectum Segmentation | ⭐⭐⭐ |

### 关键词

```
3D检测, CornerPoint3D, 角点预测, 点云, Neural Varifolds,
Varifolds度量, 跨域检测, 图割分割, 3D树木, 放疗
```

### 方法论亮点

- [2-11]: 角点替代中心的检测范式
- [2-12]: 点云神经表示学习（TPAMI）

---

## R-4: 医学前沿+补充 (11篇) ⭐⭐⭐⭐

**论文ID**: [2-21] ~ [2-31]

### 快速列表

| ID | 中文标题 | 关键词 | 重要性 |
|----|----------|--------|--------|
| [2-21] | 扩散模型脑MRI病变 | Diffusion Brain MRI | ⭐⭐⭐⭐ |
| [2-22] | 前列腺放疗器官勾画 | Prostate Radiotherapy | ⭐⭐⭐ |
| [2-23] | 直肠轮廓精度分析 | Rectal Contours Accuracy | ⭐⭐⭐ |
| [2-24] | VoxTox研究计划 | VoxTox Programme | ⭐⭐⭐⭐ |
| [2-25] | 医学图像小样本学习 | Medical Few-Shot | ⭐⭐⭐⭐⭐ |
| [2-26] | 非负子空间小样本 | Non-negative Subspace | ⭐⭐⭐⭐⭐ |
| [2-27] | 临床变量医学分类 | Medical Classification | ⭐⭐⭐ |
| [2-28] | 医学报告生成IIHT | Medical Report Generation | ⭐⭐⭐⭐ |
| [2-29] | 中心体分割网络 | CenSegNet | ⭐⭐⭐⭐ |
| [2-30] | 高效变分分类方法 | Efficient Variational | ⭐⭐⭐ |
| [2-31] | 点云神经表示补充 | Neural Varifolds Supplement | ⭐⭐⭐ |

### 关键词

```
扩散模型, 脑MRI, 小样本学习, Few-Shot, N-way K-shot,
非负子空间, 元学习, 医学报告生成, CenSegNet
```

### 方法论亮点

- [2-21]: 扩散模型医学应用
- [2-25]: 医学小样本学习（MedIA）
- [2-26]: 非负子空间特征学习（IEEE TMI）

---

## R-5: 大模型+多模态 (7篇) ⭐⭐⭐⭐⭐

**论文ID**: [3-01] ~ [3-07]

### 快速列表

| ID | 中文标题 | 关键词 | 重要性 |
|----|----------|--------|--------|
| [3-01] | 大模型高效微调 | LLM Fine-tuning | ⭐⭐⭐⭐ |
| [3-02] | 张量CUR分解LoRA | tCURLoRA | ⭐⭐⭐⭐⭐ |
| [3-03] | 自监督图神经网络 | LL4G Graph | ⭐⭐⭐⭐ |
| [3-04] | 低秩Tucker近似 | sketching Tucker | ⭐⭐⭐⭐ |
| [3-05] | 大规模张量分解 | Two-Sided Sketching | ⭐⭐⭐⭐ |
| [3-06] | 雷达语言多模态 | Talk2Radar | ⭐⭐⭐⭐⭐ |
| [3-07] | 多模态虚假新闻GAMED | GAMED Fake News | ⭐⭐⭐⭐ |

### 关键词

```
大模型, PEFT, LoRA, tCURLoRA, 张量CUR分解, ICML,
自监督GNN, Tucker分解, Sketching, Talk2Radar,
雷达语言多模态, ACM MM Oral, GAMED, 多专家解耦
```

### 方法论亮点

- [3-02]: tCURLoRA（ICML 2024）
- [3-06]: Talk2Radar（ACM MM Oral）
- [3-04/05]: 张量分解Sketching

---

## R-6: 3D生成+XAI (6篇) ⭐⭐⭐⭐

**论文ID**: [3-08] ~ [3-13]

### 快速列表

| ID | 中文标题 | 关键词 | 重要性 |
|----|----------|--------|--------|
| [3-08] | 3D人体运动生成Mogo | Mogo Motion | ⭐⭐⭐⭐ |
| [3-09] | 迁移学习动作识别 | TransNet | ⭐⭐⭐⭐ |
| [3-10] | CNN与Transformer动作识别 | CNN-ViT Action | ⭐⭐⭐⭐ |
| [3-11] | 概念级XAI指标 | Concept-based XAI | ⭐⭐⭐⭐⭐ |
| [3-12] | 多层次XAI解释 | Multilevel XAI | ⭐⭐⭐⭐ |
| [3-13] | GAMED多专家解耦 | GAMED Decoupling | ⭐⭐⭐ |

### 关键词

```
3D运动生成, Mogo, ICLR, 迁移学习, TransNet,
CNN-ViT融合, 概念级XAI, TCAV, TPAMI, 多模态XAI
```

### 方法论亮点

- [3-08]: Mogo（ICLR 2024）
- [3-11]: 概念级XAI（TPAMI）

---

## R-7: 雷达+遥感 (10篇) ⭐⭐⭐

**论文ID**: [4-01] ~ [4-10]

### 快速列表

| ID | 中文标题 | 关键词 | 重要性 |
|----|----------|--------|--------|
| [4-01] | 雷达工作模式识别 | Radar Work Mode | ⭐⭐⭐⭐ |
| [4-02] | 雷达信号去噪DNCNet | DNCNet Denoising | ⭐⭐⭐⭐ |
| [4-03] | ISAR卫星特征识别 | ISAR Satellite | ⭐⭐⭐ |
| [4-04] | 无线电干涉不确定性I | Radio Interferometric I | ⭐⭐⭐⭐⭐ |
| [4-05] | 在线无线电干涉成像 | Online Radio Imaging | ⭐⭐⭐⭐ |
| [4-06] | 分布式无线电干涉优化 | Distributed Radio | ⭐⭐⭐⭐ |
| [4-07] | 高维逆问题不确定性 | High-Dimensional Uncertainty | ⭐⭐⭐⭐⭐ |
| [4-08] | 近端嵌套采样 | Proximal Nested Sampling | ⭐⭐⭐⭐ |
| [4-09] | 数据驱动先验嵌套采样 | Data-Driven Priors | ⭐⭐⭐⭐ |
| [4-10] | 多传感器树种分类 | Tree Species Classification | ⭐⭐⭐⭐ |

### 关键词

```
雷达信号, FMCW, 4D毫米波, DNCNet, ISAR,
无线电干涉, 贝叶斯不确定性, 嵌套采样,
稀疏凸优化, 分布式优化
```

---

## R-8: 其他重要工作 (22篇) ⭐⭐⭐

**论文ID**: [4-11] ~ [4-32]

### 主要分类

**遥感与植被** (6篇):
[4-11] 非参数配准, [4-12] 球面小波, [4-13] 贝壳识别,
[4-14] 基因形态学, [4-15] 颜色空间协同, [4-29] 舰船匹配

**个性检测** (3篇):
[4-16] 情感感知, [4-17] 脑启发, [4-18] 错误标记检测

**NAS与优化** (3篇):
[4-19] 平衡NAS (TPAMI), [4-20] NAS SEI应用, [4-23] 双层优化

**多目标跟踪** (2篇):
[4-21] GRASPTrack, [4-22] 跨域LiDAR

**生物与医学** (4篇):
[4-24] 生物启发ILC, [4-26] 前质体, [4-27] 类囊体, [4-25] 社论

**贝叶斯统计** (3篇):
[4-30] 假设检验, [4-31] 可信区间, [4-32] 峰值统计

**无线电干涉补充** (1篇):
[4-28] 离线与在线重建

---

## 处理指令

### 对于researcher:

1. **每次只处理一个chunk** (R-1 到 R-8)
2. **完成后报告team-lead**: "R-N已完成，结果摘要: ..."
3. **等待指示后再处理下一个**

### 推荐处理顺序:

```
R-2 (图像分割核心) → R-5 (大模型多模态) → R-4 (医学前沿)
→ R-3 (3D视觉) → R-6 (XAI) → R-7 (雷达) → R-1/R-8 (补充)
```

---

**创建时间**: 紧急响应
**状态**: 所有chunks已准备就绪，等待researcher处理
