# 复现卡片索引

> 生成时间: 2026-02-16
> 项目: D:\Documents\zx (Xiaohao Cai 论文集)

---

## 按复现难度排序

### 容易 (★★☆☆☆)

| 卡片 | 论文 | 代码 | 数据 | 文件 |
|:---|:---|:---:|:---:|:---|
| 01 | Proximal Nested Sampling | ✅ | ✅ | [查看](./01_ProximalNestedSampling.md) |
| 02 | tCURLoRA | ✅ | ✅ | [查看](./02_tCURLoRA.md) |

### 中等 (★★★☆☆)

| 卡片 | 论文 | 代码 | 数据 | 文件 |
|:---|:---|:---:|:---:|:---|
| 03 | SLaT三阶段分割 | ⚠️ | ⚠️ | [查看](./03_SLaT三阶段分割.md) |
| 06 | T-ROF / Mumford-Shah ROF联系 | ❌ | ⚠️ | [查看](./06_TROF.md) |
| 09 | MOGO 3D运动生成 | ⚠️ | ✅ | [查看](./09_MOGO.md) |
| 10 | Concept-Based XAI | ❌ | ✅ | [查看](./10_ConceptXAI.md) |

### 困难 (★★★★☆)

| 卡片 | 论文 | 代码 | 数据 | 文件 |
|:---|:---|:---:|:---:|:---|
| 04 | HiFi-Mamba MRI | ❌ | ✅ | [查看](./04_HiFiMamba_MRI.md) |
| 07 | CornerPoint3D | ❌ | ✅ | [查看](./07_CornerPoint3D.md) |
| 08 | GAMED多模态虚假新闻 | ❌ | ✅ | [查看](./08_GAMED.md) |
| 11 | Neural Varifolds点云 | ❌ | ✅ | [查看](./11_NeuralVarifolds.md) |
| 12 | Cross-Domain LiDAR | ❌ | ✅ | [查看](./12_CrossDomainLiDAR.md) |
| 13 | 3D Tree MCGC | ❌ | ⚠️ | [查看](./13_3DTreeMCGC.md) |
| 14 | Radio Interferometric | ❌ | ⚠️ | [查看](./14_RadioInterferometric.md) |

### 极困难 (★★★★★)

| 卡片 | 论文 | 代码 | 数据 | 文件 |
|:---|:---|:---:|:---:|:---|
| 05 | Talk2Radar | ❌ | ❌ | [查看](./05_Talk2Radar.md) |

---

## 按研究领域分类

### 图像分割

| 论文 | 年份 | 方法 | 卡片 |
|:---|:---:|:---|:---|
| SLaT三阶段分割 | 2015 | 变分方法 | [查看](./03_SLaT三阶段分割.md) |
| T-ROF | 2018 | 变分方法 | [查看](./06_TROF.md) |
| Tight-Frame Vessel | 2011 | 小波框架 | 待创建 |
| 3D Tree MCGC | 2019 | 图割 | [查看](./13_3DTreeMCGC.md) |

### 医学影像

| 论文 | 年份 | 方法 | 卡片 |
|:---|:---|:---|:---|
| HiFi-Mamba MRI | 2025 | 深度学习 | [查看](./04_HiFiMamba_MRI.md) |
| tCURLoRA | 2025 | 参数高效微调 | [查看](./02_tCURLoRA.md) |
| IIHT Medical Report | 2023 | 多模态 | 待创建 |
| Few-shot Medical Imaging | 2023 | 小样本学习 | 待创建 |

### 3D感知

| 论文 | 年份 | 方法 | 卡片 |
|:---|:---|:---|:---|
| CornerPoint3D | 2025 | 3D检测 | [查看](./07_CornerPoint3D.md) |
| Neural Varifolds | 2025 | 点云处理 | [查看](./11_NeuralVarifolds.md) |
| Cross-Domain LiDAR | 2024 | 域适应 | [查看](./12_CrossDomainLiDAR.md) |
| MOGO 3D Motion | 2025 | 运动生成 | [查看](./09_MOGO.md) |

### 多模态

| 论文 | 年份 | 方法 | 卡片 |
|:---|:---|:---|:---|
| Talk2Radar | 2025 | 雷达-语言 | [查看](./05_Talk2Radar.md) |
| GAMED | 2025 | 图像-文本 | [查看](./08_GAMED.md) |

### 可解释AI

| 论文 | 年份 | 方法 | 卡片 |
|:---|:---|:---|:---|
| Concept-Based XAI | 2025 | 概念解释 | [查看](./10_ConceptXAI.md) |

### 贝叶斯方法

| 论文 | 年份 | 方法 | 卡片 |
|:---|:---|:---|:---|
| Proximal Nested Sampling | 2021 | 嵌套采样 | [查看](./01_ProximalNestedSampling.md) |
| Radio Interferometric | 2017 | 射电天文学 | [查看](./14_RadioInterferometric.md) |

---

## 快速参考表

### 开源代码论文

| 论文 | GitHub | 语言 |
|:---|:---|:---:|
| Proximal Nested Sampling | https://github.com/astro-informatics/proxnest | Python |
| tCURLoRA | https://github.com/WangangCheng/t-CURLora | Python |

### 需要自行实现的论文

| 论文 | 主要挑战 | 预计时间 |
|:---|:---|:---:|
| SLaT三阶段分割 | MATLAB转Python | 1天 |
| T-ROF | 变分算法 | 1天 |
| HiFi-Mamba | Mamba架构 | 3-4天 |
| Talk2Radar | 雷达数据 | 5-6周 |

---

## 复现优先级建议

### 高优先级

1. **Proximal Nested Sampling** - 开源代码，理论价值高
2. **tCURLoRA** - 开源代码，实用性强
3. **SLaT三阶段分割** - 经典方法，引用较多

### 中优先级

4. **T-ROF** - 理论创新，实现相对简单
5. **Concept XAI** - 热门方向
6. **MOGO** - 有公开数据

### 低优先级

7. **Talk2Radar** - 数据获取困难
8. **HiFi-Mamba** - 架构复杂
9. **CornerPoint3D** - 训练成本高

---

*更多复现卡片持续更新中...*
