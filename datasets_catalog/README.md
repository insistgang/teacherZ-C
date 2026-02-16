# D:\Documents\zx 项目数据集完整目录

> 生成时间：2026年2月16日
> 项目：Xiaohao Cai 论文精读与研究

---

## 目录

1. [数据集目录表](#一数据集目录表)
2. [按领域分类](#二按领域分类)
3. [按使用频率排名](#三按使用频率排名)
4. [数据获取指南](#四数据获取指南)
5. [推荐数据集](#五推荐数据集)

---

## 一、数据集目录表

### 1.1 医学影像数据集

| 数据集名称 | 领域 | 样本数量 | 标注类型 | 分辨率/维度 | 公开/私有 | 获取方式 | 使用论文 | 基准性能 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **FastMRI** | MRI重建 | 34,742训练/7,195测试 | 全采样k空间 | 320×320 | 公开 | fastmri.org | HiFi-Mamba, HiFi-MambaV2, tCURLoRA | PSNR: 37.43 (4x) |
| **IXI** | MRI | 577受试者 | 脑部结构 | 256×256 | 公开 | brain-development.org | HiFi-MambaV2 | PSNR: 35.2+ |
| **CMRxRecon** | 心脏MRI | - | 动态心脏 | - | 公开 | cmrxrecon.github.io | HiFi-MambaV2 | - |
| **OASIS** | 脑部MRI | 1,000+受试者 | 脑萎缩标注 | 176×256×256 | 公开 | oasis-brains.org | HiFi-MambaV2 | - |
| **BraTS** | 脑肿瘤 | 331训练/125验证 | 肿瘤分割 | 240×240×155 | 公开 | med.upenn.edu | tCURLoRA | Dice: 0.91+ |
| **DRIVE** | 视网膜血管 | 40幅 | 血管分割 | 565×584 | 公开 | isbi.uhasselt.be | Tight-Frame血管分割 | 准确率: ~85% |
| **STARE** | 视网膜 | 20幅 | 血管分割 | 700×605 | 公开 | ceas.clemson.edu | Tight-Frame血管分割 | 准确率: ~85% |
| **CheXpert** | 胸部X光 | 224,316 | 疾病标签 | - | 公开 | stanfordmlgroup.github.io | Equalizing Attributes | AUC: 0.92+ |
| **3T MRI** | 脑部分割 | - | 结构分割 | - | 公开 | - | Semantic Proportions | - |

### 1.2 自动驾驶/3D检测数据集

| 数据集名称 | 领域 | 样本数量 | 标注类型 | 分辨率/维度 | 公开/私有 | 获取方式 | 使用论文 | 基准性能 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **KITTI** | 3D检测 | 7,481训练/7,518测试 | 3D边界框 | 1242×375 | 公开 | cvlibs.net | CornerPoint3D, DetectCloser | mAP: 88.5% (Car Easy) |
| **nuScenes** | 3D检测 | 1.4M帧 | 3D边界框+轨迹 | 1600×900 | 公开 | nuscenes.org | Talk2Radar, CrossDomain LiDAR | NDS: 53.8+ |
| **Waymo Open** | 3D检测 | 1,950段视频 | 3D边界框 | 1920×1280 | 公开 | waymo.com | CornerPoint3D验证 | mAP: 70%+ |
| **View of Delft (VoD)** | 4D雷达 | 8,682样本 | 3D指代标注 | - | 公开 | github.com | Talk2Radar | Acc: 89.2% |
| **ONCE** | 3D检测 | 1M+帧 | 3D边界框 | - | 公开 | once-for-auto-driving | - | - |

### 1.3 遥感/林业数据集

| 数据集名称 | 领域 | 样本数量 | 标注类型 | 分辨率/维度 | 公开/私有 | 获取方式 | 使用论文 | 基准性能 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **Indian Pines** | 高光谱 | 145×145像素 | 16类地物 | 220波段 | 公开 | purdue.edu | 分割方法论总览 | 准确率: 98.83% |
| **LiDAR森林数据** | 林业TLS | 多站点 | 单木分割 | 5M-100M点/站 | 研究 | 作者 | 3D Tree MCGC | F1: 0.85+ |
| **RC/MCRC/TIFFS** | 森林 | 3个站点 | 树木标注 | - | 研究 | 作者 | 3D Tree Graph Cut | IoU: 0.80+ |

### 1.4 自然图像分割数据集

| 数据集名称 | 领域 | 样本数量 | 标注类型 | 分辨率/维度 | 公开/私有 | 获取方式 | 使用论文 | 基准性能 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **BSDS500** | 图像分割 | 500幅 | 边界标注 | 481×321 | 公开 | berkeley.edu | SLaT, 分割-恢复 | ODS: 0.72+ |
| **Alpert** | 显著物体 | - | 显著区域 | 300×225 | 公开 | - | 分割方法论总览 | - |
| **MSRC** | 语义分割 | 591幅 | 23类 | 320×213 | 公开 | research.microsoft.com | Semantic Proportions | - |
| **COCO** | 通用检测 | 330K图像 | 80类检测框 | 多尺寸 | 公开 | cocodataset.org | DetectCloser预训练 | - |

### 1.5 多模态/NLP数据集

| 数据集名称 | 领域 | 样本数量 | 标注类型 | 分辨率/维度 | 公开/私有 | 获取方式 | 使用论文 | 基准性能 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **Weibo** | 假新闻检测 | 8,049 | 真假标签 | 文本+图像 | 公开 | - | GAMED | Acc: 93.5% |
| **Twitter** | 假新闻检测 | 13,225 | 真假标签 | 文本+图像 | 公开 | - | GAMED | Acc: 88.7% |
| **Gossipcop** | 假新闻检测 | 17,500 | 真假标签 | 文本+图像 | 公开 | gossipcop.com | GAMED | Acc: 90%+ |
| **Talk2Car** | 3D指代 | - | 语言+3D框 | - | 公开 | - | Talk2Radar对比 | - |

### 1.6 动作识别/视频数据集

| 数据集名称 | 领域 | 样本数量 | 标注类型 | 分辨率/维度 | 公开/私有 | 获取方式 | 使用论文 | 基准性能 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **KTH** | 动作识别 | 599视频 | 6类动作 | 160×120 | 公开 | nada.kth.se | TransNet | Acc: 100% |
| **UCF101** | 动作识别 | 13,320视频 | 101类 | 320×240 | 公开 | crcv.ucf.edu | TransNet | Acc: 96.5% |
| **HMDB51** | 动作识别 | 7,000视频 | 51类 | 多尺寸 | 公开 | serre-lab.clps.brown.edu | TransNet | Acc: 73.2% |

### 1.7 3D点云/模型数据集

| 数据集名称 | 领域 | 样本数量 | 标注类型 | 分辨率/维度 | 公开/私有 | 获取方式 | 使用论文 | 基准性能 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **ModelNet40** | 3D分类 | 12,311模型 | 40类 | - | 公开 | modelnet.cs.princeton.edu | Neural Varifolds | Acc: 92%+ |
| **ShapeNet** | 3D模型 | 3M+模型 | 55类 | - | 公开 | shapenet.org | Neural Varifolds | - |
| **ScanNet** | 3D场景 | 1,513扫描 | 语义分割 | - | 公开 | scannet.s3.amazonaws.com | - | - |

### 1.8 生物学数据集

| 数据集名称 | 领域 | 样本数量 | 标注类型 | 分辨率/维度 | 公开/私有 | 获取方式 | 使用论文 | 基准性能 |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **笠贝数据集** | 物种识别 | 多物种 | 物种标签 | 多尺寸 | 研究 | 作者 | 贝壳识别 | Acc: 92%+ |
| **贝壳形态数据** | 形态分析 | - | 形态测量 | - | 研究 | 作者 | 基因形态学AI | - |
| **类囊体CT** | 电子断层 | - | 结构分割 | - | 研究 | - | 电子断层分析 | - |
| **土壤CT** | 生物孔隙 | 8GB+ | 孔隙分割 | - | 研究 | - | 生物孔隙分割 | - |

---

## 二、按领域分类

### 2.1 医学影像 (Medical Imaging)

```
医学影像数据集 (9个)
├── MRI相关
│   ├── FastMRI (膝盖/脑部) ★★★★★
│   ├── IXI (脑部)
│   ├── CMRxRecon (心脏)
│   ├── OASIS (脑萎缩)
│   └── BraTS (脑肿瘤) ★★★★
├── 血管分割
│   ├── DRIVE (视网膜) ★★★★
│   └── STARE (视网膜)
├── X光
│   └── CheXpert (胸部) ★★★★
└── 其他
    └── 3T MRI (脑部结构)
```

### 2.2 自动驾驶 (Autonomous Driving)

```
自动驾驶数据集 (5个)
├── 3D目标检测
│   ├── KITTI ★★★★★
│   ├── nuScenes ★★★★★
│   └── Waymo Open ★★★★
├── 4D雷达
│   └── View of Delft (VoD) ★★★★
└── 大规模
    └── ONCE ★★★
```

### 2.3 遥感与林业 (Remote Sensing & Forestry)

```
遥感/林业数据集 (3个)
├── 高光谱
│   └── Indian Pines ★★★★
└── 激光雷达
    ├── LiDAR森林数据 (多站点)
    └── RC/MCRC/TIFFS (森林)
```

### 2.4 自然图像 (Natural Images)

```
自然图像数据集 (4个)
├── 分割基准
│   ├── BSDS500 ★★★★★
│   └── MSRC ★★★
├── 显著性
│   └── Alpert
└── 通用
    └── COCO ★★★★★
```

### 2.5 多模态/NLP (Multimodal/NLP)

```
多模态数据集 (4个)
├── 假新闻
│   ├── Weibo ★★★★
│   ├── Twitter ★★★★
│   └── Gossipcop ★★★★
└── 3D指代
    └── Talk2Car
```

---

## 三、按使用频率排名

### 3.1 高频使用 (5次以上)

| 排名 | 数据集 | 使用次数 | 主要论文 |
|:---:|:---:|:---:|:---|
| 1 | **KITTI** | 8+ | CornerPoint3D, DetectCloser, 多篇3D检测 |
| 2 | **FastMRI** | 5+ | HiFi-Mamba, HiFi-MambaV2, tCURLoRA |
| 3 | **BSDS500** | 5+ | SLaT, 分割-恢复, 多篇分割 |
| 4 | **nuScenes** | 4+ | Talk2Radar, CrossDomain LiDAR |
| 5 | **DRIVE** | 4+ | Tight-Frame, 血管分割系列 |

### 3.2 中频使用 (2-4次)

| 排名 | 数据集 | 使用次数 | 主要论文 |
|:---:|:---:|:---:|:---|
| 6 | STARE | 3 | 血管分割系列 |
| 7 | CheXpert | 2 | 医学分类 |
| 8 | Weibo/Twitter/Gossipcop | 各2 | GAMED |
| 9 | Indian Pines | 2 | 高光谱分割 |
| 10 | ModelNet40 | 2 | 点云分类 |

### 3.3 专项使用 (1次)

其余数据集主要为特定论文专用。

---

## 四、数据获取指南

### 4.1 医学影像数据集

#### FastMRI (推荐 ★★★★★)
```
官网: https://fastmri.org/dataset/
注册: 需要邮箱注册
下载: 
  - knee_singlecoil: ~30GB
  - knee_multicoil: ~100GB
  - brain_multicoil: ~200GB
许可: 学术研究免费
```

#### DRIVE & STARE
```
DRIVE: https://drive.grand-challenge.org/
STARE: http://ceas.clemson.edu/~ahoover/stare/
下载: 直接下载，免费
大小: 各约100MB
```

#### BraTS
```
官网: https://www.med.upenn.edu/cbica/brats/
注册: Synapse平台注册
下载: 需签署数据使用协议
大小: ~10GB
```

### 4.2 自动驾驶数据集

#### KITTI (推荐 ★★★★★)
```
官网: http://www.cvlibs.net/datasets/kitti/
注册: 无需注册
下载: 直接下载
大小: 
  - raw data: ~80GB
  - object detection: ~12GB
  - tracking: ~8GB
```

#### nuScenes (推荐 ★★★★★)
```
官网: https://www.nuscenes.org/nuscenes
注册: 需要邮箱注册
下载: 
  - Full dataset: ~350GB
  - Mini: ~4.5GB (测试用)
许可: CC BY-NC-SA 4.0
```

#### Waymo Open
```
官网: https://waymo.com/open/
注册: 需要申请审批
下载: 通过Google Cloud
大小: ~1TB
```

### 4.3 自然图像数据集

#### BSDS500
```
官网: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
下载: 直接下载
大小: ~80MB
```

#### COCO
```
官网: https://cocodataset.org/
下载: 直接下载或API
大小: 
  - train2017: ~18GB
  - val2017: ~1GB
  - annotations: ~500MB
```

### 4.4 快速下载脚本

```python
# datasets_download.py
"""数据集下载辅助脚本"""

import webbrowser

DATASET_URLS = {
    "fastmri": "https://fastmri.org/dataset/",
    "kitti": "http://www.cvlibs.net/datasets/kitti/",
    "nuscenes": "https://www.nuscenes.org/nuscenes",
    "bsds500": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/",
    "coco": "https://cocodataset.org/",
    "drive": "https://drive.grand-challenge.org/",
    "modelnet40": "https://modelnet.cs.princeton.edu/",
}

def download_dataset(name: str):
    if name.lower() in DATASET_URLS:
        url = DATASET_URLS[name.lower()]
        print(f"打开 {name} 下载页面: {url}")
        webbrowser.open(url)
    else:
        print(f"未知数据集: {name}")
        print(f"可用数据集: {list(DATASET_URLS.keys())}")
```

---

## 五、推荐数据集

### 5.1 综合推荐排名

#### Top 10 必备数据集

| 排名 | 数据集 | 推荐理由 | 难度 | 用途 |
|:---:|:---|:---|:---:|:---|
| 1 | **FastMRI** | MRI重建SOTA基准，社区活跃 | ⭐⭐⭐ | 医学影像深度学习 |
| 2 | **KITTI** | 3D检测经典基准，入门友好 | ⭐⭐ | 自动驾驶感知 |
| 3 | **nuScenes** | 大规模多传感器，工业级 | ⭐⭐⭐⭐ | 多模态融合 |
| 4 | **BSDS500** | 分割算法标准评估 | ⭐ | 图像分割 |
| 5 | **COCO** | 通用视觉任务基准 | ⭐⭐ | 检测/分割/关键点 |
| 6 | **DRIVE** | 血管分割金标准 | ⭐ | 医学图像分割 |
| 7 | **ModelNet40** | 3D深度学习入门 | ⭐⭐ | 点云分类 |
| 8 | **BraTS** | 脑肿瘤分割挑战赛 | ⭐⭐⭐ | 医学分割 |
| 9 | **CheXpert** | 大规模胸部X光 | ⭐⭐ | 医学分类 |
| 10 | **Weibo/Twitter** | 假新闻检测 | ⭐⭐ | 多模态NLP |

### 5.2 按研究方向推荐

#### 医学影像方向
```
入门: DRIVE → BraTS → FastMRI
进阶: FastMRI (多线圈) → IXI (多模态) → CMRxRecon (动态)
```

#### 自动驾驶方向
```
入门: KITTI → Waymo Open
进阶: nuScenes → ONCE (大规模)
专业: VoD (4D雷达) → 自采数据
```

#### 图像分割方向
```
入门: BSDS500 → MSRC
进阶: COCO → ADE20K
专业: 医学/遥感特定数据集
```

#### 点云处理方向
```
入门: ModelNet40 → ShapeNet
进阶: ScanNet → SemanticKITTI
专业: 自采TLS数据
```

---

*本目录由多智能体论文分析系统自动生成*
*最后更新: 2026年2月16日*
