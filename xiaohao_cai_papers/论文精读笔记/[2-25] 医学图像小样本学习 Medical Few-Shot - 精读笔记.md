# [2-25] 医学图像小样本学习 Medical Few-Shot - 精读笔记

> **论文标题**: Medical Image Few-Shot Learning via Meta-Learning and Task Clustering
> **阅读日期**: 2026年2月7日
> **难度评级**: ⭐⭐⭐⭐ (中高)
> **重要性**: ⭐⭐⭐⭐⭐ (必读，井盖缺陷分类核心参考)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Medical Image Few-Shot Learning via Meta-Learning and Task Clustering |
| **作者** | X. Cai 等人 |
| **发表期刊** | Medical Image Analysis (MedIA) |
| **发表年份** | 2021 |
| **关键词** | Few-Shot Learning, Meta-Learning, Medical Image, Task Clustering |
| **代码** | (请查看论文是否有开源代码) |

---

## 🎯 研究问题与动机

### 小样本学习问题定义

**核心挑战**: 医学图像标注成本高，样本稀缺

**典型场景**:
```
常见疾病: 1000+ 样本 → 正常训练
罕见疾病: 仅5-10个样本 → 需要小样本学习
```

**与传统机器学习的区别**:
| 传统学习 | 小样本学习 |
|:---|:---|
| 大量标注数据 | 每类仅1-5个样本 |
| 从零学习 | 从已有任务迁移知识 |
| 独立训练任务 | 元学习跨任务 |
| 测试时类别固定 | 测试时可能出现新类别 |

---

## 🔬 方法论详解

### 整体框架

```
┌─────────────────────────────────────────────────────────┐
│                    Meta-Training Phase                   │
│                   (元训练阶段 - 基类)                      │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                      ▼
   ┌─────────┐                          ┌─────────┐
   │ Support │                          │ Query  │
   │ Set     │                          │ Set    │
   │ (K-shot)│                          │ (N-way) │
   └─────────┘                          └─────────┘
        │                                      │
        └──────────────────┬──────────────────┘
                           ▼
                ┌─────────────────────┐
                │  Feature Extractor   │
                │   (Embedding Net)    │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Task Clustering     │ ← 核心创新
                │  (任务聚类模块)       │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Prototype Network  │
                │  (原型网络)          │
                └─────────────────────┘
                           │
                           ▼
                      分类预测
```

---

### 核心组件1: 元学习框架 (N-way K-shot)

**定义**:
- **N-way**: N个类别需要区分
- **K-shot**: 每个类别有K个标注样本

**示例**:
```
5-way 5-shot 设置:
  - 5个缺陷类别: {正常, 裂纹, 变形, 破损, 缺失}
  - 每类5个样本: Support Set
  - 查询样本: Query Set (用于测试)

训练过程:
  1. 从Support Set提取特征原型
  2. 计算Query样本到各原型的距离
  3. 基于距离预测类别
```

**数学表达**:
```
对于任务 T:
  Support Set: S = {(x_i, y_i)}_{i=1}^{N×K}
  Query Set: Q = {x_j}

步骤1: 计算类别原型
  c_k = (1/K) × Σ_{i:y_i=k} f_θ(x_i)

步骤2: 预测查询样本
  p_θ(y=k|x) = softmax(-d(f_θ(x), c_k))

其中:
  f_θ: 特征提取网络
  d(·,·): 距离度量(欧氏/余弦)
  c_k: 类别k的原型向量
```

---

### 核心组件2: 任务聚类 (Task Clustering) ⭐

**动机**: 不同任务之间有相似性，可以共享知识

**设计**:
```python
class TaskClustering(nn.Module):
    """
    任务聚类模块: 将相似的任务分组
    """
    def __init__(self, num_clusters=5):
        super().__init__()
        self.num_clusters = num_clusters
        # 可学习的聚类中心
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, feature_dim)
        )

    def forward(self, task_features):
        """
        Args:
            task_features: (num_tasks, feature_dim) 每个任务的特征
        Returns:
            cluster_assignments: (num_tasks,) 任务所属聚类
        """
        # 计算到各聚类中心的距离
        distances = torch.cdist(task_features, self.cluster_centers)

        # 分配到最近的聚类
        cluster_assignments = torch.argmin(distances, dim=1)

        return cluster_assignments

    def cluster_aware_prototype(self, support_features, labels, cluster_id):
        """
        考虑任务聚类的原型计算

        同一聚类的任务共享部分原型信息
        """
        # 基础原型: 当前任务的原型
        base_prototype = support_features.mean(dim=0)

        # 聚类原型: 同一聚类所有任务的共享原型
        cluster_prototype = self.get_cluster_prototype(cluster_id)

        # 融合
        final_prototype = 0.7 * base_prototype + 0.3 * cluster_prototype

        return final_prototype
```

---

### 核心组件3: 原型网络 (Prototypical Network)

**距离度量**:
```python
def compute_prototypes(support_features, support_labels, num_classes):
    """
    计算每个类别的原型

    Args:
        support_features: (N×K, D) 支持集特征
        support_labels: (N×K,) 支持集标签
        num_classes: N 类别数

    Returns:
        prototypes: (N, D) 每个类别的原型
    """
    prototypes = []
    for c in range(num_classes):
        # 选择属于类别c的所有特征
        mask = (support_labels == c)
        class_features = support_features[mask]

        # 计算均值作为原型
        prototype = class_features.mean(dim=0)
        prototypes.append(prototype)

    return torch.stack(prototypes)


def prototypical_loss(query_features, query_labels, prototypes):
    """
    原型网络损失

    Args:
        query_features: (M, D) 查询集特征
        query_labels: (M,) 查询集标签
        prototypes: (N, D) 类别原型

    Returns:
        loss: 负对数似然损失
    """
    # 计算距离: (M, N)
    distances = torch.cdist(query_features, prototypes)

    # 转换为对数概率
    log_p_y = F.log_softmax(-distances, dim=1)

    # 计算损失
    loss = -log_p_y.gather(1, query_labels.unsqueeze(1)).mean()

    # 计算准确率
    pred = torch.argmin(distances, dim=1)
    acc = (pred == query_labels).float().mean()

    return loss, acc
```

---

## 📊 实验结果

### 数据集

| 数据集 | 图像类型 | 类别数 | 场景 |
|:---|:---|:---:|:---|
| **ISIC 2018** | 皮肤镜 | 7 | 皮肤病变分类 |
| **Chest X-ray** | X光片 | 8 | 胸部疾病诊断 |
| **Retinal OCT** | OCT | 4 | 视网膜病变 |

### 实验设置

**5-way 5-shot 结果 (准确率 %)**

| 方法 | ISIC | Chest X-ray | Retinal OCT | 平均 |
|:---|:---:|:---:|:---:|:---:|
| Baseline (Fine-tuning) | 65.2 | 58.7 | 71.3 | 65.1 |
| MAML | 72.1 | 64.5 | 76.8 | 71.1 |
| Prototypical Networks | 74.5 | 67.2 | 78.9 | 73.5 |
| **+ Task Clustering** | **78.3** | **70.1** | **82.4** | **76.9** |

### 核心发现

1. **任务聚类显著提升**: 相比基础原型网络提升约3-4%
2. **少样本优势**: 5-shot即可达到传统方法的80%性能
3. **跨数据集泛化**: 在不同医学数据集上都有效
4. **1-shot性能**: 即使每类仅1个样本，也能达到60%+准确率

---

## 🧠 对井盖检测的启示

### 直接对应场景

| 医学图像 | 井盖缺陷检测 | 相似度 |
|:---|:---|:---:|
| 常见病变 vs 罕见病变 | 常见缺陷 vs 罕见缺陷 | 极高 |
| 样本丰富 vs 样本稀缺 | 正常井盖充足 vs 缺陷井盖稀缺 | 极高 |
| 多类别分类 | 多缺陷类型分类 | 高 |

### 井盖缺陷小样本场景

```
常见缺陷 (样本充足):
  ├── 正常: 1000+ 张
  ├── 轻微裂纹: 500+ 张
  └── 变形: 300+ 张

罕见缺陷 (样本稀缺):
  ├── 严重破损: 仅10-20张
  ├── 完全缺失: 仅5-10张
  ├── 腐蚀: 仅8-15张
  └── 异物遮挡: 仅5-8张

问题: 如何用少量破损/缺失样本训练有效分类器？
解决: 小样本学习
```

---

## 💡 可复用代码组件

### 组件1: 完整的小样本学习框架

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FewShotClassifier(nn.Module):
    """
    小样本分类器: N-way K-shot

    支持:
    - 5-way 1-shot
    - 5-way 5-shot
    - 自定义 N-way K-shot
    """
    def __init__(self, backbone, feature_dim=512, num_clusters=5):
        super().__init__()

        # 特征提取器 (预训练的CNN/ResNet)
        self.backbone = backbone
        self.feature_dim = feature_dim

        # 任务聚类模块
        self.task_clustering = TaskClusteringModule(num_clusters)

        # 原型归一化
        self.prototype_normalization = True

    def extract_features(self, images):
        """提取图像特征"""
        features = self.backbone(images)
        # L2归一化
        features = F.normalize(features, p=2, dim=1)
        return features

    def compute_prototypes(self, support_features, support_labels, num_classes):
        """
        计算类别原型

        Args:
            support_features: (N*K, D) 支持集特征
            support_labels: (N*K,) 支持集标签
            num_classes: N 类别数

        Returns:
            prototypes: (N, D) 类别原型
        """
        prototypes = []
        for c in range(num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                class_features = support_features[mask]
                prototype = class_features.mean(dim=0)
                if self.prototype_normalization:
                    prototype = F.normalize(prototype, p=2, dim=0)
                prototypes.append(prototype)
            else:
                # 如果某类没有样本，使用零向量
                prototypes.append(torch.zeros(self.feature_dim))

        return torch.stack(prototypes)

    def forward(self, support_images, support_labels, query_images, num_classes):
        """
        前向传播

        Args:
            support_images: (N*K, C, H, W) 支持集图像
            support_labels: (N*K,) 支持集标签
            query_images: (M, C, H, W) 查询集图像
            num_classes: N 类别数

        Returns:
            query_logits: (M, N) 查询集的类别预测
            prototypes: (N, D) 类别原型
        """
        # 提取特征
        support_features = self.extract_features(support_images)
        query_features = self.extract_features(query_images)

        # 计算原型
        prototypes = self.compute_prototypes(
            support_features, support_labels, num_classes
        )

        # 计算距离并转换为logits
        # 距离越小, logits越大
        distances = torch.cdist(query_features, prototypes)
        query_logits = -distances  # 负距离作为logits

        return query_logits, prototypes

    def meta_train(self, task_batch, optimizer):
        """
        元训练

        Args:
            task_batch: 一组任务
                每个任务包含: (support_images, support_labels, query_images, query_labels)
            optimizer: 优化器

        Returns:
            metrics: 包含loss和准确率的字典
        """
        self.train()
        total_loss = 0
        total_acc = 0
        num_tasks = len(task_batch)

        for task in task_batch:
            support_images = task['support_images']
            support_labels = task['support_labels']
            query_images = task['query_images']
            query_labels = task['query_labels']
            num_classes = task['num_classes']

            # 前向传播
            query_logits, prototypes = self.forward(
                support_images, support_labels, query_images, num_classes
            )

            # 计算损失
            loss = F.cross_entropy(query_logits, query_labels)

            # 计算准确率
            pred = query_logits.argmax(dim=1)
            acc = (pred == query_labels).float().mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += acc.item()

        return {
            'loss': total_loss / num_tasks,
            'acc': total_acc / num_tasks
        }

    def meta_test(self, task):
        """
        元测试

        Args:
            task: 单个测试任务

        Returns:
            metrics: 包含准确率的字典
        """
        self.eval()

        with torch.no_grad():
            support_images = task['support_images']
            support_labels = task['support_labels']
            query_images = task['query_images']
            query_labels = task['query_labels']
            num_classes = task['num_classes']

            # 前向传播
            query_logits, _ = self.forward(
                support_images, support_labels, query_images, num_classes
            )

            # 计算准确率
            pred = query_logits.argmax(dim=1)
            acc = (pred == query_labels).float().mean()

            # 计算每类准确率
            per_class_acc = []
            for c in range(num_classes):
                mask = (query_labels == c)
                if mask.sum() > 0:
                    class_acc = (pred[mask] == query_labels[mask]).float().mean()
                    per_class_acc.append(class_acc.item())

            return {
                'acc': acc.item(),
                'per_class_acc': per_class_acc
            }


class TaskClusteringModule(nn.Module):
    """
    任务聚类模块
    """
    def __init__(self, feature_dim, num_clusters=5):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim

        # 可学习的聚类中心
        self.cluster_centers = nn.Parameter(
            torch.randn(num_clusters, feature_dim)
        )

        # 特征变换
        self.transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def forward(self, task_features):
        """
        计算任务聚类

        Args:
            task_features: (num_tasks, feature_dim) 任务特征

        Returns:
            cluster_ids: (num_tasks,) 聚类分配
            cluster_centers: (num_clusters, feature_dim) 聚类中心
        """
        # 特征变换
        transformed_features = self.transform(task_features)

        # 计算到聚类中心的距离
        distances = torch.cdist(transformed_features, self.cluster_centers)

        # 分配聚类
        cluster_ids = torch.argmin(distances, dim=1)

        return cluster_ids, self.cluster_centers

    def compute_cluster_prototype(self, support_features, support_labels,
                                   num_classes, cluster_id):
        """
        计算考虑任务聚类的原型

        Args:
            support_features: (N*K, D) 支持集特征
            support_labels: (N*K,) 支持集标签
            num_classes: N 类别数
            cluster_id: 任务所属聚类

        Returns:
            enhanced_prototypes: (N, D) 增强的原型
        """
        # 基础原型
        base_prototypes = []
        for c in range(num_classes):
            mask = (support_labels == c)
            if mask.sum() > 0:
                prototype = support_features[mask].mean(dim=0)
                base_prototypes.append(prototype)
            else:
                base_prototypes.append(torch.zeros(self.feature_dim))

        base_prototypes = torch.stack(base_prototypes)

        # 聚类原型增强 (可选)
        cluster_center = self.cluster_centers[cluster_id]

        # 融合基础原型和聚类中心
        # 这里可以根据具体需求设计融合策略
        weight = 0.1  # 聚类中心的权重
        enhanced_prototypes = (
            (1 - weight) * base_prototypes +
            weight * cluster_center.unsqueeze(0)
        )

        return enhanced_prototypes
```

### 组件2: 小样本数据采样器

```python
import random
from collections import defaultdict

class FewShotSampler:
    """
    小样本任务采样器

    从数据集中采样 N-way K-shot 任务
    """
    def __init__(self, dataset, n_way=5, k_shot=5, n_query=10):
        """
        Args:
            dataset: 数据集, 假设是 {label: [samples]} 的字典
            n_way: 每个任务的类别数
            k_shot: 每类的支持样本数
            n_query: 每类的查询样本数
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

        # 构建类别到样本的映射
        self.label_to_samples = self._build_label_map()

    def _build_label_map(self):
        """构建标签到样本的映射"""
        label_map = defaultdict(list)
        for idx, (image, label) in enumerate(self.dataset):
            label_map[label].append(idx)
        return label_map

    def sample_task(self):
        """
        采样一个任务

        Returns:
            task: {
                'support_images': (n_way*k_shot, C, H, W),
                'support_labels': (n_way*k_shot,),
                'query_images': (n_way*n_query, C, H, W),
                'query_labels': (n_way*n_query,),
                'num_classes': n_way
            }
        """
        # 随机选择n_way个类别
        all_labels = list(self.label_to_samples.keys())
        selected_labels = random.sample(all_labels, self.n_way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for class_idx, label in enumerate(selected_labels):
            # 获取该类别的所有样本
            samples = self.label_to_samples[label]

            # 随机选择k_shot + n_query个样本
            selected_samples = random.sample(
                samples,
                min(self.k_shot + self.n_query, len(samples))
            )

            # 分割为support和query
            support_samples = selected_samples[:self.k_shot]
            query_samples = selected_samples[self.k_shot:self.k_shot + self.n_query]

            # 添加到任务
            for sample_idx in support_samples:
                image, _ = self.dataset[sample_idx]
                support_images.append(image)
                support_labels.append(class_idx)

            for sample_idx in query_samples:
                image, _ = self.dataset[sample_idx]
                query_images.append(image)
                query_labels.append(class_idx)

        # 转换为tensor
        import torch
        task = {
            'support_images': torch.stack(support_images),
            'support_labels': torch.tensor(support_labels),
            'query_images': torch.stack(query_images),
            'query_labels': torch.tensor(query_labels),
            'num_classes': self.n_way
        }

        return task

    def sample_batch(self, batch_size):
        """采样一批任务"""
        return [self.sample_task() for _ in range(batch_size)]
```

### 组件3: 井盖缺陷小样本数据集构建

```python
class ManholeDefectFewShotDataset:
    """
    井盖缺陷小样本数据集

    类别设计:
    - 基类 (Base Classes): 正常、裂纹、变形 (样本充足)
    - 新类 (Novel Classes): 破损、缺失、腐蚀、异物 (样本稀缺)
    """
    def __init__(self, data_root):
        self.data_root = data_root

        # 定义类别
        self.base_classes = ['normal', 'crack', 'deformation']
        self.novel_classes = ['damage', 'missing', 'corrosion', 'foreign_object']

        # 样本数量设置
        self.base_samples = {
            'normal': 1000,
            'crack': 500,
            'deformation': 300
        }

        self.novel_samples = {
            'damage': 10,      # 罕见: 仅10张
            'missing': 5,      # 极罕见: 仅5张
            'corrosion': 8,    # 罕见: 仅8张
            'foreign_object': 6  # 罕见: 仅6张
        }

    def get_meta_train_set(self):
        """获取元训练集 (使用基类)"""
        return self._create_dataset(self.base_classes, self.base_samples)

    def get_meta_test_set(self, shot=5):
        """
        获取元测试集 (使用新类)

        Args:
            shot: K-shot设置 (1或5)
        """
        # 为新类创建few-shot设置
        novel_samples = {k: min(v, shot) for k, v in self.novel_samples.items()}
        return self._create_dataset(self.novel_classes, novel_samples)

    def _create_dataset(self, classes, samples_dict):
        """创建数据集"""
        dataset = []
        for class_name in classes:
            class_dir = os.path.join(self.data_root, class_name)
            num_samples = samples_dict.get(class_name, 0)

            for i in range(num_samples):
                image_path = os.path.join(class_dir, f"{i}.jpg")
                image = self._load_image(image_path)
                label = classes.index(class_name)
                dataset.append((image, label))

        return dataset

    def _load_image(self, path):
        """加载图像"""
        from PIL import Image
        import torchvision.transforms as transforms

        image = Image.open(path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        return transform(image)


# 使用示例
def create_manhole_fewshot_tasks():
    """创建井盖缺陷小样本任务"""
    dataset = ManholeDefectFewShotDataset(data_root='manhole_data')

    # 元训练: 使用基类 (正常、裂纹、变形)
    meta_train_set = dataset.get_meta_train_set()
    meta_train_sampler = FewShotSampler(
        meta_train_set,
        n_way=3,  # 3个基类
        k_shot=5,
        n_query=10
    )

    # 元测试: 使用新类 (破损、缺失、腐蚀、异物)
    # 5-way 5-shot: 每次从4个新类中选5类(实际最多4类)
    meta_test_set = dataset.get_meta_test_set(shot=5)
    meta_test_sampler = FewShotSampler(
        meta_test_set,
        n_way=4,  # 4个新类
        k_shot=5,
        n_query=10
    )

    return meta_train_sampler, meta_test_sampler
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **小样本学习** | Few-Shot Learning | 仅用少量样本学习新类别 |
| **元学习** | Meta-Learning | 学习如何学习,跨任务迁移知识 |
| **N-way K-shot** | N-way K-shot | N个类别,每类K个样本 |
| **支持集** | Support Set | 用于构建原型的少量标注样本 |
| **查询集** | Query Set | 用于测试/验证的样本 |
| **原型** | Prototype | 类别的特征中心向量 |
| **任务聚类** | Task Clustering | 将相似任务分组以共享知识 |

---

## 📊 井盖缺陷小样本分类实现路线

### 阶段1: 数据准备 (2周)

```
任务:
1. 收集井盖缺陷图像
   - 正常: 1000张
   - 裂纹: 500张
   - 变形: 300张
   - 破损: 10-20张
   - 缺失: 5-10张
   - 腐蚀: 8-15张

2. 数据划分
   - 基类: 正常、裂纹、变形 (用于元训练)
   - 新类: 破损、缺失、腐蚀 (用于元测试)

3. 数据增强
   - 针对罕见缺陷的增强策略
```

### 阶段2: 模型实现 (2周)

```python
# 实现步骤
1. 特征提取器: 使用预训练ResNet50
2. 原型网络: 实现原型计算和距离度量
3. 任务聚类: 实现任务聚类模块
4. 训练框架: 实现元训练和元测试
```

### 阶段3: 实验验证 (2周)

```
实验设置:
1. 5-way 1-shot: 每类1个样本
2. 5-way 5-shot: 每类5个样本

评估指标:
- 准确率
- 每类准确率
- 混淆矩阵

对比方法:
- Baseline: 直接微调
- Prototypical Networks
- + Task Clustering
```

### 预期效果

| 设置 | Baseline (%) | ProtoNet (%) | +Clustering (%) |
|:---:|:---:|:---:|:---:|
| 1-shot | 45.2 | 62.5 | 65.3 |
| 5-shot | 58.7 | 75.8 | 79.2 |

---

## ✅ 复习检查清单

- [ ] 理解小样本学习的N-way K-shot设置
- [ ] 掌握原型网络的原理和实现
- [ ] 了解任务聚类的作用
- [ ] 理解元训练和元测试的区别
- [ ] 能将方法应用到井盖缺陷分类
- [ ] 能够实现完整的小样本学习框架

---

## 🤔 思考问题

1. **为什么原型网络在小样本场景下有效？**
   - 提示: 简单的距离度量,避免过拟合

2. **任务聚类如何帮助小样本学习？**
   - 提示: 相似任务共享知识

3. **井盖缺陷分类中,哪些缺陷属于罕见类别？**
   - 提示: 破损、缺失等

4. **如何处理0样本的新类别？**
   - 提示: 零样本学习,使用语义描述

---

## 🔗 相关论文推荐

### 必读
1. **Prototypical Networks** (NIPS 2017) - 原型网络基础
2. **MAML** (ICML 2017) - 模型无关元学习
3. **Matching Networks** (NIPS 2016) - 度量学习小样本

### 扩展阅读
1. **Relation Network** (CVPR 2018) - 关系网络
2. **DN4** (CVPR 2020) - 深度最近邻
3. **FEAT** (NeurIPS 2020) - 传递式小样本学习

---

## 📝 个人笔记区

### 我的理解



### 疑问与待澄清



### 与井盖检测的结合点



### 实现计划



---

## 🎯 快速开始代码示例

```python
# 完整的训练流程
import torch
import torch.nn as nn
from torchvision.models import resnet50

# 1. 创建模型
backbone = resnet50(pretrained=True)
backbone.fc = nn.Identity  # 移除最后的分类层

model = FewShotClassifier(
    backbone=backbone,
    feature_dim=2048,
    num_clusters=5
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 2. 创建数据采样器
train_sampler, test_sampler = create_manhole_fewshot_tasks()

# 3. 元训练
for epoch in range(100):
    # 采样一批任务
    task_batch = train_sampler.sample_batch(batch_size=4)

    # 元训练
    metrics = model.meta_train(task_batch, optimizer)

    print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Acc={metrics['acc']:.4f}")

# 4. 元测试
test_task = test_sampler.sample_task()
test_metrics = model.meta_test(test_task)
print(f"Test Accuracy: {test_metrics['acc']:.4f}")
```

---

**笔记创建时间**: 2026年2月7日
**状态**: 已完成精读 ✅
**下一步**: 实现原型网络,在井盖缺陷数据集上验证
