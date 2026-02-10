# [3-09] 迁移学习动作识别 TransNet - 精读笔记

> **论文标题**: TransNet: Transfer Learning for Action Recognition with Deep Networks
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐ (中等)
> **重要性**: ⭐⭐⭐⭐ (重要，迁移学习在视频理解中的应用)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | TransNet: Transfer Learning for Action Recognition with Deep Networks |
| **作者** | X. Cai 等人 |
| **发表期刊** | IEEE Transactions on Multimedia |
| **发表年份** | 2023 |
| **关键词** | Transfer Learning, Action Recognition, Video Understanding, Domain Adaptation |
| **代码** | (请查看论文是否有开源代码) |

---

## 🎯 研究问题与动机

### 动作识别挑战

**标注数据稀缺问题**:
```
视频标注成本高:
- 需要人工观看整个视频
- 动作边界标注耗时
- 细粒度动作需要专业知识

数据集对比:
- ImageNet (图像): 1400万张图片
- Kinetics (视频): 30万个视频片段
- 自定义动作: 可能只有几百个样本
```

**迁移学习的解决方案**:
```
利用预训练模型:
- 从大规模数据集 (Kinetics, Sports1M) 预训练
- 迁移到目标域 (自定义动作)
- 大幅减少目标域标注需求
```

---

## 🔬 方法论详解

### 整体框架

```
┌─────────────────────────────────────────────────────────┐
│              TransNet 迁移学习框架                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │           阶段1: 源域预训练                       │   │
│  │                                                  │   │
│  │   大规模数据集 (Kinetics/Sports1M)               │   │
│  │         ↓                                        │   │
│  │   3D CNN 预训练                                   │   │
│  │   (C3D/I3D/SlowFast)                             │   │
│  │         ↓                                        │   │
│  │   通用动作表征学习                                │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │           阶段2: 目标域迁移                       │   │
│  │                                                  │   │
│  │   策略选择:                                       │   │
│  │   ├─ 特征提取 (Feature Extraction)               │   │
│  │   ├─ 微调 (Fine-tuning)                          │   │
│  │   └─ 领域适应 (Domain Adaptation) ⭐              │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │           阶段3: 目标域优化                       │   │
│  │                                                  │   │
│  │   - 小样本学习                                    │   │
│  │   - 时序建模                                      │   │
│  │   - 多模态融合                                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

### 核心方法1: 多阶段迁移策略

**迁移策略对比**:
```python
class TransferStrategy:
    """
    迁移学习策略

    三种主要策略:
    1. Feature Extraction: 冻结特征提取器
    2. Fine-tuning: 微调所有层
    3. Domain Adaptation: 对齐源域和目标域
    """

    @staticmethod
    def feature_extraction(model, target_data):
        """
        特征提取策略

        冻结预训练模型，只训练分类器
        """
        # 冻结所有层
        for param in model.parameters():
            param.requires_grad = False

        # 替换分类头
        num_classes = target_data.num_classes
        model.classifier = nn.Linear(model.feature_dim, num_classes)

        # 只训练分类头
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

        return model, optimizer

    @staticmethod
    def fine_tuning(model, target_data, lr=1e-4):
        """
        微调策略

        使用较小学习率微调所有层
        """
        # 分层学习率
        # 底层使用更小学习率，顶层使用较大学习率
        base_lr = lr
        param_groups = [
            {'params': model.backbone.layer1.parameters(), 'lr': base_lr * 0.1},
            {'params': model.backbone.layer2.parameters(), 'lr': base_lr * 0.2},
            {'params': model.backbone.layer3.parameters(), 'lr': base_lr * 0.5},
            {'params': model.backbone.layer4.parameters(), 'lr': base_lr},
            {'params': model.classifier.parameters(), 'lr': base_lr * 10},
        ]

        optimizer = torch.optim.Adam(param_groups)

        return model, optimizer

    @staticmethod
    def domain_adaptation(model, source_data, target_data):
        """
        领域适应策略

        最小化源域和目标域的分布差异
        """
        # 添加域分类器
        model.domain_classifier = nn.Sequential(
            nn.Linear(model.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # 源域 vs 目标域
        )

        return model
```

---

### 核心方法2: 时序特征对齐

```python
class TemporalFeatureAlignment(nn.Module):
    """
    时序特征对齐模块

    对齐源域和目标域的时序特征分布
    """
    def __init__(self, feature_dim: int, num_frames: int = 16):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frames = num_frames

        # 时序注意力
        self.temporal_attn = nn.MultiheadAttention(feature_dim, num_heads=8)

        # 域对齐投影
        self.domain_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def forward(self, source_features, target_features):
        """
        Args:
            source_features: (B, T, D) 源域特征
            target_features: (B, T, D) 目标域特征

        Returns:
            aligned_source: 对齐后的源域特征
            aligned_target: 对齐后的目标域特征
            alignment_loss: 对齐损失
        """
        # 时序建模
        source_temp, _ = self.temporal_attn(source_features, source_features, source_features)
        target_temp, _ = self.temporal_attn(target_features, target_features, target_features)

        # 域投影
        source_proj = self.domain_projector(source_temp)
        target_proj = self.domain_projector(target_temp)

        # 计算对齐损失 (最大均值差异 MMD)
        alignment_loss = self.compute_mmd(source_proj, target_proj)

        return source_proj, target_proj, alignment_loss

    def compute_mmd(self, X, Y, kernel='rbf'):
        """
        计算最大均值差异 (Maximum Mean Discrepancy)

        衡量两个分布的差异
        """
        if kernel == 'rbf':
            XX = torch.exp(-torch.cdist(X, X) ** 2 / (2 * X.size(-1)))
            YY = torch.exp(-torch.cdist(Y, Y) ** 2 / (2 * Y.size(-1)))
            XY = torch.exp(-torch.cdist(X, Y) ** 2 / (2 * X.size(-1)))

            mmd = XX.mean() + YY.mean() - 2 * XY.mean()
        else:
            # 线性核
            mmd = (X.mean(0) - Y.mean(0)).pow(2).sum()

        return mmd
```

---

### 核心方法3: 跨域注意力机制

```python
class CrossDomainAttention(nn.Module):
    """
    跨域注意力机制

    允许目标域样本关注源域的相关样本
    """
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)

        self.scale = (feature_dim // num_heads) ** -0.5

    def forward(self, target_features, source_features):
        """
        Args:
            target_features: (B_t, D) 目标域查询
            source_features: (B_s, D) 源域键值

        Returns:
            enhanced_features: 增强后的目标域特征
            attention_weights: 注意力权重
        """
        B_t, D = target_features.shape
        B_s = source_features.shape[0]

        # 投影
        Q = self.query_proj(target_features)  # (B_t, D)
        K = self.key_proj(source_features)    # (B_s, D)
        V = self.value_proj(source_features)  # (B_s, D)

        # 计算注意力
        attention_scores = torch.matmul(Q, K.T) * self.scale  # (B_t, B_s)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权聚合
        enhanced_features = torch.matmul(attention_weights, V)  # (B_t, D)

        # 残差连接
        output = target_features + enhanced_features

        return output, attention_weights
```

---

## 📊 实验结果

### 数据集

| 数据集 | 类型 | 类别数 | 视频数 | 用途 |
|:---|:---|:---:|:---:|:---|
| Kinetics-400 | 通用动作 | 400 | 306K | 源域预训练 |
| UCF-101 | 通用动作 | 101 | 13K | 目标域评估 |
| HMDB-51 | 通用动作 | 51 | 7K | 目标域评估 |
| Something-Something | 细粒度 | 174 | 108K | 目标域评估 |

### 迁移性能对比

| 方法 | UCF-101 | HMDB-51 | Something-Something |
|:---|:---:|:---:|:---:|
| From Scratch | 51.2% | 23.4% | 18.7% |
| Feature Extraction | 82.3% | 48.6% | 35.2% |
| Fine-tuning | 94.5% | 67.8% | 48.9% |
| **TransNet (DA)** | **95.8%** | **71.2%** | **52.3%** |

---

## 💡 可复用代码组件

### 组件1: 完整的迁移学习训练流程

```python
class TransferLearningTrainer:
    """
    迁移学习训练器

    完整的预训练→迁移→微调流程
    """
    def __init__(
        self,
        backbone: str = 'i3d',
        num_classes: int = 101,
        strategy: str = 'domain_adaptation'
    ):
        self.backbone = backbone
        self.num_classes = num_classes
        self.strategy = strategy

        # 加载预训练模型
        self.model = self._load_pretrained_model()

    def _load_pretrained_model(self):
        """加载预训练模型"""
        if self.backbone == 'i3d':
            model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        elif self.backbone == 'c3d':
            model = C3D(pretrained=True)
        else:
            raise ValueError(f"Unknown backbone: {self.backbone}")

        return model

    def train_source(self, source_loader, epochs=50):
        """源域预训练"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch in source_loader:
                videos, labels = batch
                videos, labels = videos.cuda(), labels.cuda()

                outputs = self.model(videos)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def train_target(self, target_loader, epochs=30):
        """目标域迁移训练"""
        if self.strategy == 'feature_extraction':
            self.model, optimizer = TransferStrategy.feature_extraction(
                self.model, target_loader.dataset
            )
        elif self.strategy == 'fine_tuning':
            self.model, optimizer = TransferStrategy.fine_tuning(
                self.model, target_loader.dataset
            )
        elif self.strategy == 'domain_adaptation':
            self.model = TransferStrategy.domain_adaptation(
                self.model, None, target_loader.dataset
            )
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            for batch in target_loader:
                videos, labels = batch
                videos, labels = videos.cuda(), labels.cuda()

                outputs = self.model(videos)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def evaluate(self, test_loader):
        """评估"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                videos, labels = batch
                videos, labels = videos.cuda(), labels.cuda()

                outputs = self.model(videos)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **迁移学习** | Transfer Learning | 将知识从源域迁移到目标域 |
| **领域适应** | Domain Adaptation | 减小源域和目标域的差异 |
| **MMD** | Maximum Mean Discrepancy | 分布差异度量 |
| **微调** | Fine-tuning | 调整预训练模型参数 |
| **特征提取** | Feature Extraction | 使用预训练特征 |
| **时序建模** | Temporal Modeling | 视频时间维度建模 |

---

## ✅ 复习检查清单

- [ ] 理解迁移学习在动作识别中的作用
- [ ] 掌握三种迁移策略的区别
- [ ] 理解领域适应的原理
- [ ] 了解时序特征对齐方法
- [ ] 能够选择合适的迁移策略

---

## 🤔 思考问题

1. **为什么视频比图像更需要迁移学习？**
   - 提示: 标注成本、数据量

2. **领域适应 vs 微调，如何选择？**
   - 提示: 域差异大小、数据量

3. **如何处理源域和目标域动作类别不同？**
   - 提示: 部分重叠、零样本

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
