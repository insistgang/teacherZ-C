# [3-03] 自监督图神经网络 LL4G - 精读笔记

> **论文标题**: LL4G: Learning to Learn for Graph Neural Networks via Knowledge Distillation
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐⭐ (中高)
> **重要性**: ⭐⭐⭐⭐ (重要，图神经网络自监督学习)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | LL4G: Learning to Learn for Graph Neural Networks via Knowledge Distillation |
| **作者** | X. Cai 等人 |
| **发表期刊** | IEEE Transactions on Neural Networks and Learning Systems |
| **发表年份** | 2023 |
| **关键词** | Graph Neural Networks, Self-Supervised Learning, Knowledge Distillation, Meta-Learning |
| **代码** | (请查看论文是否有开源代码) |

---

## 🎯 研究问题与动机

### 图神经网络挑战

**标注数据稀缺问题**:
```
图数据标注困难:
- 节点分类: 需要专家标注每个节点
- 图分类: 需要标注整个图
- 链接预测: 需要知道所有边关系

现实场景:
- 社交网络: 用户标签难以获取
- 分子图: 生物活性实验昂贵
- 知识图谱: 关系标注耗时
```

**自监督学习的优势**:
```
无需人工标注
从图结构本身学习
学习通用的节点/图表征
```

---

## 🔬 方法论详解

### 整体框架

```
┌─────────────────────────────────────────────────────────┐
│                   LL4G 整体框架                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │           阶段1: 自监督预训练                      │   │
│  │                                                  │   │
│  │   原始图 → 数据增强 → 对比学习 → 节点表征          │   │
│  │       ↓         ↓         ↓                      │   │
│  │    GNN编码器   多视图    InfoNCE损失              │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │           阶段2: 知识蒸馏                         │   │
│  │                                                  │   │
│  │   教师模型 (预训练GNN)                           │   │
│  │        ↓ 蒸馏知识                                │   │
│  │   学生模型 (轻量GNN)                             │   │
│  │        ↓                                        │   │
│  │   轻量级但高性能的节点表征                        │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│                          ▼                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │           阶段3: 下游任务微调                      │   │
│  │                                                  │   │
│  │   节点分类 / 图分类 / 链接预测                    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

### 核心方法1: 图对比学习

**图数据增强策略**:
```python
class GraphAugmentation:
    """
    图数据增强操作

    针对图结构的多种增强方式
    """
    def __init__(self, aug_type='edge_drop', aug_ratio=0.2):
        self.aug_type = aug_type
        self.aug_ratio = aug_ratio

    def __call__(self, x, edge_index):
        """
        对图进行增强

        Args:
            x: 节点特征 (num_nodes, feature_dim)
            edge_index: 边索引 (2, num_edges)

        Returns:
            aug_x: 增强后的节点特征
            aug_edge_index: 增强后的边索引
        """
        if self.aug_type == 'edge_drop':
            return self.edge_dropout(x, edge_index)
        elif self.aug_type == 'node_drop':
            return self.node_dropout(x, edge_index)
        elif self.aug_type == 'feature_mask':
            return self.feature_masking(x, edge_index)
        elif self.aug_type == 'subgraph':
            return self.subgraph_sampling(x, edge_index)
        else:
            return x, edge_index

    def edge_dropout(self, x, edge_index):
        """边丢弃: 随机移除部分边"""
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) > self.aug_ratio
        aug_edge_index = edge_index[:, mask]
        return x, aug_edge_index

    def node_dropout(self, x, edge_index):
        """节点丢弃: 随机移除部分节点及其边"""
        num_nodes = x.size(0)
        mask = torch.rand(num_nodes) > self.aug_ratio
        # 保留的节点
        keep_nodes = mask.nonzero(as_tuple=True)[0]
        # 重新索引边
        node_map = {old.item(): new for new, old in enumerate(keep_nodes)}
        mask_edges = mask[edge_index[0]] & mask[edge_index[1]]
        aug_edge_index = edge_index[:, mask_edges]
        aug_edge_index[0] = torch.tensor([node_map[n.item()] for n in aug_edge_index[0]])
        aug_edge_index[1] = torch.tensor([node_map[n.item()] for n in aug_edge_index[1]])
        aug_x = x[mask]
        return aug_x, aug_edge_index

    def feature_masking(self, x, edge_index):
        """特征掩码: 随机掩码部分特征维度"""
        aug_x = x.clone()
        num_features = x.size(1)
        mask = torch.rand(num_features) < self.aug_ratio
        aug_x[:, mask] = 0
        return aug_x, edge_index

    def subgraph_sampling(self, x, edge_index):
        """子图采样: 随机游走采样子图"""
        # 从随机节点开始采样
        num_nodes = x.size(0)
        start_node = torch.randint(0, num_nodes, (1,)).item()

        # 随机游走采样
        sampled_nodes = {start_node}
        for _ in range(int(num_nodes * (1 - self.aug_ratio))):
            # 找到当前节点的邻居
            neighbors = edge_index[1][edge_index[0] == start_node].tolist()
            if neighbors:
                start_node = random.choice(neighbors)
                sampled_nodes.add(start_node)

        sampled_nodes = sorted(list(sampled_nodes))
        node_map = {old: new for new, old in enumerate(sampled_nodes)}

        # 提取子图
        aug_x = x[sampled_nodes]
        mask = torch.tensor([n in sampled_nodes for n in edge_index[0]]) & \
               torch.tensor([n in sampled_nodes for n in edge_index[1]])
        aug_edge_index = edge_index[:, mask]
        aug_edge_index[0] = torch.tensor([node_map[n.item()] for n in aug_edge_index[0]])
        aug_edge_index[1] = torch.tensor([node_map[n.item()] for n in aug_edge_index[1]])

        return aug_x, aug_edge_index
```

**对比学习损失 (InfoNCE)**:
```python
class GraphContrastiveLearning(nn.Module):
    """
    图对比学习框架

    使用InfoNCE损失学习节点表征
    """
    def __init__(self, encoder, projection_dim=128, temperature=0.5):
        super().__init__()
        self.encoder = encoder
        self.projection_head = nn.Sequential(
            nn.Linear(encoder.hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.temperature = temperature

    def forward(self, x, edge_index):
        """
        前向传播

        Args:
            x: 节点特征
            edge_index: 边索引

        Returns:
            loss: 对比学习损失
        """
        # 生成两个增强视图
        aug1 = GraphAugmentation(aug_type='edge_drop', aug_ratio=0.2)
        aug2 = GraphAugmentation(aug_type='feature_mask', aug_ratio=0.2)

        x1, edge_index1 = aug1(x, edge_index)
        x2, edge_index2 = aug2(x, edge_index)

        # 编码
        h1 = self.encoder(x1, edge_index1)
        h2 = self.encoder(x2, edge_index2)

        # 投影
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)

        # 对比损失
        loss = self.infonce_loss(z1, z2)

        return loss

    def infonce_loss(self, z1, z2):
        """
        InfoNCE对比损失

        Args:
            z1, z2: 两个视图的投影特征 (num_nodes, projection_dim)

        Returns:
            loss: 对比损失
        """
        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 计算相似度矩阵
        similarity = torch.mm(z1, z2.t()) / self.temperature

        # 正样本: 对角线 (同一节点的两个视图)
        # 负样本: 非对角线 (不同节点)
        labels = torch.arange(z1.size(0)).to(z1.device)

        # 对称损失
        loss1 = F.cross_entropy(similarity, labels)
        loss2 = F.cross_entropy(similarity.t(), labels)

        return (loss1 + loss2) / 2
```

---

### 核心方法2: 知识蒸馏

```python
class KnowledgeDistillationForGNN(nn.Module):
    """
    GNN知识蒸馏

    将预训练教师模型的知识迁移到轻量学生模型
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5
    ):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(self, x, edge_index, labels=None):
        """
        前向传播

        Args:
            x: 节点特征
            edge_index: 边索引
            labels: 节点标签 (可选，用于监督)

        Returns:
            loss: 蒸馏损失
            student_logits: 学生模型输出
        """
        # 教师模型推理 (不计算梯度)
        with torch.no_grad():
            teacher_logits = self.teacher_model(x, edge_index)
            teacher_features = self.teacher_model.get_embeddings(x, edge_index)

        # 学生模型推理
        student_logits = self.student_model(x, edge_index)
        student_features = self.student_model.get_embeddings(x, edge_index)

        # 计算蒸馏损失
        loss = self.compute_distillation_loss(
            student_logits, teacher_logits,
            student_features, teacher_features,
            labels
        )

        return loss, student_logits

    def compute_distillation_loss(
        self,
        student_logits, teacher_logits,
        student_features, teacher_features,
        labels=None
    ):
        """
        计算蒸馏损失

        包含:
        1. 软目标蒸馏 (Soft Target Distillation)
        2. 特征蒸馏 (Feature Distillation)
        3. 监督损失 (可选)
        """
        losses = {}

        # 1. 软目标蒸馏
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        loss_soft = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        loss_soft *= (self.temperature ** 2)
        losses['soft'] = loss_soft

        # 2. 特征蒸馏
        loss_feat = F.mse_loss(student_features, teacher_features)
        losses['feature'] = loss_feat

        # 3. 监督损失 (如果有标签)
        if labels is not None:
            loss_sup = F.cross_entropy(student_logits, labels)
            losses['supervised'] = loss_sup

        # 总损失
        total_loss = losses['soft'] + 0.5 * losses['feature']
        if 'supervised' in losses:
            total_loss += self.alpha * losses['supervised']

        return total_loss
```

---

### 核心方法3: 元学习优化

```python
class MetaLearningOptimizer:
    """
    元学习优化器

    学习如何快速适应新任务
    """
    def __init__(self, model, meta_lr=1e-3, inner_lr=1e-2):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

    def meta_train_step(self, task_batch):
        """
        元训练步骤

        Args:
            task_batch: 一批图任务

        Returns:
            meta_loss: 元损失
        """
        meta_loss = 0

        for task in task_batch:
            # 内循环: 在支持集上适应
            support_loss = self.inner_loop(task['support'])

            # 外循环: 在查询集上评估
            query_loss = self.outer_loop(task['query'])

            meta_loss += query_loss

        # 元优化
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def inner_loop(self, support_data, num_steps=5):
        """
        内循环适应

        在支持集上进行几步梯度下降
        """
        # 创建临时参数副本
        fast_weights = [p.clone() for p in self.model.parameters()]

        for _ in range(num_steps):
            # 前向传播
            loss = self.compute_loss(support_data, fast_weights)

            # 计算梯度
            grads = torch.autograd.grad(loss, fast_weights, create_graph=True)

            # 更新快速权重
            fast_weights = [w - self.inner_lr * g for w, g in zip(fast_weights, grads)]

        return fast_weights

    def outer_loop(self, query_data, fast_weights):
        """
        外循环评估

        使用适应后的参数在查询集上计算损失
        """
        loss = self.compute_loss(query_data, fast_weights)
        return loss
```

---

## 📊 实验结果

### 数据集

| 数据集 | 节点数 | 边数 | 特征维度 | 任务类型 |
|:---|:---:|:---:|:---:|:---|
| Cora | 2,708 | 5,429 | 1,433 | 节点分类 |
| CiteSeer | 3,327 | 4,732 | 3,703 | 节点分类 |
| PubMed | 19,717 | 44,338 | 500 | 节点分类 |
| PPI | 56,944 | 818,716 | 50 | 多标签分类 |

### 性能对比

| 方法 | Cora | CiteSeer | PubMed |
|:---|:---:|:---:|:---:|
| GCN (监督) | 81.5% | 70.3% | 79.0% |
| GAT (监督) | 83.0% | 72.5% | 79.0% |
| DGI (自监督) | 82.3% | 71.8% | 76.8% |
| GRACE (自监督) | 83.5% | 73.0% | 80.5% |
| **LL4G** | **84.2%** | **73.8%** | **81.2%** |

---

## 💡 可复用代码组件

### 组件1: 完整的自监督GNN训练流程

```python
class SelfSupervisedGNNTrainer:
    """
    自监督GNN训练器

    完整的预训练+微调流程
    """
    def __init__(self, encoder, device='cuda'):
        self.encoder = encoder.to(device)
        self.device = device
        self.contrastive_model = GraphContrastiveLearning(encoder)

    def pretrain(self, data_loader, epochs=100, lr=1e-3):
        """
        自监督预训练
        """
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                x = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)

                # 对比学习
                loss = self.contrastive_model(x, edge_index)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader):.4f}")

        return self.encoder

    def finetune(self, train_data, val_data, epochs=50, lr=1e-3):
        """
        下游任务微调
        """
        # 添加分类头
        classifier = nn.Linear(self.encoder.hidden_dim, num_classes).to(self.device)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(classifier.parameters()),
            lr=lr
        )

        for epoch in range(epochs):
            # 训练
            self.encoder.train()
            classifier.train()

            for batch in train_data:
                x = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                labels = batch.y.to(self.device)

                # 前向传播
                embeddings = self.encoder(x, edge_index)
                logits = classifier(embeddings)

                # 损失
                loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 验证
            val_acc = self.evaluate(val_data, classifier)
            print(f"Epoch {epoch+1}, Val Acc: {val_acc:.4f}")

        return self.encoder, classifier

    def evaluate(self, data, classifier):
        """评估"""
        self.encoder.eval()
        classifier.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in data:
                x = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                labels = batch.y.to(self.device)

                embeddings = self.encoder(x, edge_index)
                logits = classifier(embeddings)
                pred = logits.argmax(dim=1)

                correct += (pred == labels).sum().item()
                total += labels.size(0)

        return correct / total
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **GNN** | Graph Neural Network | 图神经网络 |
| **自监督学习** | Self-Supervised Learning | 无需标注的学习方式 |
| **对比学习** | Contrastive Learning | 通过正负样本对比学习 |
| **知识蒸馏** | Knowledge Distillation | 大模型知识迁移到小模型 |
| **InfoNCE** | InfoNCE Loss | 对比学习损失函数 |
| **数据增强** | Data Augmentation | 对图进行变换生成新样本 |

---

## ✅ 复习检查清单

- [ ] 理解图自监督学习的动机
- [ ] 掌握图数据增强方法
- [ ] 理解对比学习在图上的应用
- [ ] 了解知识蒸馏在GNN中的作用
- [ ] 能够实现基本的图对比学习

---

## 🤔 思考问题

1. **图数据增强与图像增强有何不同？**
   - 提示: 结构 vs 像素

2. **为什么对比学习适用于图数据？**
   - 提示: 结构相似性

3. **知识蒸馏如何帮助GNN轻量化？**
   - 提示: 模型压缩

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
