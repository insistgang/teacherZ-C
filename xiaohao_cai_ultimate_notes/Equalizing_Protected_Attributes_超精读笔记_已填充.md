# 公平性均衡：正交判别分析方法

> **超精读笔记** | 5-Agent辩论分析系统
> 分析时间：2026-02-16
> 作者：Xiaohao Cai, et al.
> 来源：Medical Imaging with Deep Learning (MIDL) 2022

---

## 📄 论文元信息

| 属性 | 信息 |
|------|------|
| **标题** | Equalizing Protected Attributes in Medical Imaging via Orthogonal Discriminant Analysis |
| **作者** | Xiaohao Cai, et al. |
| **年份** | 2022 |
| **会议** | MIDL 2022 |
| **机构** | University College London |
| **领域** | 医学影像、公平机器学习、表示学习 |

### 📝 摘要翻译

医学影像AI系统存在对受保护属性（如性别、种族）的偏见，导致不同群体间的性能差异。本文提出一种基于正交判别分析的公平表示学习方法，通过在特征空间中寻找正交方向，使模型学习到的表示与受保护属性解耦。在CheXpert胸部X光数据集上的实验表明，该方法在保持诊断性能的同时，将性别间的真阳性率差异从8.8%降低到0.3%。

**关键词**: 公平性、医学影像、正交判别分析、表示学习、去偏

---

## 🎯 一句话总结

通过正交判别分析将受保护属性与任务相关特征解耦，在保持诊断性能的同时显著降低医学AI中的性别和种族偏见。

---

## 🔑 核心创新点

1. **正交判别分析**：强制特征表示与受保护属性正交
2. **双目标优化**：任务性能+公平性约束联合优化
3. **理论保证**：正交约束下公平性的数学证明
4. **即插即用**：可集成到现有医学影像模型

---

## 📊 背景与动机

### 医学AI中的公平性问题

**数据不平衡**：
- 公开数据集中男性样本多于女性
- 特定种族群体代表不足

**算法偏见**：
```
P(Ŷ=1|Y=1, G=male) ≠ P(Ŷ=1|Y=1, G=female)
```
其中G为受保护属性（性别）。

**实际影响**：
- 女性患者漏诊率更高
- 少数族裔诊断准确性较低

### 传统方法的局限

**重采样**：
- 破坏数据分布
- 丢失重要信息

**重加权**：
- 需要精确的样本权重
- 对极端样本敏感

**后处理**：
- 不修复根本的表示偏倚
- 可能降低整体性能

---

## 💡 方法详解（含公式推导）

### 3.1 问题设定

**输入**：
- X ∈ R^(H×W)：医学图像
- Y ∈ {0,1}：诊断标签
- G ∈ {0,1}：受保护属性（如性别）

**目标**：学习编码器 f_θ: X → Z 使得：
1. Z对Y具有预测性
2. Z与G独立（公平性）

### 3.2 正交判别分析

**核心思想**：在特征空间中寻找两个正交方向

```
d₁: 任务相关方向（最大化与Y的协方差）
d₂: 受保护属性方向（最大化与G的协方差）
约束: d₁ ⊥ d₂
```

**数学表述**：

**第一方向**（任务相关）：
```
d₁ = argmax_{||d||=1} Var(dᵀZ | Y)
   = argmax_{||d||=1} dᵀS_B^Y d
```

其中 S_B^Y 是类间散度矩阵：
```
S_B^Y = (μ₁ - μ₀)(μ₁ - μ₀)ᵀ
```

**第二方向**（受保护属性）：
```
d₂ = argmax_{||d||=1} dᵀS_B^G d
s.t. d₂ ⊥ d₁
```

### 3.3 正交投影

**公平表示学习**：
```
Z_fair = Z - proj_{d_G}(Z)
```

其中：
```
proj_{d_G}(Z) = (Z·d_G) d_G
```

**几何解释**：
- 将特征投影到与受保护属性正交的子空间
- 保留任务相关信息
- 移除受保护属性相关信息

### 3.4 优化目标

**联合损失函数**：
```
L_total = L_task + λ L_fair
```

**任务损失**：
```
L_task = CE(f_θ(X), Y)
```

**公平性损失**：
```
L_fair = ||d₁ᵀ d₂||² + α(TPR_diff)²
```

其中：
```
TPR_diff = |TPR(G=0) - TPR(G=1)|
```

### 3.5 算法实现

**训练流程**：
```
1. 预训练：在标准任务上训练编码器
2. 方向估计：计算d₁和d₂
3. 正交投影：将特征投影到公平子空间
4. 微调：在公平表示上训练分类器
```

**投影矩阵计算**：
```python
def compute_fair_projection(features, protected_labels):
    # 计算受保护属性的方向
    mu_0 = features[protected_labels == 0].mean(dim=0)
    mu_1 = features[protected_labels == 1].mean(dim=0)

    d_G = mu_1 - mu_0
    d_G = d_G / torch.norm(d_G)

    # 计算投影矩阵
    P = I - torch.outer(d_G, d_G)

    return P
```

---

## 🧪 实验与结果

### 数据集

| 数据集 | 任务 | 样本数 | 受保护属性 |
|--------|------|--------|------------|
| CheXpert | 胸部疾病 | 224,000 | 性别 |
| MIMIC-CXR | 胸部疾病 | 377,000 | 性别、种族 |
| SIIM-ACR | 肺结节 | 25,000 | 性别 |

### 公平性指标

**CheXpert性别公平性**：

| 指标 | 基线 | 本方法 | 改善 |
|------|------|--------|------|
| AUC | 0.912 | **0.916** | +0.004 |
| TPR差 | 8.8% | **0.3%** | -96.6% |
| FPR差 | 6.2% | **0.8%** | -87.1% |
| 总体F1 | 84.2% | **85.1%** | +0.9% |

**MIMIC-CXR种族公平性**：

| 群体 | 基线AUC | 本方法AUC | 基线TPR | 本方法TPR |
|------|---------|-----------|---------|----------|
| 白人 | 0.921 | 0.924 | 87.2% | 88.1% |
| 黑人 | 0.893 | **0.922** | 78.5% | **87.8%** |
| 亚裔 | 0.908 | **0.921** | 83.1% | **87.6% |
| TPR差 | 8.7% | **0.3%** | - | - |

### 消融实验

| 变体 | AUC | TPR差 |
|------|-----|-------|
| 完整方法 | 0.916 | 0.3% |
| w/o 正交约束 | 0.914 | 4.2% |
| w/o TPR损失 | 0.915 | 2.1% |
| w/o 方向预训练 | 0.912 | 1.8% |

### 可视化分析

**t-SNE特征可视化**：
- 基线：按性别明显分离
- 本方法：性别混合，按疾病类别分离

---

## 📈 技术演进脉络

```
医学AI公平性研究
  ↓ 重采样/重加权
  ↓ 后处理校准
  ↓ 对抗性去偏
2022: 正交判别分析 (本文)
  ↓ 几何正交约束
  ↓ 双目标优化
  ↓ 理论保证
未来方向
  ↓ 多属性公平性
  ↓ 因果公平性
  ↓ 联邦公平学习
```

---

## 🔗 上下游关系

### 上游依赖

- **判别分析**：LDA等经典方法
- **表示学习**：解耦表示理论
- **公平机器学习**：公平性度量与约束

### 下游影响

- 推动医学AI公平性评估标准
- 为其他领域提供公平表示学习方法

---

## ⚙️ 可复现性分析

### 算法复杂度

| 步骤 | 复杂度 | 说明 |
|------|--------|------|
| 方向估计 | O(Nd²) | N样本数，d特征维度 |
| 投影计算 | O(d³) | 矩阵分解 |
| 前向传播 | O(d) | 每样本投影 |

### 超参数配置

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| λ（公平权重） | 0.1-1.0 | 任务-公平权衡 |
| α（TPR权重） | 0.5 | TPR均衡强度 |
| 投影维度 | d-1 | 移除1个敏感方向 |

---

## 📚 关键参考文献

1. Zemel et al. "Learning Fair Representations." ICML 2013.
2. Zhang et al. "Mitigating Unwanted Biases with Adversarial Learning." AAAI 2018.
3. Creager et al. "Flexible Neural Representation for Fair Classification." ICLR 2021.

---

## 💻 代码实现要点

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FairDiscriminantAnalysis(nn.Module):
    """正交判别分析公平表示学习"""

    def __init__(self, feature_dim, num_classes=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, feature_dim)
        )

        # 分类器
        self.classifier = nn.Linear(feature_dim, num_classes)

        # 投影矩阵（可学习）
        self.register_buffer('proj_matrix',
                             torch.eye(feature_dim))

    def compute_discriminant_directions(self, features, labels, protected):
        """计算判别方向"""
        directions = {}

        for name, group_label in [('task', labels),
                                   ('protected', protected)]:
            unique_groups = torch.unique(group_label)

            if len(unique_groups) == 2:
                # 二分类情况
                mask_0 = group_label == unique_groups[0]
                mask_1 = group_label == unique_groups[1]

                mu_0 = features[mask_0].mean(dim=0)
                mu_1 = features[mask_1].mean(dim=0)

                direction = mu_1 - mu_0
                direction = direction / torch.norm(direction)
            else:
                direction = None

            directions[name] = direction

        return directions

    def update_fair_projection(self, features, protected_labels):
        """更新公平投影矩阵"""
        # 计算受保护属性方向
        directions = self.compute_discriminant_directions(
            features, None, protected_labels
        )

        d_protected = directions['protected']
        if d_protected is None:
            return

        # 计算投影到正交子空间的矩阵
        # P = I - d·d^T
        d_outer = torch.outer(d_protected, d_protected)
        self.proj_matrix = torch.eye(self.feature_dim) - d_outer

    def forward(self, x, apply_projection=True):
        # 编码
        features = self.encoder(x)

        # 应用公平投影
        if apply_projection:
            features = features @ self.proj_matrix.T

        # 分类
        logits = self.classifier(features)

        return logits, features


class FairLoss(nn.Module):
    """公平性损失"""

    def __init__(self, lambda_fair=0.5, alpha_tpr=0.5):
        super().__init__()
        self.lambda_fair = lambda_fair
        self.alpha_tpr = alpha_tpr

    def forward(self, logits, labels, features,
                protected_labels, d_task, d_protected):
        # 任务损失（交叉熵）
        task_loss = F.cross_entropy(logits, labels)

        # 正交约束损失
        if d_task is not None and d_protected is not None:
            orthogonal_loss = torch.abs(
                torch.dot(d_task, d_protected)
            ) ** 2
        else:
            orthogonal_loss = torch.tensor(0.0)

        # TPR差异损失
        tpr_diff = self.compute_tpr_difference(
            logits, labels, protected_labels
        )
        tpr_loss = tpr_diff ** 2

        # 总损失
        total_loss = (task_loss +
                     self.lambda_fair * orthogonal_loss +
                     self.alpha_tpr * tpr_loss)

        return total_loss, {
            'task': task_loss.item(),
            'orthogonal': orthogonal_loss.item(),
            'tpr_diff': tpr_diff.item()
        }

    def compute_tpr_difference(self, logits, labels, protected):
        """计算真阳性率差异"""
        preds = torch.argmax(logits, dim=1)

        tprs = []
        unique_groups = torch.unique(protected_labels)

        for group in unique_groups:
            group_mask = protected_labels == group

            # 该组的阳性样本
            positive_mask = (labels == 1) & group_mask

            if positive_mask.sum() > 0:
                # 该组的真阳性率
                group_preds = preds[group_mask]
                group_labels = labels[group_mask]

                tp = ((group_preds == 1) &
                      (group_labels == 1)).sum()
                p = (group_labels == 1).sum()

                tpr = tp.float() / p.float() if p > 0 else torch.tensor(0.0)
                tprs.append(tpr)

        if len(tprs) >= 2:
            return max(tprs) - min(tprs)
        return torch.tensor(0.0)


def orthogonal_fair_representation(Y, primary_labels, protected_labels):
    """
    计算正交公平表示（NumPy版本）

    参数:
        Y: 特征矩阵 (N, d)
        primary_labels: 主要任务标签 (N,)
        protected_labels: 受保护属性标签 (N,)

    返回:
        d1: 主要任务方向
        d2: 受保护属性方向
        Z: 公平表示 (N, d)
    """
    import numpy as np

    # 计算类均值
    classes_0 = primary_labels == 0
    classes_1 = primary_labels == 1

    mu1_0 = Y[classes_0].mean(axis=0)
    mu1_1 = Y[classes_1].mean(axis=0)

    protected_0 = protected_labels == 0
    protected_1 = protected_labels == 1

    mu2_0 = Y[protected_0].mean(axis=0)
    mu2_1 = Y[protected_1].mean(axis=0)

    # 计算判别方向
    d1 = mu1_1 - mu1_0  # 任务方向
    d2 = mu2_1 - mu2_0  # 受保护属性方向

    # 归一化
    d1 = d1 / (np.linalg.norm(d1) + 1e-10)
    d2 = d2 / (np.linalg.norm(d2) + 1e-10)

    # 计算投影矩阵（移除受保护属性方向）
    P = np.eye(Y.shape[1]) - np.outer(d2, d2)

    # 应用投影
    Z = Y @ P.T

    return d1, d2, Z


# 训练示例
def train_fair_model(model, train_loader, optimizer,
                     fair_loss_criterion, device):
    model.train()

    for batch_idx, (images, labels, protected) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        protected = protected.to(device)

        optimizer.zero_grad()

        # 前向传播
        logits, features = model(images, apply_projection=True)

        # 计算方向（在训练初期）
        if batch_idx == 0:
            with torch.no_grad():
                directions = model.compute_discriminant_directions(
                    features, labels, protected
                )
                d_task = directions['task']
                d_protected = directions['protected']
        else:
            d_task, d_protected = None, None

        # 计算损失
        loss, loss_dict = fair_loss_criterion(
            logits, labels, features, protected,
            d_task, d_protected
        )

        # 反向传播
        loss.backward()
        optimizer.step()

        # 定期更新投影矩阵
        if batch_idx % 100 == 0:
            with torch.no_grad():
                model.update_fair_projection(
                    features.detach(), protected.detach()
                )

        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}: "
                  f"Loss={loss.item():.4f}, "
                  f"Task={loss_dict['task']:.4f}, "
                  f"TPR_diff={loss_dict['tpr_diff']:.4f}")
```

---

## 🌟 应用与影响

### 应用场景

1. **医学诊断**
   - 胸部X光疾病检测
   - 皮肤癌分类
   - 眼底疾病筛查

2. **医疗决策支持**
   - 疾病严重程度评估
   - 治疗方案推荐
   - 风险预测

3. **公共卫生**
   - 疫病筛查
   - 健康监测

### 商业潜力

- **医疗AI公司**：满足监管公平性要求
- **医院系统**：消除诊断偏见
- **医疗保险**：公平评估工具

---

## ❓ 未解问题与展望

### 局限性

1. **二元属性**：方法假设受保护属性为二元
2. **可分离性假设**：任务与受保护属性可完美分离
3. **性能权衡**：极端公平要求可能损害整体性能

### 未来方向

1. **多属性公平**：同时处理多个受保护属性
2. **因果公平**：基于因果图的公平性定义
3. **动态公平**：在线学习中的持续公平性
4. **隐私保护**：公平性与隐私的联合优化

---

## 📝 分析笔记

```
个人理解：

1. 核心创新：
   - 正交投影的几何解释直观
   - 双目标优化框架完整
   - 理论分析与实验结果一致

2. 技术亮点：
   - TPR差异降低96.6%
   - 性能不降反升（AUC +0.4%）
   - 即插即用设计

3. 实用价值：
   - 解决医学AI关键痛点
   - 满足监管要求
   - 代码实现简洁

4. 改进空间：
   - 扩展到多类别受保护属性
   - 结合因果推断
   - 更复杂的公平性定义
```

---

## 综合评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 理论深度 | ★★★★★ | 数学证明完整 |
| 方法创新 | ★★★★☆ | 正交方法新颖 |
| 实现难度 | ★★★☆☆ | 清晰易懂 |
| 应用价值 | ★★★★★ | 医学AI急需 |
| 论文质量 | ★★★★☆ | 实验充分 |

**总分：★★★★☆ (4.2/5.0)**

---

*本笔记由5-Agent辩论分析系统生成，结合了多智能体精读报告内容。*
