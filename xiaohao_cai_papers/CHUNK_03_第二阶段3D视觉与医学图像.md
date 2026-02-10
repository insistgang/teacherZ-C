# CHUNK #3: 第二阶段 - 3D视觉与医学图像

> **Chunk ID**: #3/6
> **Token数**: ~47K
> **包含论文**: 17篇 ([2-15] ~ [2-31])
> **核心内容**: 3D树木分析 + 医学图像处理 + 小样本学习
> **优先级**: ⭐⭐⭐⭐ 高优先级 (包含医学小样本、非负子空间、扩散模型)

---

## 论文列表

| 论文ID | 中文标题 | 英文关键词 | 核心贡献 | 重要性 |
|--------|----------|------------|----------|--------|
| [2-15] | 3D树木分割图割 | 3D Tree Segmentation | 图割3D | ⭐⭐⭐⭐ |
| [2-16] | 3D树木描绘图割 | 3D Tree Delineation | 精细描绘 | ⭐⭐⭐⭐ |
| [2-17] | 形状签名Krawtchouk矩 | 3DKMI | Krawtchouk矩 | ⭐⭐⭐ |
| [2-18] | 3D方向场变换 | 3D Orientation Field | 方向场 | ⭐⭐⭐⭐ |
| [2-19] | 多传感器树木映射 | Tree Mapping | 多传感器 | ⭐⭐⭐⭐ |
| [2-20] | 放疗直肠分割 | Deep Rectum Segmentation | 放疗应用 | ⭐⭐⭐ |
| [2-21] | 扩散模型脑MRI病变 | Diffusion Brain MRI | ⭐⭐ 扩散模型 | ⭐⭐⭐⭐ |
| [2-22] | 前列腺放疗器官勾画 | Prostate Radiotherapy | 器官勾画 | ⭐⭐⭐ |
| [2-23] | 直肠轮廓精度分析 | Rectal Contours Accuracy | 精度对比 | ⭐⭐⭐ |
| [2-24] | VoxTox研究计划 | VoxTox Programme | 临床计划 | ⭐⭐⭐⭐ |
| [2-25] | 医学图像小样本学习 | Medical Few-Shot | ⭐⭐⭐ 小样本 | ⭐⭐⭐⭐⭐ |
| [2-26] | 非负子空间小样本 | Non-negative Subspace | ⭐⭐⭐⭐ 非负子空间 | ⭐⭐⭐⭐⭐ |
| [2-27] | 临床变量医学分类 | Medical Classification | 临床融合 | ⭐⭐⭐ |
| [2-28] | 医学报告生成IIHT | Medical Report Generation | 报告生成 | ⭐⭐⭐⭐ |
| [2-29] | 中心体分割网络 | CenSegNet | ⭐ 细小结构 | ⭐⭐⭐⭐ |
| [2-30] | 高效变分分类方法 | Efficient Variational | 变分分类 | ⭐⭐⭐ |
| [2-31] | 点云神经表示补充 | Neural Varifolds Supplement | 补充论文 | ⭐⭐⭐ |

---

## 重点论文详解

### [2-25] 医学图像小样本学习 Medical Few-Shot ⭐⭐⭐

**期刊**: Medical Image Analysis 2021
**核心问题**: 医学图像标注数据极其稀缺

**核心框架**:
```
医学图像小样本学习设定:

N-way K-shot:
- N: 类别数 (如5类)
- K: 每类样本数 (如1-shot或5-shot)

Episode训练:
支持集 (Support Set): 用于学习
查询集 (Query Set): 用于评估

医学应用场景:
- 疾病分类: 正常 vs 异常
- 器官分割: 多器官标注
- 病变检测: 罕见病种
```

**核心贡献**:
1. 医学图像推理的小样本学习方法
2. 解决医学数据标注稀缺问题
3. 从监督学习向小样本学习转移

---

### [2-26] 非负子空间小样本学习 Non-negative Subspace ⭐⭐⭐⭐

**期刊**: IEEE TMI 2022
**核心问题**: 小样本学习中的特征表示

**核心方法**:
```
非负子空间特征学习:

传统特征学习:
特征 = 任意实数向量
→ 可解释性弱
→ 物理意义不清晰

非负子空间方法:
特征 = 非负子空间基的线性组合
→ 可解释性强 (非负约束)
→ 符合医学数据物理意义
→ 抗噪声能力强

算法流程:
1. 支持集样本学习子空间基
2. 查询集投影到子空间
3. 基于投影距离分类
```

**核心创新**:
1. 非负约束增强可解释性
2. 子空间建模降维抗噪
3. 医学数据物理意义清晰

---

### [2-21] 扩散模型脑MRI病变 Diffusion Brain MRI ⭐⭐

**期刊**: Medical Image Analysis 2023
**核心问题**: 脑MRI病变检测

**核心方法**:
```
基于差异的扩散模型:

传统方法:
直接检测病变 → 困难 (病变多样性大)

差异扩散方法:
1. 学习健康脑MRI分布
2. 测试时计算重建差异
3. 差异大 → 病变区域

扩散模型优势:
- 生成质量高
- 样本多样性好
- 可控生成
```

**核心创新**:
1. 首次将扩散模型应用于脑MRI病变
2. 基于差异的检测策略
3. 开创扩散模型医学应用

---

## 方法论关联图

```
医学图像处理主线:
[2-24] VoxTox临床研究 (应用驱动)
    ↓
[2-20][2-22] 放疗器官分割 (深度学习)
    ↓
[2-25] 小样本学习 (范式转移) ──→ [2-26] 非负子空间
    ↓
[2-21] 扩散模型 (生成模型前沿)

3D视觉主线:
[2-15] 图割3D分割 (传统方法)
    ↓
[2-18] 3D方向场 (几何方法)
    ↓
[2-19] 多传感器融合 (多模态)
    ↓
[2-12] Neural Varifolds (神经表示，关联Chunk #2)
```

---

## 核心算法模板

### 1. 小样本学习协议

```python
# 小样本学习协议模板 (来自[2-25])
def few_shot_learning_episode(support_set, query_set, N_way, K_shot):
    """
    N-way K-shot 学习协议
    """
    # 1. 支持集: N类，每类K个样本
    support_x, support_y = support_set
    # 2. 查询集: 同N类，评估用
    query_x, query_y = query_set

    # 3. 方法选择
    # 方法A: 原型网络
    prototypes = compute_prototypes(support_x, support_y, N_way)
    predictions = prototype_classification(query_x, prototypes)

    # 方法B: 非负子空间
    subspace = learn_non_negative_subspace(support_x, support_y)
    predictions = subspace_classification(query_x, subspace)

    # 4. 评估
    accuracy = compute_accuracy(predictions, query_y)

    return accuracy
```

### 2. 非负子空间学习

```python
# 非负子空间学习模板 (来自[2-26])
def non_negative_subspace_learning(support_features, n_components):
    """
    非负子空间特征学习

    Args:
        support_features: 支持集特征 (K_samples, feature_dim)
        n_components: 子空间维度

    Returns:
        subspace_basis: 非负子空间基 (feature_dim, n_components)
    """
    # 非负矩阵分解 (NMF)
    from sklearn.decomposition import NMF

    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(support_features.T)  # 特征作为基
    H = model.components_

    # W是子空间基 (非负约束)
    subspace_basis = W

    return subspace_basis
```

### 3. 放疗器官分割网络

```python
# 放疗器官分割网络模板 (来自[2-20][2-22])
class OrganSegmentationNet(nn.Module):
    def __init__(self, in_channels=1, num_organs=5):
        super().__init__()
        # 编码器 (预训练ResNet)
        self.encoder = ResNetEncoder(in_channels)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # 解码器 (多器官输出)
        self.decoder = UNetDecoder(1024, num_organs)

    def forward(self, x):
        # CT/MRI图像
        features = self.encoder(x)
        bottleneck = self.bottleneck(features)
        segmentation = self.decoder(bottleneck)
        return segmentation
```

---

## 关键概念对比

| 概念 | 传统方法 | 创新方法 | 来源论文 |
|------|----------|----------|----------|
| 小样本学习 | 迁移学习 | 元学习 + N-way K-shot | [2-25] |
| 特征表示 | 实数向量 | 非负子空间基 | [2-26] |
| 病变检测 | 直接分类 | 差异扩散模型 | [2-21] |
| 器官分割 | 2D分割 | 3D体积分割 | [2-20][2-22] |
| 临床融合 | 仅图像 | 图像 + 临床变量 | [2-27] |

---

## 学习目标检查

**阶段目标**:
- [ ] 掌握3D视觉的前沿方法
- [ ] 了解医学图像处理的实际应用

**关键问题**:
1. N-way K-shot设定如何适应医学场景？
2. 非负约束为什么增强可解释性？
3. 扩散模型如何用于病变检测？
4. 多传感器如何融合用于3D重建？

---

**处理说明**: 本chunk为method-summarizer准备，请提取核心方法论并创建中间摘要。
