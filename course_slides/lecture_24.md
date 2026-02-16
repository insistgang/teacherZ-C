# 第二十四讲：多模态学习

## Multi-Modal Learning

---

### 📋 本讲大纲

1. 多模态学习概述
2. 视觉-语言模型
3. 跨模态对齐
4. 多模态融合
5. 前沿应用

---

### 24.1 多模态学习

#### 定义

利用多种模态（视觉、语言、音频等）的联合信息进行学习

#### 为什么多模态？

```
• 信息互补
• 更好的表示
• 更强的泛化
• 更自然的人机交互
```

#### 挑战

- 模态差异大
- 数据对齐困难
- 融合策略设计

---

### 24.2 常见模态组合

| 组合 | 应用 |
|------|------|
| 图像-文本 | 图像描述、VQA |
| 视频-音频 | 视频理解 |
| 图像-深度 | 3D感知 |
| 文本-语音 | 语音合成 |
| 多语言 | 机器翻译 |

---

### 24.3 视觉-语言模型

#### 代表模型

| 模型 | 年份 | 特点 |
|------|------|------|
| CLIP | 2021 | 对比学习 |
| ViLT | 2021 | 无卷积 |
| BLIP | 2022 | 统一架构 |
| Flamingo | 2022 | 少样本 |
| GPT-4V | 2023 | 大规模多模态 |

#### 发展趋势

- 统一架构
- 更大规模
- 更好泛化

---

### 24.4 CLIP模型

#### 对比语言-图像预训练

**架构**：
```
图像编码器 (ViT/CNN)
        ↓
    图像特征 z_I
        ↓
对    ←→    对比损失
比    ←→
        ↑
    文本特征 z_T
        ↑
文本编码器 (Transformer)
```

#### 训练目标

最大化匹配图文对的相似度：
$$\mathcal{L} = -\sum_i \log \frac{\exp(z_I^i \cdot z_T^i / \tau)}{\sum_j \exp(z_I^i \cdot z_T^j / \tau)}$$

#### 零样本分类

```
类别 → 文本模板 → 文本编码 → 与图像相似度
```

---

### 24.5 跨模态对齐

#### 特征级对齐

将不同模态映射到共同特征空间：
$$f_V: V \rightarrow \mathbb{R}^d$$
$$f_T: T \rightarrow \mathbb{R}^d$$

#### 对齐方法

| 方法 | 描述 |
|------|------|
| 对比学习 | CLIP、ALIGN |
| 匹配损失 | 视觉-语言BERT |
| 重建损失 | 跨模态自编码器 |

---

### 24.6 多模态融合

#### 融合层级

```
早期融合：原始数据融合
特征融合：中层特征融合
决策融合：各模态独立预测后融合
```

#### 注意力融合

$$\text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

- 图像作为Query，文本作为Key/Value
- 或反之

---

### 24.7 视觉Transformer (ViT)

#### 图像patch化

将图像划分为patch：
$$\mathbf{x} = [\mathbf{x}_1^p, \mathbf{x}_2^p, ..., \mathbf{x}_N^p]$$

#### Patch嵌入

$$\mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_1^p\mathbf{E}; ...; \mathbf{x}_N^p\mathbf{E}] + \mathbf{E}_{pos}$$

#### Transformer编码

与文本Transformer相同的处理方式

---

### 24.8 多模态Transformer

#### 统一架构

将图像patch和文本token作为统一输入：
$$\mathbf{H} = [\mathbf{H}_{img}; \mathbf{H}_{text}]$$

#### 自注意力

图像和文本token之间可以相互交互

#### 模型示例

- ViLT
- COCA
- BLIP-2

---

### 24.9 视觉问答(VQA)

#### 任务

给定图像和问题，生成答案

#### 方法

```
图像 → 视觉编码器 → 图像特征
                        ↓
问题 → 文本编码器 → 问题特征 → 融合 → 答案预测
```

#### 挑战

- 细粒度理解
- 推理能力
- 多步推理

---

### 24.10 图像描述生成

#### 任务

生成描述图像内容的自然语言文本

#### 编码器-解码器

```
图像 → 编码器 → 上下文向量 → 解码器 → 文本描述
```

#### 注意力机制

生成每个词时关注图像的不同区域

#### 评估指标

- BLEU
- CIDEr
- METEOR

---

### 24.11 文生图模型

#### 任务

根据文本描述生成图像

#### 代表模型

| 模型 | 方法 |
|------|------|
| DALL-E | 自回归+VAE |
| Stable Diffusion | 扩散模型 |
| Imagen | 扩散+大语言模型 |
| DALL-E 3 | 潜在扩散 |

#### 扩散模型

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$$

学习逆向过程：$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$

---

### 24.12 前沿方向

#### 统一多模态模型

- GPT-4V
- Gemini
- 能够处理任意模态组合

#### 多模态推理

- 视觉思维链
- 多模态CoT

#### 世界模型

- 视频预测
- 物理理解
- 具身智能

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           多模态学习核心                         │
├─────────────────────────────────────────────────┤
│                                                 │
│   核心任务：                                     │
│   • 表征学习：各模态编码                        │
│   • 对齐：跨模态特征空间                        │
│   • 融合：多模态信息整合                        │
│                                                 │
│   代表模型：                                     │
│   • CLIP：对比图文对                            │
│   • ViLT：统一Transformer                      │
│   • BLIP：统一预训练                            │
│                                                 │
│   应用：                                        │
│   • VQA、图像描述                               │
│   • 文生图（Stable Diffusion）                  │
│   • 多模态对话                                  │
│                                                 │
│   趋势：大规模统一模型                          │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **实现题**：使用CLIP进行零样本图像分类

2. **实验题**：比较不同视觉-语言模型的检索效果

3. **分析题**：分析跨模态对齐的关键因素

4. **研究题**：调研最新的多模态大模型

---

### 📖 扩展阅读

1. **经典论文**：
   - Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
   - Ramesh et al., "Zero-Shot Text-to-Image Generation", ICML 2021
   - Li et al., "BLIP-2", ICML 2023

2. **开源项目**：
   - OpenCLIP
   - Stable Diffusion
   - LLaVA

3. **数据集**：
   - COCO Captions
   - VQA v2
   - LAION

---

### 📖 参考文献

1. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.

2. Ramesh, A., et al. (2021). Zero-shot text-to-image generation. *ICML*.

3. Li, J., et al. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *ICML*.

4. Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. *CVPR*.

5. Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.
