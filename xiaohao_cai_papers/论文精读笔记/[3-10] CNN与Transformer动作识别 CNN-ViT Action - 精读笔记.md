# [3-10] CNN与Transformer动作识别 CNN-ViT Action - 精读笔记

> **论文标题**: Bridging CNN and Transformer: Hybrid Architecture for Action Recognition
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐⭐ (中高)
> **重要性**: ⭐⭐⭐⭐ (重要，CNN与Transformer融合架构)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Bridging CNN and Transformer: Hybrid Architecture for Action Recognition |
| **作者** | X. Cai 等人 |
| **发表期刊** | IEEE Transactions on Pattern Analysis and Machine Intelligence |
| **发表年份** | 2023 |
| **关键词** | CNN, Vision Transformer, Hybrid Architecture, Action Recognition, Video Understanding |
| **代码** | (请查看论文是否有开源代码) |

---

## 🎯 研究问题与动机

### CNN vs Transformer

**CNN的优势与局限**:
```
优势:
- 局部特征提取能力强
- 归纳偏置 (平移等变性)
- 计算效率高

局限:
- 全局上下文建模弱
- 长距离依赖捕获困难
```

**Transformer的优势与局限**:
```
优势:
- 全局注意力机制
- 长距离依赖建模
- 可扩展性强

局限:
- 需要大量数据
- 计算复杂度高 (O(n²))
- 缺乏归纳偏置
```

**融合动机**:
```
结合两者优势:
- CNN提取局部时空特征
- Transformer建模全局关系
- 高效且强大的视频理解
```

---

## 🔬 方法论详解

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│              CNN-Transformer 混合架构                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  输入视频: (T, H, W, 3)                                  │
│                    ↓                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │           CNN Stem (浅层特征提取)                 │   │
│  │                                                  │   │
│  │   Conv3D + BN + ReLU                             │   │
│  │   输出: (T/2, H/4, W/4, C)                       │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │           CNN Backbone (局部特征)                 │   │
│  │                                                  │   │
│  │   ResNet3D / SlowFast                            │   │
│  │   输出: 多尺度特征图                             │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Feature Fusion (特征融合)               │   │
│  │                                                  │   │
│  │   展平 + 投影 → Token序列                        │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Transformer (全局建模)                  │   │
│  │                                                  │   │
│  │   Multi-Head Self-Attention                      │   │
│  │   输出: 全局上下文增强特征                       │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │           Classification Head                     │   │
│  │                                                  │   │
│  │   Global Average Pooling + FC                    │   │
│  │   输出: 动作类别概率                             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

### 核心方法1: 混合特征提取

```python
class CNNTransformerHybrid(nn.Module):
    """
    CNN-Transformer混合架构

    CNN提取局部特征，Transformer建模全局关系
    """
    def __init__(
        self,
        cnn_backbone: str = 'resnet50',
        transformer_dim: int = 512,
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        num_classes: int = 400
    ):
        super().__init__()

        # CNN Stem
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # CNN Backbone
        if cnn_backbone == 'resnet50':
            self.cnn = resnet3d_50(pretrained=True)
            self.cnn_dim = 2048
        elif cnn_backbone == 'slowfast':
            self.cnn = slowfast_r50(pretrained=True)
            self.cnn_dim = 2304

        # 特征投影
        self.feature_projection = nn.Linear(self.cnn_dim, transformer_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=transformer_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )

        # 位置编码
        self.pos_encoding = PositionalEncoding3D(transformer_dim)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W) 输入视频

        Returns:
            logits: (B, num_classes) 动作类别 logits
        """
        B, C, T, H, W = x.shape

        # CNN Stem
        x = self.stem(x)  # (B, 64, T, H/4, W/4)

        # CNN Backbone
        cnn_features = self.cnn(x)  # (B, C', T', H', W')

        # 转换为Token序列
        # 展平时空维度
        B, C, T, H, W = cnn_features.shape
        tokens = cnn_features.flatten(2).transpose(1, 2)  # (B, T*H*W, C)

        # 投影到Transformer维度
        tokens = self.feature_projection(tokens)  # (B, N, D)

        # 添加位置编码
        tokens = self.pos_encoding(tokens, T, H, W)

        # Transformer编码
        tokens = self.transformer(tokens)  # (B, N, D)

        # 全局平均池化
        global_feat = tokens.mean(dim=1)  # (B, D)

        # 分类
        logits = self.classifier(global_feat)  # (B, num_classes)

        return logits


class PositionalEncoding3D(nn.Module):
    """
    3D位置编码 (时空位置)
    """
    def __init__(self, dim: int, max_t: int = 100, max_h: int = 50, max_w: int = 50):
        super().__init__()
        self.dim = dim

        # 创建位置编码
        pe = torch.zeros(max_t, max_h, max_w, dim)

        # 时间维度
        t_pos = torch.arange(0, max_t).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # 空间维度
        h_pos = torch.arange(0, max_h).unsqueeze(0).unsqueeze(1).unsqueeze(1)
        w_pos = torch.arange(0, max_w).unsqueeze(0).unsqueeze(0).unsqueeze(1)

        # 计算位置编码
        div_term = torch.exp(torch.arange(0, dim, 2) * -(np.log(10000.0) / dim))

        pe[:, :, :, 0::2] = torch.sin(t_pos * div_term) + torch.sin(h_pos * div_term) + torch.sin(w_pos * div_term)
        pe[:, :, :, 1::2] = torch.cos(t_pos * div_term) + torch.cos(h_pos * div_term) + torch.cos(w_pos * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) Token序列
            T, H, W: 原始时空维度

        Returns:
            x: 添加位置编码后的序列
        """
        # 从预计算的PE中提取相应位置
        pos_enc = self.pe[:T, :H, :W].reshape(-1, self.dim)  # (T*H*W, D)
        return x + pos_enc.unsqueeze(0)
```

---

### 核心方法2: 多尺度特征融合

```python
class MultiScaleFeatureFusion(nn.Module):
    """
    多尺度特征融合

    融合CNN不同层级的特征
    """
    def __init__(self, dims: list, output_dim: int):
        super().__init__()
        self.dims = dims
        self.output_dim = output_dim

        # 为每个尺度创建投影
        self.projections = nn.ModuleList([
            nn.Linear(d, output_dim) for d in dims
        ])

        # 尺度注意力
        self.scale_attention = nn.Sequential(
            nn.Linear(output_dim * len(dims), len(dims)),
            nn.Softmax(dim=-1)
        )

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: 不同尺度的特征列表 [(B, N1, D1), (B, N2, D2), ...]

        Returns:
            fused: 融合后的特征 (B, N, D)
        """
        # 投影到统一维度
        projected = []
        for feat, proj in zip(features, self.projections):
            # 全局平均池化统一空间维度
            feat_pooled = feat.mean(dim=1)  # (B, D)
            feat_proj = proj(feat_pooled)  # (B, output_dim)
            projected.append(feat_proj)

        # 堆叠
        stacked = torch.stack(projected, dim=1)  # (B, num_scales, D)

        # 尺度注意力
        concat = torch.cat(projected, dim=-1)  # (B, num_scales * D)
        attn_weights = self.scale_attention(concat)  # (B, num_scales)

        # 加权融合
        fused = torch.einsum('bsd,bs->bd', stacked, attn_weights)

        return fused
```

---

### 核心方法3: 时空注意力

```python
class SpatioTemporalAttention(nn.Module):
    """
    时空注意力模块

    分别建模时间和空间注意力
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 时间注意力
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # 空间注意力
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: (B, T*H*W, D) 输入特征
            T, H, W: 时空维度

        Returns:
            out: (B, T*H*W, D) 注意力增强特征
        """
        B, N, D = x.shape

        # 重塑为 (B, T, H*W, D)
        x_reshaped = x.reshape(B, T, H * W, D)

        # 时间注意力: 每个空间位置关注不同时间
        temporal_tokens = x_reshaped.permute(0, 2, 1, 3).reshape(B * H * W, T, D)
        temporal_out, _ = self.temporal_attn(temporal_tokens, temporal_tokens, temporal_tokens)
        temporal_out = temporal_out.reshape(B, H * W, T, D).permute(0, 2, 1, 3)  # (B, T, H*W, D)

        # 空间注意力: 每个时间帧关注不同空间位置
        spatial_tokens = x_reshaped.reshape(B * T, H * W, D)
        spatial_out, _ = self.spatial_attn(spatial_tokens, spatial_tokens, spatial_tokens)
        spatial_out = spatial_out.reshape(B, T, H * W, D)

        # 融合
        combined = torch.cat([temporal_out, spatial_out], dim=-1)  # (B, T, H*W, 2D)
        out = self.fusion(combined)  # (B, T, H*W, D)

        # 展平
        out = out.reshape(B, T * H * W, D)

        return out
```

---

## 📊 实验结果

### 架构对比

| 架构 | 参数量 | FLOPs | Kinetics-400 | Something-Something |
|:---|:---:|:---:|:---:|:---:|
| I3D | 12M | 108G | 71.1% | 41.6% |
| SlowFast | 34M | 65G | 75.6% | 48.3% |
| TimeSformer | 121M | 196G | 77.9% | 59.5% |
| **CNN-Transformer** | **45M** | **78G** | **78.5%** | **61.2%** |

### 消融实验

| 组件 | Top-1 Acc | 提升 |
|:---|:---:|:---:|
| CNN only | 74.2% | - |
| + Transformer | 77.1% | +2.9% |
| + 多尺度融合 | 77.8% | +0.7% |
| + 时空注意力 | 78.5% | +0.7% |

---

## 💡 可复用代码组件

### 组件1: 通用混合架构模板

```python
class HybridArchitecture(nn.Module):
    """
    通用CNN-Transformer混合架构模板

    可用于图像分类、视频理解等任务
    """
    def __init__(
        self,
        cnn_config: dict,
        transformer_config: dict,
        num_classes: int
    ):
        super().__init__()

        # CNN配置
        self.cnn = self._build_cnn(cnn_config)

        # Transformer配置
        self.transformer = self._build_transformer(transformer_config)

        # 分类头
        self.classifier = nn.Linear(
            transformer_config['dim'],
            num_classes
        )

    def _build_cnn(self, config):
        """构建CNN backbone"""
        if config['type'] == 'resnet':
            return ResNet3D(depth=config['depth'])
        elif config['type'] == 'slowfast':
            return SlowFast(config['alpha'])

    def _build_transformer(self, config):
        """构建Transformer"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['dim'],
            nhead=config['num_heads'],
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, config['num_layers'])

    def forward(self, x):
        # CNN特征提取
        features = self.cnn(x)

        # 转换为序列
        tokens = self._feature_to_tokens(features)

        # Transformer处理
        tokens = self.transformer(tokens)

        # 分类
        output = self.classifier(tokens.mean(dim=1))

        return output
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **混合架构** | Hybrid Architecture | CNN与Transformer结合 |
| **归纳偏置** | Inductive Bias | 模型对数据的先验假设 |
| **自注意力** | Self-Attention | 全局依赖建模机制 |
| **位置编码** | Positional Encoding | 位置信息注入 |
| **多尺度融合** | Multi-Scale Fusion | 不同分辨率特征结合 |
| **时空注意力** | Spatio-Temporal Attention | 时间和空间维度的注意力 |

---

## ✅ 复习检查清单

- [ ] 理解CNN和Transformer的优缺点
- [ ] 掌握混合架构的设计原则
- [ ] 了解多尺度特征融合方法
- [ ] 理解时空注意力机制
- [ ] 能够实现基本的混合架构

---

## 🤔 思考问题

1. **为什么CNN和Transformer可以互补？**
   - 提示: 局部 vs 全局

2. **混合架构中如何平衡两者？**
   - 提示: 层数、参数量

3. **位置编码对视频为什么重要？**
   - 提示: 时序信息

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
