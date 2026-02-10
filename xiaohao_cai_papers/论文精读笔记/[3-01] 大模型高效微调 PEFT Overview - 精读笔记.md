# [3-01] 大模型高效微调 PEFT Overview - 精读笔记

> **论文标题**: Parameter-Efficient Fine-Tuning (PEFT) for Large Language Models: A Survey
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐⭐ (中高，涉及多种技术)
> **重要性**: ⭐⭐⭐⭐ (LLM微调核心技术)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Parameter-Efficient Fine-Tuning (PEFT) for Large Language Models: A Survey / Methods |
| **作者** | Xiaohao Cai 等人 |
| **发表会议/期刊** | 综述/方法论文 (2023-2024) |
| **关键词** | PEFT, LoRA, Adapter, Prompt Tuning, LLM Fine-tuning |
| **核心价值** | 系统梳理大模型高效微调方法体系 |

---

## 🎯 研究背景与问题

### 大模型微调挑战

```
传统全参数微调的问题:
├── 参数量巨大 (GPT-3: 175B, LLaMA: 65B)
├── 显存需求高 (需要多卡A100)
├── 训练时间长 (数天到数周)
├── 存储成本高 (每个任务需保存完整模型)
└── 灾难性遗忘 (覆盖预训练知识)

PEFT解决方案:
├── 只微调少量参数 (0.1% - 1%)
├── 显存需求降低 (单卡可训练)
├── 训练速度提升 (数小时完成)
├── 多任务共享底座 (只存适配器)
└── 保留预训练知识
```

### PEFT方法分类

```
PEFT方法体系:
│
├── 添加参数类 (Additive Methods)
│   ├── Adapter: 插入小型适配层
│   ├── LoRA: 低秩适配
│   ├── (IA)³: 学习缩放向量
│   └── Prefix Tuning: 添加前缀嵌入
│
├── 选择参数类 (Selective Methods)
│   ├── BitFit: 只微调偏置项
│   ├── Diff Pruning: 稀疏差分更新
│   └── 层选择: 只微调特定层
│
└── 重参数化类 (Reparameterized Methods)
    ├── LoRA系列: 低秩分解
    ├── Tensor-based: 张量分解 (tCURLoRA)
    └── Kronecker Product: 克罗内克积分解
```

---

## 🔬 方法论详解

### 方法一: LoRA (Low-Rank Adaptation)

#### 核心思想

```
预训练权重: W₀ ∈ ℝ^{d×k}
传统微调: W = W₀ + ΔW (更新全部参数)

LoRA:
W = W₀ + ΔW = W₀ + BA
其中:
  - B ∈ ℝ^{d×r}
  - A ∈ ℝ^{r×k}
  - r ≪ min(d, k) (通常 r=4,8,16)

参数量: d×k → r×(d+k)
压缩比: ~d×k / (r×(d+k))
```

#### 数学公式

```python
# LoRA前向传播
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 实现
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0
    ):
        super().__init__()

        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # 冻结的预训练权重
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA可训练参数
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: y = x @ W₀ᵀ + x @ (B @ A)ᵀ * scaling
        """
        # 原始输出 (冻结)
        original_out = F.linear(x, self.weight)

        # LoRA分支 (可训练)
        lora_out = F.linear(F.linear(self.lora_dropout(x), self.lora_A.t()), self.lora_B.t())

        return original_out + lora_out * self.scaling


class LinearWithLoRA(nn.Module):
    """
    将普通Linear层转换为LoRA层
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, lora_alpha: float = 16):
        super().__init__()

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # 冻结原始权重
        self.weight = nn.Parameter(linear_layer.weight.data.clone())
        self.weight.requires_grad = False

        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone())
            self.bias.requires_grad = False
        else:
            self.register_parameter('bias', None)

        # 添加LoRA参数
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        self.scaling = lora_alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        out = F.linear(x, self.weight, self.bias)

        # LoRA适配
        out += F.linear(F.linear(x, self.lora_A.t()), self.lora_B.t()) * self.scaling

        return out
```

#### LoRA优势

```
优势:
├── 参数高效: 只训练0.1%-1%参数
├── 无推理开销: 可合并回原始权重
├── 模块化: 不同任务切换不同LoRA
├── 存储友好: 每个任务只存小矩阵
└── 效果接近: 性能与全微调相当

应用位置:
├── Transformer: 只应用于Q, V投影
├── 推荐: W_q, W_v (注意力查询和值)
├── 可选: W_k, W_o, FFN层
└── 避免: Embedding和输出层
```

---

### 方法二: Adapter

#### 核心思想

```
在Transformer层之间插入小型适配模块:

原始: x → Attention → FFN → Output
添加Adapter: x → Attention → Adapter → FFN → Adapter → Output

Adapter结构:
  输入 → Down-project (d→r) → ReLU → Up-project (r→d) → Output
       + Skip Connection

参数量: 2 × d × r (通常 r=64)
```

#### 实现代码

```python
class Adapter(nn.Module):
    """
    Adapter模块: 瓶颈结构适配器
    """
    def __init__(
        self,
        hidden_size: int,
        adapter_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()

        self.down_project = nn.Linear(hidden_size, adapter_dim)
        self.activation = nn.GELU()
        self.up_project = nn.Linear(adapter_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 初始化: 接近恒等映射
        nn.init.xavier_uniform_(self.down_project.weight)
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        residual = x

        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)

        return x + residual  # 残差连接


class TransformerLayerWithAdapter(nn.Module):
    """
    带Adapter的Transformer层
    """
    def __init__(self, d_model: int, nhead: int, adapter_dim: int = 64):
        super().__init__()

        # 原始Transformer组件
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        # LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Adapter模块
        self.adapter_after_attn = Adapter(d_model, adapter_dim)
        self.adapter_after_ffn = Adapter(d_model, adapter_dim)

    def forward(self, x: torch.Tensor, mask=None):
        # Attention子层
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.adapter_after_attn(x)  # 添加Adapter

        # FFN子层
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        x = self.adapter_after_ffn(x)  # 添加Adapter

        return x
```

---

### 方法三: Prompt Tuning / Prefix Tuning

#### 核心思想

```
不修改模型参数，而是修改输入:

Prompt Tuning:
  输入: [可训练软提示] + [真实输入]
  例如: [P1][P2]...[Pk] + "翻译这句话"

Prefix Tuning:
  在每层注意力前添加可训练前缀:
  [Prefix_K] → Key
  [Prefix_V] → Value

优势:
  - 完全不修改模型
  - 可训练参数量极少
  - 任务切换只需换prompt
```

#### 实现代码

```python
class PrefixTuning(nn.Module):
    """
    Prefix Tuning: 在注意力层添加可训练前缀
    """
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        embed_dim: int,
        prefix_length: int = 20
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.prefix_length = prefix_length

        # 每层的prefix参数
        # shape: (num_layers, 2, num_heads, prefix_length, head_dim)
        # 2 for key and value
        self.prefix_tokens = nn.Parameter(
            torch.randn(num_layers, 2, num_heads, prefix_length, self.head_dim)
        )

        # MLP重参数化 (可选，提高稳定性)
        self.prefix_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, num_layers * 2 * embed_dim)
        )

    def forward(self, batch_size: int, device: torch.device):
        """
        生成prefix embedding

        Returns:
            past_key_values: 用于Transformer的past_key_values
        """
        # 直接使用prefix tokens
        prefix = self.prefix_tokens.unsqueeze(0)  # (1, num_layers, 2, num_heads, prefix_len, head_dim)
        prefix = prefix.expand(batch_size, -1, -1, -1, -1, -1)

        # 转换为past_key_values格式
        past_key_values = []
        for layer_idx in range(self.num_layers):
            key = prefix[:, layer_idx, 0]  # (batch, num_heads, prefix_len, head_dim)
            value = prefix[:, layer_idx, 1]
            past_key_values.append((key, value))

        return past_key_values


class PromptTuning(nn.Module):
    """
    Prompt Tuning: 在输入前添加可训练软提示
    """
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        num_prompts: int = 100
    ):
        super().__init__()

        self.num_prompts = num_prompts

        # 可训练的软提示嵌入
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_prompts, token_dim)
        )

        # 提示到token的映射
        self.prompt_projection = nn.Linear(token_dim, num_tokens * token_dim)

    def forward(self, input_embeds: torch.Tensor, prompt_id: int = 0):
        """
        将软提示与输入嵌入拼接

        Args:
            input_embeds: (batch, seq_len, dim)
            prompt_id: 使用的提示ID

        Returns:
            combined_embeds: (batch, num_tokens + seq_len, dim)
        """
        batch_size = input_embeds.size(0)

        # 获取提示嵌入
        prompt_embed = self.prompt_embeddings[prompt_id]

        # 投影到多个token
        prompt_tokens = self.prompt_projection(prompt_embed)
        prompt_tokens = prompt_tokens.view(self.num_prompts, -1, input_embeds.size(-1))

        # 扩展到batch
        prompt_embeds = prompt_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        # 拼接
        combined = torch.cat([prompt_embeds, input_embeds], dim=1)

        return combined
```

---

### 方法四: (IA)³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)

#### 核心思想

```
学习缩放向量而非添加新参数:

对Transformer中的Key, Value, FFN输出进行缩放:
  Key = Key ⊙ l_k
  Value = Value ⊙ l_v
  FFN_out = FFN_out ⊙ l_ff

其中 l_k, l_v, l_ff 是可学习的缩放向量 (逐元素)

参数量: 3 × d (比Adapter和LoRA更少)
```

#### 实现代码

```python
class IA3Layer(nn.Module):
    """
    (IA)³: 学习缩放向量
    """
    def __init__(self, hidden_size: int):
        super().__init__()

        # 可学习的缩放向量
        self.scale_k = nn.Parameter(torch.ones(hidden_size))
        self.scale_v = nn.Parameter(torch.ones(hidden_size))
        self.scale_ff = nn.Parameter(torch.ones(hidden_size))

    def forward(self, key, value, ff_output):
        """
        应用学习到缩放
        """
        key_scaled = key * self.scale_k
        value_scaled = value * self.scale_v
        ff_scaled = ff_output * self.scale_ff

        return key_scaled, value_scaled, ff_scaled


class IA3Transformer(nn.Module):
    """
    集成(IA)³的Transformer
    """
    def __init__(self, d_model: int, nhead: int):
        super().__init__()

        self.attention = nn.MultiheadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        # (IA)³缩放
        self.ia3 = IA3Layer(d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # FFN
        ff_out = self.ffn(x)

        # 应用(IA)³缩放
        # 这里简化处理，实际应在attention内部应用
        _, _, ff_scaled = self.ia3(x, x, ff_out)

        x = self.norm2(x + ff_scaled)

        return x
```

---

## 📊 方法对比

### 参数量对比

| 方法 | 可训练参数量 | 相对全微调 | 适用场景 |
|:---|:---|:---:|:---|
| **Full Fine-tuning** | 100% | 100% | 数据充足，计算资源充足 |
| **BitFit** | ~0.1% | 偏置项 | 极少量参数场景 |
| **Prompt Tuning** | ~0.01% | 软提示 | 分类任务，多任务 |
| **Prefix Tuning** | ~0.1% | 前缀嵌入 | 生成任务 |
| **Adapter** | ~0.5-2% | 适配层 | 多任务，模块化 |
| **LoRA** | ~0.1-1% | 低秩矩阵 | 通用，推荐首选 |
| **(IA)³** | ~0.01% | 缩放向量 | 超轻量级适配 |
| **tCURLoRA** | ~0.1-1% | 张量分解 | 高维参数结构 |

### 性能对比

```
典型任务性能 (相对于全微调):

GLUE Benchmark:
├── Full Fine-tuning: 100% (baseline)
├── LoRA (r=8): 98-99%
├── Adapter: 97-98%
├── Prefix Tuning: 95-97%
└── Prompt Tuning: 92-95%

注意:
- 任务越复杂，差距可能越大
- 适当增大rank可缩小差距
- 组合方法通常效果更好
```

---

## 💻 可复用代码组件

### 组件1: 通用PEFT包装器

```python
import torch
import torch.nn as nn
from typing import Optional, List, Dict
from enum import Enum

class PEFTMethod(Enum):
    LORA = "lora"
    ADAPTER = "adapter"
    PREFIX_TUNING = "prefix_tuning"
    IA3 = "ia3"


class PEFTWrapper:
    """
    通用PEFT方法包装器

    自动为模型添加PEFT适配
    """

    def __init__(
        self,
        model: nn.Module,
        method: PEFTMethod,
        config: Dict
    ):
        self.model = model
        self.method = method
        self.config = config

        # 冻结原始参数
        self._freeze_base_model()

        # 添加PEFT模块
        self._add_peft_modules()

    def _freeze_base_model(self):
        """冻结基础模型参数"""
        for param in self.model.parameters():
            param.requires_grad = False

    def _add_peft_modules(self):
        """根据方法添加适配模块"""
        if self.method == PEFTMethod.LORA:
            self._apply_lora()
        elif self.method == PEFTMethod.ADAPTER:
            self._apply_adapter()
        elif self.method == PEFTMethod.PREFIX_TUNING:
            self._apply_prefix_tuning()
        elif self.method == PEFTMethod.IA3:
            self._apply_ia3()

    def _apply_lora(self):
        """应用LoRA"""
        target_modules = self.config.get('target_modules', ['q', 'v'])
        rank = self.config.get('rank', 8)
        alpha = self.config.get('alpha', 16)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 检查是否是目标模块
                if any(target in name for target in target_modules):
                    # 替换为LoRA层
                    lora_layer = LinearWithLoRA(module, rank, alpha)
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self.model.get_submodule(parent_name)
                    setattr(parent, child_name, lora_layer)

    def _apply_adapter(self):
        """应用Adapter"""
        adapter_dim = self.config.get('adapter_dim', 64)

        # 在Transformer层后添加Adapter
        for name, module in self.model.named_modules():
            if 'Transformer' in type(module).__name__:
                # 添加Adapter
                pass  # 具体实现取决于模型结构

    def _apply_prefix_tuning(self):
        """应用Prefix Tuning"""
        prefix_length = self.config.get('prefix_length', 20)
        # 实现略
        pass

    def _apply_ia3(self):
        """应用(IA)³"""
        # 实现略
        pass

    def get_trainable_parameters(self):
        """获取可训练参数"""
        return [p for p in self.model.parameters() if p.requires_grad]

    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        trainable_params = 0
        all_params = 0

        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_params:,} || "
              f"Trainable %: {100 * trainable_params / all_params:.2f}%")

    def save_adapter(self, path: str):
        """只保存适配器参数"""
        adapter_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                adapter_state[name] = param.data.cpu()

        torch.save(adapter_state, path)
        print(f"Adapter saved to {path}")

    def load_adapter(self, path: str):
        """加载适配器参数"""
        adapter_state = torch.load(path)
        self.model.load_state_dict(adapter_state, strict=False)
        print(f"Adapter loaded from {path}")


def apply_peft_to_model(
    model: nn.Module,
    method: str = "lora",
    **kwargs
) -> nn.Module:
    """
    便捷函数: 为模型添加PEFT

    Args:
        model: 原始模型
        method: PEFT方法 (lora, adapter, prefix, ia3)
        **kwargs: 方法特定参数

    Returns:
        带有PEFT的模型
    """
    method_enum = PEFTMethod(method)
    wrapper = PEFTWrapper(model, method_enum, kwargs)
    wrapper.print_trainable_parameters()

    return wrapper.model
```

### 组件2: LoRA配置工具

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoRAConfig:
    """
    LoRA配置
    """
    r: int = 8  # 秩
    lora_alpha: int = 16  # 缩放参数
    lora_dropout: float = 0.0
    target_modules: List[str] = None
    bias: str = "none"  # none, all, lora_only
    modules_to_save: List[str] = None  # 额外训练的模块

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


# 推荐配置
LORA_CONFIGS = {
    "light": LoRAConfig(r=4, lora_alpha=8),
    "default": LoRAConfig(r=8, lora_alpha=16),
    "heavy": LoRAConfig(r=16, lora_alpha=32),
    "ultra": LoRAConfig(r=32, lora_alpha=64),
}
```

---

## 🧪 应用到井盖检测

### 场景: 使用预训练视觉模型进行井盖检测微调

```python
class ManholePEFTDetector:
    """
    使用PEFT进行井盖检测
    """

    def __init__(self, pretrained_backbone: nn.Module):
        # 加载预训练骨干网络 (如ResNet, ViT)
        self.backbone = pretrained_backbone

        # 冻结骨干网络
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 添加LoRA适配器
        self._apply_lora_to_backbone()

        # 检测头 (始终可训练)
        self.detection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 5)  # x, y, w, h, confidence
        )

    def _apply_lora_to_backbone(self):
        """在骨干网络上应用LoRA"""
        # 找到所有Linear层并添加LoRA
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Linear):
                # 替换为LoRA层
                parent = self._get_parent_module(name)
                child_name = name.split('.')[-1]
                lora_layer = LinearWithLoRA(module, rank=8, lora_alpha=16)
                setattr(parent, child_name, lora_layer)

    def _get_parent_module(self, name: str):
        """获取父模块"""
        parts = name.split('.')[:-1]
        module = self.backbone
        for part in parts:
            module = getattr(module, part)
        return module

    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        return detections

    def save_checkpoint(self, path: str):
        """保存检查点 (只包含LoRA和检测头)"""
        checkpoint = {
            'lora_params': {
                k: v for k, v in self.backbone.named_parameters()
                if v.requires_grad
            },
            'head_params': self.detection_head.state_dict()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.backbone.load_state_dict(checkpoint['lora_params'], strict=False)
        self.detection_head.load_state_dict(checkpoint['head_params'])
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **PEFT** | Parameter-Efficient Fine-Tuning | 参数高效微调 |
| **LoRA** | Low-Rank Adaptation | 低秩适配 |
| **Adapter** | Adapter | 适配器，瓶颈结构 |
| **Prompt Tuning** | Prompt Tuning | 软提示微调 |
| **Prefix Tuning** | Prefix Tuning | 前缀微调 |
| **(IA)³** | Infused Adapter | 缩放向量适配 |
| **Rank** | Rank | 低秩分解的秩 |
| **Alpha** | Alpha | LoRA缩放系数 |

---

## ✅ 复习检查清单

- [ ] 理解PEFT的核心动机
- [ ] 掌握LoRA的原理和实现
- [ ] 了解Adapter的结构
- [ ] 理解Prompt/Prefix Tuning的区别
- [ ] 能选择合适的PEFT方法
- [ ] 能将PEFT应用到视觉任务

---

## 🔗 相关论文推荐

### 必读
1. **LoRA** (ICLR 2022) - 低秩适配
2. **Prefix Tuning** (ACL 2021) - 前缀微调
3. **Adapter** (ICML 2019) - 适配器

### 扩展阅读
1. **(IA)³** (EMNLP 2022) - 缩放向量
2. **Prompt Tuning** (EMNLP 2021) - 软提示
3. **BitFit** (ACL 2022) - 偏置微调
4. **tCURLoRA** (ICML 2024) - 张量CUR分解适配

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
**下一步**: 在井盖检测上实践LoRA微调
