# [3-01] 大模型高效微调 LLM Fine-tuning - 精读笔记

> **论文标题**: Parameter-Efficient Fine-Tuning for Large Language Models: A Survey
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐⭐ (中高)
> **重要性**: ⭐⭐⭐⭐ (重要，PEFT技术综述)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Parameter-Efficient Fine-Tuning for Large Language Models: A Survey |
| **作者** | Various (综述论文) |
| **发表期刊** | arXiv/相关综述 |
| **发表年份** | 2023-2024 |
| **关键词** | PEFT, LoRA, Adapter, Prompt Tuning, LLM |
| **代码** | 多种开源实现 |

---

## 🎯 研究问题与动机

### 核心挑战

**大模型微调困境**:
```
GPT-3: 175B 参数
全量微调需要:
- 显存: 700GB+ (FP32)
- 训练时间: 数天到数周
- 计算成本: 数万美元

问题: 普通研究者/企业无法承担
```

**参数高效微调(PEFT)解决方案**:
```
只微调 0.1%~1% 的参数
保持 99%+ 的原始参数冻结
达到接近全量微调的性能
```

---

## 🔬 方法论详解

### PEFT方法分类

```
┌─────────────────────────────────────────────────────────┐
│              参数高效微调 (PEFT) 方法分类                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 1. 加法方法 (Additive Methods)                    │   │
│  │    - Adapter: 插入小型适配层                      │   │
│  │    - Prompt Tuning: 学习软提示                    │   │
│  │    - Prefix Tuning: 学习前缀嵌入                  │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 2. 选择方法 (Selective Methods)                   │   │
│  │    - 只微调部分层                                 │   │
│  │    - 只微调偏置项                                 │   │
│  └─────────────────────────────────────────────────┘   │
│                          │                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 3. 重参数化方法 (Reparameterization)              │   │
│  │    - LoRA: 低秩适配 ⭐⭐⭐⭐⭐                     │   │
│  │    - tCURLoRA: 张量CUR分解 (见[3-02])             │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

### 核心方法1: LoRA (Low-Rank Adaptation)

**核心思想**:
```
传统微调: W = W₀ + ΔW (更新所有参数)
LoRA: W = W₀ + BA (只训练B和A)

其中:
- W₀ ∈ R^(d×k): 预训练权重 (冻结)
- B ∈ R^(d×r): 可训练矩阵
- A ∈ R^(r×k): 可训练矩阵
- r << min(d,k): 低秩约束

参数量: d×k → r×(d+k) (大幅减少)
```

**数学表达**:
```
h = W₀x + ΔWx = W₀x + BAx

前向传播:
1. 原始输出: h₀ = W₀x
2. 低秩更新: Δh = BAx
3. 最终输出: h = h₀ + Δh

训练时: 只优化B和A
推理时: 可以合并 W = W₀ + BA
```

**LoRA实现代码**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 层

    只对线性层进行低秩适配
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        # 冻结的预训练权重
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight.requires_grad = False

        # LoRA可训练参数
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch_size, in_features)

        Returns:
            output: (batch_size, out_features)
        """
        # 原始输出 (冻结)
        original_output = F.linear(x, self.weight)

        # LoRA分支 (可训练)
        lora_output = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling

        return original_output + lora_output


class LinearWithLoRA(nn.Module):
    """
    将现有线性层转换为LoRA层
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
            self.bias = None

        # LoRA参数
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) / rank)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        self.scaling = lora_alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始输出
        original = F.linear(x, self.weight, self.bias)

        # LoRA适配
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

        return original + lora

    def merge_lora(self):
        """
        将LoRA权重合并到原始权重 (推理优化)
        """
        self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        return self
```

---

### 核心方法2: Adapter

**核心思想**:
```
在Transformer层中插入小型适配模块
只训练Adapter参数，原始参数冻结
```

**Adapter结构**:
```
输入 → [冻结的Transformer] → 隐藏状态
                              ↓
                        ┌───────────┐
                        │  Adapter  │
                        │  ┌─────┐  │
                        │  │Down │  │  (投影到低维)
                        │  │Proj │  │
                        │  └──┬──┘  │
                        │     ↓     │
                        │  Nonlinear│
                        │     ↓     │
                        │  ┌─────┐  │
                        │  │ Up  │  │  (投影回高维)
                        │  │Proj │  │
                        │  └──┬──┘  │
                        │     +     │  (残差连接)
                        └─────┼─────┘
                              ↓
                        输出
```

**Adapter实现**:
```python
class Adapter(nn.Module):
    """
    Adapter模块

    标准的瓶颈结构: d → r → d
    """
    def __init__(self, input_dim: int, adapter_dim: int = 64):
        super().__init__()

        self.down_proj = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(adapter_dim, input_dim)

        # 初始化: 接近恒等映射
        nn.init.normal_(self.down_proj.weight, std=1e-3)
        nn.init.normal_(self.up_proj.weight, std=1e-3)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, input_dim)

        Returns:
            output: (batch_size, seq_len, input_dim)
        """
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual  # 残差连接


class TransformerWithAdapter(nn.Module):
    """
    带Adapter的Transformer层
    """
    def __init__(self, d_model: int, nhead: int, adapter_dim: int = 64):
        super().__init__()

        # 原始Transformer组件 (冻结)
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Adapter (可训练)
        self.adapter1 = Adapter(d_model, adapter_dim)
        self.adapter2 = Adapter(d_model, adapter_dim)

        # 冻结原始参数
        for param in self.self_attn.parameters():
            param.requires_grad = False
        for param in self.feed_forward.parameters():
            param.requires_grad = False
        for param in self.norm1.parameters():
            param.requires_grad = False
        for param in self.norm2.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention + Adapter
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.adapter1(x)  # Adapter插入点

        # Feed-Forward + Adapter
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        x = self.adapter2(x)  # Adapter插入点

        return x
```

---

### 核心方法3: Prompt Tuning

**核心思想**:
```
不修改模型参数，而是学习输入的"软提示"
在输入嵌入前添加可训练的提示向量
```

**Prompt Tuning实现**:
```python
class PromptTuning(nn.Module):
    """
    Prompt Tuning: 学习软提示

    在输入前添加可训练的提示嵌入
    """
    def __init__(
        self,
        num_tokens: int = 20,  # 提示长度
        token_dim: int = 768,  # 嵌入维度
        num_classes: int = 10
    ):
        super().__init__()

        # 可训练的软提示
        self.soft_prompt = nn.Parameter(torch.randn(num_tokens, token_dim))

        # 预训练模型 (冻结)
        self.backbone = AutoModel.from_pretrained('bert-base-uncased')
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 分类头
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.size(0)

        # 获取输入嵌入
        inputs_embeds = self.backbone.embeddings.word_embeddings(input_ids)

        # 扩展软提示到batch_size
        prompt_embeds = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)

        # 拼接: [软提示, 输入嵌入]
        combined_embeds = torch.cat([prompt_embeds, inputs_embeds], dim=1)

        # 更新attention mask
        prompt_attention = torch.ones(batch_size, self.soft_prompt.size(0),
                                       device=attention_mask.device)
        combined_attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        # 通过冻结的backbone
        outputs = self.backbone(inputs_embeds=combined_embeds,
                               attention_mask=combined_attention_mask)

        # 使用[CLS]或平均池化进行分类
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(pooled)

        return logits
```

---

## 📊 方法对比

### 参数效率对比

| 方法 | 可训练参数量 | 存储开销 | 推理开销 | 性能 |
|:---|:---:|:---:|:---:|:---:|
| 全量微调 | 100% | 100% | 100% | 100% |
| LoRA (r=8) | ~0.1% | ~0.1% | ~0% (可合并) | ~98% |
| Adapter | ~0.5% | ~0.5% | ~5% | ~97% |
| Prompt Tuning | ~0.01% | ~0.01% | ~0% | ~95% |
| Prefix Tuning | ~0.1% | ~0.1% | ~2% | ~96% |

### 适用场景

| 方法 | 最佳适用场景 | 注意事项 |
|:---|:---|:---|
| **LoRA** | 大多数场景，特别是大模型 | 需要选择合适rank |
| **Adapter** | 多任务场景 | 增加推理延迟 |
| **Prompt Tuning** | 超大模型 (>10B) | 需要较长提示 |
| **Prefix Tuning** | 生成任务 | 可能不稳定 |

---

## 💡 可复用代码组件

### 组件1: 通用PEFT应用器

```python
class PEFTApplier:
    """
    通用PEFT方法应用器

    支持LoRA、Adapter等多种方法
    """
    def __init__(self, model: nn.Module, method: str = 'lora', **kwargs):
        self.model = model
        self.method = method
        self.kwargs = kwargs

    def apply(self) -> nn.Module:
        """应用PEFT方法"""
        if self.method == 'lora':
            return self._apply_lora()
        elif self.method == 'adapter':
            return self._apply_adapter()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _apply_lora(self) -> nn.Module:
        """应用LoRA到所有线性层"""
        rank = self.kwargs.get('rank', 8)
        lora_alpha = self.kwargs.get('lora_alpha', 16)
        target_modules = self.kwargs.get('target_modules', ['q', 'v'])  # Attention的q,v

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 检查是否应该应用LoRA
                if any(target in name for target in target_modules):
                    # 替换为LoRA层
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = self._get_module(self.model, parent_name)

                    lora_layer = LinearWithLoRA(module, rank=rank, lora_alpha=lora_alpha)
                    setattr(parent, child_name, lora_layer)

        return self.model

    def _apply_adapter(self) -> nn.Module:
        """应用Adapter到Transformer层"""
        adapter_dim = self.kwargs.get('adapter_dim', 64)

        for name, module in self.model.named_modules():
            if 'Transformer' in type(module).__name__:
                # 添加Adapter
                # 具体实现取决于模型结构
                pass

        return self.model

    def _get_module(self, model: nn.Module, name: str) -> nn.Module:
        """通过名称获取模块"""
        if not name:
            return model
        for part in name.split('.'):
            model = getattr(model, part)
        return model

    def count_parameters(self) -> dict:
        """统计参数数量"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
            'trainable_percent': 100 * trainable / total
        }
```

### 组件2: LoRA训练配置

```python
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LoRAConfig:
    """
    LoRA训练配置
    """
    # LoRA超参数
    rank: int = 8
    lora_alpha: float = 16
    lora_dropout: float = 0.05

    # 目标模块
    target_modules: List[str] = None

    # 训练参数
    learning_rate: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100

    def __post_init__(self):
        if self.target_modules is None:
            # 默认应用到Attention的q和v
            self.target_modules = ['q_proj', 'v_proj']


# 常用配置预设
LORA_CONFIGS = {
    'default': LoRAConfig(rank=8, lora_alpha=16),
    'large': LoRAConfig(rank=16, lora_alpha=32),
    'small': LoRAConfig(rank=4, lora_alpha=8),
    'ultra_small': LoRAConfig(rank=1, lora_alpha=2),
}
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **PEFT** | Parameter-Efficient Fine-Tuning | 参数高效微调 |
| **LoRA** | Low-Rank Adaptation | 低秩适配 |
| **Adapter** | Adapter | 适配器模块 |
| **Prompt Tuning** | Prompt Tuning | 提示微调 |
| **Rank** | Rank | 低秩矩阵的秩 |
| **Alpha** | Alpha | LoRA缩放系数 |
| **Scaling** | Scaling | 实际缩放比例 = alpha / rank |

---

## ✅ 复习检查清单

- [ ] 理解PEFT的核心动机和优势
- [ ] 掌握LoRA的原理和实现
- [ ] 了解Adapter的结构和应用
- [ ] 理解Prompt Tuning的概念
- [ ] 能够选择合适的PEFT方法
- [ ] 能够实现LoRA训练流程

---

## 🤔 思考问题

1. **为什么LoRA使用低秩分解？**
   - 提示: 参数效率、过拟合控制

2. **LoRA的rank如何选择？**
   - 提示: 任务复杂度、模型大小

3. **PEFT方法能否组合使用？**
   - 提示: LoRA + Adapter等

4. **井盖检测模型能否使用PEFT？**
   - 提示: 预训练模型微调

---

## 🔗 相关论文推荐

### 必读
1. **LoRA** (ICLR 2022) - 低秩适配
2. **Adapter** (ICML 2019) - 适配器
3. **Prompt Tuning** (EMNLP 2021) - 提示微调

### 扩展阅读
1. **Prefix Tuning** (ACL 2021) - 前缀微调
2. **IA³** (EMNLP 2022) - 学习缩放向量
3. **UniPELT** (EMNLP 2022) - 统一PEFT框架

---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
**下一步**: 实践LoRA在井盖检测模型上的应用
