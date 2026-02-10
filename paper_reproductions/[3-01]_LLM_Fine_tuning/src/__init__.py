"""
大模型高效微调实现包

本包实现了基于LoRA的参数高效微调方法，支持多种大型语言模型。

主要模块:
    - lora_finetune: LoRA微调核心实现
    - model: 模型定义与包装
    - dataset: 数据处理和加载
    - trainer: 训练器

使用示例:
    >>> from src.lora_finetune import LoRAModel
    >>> model = LoRAModel(base_model="gpt2", lora_rank=8)
    >>> model.finetune(dataset)

作者: Xiaohao Cai
版本: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Xiaohao Cai"

# 主要导出类
from .lora_finetune import LoRAModel
from .model import load_base_model, load_lora_model
from .dataset import TextDataset, load_dataset
from .trainer import LoRATrainer

__all__ = [
    "LoRAModel",
    "load_base_model", 
    "load_lora_model",
    "TextDataset",
    "load_dataset",
    "LoRATrainer",
]
