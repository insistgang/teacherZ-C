"""
tCURLoRA张量分解微调实现包

本包实现了基于张量CUR分解的参数高效微调方法，
相比传统LoRA进一步减少可训练参数。

主要模块:
    - tcur_lora: tCURLoRA核心实现
    - tensor_ops: 张量操作工具
    - train: 训练框架

使用示例:
    >>> from src.tcur_lora import tCURLoRAModel
    >>> model = tCURLoRAModel(base_model="gpt2", tensor_rank=8)
    >>> model.finetune(dataset)

核心优势:
    - 比LoRA更少的参数
    - 保持模型性能
    - 更快的训练速度

作者: Xiaohao Cai
版本: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Xiaohao Cai"

from .tcur_lora import tCURLoRALayer, tCURLoRAModel
from .tensor_ops import (
    cur_decomposition,
    tensor_cur_decomposition,
    n_mode_product
)

__all__ = [
    "tCURLoRALayer",
    "tCURLoRAModel",
    "cur_decomposition",
    "tensor_cur_decomposition",
    "n_mode_product",
]
