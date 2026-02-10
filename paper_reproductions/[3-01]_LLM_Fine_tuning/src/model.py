"""
模型定义与加载模块

提供模型加载、保存和管理的工具函数。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def load_base_model(
    model_name: str,
    device: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    **kwargs
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    加载基础预训练模型
    
    参数:
        model_name: 模型名称或路径 (如 "gpt2", "meta-llama/Llama-2-7b-hf")
        device: 设备 ("auto", "cuda", "cpu")
        load_in_8bit: 是否使用8位量化 (节省显存)
        load_in_4bit: 是否使用4位量化 (更省显存)
        torch_dtype: 数据类型 (如 torch.float16)
        trust_remote_code: 是否信任远程代码
        **kwargs: 额外参数
        
    返回:
        (model, tokenizer) 元组
        
    示例:
        >>> model, tokenizer = load_base_model("gpt2")
        >>> model, tokenizer = load_base_model("gpt2", load_in_8bit=True)
    """
    logger.info(f"正在加载模型: {model_name}")
    
    # 自动选择数据类型
    if torch_dtype is None:
        if torch.cuda.is_available() and device != "cpu":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side="right"
    )
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "auto" else "auto",
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    logger.info(f"模型加载完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def load_lora_model(
    base_model_name: str,
    lora_weights_path: str,
    device: str = "auto",
    **kwargs
) -> tuple[PeftModel, AutoTokenizer]:
    """
    加载带有LoRA适配器的模型
    
    参数:
        base_model_name: 基础模型名称
        lora_weights_path: LoRA权重路径
        device: 设备
        **kwargs: 额外参数
        
    返回:
        (model, tokenizer) 元组
        
    示例:
        >>> model, tokenizer = load_lora_model("gpt2", "./lora_outputs")
    """
    logger.info(f"正在加载LoRA模型: {base_model_name} + {lora_weights_path}")
    
    # 加载基础模型
    base_model, tokenizer = load_base_model(
        base_model_name,
        device=device,
        **kwargs
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_weights_path
    )
    
    logger.info("LoRA适配器加载完成")
    
    return model, tokenizer


def merge_and_save_model(
    model: PeftModel,
    save_path: str,
    safe_serialization: bool = True
):
    """
    合并LoRA权重并保存完整模型
    
    将LoRA权重合并到基础模型中，生成一个完整的模型。
    这样加载时不需要再加载LoRA适配器。
    
    参数:
        model: LoRA模型
        save_path: 保存路径
        safe_serialization: 是否使用safetensors格式
    """
    logger.info(f"正在合并并保存模型到: {save_path}")
    
    # 合并权重
    merged_model = model.merge_and_unload()
    
    # 保存模型
    merged_model.save_pretrained(
        save_path,
        safe_serialization=safe_serialization
    )
    
    logger.info("模型合并并保存完成")


def get_model_info(model: torch.nn.Module) -> dict:
    """
    获取模型信息
    
    参数:
        model: PyTorch模型
        
    返回:
        包含模型信息的字典
        
    示例:
        >>> info = get_model_info(model)
        >>> print(f"总参数: {info['total_params']:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小 (MB)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "trainable_percentage": 100 * trainable_params / total_params if total_params > 0 else 0,
        "size_mb": size_mb,
        "num_layers": len(list(model.modules()))
    }


def print_model_info(model: torch.nn.Module, detailed: bool = False):
    """
    打印模型信息
    
    参数:
        model: PyTorch模型
        detailed: 是否打印详细信息 (各层参数)
    """
    info = get_model_info(model)
    
    print("\n" + "="*60)
    print("模型信息概览")
    print("="*60)
    print(f"总参数数量:     {info['total_params']:>15,}")
    print(f"可训练参数:     {info['trainable_params']:>15,}")
    print(f"冻结参数:       {info['frozen_params']:>15,}")
    print(f"可训练比例:     {info['trainable_percentage']:>15.4f}%")
    print(f"模型大小:       {info['size_mb']:>15.2f} MB")
    print(f"层数:           {info['num_layers']:>15,}")
    print("="*60)
    
    if detailed:
        print("\n各层参数详情:")
        print("-"*60)
        for name, param in model.named_parameters():
            status = "可训练" if param.requires_grad else "冻结"
            print(f"{name:50s} {param.numel():>10,} [{status}]")


if __name__ == "__main__":
    """
    测试模型加载功能
    
    注意: 运行此测试需要下载模型，可能需要较长时间
    """
    print("="*60)
    print("模型加载模块测试")
    print("="*60)
    
    # 测试模型信息获取 (使用简单模型)
    print("\n1. 测试模型信息获取...")
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 20)
            self.linear2 = torch.nn.Linear(20, 5)
            
        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))
    
    test_model = SimpleModel()
    # 冻结部分参数
    for param in test_model.linear1.parameters():
        param.requires_grad = False
    
    print_model_info(test_model, detailed=True)
    
    print("\n✅ 基础测试完成!")
    print("\n提示: 要测试真实模型加载，请取消注释以下代码:")
    print("  model, tokenizer = load_base_model('gpt2')")
