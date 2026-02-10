"""
数据集处理模块

提供数据加载、预处理和批处理功能。
支持多种数据集格式和来源。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset as hf_load_dataset
from typing import Optional, Union, List, Dict, Callable, Any
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    文本数据集类
    
    用于加载和预处理文本数据，支持:
    - 从文件加载
    - 分词和编码
    - 动态填充或截断
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True
    ):
        """
        初始化文本数据集
        
        参数:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 最大序列长度
            padding: 填充策略 ("max_length", "longest")
            truncation: 是否截断
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        参数:
            idx: 索引
            
        返回:
            包含input_ids和attention_mask的字典
        """
        text = self.texts[idx]
        
        # 分词和编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        # 移除batch维度
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }


class InstructionDataset(Dataset):
    """
    指令微调数据集
    
    用于指令微调（Instruction Tuning）格式的数据，
    每条数据包含instruction、input和output字段。
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer,
        max_length: int = 512,
        template: str = "alpaca"
    ):
        """
        初始化指令数据集
        
        参数:
            data: 数据列表，每项为 {"instruction": ..., "input": ..., "output": ...}
            tokenizer: 分词器
            max_length: 最大序列长度
            template: 模板类型 ("alpaca", "chatml", "vicuna")
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
        
    def _format_example(self, example: Dict[str, str]) -> str:
        """
        格式化单个样本
        
        参数:
            example: 包含instruction、input、output的字典
            
        返回:
            格式化后的文本
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")
        
        if self.template == "alpaca":
            # Alpaca格式
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            return prompt + output
            
        elif self.template == "chatml":
            # ChatML格式
            if input_text:
                prompt = f"<|im_start|>user\n{instruction}\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
            else:
                prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            return prompt + output + "<|im_end|>"
            
        else:
            # 简单格式
            if input_text:
                return f"{instruction}\nInput: {input_text}\nOutput: {output}"
            else:
                return f"{instruction}\nOutput: {output}"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self._format_example(self.data[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 对于因果语言模型，labels与input_ids相同
        input_ids = encoding["input_ids"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": input_ids.clone()  # 用于计算loss
        }


def load_dataset(
    dataset_name: str,
    tokenizer,
    split: str = "train",
    max_length: int = 512,
    text_column: str = "text",
    subset: Optional[str] = None,
    num_samples: Optional[int] = None,
    **kwargs
) -> Dataset:
    """
    加载Hugging Face数据集
    
    参数:
        dataset_name: 数据集名称 (如 "wikitext", "openwebtext")
        tokenizer: 分词器
        split: 数据集划分 ("train", "validation", "test")
        max_length: 最大序列长度
        text_column: 文本列名
        subset: 子集名称 (如 "wikitext-2-raw-v1")
        num_samples: 使用的样本数 (None表示全部)
        **kwargs: 额外参数传递给load_dataset
        
    返回:
        TextDataset实例
        
    示例:
        >>> dataset = load_dataset("wikitext", tokenizer, subset="wikitext-2-raw-v1")
        >>> dataset = load_dataset("openwebtext", tokenizer, num_samples=10000)
    """
    logger.info(f"正在加载数据集: {dataset_name}, subset={subset}, split={split}")
    
    # 加载数据集
    if subset:
        dataset = hf_load_dataset(dataset_name, subset, split=split, **kwargs)
    else:
        dataset = hf_load_dataset(dataset_name, split=split, **kwargs)
    
    # 限制样本数
    if num_samples is not None and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
    
    # 过滤空文本
    texts = [text for text in dataset[text_column] if text and len(text.strip()) > 0]
    
    logger.info(f"数据集加载完成，共 {len(texts)} 条样本")
    
    return TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length
    )


def load_instruction_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    template: str = "alpaca"
) -> InstructionDataset:
    """
    加载指令微调数据集（JSON格式）
    
    参数:
        data_path: JSON文件路径
        tokenizer: 分词器
        max_length: 最大序列长度
        template: 模板类型
        
    返回:
        InstructionDataset实例
        
    示例:
        >>> dataset = load_instruction_dataset("./alpaca_data.json", tokenizer)
    """
    import json
    
    logger.info(f"正在加载指令数据集: {data_path}")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"指令数据集加载完成，共 {len(data)} 条样本")
    
    return InstructionDataset(
        data=data,
        tokenizer=tokenizer,
        max_length=max_length,
        template=template
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    创建DataLoader
    
    参数:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 数据加载进程数
        pin_memory: 是否固定内存
        
    返回:
        DataLoader实例
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None  # 使用默认的collate_fn
    )


def split_dataset(
    dataset: Dataset,
    train_ratio: float = 0.9,
    seed: int = 42
) -> tuple[Dataset, Dataset]:
    """
    划分训练集和验证集
    
    参数:
        dataset: 原始数据集
        train_ratio: 训练集比例
        seed: 随机种子
        
    返回:
        (train_dataset, val_dataset) 元组
    """
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    logger.info(f"数据集划分: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    """
    测试数据集模块
    
    运行此测试需要安装transformers和datasets
    """
    print("="*60)
    print("数据集处理模块测试")
    print("="*60)
    
    # 创建模拟数据测试
    print("\n1. 测试TextDataset...")
    mock_texts = [
        "这是一个测试句子。",
        "这是第二个测试句子，稍微长一点。",
        "Short text.",
        "This is a longer piece of text that might need truncation if the max length is set very short."
    ]
    
    # 注意: 实际测试需要真实的tokenizer
    print(f"   模拟文本数: {len(mock_texts)}")
    print("   (完整测试需要真实的tokenizer)")
    
    # 测试指令数据集格式化
    print("\n2. 测试指令格式化...")
    mock_data = [
        {
            "instruction": "翻译成英文",
            "input": "你好世界",
            "output": "Hello world"
        },
        {
            "instruction": "解释概念",
            "input": "",
            "output": "这是一个概念解释..."
        }
    ]
    
    # 展示不同模板
    templates = ["alpaca", "chatml", "simple"]
    for template in templates:
        print(f"\n   模板: {template}")
        for item in mock_data:
            # 模拟格式化
            inst = item["instruction"]
            inp = item.get("input", "")
            out = item["output"]
            
            if template == "alpaca":
                if inp:
                    formatted = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                else:
                    formatted = f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            elif template == "chatml":
                if inp:
                    formatted = f"<|im_start|>user\n{inst}\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
                else:
                    formatted = f"<|im_start|>user\n{inst}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"
            else:
                formatted = f"{inst}\nOutput: {out}"
            
            print(f"      {formatted[:60]}...")
    
    print("\n✅ 测试完成!")
    print("\n提示: 要测试真实数据加载，请运行:")
    print("  from transformers import AutoTokenizer")
    print("  tokenizer = AutoTokenizer.from_pretrained('gpt2')")
    print("  dataset = load_dataset('wikitext', tokenizer, subset='wikitext-2-raw-v1')")
