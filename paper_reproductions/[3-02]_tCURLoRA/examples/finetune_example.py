"""
tCURLoRA微调示例

演示如何使用tCURLoRA对预训练模型进行参数高效微调。

使用方法:
    python examples/finetune_example.py --model gpt2 --dataset wikitext

参数:
    --model: 基础模型名称
    --dataset: 数据集名称
    --rank: CUR分解秩
    --output: 输出目录
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
from tqdm import tqdm

from src.tcur_lora import convert_to_tcur_lora


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='tCURLoRA微调示例')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='gpt2',
                        help='基础模型名称')
    parser.add_argument('--rank', type=int, default=8,
                        help='CUR分解秩')
    parser.add_argument('--alpha', type=int, default=32,
                        help='缩放因子')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='wikitext',
                        help='数据集名称')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1',
                        help='数据集配置')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    
    # 其他参数
    parser.add_argument('--output', type=str, default='./outputs/tcur_lora',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                        help='设备')
    
    return parser.parse_args()


def get_device(device: str):
    """获取设备"""
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def prepare_dataset(dataset_name, config_name, tokenizer, max_length):
    """准备数据集"""
    print(f"\n加载数据集: {dataset_name}")
    
    dataset = load_dataset(dataset_name, config_name)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    return tokenized


def main():
    """主函数"""
    args = parse_args()
    device = get_device(args.device)
    
    print("="*70)
    print("tCURLoRA微调示例")
    print("="*70)
    print(f"设备: {device}")
    print(f"基础模型: {args.model}")
    print(f"CUR秩: {args.rank}")
    print(f"缩放因子: {args.alpha}")
    
    # 加载模型和分词器
    print("\n加载预训练模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(args.model)
    base_model = base_model.to(device)
    
    # 转换为tCURLoRA
    print("\n应用tCURLoRA...")
    model = convert_to_tcur_lora(
        base_model,
        target_modules=['c_attn', 'c_proj'],  # GPT-2的注意力层
        rank=args.rank,
        alpha=args.alpha
    )
    model = model.to(device)
    
    # 打印参数统计
    model.print_trainable_parameters()
    
    # 准备数据
    dataset = prepare_dataset(
        args.dataset,
        args.dataset_config,
        tokenizer,
        args.max_length
    )
    
    # 创建数据加载器
    train_dataloader = DataLoader(
        dataset['train'],
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )
    
    # 学习率调度
    num_training_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps
    )
    
    # 训练循环
    print("\n开始训练...")
    model.train()
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f'Training')
        for batch in progress_bar:
            # 准备输入
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"平均损失: {avg_loss:.4f}")
    
    # 保存模型
    import os
    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, 'tcur_lora_weights.pt')
    model.save_pretrained(save_path)
    
    print("\n" + "="*70)
    print("训练完成!")
    print(f"模型保存: {save_path}")
    print("="*70)
    
    # 对比LoRA
    print("\n【对比】传统LoRA vs tCURLoRA:")
    print("  LoRA: W = W_0 + BA")
    print("        参数: r × (d_in + d_out)")
    print("  tCURLoRA: W = W_0 + CUR")
    print("            参数: c×d_in + c×r + r×d_out")
    print(f"  本例中可训练参数: {model.get_trainable_parameters():,}")
    print("\ntCURLoRA优势:")
    print("  - 参数更少")
    print("  - 使用实际权重列/行，可解释性更好")
    print("  - 训练更稳定")


if __name__ == '__main__':
    main()
