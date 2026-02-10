"""
LoRA微调快速开始示例

本示例演示如何使用LoRA对GPT-2模型进行微调。
适用于快速验证和理解LoRA微调流程。

运行前请确保:
1. 已安装依赖: pip install -r requirements.txt
2. 有足够的GPU显存 (或使用CPU，但会很慢)
3. 网络连接正常 (需要下载预训练模型)

使用方法:
    python examples/quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.lora_finetune import LoRAModel
from src.dataset import TextDataset
from transformers import AutoTokenizer
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    主函数: 完整的LoRA微调流程
    """
    print("="*70)
    print("LoRA微调快速开始示例")
    print("="*70)
    
    # ==================== 1. 配置 ====================
    print("\n[1/6] 配置参数...")
    
    # 模型配置
    BASE_MODEL = "gpt2"  # 使用GPT-2作为示例 (小模型，适合测试)
    LORA_RANK = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    
    # 训练配置
    MAX_LENGTH = 128
    NUM_EPOCHS = 1  # 示例中只训练1轮
    BATCH_SIZE = 4  # 小批次以适应显存
    LEARNING_RATE = 3e-4
    
    # 输出目录
    OUTPUT_DIR = "./outputs/quickstart"
    
    print(f"  基础模型: {BASE_MODEL}")
    print(f"  LoRA秩: {LORA_RANK}, Alpha: {LORA_ALPHA}")
    print(f"  训练轮数: {NUM_EPOCHS}, 批次大小: {BATCH_SIZE}")
    
    # ==================== 2. 加载模型 ====================
    print("\n[2/6] 加载预训练模型...")
    
    model = LoRAModel(
        base_model=BASE_MODEL,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        device="auto"
    )
    
    # 打印模型信息
    model.print_trainable_parameters()
    
    # ==================== 3. 准备数据 ====================
    print("\n[3/6] 准备训练数据...")
    
    # 使用示例文本数据 (实际应用中应使用真实数据集)
    sample_texts = [
        "人工智能是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的系统。",
        "机器学习是人工智能的一个重要子领域，它使计算机能够从数据中学习。",
        "深度学习是机器学习的一种方法，使用多层神经网络来模拟人脑的工作方式。",
        "自然语言处理是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
        "计算机视觉是人工智能领域，使计算机能够从数字图像或视频中获得高层次的理解。",
        "强化学习是一种机器学习方法，通过与环境交互来学习如何做出决策。",
        "神经网络是受生物神经网络启发的计算模型，用于估计或近似函数。",
        "Transformer是一种神经网络架构，广泛应用于自然语言处理任务。",
        "注意力机制是深度学习中的关键技术，允许模型聚焦于输入的重要部分。",
        "预训练语言模型是在大规模文本数据上训练的模型，可以微调到特定任务。",
        "迁移学习是一种机器学习方法，将已学习的知识应用到不同但相关的问题上。",
        "生成式AI能够创建新的内容，包括文本、图像、音频和视频。",
        "ChatGPT是一种大型语言模型，能够进行自然对话和完成各种任务。",
        "数据增强是一种技术，通过创建训练数据的修改版本来增加数据量。",
        "过拟合是机器学习中的一种现象，模型在训练数据上表现很好但在新数据上表现差。",
        "正则化是用于防止过拟合的技术，通过在损失函数中添加惩罚项。",
        "梯度下降是一种优化算法，用于最小化损失函数。",
        "反向传播是训练神经网络的核心算法，用于计算梯度。",
        "卷积神经网络是专门用于处理网格状数据（如图像）的神经网络。",
        "循环神经网络是处理序列数据的神经网络，具有记忆功能。",
    ]
    
    # 创建数据集
    train_dataset = TextDataset(
        texts=sample_texts,
        tokenizer=model.tokenizer,
        max_length=MAX_LENGTH
    )
    
    print(f"  训练样本数: {len(train_dataset)}")
    
    # ==================== 4. 测试生成 (微调前) ====================
    print("\n[4/6] 测试生成能力 (微调前)...")
    
    test_prompt = "人工智能"
    print(f"  提示: '{test_prompt}'")
    
    try:
        generated_before = model.generate(
            prompt=test_prompt,
            max_length=50,
            temperature=0.7
        )
        print(f"  生成结果: {generated_before}")
    except Exception as e:
        print(f"  生成测试跳过: {e}")
    
    # ==================== 5. 微调 ====================
    print("\n[5/6] 开始微调...")
    print("  (示例中使用少量数据，实际应用需要更多数据和更长的训练时间)")
    
    try:
        model.finetune(
            train_dataset=train_dataset,
            output_dir=OUTPUT_DIR,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            logging_steps=5,
            save_steps=100  # 示例中不频繁保存
        )
        print(f"  微调完成！模型已保存到: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"微调失败: {e}")
        print(f"\n  错误信息: {e}")
        print("  提示: 如果显存不足，请尝试:")
        print("    - 减小 BATCH_SIZE")
        print("    - 减小 MAX_LENGTH")
        print("    - 使用更小的基础模型")
        print("    - 启用8位量化 (修改代码添加 load_in_8bit=True)")
        return
    
    # ==================== 6. 测试生成 (微调后) ====================
    print("\n[6/6] 测试生成能力 (微调后)...")
    
    try:
        generated_after = model.generate(
            prompt=test_prompt,
            max_length=50,
            temperature=0.7
        )
        print(f"  生成结果: {generated_after}")
    except Exception as e:
        print(f"  生成测试跳过: {e}")
    
    # ==================== 完成 ====================
    print("\n" + "="*70)
    print("快速开始示例完成!")
    print("="*70)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n后续步骤:")
    print("  1. 查看生成的模型和日志")
    print("  2. 使用更大规模的数据进行完整训练")
    print("  3. 尝试不同的LoRA超参数 (rank, alpha)")
    print("  4. 评估模型在下游任务上的表现")
    
    print("\n提示: 要加载已保存的模型，使用:")
    print(f"  model.load_adapter('{OUTPUT_DIR}/final')")


if __name__ == "__main__":
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  未检测到GPU，将使用CPU进行训练 (速度较慢)")
    
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断训练")
    except Exception as e:
        logger.exception("运行出错")
        print(f"\n错误: {e}")
        sys.exit(1)
