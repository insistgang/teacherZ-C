"""
LoRA (Low-Rank Adaptation) 微调实现

本模块实现了LoRA微调的核心功能，包括:
1. LoRA层的定义和初始化
2. 模型包装和注入LoRA层
3. 训练流程

参考文献:
    Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import logging

# 设置日志
logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """
    LoRA层实现
    
    LoRA通过在原始权重矩阵旁添加低秩分解矩阵来进行微调:
    W = W_0 + BA
    其中 B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d, k)
    
    属性:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: 低秩矩阵的秩
        alpha: 缩放因子
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 32,
        dropout: float = 0.0
    ):
        """
        初始化LoRA层
        
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度  
            rank: 低秩矩阵的秩 (r)
            alpha: 缩放因子
            dropout: dropout概率
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放系数
        
        # 低秩分解矩阵
        # A矩阵: 使用随机高斯初始化
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        # B矩阵: 使用零初始化
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout层
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """
        初始化参数
        
        按照论文建议:
        - A使用Kaiming均匀初始化
        - B使用零初始化
        """
        # A使用Kaiming初始化
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        # B使用零初始化，确保训练开始时LoRA的影响为零
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        计算: dropout(x) @ A @ B * scaling
        
        参数:
            x: 输入张量，形状为 (..., in_features)
            
        返回:
            LoRA输出，形状为 (..., out_features)
        """
        # 应用dropout
        x_dropped = self.dropout(x)
        # 计算低秩更新: x @ A @ B
        # x_dropped @ lora_A: (..., in_features) @ (in_features, rank) -> (..., rank)
        # @ lora_B: (..., rank) @ (rank, out_features) -> (..., out_features)
        output = (x_dropped @ self.lora_A @ self.lora_B) * self.scaling
        return output
    
    def merge_weights(self, base_weight: torch.Tensor) -> torch.Tensor:
        """
        合并LoRA权重到基础权重
        
        计算: W_merged = W_base + BA * scaling
        
        参数:
            base_weight: 基础权重矩阵，形状为 (out_features, in_features)
            
        返回:
            合并后的权重矩阵
        """
        # 计算低秩更新
        lora_update = (self.lora_A @ self.lora_B).T * self.scaling
        return base_weight + lora_update


class LinearWithLoRA(nn.Module):
    """
    带LoRA的线性层
    
    将原始线性层与LoRA层结合，实现:
    output = x @ W_0^T + x @ (BA)^T * scaling
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 32,
        dropout: float = 0.0
    ):
        """
        初始化
        
        参数:
            base_layer: 原始线性层
            rank: LoRA秩
            alpha: 缩放因子
            dropout: dropout概率
        """
        super().__init__()
        self.base_layer = base_layer
        # 冻结基础层参数
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
        # 创建LoRA层
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量
            
        返回:
            输出张量 (基础层输出 + LoRA输出)
        """
        # 基础层输出
        base_output = self.base_layer(x)
        # LoRA输出
        lora_output = self.lora(x)
        return base_output + lora_output


class LoRAModel:
    """
    LoRA模型包装类
    
    提供高层次的LoRA微调接口，包括:
    - 加载预训练模型
    - 注入LoRA层
    - 训练流程
    - 模型保存和加载
    """
    
    def __init__(
        self,
        base_model: str,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        device: str = "auto"
    ):
        """
        初始化LoRA模型
        
        参数:
            base_model: 基础模型名称或路径 (如 "gpt2", "meta-llama/Llama-2-7b")
            lora_rank: LoRA秩
            lora_alpha: LoRA缩放因子
            lora_dropout: LoRA dropout率
            target_modules: 目标模块列表 (如 ["q_proj", "v_proj"])
            device: 设备 ("auto", "cuda", "cpu")
        """
        self.base_model_name = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device = device
        
        # 默认目标模块 (针对Transformer模型)
        if target_modules is None:
            # 检测模型类型并设置默认模块
            target_modules = self._get_default_target_modules(base_model)
        self.target_modules = target_modules
        
        # 加载模型和分词器
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        logger.info(f"LoRA模型初始化完成: rank={lora_rank}, alpha={lora_alpha}")
        logger.info(f"可训练参数: {self.get_trainable_parameters():,}")
        
    def _get_default_target_modules(self, model_name: str) -> List[str]:
        """
        获取默认目标模块
        
        根据模型类型返回适合的目标模块列表
        
        参数:
            model_name: 模型名称
            
        返回:
            目标模块列表
        """
        # GPT-2/OPT类模型
        if any(x in model_name.lower() for x in ["gpt", "opt", "bloom"]):
            return ["c_attn", "c_proj"]
        # LLaMA类模型
        elif any(x in model_name.lower() for x in ["llama", "mistral", "mixtral"]):
            return ["q_proj", "k_proj", "v_proj", "o_proj"]
        # BART/T5类模型
        elif any(x in model_name.lower() for x in ["bart", "t5", "flan-t5"]):
            return ["q", "k", "v", "o"]
        else:
            # 默认模块
            return ["q_proj", "v_proj"]
    
    def _load_model(self):
        """
        加载基础模型并应用LoRA配置
        """
        logger.info(f"正在加载基础模型: {self.base_model_name}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 创建LoRA配置
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device if self.device != "auto" else "auto",
            trust_remote_code=True
        )
        
        # 应用LoRA配置
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info("模型加载完成")
        
    def get_trainable_parameters(self) -> int:
        """
        获取可训练参数数量
        
        返回:
            可训练参数总数
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """
        获取总参数数量
        
        返回:
            总参数数
        """
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters())
    
    def print_trainable_parameters(self):
        """
        打印可训练参数信息
        """
        trainable_params = self.get_trainable_parameters()
        total_params = self.get_total_parameters()
        percentage = 100 * trainable_params / total_params if total_params > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  可训练比例: {percentage:.4f}%")
        print(f"{'='*50}\n")
        
    def finetune(
        self,
        train_dataset,
        eval_dataset=None,
        output_dir: str = "./outputs",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """
        微调模型
        
        参数:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集 (可选)
            output_dir: 输出目录
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            **kwargs: 额外参数传递给训练器
        """
        from .trainer import LoRATrainer
        
        trainer = LoRATrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs
        )
        
        trainer.train()
        
    def save_model(self, save_path: str):
        """
        保存LoRA权重
        
        参数:
            save_path: 保存路径
        """
        if self.model is None:
            raise ValueError("模型未加载")
            
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"模型已保存到: {save_path}")
        
    def load_adapter(self, adapter_path: str):
        """
        加载LoRA适配器
        
        参数:
            adapter_path: 适配器路径
        """
        if self.model is None:
            raise ValueError("模型未加载")
            
        self.model = PeftModel.from_pretrained(
            self.model,
            adapter_path
        )
        logger.info(f"适配器已加载: {adapter_path}")
        
    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        生成文本
        
        参数:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            **kwargs: 额外生成参数
            
        返回:
            生成的文本
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("模型未加载")
            
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                **kwargs
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    """
    示例用法
    
    演示如何创建LoRA模型并进行简单测试
    """
    print("="*60)
    print("LoRA微调模块测试")
    print("="*60)
    
    # 创建简单的LoRA层测试
    print("\n1. 测试LoRA层...")
    lora_layer = LoRALayer(in_features=512, out_features=512, rank=8, alpha=32)
    test_input = torch.randn(2, 10, 512)  # (batch, seq_len, features)
    test_output = lora_layer(test_input)
    print(f"   输入形状: {test_input.shape}")
    print(f"   输出形状: {test_output.shape}")
    print(f"   LoRA参数量: A={lora_layer.lora_A.numel()}, B={lora_layer.lora_B.numel()}")
    
    # 测试完整模型 (需要下载模型，默认注释掉)
    # print("\n2. 测试完整LoRA模型...")
    # model = LoRAModel(
    #     base_model="gpt2",
    #     lora_rank=8,
    #     lora_alpha=32
    # )
    # model.print_trainable_parameters()
    
    print("\n✅ 测试完成!")
