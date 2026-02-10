"""
训练器模块

提供LoRA微调的完整训练流程，包括:
- 训练循环
- 评估
- 日志记录
- 模型保存
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import logging
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    LoRA训练器
    
    提供完整的训练流程，包括训练、评估、日志记录和检查点保存。
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./outputs",
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        eval_steps: int = 500,
        save_steps: int = 500,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        device: str = "auto"
    ):
        """
        初始化训练器
        
        参数:
            model: LoRA模型
            tokenizer: 分词器
            train_dataset: 训练数据集
            eval_dataset: 评估数据集 (可选)
            output_dir: 输出目录
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            warmup_steps: 预热步数
            logging_steps: 日志记录步数
            eval_steps: 评估步数
            save_steps: 保存步数
            max_grad_norm: 最大梯度范数 (用于梯度裁剪)
            weight_decay: 权重衰减
            use_wandb: 是否使用Weights & Biases
            use_tensorboard: 是否使用TensorBoard
            device: 设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 如果模型不在正确的设备上，移动它
        if hasattr(model, 'device') and str(model.device) != str(self.device):
            model = model.to(self.device)
        
        # 设置日志记录
        self.writer = None
        if use_tensorboard:
            log_dir = os.path.join(output_dir, "logs")
            self.writer = SummaryWriter(log_dir)
            
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(project="lora-finetuning")
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb未安装，禁用wandb日志")
                self.use_wandb = False
        
        # 初始化优化器和学习率调度器 (在训练时)
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        
    def _create_optimizer(self):
        """创建优化器"""
        # 只优化可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
    def _create_scheduler(self, num_training_steps: int):
        """创建学习率调度器"""
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
        
    def _create_dataloader(self, dataset: Dataset, shuffle: bool = True) -> DataLoader:
        """创建DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """
        批处理函数
        
        将样本列表转换为批次张量
        """
        # 获取所有键
        keys = batch[0].keys()
        
        # 对每个键堆叠张量
        collated = {}
        for key in keys:
            collated[key] = torch.stack([item[key] for item in batch])
            # 移动到设备
            collated[key] = collated[key].to(self.device)
        
        return collated
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        单步训练
        
        参数:
            batch: 批次数据
            
        返回:
            损失值
        """
        self.model.train()
        
        # 前向传播
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm
        )
        
        # 优化器步进
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
    def _eval_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        单步评估
        
        参数:
            batch: 批次数据
            
        返回:
            损失值
        """
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(**batch)
            loss = outputs.loss
        
        return loss.item()
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型
        
        返回:
            包含评估指标的字典
        """
        if self.eval_dataset is None:
            return {}
        
        eval_dataloader = self._create_dataloader(self.eval_dataset, shuffle=False)
        
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(eval_dataloader, desc="评估"):
            loss = self._eval_step(batch)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity
        }
    
    def _log_metrics(self, metrics: Dict[str, float], step: int):
        """记录指标"""
        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        
        # Weights & Biases
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
    
    def _save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # 保存分词器
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        # 保存训练状态
        state = {
            "global_step": self.global_step,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
        }
        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
        
        logger.info(f"检查点已保存: {checkpoint_dir}")
    
    def train(self):
        """
        主训练循环
        """
        # 创建数据加载器
        train_dataloader = self._create_dataloader(self.train_dataset, shuffle=True)
        
        # 计算总训练步数
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = num_update_steps_per_epoch * self.num_epochs
        
        # 创建优化器和调度器
        self._create_optimizer()
        self._create_scheduler(num_training_steps)
        
        logger.info(f"***** 开始训练 *****")
        logger.info(f"  训练样本数: {len(self.train_dataset)}")
        logger.info(f"  训练轮数: {self.num_epochs}")
        logger.info(f"  批次大小: {self.batch_size}")
        logger.info(f"  总训练步数: {num_training_steps}")
        logger.info(f"  学习率: {self.learning_rate}")
        
        self.global_step = 0
        
        # 训练循环
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{self.num_epochs}"
            )
            
            for batch in progress_bar:
                loss = self._train_step(batch)
                epoch_loss += loss
                self.global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})
                
                # 日志记录
                if self.global_step % self.logging_steps == 0:
                    avg_loss = epoch_loss / (self.global_step % num_update_steps_per_epoch + 1)
                    lr = self.scheduler.get_last_lr()[0]
                    
                    metrics = {
                        "train/loss": loss,
                        "train/avg_loss": avg_loss,
                        "train/learning_rate": lr
                    }
                    self._log_metrics(metrics, self.global_step)
                
                # 评估
                if self.eval_dataset is not None and self.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self._log_metrics(eval_metrics, self.global_step)
                    
                    logger.info(f"评估结果: {eval_metrics}")
                
                # 保存检查点
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(self.global_step)
            
            # 每轮结束的平均损失
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} 完成，平均损失: {avg_epoch_loss:.4f}")
        
        # 保存最终模型
        final_dir = os.path.join(self.output_dir, "final")
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(final_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(final_dir)
        
        logger.info(f"训练完成！最终模型已保存到: {final_dir}")
        
        # 关闭日志记录
        if self.writer is not None:
            self.writer.close()
        if self.use_wandb:
            self.wandb.finish()


if __name__ == "__main__":
    """
    训练器模块测试
    """
    print("="*60)
    print("训练器模块测试")
    print("="*60)
    
    print("\n1. 训练器组件说明:")
    print("   - LoRATrainer: 完整的训练流程管理")
    print("   - 支持: 训练/评估/日志记录/检查点保存")
    print("   - 集成: TensorBoard 和 Weights & Biases")
    
    print("\n2. 使用方法:")
    print("""
    from src.trainer import LoRATrainer
    
    trainer = LoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir="./outputs",
        num_epochs=3,
        batch_size=8,
        learning_rate=3e-4
    )
    
    trainer.train()
    """)
    
    print("\n✅ 模块定义完成!")
    print("   要运行完整训练，请使用 examples/quickstart.py")
