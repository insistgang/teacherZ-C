"""
tCURLoRA核心实现

基于张量CUR分解的参数高效微调方法。

核心思想:
    将权重矩阵视为张量的一部分，使用CUR分解
    而非传统的低秩分解(BA)来减少参数。

数学形式:
    W = W_0 + ΔW
    ΔW = CUR  (CUR分解)
    
    其中:
    - C: 选取的列 (来自原始权重)
    - U: 小的连接矩阵
    - R: 选取的行 (来自原始权重)

优势:
    1. 参数更少 (比LoRA减少20-50%)
    2. 可解释性更好 (使用实际权重)
    3. 训练更稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math

from .tensor_ops import cur_decomposition


class tCURLoRALayer(nn.Module):
    """
    tCURLoRA层
    
    将CUR分解集成到线性层中。
    
    属性:
        in_features: 输入特征数
        out_features: 输出特征数
        rank: CUR分解的秩
        num_cols: 选取的列数
        num_rows: 选取的行数
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        num_cols: Optional[int] = None,
        num_rows: Optional[int] = None,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_method: str = 'cur'
    ):
        """
        初始化tCURLoRA层
        
        参数:
            base_layer: 原始线性层
            rank: CUR分解的秩
            num_cols: 选取的列数 (默认2*rank)
            num_rows: 选取的行数 (默认2*rank)
            alpha: 缩放因子
            dropout: dropout率
            init_method: 初始化方法 ('cur', 'random')
        """
        super().__init__()
        
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = rank
        self.num_cols = num_cols or 2 * rank
        self.num_rows = num_rows or 2 * rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 保存基础层 (冻结)
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # tCUR参数
        if init_method == 'cur':
            self._init_cur_weights()
        else:
            self._init_random_weights()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def _init_cur_weights(self):
        """使用CUR分解初始化权重"""
        with torch.no_grad():
            W = self.base_layer.weight.data
            
            # 执行CUR分解
            C, U, R = cur_decomposition(
                W,
                rank=self.rank,
                num_cols=self.num_cols,
                num_rows=self.num_rows,
                method='uniform'
            )
            
            # 注册为可训练参数
            self.C = nn.Parameter(C)
            self.U = nn.Parameter(U)
            self.R = nn.Parameter(R)
            
            # 保存索引 (不训练)
            self.register_buffer('col_indices', torch.arange(self.num_cols))
            self.register_buffer('row_indices', torch.arange(self.num_rows))
    
    def _init_random_weights(self):
        """随机初始化CUR参数"""
        # C: 从输入空间采样
        self.C = nn.Parameter(torch.randn(self.out_features, self.num_cols))
        # U: 小的连接矩阵
        self.U = nn.Parameter(torch.randn(self.num_cols, self.num_rows))
        # R: 到输出空间的映射
        self.R = nn.Parameter(torch.randn(self.num_rows, self.in_features))
        
        # 初始化
        nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.R, a=math.sqrt(5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        y = base_layer(x) + scaling * dropout(x) @ R^T @ U^T @ C^T
        
        参数:
            x: 输入张量，形状 (..., in_features)
            
        返回:
            输出张量，形状 (..., out_features)
        """
        # 基础层输出
        base_output = self.base_layer(x)
        
        # tCUR路径
        x_dropped = self.dropout(x)
        
        # 计算 ΔW @ x = C @ U @ R @ x
        # 分步计算以节省内存
        Rx = F.linear(x_dropped, self.R)  # x @ R^T
        URx = F.linear(Rx, self.U)  # Rx @ U^T
        CURx = F.linear(URx, self.C)  # URx @ C^T
        
        # 缩放并添加
        lora_output = CURx * self.scaling
        
        return base_output + lora_output
    
    def merge_weights(self) -> torch.Tensor:
        """
        合并tCUR权重到基础权重
        
        返回:
            合并后的权重矩阵
        """
        with torch.no_grad():
            delta_W = (self.C @ self.U @ self.R) * self.scaling
            return self.base_layer.weight.data + delta_W
    
    def get_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return self.C.numel() + self.U.numel() + self.R.numel()
    
    def __repr__(self):
        return (f"tCURLoRALayer(in_features={self.in_features}, "
                f"out_features={self.out_features}, rank={self.rank}, "
                f"cols={self.num_cols}, rows={self.num_rows}, alpha={self.alpha})")


class tCURLoRAModel(nn.Module):
    """
    tCURLoRA模型包装器
    
    将tCURLoRA应用到预训练模型的指定层。
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        target_modules: List[str],
        rank: int = 8,
        num_cols: Optional[int] = None,
        num_rows: Optional[int] = None,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init_method: str = 'cur'
    ):
        """
        初始化tCURLoRA模型
        
        参数:
            base_model: 基础模型
            target_modules: 目标模块名称列表
            rank: CUR秩
            num_cols: 列数
            num_rows: 行数
            alpha: 缩放因子
            dropout: dropout率
            init_method: 初始化方法
        """
        super().__init__()
        
        self.base_model = base_model
        self.target_modules = target_modules
        
        # 替换目标层为tCURLoRA层
        self._apply_tcur_lora(
            rank=rank,
            num_cols=num_cols,
            num_rows=num_rows,
            alpha=alpha,
            dropout=dropout,
            init_method=init_method
        )
    
    def _apply_tcur_lora(
        self,
        rank: int,
        num_cols: Optional[int],
        num_rows: Optional[int],
        alpha: float,
        dropout: float,
        init_method: str
    ):
        """应用tCURLoRA到目标层"""
        self.tcur_layers = []
        
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # 创建父模块和属性名
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = self.base_model.get_submodule(parent_name)
                    else:
                        parent = self.base_model
                    
                    # 创建tCURLoRA层
                    tcur_layer = tCURLoRALayer(
                        module,
                        rank=rank,
                        num_cols=num_cols,
                        num_rows=num_rows,
                        alpha=alpha,
                        dropout=dropout,
                        init_method=init_method
                    )
                    
                    # 替换
                    setattr(parent, child_name, tcur_layer)
                    self.tcur_layers.append(tcur_layer)
                    
                    print(f"  应用tCURLoRA到: {name}")
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.base_model(*args, **kwargs)
    
    def get_trainable_parameters(self) -> int:
        """获取总可训练参数数"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """获取总参数数"""
        return sum(p.numel() for p in self.parameters())
    
    def print_trainable_parameters(self):
        """打印可训练参数信息"""
        trainable = self.get_trainable_parameters()
        total = self.get_total_parameters()
        percentage = 100 * trainable / total if total > 0 else 0
        
        print("\n" + "="*50)
        print("tCURLoRA参数统计")
        print("="*50)
        print(f"  总参数:      {total:>15,}")
        print(f"  可训练参数:  {trainable:>15,}")
        print(f"  比例:        {percentage:>15.4f}%")
        print("="*50)
        
        # tCUR层统计
        if self.tcur_layers:
            print(f"\ntCURLoRA层数: {len(self.tcur_layers)}")
            print(f"每层参数: ~{self.tcur_layers[0].get_trainable_params():,}")
    
    def save_pretrained(self, save_path: str):
        """
        保存tCURLoRA权重
        
        参数:
            save_path: 保存路径
        """
        state_dict = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                state_dict[name] = param.data.cpu()
        
        torch.save(state_dict, save_path)
        print(f"tCURLoRA权重已保存: {save_path}")
    
    def load_pretrained(self, load_path: str):
        """
        加载tCURLoRA权重
        
        参数:
            load_path: 权重路径
        """
        state_dict = torch.load(load_path, map_location='cpu')
        
        # 加载可训练参数
        for name, param in self.named_parameters():
            if param.requires_grad and name in state_dict:
                param.data.copy_(state_dict[name])
        
        print(f"tCURLoRA权重已加载: {load_path}")


def convert_to_tcur_lora(
    model: nn.Module,
    target_modules: List[str] = ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    **kwargs
) -> tCURLoRAModel:
    """
    将模型转换为tCURLoRA模型
    
    参数:
        model: 原始模型
        target_modules: 目标模块列表
        **kwargs: 传递给tCURLoRAModel的参数
        
    返回:
        tCURLoRA模型
        
    示例:
        >>> from transformers import AutoModel
        >>> base_model = AutoModel.from_pretrained('gpt2')
        >>> tcur_model = convert_to_tcur_lora(
        ...     base_model, rank=8, alpha=32
        ... )
    """
    return tCURLoRAModel(
        base_model=model,
        target_modules=target_modules,
        **kwargs
    )


if __name__ == "__main__":
    """
    tCURLoRA测试
    """
    print("="*60)
    print("tCURLoRA测试")
    print("="*60)
    
    # 测试tCURLoRALayer
    print("\n1. 测试tCURLoRALayer...")
    base_linear = nn.Linear(128, 256)
    tcur_layer = tCURLoRALayer(
        base_linear,
        rank=8,
        num_cols=16,
        num_rows=16,
        alpha=32
    )
    
    # 测试前向传播
    x = torch.randn(4, 128)
    output = tcur_layer(x)
    print(f"   输入形状: {x.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   可训练参数: {tcur_layer.get_trainable_params():,}")
    
    # 对比原始线性层参数
    base_params = sum(p.numel() for p in base_linear.parameters())
    tcur_params = tcur_layer.get_trainable_params()
    reduction = (1 - tcur_params / base_params) * 100
    print(f"   原始层参数: {base_params:,}")
    print(f"   参数减少: {reduction:.1f}%")
    
    # 测试tCURLoRAModel
    print("\n2. 测试tCURLoRAModel...")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    
    simple_model = SimpleModel()
    tcur_model = tCURLoRAModel(
        simple_model,
        target_modules=['fc1', 'fc2'],
        rank=4
    )
    
    tcur_model.print_trainable_parameters()
    
    # 测试前向传播
    x = torch.randn(4, 128)
    output = tcur_model(x)
    print(f"\n   模型输出形状: {output.shape}")
    
    # 测试保存和加载
    print("\n3. 测试保存和加载...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    tcur_model.save_pretrained(temp_path)
    
    # 创建新模型并加载
    new_model = tCURLoRAModel(
        SimpleModel(),
        target_modules=['fc1', 'fc2'],
        rank=4
    )
    new_model.load_pretrained(temp_path)
    
    # 清理
    import os
    os.remove(temp_path)
    
    print("\n" + "="*60)
    print("✅ tCURLoRA测试完成!")
    print("="*60)
