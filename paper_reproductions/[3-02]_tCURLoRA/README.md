# [3-02] tCURLoRA张量分解 (Tensor CUR Decomposition LoRA)

## 论文信息

**标题**: tCURLoRA: Tensor CUR Decomposition for Low-Rank Adaptation of Large Language Models

**作者**: Xiaohao Cai 等

**发表**: 2024 (张量分解 + 参数高效微调)

**论文路径**: `xiaohao_cai_papers/[3-02] 张量CUR分解LoRA tCURLoRA.pdf`

---

## 核心贡献简介

本论文提出了tCURLoRA，一种基于张量CUR分解的参数高效微调方法：

### 1. 张量CUR分解

**CUR分解**:
- C: 选取的列 (Columns)
- U: 连接矩阵
- R: 选取的行 (Rows)

**张量扩展 (tCUR)**:
- 将矩阵CUR扩展到高阶张量
- 保持低秩结构的同时减少参数

### 2. LoRA改进

**传统LoRA**:
```
W = W_0 + BA
参数: r × (d_in + d_out)
```

**tCURLoRA**:
```
W = W_0 + CUR
参数: 大大减少，特别是高维情况
```

**优势**:
- ✅ 更少的可训练参数
- ✅ 保持或提升模型性能
- ✅ 更快的训练速度
- ✅ 更好的可解释性

### 3. 应用场景

- 大语言模型微调
- 多模态模型适应
- 跨语言迁移学习

---

## 复现状态

| 组件 | 状态 | 说明 |
|:---|:---:|:---|
| 张量CUR分解 | 🟡 进行中 | 核心算法框架已搭建 |
| tCURLoRA层 | 🟡 进行中 | 基础实现完成 |
| 训练框架 | 🔴 待完成 | 待集成 |
| 评估指标 | 🔴 待完成 | 待实现 |
| 示例脚本 | 🟡 进行中 | 基础示例可用 |

**总体状态**: 🟡 **进行中** (约50%完成)

---

## 文件结构说明

```
[3-02]_tCURLoRA/
├── README.md                    # 本文件
├── requirements.txt             # Python依赖
├── src/                         # 源代码
│   ├── __init__.py             # 包初始化
│   ├── tcur_lora.py            # tCURLoRA核心实现
│   ├── tensor_ops.py           # 张量操作
│   └── train.py                # 训练脚本
└── examples/                    # 示例代码
    └── finetune_example.py     # 微调示例
```

---

## 使用方法

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

### 快速开始

```python
# 导入模块
from src.tcur_lora import tCURLoRAModel

# 创建tCURLoRA模型
model = tCURLoRAModel(
    base_model="gpt2",
    tensor_rank=8,
    num_columns=16,
    num_rows=16
)

# 打印参数统计
model.print_trainable_parameters()

# 训练
model.finetune(dataset, num_epochs=3)
```

### 使用示例脚本

```bash
# 运行微调示例
python examples/finetune_example.py --model gpt2 --dataset wikitext
```

---

## 核心概念

### 矩阵CUR分解

给定矩阵 A ∈ ℝ^{m×n}:
```
A ≈ CUR
```

其中:
- C ∈ ℝ^{m×c}: 选取的c列
- U ∈ ℝ^{c×r}: 连接矩阵
- R ∈ ℝ^{r×n}: 选取的r行

### 张量CUR分解

对于3阶张量 𝒜 ∈ ℝ^{I×J×K}:
```
𝒜 ≈ 𝒞 ×₁ U₁ ×₂ U₂ ×₃ U₃ × ℛ
```

其中 ×ₙ 表示n-模乘积。

### tCURLoRA的优势

| 方法 | 参数数量 | 存储效率 |
|:---|:---:|:---:|
| Full Fine-tuning | d × d | 1× |
| LoRA | r×(d_in+d_out) | ~10× |
| **tCURLoRA** | c×d + r×c + r×d | ~20× |

---

## 依赖要求

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.30
- NumPy >= 1.24
- tensorly >= 0.8 (张量分解)

---

## 参考文献

1. Cai, X., et al. (2024). tCURLoRA: Tensor CUR Decomposition for Low-Rank Adaptation.
2. Mahoney, M. W., & Drineas, P. (2009). CUR matrix decompositions for improved data analysis.
3. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
4. Kolda, T. G., & Bader, B. W. (2009). Tensor decompositions and applications.

---

## 更新日志

- **2024-XX-XX**: 创建复现框架
- **2024-XX-XX**: 实现基础张量CUR分解
- **2024-XX-XX**: 集成LoRA框架
