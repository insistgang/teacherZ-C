# TOP 5 可复现论文代码框架

本目录包含TOP 5可复现优先级论文的复现代码框架。

## 📚 论文列表

| 优先级 | 编号 | 论文名称 | 复现状态 | 说明 |
|:---:|:---:|:---|:---:|:---|
| 1 | [3-01] | 大模型高效微调 (LLM Fine-tuning) | 🟡 进行中 | LoRA改进实现 |
| 2 | [1-04] | 变分法基础 (Mumford-Shah/ROF) | ✅ 已完成 | 总结现有实现 |
| 3 | [2-09] | 框架分割管状结构 (Framelet Tubular) | 🟡 进行中 | 小波框架分割 |
| 4 | [2-08] | 小波框架血管分割 (Vessel Segmentation) | 🟡 进行中 | DRIVE数据集 |
| 5 | [3-02] | tCURLoRA张量分解 (tCURLoRA) | 🟡 进行中 | 张量CUR分解 |

---

## 📁 目录结构

```
paper_reproductions/
├── README.md                              # 本文件
├── [3-01]_LLM_Fine_tuning/               # 大模型高效微调
│   ├── README.md                         # 论文说明
│   ├── requirements.txt                  # 依赖
│   ├── config.yaml                       # 配置
│   ├── src/                              # 源代码
│   │   ├── lora_finetune.py             # LoRA核心实现
│   │   ├── model.py                      # 模型定义
│   │   ├── dataset.py                    # 数据处理
│   │   └── trainer.py                    # 训练器
│   └── examples/
│       └── quickstart.py                 # 快速开始
├── [1-04]_Variational_Methods/           # 变分法基础
│   └── README.md                         # 现有实现总结
├── [2-09]_Framelet_Tubular/              # 框架分割管状结构
│   ├── README.md
│   ├── requirements.txt
│   ├── src/
│   │   ├── framelet.py                   # 小波框架
│   │   ├── shape_prior.py                # 形状先验
│   │   ├── segmentation.py               # 分割算法
│   │   └── utils.py                      # 工具函数
│   └── examples/
│       └── demo.py                       # 演示脚本
├── [2-08]_Vessel_Segmentation/           # 小波框架血管分割
│   ├── README.md
│   ├── requirements.txt
│   ├── data/
│   │   └── download_drive.py             # DRIVE下载
│   ├── src/
│   │   ├── vessel_net.py                 # 分割网络
│   │   ├── wavelet_frame.py              # 小波框架
│   │   ├── evaluate.py                   # 评估指标
│   │   └── dataset.py                    # 数据集
│   └── examples/
│       └── train.py                      # 训练脚本
└── [3-02]_tCURLoRA/                      # 张量分解LoRA
    ├── README.md
    ├── requirements.txt
    ├── src/
    │   ├── tcur_lora.py                  # tCURLoRA核心
    │   └── tensor_ops.py                 # 张量操作
    └── examples/
        └── finetune_example.py           # 微调示例
```

---

## 🚀 快速开始

### 1. 环境准备

每个论文目录下都有独立的 `requirements.txt`，请分别安装依赖：

```bash
# 进入对应论文目录
cd "[3-01]_LLM_Fine_tuning"
pip install -r requirements.txt
```

### 2. 运行示例

每个论文都提供了快速开始示例：

```bash
# [3-01] LoRA微调
python "[3-01]_LLM_Fine_tuning/examples/quickstart.py"

# [2-09] 管状结构分割
python "[2-09]_Framelet_Tubular/examples/demo.py"

# [2-08] 血管分割训练
python "[2-08]_Vessel_Segmentation/examples/train.py"

# [3-02] tCURLoRA微调
python "[3-02]_tCURLoRA/examples/finetune_example.py"
```

### 3. 查看现有实现

[1-04] 变分法已有完整实现，请查看：
- `../rof_from_scratch.py` - ROF去噪
- `../chan_vese_implementation.py` - Chan-Vese分割
- `../convex_relaxation_graph_cut.py` - 凸松弛与图割

---

## 📊 复现进度

### [3-01] 大模型高效微调
- ✅ LoRA核心层实现
- ✅ 训练框架搭建
- ✅ 快速开始示例
- 🟡 动态秩选择 (待优化)
- 🔴 多任务评估 (待完成)

### [1-04] 变分法基础
- ✅ ROF模型实现
- ✅ Chan-Vese分割
- ✅ 凸松弛方法
- ✅ 图割算法
- ✅ 完整文档总结

### [2-09] 框架分割管状结构
- ✅ 小波框架变换
- ✅ 形状先验建模
- 🟡 分割算法 (基础版完成)
- 🔴 3D扩展 (待开发)
- 🔴 真实数据测试 (待完成)

### [2-08] 小波框架血管分割
- ✅ 小波框架模块
- ✅ 分割网络架构
- ✅ DRIVE数据加载器
- ✅ 评估指标
- 🟡 训练流程
- 🔴 预训练权重 (待训练)

### [3-02] tCURLoRA张量分解
- ✅ 张量CUR分解
- ✅ tCURLoRA层实现
- ✅ 基础微调示例
- 🟡 大规模模型测试
- 🔴 性能对比实验 (待完成)

---

## 📝 代码规范

每个代码文件都包含：
1. **模块文档字符串**: 说明功能和理论基础
2. **详细中文注释**: 关键步骤解释
3. **清晰的API接口**: 易于使用和扩展
4. **示例用法**: `if __name__ == "__main__"` 中的测试代码

---

## 📖 学习建议

### 初学者路径
1. 从 [1-04] 开始，理解变分法基础
2. 运行ROF和Chan-Vese示例
3. 学习 [3-01] 的LoRA实现

### 进阶路径
1. 深入理解 [2-09] 小波框架理论
2. 实现管状结构分割
3. 尝试 [2-08] 血管分割

### 研究路径
1. 对比 [3-01] LoRA 和 [3-02] tCURLoRA
2. 分析参数效率和性能
3. 探索改进方向

---

## 🔗 相关资源

- **论文集合**: `../xiaohao_cai_papers/`
- **学习笔记**: `../第一课_变分法直观理解.md` 等
- **精读指南**: `../PDF精读指南_你的论文集合.md`

---

## 📞 问题反馈

如有问题或建议，请参考各论文目录下的README.md获取详细信息。

---

**创建时间**: 2024年  
**作者**: Xiaohao Cai  
**用途**: 学术研究和学习
