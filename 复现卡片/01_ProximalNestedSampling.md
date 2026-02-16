# 复现卡片: Proximal Nested Sampling

> arXiv: 2106.03646 | 高维贝叶斯模型选择 | 复现难度: ★★☆☆☆

---

## 基本信息

| 项目 | 信息 |
|:---|:---|
| **标题** | Proximal nested sampling for high-dimensional Bayesian model selection |
| **作者** | Xiaohao Cai, Jason D. McEwen, Marcelo Pereyra |
| **年份** | 2021 (v3: 2022) |
| **领域** | 贝叶斯统计、计算成像 |
| **期刊** | Statistics and Computing |

---

## 代码可用性

| 检查项 | 状态 | 详情 |
|:---|:---:|:---|
| **开源代码** | ✅ | 完整Python实现 |
| **代码仓库** | ✅ | https://github.com/astro-informatics/proxnest |
| **许可证** | GPL-3.0 | 开源协议 |
| **维护状态** | 活跃 | 持续更新 |

### 编程语言与框架

```
主要语言: Python 3.8+
依赖库:
├── numpy >= 1.20
├── scipy >= 1.6
├── matplotlib >= 3.3
└── tqdm (进度条)
```

---

## 数据集可用性

| 检查项 | 状态 | 详情 |
|:---|:---:|:---|
| **数据类型** | 仿真 | 代码生成测试数据 |
| **获取难度** | 简单 | 无需外部下载 |
| **预处理** | ✅ | 包含在代码中 |

### 数据集说明

- **类型**: 仿真数据（高维高斯分布、混合模型）
- **规模**: 可调节维度 (d = 2 to 10^6)
- **获取**: 运行示例脚本自动生成

---

## 实验复现步骤

### 环境配置

```bash
# 方法1: pip安装
pip install proxnest

# 方法2: 从源码安装
git clone https://github.com/astro-informatics/proxnest.git
cd proxnest
pip install -e .
```

### 运行示例

```bash
# 基本示例
python examples/gaussian_example.py

# 高维测试
python examples/highdim_example.py --dim 100000

# 复现论文图表
python scripts/reproduce_figures.py
```

### 核心代码示例

```python
import numpy as np
from proxnest import ProximalNestedSampling

# 定义似然和先验
def log_likelihood(x):
    return -0.5 * np.sum(x**2)

def prior_sample(n):
    return np.random.randn(n, dim)

# 运行嵌套采样
pns = ProximalNestedSampling(
    log_likelihood=log_likelihood,
    prior_sample=prior_sample,
    n_dim=100,
    n_live=500
)

# 计算边缘似然
log_Z = pns.run()
print(f"Log marginal likelihood: {log_Z:.4f}")
```

---

## 超参数设置

| 参数 | 论文值 | 建议范围 | 说明 |
|:---|:---:|:---:|:---|
| `n_live` | 500 | 100-1000 | 活跃样本数 |
| `n_dim` | 10-10^6 | - | 问题维度 |
| `tolerance` | 0.1 | 0.01-1.0 | 收敛阈值 |
| `max_iter` | 10000 | - | 最大迭代次数 |

---

## 结果验证

### 论文报告指标

| 实验 | 报告值 | 单位 |
|:---|:---:|:---|
| 10维高斯 log(Z) | -11.41 | nats |
| 100维高斯 log(Z) | -115.1 | nats |
| 收敛迭代数 | ~5000 | iterations |

### 验证标准

```python
# 验证log(Z)误差
def validate_result(computed_Z, true_Z):
    error = abs(computed_Z - true_Z) / abs(true_Z)
    assert error < 0.05, f"Error {error:.2%} exceeds 5%"
    print(f"Validation passed! Error: {error:.2%}")
```

### 预期差异

- **数值精度**: < 1% (浮点误差)
- **随机性**: < 3% (蒙特卡洛变异)
- **总体误差**: < 5%

---

## 常见问题

### Q1: 安装失败

```bash
# 解决依赖冲突
pip install --upgrade pip setuptools wheel
pip install proxnest --no-deps
pip install numpy scipy matplotlib tqdm
```

### Q2: 高维问题内存不足

```python
# 使用分块处理
pns = ProximalNestedSampling(
    ...,
    chunk_size=10000  # 分块大小
)
```

### Q3: 收敛慢

- 增加 `n_live` 参数
- 检查似然函数是否正确
- 尝试不同的近端算子

---

## 复现时间估计

| 任务 | 时间 |
|:---|:---|
| 环境配置 | 10分钟 |
| 运行示例 | 5分钟 |
| 复现论文图 | 30分钟 |
| 自定义实验 | 1-2小时 |

---

## 联系方式

- **GitHub Issues**: https://github.com/astro-informatics/proxnest/issues
- **论文作者**: x.cai@ucl.ac.uk

---

*最后更新: 2026-02-16*
