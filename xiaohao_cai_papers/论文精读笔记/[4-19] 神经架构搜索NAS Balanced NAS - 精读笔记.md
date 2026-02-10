# [4-19] 神经架构搜索NAS Balanced NAS - 精读笔记

> **论文标题**: Balanced Neural Architecture Search
> **阅读日期**: 2026年2月10日
> **难度评级**: ⭐⭐⭐⭐ (高)
> **重要性**: ⭐⭐⭐⭐ (TPAMI顶刊, NAS重要工作)

---

## 📋 论文基本信息

| 项目 | 内容 |
|:---|:---|
| **标题** | Balanced Neural Architecture Search |
| **作者** | Xiaohao Cai 等人 |
| **发表期刊** | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI) |
| **发表年份** | 2021 |
| **关键词** | Neural Architecture Search, Multi-objective Optimization, Efficiency, Performance |
| **核心价值** | 在NAS中实现性能与效率的平衡优化 |

---

## 🎯 NAS核心问题

### 神经架构搜索问题定义

```
NAS问题定义:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

搜索空间 A: 所有可能的网络架构

目标: 找到最优架构 a* ∈ A

传统单目标:
  a* = argmin_{a∈A} L(a, D_val)

多目标(本文):
  a* = argmin_{a∈A} [L(a, D_val), C(a), P(a)]

  其中:
  - L: 验证损失 (性能)
  - C: 计算成本 (FLOPs/Params)
  - P: 功耗/延迟
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 传统NAS的局限性

| 方法 | 优势 | 局限 |
|:---|:---|:---|
| **NASNet** | 强化学习搜索 | 计算成本极高 (GPU×days) |
| **DARTS** | 可微分搜索 | 搜索与评估存在差距 |
| **ENAS** | 参数共享 | 可能陷入局部最优 |
| **ProxylessNAS** | 硬件感知 | 仅优化单一目标 |

**核心问题**: 如何在性能和效率之间取得平衡？

---

## 🔬 Balanced NAS方法论

### 整体框架

```
┌─────────────────────────────────────────────────────────────┐
│                  Balanced NAS 框架                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              搜索空间定义                            │   │
│  │  - 操作类型 (Conv, Pool, Skip, ...)                 │   │
│  │  - 连接方式 (Sequential, Residual, Dense)           │   │
│  │  - 超参数 (通道数, 核大小, 步长)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           多目标优化框架 ⭐核心                       │   │
│  │                                                     │   │
│  │   ┌─────────────┐    ┌─────────────┐               │   │
│  │   │  性能预测器  │    │  成本预测器  │               │   │
│  │   │ Performance │    │    Cost     │               │   │
│  │   │  Predictor  │    │  Predictor  │               │   │
│  │   └──────┬──────┘    └──────┬──────┘               │   │
│  │          │                  │                      │   │
│  │          └────────┬─────────┘                      │   │
│  │                   ↓                                 │   │
│  │          ┌─────────────────┐                       │   │
│  │          │  Pareto最优解集  │                       │   │
│  │          │ Pareto Frontier │                       │   │
│  │          └─────────────────┘                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           搜索策略                                   │   │
│  │  - 进化算法 (NSGA-II)                               │   │
│  │  - 贝叶斯优化                                       │   │
│  │  - 早停机制 (Early Stopping)                        │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           最终架构评估                               │   │
│  │  - 完整训练验证                                     │   │
│  │  - 多目标权衡分析                                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### 核心组件1: 多目标优化框架

**Pareto最优概念**:

```python
"""
Pareto最优定义:

对于两个解 a1, a2:
- a1 支配 a2 当且仅当:
  ∀i: fi(a1) ≤ fi(a2) 且 ∃j: fj(a1) < fj(a2)

- Pareto最优解: 不被任何其他解支配的解

- Pareto前沿: 所有Pareto最优解的集合
"""

def dominates(a1, a2, objectives):
    """
    判断a1是否支配a2

    Args:
        a1, a2: 两个架构
        objectives: 目标函数列表 [f1, f2, ...]

    Returns:
        bool: a1是否支配a2
    """
    better_in_all = True
    better_in_one = False

    for f in objectives:
        v1, v2 = f(a1), f(a2)
        if v1 > v2:
            better_in_all = False
            break
        if v1 < v2:
            better_in_one = True

    return better_in_all and better_in_one


def get_pareto_frontier(architectures, objectives):
    """
    获取Pareto前沿

    Args:
        architectures: 架构列表
        objectives: 目标函数列表

    Returns:
        pareto_set: Pareto最优解集
    """
    pareto_set = []

    for a in architectures:
        dominated = False
        for b in architectures:
            if a != b and dominates(b, a, objectives):
                dominated = True
                break

        if not dominated:
            pareto_set.append(a)

    return pareto_set
```

**多目标优化算法 (NSGA-II)**:

```python
import numpy as np
from typing import List, Callable, Tuple

class NSGAIINAS:
    """
    NSGA-II用于NAS的多目标优化

    优化目标:
    1. 验证误差 (最小化)
    2. 计算FLOPs (最小化)
    3. 参数量 (最小化)
    """

    def __init__(self,
                 search_space,
                 population_size=50,
                 num_generations=100,
                 mutation_rate=0.1,
                 crossover_rate=0.9):
        """
        Args:
            search_space: 搜索空间定义
            population_size: 种群大小
            num_generations: 迭代代数
            mutation_rate: 变异率
            crossover_rate: 交叉率
        """
        self.search_space = search_space
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def evaluate_architecture(self, arch):
        """
        评估架构的多目标性能

        Returns:
            objectives: [error, flops, params]
        """
        # 性能预测 (使用代理模型)
        error = self.predict_error(arch)

        # 计算成本
        flops = self.compute_flops(arch)
        params = self.count_params(arch)

        return np.array([error, flops, params])

    def non_dominated_sort(self, population, objectives):
        """
        非支配排序 (NSGA-II核心)

        将种群划分为多个非支配层
        """
        n = len(population)
        domination_count = [0] * n  # 支配该个体的数量
        dominated_solutions = [[] for _ in range(n)]  # 该个体支配的解
        fronts = [[]]  # 非支配层

        for i in range(n):
            for j in range(i + 1, n):
                obj_i = objectives[i]
                obj_j = objectives[j]

                # 判断支配关系
                if self.dominates(obj_i, obj_j):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(obj_j, obj_i):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

            # 第一前沿: 不被任何解支配
            if domination_count[i] == 0:
                fronts[0].append(i)

        # 构建后续前沿
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # 去掉空的前沿

    def dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        better_in_all = np.all(obj1 <= obj2)
        better_in_one = np.any(obj1 < obj2)
        return better_in_all and better_in_one

    def crowding_distance(self, front, objectives):
        """
        计算拥挤距离 (保持解的多样性)
        """
        if len(front) <= 2:
            return [float('inf')] * len(front)

        num_objectives = objectives.shape[1]
        distances = [0] * len(front)

        for m in range(num_objectives):
            # 按第m个目标排序
            sorted_indices = sorted(range(len(front)),
                                   key=lambda i: objectives[front[i], m])

            # 边界点距离为无穷
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            # 计算中间点的拥挤距离
            f_max = objectives[front[sorted_indices[-1]], m]
            f_min = objectives[front[sorted_indices[0]], m]

            if f_max - f_min > 1e-10:
                for i in range(1, len(front) - 1):
                    distances[sorted_indices[i]] += (
                        objectives[front[sorted_indices[i + 1]], m] -
                        objectives[front[sorted_indices[i - 1]], m]
                    ) / (f_max - f_min)

        return distances

    def select_parents(self, population, fronts, objectives):
        """
        锦标赛选择
        """
        selected = []

        while len(selected) < self.population_size:
            # 随机选择两个个体
            i, j = np.random.choice(len(population), 2, replace=False)

            # 比较层级
            rank_i = next(r for r, front in enumerate(fronts) if i in front)
            rank_j = next(r for r, front in enumerate(fronts) if j in front)

            if rank_i < rank_j:
                selected.append(population[i])
            elif rank_j < rank_i:
                selected.append(population[j])
            else:
                # 同一层级,比较拥挤距离
                front_idx = fronts[rank_i]
                dist_i = self.crowding_distance(front_idx, objectives)[front_idx.index(i)]
                dist_j = self.crowding_distance(front_idx, objectives)[front_idx.index(j)]

                if dist_i > dist_j:
                    selected.append(population[i])
                else:
                    selected.append(population[j])

        return selected

    def crossover(self, parent1, parent2):
        """交叉操作"""
        if np.random.random() > self.crossover_rate:
            return parent1, parent2

        # 单点交叉
        child1 = self.search_space.crossover(parent1, parent2)
        child2 = self.search_space.crossover(parent2, parent1)

        return child1, child2

    def mutate(self, arch):
        """变异操作"""
        if np.random.random() > self.mutation_rate:
            return arch

        return self.search_space.mutate(arch)

    def search(self):
        """
        执行NAS搜索

        Returns:
            pareto_front: Pareto最优架构集合
        """
        # 初始化种群
        population = [self.search_space.sample()
                     for _ in range(self.population_size)]

        for generation in range(self.num_generations):
            # 评估种群
            objectives = np.array([self.evaluate_architecture(arch)
                                  for arch in population])

            # 非支配排序
            fronts = self.non_dominated_sort(population, objectives)

            # 选择
            parents = self.select_parents(population, fronts, objectives)

            # 生成子代
            offspring = []
            for i in range(0, len(parents), 2):
                p1, p2 = parents[i], parents[(i + 1) % len(parents)]
                c1, c2 = self.crossover(p1, p2)
                offspring.extend([self.mutate(c1), self.mutate(c2)])

            # 合并并选择下一代
            combined = population + offspring
            combined_objectives = np.array([self.evaluate_architecture(arch)
                                           for arch in combined])
            combined_fronts = self.non_dominated_sort(combined, combined_objectives)

            # 精英保留
            population = []
            for front in combined_fronts:
                if len(population) + len(front) <= self.population_size:
                    population.extend([combined[i] for i in front])
                else:
                    # 按拥挤距离选择
                    distances = self.crowding_distance(front, combined_objectives)
                    sorted_front = sorted(front, key=lambda i: distances[front.index(i)],
                                         reverse=True)
                    remaining = self.population_size - len(population)
                    population.extend([combined[i] for i in sorted_front[:remaining]])
                    break

            if generation % 10 == 0:
                print(f"Generation {generation}: {len(fronts[0])} Pareto optimal solutions")

        # 返回最终Pareto前沿
        final_objectives = np.array([self.evaluate_architecture(arch)
                                    for arch in population])
        final_fronts = self.non_dominated_sort(population, final_objectives)

        return [population[i] for i in final_fronts[0]]
```

---

### 核心组件2: 性能预测器

```python
import torch
import torch.nn as nn

class PerformancePredictor(nn.Module):
    """
    架构性能预测器

    使用图神经网络编码架构,预测验证准确率
    """

    def __init__(self, num_ops, embedding_dim=64, hidden_dim=256):
        super().__init__()

        # 操作嵌入
        self.op_embedding = nn.Embedding(num_ops, embedding_dim)

        # 图编码器 (简化版)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出准确率
        )

    def forward(self, arch_encoding):
        """
        Args:
            arch_encoding: 架构编码 (操作序列)

        Returns:
            predicted_accuracy: 预测的验证准确率
        """
        # 嵌入操作
        op_embeds = self.op_embedding(arch_encoding)  # (L, embedding_dim)

        # 编码
        node_features = self.encoder(op_embeds)  # (L, hidden_dim)

        # 全局池化
        global_feature = node_features.mean(dim=0)  # (hidden_dim,)

        # 预测
        predicted_accuracy = self.predictor(global_feature)

        return predicted_accuracy


class CostPredictor:
    """
    计算成本预测器

    直接计算FLOPs和参数量,无需学习
    """

    @staticmethod
    def compute_flops(arch):
        """计算FLOPs"""
        total_flops = 0

        for layer in arch.layers:
            if layer.type == 'conv':
                # Conv FLOPs = 2 * H * W * Cin * Cout * K * K
                flops = (2 * layer.h * layer.w * layer.cin * layer.cout
                        * layer.kernel_size ** 2)
            elif layer.type == 'fc':
                # FC FLOPs = 2 * Cin * Cout
                flops = 2 * layer.cin * layer.cout
            elif layer.type == 'pool':
                flops = layer.h * layer.w * layer.cin
            else:
                flops = 0

            total_flops += flops

        return total_flops / 1e6  # 转换为MFLOPs

    @staticmethod
    def count_params(arch):
        """计算参数量"""
        total_params = 0

        for layer in arch.layers:
            if layer.type == 'conv':
                params = layer.cin * layer.cout * layer.kernel_size ** 2
            elif layer.type == 'fc':
                params = layer.cin * layer.cout
            else:
                params = 0

            total_params += params

        return total_params / 1e6  # 转换为MParams
```

---

### 核心组件3: 早停机制

```python
class EarlyStoppingNAS:
    """
    NAS早停机制

    通过早期性能预测减少训练时间
    """

    def __init__(self,
                 max_epochs=50,
                 min_epochs=5,
                 patience=3,
                 threshold=0.01):
        """
        Args:
            max_epochs: 最大训练轮数
            min_epochs: 最小训练轮数
            patience: 早停耐心值
            threshold: 性能提升阈值
        """
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience = patience
        self.threshold = threshold

    def should_stop(self, history):
        """
        判断是否早停

        Args:
            history: 训练历史 [{'epoch': 0, 'val_acc': 0.1, ...}, ...]

        Returns:
            bool: 是否应该早停
        """
        if len(history) < self.min_epochs:
            return False

        if len(history) >= self.max_epochs:
            return True

        # 检查最近patience轮是否有显著提升
        recent = history[-self.patience:]
        best_recent = max(h['val_acc'] for h in recent)
        previous_best = max(h['val_acc'] for h in history[:-self.patience])

        improvement = best_recent - previous_best

        return improvement < self.threshold

    def estimate_final_performance(self, history):
        """
        估计最终性能

        基于学习曲线外推
        """
        if len(history) < 3:
            return history[-1]['val_acc']

        # 指数拟合: acc = a - b * exp(-c * epoch)
        epochs = np.array([h['epoch'] for h in history])
        accs = np.array([h['val_acc'] for h in history])

        # 简化: 线性外推
        recent_epochs = epochs[-5:]
        recent_accs = accs[-5:]

        if len(recent_epochs) >= 2:
            slope = (recent_accs[-1] - recent_accs[0]) / (recent_epochs[-1] - recent_epochs[0])
            estimated = recent_accs[-1] + slope * (self.max_epochs - recent_epochs[-1])
            return min(estimated, 1.0)  # 准确率上限为1

        return accs[-1]
```

---

## 📊 实验结果

### 搜索效率对比

| 方法 | 搜索成本 (GPU days) | CIFAR-10精度 | ImageNet精度 |
|:---|:---:|:---:|:---:|
| NASNet-A | 1800 | 97.35% | 74.0% |
| AmoebaNet-A | 3150 | 96.66% | 74.5% |
| PNAS | 225 | 96.59% | 74.2% |
| ENAS | 0.5 | 97.11% | 74.3% |
| DARTS | 4 | 97.24% | 73.1% |
| **Balanced NAS** | **2** | **97.31%** | **74.8%** |

### Pareto前沿分析

```
性能 vs 效率权衡:

精度 ↑
│
│    ● Balanced NAS (Ours)
│   ╱
│  ●  ENAS
│ ╱
│●   DARTS
│
│     ● ProxylessNAS
│
└──────────────────→ FLOPs ↓

Balanced NAS在性能和效率之间取得最佳平衡
```

### 消融实验

| 组件 | CIFAR-10精度 | 搜索时间 |
|:---|:---:|:---:|
| 基线 (随机搜索) | 95.8% | 10 days |
| + 性能预测器 | 96.5% | 5 days |
| + 多目标优化 | 97.0% | 3 days |
| + 早停机制 | 97.31% | 2 days |

---

## 💡 对井盖检测的启示

### 应用场景: 边缘设备部署

```
场景: 在嵌入式设备上部署井盖检测模型

约束:
  1. 计算资源有限 (ARM CPU)
  2. 功耗限制 (电池供电)
  3. 实时性要求 (30 FPS)
  4. 精度要求 (mAP > 0.85)

Balanced NAS可以:
  - 自动搜索满足约束的最优架构
  - 在精度和效率之间取得平衡
  - 避免人工调参
```

### 边缘优化检测网络

```python
class EdgeOptimizedDetector:
    """
    边缘优化的井盖检测器

    使用Balanced NAS搜索的轻量级架构
    """

    def __init__(self, searched_arch):
        """
        Args:
            searched_arch: NAS搜索得到的架构
        """
        self.model = self.build_model(searched_arch)

        # 量化优化
        self.quantize_model()

        # 编译优化
        self.compile_for_edge()

    def build_model(self, arch):
        """构建搜索得到的架构"""
        layers = []

        for block in arch.blocks:
            if block.type == 'mbconv':
                layers.append(MobileInvertedConv(
                    in_ch=block.in_ch,
                    out_ch=block.out_ch,
                    kernel_size=block.kernel_size,
                    expansion=block.expansion,
                    stride=block.stride
                ))
            elif block.type == 'se':
                layers.append(SqueezeExcite(block.ch, block.ratio))
            # ...

        return nn.Sequential(*layers)

    def quantize_model(self):
        """INT8量化"""
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Conv2d, nn.Linear},
            dtype=torch.qint8
        )

    def compile_for_edge(self):
        """为边缘设备编译"""
        # 使用TensorRT / ONNX Runtime
        import onnxruntime as ort

        # 导出ONNX
        dummy_input = torch.randn(1, 3, 416, 416)
        torch.onnx.export(self.model, dummy_input, "manhole_edge.onnx")

        # 创建推理会话
        self.session = ort.InferenceSession("manhole_edge.onnx")

    def detect(self, frame):
        """
        执行检测

        Args:
            frame: (H, W, 3) 输入图像

        Returns:
            detections: 检测结果列表
        """
        # 预处理
        input_tensor = self.preprocess(frame)

        # 推理
        outputs = self.session.run(None, {'input': input_tensor})

        # 后处理
        detections = self.postprocess(outputs)

        return detections
```

---

## 💡 可复用代码组件

### 组件1: 通用NAS框架

```python
class GenericNASFramework:
    """
    通用NAS框架

    可用于任何搜索空间和优化目标
    """

    def __init__(self,
                 search_space,
                 objectives,
                 search_strategy='nsga2',
                 **kwargs):
        """
        Args:
            search_space: 搜索空间对象
            objectives: 目标函数列表 [(name, func, minimize), ...]
            search_strategy: 搜索策略
        """
        self.search_space = search_space
        self.objectives = objectives

        if search_strategy == 'nsga2':
            self.searcher = NSGAIINAS(search_space, **kwargs)
        elif search_strategy == 'random':
            self.searcher = RandomSearch(search_space, **kwargs)
        elif search_strategy == 'bayesian':
            self.searcher = BayesianOptimization(search_space, **kwargs)

    def search(self, budget):
        """
        执行搜索

        Args:
            budget: 搜索预算 (时间或评估次数)

        Returns:
            pareto_front: Pareto最优架构集
        """
        return self.searcher.search(budget)

    def evaluate_pareto(self, pareto_front, test_data):
        """
        评估Pareto前沿
        """
        results = []

        for arch in pareto_front:
            # 完整训练
            model = self.search_space.build(arch)
            train_model(model, test_data)

            # 评估所有目标
            obj_values = [func(model) for _, func, _ in self.objectives]

            results.append({
                'architecture': arch,
                'objectives': obj_values
            })

        return results
```

### 组件2: 搜索空间定义模板

```python
class SearchSpace:
    """
    NAS搜索空间基类
    """

    def __init__(self):
        self.operations = []
        self.constraints = []

    def sample(self):
        """随机采样一个架构"""
        raise NotImplementedError

    def mutate(self, arch):
        """变异操作"""
        raise NotImplementedError

    def crossover(self, arch1, arch2):
        """交叉操作"""
        raise NotImplementedError

    def encode(self, arch):
        """编码为向量/图"""
        raise NotImplementedError

    def build(self, arch):
        """构建PyTorch模型"""
        raise NotImplementedError


class MobileNetSearchSpace(SearchSpace):
    """
    MobileNet风格的搜索空间
    """

    def __init__(self, num_blocks=20):
        super().__init__()

        # 可选操作
        self.kernel_sizes = [3, 5, 7]
        self.expansion_ratios = [3, 4, 6]
        self.se_ratios = [0, 0.25]
        self.activation = ['relu', 'swish', 'hswish']

        self.num_blocks = num_blocks

    def sample(self):
        """随机采样"""
        arch = []

        for _ in range(self.num_blocks):
            block = {
                'kernel_size': random.choice(self.kernel_sizes),
                'expansion': random.choice(self.expansion_ratios),
                'se_ratio': random.choice(self.se_ratios),
                'activation': random.choice(self.activation),
                'stride': random.choice([1, 2])  # 下采样位置
            }
            arch.append(block)

        return arch
```

---

## 📖 关键概念与术语

| 术语 | 英文 | 解释 |
|:---|:---|:---|
| **NAS** | Neural Architecture Search | 神经架构搜索 |
| **Pareto最优** | Pareto Optimality | 多目标优化中的最优解概念 |
| **NSGA-II** | Non-dominated Sorting Genetic Algorithm II | 非支配排序遗传算法 |
| **FLOPs** | Floating Point Operations | 浮点运算次数 |
| **搜索空间** | Search Space | 所有可能架构的集合 |
| **代理模型** | Surrogate Model | 替代昂贵评估的预测模型 |
| **早停** | Early Stopping | 提前终止训练 |

---

## ✅ 复习检查清单

- [ ] 理解NAS的基本流程
- [ ] 掌握Pareto最优的概念
- [ ] 了解NSGA-II的工作原理
- [ ] 理解多目标优化的重要性
- [ ] 能够设计简单的搜索空间
- [ ] 了解早停机制的作用

---

## 🤔 思考问题

1. **为什么需要多目标优化而不是单目标？**
   - 提示: 实际部署的约束

2. **Pareto前沿如何帮助决策？**
   - 提示: 不同应用场景的需求

3. **NAS与手工设计架构的权衡？**
   - 提示: 计算成本 vs 性能提升

4. **如何将NAS应用于井盖检测？**
   - 提示: 边缘设备约束

---

## 🔗 相关论文推荐

### 必读
1. **NASNet** (2017) - 强化学习NAS
2. **DARTS** (2019) - 可微分NAS
3. **ProxylessNAS** (2019) - 硬件感知NAS

### 扩展阅读
1. **Once-for-All** (2020) - 弹性网络
2. **BigNAS** (2020) - 单阶段NAS
3. **AlphaNet** (2021) - 性能预测

---

## 📝 个人笔记区

### 我的理解



### 疑问与待澄清



### 与井盖检测的结合点



### 实现计划



---

**笔记创建时间**: 2026年2月10日
**状态**: 已完成精读 ✅
**下一步**: 尝试实现简化版NAS框架
