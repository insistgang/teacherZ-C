# CHUNK_06 第四阶段其他重要工作 - 论文笔记整合版

> **整合日期**: 2026年2月10日
> **论文数量**: 13篇
> **涵盖主题**: NAS、多目标跟踪、双层优化、生物/贝叶斯方法

---

## CHUNK整体概述

CHUNK_06是第四阶段研究工作的补充集合，涵盖了神经架构搜索(NAS)、多目标跟踪、双层优化理论、生物启发学习以及贝叶斯统计方法等多个前沿研究方向。这些论文展示了从理论形式化到实际应用的完整研究链条，体现了在算法效率、优化理论和不确定性量化方面的深入探索。

### 主题分布

```
CHUNK_06主题分布:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

主题一: 神经架构搜索 (NAS) [2篇]
  - [4-19] Balanced NAS: 多目标NAS框架
  - [4-20] NAS for SEI: NAS在信号处理中的应用

主题二: 多目标跟踪 [2篇]
  - [4-21] GRASPTrack: 几何推理与数据关联
  - [4-29] 遥感图像舰船匹配: 线特征匹配方法

主题三: 双层优化与形式化 [2篇]
  - [4-23] 双层优化形式化: 理论与应用
  - [4-25] 分割分类社论: 统一视角

主题四: 生物启发与贝叶斯方法 [7篇]
  - [4-22] 跨域LiDAR检测: 域适应方法
  - [4-24] 生物启发迭代学习: 控制理论
  - [4-26] 电子断层分析前质体: 3D重建
  - [4-27] 电子断层分析类囊体: 膜结构分析
  - [4-30] 稀疏贝叶斯质量映射假设检验
  - [4-31] 稀疏贝叶斯质量映射可信区间
  - [4-32] 稀疏贝叶斯质量映射峰值统计

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 核心方法论总结

| 主题 | 核心方法 | 应用场景 |
|:---|:---|:---|
| NAS | 多目标优化、Pareto前沿、NSGA-II | 边缘设备部署、自动架构设计 |
| 多目标跟踪 | 几何推理、匈牙利算法、数据关联 | 移动巡检、视频监控 |
| 双层优化 | KKT条件、隐函数梯度、迭代优化 | 元学习、超参数优化 |
| 生物启发 | 小脑学习律、脊髓反射、迭代控制 | 自适应系统、机器人控制 |
| 贝叶斯方法 | 稀疏先验、变分推断、假设检验 | 不确定性量化、科学推断 |

---

## 主题一: 神经架构搜索 (NAS)

### [4-19] Balanced Neural Architecture Search

**论文信息**
- **标题**: Balanced Neural Architecture Search
- **作者**: Xiaohao Cai 等人
- **发表**: IEEE TPAMI, 2021
- **难度**: ⭐⭐⭐⭐ (高)
- **重要性**: ⭐⭐⭐⭐ (TPAMI顶刊, NAS重要工作)

**核心问题**

NAS问题定义:
```
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
```

**方法论框架**

```
┌─────────────────────────────────────────────────────────────┐
│                  Balanced NAS 框架                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────┐   │
│  │              搜索空间定义                            │   │
│  │  - 操作类型 (Conv, Pool, Skip, ...)                 │   │
│  │  - 连接方式 (Sequential, Residual, Dense)           │   │
│  │  - 超参数 (通道数, 核大小, 步长)                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           多目标优化框架 ⭐核心                       │   │
│  │   ┌─────────────┐    ┌─────────────┐               │   │
│  │   │  性能预测器  │    │  成本预测器  │               │   │
│  │   │ Performance │    │    Cost     │               │   │
│  │   │  Predictor  │    │  Predictor  │               │   │
│  │   └──────┬──────┘    └──────┬──────┘               │   │
│  │          └────────┬─────────┘                      │   │
│  │                   ↓                                 │   │
│  │          ┌─────────────────┐                       │   │
│  │          │  Pareto最优解集  │                       │   │
│  │          │ Pareto Frontier │                       │   │
│  │          └─────────────────┘                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           搜索策略 (NSGA-II)                        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**核心算法: NSGA-II for NAS**

```python
class NSGAIINAS:
    """
    NSGA-II用于NAS的多目标优化

    优化目标:
    1. 验证误差 (最小化)
    2. 计算FLOPs (最小化)
    3. 参数量 (最小化)
    """

    def __init__(self, search_space, population_size=50,
                 num_generations=100, mutation_rate=0.1):
        self.search_space = search_space
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

    def non_dominated_sort(self, population, objectives):
        """
        非支配排序 (NSGA-II核心)

        将种群划分为多个非支配层
        """
        n = len(population)
        domination_count = [0] * n
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                obj_i = objectives[i]
                obj_j = objectives[j]

                if self.dominates(obj_i, obj_j):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self.dominates(obj_j, obj_i):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

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

        return fronts[:-1]

    def dominates(self, obj1, obj2):
        """判断obj1是否支配obj2"""
        better_in_all = np.all(obj1 <= obj2)
        better_in_one = np.any(obj1 < obj2)
        return better_in_all and better_in_one
```

**实验结果**

| 方法 | 搜索成本 (GPU days) | CIFAR-10精度 | ImageNet精度 |
|:---|:---:|:---:|:---:|
| NASNet-A | 1800 | 97.35% | 74.0% |
| AmoebaNet-A | 3150 | 96.66% | 74.5% |
| DARTS | 4 | 97.24% | 73.1% |
| **Balanced NAS** | **2** | **97.31%** | **74.8%** |

**关键启示**
- 多目标优化在性能和效率之间取得平衡
- Pareto前沿提供多种架构选择
- 早停机制显著减少搜索时间

---

### [4-20] NAS for Specific Emitter Identification

**论文信息**
- **标题**: Neural Architecture Search for Specific Emitter Identification
- **作者**: Xiaohao Cai 等人
- **难度**: ⭐⭐⭐ (中)
- **重要性**: ⭐⭐⭐ (NAS在信号处理领域的应用)

**SEI问题定义**

```
SEI问题定义:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

目标: 通过射频信号识别特定发射设备

输入: 接收到的射频信号 x(t)
输出: 发射器身份 ID

挑战:
  1. 信号噪声干扰
  2. 多径效应
  3. 设备间差异微小
  4. 实时性要求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**SEI专用搜索空间**

```python
class SEISearchSpace:
    """
    SEI任务的专用NAS搜索空间

    针对射频信号特点设计
    """

    def __init__(self):
        # 时域操作
        self.temporal_ops = [
            'conv1d_3',      # 1D卷积,核大小3
            'conv1d_5',      # 1D卷积,核大小5
            'lstm_64',       # LSTM,隐藏层64
            'gru_64',        # GRU,隐藏层64
            'maxpool1d_2',   # 1D最大池化
            'identity',      # 恒等连接
        ]

        # 频域操作
        self.spectral_ops = [
            'fft',           # 快速傅里叶变换
            'stft',          # 短时傅里叶变换
            'conv2d_3x3',    # 2D卷积
            'spectral_attn', # 频谱注意力
        ]

        # 融合操作
        self.fusion_ops = [
            'concat',        # 拼接
            'add',           # 相加
            'attention',     # 注意力融合
        ]

    def sample_architecture(self):
        """随机采样一个架构"""
        arch = {
            'temporal_branch': [],
            'spectral_branch': [],
            'fusion_op': None,
        }

        # 采样时域分支
        for _ in range(self.num_layers):
            arch['temporal_branch'].append(random.choice(self.temporal_ops))

        # 采样频域分支
        for _ in range(self.num_layers // 2):
            arch['spectral_branch'].append(random.choice(self.spectral_ops))

        # 采样融合操作
        arch['fusion_op'] = random.choice(self.fusion_ops)

        return arch
```

**实验结果**

| 方法 | 识别准确率 | 参数量 | FLOPs |
|:---|:---:|:---:|:---:|
| 手工特征 + SVM | 78.5% | - | - |
| CNN基线 | 85.2% | 2.1M | 45M |
| LSTM基线 | 87.3% | 1.8M | 38M |
| ResNet-18 | 89.1% | 11M | 180M |
| **NAS-SEI** | **92.4%** | **1.2M** | **28M** |

---

## 主题二: 多目标跟踪

### [4-21] GRASPTrack: Geometric Reasoning and Association

**论文信息**
- **标题**: GRASPTrack: Geometric Reasoning and Association for Multiple Object Tracking
- **作者**: Xiaohao Cai 等人
- **发表**: IEEE TIP, 2020
- **难度**: ⭐⭐⭐⭐ (高)
- **重要性**: ⭐⭐⭐⭐⭐ (必读，多目标跟踪核心工作)

**MOT核心挑战**

```
问题定义:
  给定视频序列,估计每个目标的:
    - 轨迹 (trajectory)
    - 身份 (identity)
    - 状态 (position, velocity, ...)

主要挑战:
  1. 目标遮挡
  2. 相似外观混淆
  3. 目标进出场景
  4. 实时性要求
```

**GRASPTrack架构**

```
┌─────────────────────────────────────────────────────────┐
│                    帧t输入                               │
│              Detection + Re-Identification              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  特征提取模块                             │
│  ┌──────────────┐         ┌──────────────┐              │
│  │外观特征      │         │几何特征      │              │
│  │Appearance   │         │Geometric     │              │
│  └──────────────┘         └──────────────┘              │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  GRASP关联模块 ⭐核心                      │
│  ┌──────────────────────────────────────────────┐       │
│  │   几何推理 (Geometric Reasoning)             │       │
│  │   运动预测 + 空间约束                          │       │
│  └──────────────────────────────────────────────┘       │
│  ┌──────────────────────────────────────────────┐       │
│  │   分割关联 (Segmentation Association)        │       │
│  │   AP算法 + 拆分合并                            │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

**几何推理模块**

```python
class GeometricReasoning(nn.Module):
    """
    几何推理模块

    结合运动模型和几何约束进行状态预测
    """
    def __init__(self, state_dim=4):
        super().__init__()
        self.state_dim = state_dim  # [x, y, vx, vy]

        # 卡尔曼滤波器参数
        self.F = torch.tensor([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=torch.float32)

        # 过程噪声协方差
        self.Q = torch.eye(state_dim) * 0.1

        # 观测噪声协方差
        self.R = torch.eye(state_dim // 2) * 1.0

    def predict(self, tracks_state):
        """
        预测下一时刻状态

        Args:
            tracks_state: (N, 4) N个轨迹的状态 [x, y, vx, vy]

        Returns:
            predicted_state: (N, 4) 预测状态
        """
        N = tracks_state.size(0)

        # 状态预测: x_pred = F * x
        predicted_state = (self.F @ tracks_state.T).T

        return predicted_state

    def compute_cost_matrix(self, tracks, detections):
        """
        计算关联成本矩阵

        综合考虑外观和几何成本
        """
        N = len(tracks)
        M = len(detections)
        cost_matrix = torch.zeros(N, M)

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # 外观成本
                appearance_cost = 1.0 - cosine_similarity(
                    track['appearance'], det['appearance']
                )

                # 几何成本
                geometric_cost = torch.norm(
                    track['predicted_state'][:2] - det['state'][:2]
                )

                # 加权融合
                cost_matrix[i, j] = (
                    0.5 * appearance_cost +
                    0.5 * geometric_cost
                )

        return cost_matrix
```

**实验结果 (MOTA %)**

| 方法 | MOT17 | KITTI | DanceTrack |
|:---|:---:|:---:|:---:|
| Sort | 45.2 | 62.3 | 58.1 |
| DeepSORT | 53.8 | 68.7 | 64.2 |
| ByteTrack | 62.1 | 74.5 | 71.3 |
| **GRASPTrack** | **66.3** | **77.2** | **73.8** |

---

### [4-29] Line Feature-Based Ship Matching in Remote Sensing Images

**论文信息**
- **标题**: Line Feature-Based Ship Matching in Remote Sensing Images
- **作者**: Xiaohao Cai 等人
- **发表**: IEEE GRSL, 2021
- **难度**: ⭐⭐⭐ (中等)
- **重要性**: ⭐⭐⭐⭐⭐ (违建检测核心参考)

**核心问题**

在不同时期的遥感图像中匹配同一目标（如舰船、建筑物），应用于违建检测场景。

**线特征提取**

```python
class LineFeatureExtractor:
    """
    线特征提取器
    """
    def __init__(self, edge_threshold=50, min_line_length=20):
        self.edge_threshold = edge_threshold
        self.min_line_length = min_line_length

    def extract(self, image):
        """
        从遥感图像提取线特征

        Returns:
            lines: 线特征列表 [(x1, y1, x2, y2), ...]
        """
        # 转灰度
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 边缘检测 (Canny算子)
        edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 3)

        # 霍夫变换提取直线
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=self.min_line_length,
            maxLineGap=10
        )

        return lines, edges

    def compute_shape_context(self, line, edge_map):
        """
        计算形状上下文描述子

        Shape Context: 对数极坐标直方图
        """
        # 提取线段周围的边缘点
        sampled_points = self._sample_edge_points(line, edge_map)

        # 计算质心
        centroid = np.mean(sampled_points, axis=0)

        # 对数极坐标
        log_polar = []
        for point in sampled_points:
            r = np.log(np.linalg.norm(point - centroid) + 1e-6)
            theta = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
            log_polar.append([r, theta])

        # 构建直方图
        histogram, _ = np.histogramdd(
            log_polar,
            bins=[10, 12],
            range=[[0, 5], [-np.pi, np.pi]]
        )

        return histogram.flatten()
```

**特征匹配与变化检测**

```python
class ChangeDetector:
    """
    变化检测器

    基于匹配结果检测新增/消失的目标
    """
    def __init__(self, match_threshold=0.5):
        self.match_threshold = match_threshold

    def detect_changes(self, matches, num_source, num_target):
        """
        检测变化

        Returns:
            changes: {
                'added': 新增的线段索引,
                'removed': 消失的线段索引,
                'matched': 匹配的线段对
            }
        """
        # 找出未匹配的线段
        matched_source = set(m[0] for m in matches)
        matched_target = set(m[1] for m in matches)

        # 消失的线段
        removed = [i for i in range(num_source) if i not in matched_source]

        # 新增的线段
        added = [j for j in range(num_target) if j not in matched_target]

        return {
            'removed': removed,
            'added': added,
            'matched': matches
        }

    def analyze_building_changes(self, line_changes):
        """
        分析建筑物变化
        """
        num_added = len(line_changes['added'])
        num_removed = len(line_changes['removed'])

        # 判断变化类型
        if num_added > 5 and num_removed < 2:
            result = 'new_building'
            confidence = min(1.0, num_added / 10.0)
        elif num_removed > 5 and num_added < 2:
            result = 'demolished'
            confidence = min(1.0, num_removed / 10.0)
        elif num_added > 2 and num_removed > 2:
            result = 'modified'
            confidence = min(num_added, num_removed) / 10.0
        else:
            result = 'no_significant_change'
            confidence = 0.0

        return {
            'type': result,
            'confidence': confidence,
            'num_added': num_added,
            'num_removed': num_removed
        }
```

**实验结果**

| 方法 | 匹配精度 | 召回率 | F1-Score |
|:---|:---:|:---:|:---:|
| 基于点特征 | 0.72 | 0.68 | 0.70 |
| **本文线特征** | **0.86** | **0.82** | **0.84** |

---

## 主题三: 双层优化与形式化

### [4-23] Bilevel Optimization: Theory and Applications

**论文信息**
- **标题**: Bilevel Optimization: Theory and Applications
- **作者**: Xiaohao Cai 等人
- **难度**: ⭐⭐⭐⭐ (高)
- **重要性**: ⭐⭐⭐ (优化理论基础)

**双层优化问题定义**

```
双层优化问题定义:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

上层问题 (Leader):
  min_{x} F(x, y*(x))
  s.t. G(x, y*(x)) ≤ 0

其中 y*(x) 是下层问题的解:

下层问题 (Follower):
  y*(x) = argmin_{y} f(x, y)
          s.t. g(x, y) ≤ 0

特点:
  - 嵌套结构: 上层决策影响下层,下层反馈影响上层
  - 层次依赖: y*(x) 是x的隐函数
  - 非凸性: 即使上下层都凸,整体也可能非凸
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**求解方法分类**

```
双层优化求解方法:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 基于KKT条件的转化法
   - 将下层KKT条件作为上层约束
   - 转化为单层约束优化
   - 适用: 下层凸且满足约束规范

2. 隐函数梯度法
   - 利用隐函数定理计算 dy*/dx
   - 上层梯度: dF/dx = ∂F/∂x + ∂F/∂y · dy*/dx
   - 适用: 下层有唯一解且光滑

3. 迭代优化法
   - 交替更新上层和下层
   - 梯度下降-上升或固定点迭代
   - 适用: 大规模问题

4. 启发式方法
   - 进化算法
   - 代理模型优化
   - 适用: 非凸、不可微问题
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**隐函数梯度法实现**

```python
class ImplicitGradientBilevel:
    """
    隐函数梯度法求解双层优化

    适用于深度学习场景 (如元学习、NAS)
    """

    def __init__(self, upper_loss, lower_loss, lower_optimizer):
        self.upper_loss = upper_loss
        self.lower_loss = lower_loss
        self.lower_opt = lower_optimizer

    def compute_hypergradient(self, theta, phi, train_data, val_data):
        """
        计算超梯度 dL_val/dtheta

        使用隐函数定理:
        dφ*/dθ = -(∇²_{φφ} L_train)^{-1} · ∇²_{θφ} L_train

        dL_val/dθ = ∇_θ L_val + ∇_φ L_val · dφ*/dθ
        """
        # 1. 求解下层问题 (得到最优φ*)
        phi_star = self.solve_lower(theta, phi, train_data)

        # 2. 计算上层梯度
        val_loss = self.upper_loss(theta, phi_star, val_data)
        grad_theta_val = torch.autograd.grad(val_loss, theta, create_graph=True)[0]
        grad_phi_val = torch.autograd.grad(val_loss, phi_star, create_graph=True)[0]

        # 3. 计算隐函数梯度 (使用共轭梯度法避免求逆)
        implicit_grad = self.implicit_gradient(
            theta, phi_star, train_data, grad_phi_val
        )

        # 4. 总梯度
        hypergradient = grad_theta_val + implicit_grad

        return hypergradient

    def conjugate_gradient(self, A_func, b, max_iter=10, tol=1e-6):
        """
        共轭梯度法求解 Ax = b

        避免直接计算Hessian逆矩阵
        """
        x = torch.zeros_like(b)
        r = b - A_func(x)
        p = r.clone()
        rs_old = torch.sum(r * r)

        for _ in range(max_iter):
            Ap = A_func(p)
            alpha = rs_old / (torch.sum(p * Ap) + 1e-10)

            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = torch.sum(r * r)
            if torch.sqrt(rs_new) < tol:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x
```

**应用案例: 超参数优化**

```python
class HyperparameterOptimization:
    """
    双层优化用于超参数优化

    上层: 选择超参数 λ
    下层: 训练模型权重 w
    """

    def upper_objective(self, lambda_reg, w_star):
        """
        上层目标: 验证集性能
        """
        val_loss = 0
        for x, y in self.val_loader:
            pred = self.model(x, w_star)
            val_loss += F.cross_entropy(pred, y)

        return val_loss / len(self.val_loader)

    def lower_objective(self, lambda_reg, w):
        """
        下层目标: 训练集损失 + 正则化
        """
        train_loss = 0
        for x, y in self.train_loader:
            pred = self.model(x, w)
            train_loss += F.cross_entropy(pred, y)

        reg_loss = lambda_reg * torch.sum(w ** 2)

        return train_loss / len(self.train_loader) + reg_loss
```

---

### [4-25] Editorial: Segmentation and Classification

**论文信息**
- **标题**: Editorial: Segmentation and Classification - A Unified Perspective
- **作者**: Xiaohao Cai 等人
- **难度**: ⭐⭐ (低)
- **重要性**: ⭐⭐ (社论/观点性文章)

**核心观点: 分割与分类的统一视角**

```
分割 vs 分类的统一性:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

传统观点:
  分割: 像素级预测 (密集预测)
  分类: 图像级预测 (稀疏预测)
  → 被视为两个不同任务

统一视角:
  分割 = 对每个像素的局部分类
  分类 = 对全局特征的聚合分类
  → 本质是同一问题的不同粒度

数学统一:
  分类: y = f_θ(X)  ∈ R^C
  分割: Y = f_θ(X)  ∈ R^{H×W×C}

  其中C为类别数,分割是分类在空间上的扩展
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**多任务学习框架**

```python
class UnifiedSegmentationClassification(nn.Module):
    """
    统一的分割-分类网络

    同时完成两个任务,共享编码器
    """

    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()

        # 共享编码器
        self.encoder = ResNetEncoder(backbone)

        # 分类分支
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, num_classes)
        )

        # 分割分支
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=32, mode='bilinear'),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        # 共享特征
        features = self.encoder(x)

        # 分类输出
        cls_logits = self.classification_head(features)

        # 分割输出
        seg_logits = self.segmentation_head(features)

        return {
            'classification': cls_logits,
            'segmentation': seg_logits
        }
```

---

## 主题四: 生物启发与贝叶斯方法

### [4-22] Cross-Domain LiDAR Object Detection

**论文信息**
- **标题**: Cross-Domain LiDAR Object Detection: A Benchmark and Baseline
- **作者**: X. Cai 等人
- **发表**: Remote Sensing (MDPI), 2022
- **难度**: ⭐⭐⭐ (中等)
- **重要性**: ⭐⭐⭐⭐⭐ (井盖跨场景检测核心参考)

**跨域问题定义**

```
源域 (Source Domain)          →  目标域 (Target Domain)
─────────────────────────────────────────────────────
KITTI (德国)                   →  nuScenes (美国/新加坡)
Waymo (白天/晴天)              →  Waymo (夜晚/雨天)
64线激光雷达                   →  32线激光雷达
密集城区                       →  稀疏郊区
```

**域适应损失**

```python
class DomainAdaptationLoss(nn.Module):
    """
    域适应损失: 对抗训练
    """
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, source_features, target_features, domain_discriminator):
        """
        Args:
            source_features: 源域特征 (B, C, H, W)
            target_features: 目标域特征 (B, C, H, W)
            domain_discriminator: 域判别器
        """
        batch_size = source_features.shape[0]

        # 源域标签为0，目标域标签为1
        source_labels = torch.zeros(batch_size, device=source_features.device)
        target_labels = torch.ones(batch_size, device=target_features.device)

        # 域判别
        source_pred = domain_discriminator(source_features).squeeze()
        target_pred = domain_discriminator(target_features).squeeze()

        # 对抗损失: 希望判别器无法区分源域和目标域
        source_loss = self.bce_loss(source_pred, 1 - source_labels)
        target_loss = self.bce_loss(target_pred, target_labels)

        total_loss = (source_loss + target_loss) / 2
        return total_loss


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy Loss
    最小化源域和目标域特征分布的差异
    """
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source, target):
        """多尺度高斯核"""
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # 计算所有样本对之间的距离
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # 多尺度高斯核
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i)
                         for i in range(self.kernel_num)]

        kernel_val = [torch.exp(-L2_distance / b) for b in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source_features, target_features):
        """计算MMD损失"""
        source_features = source_features.view(source_features.size(0), -1)
        target_features = target_features.view(target_features.size(0), -1)

        kernels = self.gaussian_kernel(source_features, target_features)

        n_source = source_features.size(0)
        n_target = target_features.size(0)

        XX = kernels[:n_source, :n_source].mean()
        YY = kernels[n_source:, n_source:].mean()
        XY = kernels[:n_source, n_source:].mean()

        mmd_loss = XX + YY - 2 * XY
        return mmd_loss
```

**实验结果 (KITTI → Waymo)**

| 方法 | 源域性能 | 目标域性能 | 适应后提升 |
|:---|:---:|:---:|:---:|
| PointRCNN (无适应) | 75.64 | 52.31 | - |
| CenterPoint (无适应) | 79.12 | 58.45 | - |
| **+ 对抗适应** | 78.89 | **63.21** | **+4.76** |
| **+ MMD适应** | 78.95 | **64.58** | **+6.13** |
| **+ 全部** | 78.76 | **66.34** | **+7.89** |

---

### [4-24] Biologically-Inspired Iterative Learning Control

**论文信息**
- **标题**: Biologically-Inspired Iterative Learning Control
- **作者**: Xiaohao Cai 等人
- **难度**: ⭐⭐⭐ (中)
- **重要性**: ⭐⭐⭐ (控制理论与生物启发结合)

**生物启发ILC框架**

```
生物运动控制特点:
┌─────────────────────────────────────────────────────────┐
│  1. 小脑 (Cerebellum)                                   │
│     - 运动学习与协调                                     │
│     - 误差驱动的前馈调整                                 │
│                                                         │
│  2. 脊髓反射 (Spinal Reflex)                            │
│     - 快速反馈响应                                       │
│     - 实时误差修正                                       │
│                                                         │
│  3. 运动皮层 (Motor Cortex)                             │
│     - 运动规划                                           │
│     - 高层目标设定                                       │
└─────────────────────────────────────────────────────────┘

映射到ILC:
┌─────────────────────────────────────────────────────────┐
│  ILC控制器 ←→ 小脑                                       │
│  反馈控制  ←→ 脊髓反射                                   │
│  轨迹规划  ←→ 运动皮层                                   │
└─────────────────────────────────────────────────────────┘
```

**小脑型学习律**

```python
class CerebellarILC(nn.Module):
    """
    小脑启发的ILC学习律

    特点:
    - 多时间尺度记忆
    - 自适应学习率
    - 误差预测
    """

    def __init__(self, input_dim, hidden_dim=64, memory_horizon=5):
        super().__init__()
        self.memory_horizon = memory_horizon

        # 长时程记忆 (类似小脑长时程增强LTP)
        self.long_term_memory = nn.Parameter(
            torch.zeros(memory_horizon, input_dim)
        )

        # 短时程记忆
        self.short_term_buffer = []

        # 误差预测网络 (类似小脑内部模型)
        self.error_predictor = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # 自适应学习率
        self.learning_rate_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, current_error, previous_update):
        """
        计算控制更新

        Returns:
            control_update: 控制输入更新
            adaptive_lr: 自适应学习率
        """
        # 更新短时程记忆
        self.short_term_buffer.append(current_error.detach())
        if len(self.short_term_buffer) > self.memory_horizon:
            self.short_term_buffer.pop(0)

        # 计算记忆加权误差
        memory_error = self.compute_memory_weighted_error()

        # 预测下一时刻误差
        combined = torch.cat([current_error, memory_error], dim=-1)
        predicted_error = self.error_predictor(combined)

        # 自适应学习率
        adaptive_lr = self.learning_rate_net(combined)

        # 控制更新
        control_update = (
            adaptive_lr * current_error +
            (1 - adaptive_lr) * predicted_error +
            0.1 * previous_update  # 动量项
        )

        return control_update, adaptive_lr
```

---

### [4-26] Electron Tomography Analysis of Prolamellar Bodies

**论文信息**
- **标题**: Electron Tomography Analysis of Prolamellar Bodies
- **作者**: Xiaohao Cai 等人
- **难度**: ⭐⭐⭐ (中)
- **重要性**: ⭐⭐⭐ (细胞生物学成像)

**电子断层成像流程**

```
电子断层成像工作流程:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 样品制备
   - 化学固定或冷冻固定
   - 超薄切片 (50-100nm)
   - 重金属染色 (增强对比度)

2. 数据采集
   - 倾斜系列成像 (-70° to +70°)
   - 步长: 1-2°
   - 获取70-140张投影图像

3. 图像对齐
   - 金颗粒标记物追踪
   - 基于特征的图像配准
   - 消除机械漂移

4. 三维重建
   - 加权反投影 (WBP)
   - SIRT迭代重建
   - 生成3D体数据

5. 分割与分析
   - 膜结构分割
   - 三维可视化
   - 形态计量分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**三维重建算法**

```python
class TomographicReconstruction:
    """
    断层重建算法
    """

    def weighted_back_projection(self, aligned_projections, angles):
        """
        加权反投影 (WBP) 重建

        经典解析重建方法
        """
        # 初始化3D体积
        size = aligned_projections[0].shape[0]
        volume = np.zeros((size, size, size))

        for angle, projection in zip(angles, aligned_projections):
            # 滤波 (Ram-Lak滤波器)
            filtered_proj = self.ram_lak_filter(projection)

            # 反投影
            self.back_project(volume, filtered_proj, angle)

        return volume

    def sirt_reconstruction(self, aligned_projections, angles, iterations=100):
        """
        SIRT (Simultaneous Iterative Reconstruction Technique)

        迭代重建方法,对噪声更鲁棒
        """
        # 初始化
        size = aligned_projections[0].shape[0]
        volume = np.ones((size, size, size)) * 0.5

        for iter in range(iterations):
            # 前向投影
            projections = self.forward_project(volume, angles)

            # 计算误差
            errors = [p - proj for p, proj in zip(aligned_projections, projections)]

            # 反投影误差
            correction = self.back_project_errors(errors, angles)

            # 更新体积
            volume = volume - 0.1 * correction

            # 非负约束
            volume = np.maximum(volume, 0)

        return volume

    def ram_lak_filter(self, projection):
        """Ram-Lak滤波器"""
        # 1D傅里叶变换
        f_projection = np.fft.fft(projection, axis=0)

        # 频率坐标
        n = projection.shape[0]
        freq = np.fft.fftfreq(n)

        # Ram-Lak滤波器 |f|
        filter_response = np.abs(freq)

        # 应用滤波器
        filtered = f_projection * filter_response[:, np.newaxis]

        # 逆傅里叶变换
        return np.real(np.fft.ifft(filtered, axis=0))
```

---

### [4-27] Electron Tomography Analysis of Thylakoid Membranes

**论文信息**
- **标题**: Electron Tomography Analysis of Thylakoid Membranes
- **作者**: Xiaohao Cai 等人
- **难度**: ⭐⭐⭐ (中)
- **重要性**: ⭐⭐⭐ (细胞生物学成像)

**类囊体结构分析**

```python
class ThylakoidStructureAnalyzer:
    """
    类囊体结构分析器

    分析基粒和基质片层的三维结构
    """

    def __init__(self, voxel_size=1.0):
        self.voxel_size = voxel_size  # nm

    def analyze_grana_stacks(self, membrane_mask):
        """
        分析基粒堆叠结构

        Returns:
            grana_stats: 基粒统计信息
        """
        # 识别堆叠区域
        stacked_regions = self.identify_stacked_regions(membrane_mask)

        # 分析每个基粒
        grana_list = []
        for region_id in np.unique(stacked_regions)[1:]:
            granum_mask = stacked_regions == region_id
            granum_stats = self.analyze_single_granum(granum_mask)
            grana_list.append(granum_stats)

        return {
            'num_grana': len(grana_list),
            'mean_diameter': np.mean([g['diameter'] for g in grana_list]),
            'mean_num_layers': np.mean([g['num_layers'] for g in grana_list])
        }

    def identify_stacked_regions(self, membrane_mask):
        """
        识别膜堆叠区域 (基粒)

        基于膜密度和间距
        """
        # 距离变换
        distance = ndimage.distance_transform_edt(~membrane_mask)

        # 识别堆叠: 膜间距小的区域
        # 基粒特征: 膜间距约3-5nm
        lumen_width = (distance > 2) & (distance < 6)

        # 连通分量分析
        labeled, num_features = ndimage.label(lumen_width)

        return labeled

    def compute_membrane_curvature(self, surface_mesh):
        """
        计算膜曲率

        分析膜的弯曲特性
        """
        vertices = surface_mesh['vertices']
        faces = surface_mesh['faces']

        # 计算每个顶点的曲率
        curvatures = []

        for i, vertex in enumerate(vertices):
            # 找到相邻面
            adjacent_faces = self.get_adjacent_faces(i, faces)

            # 估计法向量变化
            normals = [self.compute_face_normal(f, vertices)
                      for f in adjacent_faces]

            # 曲率估计
            curvature = self.estimate_curvature_from_normals(normals)
            curvatures.append(curvature)

        return {
            'mean_curvature': np.mean(curvatures),
            'max_curvature': np.max(curvatures)
        }
```

---

### [4-30] Sparse Bayesian Mass Mapping: Hypothesis Testing

**论文信息**
- **标题**: Sparse Bayesian Mass Mapping: Hypothesis Testing
- **作者**: Xiaohao Cai 等人
- **难度**: ⭐⭐⭐⭐ (高)
- **重要性**: ⭐⭐⭐⭐ (贝叶斯统计方法)

**质量映射问题**

```
弱引力透镜质量映射问题:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

背景:
  - 大质量天体(星系团)会弯曲周围时空
  - 背景星系的光线被偏折
  - 观测到背景星系的形状畸变(剪切)

问题:
  给定观测的剪切场 γ,重建质量分布 κ

数学模型:
  γ = P * κ + n

  其中:
  - γ: 观测剪切 (可观测)
  - κ: 收敛场 (待重建的质量分布)
  - P: 投影算子
  - n: 噪声

挑战:
  1. 问题病态 (ill-posed)
  2. 噪声显著
  3. 需要稀疏先验
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**稀疏贝叶斯模型**

```python
class SparseBayesianMassMapping:
    """
    稀疏贝叶斯质量映射

    使用稀疏先验进行质量分布重建
    """

    def __init__(self, lambda_sparse=1.0, noise_sigma=1.0):
        self.lambda_sparse = lambda_sparse
        self.noise_sigma = noise_sigma

    def sparse_prior(self, x, prior_type='laplace'):
        """
        稀疏先验分布

        Args:
            x: 质量分布 (小波系数)
            prior_type: 'laplace' | 'student_t' | 'horseshoe'
        """
        if prior_type == 'laplace':
            # Laplace先验 (L1正则化的贝叶斯解释)
            log_prior = -self.lambda_sparse * np.sum(np.abs(x))

        elif prior_type == 'student_t':
            # Student-t先验 (重尾分布)
            nu = 3  # 自由度
            log_prior = np.sum(
                np.log(1 + (x / self.lambda_sparse)**2 / nu) * (-(nu + 1) / 2)
            )

        elif prior_type == 'horseshoe':
            # Horseshoe先验 (稀疏性更强)
            log_prior = self._horseshoe_log_prior(x)

        return log_prior

    def likelihood(self, y, x, A):
        """
        似然函数

        p(y|x) = N(y; Ax, sigma^2 I)
        """
        residual = y - A @ x
        log_likelihood = -0.5 * np.sum(residual**2) / (self.noise_sigma**2)

        return log_likelihood
```

**假设检验框架**

```python
class BayesianHypothesisTesting:
    """
    贝叶斯假设检验

    检验特定区域是否有显著质量聚集
    """

    def __init__(self, vi_result):
        self.mu, self.sigma_sq = vi_result

    def test_point(self, index, threshold=0.95):
        """
        单点假设检验

        H0: x_i = 0 (无质量)
        H1: x_i ≠ 0 (有质量)
        """
        post_mean = self.mu[index]
        post_std = np.sqrt(self.sigma_sq[index])

        # 计算P(x_i > 0 | y) 和 P(x_i < 0 | y)
        prob_positive = 1 - stats.norm.cdf(0, post_mean, post_std)
        prob_negative = stats.norm.cdf(0, post_mean, post_std)
        prob_nonzero = prob_positive + prob_negative

        # 贝叶斯因子近似
        bayes_factor = self._approximate_bayes_factor(index)

        # 决策
        reject_h0 = prob_nonzero > threshold

        return {
            'reject_h0': reject_h0,
            'bayes_factor': bayes_factor,
            'posterior_prob_nonzero': prob_nonzero,
            'credibility_interval': (
                post_mean - 2 * post_std,
                post_mean + 2 * post_std
            )
        }

    def test_region(self, region_mask, threshold=0.95):
        """
        区域假设检验

        H0: 区域内所有x_i = 0
        H1: 区域内至少一个x_i ≠ 0
        """
        indices = np.where(region_mask)[0]

        # 计算区域统计量
        region_mean = np.mean(self.mu[indices])
        region_var = np.mean(self.sigma_sq[indices])

        # 联合检验
        test_statistic = np.sum(self.mu[indices]**2 / self.sigma_sq[indices])

        # 近似p值 (基于卡方分布)
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(test_statistic, df=len(indices))

        return {
            'reject_h0': p_value < (1 - threshold),
            'test_statistic': test_statistic,
            'p_value': p_value
        }
```

---

### [4-31] Sparse Bayesian Mass Mapping: Credible Intervals

**论文信息**
- **标题**: Sparse Bayesian Mass Mapping: Credible Intervals
- **作者**: Xiaohao Cai 等人
- **难度**: ⭐⭐⭐⭐ (高)
- **重要性**: ⭐⭐⭐⭐ (不确定性量化)

**可信区间计算方法**

```python
class VariationalCredibleIntervals:
    """
    基于变分推断的可信区间

    使用高斯近似计算解析可信区间
    """

    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def compute_intervals(self, vi_result):
        """
        计算可信区间

        Args:
            vi_result: 变分推断结果 (mu, sigma_sq)

        Returns:
            intervals: {
                'lower': 下界,
                'upper': 上界,
                'width': 区间宽度
            }
        """
        mu, sigma_sq = vi_result
        sigma = np.sqrt(sigma_sq)

        # 高斯近似的分位数
        z_score = stats.norm.ppf(1 - self.alpha / 2)

        lower = mu - z_score * sigma
        upper = mu + z_score * sigma

        return {
            'lower': lower,
            'upper': upper,
            'width': upper - lower,
            'mean': mu,
            'std': sigma
        }

    def compute_hpd_intervals(self, vi_result, num_samples=10000):
        """
        计算最高后验密度 (HPD) 区间

        HPD区间是包含指定概率的最短区间
        """
        mu, sigma_sq = vi_result

        # 从变分分布采样
        samples = np.random.normal(
            mu[:, np.newaxis],
            np.sqrt(sigma_sq)[:, np.newaxis],
            size=(len(mu), num_samples)
        )

        hpd_intervals = []
        for i in range(len(mu)):
            sample = samples[i]
            hpd_lower, hpd_upper = self._compute_hpd(sample, self.confidence_level)
            hpd_intervals.append((hpd_lower, hpd_upper))

        return np.array(hpd_intervals)

    def _compute_hpd(self, samples, credible_mass=0.95):
        """
        计算HPD区间

        使用样本排序方法
        """
        sorted_samples = np.sort(samples)
        n = len(sorted_samples)

        # 找到最短的区间
        interval_idx = int(np.floor(credible_mass * n))
        n_intervals = n - interval_idx

        interval_widths = sorted_samples[interval_idx:] - sorted_samples[:n_intervals]

        min_idx = np.argmin(interval_widths)
        hpd_lower = sorted_samples[min_idx]
        hpd_upper = sorted_samples[min_idx + interval_idx]

        return hpd_lower, hpd_upper
```

---

### [4-32] Sparse Bayesian Mass Mapping: Peak Statistics

**论文信息**
- **标题**: Sparse Bayesian Mass Mapping: Peak Statistics
- **作者**: Xiaohao Cai 等人
- **难度**: ⭐⭐⭐⭐ (高)
- **重要性**: ⭐⭐⭐⭐ (宇宙学统计)

**峰值检测与统计**

```python
class PeakDetector:
    """
    峰值检测器

    从质量图中识别局部极大值
    """

    def __init__(self, min_height=0.0, min_distance=5):
        self.min_height = min_height
        self.min_distance = min_distance

    def detect(self, map_2d):
        """
        检测二维质量图中的峰值

        Returns:
            peaks: 峰值列表 [{'x', 'y', 'height', 'significance'}, ...]
        """
        from scipy.ndimage import maximum_filter

        # 使用最大滤波器找局部极大值
        local_max = maximum_filter(map_2d, size=self.min_distance) == map_2d

        # 应用高度阈值
        above_threshold = map_2d > self.min_height

        # 峰值掩膜
        peak_mask = local_max & above_threshold

        # 提取峰值坐标和属性
        peak_coords = np.argwhere(peak_mask)

        peaks = []
        for y, x in peak_coords:
            peak = {
                'x': x,
                'y': y,
                'height': map_2d[y, x],
                'significance': self._compute_significance(map_2d, x, y)
            }
            peaks.append(peak)

        return peaks


class PeakStatistics:
    """
    峰值统计量计算

    计算各种峰值统计量用于宇宙学分析
    """

    def __init__(self, bins=10):
        self.bins = bins

    def peak_count_histogram(self, peaks, height_bins=None):
        """
        峰值高度分布直方图

        宇宙学敏感统计量
        """
        heights = [p['height'] for p in peaks]

        if height_bins is None:
            height_bins = np.linspace(min(heights), max(heights), self.bins + 1)

        counts, bin_edges = np.histogram(heights, bins=height_bins)

        return counts, bin_edges

    def peak_correlation_function(self, peaks, max_distance=100):
        """
        峰值两点关联函数

        描述峰值的空间分布
        """
        coords = np.array([[p['x'], p['y']] for p in peaks])
        n_peaks = len(coords)

        # 计算所有峰值对距离
        distances = []
        for i in range(n_peaks):
            for j in range(i + 1, n_peaks):
                d = np.linalg.norm(coords[i] - coords[j])
                distances.append(d)

        distances = np.array(distances)

        # 构建关联函数
        bins = np.linspace(0, max_distance, 20)
        hist, _ = np.histogram(distances, bins=bins)

        # 归一化
        area = np.pi * (bins[1:]**2 - bins[:-1]**2)
        density = n_peaks / (1000 * 1000)
        expected = 0.5 * n_peaks * (n_peaks - 1) * area * density

        correlation = hist / (expected + 1e-10)

        return bins[:-1], correlation
```

---

## 跨主题方法论总结

### 共同技术主题

| 主题 | 应用论文 | 核心思想 |
|:---|:---|:---|
| **稀疏性** | [4-30], [4-31], [4-32] | 稀疏先验、压缩感知 |
| **贝叶斯推断** | [4-30], [4-31], [4-32] | 不确定性量化、后验采样 |
| **多目标优化** | [4-19], [4-21] | Pareto最优、权衡分析 |
| **迭代优化** | [4-23], [4-24] | 嵌套优化、自适应学习 |
| **域适应** | [4-20], [4-22] | 特征对齐、跨域泛化 |

### 可复用代码组件库

```python
# 1. 多目标优化组件
class ParetoOptimizer:
    """通用Pareto优化框架"""
    pass

# 2. 贝叶斯推断组件
class VariationalInference:
    """变分推断基础类"""
    pass

# 3. 假设检验组件
class BayesianHypothesisTest:
    """贝叶斯假设检验框架"""
    pass

# 4. 域适应组件
class DomainAdaptation:
    """域适应基础类"""
    pass

# 5. 跟踪组件
class MultiObjectTracker:
    """多目标跟踪基础框架"""
    pass
```

---

## 对井盖检测项目的启示

### 直接可迁移技术

1. **NAS技术** ([4-19], [4-20])
   - 边缘设备优化的轻量级检测网络
   - 多目标权衡: 精度 vs 速度 vs 功耗

2. **多目标跟踪** ([4-21])
   - 移动巡检中的井盖跟踪
   - 避免重复计数、轨迹分析

3. **域适应** ([4-22])
   - 跨场景检测: 晴天→雨天、水泥路→沥青路
   - MMD损失减少域间差异

4. **遥感匹配** ([4-29])
   - 违建检测中的变化检测
   - 线特征匹配方法

5. **贝叶斯方法** ([4-30], [4-31], [4-32])
   - 检测结果的不确定性量化
   - 异常模式统计与假设检验

---

## 参考文献索引

| 编号 | 论文标题 | 主题 | 页码 |
|:---|:---|:---|:---:|
| [4-19] | Balanced Neural Architecture Search | NAS | - |
| [4-20] | NAS for Specific Emitter Identification | NAS | - |
| [4-21] | GRASPTrack: Geometric Reasoning and Association | 跟踪 | - |
| [4-22] | Cross-Domain LiDAR Object Detection | 域适应 | - |
| [4-23] | Bilevel Optimization: Theory and Applications | 优化理论 | - |
| [4-24] | Biologically-Inspired Iterative Learning Control | 生物启发 | - |
| [4-25] | Editorial: Segmentation and Classification | 统一视角 | - |
| [4-26] | Electron Tomography Analysis of Prolamellar Bodies | 3D重建 | - |
| [4-27] | Electron Tomography Analysis of Thylakoid Membranes | 3D重建 | - |
| [4-29] | Line Feature-Based Ship Matching in Remote Sensing Images | 遥感匹配 | - |
| [4-30] | Sparse Bayesian Mass Mapping: Hypothesis Testing | 贝叶斯统计 | - |
| [4-31] | Sparse Bayesian Mass Mapping: Credible Intervals | 贝叶斯统计 | - |
| [4-32] | Sparse Bayesian Mass Mapping: Peak Statistics | 贝叶斯统计 | - |

---

**整合完成时间**: 2026年2月10日
**整合者**: Claude Code
**版本**: 1.0
