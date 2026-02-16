# 第十六讲：图割与能量优化

## Graph Cuts and Energy Optimization

---

### 📋 本讲大纲

1. 图论基础
2. 能量函数设计
3. 最大流最小割
4. α-expansion算法
5. 图像分割应用

---

### 16.1 图论基础

#### 图的定义

图 $G = (V, E)$：
- $V$：顶点集（如像素）
- $E$：边集（如邻接关系）
- $w(e)$：边权重

#### 割(Cut)

将顶点分为两部分 $S$ 和 $\bar{S}$：
$$\text{Cut}(S, \bar{S}) = \sum_{u \in S, v \in \bar{S}} w(u, v)$$

---

### 16.2 能量最小化问题

#### 一般形式

$$E(L) = \sum_{p \in V} D_p(L_p) + \sum_{(p,q) \in E} V_{pq}(L_p, L_q)$$

| 项 | 含义 |
|---|------|
| $D_p(L_p)$ | 数据项（一元势能） |
| $V_{pq}(L_p, L_q)$ | 平滑项（二元势能） |

#### 应用

- 图像分割
- 立体匹配
- 光流估计

---

### 16.3 s-t割

#### 定义

给定源点 $s$ 和汇点 $t$，s-t割将顶点分为 $S$（包含 $s$）和 $T$（包含 $t$）

#### 最大流最小割定理

$$\text{最大s-t流} = \text{最小s-t割}$$

**动画建议**：展示Ford-Fulkerson算法求最大流的过程

---

### 16.4 最大流算法

#### Ford-Fulkerson

```
初始化：流 f = 0
while 存在增广路径 P do
  计算路径容量 c = min{c_e : e ∈ P}
  对路径上每条边：f_e += c
  对反向边：f_e -= c
end while
返回 f
```

#### 更高效算法

- **Edmonds-Karp**：BFS找增广路径，$O(VE^2)$
- **Dinic**：层次图，$O(V^2E)$
- **Push-Relabel**：$O(V^2\sqrt{E})$

---

### 16.5 二值标注的图割

#### 问题

两标签分割：$L_p \in \{0, 1\}$

#### 图构造

- 每个像素一个节点
- 添加源点 $s$（标签0）和汇点 $t$（标签1）
- t-link：$s \to p$ 权重为 $D_p(0)$，$p \to t$ 权重为 $D_p(1)$
- n-link：相邻像素间，权重为 $V_{pq}$

#### 最优性

对于子模势能，图割给出全局最优解

---

### 16.6 子模性

#### 定义

二元势能 $V(L_p, L_q)$ 是子模的，当且仅当：
$$V(0,0) + V(1,1) \leq V(0,1) + V(1,0)$$

#### 常见子模势能

- Potts模型：$V(L_p, L_q) = \lambda \cdot [L_p \neq L_q]$
- 截断线性：$V = \min(\lambda|L_p - L_q|, K)$

**动画建议**：展示子模性如何保证全局最优

---

### 16.7 多标签扩展

#### 挑战

多标签问题（$K > 2$）通常是NP-hard

#### 近似算法

1. **α-β swap**
2. **α-expansion**
3. **线性规划松弛**

---

### 16.8 α-expansion算法

#### 思想

每次迭代将部分像素"扩展"到标签α

#### 算法

```
repeat
  for 每个标签 α do
    构造图割问题：
    - 选择：保持当前标签 或 扩展到α
    - 求解最小割
    - 更新标签
  end for
until 收敛
```

#### 近似保证

对于Potts模型：
$$E(\hat{L}) \leq 2 \cdot E(L^*)$$

---

### 16.9 α-β swap算法

#### 算法

```
repeat
  for 每对标签 (α, β) do
    构造图割问题：
    - 只涉及标签为α或β的像素
    - 选择：保持α 或 换成β
    - 求解最小割
  end for
until 收敛
```

#### 适用范围

适用于更广泛的势能函数

---

### 16.10 图像分割应用

#### Boykov-Kolmogorov模型

$$E(L) = \sum_p D_p(L_p) + \lambda \sum_{(p,q)} [L_p \neq L_q] \cdot e^{-\frac{(I_p - I_q)^2}{2\sigma^2}}$$

#### 交互式分割

- 用户标记前景/背景种子
- 构建相应的t-link权重
- 求解图割

#### GrabCut

迭代优化：
1. 固定分割，更新GMM参数
2. 固定GMM，更新分割（图割）

---

### 16.11 高阶能量

#### 超像素约束

$$E_{superpixel} = \sum_S w_S \cdot [L_p \neq L_S, \forall p \in S]$$

#### 共现约束

某些标签组合更可能同时出现

#### 求解方法

- 归约到成对能量
- 使用多标签图割

---

### 16.12 与变分方法的联系

#### 水平集 vs 图割

| 方面 | 水平集 | 图割 |
|------|--------|------|
| 连续性 | 连续 | 离散 |
| 优化 | 局部最优 | 全局最优（部分） |
| 拓扑变化 | 自动 | 需特殊处理 |
| 速度 | 慢 | 快 |

#### 离散-连续统一

- 连续能量 → 离散图
- 离散最优 → 连续解

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           图割与能量优化                         │
├─────────────────────────────────────────────────┤
│                                                 │
│   能量函数：                                     │
│   E(L) = Σ D_p(L_p) + Σ V_pq(L_p, L_q)         │
│         数据项        平滑项                    │
│                                                 │
│   核心定理：                                     │
│   最大流 = 最小割                               │
│                                                 │
│   算法：                                        │
│   • 二值：全局最优（子模）                      │
│   • 多标签：α-expansion（2-近似）               │
│                                                 │
│   应用：                                        │
│   图像分割、立体匹配、光流                      │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **实现题**：实现Ford-Fulkerson最大流算法

2. **实现题**：使用图割实现二值图像分割

3. **分析题**：比较α-expansion和α-β swap的近似比

4. **应用题**：实现交互式图像分割（GrabCut简化版）

---

### 📖 扩展阅读

1. **经典论文**：
   - Boykov & Kolmogorov, "An Experimental Comparison of Min-Cut/Max-Flow Algorithms", PAMI, 2004
   - Boykov et al., "Fast Approximate Energy Minimization via Graph Cuts", PAMI, 2001

2. **软件库**：
   - Boost Graph Library
   - gco-v3.0 (多标签图割)
   - OpenCV GrabCut

3. **Cai相关论文**：
   - 图割在点云分割中的应用

---

### 📖 参考文献

1. Boykov, Y. & Kolmogorov, V. (2004). An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision. *IEEE PAMI*, 26(9), 1124-1137.

2. Boykov, Y., Veksler, O., & Zabih, R. (2001). Fast approximate energy minimization via graph cuts. *IEEE PAMI*, 23(11), 1222-1239.

3. Kolmogorov, V. & Zabih, R. (2004). What energy functions can be minimized via graph cuts? *IEEE PAMI*, 26(2), 147-159.

4. Rother, C., Kolmogorov, V., & Blake, A. (2004). GrabCut: Interactive foreground extraction using iterated graph cuts. *ACM TOG*, 23(3), 309-314.
