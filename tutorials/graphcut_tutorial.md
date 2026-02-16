# Graph Cut图割算法教程

## 目录
1. [理论讲解](#1-理论讲解)
2. [算法详解](#2-算法详解)
3. [代码实现](#3-代码实现)
4. [实验指南](#4-实验指南)
5. [习题与答案](#5-习题与答案)

---

## 1. 理论讲解

### 1.1 图论基础

**图的定义**

图 $G = (V, E)$ 由顶点集 $V$ 和边集 $E$ 组成。在图像分割中：
- 顶点 $V$：图像像素（加上额外的源点和汇点）
- 边 $E$：像素间的邻接关系

**割（Cut）的定义**

对于两个顶点子集 $S, T \subset V$，$S \cap T = \emptyset$，割定义为：
$$C(S, T) = \sum_{u \in S, v \in T} w(u, v)$$

其中 $w(u, v)$ 为边权重。

**最小割问题**

给定源点 $s$ 和汇点 $t$，找到将 $s$ 和 $t$ 分开的割，使得割的权重最小：
$$\min_{S: s \in S, t \notin S} C(S, S^c)$$

### 1.2 图像分割的能量模型

**Gibbs能量形式**

图像分割问题可以形式化为最小化能量函数：
$$E(L) = \sum_{p \in V} D_p(L_p) + \sum_{(p,q) \in N} V_{p,q}(L_p, L_q)$$

其中：
- $L = \{L_p\}_{p \in V}$ 为标签分配
- $D_p(L_p)$：数据项（一元势能），表示像素 $p$ 属于标签 $L_p$ 的代价
- $V_{p,q}(L_p, L_q)$：平滑项（二元势能），表示相邻像素标签不一致的代价
- $N$：邻域系统

**数据项设计**

对于前景/背景分割：
$$D_p(L_p) = -\log P(I_p | L_p)$$

其中 $I_p$ 为像素 $p$ 的强度，$P(I_p | L_p)$ 为条件概率（可从直方图估计）。

**平滑项设计**

常用的Ising/Potts模型：
$$V_{p,q}(L_p, L_q) = \begin{cases} 0 & \text{if } L_p = L_q \\ \lambda \cdot \exp\left(-\frac{(I_p - I_q)^2}{2\sigma^2}\right) & \text{if } L_p \neq L_q \end{cases}$$

### 1.3 最大流最小割定理

**定理内容**

在流网络中，从源点 $s$ 到汇点 $t$ 的最大流值等于分离 $s$ 和 $t$ 的最小割容量。

**数学表述**

$$\max_{f} |f| = \min_{C} c(C)$$

其中 $|f|$ 为流值，$c(C)$ 为割容量。

**证明思路（直观）**

1. 最大流存在时，残量网络中无增广路径
2. 无增广路径意味着存在割，且该割的容量等于当前流值
3. 因此最大流值 = 该割容量 ≥ 最小割容量
4. 另一方面，任何割都是流的瓶颈，故最大流 ≤ 最小割
5. 综合得：最大流 = 最小割

### 1.4 图构建方法

**s-t图构建**

将分割能量嵌入s-t图：

1. **顶点**：每个像素 $p$ 对应一个顶点，加上源点 $s$（前景）和汇点 $t$（背景）

2. **t-link（终端边）**：
   - $(s, p)$：容量 $D_p(\text{background})$
   - $(p, t)$：容量 $D_p(\text{foreground})$

3. **n-link（邻域边）**：
   - $(p, q)$ 对于邻域像素对：容量 $V_{p,q}$

**分割结果**

最小割将顶点分为两部分：
- 与 $s$ 相连：前景
- 与 $t$ 相连：背景

### 1.5 可解性分析

**次模性条件**

二元势能 $V_{p,q}$ 需要满足次模性才能保证多项式时间可解：
$$V_{p,q}(0,0) + V_{p,q}(1,1) \leq V_{p,q}(0,1) + V_{p,q}(1,0)$$

对于Ising/Potts模型，等式成立，是次模的。

**NP困难情况**

多标签（$K > 2$）的一般情况是NP困难的。但以下情况可解：
- 子模势能 + $\alpha$-expansion
- 2D网格 + 凸势能

---

## 2. 算法详解

### 2.1 Ford-Fulkerson方法

**基本思想**

迭代寻找增广路径，增加流值，直到无法增广。

**算法步骤**

```
1. 初始化流 f = 0
2. While 存在增广路径 P in 残量网络 G_f:
   a. 计算 P 上的瓶颈容量 δ = min{(u,v) in P} c_f(u,v)
   b. 增广: f(u,v) += δ for forward edges
           f(v,u) -= δ for backward edges
3. Return f
```

**复杂度**

- 使用BFS找最短增广路径（Edmonds-Karp）：$O(VE^2)$
- 使用DFS：$O(E \cdot |f^*|)$，其中 $|f^*|$ 为最大流值

### 2.2 Push-Relabel算法

**基本思想**

允许流"溢出"，然后通过推流和重标记操作逐步平衡。

**核心操作**

1. **Push(u, v)**：如果 $u$ 有超额流且高度大于 $v$，将流推给 $v$
2. **Relabel(u)**：如果 $u$ 有超额流但无法推流，增加 $u$ 的高度

**算法步骤**

```
1. 初始化:
   - h[s] = |V|, h[v] = 0 for v != s
   - 饱和推流: s -> 所有邻居
2. While 存在溢出顶点:
   a. 选择溢出顶点 u
   b. If 可以推流: Push(u, v)
   c. Else: Relabel(u)
3. Return f
```

**复杂度**

- 基本版本：$O(V^2 E)$
- 使用高度估计（HLPP）：$O(V^2 \sqrt{E})$

### 2.3 Boykov-Kolmogorov算法

**核心创新**

结合BFS和DFS的优势，使用两棵搜索树：
- $S$-tree：从源点生长
- $T$-tree：向汇点生长

**三个阶段**

1. **Growth**：扩展搜索树直到相遇
2. **Augmentation**：沿找到的路径增广
3. **Adoption**：重新确定孤立节点的归属

**优势**

- 对于图像分割的典型图结构效率高
- 增量式处理，适合交互式分割

### 2.4 多标签扩展

**$\alpha$-expansion**

每次将一部分像素的标签变为 $\alpha$：

```
For each label α in {1,...,K}:
    1. 构建辅助图: 允许像素选择保持原标签或变为α
    2. 求解二值Graph Cut
    3. 更新标签
Until 收敛
```

**$\alpha$-$\beta$-swap**

每次在两个标签之间交换：

```
For each pair (α, β):
    1. 构建辅助图: 只涉及标签为α或β的像素
    2. 求解二值Graph Cut
    3. 更新标签
Until 收敛
```

---

## 3. 代码实现

### 3.1 图数据结构

```python
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import heapq


class GraphCut:
    """
    最大流最小割算法实现（基于Edmonds-Karp）
    """
    
    def __init__(self, n_vertices: int):
        """
        初始化图
        
        参数:
            n_vertices: 顶点数
        """
        self.n = n_vertices
        self.capacity = {}  # (u, v) -> 容量
        self.flow = {}      # (u, v) -> 当前流
        self.adj = [[] for _ in range(n_vertices)]  # 邻接表
    
    def add_edge(self, u: int, v: int, cap: float):
        """
        添加边
        
        参数:
            u: 起点
            v: 终点
            cap: 容量
        """
        if (u, v) not in self.capacity:
            self.adj[u].append(v)
            self.adj[v].append(u)
            self.capacity[(u, v)] = 0
            self.capacity[(v, u)] = 0
            self.flow[(u, v)] = 0
            self.flow[(v, u)] = 0
        
        self.capacity[(u, v)] += cap
    
    def _bfs_augmenting_path(self, s: int, t: int) -> Tuple[List[int], float]:
        """
        BFS寻找增广路径
        
        返回:
            path: 路径
            bottleneck: 瓶颈容量
        """
        parent = [-1] * self.n
        parent[s] = s
        queue = deque([s])
        
        while queue and parent[t] == -1:
            u = queue.popleft()
            for v in self.adj[u]:
                if parent[v] == -1:
                    residual = self.capacity[(u, v)] - self.flow[(u, v)]
                    if residual > 0:
                        parent[v] = u
                        queue.append(v)
        
        if parent[t] == -1:
            return [], 0
        
        # 重建路径
        path = []
        v = t
        while v != s:
            u = parent[v]
            path.append((u, v))
            v = u
        path.reverse()
        
        # 计算瓶颈
        bottleneck = float('inf')
        for u, v in path:
            residual = self.capacity[(u, v)] - self.flow[(u, v)]
            bottleneck = min(bottleneck, residual)
        
        return path, bottleneck
    
    def max_flow(self, s: int, t: int) -> float:
        """
        计算最大流
        
        参数:
            s: 源点
            t: 汇点
        
        返回:
            最大流值
        """
        total_flow = 0
        
        while True:
            path, bottleneck = self._bfs_augmenting_path(s, t)
            if bottleneck == 0:
                break
            
            # 增广
            for u, v in path:
                self.flow[(u, v)] += bottleneck
                self.flow[(v, u)] -= bottleneck
            
            total_flow += bottleneck
        
        return total_flow
    
    def min_cut(self, s: int) -> Tuple[set, set]:
        """
        计算最小割
        
        参数:
            s: 源点
        
        返回:
            S: 源点侧顶点集
            T: 汇点侧顶点集
        """
        # BFS找从源点可达的顶点
        visited = [False] * self.n
        queue = deque([s])
        visited[s] = True
        
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if not visited[v]:
                    residual = self.capacity[(u, v)] - self.flow[(u, v)]
                    if residual > 0:
                        visited[v] = True
                        queue.append(v)
        
        S = {i for i in range(self.n) if visited[i]}
        T = {i for i in range(self.n) if not visited[i]}
        
        return S, T


class PushRelabel:
    """
    Push-Relabel最大流算法（高效实现）
    """
    
    def __init__(self, n_vertices: int):
        self.n = n_vertices
        self.capacity = [[0] * n_vertices for _ in range(n_vertices)]
        self.flow = [[0] * n_vertices for _ in range(n_vertices)]
        self.height = [0] * n_vertices
        self.excess = [0] * n_vertices
    
    def add_edge(self, u: int, v: int, cap: float):
        self.capacity[u][v] += cap
    
    def _push(self, u: int, v: int) -> bool:
        residual = self.capacity[u][v] - self.flow[u][v]
        if residual <= 0 or self.height[u] <= self.height[v]:
            return False
        
        push_amount = min(self.excess[u], residual)
        self.flow[u][v] += push_amount
        self.flow[v][u] -= push_amount
        self.excess[u] -= push_amount
        self.excess[v] += push_amount
        return True
    
    def _relabel(self, u: int):
        min_height = float('inf')
        for v in range(self.n):
            if self.capacity[u][v] - self.flow[u][v] > 0:
                min_height = min(min_height, self.height[v])
        if min_height < float('inf'):
            self.height[u] = min_height + 1
    
    def max_flow(self, s: int, t: int) -> float:
        # 初始化
        self.height[s] = self.n
        for v in range(self.n):
            if self.capacity[s][v] > 0:
                self.flow[s][v] = self.capacity[s][v]
                self.flow[v][s] = -self.capacity[s][v]
                self.excess[v] = self.capacity[s][v]
                self.excess[s] -= self.capacity[s][v]
        
        # 主循环
        while True:
            # 找溢出顶点
            u = -1
            for i in range(self.n):
                if i != s and i != t and self.excess[i] > 0:
                    u = i
                    break
            
            if u == -1:
                break
            
            # 尝试推流
            pushed = False
            for v in range(self.n):
                if self._push(u, v):
                    pushed = True
                    break
            
            if not pushed:
                self._relabel(u)
        
        return -self.excess[s]
```

### 3.2 图像分割Graph Cut

```python
class ImageGraphCut:
    """
    图像分割Graph Cut实现
    """
    
    def __init__(self, image: np.ndarray):
        """
        初始化
        
        参数:
            image: 输入图像 (H, W) 或 (H, W, 3)
        """
        self.image = image
        self.h, self.w = image.shape[:2]
        self.n_pixels = self.h * self.w
        
        # 节点编号: 0~n_pixels-1为像素, n_pixels为源, n_pixels+1为汇
        self.source = self.n_pixels
        self.sink = self.n_pixels + 1
    
    def _pixel_to_node(self, i: int, j: int) -> int:
        """像素坐标转节点编号"""
        return i * self.w + j
    
    def _node_to_pixel(self, node: int) -> Tuple[int, int]:
        """节点编号转像素坐标"""
        return node // self.w, node % self.w
    
    def build_graph(self, 
                    foreground_seeds: List[Tuple[int, int]],
                    background_seeds: List[Tuple[int, int]],
                    lambda_: float = 1.0,
                    sigma: float = 10.0) -> GraphCut:
        """
        构建s-t图
        
        参数:
            foreground_seeds: 前景种子点
            background_seeds: 背景种子点
            lambda_: 平滑项权重
            sigma: 强度差异敏感度
        
        返回:
            GraphCut对象
        """
        graph = GraphCut(self.n_pixels + 2)
        
        # 计算前景和背景的直方图
        fg_intensities = [self.image[i, j] for i, j in foreground_seeds]
        bg_intensities = [self.image[i, j] for i, j in background_seeds]
        
        # 构建直方图
        fg_hist, bins = np.histogram(fg_intensities, bins=256, range=(0, 256), density=True)
        bg_hist, _ = np.histogram(bg_intensities, bins=256, range=(0, 256), density=True)
        
        # 平滑直方图
        fg_hist = np.clip(fg_hist, 1e-10, None)
        bg_hist = np.clip(bg_hist, 1e-10, None)
        
        # 添加t-links和n-links
        for i in range(self.h):
            for j in range(self.w):
                node = self._pixel_to_node(i, j)
                intensity = self.image[i, j]
                bin_idx = min(int(intensity), 255)
                
                # 检查是否为种子点
                is_fg_seed = (i, j) in foreground_seeds
                is_bg_seed = (i, j) in background_seeds
                
                if is_fg_seed:
                    # 前景种子：强制连接到源点
                    graph.add_edge(self.source, node, float('inf'))
                    graph.add_edge(node, self.sink, 0)
                elif is_bg_seed:
                    # 背景种子：强制连接到汇点
                    graph.add_edge(self.source, node, 0)
                    graph.add_edge(node, self.sink, float('inf'))
                else:
                    # 普通像素：根据直方图设置容量
                    fg_prob = fg_hist[bin_idx]
                    bg_prob = bg_hist[bin_idx]
                    
                    # 负对数似然作为代价
                    cost_fg = -np.log(fg_prob / (fg_prob + bg_prob + 1e-10))
                    cost_bg = -np.log(bg_prob / (fg_prob + bg_prob + 1e-10))
                    
                    # 归一化
                    max_cost = max(cost_fg, cost_bg, 1)
                    cost_fg = cost_fg / max_cost * 10
                    cost_bg = cost_bg / max_cost * 10
                    
                    graph.add_edge(self.source, node, cost_bg)
                    graph.add_edge(node, self.sink, cost_fg)
                
                # n-links: 4邻域
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.h and 0 <= nj < self.w:
                        neighbor = self._pixel_to_node(ni, nj)
                        
                        # 计算边界代价
                        diff = abs(intensity - self.image[ni, nj])
                        boundary_cost = lambda_ * np.exp(-diff**2 / (2 * sigma**2))
                        
                        graph.add_edge(node, neighbor, boundary_cost)
        
        return graph
    
    def segment(self, 
                foreground_seeds: List[Tuple[int, int]],
                background_seeds: List[Tuple[int, int]],
                lambda_: float = 1.0,
                sigma: float = 10.0) -> np.ndarray:
        """
        执行分割
        
        返回:
            分割掩码 (H, W), True为前景
        """
        graph = self.build_graph(foreground_seeds, background_seeds, lambda_, sigma)
        
        # 计算最大流
        max_flow = graph.max_flow(self.source, self.sink)
        print(f"Max flow: {max_flow:.2f}")
        
        # 获取最小割
        S, T = graph.min_cut(self.source)
        
        # 生成分割掩码
        mask = np.zeros((self.h, self.w), dtype=bool)
        for i in range(self.h):
            for j in range(self.w):
                node = self._pixel_to_node(i, j)
                mask[i, j] = node in S
        
        return mask
```

### 3.3 交互式分割工具

```python
class InteractiveSegmentation:
    """
    交互式Graph Cut分割
    """
    
    def __init__(self, image: np.ndarray):
        self.image = image
        self.gc = ImageGraphCut(image)
        self.fg_seeds = []
        self.bg_seeds = []
    
    def add_foreground(self, points: List[Tuple[int, int]]):
        """添加前景种子"""
        self.fg_seeds.extend(points)
    
    def add_background(self, points: List[Tuple[int, int]]):
        """添加背景种子"""
        self.bg_seeds.extend(points)
    
    def clear_seeds(self):
        """清除所有种子"""
        self.fg_seeds = []
        self.bg_seeds = []
    
    def run(self, lambda_: float = 1.0, sigma: float = 10.0) -> np.ndarray:
        """执行分割"""
        if not self.fg_seeds or not self.bg_seeds:
            raise ValueError("需要同时提供前景和背景种子点")
        
        return self.gc.segment(self.fg_seeds, self.bg_seeds, lambda_, sigma)
    
    def visualize(self, mask: Optional[np.ndarray] = None):
        """可视化结果"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3 if mask is not None else 2, figsize=(15, 5))
        
        # 原图 + 种子
        axes[0].imshow(self.image, cmap='gray')
        if self.fg_seeds:
            fg_y, fg_x = zip(*self.fg_seeds)
            axes[0].scatter(fg_x, fg_y, c='red', s=10, label='FG')
        if self.bg_seeds:
            bg_y, bg_x = zip(*self.bg_seeds)
            axes[0].scatter(bg_x, bg_y, c='blue', s=10, label='BG')
        axes[0].set_title('Image with Seeds')
        axes[0].legend()
        axes[0].axis('off')
        
        if mask is not None:
            # 分割结果
            axes[1].imshow(mask, cmap='gray')
            axes[1].set_title('Segmentation')
            axes[1].axis('off')
            
            # 叠加
            overlay = self.image.copy()
            if len(overlay.shape) == 2:
                overlay = np.stack([overlay]*3, axis=-1)
            overlay[mask] = [1, 0, 0]  # 前景标红
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
```

### 3.4 GrabCut变体

```python
def grabcut_simplified(image: np.ndarray, 
                       rect: Tuple[int, int, int, int],
                       n_iter: int = 5) -> np.ndarray:
    """
    简化版GrabCut算法
    
    参数:
        image: 输入图像
        rect: 感兴趣区域 (x1, y1, x2, y2)
        n_iter: 迭代次数
    
    返回:
        分割掩码
    """
    from sklearn.mixture import GaussianMixture
    
    h, w = image.shape[:2]
    x1, y1, x2, y2 = rect
    
    # 初始化掩码
    # 0: 背景, 1: 前景, 2: 可能背景, 3: 可能前景
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 3  # ROI内为可能前景
    mask[:y1, :] = 0  # ROI外为背景
    mask[y2:, :] = 0
    mask[:, :x1] = 0
    mask[:, x2:] = 0
    
    # 转换为特征向量
    if len(image.shape) == 3:
        features = image.reshape(-1, 3)
    else:
        features = image.reshape(-1, 1)
    
    for it in range(n_iter):
        # 使用GMM建模前景和背景
        fg_mask = (mask == 1) | (mask == 3)
        bg_mask = (mask == 0) | (mask == 2)
        
        if fg_mask.sum() < 10 or bg_mask.sum() < 10:
            break
        
        fg_features = features[fg_mask.flatten()]
        bg_features = features[bg_mask.flatten()]
        
        # 训练GMM
        gmm_fg = GaussianMixture(n_components=5, random_state=42)
        gmm_bg = GaussianMixture(n_components=5, random_state=42)
        
        gmm_fg.fit(fg_features)
        gmm_bg.fit(bg_features)
        
        # 计算概率
        fg_probs = gmm_fg.score_samples(features)
        bg_probs = gmm_bg.score_samples(features)
        
        # 更新掩码
        for i in range(h):
            for j in range(w):
                if mask[i, j] == 0:  # 确定背景，不更新
                    continue
                idx = i * w + j
                if fg_probs[idx] > bg_probs[idx]:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0
    
    return mask == 1
```

### 3.5 完整示例

```python
def demo_graphcut():
    """
    Graph Cut分割演示
    """
    from skimage import data, img_as_ubyte
    
    # 加载图像
    image = img_as_ubyte(data.camera())
    
    print("=" * 50)
    print("Graph Cut图像分割演示")
    print("=" * 50)
    
    # 创建交互式分割器
    seg = InteractiveSegmentation(image)
    
    # 模拟用户输入（实际应用中通过GUI获取）
    # 前景种子：图像中心区域
    h, w = image.shape
    fg_seeds = [(i, j) for i in range(h//3, 2*h//3, 5) 
                        for j in range(w//3, 2*w//3, 5)]
    bg_seeds = [(i, j) for i in range(0, h//4, 10) 
                        for j in range(0, w, 10)]
    
    seg.add_foreground(fg_seeds[:50])
    seg.add_background(bg_seeds[:50])
    
    # 执行分割
    mask = seg.run(lambda_=5.0, sigma=20.0)
    
    # 可视化
    seg.visualize(mask)
    
    return mask


def demo_rect_segmentation():
    """
    基于矩形框的分割演示
    """
    from skimage import data
    
    image = data.astronaut()
    h, w = image.shape[:2]
    
    # 定义感兴趣区域
    rect = (w//4, h//4, 3*w//4, 3*h//4)
    
    print("Running GrabCut...")
    mask = grabcut_simplified(image, rect, n_iter=5)
    
    # 可视化
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    x1, y1, x2, y2 = rect
    axes[0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     fill=False, edgecolor='red', linewidth=2))
    axes[0].set_title('Input with ROI')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    result = image.copy()
    result[~mask] = result[~mask] // 3
    axes[2].imshow(result)
    axes[2].set_title('Result')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_graphcut()
    demo_rect_segmentation()
```

---

## 4. 实验指南

### 4.1 数据集

| 数据集 | 用途 | 特点 |
|--------|------|------|
| GrabCut Dataset | 评估 | 50张带框标注 |
| Berkeley Segmentation | 评估 | 多人标注 |
| PASCAL VOC | 目标分割 | 多类别 |

### 4.2 参数调优

```python
def parameter_study(image, fg_seeds, bg_seeds):
    """
    参数敏感性分析
    """
    lambdas = [0.5, 1, 2, 5, 10]
    sigmas = [5, 10, 20, 50]
    
    results = []
    for lam in lambdas:
        for sig in sigmas:
            seg = InteractiveSegmentation(image)
            seg.add_foreground(fg_seeds)
            seg.add_background(bg_seeds)
            mask = seg.run(lambda_=lam, sigma=sig)
            
            # 计算分割质量（需要ground truth）
            results.append({
                'lambda': lam, 'sigma': sig,
                'fg_ratio': mask.mean()
            })
    
    return results
```

### 4.3 评估指标

- **IoU (Intersection over Union)**
- **Precision / Recall**
- **Boundary F-measure**

### 4.4 扩展实验

1. 多尺度Graph Cut
2. 彩色+纹理特征
3. 与深度学习方法对比

---

## 5. 习题与答案

### 5.1 理论题

**题目1**: 证明最大流最小割定理中的不等式：最大流 ≤ 最小割。

**答案**:
设 $f$ 为任意可行流，$C$ 为任意 $s$-$t$ 割。由于流从 $s$ 流向 $t$，必须穿过割 $C$，因此：
$$|f| = \sum_{u \in S, v \in T} f(u,v) - \sum_{v \in T, u \in S} f(v,u)$$

由容量约束：$f(u,v) \leq c(u,v)$，且反向流非负：
$$|f| \leq \sum_{u \in S, v \in T} c(u,v) = c(C)$$

由于这对任意流和割成立，故最大流值 ≤ 最小割容量。

**题目2**: 解释为什么一般多标签能量最小化是NP困难的。

**答案**:
多标签能量最小化可规约到Potts模型：
$$E(L) = \sum_p D_p(L_p) + \lambda \sum_{(p,q) \in N} [L_p \neq L_q]$$

当 $\lambda \to \infty$ 时，所有相邻像素必须同标签，退化为图着色问题（NP完全）。

**题目3**: 比较Ford-Fulkerson和Push-Relabel的时间复杂度。

**答案**:
- Ford-Fulkerson (Edmonds-Karp): $O(VE^2)$，通过BFS找最短增广路
- Push-Relabel: $O(V^2E)$，使用高度标号
- 实际中，Push-Relabel通常更快，特别是对于密集图

### 5.2 编程题

**题目1**: 实现基于超像素的Graph Cut。

**答案**:
```python
def superpixel_graphcut(image, n_segments=200):
    """
    基于超像素的Graph Cut分割
    """
    from skimage.segmentation import slic
    
    # 生成超像素
    segments = slic(image, n_segments=n_segments, compactness=10)
    
    # 计算超像素特征
    n_sp = segments.max() + 1
    features = np.zeros((n_sp, 3))
    for i in range(n_sp):
        mask = segments == i
        features[i] = image[mask].mean(axis=0)
    
    # 构建超像素邻接图
    from skimage.future import graph
    rag = graph.rag_mean_color(image, segments)
    
    # 在超像素图上执行Graph Cut
    # ... (使用前面定义的GraphCut类)
    
    return segments
```

**题目2**: 实现软边界输出的Graph Cut。

**答案**:
```python
def soft_graphcut(image, fg_seeds, bg_seeds, n_runs=10):
    """
    通过多次运行获得软边界
    """
    masks = []
    for i in range(n_runs):
        # 添加随机扰动到参数
        lam = np.random.uniform(0.8, 1.2) * 5.0
        sig = np.random.uniform(0.8, 1.2) * 20.0
        
        seg = InteractiveSegmentation(image)
        seg.add_foreground(fg_seeds)
        seg.add_background(bg_seeds)
        mask = seg.run(lambda_=lam, sigma=sig)
        masks.append(mask)
    
    # 平均得到概率图
    prob = np.mean(masks, axis=0)
    return prob
```

---

## 参考文献

1. Boykov, Y., & Kolmogorov, V. (2004). An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision. *IEEE PAMI*, 26(9), 1124-1137.

2. Boykov, Y. Y., & Jolly, M. P. (2001). Interactive graph cuts for optimal boundary & region segmentation of objects in N-D images. *ICCV*.

3. Rother, C., Kolmogorov, V., & Blake, A. (2004). GrabCut: Interactive foreground extraction using iterated graph cuts. *ACM TOG*.

4. Kolmogorov, V., & Zabih, R. (2004). What energy functions can be minimized via graph cuts? *IEEE PAMI*.
