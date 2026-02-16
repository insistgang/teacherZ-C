# Challenge 3: 最快最大流 ⭐⭐⭐⭐⭐

## 题目描述

实现Ford-Fulkerson或Edmonds-Karp最大流算法，要求在1秒内处理节点数高达10000的图。

最大流问题：给定有向图G=(V,E)，源点s，汇点t，边的容量c，求s到t的最大流量。

## 输入输出格式

**输入**:
```python
n: int           # 节点数，节点编号0到n-1
edges: List[Tuple[int, int, int]]  # 边列表，(u, v, capacity)
s: int           # 源点
t: int           # 汇点
```

**输出**:
```python
max_flow: int    # 最大流量
```

## 示例

```python
n = 4
edges = [(0, 1, 3), (0, 2, 2), (1, 2, 2), (1, 3, 3), (2, 3, 2)]
s, t = 0, 3
flow = max_flow(n, edges, s, t)
# 期望输出: 5
```

图示:
```
    3
  0→→1
  ↓ ↗↓ 3
  2↘ ↓
    →→3
    2
```

## 数据范围

| 参数 | 最小值 | 最大值 |
|------|--------|--------|
| 节点数n | 2 | 10000 |
| 边数m | 1 | 100000 |
| 容量 | 1 | 10^9 |
| 测试用例数 | - | 10 |

## 限制条件

| 限制项 | 要求 |
|--------|------|
| 时间限制 | 1秒/用例 |
| 内存限制 | 256MB |
| 语言 | Python 3.8+ |
| 禁止使用 | networkx等图库 |

## 评分标准

| 维度 | 分数 | 说明 |
|------|------|------|
| 正确性 | 50分 | 全部通过得满分 |
| 时间 | 30分 | <0.5s满分，>1s扣完 |
| 内存 | 20分 | <128MB满分 |

## 参考答案

### Edmonds-Karp (BFS增广)
```python
from collections import deque

def max_flow(n, edges, s, t):
    # 构建残量图
    cap = [[0]*n for _ in range(n)]
    for u, v, c in edges:
        cap[u][v] += c
    
    flow = 0
    while True:
        # BFS找增广路径
        parent = [-1]*n
        parent[s] = s
        q = deque([s])
        while q and parent[t] == -1:
            u = q.popleft()
            for v in range(n):
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    q.append(v)
        
        if parent[t] == -1:
            break
        
        # 计算瓶颈容量
        path_flow = float('inf')
        v = t
        while v != s:
            u = parent[v]
            path_flow = min(path_flow, cap[u][v])
            v = u
        
        # 更新残量
        v = t
        while v != s:
            u = parent[v]
            cap[u][v] -= path_flow
            cap[v][u] += path_flow
            v = u
        
        flow += path_flow
    
    return flow
```

### Dinic算法 (优化版)
```python
from collections import deque

def max_flow(n, edges, s, t):
    # 邻接表存储
    graph = [[] for _ in range(n)]
    edge_list = []
    for u, v, c in edges:
        edge_list.extend([c, 0])  # [正向容量, 反向容量]
        graph[u].append([v, len(edge_list)-2])
        graph[v].append([u, len(edge_list)-1])
    
    flow = 0
    while True:
        # BFS分层
        level = [-1]*n
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v, i in graph[u]:
                if level[v] == -1 and edge_list[i] > 0:
                    level[v] = level[u] + 1
                    q.append(v)
        
        if level[t] == -1:
            break
        
        # DFS多路增广
        it = [0]*n
        def dfs(u, f):
            if u == t:
                return f
            for j in range(it[u], len(graph[u])):
                it[u] = j
                v, i = graph[u][j]
                if level[v] == level[u] + 1 and edge_list[i] > 0:
                    ret = dfs(v, min(f, edge_list[i]))
                    if ret:
                        edge_list[i] -= ret
                        edge_list[i^1] += ret
                        return ret
            return 0
        
        while True:
            f = dfs(s, float('inf'))
            if not f:
                break
            flow += f
    
    return flow
```

### ISAP算法 (最快)
```python
def max_flow(n, edges, s, t):
    # 初始化邻接表
    head = [-1]*n
    to, cap, nxt = [], [], []
    def add_edge(u, v, c):
        to.append(v); cap.append(c); nxt.append(head[u]); head[u] = len(to)-1
        to.append(u); cap.append(0); nxt.append(head[v]); head[v] = len(to)-1
    for u, v, c in edges:
        add_edge(u, v, c)
        add_edge(v, u, 0) if False else None  # 反向边已添加
    
    # ISAP主循环
    level = [0]*n
    gap = [0]*(n+1)
    flow = 0
    
    def bfs():
        level[:] = [n]*n
        level[t] = 0
        q = [t]
        for u in q:
            for i in range(head[u], -1, -1):
                if i % 2 == 0: continue
                v = to[i]
                if level[v] == n:
                    level[v] = level[u] + 1
                    q.append(v)
    
    def dfs(u, f):
        if u == t:
            return f
        ret = 0
        i = head[u]
        while i != -1:
            v = to[i]
            if cap[i] > 0 and level[v] == level[u] - 1:
                tmp = dfs(v, min(f-ret, cap[i]))
                cap[i] -= tmp
                cap[i^1] += tmp
                ret += tmp
                if ret == f:
                    return ret
            i = nxt[i]
        gap[level[u]] -= 1
        if gap[level[u]] == 0:
            level[s] = n
        level[u] += 1
        gap[level[u]] += 1
        return ret
    
    bfs()
    for i in range(n):
        if level[i] < n:
            gap[level[i]] += 1
    
    while level[s] < n:
        flow += dfs(s, float('inf'))
    
    return flow
```

## 优化提示

1. **算法选择**:
   - 小图(<100): Edmonds-Karp
   - 中图(100-1000): Dinic
   - 大图(>1000): ISAP或Push-Relabel

2. **数据结构**:
   - 使用邻接表代替邻接矩阵
   - 链式前向星存储边
   - 当前弧优化避免重复访问

3. **实现技巧**:
   - 多路增广减少BFS次数
   - Gap优化提前终止
   - 使用数组代替字典

4. **Python特定优化**:
   - 使用局部变量缓存
   - 避免递归(用栈模拟)
   - 考虑PyPy加速

## 测试用例

### 测试1: 简单图
```python
def test_simple():
    n = 4
    edges = [(0,1,3),(0,2,2),(1,2,2),(1,3,3),(2,3,2)]
    assert max_flow(n, edges, 0, 3) == 5
```

### 测试2: 无路径
```python
def test_no_path():
    n = 3
    edges = [(0, 1, 5)]
    assert max_flow(n, edges, 0, 2) == 0
```

### 测试3: 多重边
```python
def test_multi_edges():
    n = 3
    edges = [(0,1,2),(0,1,3),(1,2,4),(1,2,1)]
    assert max_flow(n, edges, 0, 2) == 5
```

### 测试4: 大规模随机图
```python
import random

def test_large():
    random.seed(42)
    n = 1000
    m = 10000
    edges = []
    for _ in range(m):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        c = random.randint(1, 1000)
        edges.append((u, v, c))
    
    import time
    start = time.time()
    result = max_flow(n, edges, 0, n-1)
    elapsed = time.time() - start
    assert elapsed < 1.0
```
