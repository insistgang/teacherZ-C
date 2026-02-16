---
title: 图割 (Graph Cut)
tags: [分割, 组合优化, 能量最小化]
related: [convex_relaxation, watershed]
---

## 核心思想

将图像分割建模为图上的能量最小化问题。

## 图构造

- **节点**: 像素 + 源点S + 汇点T
- **边**: 
  - n-links: 相邻像素间（平滑项）
  - t-links: 像素到S/T（数据项）

## 能量函数

$$E(L) = \sum_p D_p(L_p) + \sum_{p,q \in N} V_{p,q}(L_p, L_q)$$

### 数据项 $D_p$

$$D_p(L_p) = -\log P(I_p | L_p)$$

### 平滑项 $V_{p,q}$

$$V_{p,q} = \lambda \cdot \exp\left(-\frac{|I_p - I_q|^2}{2\sigma^2}\right) \cdot [L_p \neq L_q]$$

## 最大流/最小割

全局最优解可通过最大流算法求得：
- Ford-Fulkerson
- Push-Relabel
- Boykov-Kolmogorov (图像专用)

## 扩展

### GrabCut
迭代优化 + GMM颜色模型

### Multi-label Graph Cut
- $\alpha$-expansion
- $\alpha$-$\beta$-swap

## 优缺点

| 优点 | 缺点 |
|------|------|
| 全局最优 | 内存消耗大 |
| 速度快 | 硬约束难处理 |
| 可加入先验 | 边界有网格伪影 |

## 相关链接

- [[convex_relaxation]] - 凸松弛
- [[watershed]] - 分水岭
