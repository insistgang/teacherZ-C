---
title: 凸松弛 (Convex Relaxation)
tags: [变分, 优化, 凸优化]
related: [total_variation, graph_cut]
---

## 核心思想

将非凸优化问题松弛为凸问题，保证全局最优解。

## 典型方法

### 1. 区域竞争的凸松弛

原问题：$\min_C E(C)$ （组合优化）

松弛后：$\min_u E(u)$ 其中 $u \in [0,1]$

### 2. 全局凸松弛分割

Chan-Vese模型的凸松弛：

$$\min_u \int_\Omega |Du| + \lambda \int_\Omega (c_1 - c_2)(u - f) dx$$

### 3. 多标签凸松弛

Potts模型的松弛：

$$\min_{u_i} \sum_i \int_\Omega |Du_i| + \sum_i \int_\Omega \rho_i(x) u_i(x) dx$$

约束：$u_i \geq 0, \sum_i u_i = 1$

## 求解算法

| 算法 | 类型 | 收敛性 |
|------|------|--------|
| Primal-Dual | 一阶方法 | $O(1/n)$ |
| ADMM | 乘子法 | $O(1/n^2)$ |
| Split Bregman | 迭代收缩 | 线性收敛 |

## 阈值化

松弛解 $u^* \in [0,1]$ 可以通过阈值化得到二值解：

$$u_{binary} = \begin{cases} 1 & u^* > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

## 相关链接

- [[graph_cut]] - 图割
- [[total_variation]] - 全变分
- [[slat_segmentation]] - SLaT分割
