---
title: 水平集方法 (Level Set)
tags: [分割, 曲线演化, 隐式表示]
related: [active_contour, chan_vese]
---

## 核心思想

将曲线$C$表示为更高维函数的零等值线：

$$C(t) = \{x : \phi(x, t) = 0\}$$

## 演化方程

曲线沿法向$N$以速度$v$演化：

$$\frac{\partial \phi}{\partial t} + v|\nabla \phi| = 0$$

## 初始化

通常用符号距离函数(SDF)：

$$\phi(x) = \begin{cases}
+d(x, C) & x \in \text{inside} \\
-d(x, C) & x \in \text{outside}
\end{cases}$$

## 优势

| 特性 | 说明 |
|------|------|
| 拓扑变化 | 自动处理分裂和合并 |
| 几何性质 | 曲率、法向易于计算 |
| 数值稳定 | 可用隐式格式 |

## 重新初始化

演化过程中$\phi$会偏离SDF，需定期重新初始化：

$$\frac{\partial \phi}{\partial t} = \text{sign}(\phi)(1-|\nabla \phi|)$$

## 变体

### 窄带水平集
只在曲线附近更新，降低复杂度 $O(N)$ → $O(k\sqrt{N})$

### 多相水平集
用多个$\phi$表示多区域分割

## 相关链接

- [[active_contour]] - 活动轮廓
- [[chan_vese]] - Chan-Vese模型
