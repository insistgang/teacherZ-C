---
title: 欧拉-拉格朗日方程
tags: [变分, 优化, 数学基础]
related: [total_variation, mumford_shah]
---

## 定义

对于泛函 $J(u) = \int_\Omega L(x, u, \nabla u) dx$，其极值点满足：

$$\frac{\partial L}{\partial u} - \nabla \cdot \frac{\partial L}{\partial \nabla u} = 0$$

## 推导示例

### ROF模型

能量：$E(u) = \frac{1}{2}\|u-f\|^2 + \lambda TV(u)$

欧拉-拉格朗日方程：

$$u - f - \lambda \nabla \cdot \left(\frac{\nabla u}{|\nabla u|}\right) = 0$$

正则化后：

$$u - f - \lambda \nabla \cdot \left(\frac{\nabla u}{\sqrt{|\nabla u|^2 + \epsilon^2}}\right) = 0$$

## 数值求解

### 梯度下降

$$\frac{\partial u}{\partial t} = f - u + \lambda \nabla \cdot \left(\frac{\nabla u}{|\nabla u|_\epsilon}\right)$$

### 离散化

$$u^{n+1}_{ij} = u^n_{ij} + \Delta t \left[ f_{ij} - u^n_{ij} + \lambda \cdot div\left(\frac{\nabla u^n}{|\nabla u^n|_\epsilon}\right)\right]$$

## 边界条件

- **Dirichlet**: $u|_{\partial\Omega} = g$
- **Neumann**: $\frac{\partial u}{\partial n}|_{\partial\Omega} = 0$

## 相关链接

- [[total_variation]] - 全变分
- [[rof_denoising]] - ROF去噪
