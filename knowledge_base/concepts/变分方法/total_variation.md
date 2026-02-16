---
title: 全变分 (Total Variation)
tags: [变分, 正则化, 图像去噪]
related: [rof_denoising, euler_lagrange]
---

## 定义

全变分是图像梯度的L1范数：

$$TV(u) = \int_\Omega |\nabla u| dx = \int_\Omega \sqrt{u_x^2 + u_y^2} dx$$

## 特性

- **边缘保持**: 不会过度平滑边缘
- **分段常数**: 倾向于产生分片常数解
- **凸性**: 优化问题是凸的

## 变体

| 变体 | 公式 | 特点 |
|------|------|------|
| 各向同性TV | $\sqrt{u_x^2+u_y^2}$ | 旋转不变 |
| 各向异性TV | $|u_x| + |u_y|$ | 计算更快 |
| TV-L2 | $\|Ku-f\|_2^2 + \lambda TV(u)$ | ROF模型 |

## 应用

- 图像去噪
- 图像修复
- 图像分割
- MRI重建

## 相关链接

- [[rof_denoising]] - ROF去噪模型
- [[euler_lagrange]] - 欧拉-拉格朗日方程
- [[convex_relaxation]] - 凸松弛
