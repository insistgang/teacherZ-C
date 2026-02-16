---
title: Mumford-Shah泛函
tags: [变分, 分割, 能量泛函]
related: [active_contour, chan_vese]
---

## 定义

Mumford-Shah泛函是图像分割的经典能量函数：

$$E(u, C) = \alpha \int_\Omega (u - f)^2 dx + \beta \int_{\Omega \setminus C} |\nabla u|^2 dx + \gamma |C|$$

其中：
- $f$: 原始图像
- $u$: 分割后的近似图像
- $C$: 边缘曲线
- $|C|$: 边缘长度

## 三项含义

1. **保真项**: $\int (u-f)^2$ - 保持与原图相似
2. **平滑项**: $\int |\nabla u|^2$ - 区域内平滑
3. **边缘项**: $|C|$ - 边缘简洁

## 简化版本

### Chan-Vese模型
当 $\beta \to \infty$ 时，区域内变为常数：

$$E(C) = \int_{inside(C)}(f-c_1)^2 + \int_{outside(C)}(f-c_2)^2 + \gamma |C|$$

## 求解难点

- 泛函定义在不同空间上
- 边缘$C$的表示困难
- 非凸优化问题

## 相关链接

- [[active_contour]] - 活动轮廓
- [[chan_vese]] - Chan-Vese模型
- [[level_set]] - 水平集方法
