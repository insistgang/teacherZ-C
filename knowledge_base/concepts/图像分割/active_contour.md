---
title: 活动轮廓 (Active Contour)
tags: [分割, 变分, 曲线演化]
related: [level_set, chan_vese, mumford_shah]
---

## 定义

活动轮廓（Snake模型）通过能量最小化演化曲线实现分割。

## 经典Snake模型

$$E_{snake} = \int_0^1 \left[ E_{int}(v(s)) + E_{ext}(v(s)) \right] ds$$

### 内部能量

$$E_{int} = \alpha |v'(s)|^2 + \beta |v''(s)|^2$$

- 第一项：弹性（保持连续）
- 第二项：刚度（保持平滑）

### 外部能量

$$E_{ext} = -|\nabla I(x,y)|^2$$

吸引曲线到图像边缘。

## 几何活动轮廓

使用水平集隐式表示：

$$\frac{\partial \phi}{\partial t} = |\nabla \phi| \cdot v(\phi, I)$$

优势：
- 自动处理拓扑变化
- 不需要参数化

## Chan-Vese模型

基于区域的水平集方法：

$$\frac{\partial \phi}{\partial t} = \delta(\phi) \left[ \lambda_1(f-c_1)^2 - \lambda_2(f-c_2)^2 + \nu \right]$$

## 求解方法

1. 显式差分
2. 半隐式差分
3. 加水平集正则化

## 相关链接

- [[level_set]] - 水平集方法
- [[chan_vese]] - Chan-Vese模型
- [[mumford_shah]] - Mumford-Shah泛函
