---
title: 分水岭 (Watershed)
tags: [分割, 形态学, 标记驱动]
related: [graph_cut, active_contour]
---

## 核心思想

将图像视为地形，水从局部极小值开始上涨，不同集水盆的边界即为分割线。

## 基于浸没的算法

1. 将像素按灰度排序
2. 从极小值开始逐层"淹没"
3. 当不同区域相遇时构建坝（分割边界）

## 标记控制的分水岭

避免过分割的关键：

```python
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# 生成标记
markers = peak_local_max(distance, labels=mask)

# 分水岭
labels = watershed(-distance, markers, mask=image)
```

## 变体

| 方法 | 特点 |
|------|------|
| Meyer算法 | 基于队列的高效实现 |
| Topological Watershed | 拓扑学视角 |
| Waterfall | 层级分割 |

## 过分割问题

分水岭容易产生过多区域，解决方案：

1. **标记预滤波**: 形态学开闭运算
2. **合并后处理**: 区域合并
3. **梯度修正**: 先平滑梯度图

## 与其他方法对比

| 方法 | 速度 | 过分割 | 需要标记 |
|------|------|--------|----------|
| Watershed | 快 | 严重 | 是 |
| Graph Cut | 中 | 无 | 可选 |
| Level Set | 慢 | 无 | 否 |

## 相关链接

- [[graph_cut]] - 图割
- [[active_contour]] - 活动轮廓
