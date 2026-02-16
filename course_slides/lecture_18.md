# 第十八讲：3D目标检测

## 3D Object Detection

---

### 📋 本讲大纲

1. 3D目标检测概述
2. 点云检测方法
3. 跨域适应方法
4. 评估指标与数据集
5. 前沿进展

---

### 18.1 任务定义

#### 问题

给定点云 $\mathcal{P} = \{p_i\}$，输出：
$$\{(B_k, c_k, s_k)\}_{k=1}^K$$

- $B_k$：3D边界框（位置、尺寸、朝向）
- $c_k$：类别
- $s_k$：置信度

#### 挑战

```
• 点云稀疏性
• 遮挡与截断
• 类别不平衡
• 实时性要求
```

---

### 18.2 3D边界框表示

#### 参数化

$$B = (x, y, z, l, w, h, \theta)$$

| 参数 | 含义 |
|------|------|
| $(x, y, z)$ | 中心位置 |
| $(l, w, h)$ | 长、宽、高 |
| $\theta$ | 偏航角 |

#### 8角点表示

$$Corners = f(B) \in \mathbb{R}^{8 \times 3}$$

---

### 18.3 体素化方法

#### VoxelNet (2017)

```
点云 → 体素化 → 3D卷积 → 检测头
```

#### PointPillars

```
点云 → 柱状体素 → 2D卷积 → 检测头
```

- 更快的推理速度
- 适用于自动驾驶

**动画建议**：展示体素化过程

---

### 18.4 点云方法

#### PointRCNN (2019)

两阶段检测：
1. 前景点分割 + 边界框生成
2. 特征精炼

#### 3DSSD

单阶段检测：
- 下采样保留前景点
- 预测边界框

#### CenterPoint

中心点检测：
- 点云 → BEV特征图
- 检测中心点 → 边界框

---

### 18.5 多传感器融合

#### Late Fusion

各传感器独立检测 → 结果融合

#### Early Fusion

原始数据融合 → 统一检测

#### BEVFusion

```
相机图像 → BEV特征
              ↘
               → 融合 → 检测
              ↗
LiDAR点云 → BEV特征
```

---

### 18.6 跨域适应

#### 问题

训练域（源域）与测试域（目标域）分布不同

```
源域：晴天数据
目标域：雨天数据
```

#### 域差异来源

- 天气变化
- 传感器差异
- 地理位置差异

---

### 18.7 域适应方法

#### 特征对齐

最小化源域和目标域特征分布差异：
$$\mathcal{L}_{DA} = \text{MMD}(F_s, F_t)$$

#### 自训练

```
1. 源域预训练
2. 目标域伪标签生成
3. 自监督微调
```

#### ST3D

自训练框架：
- 3D目标检测器预训练
- 目标域伪标签
- 渐进式更新

---

### 18.8 跨传感器适应

#### LiDAR → LiDAR

不同型号激光雷达之间的适应

#### 相机 → LiDAR

利用图像信息辅助点云检测

#### 跨模态蒸馏

用教师模型（源域）指导学生模型（目标域）

---

### 18.9 评估指标

#### 平均精度(AP)

$$AP = \int_0^1 P(R) dR$$

#### 3D IoU

$$IoU = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}$$

#### 指标

- AP@0.5
- AP@0.7
- mAP (多类别平均)

---

### 18.10 数据集

#### KITTI

- 自动驾驶场景
- 3类：Car, Pedestrian, Cyclist
- 标准benchmark

#### Waymo Open Dataset

- 大规模
- 多传感器
- 多样场景

#### nuScenes

- 全场景3D检测
- 23类物体
- 360°覆盖

#### ONCE

- 百万级标注帧
- 自动驾驶数据

---

### 18.11 实时检测

#### 速度要求

| 应用 | 延迟要求 |
|------|----------|
| 自动驾驶 | < 100ms |
| 机器人 | < 200ms |
| 离线分析 | 无限制 |

#### 加速技术

- TensorRT
- 量化
- 模型剪枝

---

### 18.12 未来方向

- 更好的泛化能力
- 长尾类别检测
- 时序信息利用
- 可解释性

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           3D目标检测核心                         │
├─────────────────────────────────────────────────┤
│                                                 │
│   方法分类：                                     │
│   • 体素化：VoxelNet, PointPillars              │
│   • 点云：PointRCNN, CenterPoint               │
│   • 融合：BEVFusion                            │
│                                                 │
│   跨域适应：                                     │
│   • 特征对齐                                    │
│   • 自训练                                      │
│   • 跨模态蒸馏                                  │
│                                                 │
│   评估：AP@IoU阈值                             │
│                                                 │
│   数据集：KITTI, Waymo, nuScenes               │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **实现题**：实现3D IoU计算

2. **分析题**：比较体素方法和点云方法的优缺点

3. **实验题**：在KITTI数据集上评估检测器性能

4. **研究题**：调研最新的跨域适应方法

---

### 📖 扩展阅读

1. **经典论文**：
   - Zhou & Tuzel, "VoxelNet", CVPR 2018
   - Lang et al., "PointPillars", CVPR 2019
   - Yin et al., "Center-based 3D Object Detection", CVPR 2021

2. **代码库**：
   - OpenPCDet
   - mmdetection3d

3. **数据集**：
   - KITTI 3D Object Detection Benchmark
   - Waymo Open Dataset

---

### 📖 参考文献

1. Zhou, Y. & Tuzel, O. (2018). VoxelNet: End-to-end learning for point cloud based 3D object detection. *CVPR*.

2. Lang, A.H., et al. (2019). PointPillars: Fast encoders for object detection from point clouds. *CVPR*.

3. Yin, T., et al. (2021). Center-based 3D object detection and tracking. *CVPR*.

4. Yang, Z., et al. (2021). ST3D: Self-training for unsupervised domain adaptation on 3D object detection. *CVPR*.
