# 第十五讲：点云处理基础

## Point Cloud Processing Fundamentals

---

### 📋 本讲大纲

1. 点云数据概述
2. 点云采样与滤波
3. 点云配准
4. 特征提取
5. 应用场景

---

### 15.1 点云数据

#### 定义

点云是三维空间中点的集合：
$$\mathcal{P} = \{p_i = (x_i, y_i, z_i)\}_{i=1}^N$$

可能包含属性：颜色、法向量、强度等

#### 数据来源

| 设备 | 原理 | 特点 |
|------|------|------|
| LiDAR | 激光测距 | 高精度，远距离 |
| RGB-D相机 | 结构光/ToF | 实时，近距离 |
| 双目视觉 | 视差计算 | 被动，低成本 |
| 激光扫描仪 | 机械旋转 | 高精度，慢 |

---

### 15.2 点云挑战

#### 技术挑战

```
• 数据量大（百万级点）
• 噪声和离群点
• 不均匀采样
• 遮挡和缺失
• 无序性（排列不变）
```

**动画建议**：展示典型点云的噪声和不完整性

---

### 15.3 点云采样

#### 下采样方法

**体素网格采样**：
1. 将空间划分为体素网格
2. 每个体素保留一个代表点（中心点或均值）

**随机采样**：
$$\mathcal{P}' = \text{RandomSample}(\mathcal{P}, k)$$

**最远点采样(FPS)**：
```
初始化：选择一个起点
repeat
  选择距离已选点最远的点
until 选够k个点
```

---

### 15.4 点云滤波

#### 统计滤波

**统计离群点移除(SOR)**：
- 计算每个点到k近邻的平均距离
- 移除距离超过 $\mu + \alpha \sigma$ 的点

#### 半径滤波

移除指定半径内邻居少于阈值的点

#### 双边滤波

$$p'_i = p_i + \frac{1}{W}\sum_{j \in N(i)} (p_j - p_i) \cdot w_d(\|p_i - p_j\|) \cdot w_n(\|\mathbf{n}_i - \mathbf{n}_j\|)$$

---

### 15.5 点云配准

#### 问题定义

给定源点云 $\mathcal{P}$ 和目标点云 $\mathcal{Q}$，求变换 $T$ 使得：
$$T^* = \arg\min_T \sum_i \|T(p_i) - q_{\pi(i)}\|^2$$

其中 $\pi$ 是对应关系

#### 刚体变换

$$T(p) = Rp + t$$

- $R \in SO(3)$：旋转矩阵
- $t \in \mathbb{R}^3$：平移向量

---

### 15.6 ICP算法

#### Iterative Closest Point

```
初始化：T = I（单位变换）
repeat
  1. 对应：对于每个p_i，找q中最近点
  2. 变换：计算最优R, t使Σ||R·p_i + t - q_i||²最小
  3. 更新：应用变换
until 收敛
```

#### 最优变换求解

使用SVD：
1. 中心化：$\hat{p} = p - \bar{p}$, $\hat{q} = q - \bar{q}$
2. 计算协方差：$H = \hat{P}\hat{Q}^T$
3. SVD：$H = U\Sigma V^T$
4. 旋转：$R = VU^T$
5. 平移：$t = \bar{q} - R\bar{p}$

---

### 15.7 配准变体

#### 点到面ICP

$$\min_T \sum_i \left((T(p_i) - q_i) \cdot \mathbf{n}_i\right)^2$$

#### 彩色ICP

结合几何和颜色信息

#### 全局配准

- FPFH特征匹配
- RANSAC验证
- 避免局部最优

---

### 15.8 特征提取

#### 法向量估计

使用PCA：
1. 找k近邻
2. 构建协方差矩阵
3. 最小特征值对应特征向量为法向量

$$C = \frac{1}{k}\sum_{j \in N(i)}(p_j - \bar{p})(p_j - \bar{p})^T$$

#### 局部描述符

| 描述符 | 维度 | 特点 |
|--------|------|------|
| FPFH | 33 | 快速，配准 |
| SHOT | 352 | 高精度 |
| 3DMatch | 512 | 学习方法 |

---

### 15.9 点云分割

#### 平面分割（RANSAC）

```
repeat
  1. 随机选3个点，拟合平面
  2. 计算内点数（距离<阈值）
  3. 保留内点最多的平面
until 达到迭代次数
```

#### 区域生长

基于法向量/曲率一致性

#### 深度学习方法

- PointNet
- PointNet++
- DGCNN

---

### 15.10 点云重建

#### 泊松重建

求解泊松方程：
$$\Delta \chi = \nabla \cdot \vec{V}$$

其中 $\vec{V}$ 是指示函数的梯度场

#### Marching Cubes

从隐式函数提取等值面

#### Delaunay三角剖分

基于点的三角剖分

---

### 15.11 应用场景

| 应用 | 描述 |
|------|------|
| 自动驾驶 | 环境感知、障碍物检测 |
| 机器人 | 导航、抓取 |
| 测绘 | 地形建模 |
| 建筑BIM | 三维重建 |
| 文化遗产 | 数字化保存 |

---

### 15.12 Cai的点云工作

#### 研究方向

- 点云分割的变分方法
- 图割优化
- 神经变分(Neural Varifolds)

#### 相关贡献

- 点云上的框架变换
- 图优化方法

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│           点云处理核心                           │
├─────────────────────────────────────────────────┤
│                                                 │
│   基本操作：                                     │
│   • 采样：体素网格、FPS                         │
│   • 滤波：统计滤波、双边滤波                     │
│   • 配准：ICP及其变体                           │
│                                                 │
│   特征提取：                                     │
│   • 法向量：PCA                                 │
│   • 描述符：FPFH、SHOT                          │
│                                                 │
│   分割与重建：                                   │
│   • RANSAC、区域生长                            │
│   • 泊松重建、Marching Cubes                    │
│                                                 │
│   深度学习：PointNet系列                        │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **实现题**：实现FPS下采样算法

2. **实现题**：实现ICP点云配准

3. **应用题**：使用Open3D进行点云滤波和配准

4. **分析题**：比较不同下采样方法的效果

---

### 📖 扩展阅读

1. **教材**：
   - Rusu, "Semantic 3D Object Maps for Everyday Manipulation"
   
2. **软件工具**：
   - Open3D (Python/C++)
   - PCL (C++)
   - PointNet++ (PyTorch)

3. **论文**：
   - Besl & McKay, "A Method for Registration of 3-D Shapes", PAMI, 1992

---

### 📖 参考文献

1. Besl, P.J. & McKay, N.D. (1992). A method for registration of 3-D shapes. *IEEE PAMI*, 14(2), 239-256.

2. Rusu, R.B., et al. (2009). Fast point feature histograms (FPFH) for 3D registration. *ICRA*.

3. Qi, C.R., et al. (2017). PointNet: Deep learning on point sets for 3D classification and segmentation. *CVPR*.

4. Kazhdan, M., et al. (2006). Poisson surface reconstruction. *SGP*.
