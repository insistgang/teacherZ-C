# 第二十一讲：分割与检测

## Segmentation and Detection in Medical Imaging

---

### 📋 本讲大纲

1. 医学图像分割任务
2. 传统分割方法
3. 深度学习方法
4. 病灶检测
5. 临床应用

---

### 21.1 医学图像分割

#### 任务定义

将医学图像划分为有意义的区域：
- 器官分割
- 组织分割
- 病灶分割

#### 临床意义

- 体积量化
- 治疗规划
- 疾病诊断
- 手术导航

---

### 21.2 分割挑战

```
• 边界模糊
• 形状变异大
• 类别不平衡
• 标注数据少
• 多模态融合
```

---

### 21.3 传统分割方法

#### 阈值方法

OTSU、自适应阈值

#### 区域方法

区域生长、分水岭

#### 边缘方法

Canny、水平集

#### 统计方法

GMM、MRF

---

### 21.4 活动轮廓模型

#### Snake模型

$$E = \int_0^1 \left[\alpha|v'|^2 + \beta|v''|^2 + E_{ext}(v)\right] ds$$

- 内部能量：平滑性
- 外部能量：图像力

#### 水平集方法

$$\frac{\partial \phi}{\partial t} + F|\nabla \phi| = 0$$

#### Chan-Vese模型

医学图像分割的经典方法

---

### 21.5 深度学习分割

#### FCN (2015)

全卷积网络，端到端分割

#### U-Net (2015)

医学图像分割标准架构

```
编码器 → 瓶颈 → 解码器
      ↘ 跳跃连接 ↙
```

#### nnU-Net (2020)

自适应配置的U-Net

---

### 21.6 U-Net详解

#### 架构

```
输入图像
    │
    ▼
┌─────────────────┐
│ 编码器 (下采样)  │ ──┐
│  64 → 128 → 256  │   │ 跳跃
└─────────────────┘   │ 连接
         │            │
         ▼            │
┌─────────────────┐   │
│    瓶颈层 512    │   │
└─────────────────┘   │
         │            │
         ▼            │
┌─────────────────┐   │
│ 解码器 (上采样)  │ ←─┘
│ 256 ← 128 ← 64  │
└─────────────────┘
    │
    ▼
分割输出
```

#### 关键特点

- 对称结构
- 跳跃连接
- 多尺度特征融合

---

### 21.7 3D分割网络

#### 3D U-Net

扩展到3D体积数据

#### V-Net

添加残差连接

#### Attention U-Net

注意力机制聚焦关键区域

#### TransUNet

Transformer + CNN混合

---

### 21.8 多器官分割

#### 挑战

- 器官数量多
- 大小差异大
- 边界复杂

#### 方法

- 多类分割
- 级联网络
- 分层分割

#### 数据集

- Synapse Multi-organ
- BTCV (腹部器官)

---

### 21.9 病灶检测

#### 任务

定位并识别异常区域

#### 挑战

- 病灶小
- 形状不规则
- 背景复杂

#### 方法

| 方法 | 描述 |
|------|------|
| Faster R-CNN | 两阶段检测 |
| YOLO | 单阶段检测 |
| RetinaNet | Focal Loss |
| DETR | Transformer检测 |

---

### 21.10 小目标检测

#### 问题

病灶通常很小，容易被忽略

#### 解决方案

- 多尺度特征融合 (FPN)
- 注意力机制
- 数据增强

#### 专用方法

- 3D CE (上下文编码)
- 小目标专用损失

---

### 21.11 类别不平衡

#### 问题

病灶像素远少于正常像素

#### 损失函数

**Focal Loss**：
$$L_{focal} = -\alpha(1-p_t)^\gamma \log(p_t)$$

**Dice Loss**：
$$L_{dice} = 1 - \frac{2|P \cap G|}{|P| + |G|}$$

**Tversky Loss**：
$$L_{Tversky} = 1 - \frac{TP}{TP + \alpha FP + \beta FN}$$

---

### 21.12 临床应用

#### 脑部分割

- 白质/灰质/脑脊液
- 脑肿瘤分割
- 脑卒中检测

#### 心脏分割

- 心室分割
- 心功能评估
- 冠脉提取

#### 肺部分割

- 肺结节检测
- COVID-19诊断
- 肺气肿量化

---

### 📊 本讲总结

```
┌─────────────────────────────────────────────────┐
│       医学图像分割与检测                         │
├─────────────────────────────────────────────────┤
│                                                 │
│   分割方法：                                     │
│   • 传统：阈值、水平集、MRF                     │
│   • 深度学习：U-Net、nnU-Net                    │
│   • 3D：3D U-Net、V-Net                        │
│                                                 │
│   检测方法：                                     │
│   • 通用：Faster R-CNN、YOLO                   │
│   • 医学专用：小目标检测、3D检测                │
│                                                 │
│   挑战：                                        │
│   • 类别不平衡 → Dice/Focal Loss               │
│   • 小目标 → FPN、注意力                        │
│   • 数据少 → 迁移学习、数据增强                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

### 📚 课后作业

1. **实现题**：实现U-Net进行肝脏分割

2. **分析题**：比较Dice Loss和交叉熵损失的效果

3. **实验题**：使用nnU-Net训练分割模型

4. **研究题**：调研最新的医学图像分割方法

---

### 📖 扩展阅读

1. **经典论文**：
   - Ronneberger et al., "U-Net", MICCAI 2015
   - Isensee et al., "nnU-Net", Nature Methods 2021

2. **挑战赛**：
   - BraTS (脑肿瘤)
   - LiTS (肝脏肿瘤)
   - KiTS (肾脏肿瘤)

3. **代码库**：
   - MONAI
   - nnU-Net
   - segmentation_models.pytorch

---

### 📖 参考文献

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.

2. Isensee, F., et al. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203-211.

3. Milletari, F., Navab, N., & Ahmadi, S.A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. *3DV*.

4. Lin, T.Y., et al. (2017). Focal loss for dense object detection. *ICCV*.
