# ROF与Mumford-Shah学习路线图

## 📅 Week 1: ROF去噪 (快速入门)

### Day 1-2: 理解基础概念
- [ ] **数学基础**
  - 阅读ROF原始论文第1-3节（Rudin, Osher, Fatemi, 1992）
  - 理解Total Variation (TV) 半范数的定义
  - 理解为什么TV可以保持边缘

- [ ] **运行快速演示**
  ```bash
  python rof_denoise_quickstart.py
  ```

- [ ] **关键公式理解**
  ```
  ROF能量泛函:
  min_u ∫_Ω |∇u| dx + λ/2 ∫_Ω (u - f)² dx

  其中:
  - u: 去噪后的图像
  - f: 原始含噪图像
  - |∇u|: 全变分项（保持边缘）
  - (u-f)²: 数据保真项（保持相似性）
  - λ: 平衡参数
  ```

### Day 3-4: 深入算法实现
- [ ] **学习Chambolle投影算法**
  - 阅读: [IPOL教程](https://www.ipol.im/pub/art/2013/61/)
  - 理解对偶问题的引入
  - 手动实现Chambolle算法（不依赖scikit-image）

- [ ] **算法对比**
  - 梯度下降法（慢但易懂）
  - Chambolle投影算法（快且稳定）
  - Split Bregman方法（适合复杂问题）

### Day 5-7: 实验与分析
- [ ] **参数调优实验**
  - 测试不同λ值的影响: [0.01, 0.05, 0.1, 0.2, 0.5]
  - 观察边缘保持效果
  - 对比高斯滤波（为什么ROF更好）

- [ ] **扩展实验**
  - 彩色图像去噪
  - 不同噪声水平
  - 真实噪声图像

- [ ] **Week 1总结**
  - 写一份实验报告
  - 记录观察到的现象
  - 思考TV去噪的局限性（阶梯效应）

---

## 📅 Week 2-3: Chan-Vese活动轮廓模型

### 为什么学习Chan-Vese?
- Chan-Vese是Mumford-Shah的**简化版本**（无边缘检测器）
- 数学上更简单，但保留了核心思想
- 可以理解为：基于区域的活动轮廓模型

### 学习路径
- [ ] **理论理解**
  - Chan-Vese能量泛函
  - 水平集方法 (Level Set Method)
  - 演化方程

- [ ] **代码实现**
  - 实现基本的Chan-Vese分割
  - 使用scikit-image的`segmentation.chan_vese`
  - 测试合成图像（圆形、方形等）

- [ ] **实验验证**
  - 测试不同形状的目标
  - 观察初始化的影响
  - 对比传统边缘检测（Canny）

### 参考资源
- IPOL Chan-Vese教程: https://www.ipol.im/pub/art/2012/g-cv/
- 原始论文: Chan, Vese (2001) "Active contours without edges"

---

## 📅 Week 4-6: 完整Mumford-Shah模型

### 从简化版到完整版
```
Chan-Vese (无边缘) → Mumford-Shah (有边缘检测器)
```

### Mumford-Shah泛函
```
E(u, Γ) = ∫_Ω\Γ |∇u|² dx + μ∫_Ω\Γ (u-f)² dx + ν|Γ|

其中:
- u: 分段光滑的近似图像
- Γ: 边缘集合（不连续点集）
- 第一项: u的光滑性
- 第二项: u与f的保真度
- 第三项: 边缘长度（正则化）
```

### 实现策略
1. **凸松弛方法** (推荐)
   - 使用[2-01]论文的方法
   - 引入松弛变量
   - 转化为凸优化问题

2. **图割方法**
   - 将连续问题离散化
   - 使用max-flow/min-cut算法

3. **Split Bregman方法**
   - 适合大规模问题
   - 收敛速度快

### 实验项目
- [ ] 在Berkeley分割数据集上测试
- [ ] 对比多种方法（Chan-Vese, 图割, 深度学习）
- [ ] 分析各方法的优缺点

---

## 📅 Week 7-8: 应用扩展

### 应用1: 小波框架血管分割 [2-08]
```
理论方法 → 医学应用
```

- [ ] 下载DRIVE数据集
- [ ] 实现小波框架分割
- [ ] 评估指标：Sensitivity, Specificity, AUC

### 应用2: 3D分割（可选）
- 扩展到3D医学图像
- 理解3D中的TV定义

---

## 🎯 学习成果验收

### 理论掌握
- [ ] 能够推导ROF能量泛函
- [ ] 理解TV正则化的几何意义
- [ ] 掌握凸松弛的基本思想

### 编程能力
- [ ] 不依赖库实现ROF去噪
- [ ] 实现Chan-Vese分割
- [ ] 理解并能修改现有算法

### 应用能力
- [ ] 能够处理真实图像
- [ ] 能够调整参数适应不同场景
- [ ] 理解算法的适用范围

---

## 📚 推荐阅读顺序

### 必读论文
1. **[1-04] 变分法基础 Mumford-Shah与ROF** (你的PDF)
   - 第1-3章：基础理论
   - 第4-5章：算法实现

2. **Rudin-Osher-Fatemi (1992)** - ROF原始论文
   - Physica D journal
   - 可在Google Scholar找到

3. **Chan-Vese (2001)** - 简化版Mumford-Shah
   - IEEE Trans. Image Processing

4. **[2-01] 凸优化分割 Convex Mumford-Shah** (你的PDF)
   - 现代优化方法

### 推荐书籍
- "Variational Methods in Image Segmentation" by Jean-Michel Morel
- "Image Processing and Analysis" by Tony Chan

---

## 💡 常见问题

### Q1: ROF为什么能保持边缘？
**A**: TV半范数 |∇u| 允许不连续性（边缘存在时|∇u|大但积分有限），而Tikhonov正则化 |∇u|² 惩罚所有梯度，会平滑边缘。

### Q2: λ参数如何选择？
**A**:
- λ小 → 去噪弱，保留细节
- λ大 → 去噪强，但可能过度平滑
- 经验值: 0.1-0.3（根据噪声水平调整）

### Q3: 什么时候ROF不适用？
**A**:
- 纹理丰富的图像（TV会过度平滑纹理）
- 需要非常精细的边缘（有阶梯效应）
- 实时性要求高的场景（迭代算法慢）

---

## 🚀 下一步：深度学习结合

完成上述学习后，可以探索：
- 将TV正则化引入深度学习损失函数
- 学习深度学习图像去噪（DnCNN, UNet等）
- 理解为什么深度学习可以学习隐式的正则化

---

**开始你的学习之旅吧！** 🎉

有问题随时问我！
