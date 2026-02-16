# 52周论文精读计划

## 概述

本计划为期52周，分四个季度系统精读论文，构建从数学基础到前沿应用的完整知识体系。

---

## 第一季度: 数学基础 (1-13周)

### 第1-4周: 变分方法

## 第1周: 泛函分析与变分法基础

### 本周目标
- 理解Banach空间与Hilbert空间基本概念
- 掌握泛函极值问题的数学表述
- 理解变分法的基本思想

### 忽读论文
1. Gelfand & Fomin - Calculus of Variations (Ch1-2) - 60页
2. Struwe - Variational Methods (Ch1) - 40页

### 选读论文
1. Evans - Partial Differential Equations (Ch8) - 补充PDE背景

### 实践任务
- 编程实现: 一维泛函极值问题的数值求解
- 实验: 对比不同泛函的极值性质

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第2周: Euler-Lagrange方程

### 本周目标
- 推导Euler-Lagrange方程
- 理解边界条件与自然边界条件
- 掌握高阶变分问题

### 忽读论文
1. Gelfand & Fomin - Calculus of Variations (Ch3) - 50页
2. Weinstock - Calculus of Variations (Ch2-3) - 45页

### 选读论文
1. Bliss - Lectures on the Calculus of Variations - 历史视角

### 实践任务
- 编程实现: 多变量Euler-Lagrange方程求解器
- 实验: 验证最速降线问题

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第3周: 全变分与ROF模型

### 本周目标
- 理解全变分(TV)的定义与性质
- 掌握ROF去噪模型
- 理解BV空间

### 忽读论文
1. Rudin, Osher, Fatemi - Nonlinear total variation based noise removal algorithms (1992) - 12页
2. Chambolle et al. - An introduction to total variation for image analysis - 40页

### 选读论文
1. Chan & Shen - Image Processing and Analysis: Variational, PDE, Wavelet, and Stochastic Methods (Ch3-4) - 系统性参考

### 实践任务
- 编程实现: ROF模型去噪算法
- 实验: 对比不同正则化参数的效果

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第4周: Mumford-Shah泛函

### 本周目标
- 理解Mumford-Shah泛函的数学结构
- 掌握分段光滑逼近
- 理解自由边界问题

### 忽读论文
1. Mumford & Shah - Optimal approximations by piecewise smooth functions and associated variational problems (1989) - 30页
2. Ambrosio et al. - Approximation of functionals depending on jumps by elliptic functionals - 25页

### 选读论文
1. Morel & Solimini - Variational Methods in Image Segmentation - 深入理解

### 实践任务
- 编程实现: 简化的Mumford-Shah模型
- 实验: 在合成图像上测试

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

### 第5-8周: 优化理论

## 第5周: 凸优化基础

### 本周目标
- 理解凸集、凸函数的定义与性质
- 掌握凸优化问题的标准形式
- 理解强对偶性

### 忽读论文
1. Boyd & Vandenberghe - Convex Optimization (Ch1-4) - 120页
2. Rockafellar - Convex Analysis (Ch1-2) - 60页

### 选读论文
1. Bertsekas - Convex Optimization Theory - 补充证明细节

### 实践任务
- 编程实现: 凸函数性质验证工具
- 实验: 使用CVX求解标准凸优化问题

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第6周: 对偶理论与KKT条件

### 本周目标
- 理解Lagrange对偶
- 掌握KKT条件
- 理解对偶间隙

### 忽读论文
1. Boyd & Vandenberghe - Convex Optimization (Ch5) - 60页
2. Nocedal & Wright - Numerical Optimization (Ch12) - 35页

### 选读论文
1. Bertsekas - Nonlinear Programming (Ch5-6) - 深入对偶理论

### 实践任务
- 编程实现: 对偶问题构造与求解
- 实验: 验证原始-对偶解的关系

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第7周: 近端算法

### 本周目标
- 理解近端算子的定义与性质
- 掌握近端梯度法
- 理解FISTA加速

### 忽读论文
1. Parikh & Boyd - Proximal Algorithms - 45页
2. Beck & Teboulle - A fast iterative shrinkage-thresholding algorithm (2009) - 15页

### 选读论文
1. Combettes & Pesquet - Proximal splitting methods in signal processing - 系统综述

### 实践任务
- 编程实现: 近端梯度法和FISTA
- 实验: L1正则化问题求解

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第8周: ADMM算法

### 本周目标
- 理解ADMM算法原理
- 掌握一致性优化问题
- 理解收敛性分析

### 忽读论文
1. Boyd et al. - Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers - 50页
2. Glowinski - On Alternating Direction Methods of Multipliers - 30页

### 选读论文
1. Eckstein & Yao - Understanding the convergence of the alternating direction method of multipliers - 收敛性分析

### 实践任务
- 编程实现: ADMM求解LASSO问题
- 实验: 对比ADMM与其他方法

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

### 第9-13周: 小波与变换

## 第9周: 傅里叶变换

### 本周目标
- 理解连续与离散傅里叶变换
- 掌握傅里叶变换的性质
- 理解采样定理

### 忽读论文
1. Stein & Shakarchi - Fourier Analysis: An Introduction (Ch1-3) - 90页
2. Mallat - A Wavelet Tour of Signal Processing (Ch2-3) - 60页

### 选读论文
1. Bracewell - The Fourier Transform and Its Applications - 经典参考

### 实践任务
- 编程实现: FFT及其逆变换
- 实验: 频域滤波

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第10周: 小波变换

### 本周目标
- 理解连续与离散小波变换
- 掌握多分辨率分析
- 理解小波基的构造

### 忽读论文
1. Mallat - A Wavelet Tour of Signal Processing (Ch4-6) - 100页
2. Daubechies - Ten Lectures on Wavelets (Lecture 1-3) - 80页

### 选读论文
1. Strang & Nguyen - Wavelets and Filter Banks - 滤波器视角

### 实践任务
- 编程实现: DWT与IDWT
- 实验: 图像小波去噪

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第11周: 紧框架

### 本周目标
- 理解框架理论
- 掌握紧框架的构造
- 理解冗余表示的优势

### 忽读论文
1. Mallat - A Wavelet Tour of Signal Processing (Ch5) - 50页
2. Christensen - An Introduction to Frames and Riesz Bases (Ch1-3) - 70页

### 选读论文
1. Kovačević & Chebira - An introduction to frames - 综述文章

### 实践任务
- 编程实现: 紧框架变换
- 实验: 对比正交基与紧框架

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第12周: 球面小波

### 本周目标
- 理解球面调和函数
- 掌握球面小波构造
- 理解球面上的多分辨率分析

### 忽读论文
1. Antoine et al. - Wavelets on the Sphere (1999) - 20页
2. Narcowich et al. - A Continuous Frame Associated with Spherical Wavelets - 18页

### 选读论文
1. Freeden & Schreiner - Spherical Functions of Mathematical Geosciences - 系统参考

### 实践任务
- 编程实现: 球面小波变换
- 实验: 球面数据处理

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第13周: 复习与总结

### 本周目标
- 系统回顾数学基础知识
- 建立知识点之间的联系
- 查漏补缺

### 忽读论文
1. 复习前12周笔记 - 全部

### 选读论文
1. 根据薄弱环节选择补充阅读

### 实践任务
- 编程实现: 综合项目(图像去噪流水线)
- 实验: 整合多种方法对比

### 笔记输出
- 完成季度总结笔记
- 绘制知识图谱
- 规划下季度重点

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第二季度: 图像分割 (14-26周)

### 第14-17周: 活动轮廓模型

## 第14周: Snake模型

### 本周目标
- 理解参数活动轮廓的基本原理
- 掌握能量函数设计
- 理解内力与外力

### 忽读论文
1. Kass et al. - Snakes: Active contour models (1988) - 15页
2. Xu & Prince - Snakes, Shapes, and Gradient Vector Flow (1998) - 20页

### 选读论文
1. Cohen - On active contour models and balloons - 改进模型

### 实践任务
- 编程实现: 经典Snake模型
- 实验: 在简单图像上测试边缘检测

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第15周: 水平集方法基础

### 本周目标
- 理解隐式曲线表示
- 掌握水平集方程
- 理解曲率流

### 忽读论文
1. Osher & Sethian - Fronts propagating with curvature-dependent speed (1988) - 18页
2. Sethian - Level Set Methods and Fast Marching Methods (Ch1-4) - 80页

### 选读论文
1. Osher & Fedkiw - Level Set Methods and Dynamic Implicit Surfaces - 系统参考

### 实践任务
- 编程实现: 基本水平集演化
- 实验: 曲率驱动的曲线收缩

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第16周: Chan-Vese模型

### 本周目标
- 理解区域竞争思想
- 掌握Chan-Vese模型的数学推导
- 理解与Mumford-Shah的联系

### 忽读论文
1. Chan & Vese - Active contours without edges (2001) - 15页
2. Chan & Vese - A level set algorithm for minimizing the Mumford-Shah functional (2000) - 12页

### 选读论文
1. Vese & Chan - A multiphase level set framework for image segmentation - 多相扩展

### 实践任务
- 编程实现: Chan-Vese分割算法
- 实验: 弱边缘图像分割

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第17周: 几何活动轮廓

### 本周目标
- 理解几何活动轮廓模型
- 掌握测地线活动轮廓(GAC)
- 理解边缘指示函数

### 忽读论文
1. Caselles et al. - Geodesic active contours (1997) - 18页
2. Kichenassamy et al. - Conformal curvature flows (1995) - 15页

### 选读论文
1. Paragios et al. - Geodesic active regions - 区域与边缘结合

### 实践任务
- 编程实现: GAC模型
- 实验: 对比GAC与Chan-Vese

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

### 第18-21周: 图割与能量优化

## 第18周: 图割基础

### 本周目标
- 理解图论在图像处理中的应用
- 掌握最大流/最小割定理
- 理解s-t割

### 忽读论文
1. Boykov & Kolmogorov - An Experimental Comparison of Min-Cut/Max-Flow Algorithms (2004) - 20页
2. Kolmogorov & Zabih - What Energy Functions Can Be Minimized via Graph Cuts? (2004) - 25页

### 选读论文
1. Ford & Fulkerson - Flows in Networks - 经典图论参考

### 实践任务
- 编程实现: 最大流算法(Ford-Fulkerson或Push-Relabel)
- 实验: 二值图像分割

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第19周: GrabCut算法

### 本周目标
- 理解交互式分割框架
- 掌握GMM颜色模型
- 理解迭代优化策略

### 忽读论文
1. Rother et al. - GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts (2004) - 10页
2. Boykov & Jolly - Interactive Graph Cuts for Optimal Boundary & Region Segmentation (2001) - 8页

### 选读论文
1. Blake et al. - Interactive Image Segmentation using an Adaptive GMMRF Model - 扩展模型

### 实践任务
- 编程实现: GrabCut算法
- 实验: 交互式前景提取

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第20周: 多标签分割

### 本周目标
- 理解多标签能量函数
- 掌握α-expansion和α-β-swap
- 理解子模性

### 忽读论文
1. Boykov et al. - Fast Approximate Energy Minimization via Graph Cuts (2001) - 20页
2. Kolmogorov & Rother - Comparison of Energy Minimization Algorithms for MRFs (2007) - 15页

### 选读论文
1. Delong et al. - Fast Approximate Energy Minimization with Label Costs - 标签代价

### 实践任务
- 编程实现: α-expansion算法
- 实验: 语义分割任务

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第21周: 条件随机场

### 本周目标
- 理解CRF的基本概念
- 掌握全连接CRF
- 理解平均场推断

### 忽读论文
1. Krähenbühl & Koltun - Efficient Inference in Fully Connected CRFs (2011) - 12页
2. Krähenbühl & Koltun - Parameter Learning and Convergent Inference for Dense CRFs (2013) - 10页

### 选读论文
1. Sutton & McCallum - An Introduction to Conditional Random Fields for Relational Learning - 系统教程

### 实践任务
- 编程实现: 全连接CRF推断
- 实验: 语义分割后处理

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

### 第22-26周: 深度学习分割与总结

## 第22周: FCN

### 本周目标
- 理解全卷积网络架构
- 掌握转置卷积
- 理解跳跃连接

### 忽读论文
1. Long et al. - Fully Convolutional Networks for Semantic Segmentation (2015) - 10页
2. Long et al. - Fully Convolutional Networks for Semantic Segmentation (CVPR 2015 slides) - 补充材料

### 选读论文
1. Zeiler & Fergus - Visualizing and Understanding Convolutional Networks - 反卷积理解

### 实践任务
- 编程实现: FCN-8s
- 实验: PASCAL VOC分割

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第23周: U-Net

### 本周目标
- 理解编码器-解码器架构
- 掌握U-Net跳跃连接
- 理解小样本学习能力

### 忽读论文
1. Ronneberger et al. - U-Net: Convolutional Networks for Biomedical Image Segmentation (2015) - 8页
2. Çiçek et al. - 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation (2016) - 8页

### 选读论文
1. Milletari et al. - V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation - 3D扩展

### 实践任务
- 编程实现: U-Net
- 实验: 医学图像分割

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第24周: DeepLab系列

### 本周目标
- 理解空洞卷积
- 掌握ASPP模块
- 理解DeepLabv3+架构

### 忽读论文
1. Chen et al. - DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution (2017) - 12页
2. Chen et al. - Encoder-Decoder with Atrous Separable Convolution (2018) - 12页

### 选读论文
1. Yu & Koltun - Multi-Scale Context Aggregation by Dilated Convolutions - 空洞卷积起源

### 实践任务
- 编程实现: ASPP模块
- 实验: 对比不同backbone

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第25周: 实例分割

### 本周目标
- 理解实例分割问题
- 掌握Mask R-CNN架构
- 理解ROI Align

### 忽读论文
1. He et al. - Mask R-CNN (2017) - 12页
2. Ren et al. - Faster R-CNN: Towards Real-Time Object Detection (2015) - 12页

### 选读论文
1. Bolya et al. - YOLACT: Real-time Instance Segmentation - 实时方法

### 实践任务
- 编程实现: Mask R-CNN关键模块
- 实验: COCO实例分割

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第26周: 复习与总结

### 本周目标
- 系统回顾图像分割方法
- 建立传统方法与深度学习的联系
- 查漏补缺

### 忽读论文
1. 复习第14-25周笔记 - 全部

### 选读论文
1. 根据薄弱环节选择补充阅读

### 实践任务
- 编程实现: 综合分割项目
- 实验: 对比多种方法性能

### 笔记输出
- 完成季度总结笔记
- 绘制分割方法演进图谱
- 规划下季度重点

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第三季度: 3D视觉与医学影像 (27-39周)

### 第27-30周: 3D数据处理

## 第27周: 点云基础

### 本周目标
- 理解点云数据表示
- 掌握点云处理基本操作
- 理解点云配准问题

### 忽读论文
1. Qi et al. - PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation (2017) - 12页
2. Rusu et al. - 3D is here: Point Cloud Library (PCL) - 8页

### 选读论文
1. Kaaz et al. - PointNet++: Deep Hierarchical Feature Learning on Point Sets - 层次化扩展

### 实践任务
- 编程实现: PointNet分类网络
- 实验: ModelNet40分类

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第28周: 点云分割

### 本周目标
- 理解点云分割问题
- 掌握PointNet++架构
- 理解局部特征聚合

### 忽读论文
1. Qi et al. - PointNet++ (2017) - 12页
2. Zhao et al. - Point Transformer (2021) - 12页

### 选读论文
1. Engelmann et al. - RGBD to 3D Point Cloud - 数据处理

### 实践任务
- 编程实现: PointNet++分割
- 实验: ShapeNet部件分割

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第29周: 体素与网格

### 本周目标
- 理解体素表示
- 掌握3D卷积
- 理解网格表示与处理

### 忽读论文
1. Maturana & Scherer - VoxNet: A 3D Convolutional Neural Network for real-time object recognition (2015) - 6页
2. Wang et al. - OctNet: Learning Deep 3D Representations at High Resolutions (2017) - 10页

### 选读论文
1. Wu et al. - 3D ShapeNets - 3D深度学习开创工作

### 实践任务
- 编程实现: 3D CNN分类网络
- 实验: 体素分类任务

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第30周: 多视图方法

### 本周目标
- 理解多视图表示
- 掌握视图聚合策略
- 理解投影不变性

### 忽读论文
1. Su et al. - Multi-view Convolutional Neural Networks for 3D Shape Recognition (2015) - 10页
2. Kanezaki et al. - RotationNet: Learning Object Classification Using Unsupervised Viewpoint Estimation (2018) - 10页

### 选读论文
1. Wei et al. - View-GCN: View-based Graph Convolutional Network for 3D Shape Analysis - 图视角

### 实践任务
- 编程实现: MVCNN
- 实验: 多视图3D分类

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

### 第31-34周: 医学图像分割

## 第31周: 医学图像特点与预处理

### 本周目标
- 理解CT/MRI成像原理
- 掌握医学图像预处理
- 理解医学图像标注特点

### 忽读论文
1. Clark et al. - The Cancer Imaging Archive (TCIA) - 8页
2. preprocessing综述 - 20页

### 选读论文
1. 相关医学成像教材

### 实践任务
- 编程实现: 医学图像预处理流水线
- 实验: 窗宽窗位调整

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第32周: 器官分割

### 本周目标
- 理解器官分割问题
- 掌握nnU-Net框架
- 理解医学图像挑战

### 忽读论文
1. Isensee et al. - nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation (2021) - 15页
2. Myronenko - 3D MRI Brain Tumor Segmentation Using Autoencoder Regularization (2018) - 10页

### 选读论文
1. Milletari et al. - V-Net - 3D医学分割

### 实践任务
- 编程实现: nnU-Net关键组件
- 实验: 器官分割任务

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第33周: 病灶分割

### 本周目标
- 理解病灶分割挑战
- 掌握注意力机制应用
- 理解类别不平衡处理

### 忽读论文
1. Oktay et al. - Attention U-Net (2018) - 12页
2. Wang et al. - Interactive Medical Image Segmentation (2021) - 10页

### 选读论文
1. Zhou et al. - UNet++: A Nested U-Net Architecture - 改进架构

### 实践任务
- 编程实现: Attention U-Net
- 实验: 肿瘤分割任务

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第34周: 多模态融合

### 本周目标
- 理解多模态医学图像
- 掌握融合策略
- 理解跨模态学习

### 忽读论文
1. Myronenko - 3D MRI Brain Tumor Segmentation (2018) - 多模态输入
2. Chen et al. - TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation (2021) - 12页

### 选读论文
1. Hatamizadeh et al. - UNetr: Transformers for 3D Medical Image Segmentation - Transformer应用

### 实践任务
- 编程实现: 多模态分割网络
- 实验: BraTS多模态分割

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

### 第35-39周: 图像配准与重建

## 第35周: 图像配准基础

### 本周目标
- 理解图像配准问题
- 掌握刚性/仿射配准
- 理解配准评价指标

### 忽读论文
1. Szeliski - Image Alignment and Stitching: A Tutorial (2006) - 40页
2. Maintz & Viergever - A survey of medical image registration (1998) - 30页

### 选读论文
1. 相关医学图像配准教材

### 实践任务
- 编程实现: 基于特征的刚性配准
- 实验: 医学图像刚性配准

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第36周: 可变形配准

### 本周目标
- 理解非刚性配准
- 掌握Demons算法
- 理解正则化策略

### 忽读论文
1. Thirion - Image matching as a diffusion process: an explanation of the demons algorithm (1998) - 15页
2. Vercauteren et al. - Diffeomorphic Demons (2009) - 12页

### 选读论文
1. Ashburner - A fast diffeomorphic image registration algorithm - DARTEL

### 实践任务
- 编程实现: Demons配准算法
- 实验: 肺部图像配准

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第37周: 深度学习配准

### 本周目标
- 理解学习型配准方法
- 掌握VoxelMorph架构
- 理解无监督配准

### 忽读论文
1. Balakrishnan et al. - VoxelMorph: A Learning Framework for Deformable Medical Image Registration (2019) - 12页
2. de Vos et al. - End-to-End Unsupervised Deformable Image Registration (2019) - 10页

### 选读论文
1. Dalca et al. - Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration - 概率方法

### 实践任务
- 编程实现: VoxelMorph
- 实验: 脑部MRI配准

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第38周: 3D重建

### 本周目标
- 理解三维重建问题
- 掌握NeRF基本原理
- 理解体渲染

### 忽读论文
1. Mildenhall et al. - NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (2020) - 12页
2. Kerbl et al. - 3D Gaussian Splatting for Real-Time Radiance Field Rendering (2023) - 15页

### 选读论文
1. Zhang et al. - Nerf++: Analyzing and Improving Neural Radiance Fields - NeRF改进

### 实践任务
- 编程实现: 简化NeRF
- 实验: 简单场景重建

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第39周: 复习与总结

### 本周目标
- 系统回顾3D视觉与医学影像方法
- 建立2D到3D的知识迁移
- 查漏补缺

### 忽读论文
1. 复习第27-38周笔记 - 全部

### 选读论文
1. 根据薄弱环节选择补充阅读

### 实践任务
- 编程实现: 综合3D医学影像项目
- 实验: 完整分割配准流水线

### 笔记输出
- 完成季度总结笔记
- 绘制3D视觉知识图谱
- 规划下季度重点

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第四季度: 深度学习前沿 (40-52周)

### 第40-43周: Transformer架构

## 第40周: Transformer基础

### 本周目标
- 理解自注意力机制
- 掌握Transformer架构
- 理解位置编码

### 忽读论文
1. Vaswani et al. - Attention is All You Need (2017) - 15页
2. Hamilton - Attention is not all you need - 补充理解

### 选读论文
1. Devlin et al. - BERT: Pre-training of Deep Bidirectional Transformers - NLP应用

### 实践任务
- 编程实现: Transformer编码器
- 实验: 序列到序列任务

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第41周: Vision Transformer

### 本周目标
- 理解图像patch化
- 掌握ViT架构
- 理解大规模预训练

### 忽读论文
1. Dosovitskiy et al. - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (2020) - 15页
2. Steiner et al. - How to train your ViT? Training Vision Transformers (2021) - 12页

### 选读论文
1. Touvron et al. - Training data-efficient image transformers & distillation through attention - DeiT

### 实践任务
- 编程实现: ViT
- 实验: ImageNet分类

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第42周: Swin Transformer

### 本周目标
- 理解层次化Transformer
- 掌握移动窗口注意力
- 理解相对位置编码

### 忽读论文
1. Liu et al. - Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (2021) - 15页
2. Liu et al. - Swin Transformer V2 (2022) - 12页

### 选读论文
1. Chu et al. - Twins: Revisiting the Design of Spatial Attention in Vision Transformers - 双分支设计

### 实践任务
- 编程实现: Swin Transformer
- 实验: 密集预测任务

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第43周: 分割Transformer

### 本周目标
- 理解Transformer在分割中的应用
- 掌握SegFormer架构
- 理解Mask Transformer

### 忽读论文
1. Xie et al. - SegFormer: Simple and Efficient Design for Semantic Segmentation (2021) - 12页
2. Cheng et al. - Mask2Former (2022) - 15页

### 选读论文
1. Strudel et al. - Segmenter: Transformer for Semantic Segmentation - 另一种设计

### 实践任务
- 编程实现: SegFormer
- 实验: ADE20K语义分割

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

### 第44-47周: 扩散模型

## 第44周: 扩散模型基础

### 本周目标
- 理解扩散过程数学原理
- 掌握分数匹配
- 理解去噪扩散概率模型(DDPM)

### 忽读论文
1. Ho et al. - Denoising Diffusion Probabilistic Models (2020) - 12页
2. Sohl-Dickstein et al. - Deep Unsupervised Learning using Nonequilibrium Thermodynamics (2015) - 15页

### 选读论文
1. Song & Ermon - Generative Modeling by Estimating Gradients of the Data Distribution - 分数生成模型

### 实践任务
- 编程实现: DDPM
- 实验: MNIST生成

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第45周: 条件扩散模型

### 本周目标
- 理解条件生成
- 掌握Classifier-Free Guidance
- 理解文本条件生成

### 忽读论文
1. Ho & Salimans - Classifier-Free Diffusion Guidance (2022) - 8页
2. Rombach et al. - High-Resolution Image Synthesis with Latent Diffusion Models (2022) - 15页

### 选读论文
1. Nichol et al. - GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion - 文本生成

### 实践任务
- 编程实现: 条件DDPM
- 实验: 类别条件生成

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第46周: 扩散模型加速

### 本周目标
- 理解采样加速方法
- 掌握DDIM采样
- 理解一致性模型

### 忽读论文
1. Song et al. - Denoising Diffusion Implicit Models (2020) - 12页
2. Song et al. - Consistency Models (2023) - 15页

### 选读论文
1. Lu et al. - DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling - ODE求解器

### 实践任务
- 编程实现: DDIM采样
- 实验: 对比采样速度与质量

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第47周: 扩散模型应用

### 本周目标
- 理解扩散模型在分割中的应用
- 掌握条件图像生成
- 理解图像编辑

### 忽读论文
1. Amit et al. - SegDiff: Image Segmentation with Diffusion Probabilistic Models (2022) - 10页
2. Lugmayr et al. - RePaint: Inpainting using Denoising Diffusion Probabilistic Models (2022) - 10页

### 选读论文
1. Baranchuk et al. - Label-Efficient Semantic Segmentation with Diffusion Models - 语义分割

### 实践任务
- 编程实现: 扩散模型图像修复
- 实验: 掩码图像修复

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

### 第48-52周: 自监督学习与总结

## 第48周: 对比学习

### 本周目标
- 理解对比学习原理
- 掌握SimCLR/MoCo
- 理解负样本策略

### 忽读论文
1. Chen et al. - A Simple Framework for Contrastive Learning of Visual Representations (2020) - 12页
2. He et al. - Momentum Contrast for Unsupervised Visual Representation Learning (2020) - 12页

### 选读论文
1. Chen et al. - Improved Baselines with Momentum Contrast Verification - MoCo v2

### 实践任务
- 编程实现: SimCLR
- 实验: 线性评估协议

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第49周: 掩码自编码器

### 本周目标
- 理解掩码建模
- 掌握MAE架构
- 理解自监督预训练

### 忽读论文
1. He et al. - Masked Autoencoders Are Scalable Vision Learners (2022) - 12页
2. Bao et al. - BEiT: BERT Pre-Training of Image Transformers (2022) - 12页

### 选读论文
1. Wei et al. - Masked Feature Prediction for Self-Supervised Visual Pre-Training - MaskFeat

### 实践任务
- 编程实现: MAE
- 实验: ViT预训练

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第50周: 多模态学习

### 本周目标
- 理解多模态表示学习
- 掌握CLIP架构
- 理解图文对齐

### 忽读论文
1. Radford et al. - Learning Transferable Visual Models From Natural Language Supervision (2021) - 15页
2. Jia et al. - Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision (2021) - 12页

### 选读论文
1. Li et al. - BLIP: Bootstrapping Language-Image Pre-training - 多模态预训练

### 实践任务
- 编程实现: CLIP风格模型
- 实验: 零样本分类

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第51周: SAM与基础模型

### 本周目标
- 理解分割基础模型
- 掌握SAM架构
- 理解提示学习

### 忽读论文
1. Kirillov et al. - Segment Anything (2023) - 15页
2. Zhang et al. - Personalize Segment Anything - 个人化扩展

### 选读论文
1. Ma et al. - Segment Anything in Medical Images - 医学SAM

### 实践任务
- 编程实现: SAM关键组件
- 实验: 提示式分割

### 笔记输出
- 完成论文精读笔记
- 总结关键公式
- 记录疑问

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 第52周: 年度总结与展望

### 本周目标
- 系统回顾全年学习内容
- 建立完整知识体系
- 规划未来研究方向

### 忽读论文
1. 复习全年笔记 - 全部

### 选读论文
1. 根据研究方向选择前沿论文

### 实践任务
- 编程实现: 综合项目(结合多季度知识)
- 实验: 完整研究项目

### 笔记输出
- 完成年度总结笔记
- 绘制完整知识图谱
- 撰写未来研究计划

### 检查点
- [ ] 理解核心概念
- [ ] 完成代码实现
- [ ] 输出学习笔记

---

## 附录

### A. 推荐学习资源

#### 经典教材
1. Boyd & Vandenberghe - Convex Optimization
2. Mallat - A Wavelet Tour of Signal Processing
3. Szeliski - Computer Vision: Algorithms and Applications
4. Goodfellow et al. - Deep Learning

#### 在线课程
1. Stanford CS231n: Convolutional Neural Networks
2. Stanford CS224n: Natural Language Processing with Deep Learning
3. MIT 6.S191: Introduction to Deep Learning

#### 开源代码库
1. PyTorch
2. MONAI (Medical Open Network for AI)
3. OpenMMLab

### B. 论文管理建议

1. 使用Zotero/Mendeley管理文献
2. 每周精读2-3篇论文
3. 建立个人论文笔记库
4. 定期回顾重要论文

### C. 代码实践建议

1. 建立个人代码库
2. 注重代码可复现性
3. 记录实验日志
4. 参与开源项目

### D. 进度跟踪

| 周次 | 主题 | 完成状态 | 备注 |
|------|------|----------|------|
| 1 | 泛函分析基础 | | |
| 2 | Euler-Lagrange方程 | | |
| ... | ... | | |
| 52 | 年度总结 | | |

---

*计划制定时间: 2026年2月*
*预计完成时间: 2027年2月*
