================================
Neural Varifold 点云核
================================

.. automodule:: neural_varifold
   :members:
   :show-inheritance:

模块概述
========

Neural Varifold 是一种用于 3D 点云处理的神经表示方法。

**论文**: Neural Varifolds for 3D Point Cloud Processing  
**来源**: IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022  
**作者**: Xiaohao Cai, et al.

核心概念
--------

**Varifold** 是一种几何测度，将点云表示为带权重的分布：

.. math::

   V = \sum_i w_i \cdot \phi(p_i, n_i) \cdot \delta_{p_i}

其中:
- :math:`p_i`: 点位置
- :math:`n_i`: 点法向量
- :math:`w_i`: Varifold 权重
- :math:`\phi`: 特征映射

依赖关系
========

- numpy >= 1.20
- torch >= 1.9

核函数参考
==========

PositionKernel
--------------

位置核，计算点之间的空间相似性。

.. autoclass:: PositionKernel
   :members:
   :special-members: __init__

   .. automethod:: forward

**数学定义**:

.. math::

   k(p, q) = \exp\left(-\frac{\|p - q\|^2}{2\sigma^2}\right)

NormalKernel
------------

法向量核，测量法向量的对齐程度。

.. autoclass:: NormalKernel
   :members:
   :special-members: __init__

   .. automethod:: forward

**数学定义**:

.. math::

   k_n(n_1, n_2) = \langle n_1, n_2 \rangle^k

Varifold 表示
=============

VarifoldRepresentation
----------------------

将点云编码为 Varifold 表示。

.. autoclass:: VarifoldRepresentation
   :members:
   :special-members: __init__

   .. automethod:: forward

组件:

1. **位置编码器**: MLP 编码 3D 坐标
2. **法向量编码器**: MLP 编码法向量 (可选)
3. **权重网络**: 输出 Varifold 权重
4. **位置核**: 高斯核
5. **法向量核**: 内积核

VarifoldKernel
--------------

计算两个 Varifold 之间的核相似性。

.. autoclass:: VarifoldKernel
   :members:
   :special-members: __init__

   .. automethod:: forward

**数学定义**:

.. math::

   K(V_1, V_2) = \sum_{i,j} w^1_i \cdot w^2_j \cdot k(p^1_i, p^2_j) \cdot \langle f^1_i, f^2_j \rangle

VarifoldDistance
----------------

计算两个 Varifold 之间的距离。

.. autoclass:: VarifoldDistance
   :members:
   :special-members: __init__

   .. automethod:: forward

**数学定义**:

.. math::

   \|V_1 - V_2\|_V^2 = K(V_1, V_1) + K(V_2, V_2) - 2K(V_1, V_2)

网络架构
========

NeuralVarifoldNet
-----------------

完整的点云分割/分类网络。

.. autoclass:: NeuralVarifoldNet
   :members:
   :special-members: __init__

   .. automethod:: forward

架构:

.. mermaid::
   :caption: NeuralVarifoldNet 架构
   
   graph TB
       A[输入: positions, normals] --> B[VarifoldRepresentation]
       B --> C[VarifoldLayer × 3]
       C --> D[MLP Head]
       D --> E[输出: logits]

VarifoldLayer
-------------

基于 Varifold 核的特征更新层。

.. autoclass:: VarifoldLayer
   :members:
   :special-members: __init__

   .. automethod:: forward

**特征更新**:

.. math::

   f'_i = f_i + \alpha_i \cdot \text{MLP}\left(\sum_j \frac{k(p_i, p_j) \cdot w_j}{\sum_l k(p_i, p_l) \cdot w_l} \cdot f_j\right)

函数参考
========

compute_varifold_norm
---------------------

.. autofunction:: compute_varifold_norm

**数学定义**:

.. math::

   \|V\|_V^2 = \sum_{i,j} w_i \cdot w_j \cdot k(p_i, p_j) \cdot \langle n_i, n_j \rangle

使用示例
========

快速开始
--------

.. code-block:: python

   import torch
   from neural_varifold import NeuralVarifoldNet
   
   # 创建网络
   net = NeuralVarifoldNet(
       num_classes=10,    # 分割类别数
       feat_dim=64,       # 特征维度
       use_normals=True   # 使用法向量
   )
   
   # 前向传播
   positions = torch.randn(2, 256, 3)      # (B, N, 3)
   normals = torch.randn(2, 256, 3)
   normals = torch.nn.functional.normalize(normals, dim=-1)
   
   logits = net(positions, normals)        # (B, N, num_classes)
   predictions = logits.argmax(dim=-1)

核函数使用
----------

.. code-block:: python

   from neural_varifold import PositionKernel, NormalKernel
   
   # 位置核
   pos_kernel = PositionKernel(sigma=0.5, learnable=True)
   K_pos = pos_kernel(positions, positions)  # (N, N)
   
   # 法向量核
   normal_kernel = NormalKernel(exponent=1)
   K_normal = normal_kernel(normals, normals)  # (N, N)

Varifold 距离计算
-----------------

.. code-block:: python

   from neural_varifold import (
       VarifoldRepresentation, 
       VarifoldDistance
   )
   
   # 编码
   encoder = VarifoldRepresentation(feat_dim=64)
   feat1, w1, pos1 = encoder(positions1, normals1)
   feat2, w2, pos2 = encoder(positions2, normals2)
   
   # 计算距离
   dist_fn = VarifoldDistance(sigma_pos=0.5, sigma_feat=1.0)
   distance = dist_fn(
       (pos1, feat1, w1),
       (pos2, feat2, w2)
   )

Varifold 范数
-------------

.. code-block:: python

   from neural_varifold import compute_varifold_norm
   
   norm = compute_varifold_norm(positions, normals, weights, sigma=0.5)

参数速查表
==========

.. list-table:: PositionKernel 参数
   :widths: 20 15 20 45
   :header-rows: 1

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - sigma
     - float
     - 1.0
     - 高斯核带宽
   * - learnable
     - bool
     - True
     - 是否可学习 sigma

.. list-table:: NeuralVarifoldNet 参数
   :widths: 20 15 20 45
   :header-rows: 1

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - num_classes
     - int
     - 10
     - 分割类别数
   * - feat_dim
     - int
     - 64
     - 特征维度
   * - use_normals
     - bool
     - True
     - 是否使用法向量

最佳实践
========

数据预处理
----------

.. code-block:: python

   # 1. 位置归一化
   positions = (positions - positions.mean(dim=1, keepdim=True)) / positions.std()
   
   # 2. 法向量归一化
   normals = torch.nn.functional.normalize(normals, dim=-1)
   
   # 3. 点云采样 (如果点数过多)
   if N > 1024:
       idx = torch.randperm(N)[:1024]
       positions = positions[:, idx]
       normals = normals[:, idx]

sigma 参数调优
--------------

.. list-table:: sigma 选择指南
   :widths: 25 75
   :header-rows: 1

   * - 场景
     - 推荐 sigma
   * - 密集点云
     - 0.1 - 0.3
   * - 稀疏点云
     - 0.5 - 1.0
   * - 大尺度变化
     - 0.3 - 0.5 + 可学习

FAQ
===

Q: Varifold 与 PointNet 的区别？
   - PointNet: 独立处理每个点，全局池化
   - Varifold: 显式建模点之间的关系，使用核函数

Q: 为什么使用法向量？
   法向量提供局部几何信息，增强对形状变化的鲁棒性。

Q: 计算复杂度如何？
   核矩阵计算为 :math:`O(N^2)`，对于大点云建议使用采样或近似方法。

Q: 支持哪些任务？
   - 点云分割
   - 点云分类
   - 形状匹配
   - 点云配准
