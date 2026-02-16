================================
ROF 迭代阈值分割算法
================================

.. automodule:: rof_iterative_segmentation
   :members:
   :show-inheritance:

模块概述
========

基于近似 ROF 模型的多类图像分割算法，将多类分割分解为一系列二值分割问题。

**论文**: A Multiphase Image Segmentation Based on Approximate ROF Models  
**来源**: IEEE Transactions on Image Processing, 2013  
**作者**: Xiaohao Cai, et al.

核心思想
--------

- 将 K 类分割分解为 K-1 个二值分割
- 使用标签树 (Label Tree) 组织层次分割
- 每步求解标准 ROF 模型

标签树类型
----------

.. mermaid::
   :caption: 标签树结构对比
   
   graph TB
       subgraph 平衡树
           A1[{0,1,2,3}] --> B1[{0,1}]
           A1 --> C1[{2,3}]
           B1 --> D1[0]
           B1 --> E1[1]
           C1 --> F1[2]
           C1 --> G1[3]
       end
       
       subgraph 顺序树
           A2[{0,1,2,3}] --> B2[0]
           A2 --> C2[{1,2,3}]
           C2 --> D2[1]
           C2 --> E2[{2,3}]
           E2 --> F2[2]
           E2 --> G2[3]
       end

依赖关系
========

- numpy >= 1.20
- scipy >= 1.7

类参考
======

ROFDenoiser
-----------

ROF 模型求解器，使用 Chambolle 投影算法。

.. autoclass:: ROFDenoiser
   :members:
   :special-members: __init__

   .. automethod:: solve

IterativeROFSegmentation
------------------------

迭代 ROF 多类分割主类。

.. autoclass:: IterativeROFSegmentation
   :members:
   :special-members: __init__

   .. automethod:: segment
   .. automethod:: _build_label_tree
   .. automethod:: _iterative_segment

AutomaticThresholdROF
---------------------

带自动阈值选择的 ROF 分割。

.. autoclass:: AutomaticThresholdROF
   :members:
   :special-members: __init__

   .. automethod:: segment_with_auto_threshold

数学原理
========

ROF 模型
--------

标准的 Rudin-Osher-Fatemi 模型：

.. math::

   \min_u \frac{\lambda}{2} \|f - u\|^2 + \text{TV}(u)

其中 :math:`\text{TV}(u) = \|\nabla u\|` 是全变分。

**Chambolle 对偶算法**:

.. math::

   p^{n+1} = \frac{p^n + \tau \nabla (f - \lambda \cdot \text{div}(p^n))}{1 + \tau |\nabla (f - \lambda \cdot \text{div}(p^n))|}

   u = f - \lambda \cdot \text{div}(p)

迭代分割
--------

对于二值分割问题:

.. math::

   \min_u \int_\Omega u(x) (f_A(x) - f_B(x)) \, dx + \lambda \cdot \text{TV}(u)

其中:
- :math:`f_A, f_B`: 类别 A/B 的数据项
- :math:`u \in [0, 1]`: 指示函数

数据项计算:

.. math::

   f_k(x) = (I(x) - c_k)^2

使用示例
========

基本用法
--------

.. code-block:: python

   from rof_iterative_segmentation import IterativeROFSegmentation
   
   # 创建分割器
   segmenter = IterativeROFSegmentation(
       lambda_param=0.1,    # ROF正则化参数
       max_iter=100,        # ROF迭代次数
       tree_type='balanced' # 标签树类型
   )
   
   # 执行分割
   segmentation = segmenter.segment(image, K=4)

自动阈值
--------

.. code-block:: python

   from rof_iterative_segmentation import AutomaticThresholdROF
   
   auto_seg = AutomaticThresholdROF(lambda_param=0.15)
   segmentation, thresholds = auto_seg.segment_with_auto_threshold(image, K=3)
   
   print(f"自动选择的阈值: {thresholds}")

ROF 去噪
--------

.. code-block:: python

   from rof_iterative_segmentation import ROFDenoiser
   
   denoiser = ROFDenoiser(max_iter=100, tol=1e-4)
   denoised = denoiser.solve(noisy_image, lambda_param=0.2)

参数速查表
==========

.. list-table:: IterativeROFSegmentation 参数
   :widths: 20 15 25 40
   :header-rows: 1

   * - 参数
     - 类型
     - 默认值
     - 说明
   * - lambda_param
     - float
     - 0.1
     - ROF 正则化强度
   * - max_iter
     - int
     - 100
     - ROF 迭代最大次数
   * - tree_type
     - str
     - 'balanced'
     - 'balanced' 或 'sequential'

.. list-table:: lambda_param 调优指南
   :widths: 25 75
   :header-rows: 1

   * - 参数值
     - 效果
   * - 0.05 - 0.1
     - 强去噪，边缘模糊
   * - 0.1 - 0.3
     - 平衡，推荐范围
   * - 0.3 - 0.5
     - 弱去噪，保持细节

FAQ
===

Q: 平衡树和顺序树如何选择？
   - **平衡树**: 各类别像素数相近时，分割更均匀
   - **顺序树**: 类别有自然顺序时 (如灰度分割)，效率更高

Q: 与 SLaT 的区别？
   - ROF 迭代分割适用于灰度图像
   - SLaT 适用于彩色图像，使用 Lab 空间增强
