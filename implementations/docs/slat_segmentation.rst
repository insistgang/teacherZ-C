================================
SLaT 三阶段分割算法
================================

.. automodule:: slat_segmentation
   :members:
   :show-inheritance:
   :undoc-members:

模块概述
========

SLaT (Smoothing-Lifting-Thresholding) 三阶段分割算法，用于退化彩色图像分割。

**论文**: A Three-Stage Approach for Segmenting Degraded Color Images  
**来源**: Journal of Scientific Computing (2017) 72:1313–1332  
**作者**: Xiaohao Cai, Raymond Chan, Mila Nikolova, Tieyong Zeng

算法流程
--------

.. mermaid::
   :caption: SLaT 三阶段流程图
   
   graph LR
       A[输入RGB图像] --> B[Stage 1: ROF平滑]
       B --> C[Stage 2: RGB→Lab升维]
       C --> D[Stage 3: K-means分割]
       D --> E[分割标签图]

依赖关系
========

- numpy >= 1.20
- opencv-python >= 4.5
- scikit-learn >= 1.0

类参考
======

SLATSegmentation
----------------

.. autoclass:: SLATSegmentation
   :members:
   :inherited-members:
   :special-members: __init__

   .. rubric:: 方法

   .. automethod:: segment
   
   .. rubric:: 内部方法
   
   .. automethod:: _stage1_smoothing
   .. automethod:: _stage2_lifting
   .. automethod:: _stage3_thresholding

数学原理
========

Stage 1: ROF 变分平滑
---------------------

对每个 RGB 通道求解 ROF 模型：

.. math::

   \min_g \frac{\lambda}{2} \|f - g\|^2 + \frac{\mu}{2} \|\nabla g\|^2 + \|\nabla g\|_{TV}

其中:
- :math:`f`: 原始图像
- :math:`g`: 平滑后图像
- :math:`\lambda`: 数据保真项权重
- :math:`\mu`: H1 正则化权重 (论文固定为1)

**Chambolle 对偶算法**:

.. math::

   u = f - \lambda \cdot \text{div}(p)

   p^{n+1} = \frac{p^n + \tau \nabla u}{1 + \tau |\nabla u|}

Stage 2: 维度提升
-----------------

RGB → RGB + Lab (6通道):

.. math::

   g^* = [g_R, g_G, g_B, g_L, g_a, g_b]

Lab 空间归一化:

.. math::

   L_{norm} = \frac{L}{100}, \quad a_{norm} = \frac{a + 127}{254}, \quad b_{norm} = \frac{b + 127}{254}

Stage 3: K-means 分割
---------------------

在 6 维特征空间中进行聚类:

.. math::

   \min_{\{c_k\}} \sum_{k=1}^{K} \sum_{x \in C_k} \|g^*(x) - c_k\|^2

函数参考
========

slat_segment
------------

.. autofunction:: slat_segment

使用示例
========

快速开始
--------

.. code-block:: python

   from slat_segmentation import SLATSegmentation
   
   # 创建分割器
   segmenter = SLATSegmentation(
       lambda_param=1.0,    # 数据项权重
       mu=1.0,              # H1正则化权重
       rof_iterations=100,  # ROF迭代次数
       rof_tol=1e-4         # 收敛阈值
   )
   
   # 执行分割
   result = segmenter.segment(image, K=4)
   segmentation = result['segmentation']

进阶用法
--------

获取中间结果:

.. code-block:: python

   result = segmenter.segment(image, K=4, return_intermediate=True)
   
   print(f"平滑后: {result['smoothed'].shape}")      # (H, W, 3)
   print(f"升维后: {result['lifted'].shape}")        # (H, W, 6)
   print(f"分割结果: {result['segmentation'].shape}") # (H, W)

便捷函数:

.. code-block:: python

   from slat_segmentation import slat_segment
   
   segmentation = slat_segment(image, K=4, lambda_param=1.0)

参数调优
--------

.. list-table:: 参数调优建议
   :widths: 20 30 50
   :header-rows: 1

   * - 图像特征
     - lambda_param
     - 说明
   * - 高噪声
     - 0.5 - 1.0
     - 增强去噪效果
   * - 低噪声
     - 1.5 - 3.0
     - 保持更多细节
   * - 边缘重要
     - 2.0 - 5.0
     - 保留边缘信息

FAQ
===

.. rst-class:: faq

Q: 为什么使用 Lab 色彩空间？
   Lab 空间是感知均匀的，a/b 通道捕获与 RGB 互补的色彩信息，提高分割精度。

Q: 如何选择 K 值？
   K 值取决于图像中的实际区域数。可以使用肘部法则或轮廓系数确定。

Q: 计算复杂度如何？
   复杂度为 :math:`O(H \times W \times \text{rof\_iterations} + H \times W \times K \times \text{kmeans\_iter})`
