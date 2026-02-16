================================
Tucker 分解加速算法
================================

.. automodule:: tucker_decomposition
   :members:
   :show-inheritance:

模块概述
========

基于随机 Sketching 的低秩 Tucker 分解算法。

**论文**: Low-Rank Tucker Approximation with Sketching  
**来源**: SIAM Journal on Scientific Computing / JMLR

核心贡献
--------

- **随机 Sketching**: 降低计算复杂度
- **Leverage Score 采样**: 自适应采样策略
- **HOOI 迭代**: 高阶正交迭代优化

Tucker 分解形式
---------------

.. math::

   \mathcal{X} \approx \mathcal{G} \times_1 \mathbf{A}^{(1)} \times_2 \mathbf{A}^{(2)} \cdots \times_N \mathbf{A}^{(N)}

其中:
- :math:`\mathcal{G}`: 核张量 (Core Tensor)
- :math:`\mathbf{A}^{(n)}`: 因子矩阵 (Factor Matrices)
- :math:`\times_n`: Mode-n 乘积

依赖关系
========

- numpy >= 1.20
- scipy >= 1.7

类参考
======

TensorOperations
----------------

张量操作工具类。

.. autoclass:: TensorOperations
   :members:
   :special-members:

   .. automethod:: mode_n_product
   .. automethod:: unfold
   .. automethod:: fold

RandomizedSVD
-------------

随机 SVD 分解器。

.. autoclass:: RandomizedSVD
   :members:
   :special-members: __init__

   .. automethod:: fit

SketchingTucker
---------------

基于 Sketching 的 Tucker 分解主类。

.. autoclass:: SketchingTucker
   :members:
   :special-members: __init__

   .. automethod:: fit
   .. automethod:: _build_sketch_matrices
   .. automethod:: _build_sketches
   .. automethod:: _extract_factors_from_sketch
   .. automethod:: _hooi_refinement

HOOIDecomposition
-----------------

标准 HOOI 分解。

.. autoclass:: HOOIDecomposition
   :members:
   :special-members: __init__

   .. automethod:: fit

函数参考
========

reconstruct_tucker
------------------

.. autofunction:: reconstruct_tucker

数学原理
========

Mode-n 乘积
-----------

张量 :math:`\mathcal{X}` 与矩阵 :math:`\mathbf{A}` 的 Mode-n 乘积：

.. math::

   \mathcal{Y} = \mathcal{X} \times_n \mathbf{A}

   Y_{i_1, \ldots, j, \ldots, i_N} = \sum_{i_n} X_{i_1, \ldots, i_n, \ldots, i_N} A_{j, i_n}

张量展开
--------

Mode-n 展开 (Matricization):

.. math::

   \mathbf{X}_{(n)} \in \mathbb{R}^{I_n \times \prod_{k \neq n} I_k}

随机 Sketching
--------------

构建 Gaussian 随机 Sketch 矩阵:

.. math::

   \mathbf{S}^{(n)} \in \mathbb{R}^{s_n \times I_n}, \quad s_n = \alpha \cdot R_n

Range Sketch:

.. math::

   \mathbf{Y}^{(n)} = \mathcal{X} \times_1 \mathbf{S}^{(1)} \cdots \times_{n-1} \mathbf{S}^{(n-1)} \times_{n+1} \mathbf{S}^{(n+1)} \cdots \times_N \mathbf{S}^{(N)}

Core Sketch:

.. math::

   \mathcal{Z} = \mathcal{X} \times_1 \mathbf{S}^{(1)} \times_2 \mathbf{S}^{(2)} \cdots \times_N \mathbf{S}^{(N)}

HOOI 算法
---------

Higher-Order Orthogonal Iteration:

.. math::

   \mathbf{A}^{(n)} = \text{SVD}_R \left( \mathcal{X} \times_1 {\mathbf{A}^{(1)}}^\top \cdots \times_{n-1} {\mathbf{A}^{(n-1)}}^\top \times_{n+1} {\mathbf{A}^{(n+1)}}^\top \cdots \times_N {\mathbf{A}^{(N)}}^\top \right)

复杂度分析
----------

.. list-table:: 复杂度对比
   :widths: 30 35 35
   :header-rows: 1

   * - 方法
     - 时间复杂度
     - 空间复杂度
   * - 标准 HOOI
     - :math:`O(\prod I_n \cdot R \cdot N)`
     - :math:`O(\prod I_n)`
   * - Sketching
     - :math:`O(\prod I_n \cdot s + s^N)`
     - :math:`O(\prod_{k \neq n} I_k \cdot s_n)`

使用示例
========

基本用法
--------

.. code-block:: python

   from tucker_decomposition import SketchingTucker, reconstruct_tucker
   
   # 创建分解器
   tucker = SketchingTucker(
       ranks=[5, 4, 3],        # 目标Tucker秩
       sketch_multipliers=2.0   # Sketch尺寸倍数
   )
   
   # 执行分解
   core, factors = tucker.fit(tensor, max_iter=50, tol=1e-6)
   
   # 重构张量
   reconstructed = reconstruct_tucker(core, factors)

标准 HOOI
---------

.. code-block:: python

   from tucker_decomposition import HOOIDecomposition
   
   hooi = HOOIDecomposition(ranks=[5, 4, 3], max_iter=100, tol=1e-6)
   core, factors = hooi.fit(tensor)

张量操作
--------

.. code-block:: python

   from tucker_decomposition import TensorOperations as TO
   
   # Mode-n 乘积
   Y = TO.mode_n_product(tensor, matrix, mode=0)
   
   # 展开与折叠
   matrix = TO.unfold(tensor, mode=0)
   tensor_recovered = TO.fold(matrix, mode=0, shape=tensor.shape)

随机 SVD
--------

.. code-block:: python

   from tucker_decomposition import RandomizedSVD
   
   rsvd = RandomizedSVD(n_oversamples=10, n_power_iter=2)
   U, S, Vh = rsvd.fit(matrix, rank=10)

最佳实践
========

秩选择
------

.. list-table:: 秩选择指南
   :widths: 25 75
   :header-rows: 1

   * - 场景
     - 建议
   * - 未知数据
     - 从小秩开始，逐步增加
   * - 视频数据
     - 时间维度秩通常较小
   * - 高压缩比
     - 秩为原始维度的 5%-10%

Sketch 参数
-----------

- **sketch_multipliers**: 1.5 - 3.0
  - 较小值: 更快，但精度略低
  - 较大值: 更高精度，但更慢

FAQ
===

Q: Sketching 与标准 HOOI 的精度差异？
   通常在 1% 以内，但速度可提升 5-10 倍。

Q: 如何确定最优 Tucker 秩？
   可以使用 Core Consistency Diagnostic (CORCONDIA) 或交叉验证。

Q: 适用于哪些应用？
   - 视频压缩
   - 高光谱图像分析
   - 推荐系统
   - 神经网络压缩
