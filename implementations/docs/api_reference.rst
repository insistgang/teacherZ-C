====================================
论文核心算法 API 参考手册
====================================

.. toctree::
   :maxdepth: 2
   
   slat_segmentation
   rof_iterative_segmentation
   tucker_decomposition
   neural_varifold

概述
====

本模块基于以下核心论文实现：

- **SLaT 三阶段分割**: Journal of Scientific Computing (2017)
- **ROF 迭代阈值分割**: IEEE Transactions on Image Processing (2013)  
- **Tucker 分解加速**: SIAM Journal on Scientific Computing
- **Neural Varifolds**: IEEE TPAMI (2022)

模块索引
========

.. autosummary::
   :toctree: _autosummary
   :template: modules.rst
   
   slat_segmentation
   rof_iterative_segmentation
   tucker_decomposition
   neural_varifold
