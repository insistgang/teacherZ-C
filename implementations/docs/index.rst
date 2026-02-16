论文核心算法文档
=================

.. toctree::
   :maxdepth: 2
   :caption: 目录:

   api_reference
   slat_segmentation
   rof_iterative_segmentation
   tucker_decomposition
   neural_varifold

安装
====

依赖安装::

    pip install numpy opencv-python scipy scikit-learn torch

文档构建
========

使用 Sphinx 构建 HTML 文档::

    cd implementations/docs
    sphinx-build -b html . _build/html

或使用 Makefile::

    make html

快速开始
========

.. code-block:: python

    # SLaT 分割
    from slat_segmentation import SLATSegmentation
    segmenter = SLATSegmentation(lambda_param=1.0)
    result = segmenter.segment(image, K=4)

    # ROF 分割
    from rof_iterative_segmentation import IterativeROFSegmentation
    seg = IterativeROFSegmentation(lambda_param=0.1).segment(image, K=3)

    # Tucker 分解
    from tucker_decomposition import SketchingTucker
    core, factors = SketchingTucker(ranks=[5,4,3]).fit(tensor)

    # Neural Varifold
    from neural_varifold import NeuralVarifoldNet
    logits = NeuralVarifoldNet(num_classes=10)(positions, normals)

索引
====

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
