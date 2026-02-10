#!/usr/bin/env python3
"""
按照阅读顺序重命名论文
格式: [阶段-序号] 论文标题.pdf
"""

import os
import shutil

download_dir = "xiaohao_cai_papers"

# 阅读顺序列表: (原文件名关键词, 新文件名)
reading_order = [
    # ========== 第一阶段：建立基础（第1-3周）==========
    ("CNNs RNNs and Transformers in human action recognition a survey", "[1-01] 深度学习架构综述 CNNs RNNs Transformers.pdf"),
    ("An overview of SaT segmentation methodology", "[1-02] 分割方法论总览 SaT Overview.pdf"),
    ("Data augmentation in classification and segmentation", "[1-03] 数据增强基础 Data Augmentation.pdf"),
    ("Linkage between piecewise constant Mumford-Shah and ROF model", "[1-04] 变分法基础 Mumford-Shah与ROF.pdf"),
    ("A two-stage classification method for high-dimensional data", "[1-05] 高维数据分类 Two-Stage Classification.pdf"),
    ("Explainable artificial intelligence advancements and limitations", "[1-06] 可解释AI综述 XAI Advancements.pdf"),

    # ========== 第二阶段：深入核心 - 图像分割与变分法（第4-5周）==========
    ("Two-stage image segmentation using convex variant of Mumford-Shah", "[2-01] 凸优化分割 Convex Mumford-Shah.pdf"),
    ("Multiclass segmentation by iterated ROF thresholding", "[2-02] 多类分割迭代ROF Iterated ROF.pdf"),
    ("A three-stage approach for segmenting degraded color images SLaT", "[2-03] SLaT三阶段分割 SLaT Segmentation.pdf"),
    ("Variational image segmentation model coupled with image restoration", "[2-04] 分割与恢复联合模型 Segmentation Restoration.pdf"),
    ("Semantic segmentation by semantic proportions", "[2-05] 语义比例分割 Semantic Proportions.pdf"),
    ("Detect closer surfaces that can be seen", "[2-06] 可见表面检测 Detect Closer Surfaces.pdf"),
    ("Disparity and optical flow partitioning using extended Potts priors", "[2-07] 光流分割 Potts Priors.pdf"),
    ("Vessel segmentation in medical imaging using tight-frame-based algorithm", "[2-08] 小波框架血管分割 Vessel Segmentation.pdf"),
    ("Framelet-based algorithm for segmentation of tubular structures", "[2-09] 框架分割管状结构 Framelet Tubular.pdf"),
    ("Variational-based segmentation of bio-pores in tomographic images", "[2-10] 生物孔隙变分分割 Bio-Pores Segmentation.pdf"),

    # ========== 第二阶段：深入核心 - 3D计算机视觉（第6周）==========
    ("CornerPoint3D look at the nearest corner instead of the center", "[2-11] 3D检测新范式 CornerPoint3D.pdf"),
    ("Neural varifolds an aggregate representation for quantifying geometry of point clouds", "[2-12] 点云神经表示 Neural Varifolds.pdf"),
    ("Detect closer surfaces new modeling in cross-domain 3D object detection", "[2-13] 跨域3D目标检测 Cross-Domain 3D Detection.pdf"),
    ("From instance segmentation to 3D growth trajectory reconstruction", "[2-14] 3D生长轨迹重建 3D Growth Trajectory.pdf"),
    ("3D segmentation of trees through flexible multiclass graph cut", "[2-15] 3D树木分割图割 3D Tree Segmentation.pdf"),
    ("A graph cut approach to 3D tree delineation", "[2-16] 3D树木描绘图割 3D Tree Delineation.pdf"),
    ("3DKMI MATLAB package for shape signatures from Krawtchouk moments", "[2-17] 形状签名Krawtchouk矩 3DKMI.pdf"),
    ("3D orientation field transform", "[2-18] 3D方向场变换 3D Orientation Field.pdf"),
    ("Mapping individual trees from airborne multi-sensor imagery", "[2-19] 多传感器树木映射 Tree Mapping.pdf"),

    # ========== 第二阶段：深入核心 - 医学图像处理（第7周）==========
    ("Deep Rectum Segmentation for Image Guided Radiation Therapy", "[2-20] 放疗直肠分割 Deep Rectum Segmentation.pdf"),
    ("Discrepancy-based diffusion models for lesion detection in brain MRI", "[2-21] 扩散模型脑MRI病变 Diffusion Brain MRI.pdf"),
    ("Automatic contouring of soft organs for image-guided prostate radiotherapy", "[2-22] 前列腺放疗器官勾画 Prostate Radiotherapy.pdf"),
    ("Accuracy of manual and automated rectal contours", "[2-23] 直肠轮廓精度分析 Rectal Contours Accuracy.pdf"),
    ("VoxTox research programme", "[2-24] VoxTox研究计划 VoxTox Programme.pdf"),
    ("Few-shot learning for inference in medical imaging", "[2-25] 医学图像小样本学习 Medical Few-Shot.pdf"),
    ("Non-negative subspace feature representation for few-shot learning", "[2-26] 非负子空间小样本学习 Non-negative Subspace.pdf"),
    ("Medical image classification by incorporating clinical variables", "[2-27] 临床变量医学分类 Medical Classification.pdf"),
    ("IIHT medical report generation with image-to-indicator hierarchical transformer", "[2-28] 医学报告生成IIHT Medical Report Generation.pdf"),
    ("CenSegNet a generalist high-throughput deep learning framework", "[2-29] 中心体分割网络 CenSegNet.pdf"),

    # ========== 第三阶段：前沿探索 - 大模型与高效微调（第8-9周）==========
    ("Less but better parameter-efficient fine-tuning of large language models", "[3-01] 大模型高效微调 LLM Fine-tuning.pdf"),
    ("tCURLoRA tensor CUR decomposition based low-rank parameter adaptation", "[3-02] 张量CUR分解LoRA tCURLoRA.pdf"),
    ("LL4G self-supervised dynamic optimization for graph-based personality detection", "[3-03] 自监督图神经网络LL4G LL4G Graph.pdf"),
    ("Practical sketching algorithms for low-rank tucker approximation", "[3-04] 低秩Tucker近似 sketching Tucker Approximation.pdf"),
    ("An efficient two-sided sketching method for large-scale tensor decomposition", "[3-05] 大规模张量分解 Two-Sided Sketching.pdf"),

    # ========== 第三阶段：前沿探索 - 多模态与跨领域（第10-11周）==========
    ("Talk2Radar bridging natural language with 4D mmWave radar", "[3-06] 雷达语言多模态 Talk2Radar.pdf"),
    ("GAMED knowledge adaptive multi-experts decoupling for multimodal fake news detection", "[3-07] 多模态虚假新闻检测GAMED GAMED Fake News.pdf"),
    ("Mogo residual quantized hierarchical causal transformer for 3D human motion generation", "[3-08] 3D人体运动生成Mogo Mogo Motion Generation.pdf"),
    ("TransNet transfer learning-based network for human action recognition", "[3-09] 迁移学习动作识别 TransNet.pdf"),
    ("Human action recognition based on CNNs and vision transformers", "[3-10] CNN与Transformer动作识别 CNN-ViT Action.pdf"),

    # ========== 第三阶段：前沿探索 - 可解释AI（第12周）==========
    ("Concept-based explainable artificial intelligence metrics and benchmarks", "[3-11] 概念级XAI指标 Concept-based XAI.pdf"),
    ("Multilevel explainable artificial intelligence visual and linguistic explanations", "[3-12] 多层次XAI解释 Multilevel XAI.pdf"),

    # ========== 第三阶段：前沿探索 - 扩散模型与生成（第12周）==========
    ("Discrepancy-based diffusion models for lesion detection in brain MRI", "[3-13] 扩散模型病变检测 Diffusion Lesion Detection.pdf"),

    # ========== 第四阶段：专业深化 - 雷达信号处理（第13-14周）==========
    ("Robust Bayesian attention belief network for radar work mode recognition 2023", "[4-01] 雷达工作模式识别 Radar Work Mode Recognition.pdf"),
    ("DNCNet deep radar signal denoising and recognition", "[4-02] 雷达信号去噪DNCNet DNCNet Radar Denoising.pdf"),
    ("Automatic identification of satellite features from ISAR images", "[4-03] ISAR卫星特征识别 ISAR Satellite.pdf"),
    ("Uncertainty quantification for radio interferometric imaging I", "[4-04] 无线电干涉不确定性I Radio Interferometric I.pdf"),
    ("Online radio interferometric imaging Assimilating and discarding visibilities", "[4-05] 在线无线电干涉成像 Online Radio Imaging.pdf"),
    ("Distributed and parallel sparse convex optimization for radio interferometry", "[4-06] 分布式无线电干涉优化 Distributed Radio Optimization.pdf"),
    ("Quantifying uncertainty in high dimensional inverse problems", "[4-07] 高维逆问题不确定性 High-Dimensional Uncertainty.pdf"),
    ("Proximal nested sampling for high-dimensional Bayesian model selection", "[4-08] 近端嵌套采样 Proximal Nested Sampling.pdf"),
    ("Proximal nested sampling with data-driven priors", "[4-09] 数据驱动先验嵌套采样 Data-Driven Priors.pdf"),

    # ========== 第四阶段：专业深化 - 遥感与植被分析（第15周）==========
    ("Individual tree species classification from airborne multisensor imagery", "[4-10] 多传感器树种分类 Tree Species Classification.pdf"),
    ("Nonparametric image registration of airborne LiDAR hyperspectral", "[4-11] 非参数图像配准 Nonparametric Registration.pdf"),
    ("Wavelet-based segmentation on the sphere", "[4-12] 球面小波分割 Wavelet Sphere.pdf"),
    ("Using computer vision to identify limpets from their shells", "[4-13] 贝壳计算机视觉识别 Limpets Identification.pdf"),
    ("Genes shells and AI Using computer vision to detect cryptic morphological divergence", "[4-14] 基因与形态学分析 Genes Shells AI.pdf"),
    ("The synergy between different colour spaces for degraded images", "[4-15] 颜色空间协同 Colour Spaces.pdf"),

    # ========== 第四阶段：专业深化 - 个性检测与社交计算（第16周）==========
    ("EmoPerso enhancing personality detection with self-supervised emotion-aware modelling", "[4-16] 情感感知个性检测 EmoPerso.pdf"),
    ("Hippd brain-inspired hierarchical information processing for personality detection", "[4-17] 脑启发个性检测Hippd Hippd Brain-Inspired.pdf"),
    ("A computer vision method for finding mislabelled specimens", "[4-18] 错误标记样本检测 Mislabelled Specimens.pdf"),

    # ========== 第四阶段：专业深化 - 其他重要工作 ==========
    ("Balanced neural architecture search and optimization", "[4-19] 神经架构搜索NAS Balanced NAS.pdf"),
    ("Balanced Neural Architecture Search and Its Application in SEI", "[4-20] NAS在SEI应用 NAS for SEI.pdf"),
    ("GRASPTrack geometry-reasoned association via segmentation and projection", "[4-21] 多目标跟踪GRASPTrack GRASPTrack MOT.pdf"),
    ("Revisiting cross-domain problem for LiDAR-based 3D object detection", "[4-22] 跨域LiDAR检测 Cross-Domain LiDAR.pdf"),
    ("A bilevel formalism for the peer-reviewing problem", "[4-23] 双层优化形式化 Bilevel Formalism.pdf"),
    ("Biologically-inspired iterative learning control design", "[4-24] 生物启发迭代学习 Biologically-Inspired ILC.pdf"),
    ("Editorial segmentation and classification theories algorithms and applications", "[4-25] 分割分类社论 Editorial Segmentation.pdf"),
]

def main():
    print("开始按阅读顺序重命名论文...")
    print("="*60)

    # 获取所有PDF文件
    pdf_files = [f for f in os.listdir(download_dir) if f.endswith('.pdf') and not f.startswith('[')]

    renamed_count = 0
    skipped_count = 0

    for original_file in pdf_files:
        original_path = os.path.join(download_dir, original_file)

        # 查找匹配的新文件名
        new_name = None
        for keyword, new_filename in reading_order:
            if keyword.lower() in original_file.lower():
                new_name = new_filename
                break

        if new_name:
            new_path = os.path.join(download_dir, new_name)

            # 检查目标文件是否已存在
            if os.path.exists(new_path):
                print(f"[SKIP] 已存在: {new_name}")
                skipped_count += 1
            else:
                try:
                    shutil.move(original_path, new_path)
                    print(f"[OK] {original_file[:50]}...")
                    print(f"    -> {new_name}")
                    renamed_count += 1
                except Exception as e:
                    print(f"[ERROR] 重命名失败: {e}")
        else:
            print(f"[NOT FOUND] 未找到匹配: {original_file[:50]}...")
            skipped_count += 1

    print("="*60)
    print(f"重命名完成: {renamed_count} 篇")
    print(f"跳过/未匹配: {skipped_count} 篇")

    # 列出重命名后的文件
    print("\n前20个文件预览:")
    renamed_files = sorted([f for f in os.listdir(download_dir) if f.startswith('[')])
    for f in renamed_files[:20]:
        print(f"  {f}")

if __name__ == "__main__":
    main()
