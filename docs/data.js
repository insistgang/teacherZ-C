// 蔡晓浩论文研究可视化系统数据
// 生成时间: 2026-02-20
// 数据来源: web-viewer/00_papers/ (80个实际PDF文件)

const PAPERS_DATA = {
    summary: {
        total: 80,
        filled: 80,
        templates: 0,
        hasPDF: 80,
        noPDF: 0,
        yearRange: "2011-2026"
    },
    categories: {
        "基础理论": { count: 10, filled: 10, color: "#ef4444", description: "图像处理与机器学习的理论基础" },
        "变分分割": { count: 16, filled: 16, color: "#3b82f6", description: "基于变分方法的图像分割技术" },
        "深度学习": { count: 12, filled: 12, color: "#10b981", description: "深度学习在医学影像、雷达等领域的应用" },
        "雷达与无线电": { count: 8, filled: 8, color: "#8b5cf6", description: "雷达信号处理与射电干涉成像" },
        "医学图像": { count: 14, filled: 14, color: "#f59e0b", description: "医学图像重建、分割与分析" },
        "张量分解": { count: 6, filled: 6, color: "#ec4899", description: "张量分解与降维技术" },
        "3D视觉与点云": { count: 14, filled: 14, color: "#06b6d4", description: "3D场景理解、点云处理与树木分析" }
    },
    papers: [
        { id: 1, title: "Thylakoid Electron Tomography", titleEn: "Thylakoid Electron Tomography", year: 2019, category: "医学图像", status: "filled", pdfFile: "2019_Thylakoid_Electron_Tomography.pdf", noteFile: "叶绿体电子断层扫描 Thylakoid.md" },
        { id: 2, title: "Proximal Nested Sampling", titleEn: "Proximal Nested Sampling", year: 2023, category: "基础理论", status: "filled", pdfFile: "2023_2307.00056_Proximal_Nested_Sampling.pdf", noteFile: "Proximal_Nested_Sampling_超精读笔记_已填充.md" },
        { id: 3, title: "Bilevel Peer Review", titleEn: "Bilevel Peer Review", year: 2023, category: "基础理论", status: "filled", pdfFile: "2023_2307.12248_Bilevel_Peer_Review.pdf", noteFile: "Bilevel_Peer-Reviewing_Problem_超精读笔记_已填充.md" },
        { id: 4, title: "Tensor Train Approximation", titleEn: "Tensor Train Approximation", year: 2023, category: "张量分解", status: "filled", pdfFile: "2023_2308.01480_Tensor_Train_Approximation.pdf", noteFile: "Tensor_Train_Approximation_超精读笔记_已填充.md" },
        { id: 5, title: "Equalizing Protected Attributes", titleEn: "Equalizing Protected Attributes", year: 2023, category: "深度学习", status: "filled", pdfFile: "2023_2311.14733_Equalizing_Protected_Attributes.pdf", noteFile: "Equalizing_Protected_Attributes_超精读笔记_已填充.md" },
        { id: 6, title: "Limpets Computer Vision", titleEn: "Limpets Computer Vision", year: 2023, category: "3D视觉与点云", status: "filled", pdfFile: "2023_Limpets_Computer_Vision_Frontiers.pdf", noteFile: "2023_Limpets_Computer_Vision_Identification_超精读笔记.md" },
        { id: 7, title: "Diffusion Brain MRI", titleEn: "Diffusion Brain MRI", year: 2024, category: "医学图像", status: "filled", pdfFile: "2024_2405.04974_Diffusion_Brain_MRI.pdf", noteFile: "Discrepancy-based_Diffusion_MRI_超精读笔记_已填充.md" },
        { id: 8, title: "GRASPTrack Multi-Object Tracking", titleEn: "GRASPTrack", year: 2025, category: "深度学习", status: "filled", pdfFile: "2025_2508.08117_GRASPTrack_MultiObject_Tracking.pdf", noteFile: "GRASPTrack_Multi-Object_Tracking_超精读笔记_已填充.md" },
        { id: 9, title: "HIPPD Brain-Inspired Personality", titleEn: "HIPPD", year: 2025, category: "深度学习", status: "filled", pdfFile: "2025_2510.09893_HIPPD_Brain-Inspired_Personality_Detection.pdf", noteFile: "HIPPD_Brain-Inspired_Personality_Detection_超精读笔记_已填充.md" },
        { id: 10, title: "3D Growth Trajectory Reconstruction", titleEn: "3D Growth Trajectory", year: 2025, category: "3D视觉与点云", status: "filled", pdfFile: "2025_2511.02142_3D_Growth_Trajectory_Reconstruction.pdf", noteFile: "3D_Growth_Trajectory_Reconstruction_超精读笔记_已填充.md" },
        { id: 11, title: "MotionDuet 3D Motion Generation", titleEn: "MotionDuet", year: 2025, category: "3D视觉与点云", status: "filled", pdfFile: "2025_2511.18209_MotionDuet_3D_Motion_Generation.pdf", noteFile: "MotionDuet_3D_Motion_Generation_超精读笔记_已填充.md" },
        { id: 12, title: "Genes Shells AI", titleEn: "Genes Shells AI", year: 2025, category: "深度学习", status: "filled", pdfFile: "2025_Genes_Shells_AI_Scientific_Reports.pdf", noteFile: "2025_Genes_Shells_AI_Scientific_Reports_超精读笔记.md" },
        { id: 13, title: "3DKMI Krawtchouk矩形状签名", titleEn: "3DKMI", year: 2015, category: "3D视觉与点云", status: "filled", pdfFile: "3DKMI Krawtchouk矩形状签名 3DKMI.pdf", noteFile: "3DKMI Krawtchouk矩形状签名 3DKMI.md" },
        { id: 14, title: "3D方向场变换", titleEn: "3D Orientation Field", year: 2020, category: "3D视觉与点云", status: "filled", pdfFile: "3D方向场变换 3D Orientation Field.pdf", noteFile: "3D_Orientation_Field_Transform_超精读笔记_已填充.md" },
        { id: 15, title: "3D树木分割图", titleEn: "3D Tree Segmentation", year: 2019, category: "3D视觉与点云", status: "filled", pdfFile: "3D树木分割图 3D Tree Segmentation.pdf", noteFile: "3D_Tree_Segmentation_MCGC_超精读笔记_已填充.md" },
        { id: 16, title: "3D树木分割图割", titleEn: "3D Tree Graph Cut", year: 2017, category: "3D视觉与点云", status: "filled", pdfFile: "3D树木分割图割 3D Tree Graph Cut.pdf", noteFile: "3D_Tree_Delineation_Graph_Cut_超精读笔记_已填充.md" },
        { id: 17, title: "3D树木描绘图割", titleEn: "3D Tree Delineation", year: 2017, category: "3D视觉与点云", status: "filled", pdfFile: "3D树木描绘图割 3D Tree Delineation.pdf", noteFile: "3D_Tree_Delineation_Graph_Cut_超精读笔记_已填充.md" },
        { id: 18, title: "Biologically-Inspired ILC", titleEn: "ILC", year: 2020, category: "基础理论", status: "filled", pdfFile: "Biologically-Inspired_Iterative_Learning_Control_Design_A_Modular-Based_Approach.pdf", noteFile: "Biologically-Inspired_Iterative_Learning_Control_Design_A_Modular-Based_Approach.md" },
        { id: 19, title: "CenSegNet中心体", titleEn: "CenSegNet", year: 2025, category: "医学图像", status: "filled", pdfFile: "CenSegNet中心体 CenSegNet.pdf", noteFile: "CenSegNet_中心体表型分析_超精读笔记.md" },
        { id: 20, title: "CornerPoint3D 3D检测新尺度", titleEn: "CornerPoint3D", year: 2025, category: "3D视觉与点云", status: "filled", pdfFile: "CornerPoint3D 3D检测新尺度 CornerPoint3D.pdf", noteFile: "CornerPoint3D_Nearest_Corner_3D_Detection_超精读笔记_已填充.md" },
        { id: 21, title: "DNCNet雷达去噪", titleEn: "DNCNet", year: 2022, category: "雷达与无线电", status: "filled", pdfFile: "DNCNet雷达去噪 DNCNet.pdf", noteFile: "DNCNet雷达去噪 DNCNet.md" },
        { id: 22, title: "GAMED虚假新闻检测", titleEn: "GAMED", year: 2025, category: "深度学习", status: "filled", pdfFile: "GAMED虚假新闻检测 GAMED Fake News.pdf", noteFile: "GAMED_Multimodal_Fake_News_Detection_超精读笔记_已填充.md" },
        { id: 23, title: "HiFi-Mamba MRI重建", titleEn: "HiFi-Mamba", year: 2025, category: "医学图像", status: "filled", pdfFile: "HiFi-Mamba MRI重建 HiFi-Mamba MRI Reconstruction.pdf", noteFile: "HiFi-Mamba_MRI_Reconstruction_超精读笔记_已填充.md" },
        { id: 24, title: "HiFi-MambaV2分层MRI", titleEn: "HiFi-MambaV2", year: 2025, category: "医学图像", status: "filled", pdfFile: "HiFi-MambaV2分层MRI HiFi-MambaV2 Hierarchical MRI.pdf", noteFile: "HiFi-MambaV2_Hierarchical_MRI_超精读笔记_已填充.md" },
        { id: 25, title: "ISAR卫星特征识别", titleEn: "ISAR", year: 2020, category: "雷达与无线电", status: "filled", pdfFile: "ISAR卫星特征识别 ISAR.pdf", noteFile: "ISAR_Satellite_Feature_Recognition_超精读笔记.md" },
        { id: 26, title: "MOGO 3D人体运动生成", titleEn: "MOGO", year: 2025, category: "3D视觉与点云", status: "filled", pdfFile: "MOGO 3D人体运动生成 MOGO Motion.pdf", noteFile: "MOGO_3D_Motion_Generation_超精读笔记.md" },
        { id: 27, title: "PURIFY分布式优化", titleEn: "PURIFY", year: 2018, category: "雷达与无线电", status: "filled", pdfFile: "PURIFY分布式优化 PURIFY.pdf", noteFile: "PURIFY分布式优化 PURIFY.md" },
        { id: 28, title: "RobustPCA树木分类", titleEn: "Robust PCA Trees", year: 2019, category: "3D视觉与点云", status: "filled", pdfFile: "RobustPCA树木分类 Robust PCA Trees.pdf", noteFile: "RobustPCA树木分类 Robust PCA Trees.md" },
        { id: 29, title: "SLaT三阶段分割", titleEn: "SLaT", year: 2015, category: "变分分割", status: "filled", pdfFile: "SLaT三阶段分割 SLaT Segmentation.pdf", noteFile: "SLaT_Three-stage_Segmentation_超精读笔记_已填充.md" },
        { id: 30, title: "Talk2Radar 雷达语言多模态", titleEn: "Talk2Radar", year: 2025, category: "雷达与无线电", status: "filled", pdfFile: "Talk2Radar 雷达语言多模态 Talk2Radar.pdf", noteFile: "Talk2Radar_Language_Radar_Multimodal_超精读笔记_已填充.md" },
        { id: 31, title: "TransNet动作识别", titleEn: "TransNet", year: 2023, category: "深度学习", status: "filled", pdfFile: "TransNet动作识别 TransNet HAR.pdf", noteFile: "TransNet_Transfer_Learning_HAR_超精读笔记_已填充.md" },
        { id: 32, title: "Tucker近似", titleEn: "Tucker Approximation", year: 2023, category: "张量分解", status: "filled", pdfFile: "Tucker近似 Tucker Approximation.pdf", noteFile: "Practical_Sketching_Tucker_Approximation_超精读笔记_已填充.md" },
        { id: 33, title: "变分分割基础Mumford-Shah与ROF", titleEn: "Mumford-Shah ROF", year: 2018, category: "变分分割", status: "filled", pdfFile: "变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf", noteFile: "Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md" },
        { id: 34, title: "船舶匹配遥感", titleEn: "Ship Matching", year: 2023, category: "雷达与无线电", status: "filled", pdfFile: "船舶匹配遥感 Ship Matching.pdf", noteFile: "Ship_Matching_Remote_Sensing_超精读笔记.md" },
        { id: 35, title: "大模型高效微调CALM", titleEn: "CALM Fine-tuning", year: 2026, category: "深度学习", status: "filled", pdfFile: "大模型高效微调CALM CALM Fine-tuning.pdf", noteFile: "CALM_Culturally_Aware_Language_Model_超精读笔记_已填充.md" },
        { id: 36, title: "蛋白质结构网络图LL4G", titleEn: "LL4G Graph", year: 2025, category: "深度学习", status: "filled", pdfFile: "蛋白质结构网络图LL4G LL4G Graph.pdf", noteFile: "LL4G_Graph-Based_Personality_Detection_超精读笔记_已填充.md" },
        { id: 37, title: "点云神经表示", titleEn: "Neural Varifolds", year: 2025, category: "3D视觉与点云", status: "filled", pdfFile: "点云神经表示 Neural Varifolds.pdf", noteFile: "Neural_varifolds_quantifying_point_cloud_geometry_超精读笔记_已填充.md" },
        { id: 38, title: "多层次可解释AI", titleEn: "Multilevel XAI", year: 2023, category: "基础理论", status: "filled", pdfFile: "多层次可解释AI Multilevel XAI.pdf", noteFile: "Multilevel_Explainable_AI_Review_超精读笔记.md" },
        { id: 39, title: "多传感器树木制图", titleEn: "Multi-Sensor Trees", year: 2017, category: "3D视觉与点云", status: "filled", pdfFile: "多传感器树木制图 Multi-Sensor Trees.pdf", noteFile: "多传感器树木制图 Multi-Sensor Trees.md" },
        { id: 40, title: "多功能传感器树木分类", titleEn: "Tree Species Classification", year: 2017, category: "3D视觉与点云", status: "filled", pdfFile: "多功能传感器树木分类 Tree Species Classification.pdf", noteFile: "多功能传感器树木分类 Tree Species Classification.md" },
        { id: 41, title: "多类ROF分割", titleEn: "Iterated ROF", year: 2018, category: "变分分割", status: "filled", pdfFile: "多类ROF分割 Iterated ROF.pdf", noteFile: "多类分割迭代ROF_Iterated_ROF_超精读笔记_已填充.md" },
        { id: 42, title: "非负子空间小样本学习", titleEn: "Non-negative Subspace", year: 2024, category: "深度学习", status: "filled", pdfFile: "非负子空间小样本学习 Non-negative Subspace.pdf", noteFile: "Non-negative_Subspace_Few-Shot_Learning_超精读笔记_已填充.md" },
        { id: 43, title: "分布式无线电优化", titleEn: "Distributed Radio", year: 2018, category: "雷达与无线电", status: "filled", pdfFile: "分布式无线电优化 Distributed Radio Optimization.pdf", noteFile: "Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md" },
        { id: 44, title: "分割方法论总览", titleEn: "SaT Overview", year: 2023, category: "变分分割", status: "filled", pdfFile: "分割方法论总览 SaT Overview.pdf", noteFile: "分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md" },
        { id: 45, title: "分割恢复联合模型", titleEn: "Segmentation Restoration", year: 2014, category: "变分分割", status: "filled", pdfFile: "分割恢复联合模型 Segmentation Restoration.pdf", noteFile: "Variational_Segmentation-Restoration_超精读笔记_已填充.md" },
        { id: 46, title: "概念级XAI指标", titleEn: "Concept XAI", year: 2025, category: "基础理论", status: "filled", pdfFile: "概念级XAI指标 Concept XAI.pdf", noteFile: "Concept-Based_XAI_Metrics_超精读笔记_已填充.md" },
        { id: 47, title: "高维逆问题不确定性量化", titleEn: "Uncertainty Quantification", year: 2018, category: "基础理论", status: "filled", pdfFile: "高维逆问题不确定性量化 Uncertainty Quantification.pdf", noteFile: "High-Dimensional_Inverse_Problems_超精读笔记_已填充.md" },
        { id: 48, title: "高效PEFT微调", titleEn: "Less but Better PEFT", year: 2025, category: "深度学习", status: "filled", pdfFile: "高效PEFT微调 Less but Better PEFT.pdf", noteFile: "Less_but_Better_PEFT_for_Personality_Detection_超精读笔记_已填充.md" },
        { id: 49, title: "高效变分分类", titleEn: "Efficient Variational", year: 2016, category: "变分分割", status: "filled", pdfFile: "高效变分分类 Efficient Variational.pdf", noteFile: "高效变分分类方法_Efficient_Variational_Classification_超精读笔记_已填充.md" },
        { id: 50, title: "近端嵌套采样", titleEn: "Proximal Nested", year: 2021, category: "基础理论", status: "filled", pdfFile: "近端嵌套采样 Proximal Nested Sampling.pdf", noteFile: "近端嵌套采样 Proximal Nested Sampling.md" },
        { id: 51, title: "可见表面检测", titleEn: "Detect Closer Surfaces", year: 2024, category: "3D视觉与点云", status: "filled", pdfFile: "可见表面检测 Detect Closer Surfaces.pdf", noteFile: "Detect_Closer_Surfaces_3D_Detection_超精读笔记_已填充.md" },
        { id: 52, title: "可解释AI综述", titleEn: "XAI Survey", year: 2023, category: "基础理论", status: "filled", pdfFile: "可解释AI综述 XAI Survey.pdf", noteFile: "可解释AI综述 XAI Survey.md" },
        { id: 53, title: "跨域LiDAR检测", titleEn: "Cross-Domain LiDAR", year: 2024, category: "3D视觉与点云", status: "filled", pdfFile: "跨域LiDAR检测 Cross-Domain LiDAR.pdf", noteFile: "Cross-Domain_LiDAR_Detection_超精读笔记_已填充.md" },
        { id: 54, title: "框架分割管状结构", titleEn: "Framelet Tubular", year: 2011, category: "变分分割", status: "filled", pdfFile: "框架分割管状结构 Framelet Tubular.pdf", noteFile: "框架分割管状结构_Framelet_Tubular_超精读笔记_已填充.md" },
        { id: 55, title: "框架管状结构分割", titleEn: "Framelet", year: 2011, category: "变分分割", status: "filled", pdfFile: "框架管状结构分割 Framelet.pdf", noteFile: "框架分割管状结构_Framelet_Tubular_超精读笔记_已填充.md" },
        { id: 56, title: "雷达工作模式识别", titleEn: "Radar Work Mode", year: 2020, category: "雷达与无线电", status: "filled", pdfFile: "雷达工作模式识别 Radar Work Mode.pdf", noteFile: "雷达工作模式识别 Radar Work Mode.md" },
        { id: 57, title: "两阶段分类", titleEn: "Two-Stage", year: 2019, category: "变分分割", status: "filled", pdfFile: "两阶段分类 Two-Stage.pdf", noteFile: "Two_Stage_High_Dimensional_Classification_超精读笔记_已填充.md" },
        { id: 58, title: "平衡神经网络搜索I", titleEn: "Balanced NAS I", year: 2020, category: "深度学习", status: "filled", pdfFile: "平衡神经网络搜索I Balanced NAS I.pdf", noteFile: "平衡神经网络搜索I Balanced NAS I.md" },
        { id: 59, title: "平衡神经网络搜索II", titleEn: "Balanced NAS II", year: 2020, category: "深度学习", status: "filled", pdfFile: "平衡神经网络搜索II Balanced NAS II.pdf", noteFile: "平衡神经网络搜索II Balanced NAS II.md" },
        { id: 60, title: "情感感知人格检测", titleEn: "EmoPerso", year: 2025, category: "深度学习", status: "filled", pdfFile: "情感感知人格检测 EmoPerso Emotion-Aware.pdf", noteFile: "EmoPerso_Emotion-Aware_Personality_Detection_超精读笔记_已填充.md" },
        { id: 61, title: "球面小波分割", titleEn: "Wavelet Sphere", year: 2016, category: "变分分割", status: "filled", pdfFile: "球面小波分割 Wavelet Sphere.pdf", noteFile: "Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md" },
        { id: 62, title: "深度学习架构综述", titleEn: "CNNs RNNs Transformers", year: 2024, category: "深度学习", status: "filled", pdfFile: "深度学习架构综述 CNNs RNNs Transformers.pdf", noteFile: "CNNs_RNNs_Transformers_HAR_Survey_超精读笔记_已填充.md" },
        { id: 63, title: "生物孔隙分割", titleEn: "Bio-Pores", year: 2016, category: "变分分割", status: "filled", pdfFile: "生物孔隙分割 Bio-Pores.pdf", noteFile: "生物孔隙变分分割_Bio-Pores_Segmentation_超精读笔记_已填充.md" },
        { id: 64, title: "生物启发迭代学习控制", titleEn: "ILC", year: 2020, category: "基础理论", status: "filled", pdfFile: "生物启发迭代学习控制 ILC.pdf", noteFile: "生物启发迭代学习控制 ILC.md" },
        { id: 65, title: "数据增强综述", titleEn: "Data Augmentation", year: 2023, category: "深度学习", status: "filled", pdfFile: "数据增强综述 Data Augmentation.pdf", noteFile: "数据增强综述 Data Augmentation.md" },
        { id: 66, title: "双面Sketching张量", titleEn: "Two-Sided Sketching", year: 2018, category: "张量分解", status: "filled", pdfFile: "双面Sketching张量 Two-Sided Sketching.pdf", noteFile: "大规模张量分解_Large_Scale_Tensor_Decomposition_超精读笔记_已填充.md" },
        { id: 67, title: "无线电干涉不确定性I", titleEn: "Radio Interferometric I", year: 2017, category: "雷达与无线电", status: "filled", pdfFile: "无线电干涉不确定性I Radio Interferometric I.pdf", noteFile: "Radio_Interferometric_Imaging_I_超精读笔记_已填充.md" },
        { id: 68, title: "无线电干涉不确定性II", titleEn: "Radio Interferometric II", year: 2017, category: "雷达与无线电", status: "filled", pdfFile: "无线电干涉不确定性II Radio Interferometric II.pdf", noteFile: "Radio_Interferometric_Imaging_II_超精读笔记_已填充.md" },
        { id: 69, title: "稀疏贝叶斯质量映射I", titleEn: "Sparse Bayesian I", year: 2021, category: "雷达与无线电", status: "filled", pdfFile: "稀疏贝叶斯质量映射I Sparse Bayesian I.pdf", noteFile: "稀疏贝叶斯质量映射I Sparse Bayesian I.md" },
        { id: 70, title: "稀疏贝叶斯质量映射II", titleEn: "Sparse Bayesian II", year: 2021, category: "雷达与无线电", status: "filled", pdfFile: "稀疏贝叶斯质量映射II Sparse Bayesian II.pdf", noteFile: "稀疏贝叶斯质量映射II Sparse Bayesian II.md" },
        { id: 71, title: "稀疏贝叶斯质量映射III", titleEn: "Sparse Bayesian III", year: 2021, category: "雷达与无线电", status: "filled", pdfFile: "稀疏贝叶斯质量映射III Sparse Bayesian III.pdf", noteFile: "稀疏贝叶斯质量映射III Sparse Bayesian III.md" },
        { id: 72, title: "叶绿体电子断层扫描", titleEn: "Thylakoid", year: 2019, category: "医学图像", status: "filled", pdfFile: "叶绿体电子断层扫描 Thylakoid.pdf", noteFile: "叶绿体电子断层扫描 Thylakoid.md" },
        { id: 73, title: "医学报告生成IIHT", titleEn: "Medical Report IIHT", year: 2023, category: "医学图像", status: "filled", pdfFile: "医学报告生成IIHT Medical Report IIHT.pdf", noteFile: "IIHT_Medical_Report_Generation_超精读笔记_已填充.md" },
        { id: 74, title: "医学图像分类", titleEn: "Medical Classification", year: 2023, category: "医学图像", status: "filled", pdfFile: "医学图像分类 Medical Classification.pdf", noteFile: "医学图像分类 Medical Classification.md" },
        { id: 75, title: "医学图像小样本学习", titleEn: "Medical Few-Shot", year: 2023, category: "医学图像", status: "filled", pdfFile: "医学图像小样本学习 Medical Few-Shot.pdf", noteFile: "Few-shot_Medical_Imaging_Inference_超精读笔记_已填充.md" },
        { id: 76, title: "语义比例分割", titleEn: "Semantic Proportions", year: 2023, category: "变分分割", status: "filled", pdfFile: "语义比例分割 Semantic Proportions.pdf", noteFile: "Semantic_Segmentation_by_Proportions_超精读笔记_已填充.md" },
        { id: 77, title: "在线无线电干涉成像", titleEn: "Online Radio Imaging", year: 2017, category: "雷达与无线电", status: "filled", pdfFile: "在线无线电干涉成像 Online Radio Imaging.pdf", noteFile: "Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md" },
        { id: 78, title: "张量CUR分解LoRA", titleEn: "tCURLoRA", year: 2025, category: "张量分解", status: "filled", pdfFile: "张量CUR分解LoRA tCURLoRA.pdf", noteFile: "tCURLoRA_Tensor_CUR_for_Medical_Imaging_超精读笔记_已填充.md" },
        { id: 79, title: "直肠分割放疗", titleEn: "Deep Rectum", year: 2023, category: "医学图像", status: "filled", pdfFile: "直肠分割放疗 Deep Rectum.pdf", noteFile: "Deep_Learning_Rectum_Segmentation_Radiotherapy_超精读笔记.md" },
        { id: 80, title: "自然历史馆藏错误标本", titleEn: "Mislabelled Specimens", year: 2023, category: "深度学习", status: "filled", pdfFile: "自然历史馆藏错误标本能计算机视觉方法 Mislabelled Specimens.pdf", noteFile: "自然历史馆藏错误标本能计算机视觉方法 Mislabelled Specimens.md" }
    ],
    timeline: [
        { period: "2011-2015", count: 6, description: "早期变分方法与分割基础" },
        { period: "2016-2020", count: 25, description: "方法扩展与3D应用" },
        { period: "2021-2023", count: 28, description: "深度学习融合与理论成熟" },
        { period: "2024-2026", count: 21, description: "前沿创新：大模型、Mamba、多模态" }
    ],
    citations: [
        { source: 54, target: 29, type: "predecessor", description: "Tight-Frame → SLaT三阶段" },
        { source: 29, target: 45, type: "extension", description: "SLaT → 联合分割恢复" },
        { source: 45, target: 33, type: "theory", description: "分割恢复 → MS-ROF联系" },
        { source: 33, target: 41, type: "extension", description: "MS-ROF → 多类ROF" },
        { source: 23, target: 24, type: "improvement", description: "HiFi-Mamba → HiFi-MambaV2" },
        { source: 2, target: 50, type: "extension", description: "近端采样 → 嵌套采样" },
        { source: 67, target: 68, type: "series", description: "无线电干涉I → II" },
        { source: 68, target: 77, type: "extension", description: "干涉II → 在线成像" },
        { source: 32, target: 4, type: "alternative", description: "Tucker → TT近似" },
        { source: 66, target: 32, type: "improvement", description: "双边Sketching → Tucker" },
        { source: 78, target: 66, type: "application", description: "tCURLoRA → 张量分解" },
        { source: 14, target: 15, type: "extension", description: "3D方向场 → 树木分割" },
        { source: 15, target: 16, type: "improvement", description: "树木分割 → 图割" },
        { source: 16, target: 10, type: "extension", description: "图割 → 生长轨迹" },
        { source: 7, target: 23, type: "predecessor", description: "扩散MRI → HiFi-Mamba" },
        { source: 8, target: 11, type: "related", description: "GRASPTrack → MotionDuet" },
        { source: 61, target: 63, type: "extension", description: "球面小波 → 生物孔隙" },
        { source: 57, target: 49, type: "related", description: "两阶段 → 高效分类" }
    ],
    methods: {
        "变分分割": {
            description: "基于变分原理的图像分割方法",
            papers: [29, 33, 41, 45, 54, 55, 61, 63, 76],
            evolution: "Tight-Frame → SLaT → MS-ROF → 多类ROF"
        },
        "MRI重建": {
            description: "磁共振图像快速重建技术",
            papers: [7, 23, 24],
            evolution: "传统方法 → 扩散模型 → HiFi-Mamba → HiFi-MambaV2"
        },
        "张量分解": {
            description: "高维数据的张量分解与降维",
            papers: [4, 32, 66, 78],
            evolution: "Tucker → TT → 双边Sketching → tCURLoRA"
        },
        "3D视觉": {
            description: "三维场景理解与点云处理",
            papers: [10, 11, 14, 15, 16, 17, 20, 26, 37, 51, 53],
            evolution: "3D方向场 → 树木分割 → 生长轨迹"
        },
        "射电干涉": {
            description: "射电天文干涉成像方法",
            papers: [2, 43, 50, 67, 68, 69, 70, 71, 77],
            evolution: "干涉I → 干涉II → 在线成像"
        }
    }
};

// 工具函数
const Utils = {
    getCategoryColor: (category) => {
        const colors = {
            "基础理论": "#ef4444",
            "变分分割": "#3b82f6",
            "深度学习": "#10b981",
            "雷达与无线电": "#8b5cf6",
            "医学图像": "#f59e0b",
            "张量分解": "#ec4899",
            "3D视觉与点云": "#06b6d4"
        };
        return colors[category] || "#6b7280";
    },

    getCategoryClass: (category) => {
        const classes = {
            "基础理论": "category-theory",
            "变分分割": "category-segmentation",
            "深度学习": "category-dl",
            "雷达与无线电": "category-radar",
            "医学图像": "category-medical",
            "张量分解": "category-tensor",
            "3D视觉与点云": "category-3d"
        };
        return classes[category] || "category-other";
    },

    getPDFPath: (filename) => {
        if (!filename) return null;
        return '00_papers/' + encodeURIComponent(filename);
    },

    getNotePath: (filename) => {
        if (!filename) return null;
        return 'notes/' + encodeURIComponent(filename);
    },

    getPaperById: (id) => {
        return PAPERS_DATA.papers.find(p => p.id === id);
    },

    getPapersByCategory: (category) => {
        return PAPERS_DATA.papers.filter(p => p.category === category);
    },

    searchPapers: (query) => {
        const q = query.toLowerCase();
        return PAPERS_DATA.papers.filter(p =>
            p.title.toLowerCase().includes(q) ||
            (p.titleEn && p.titleEn.toLowerCase().includes(q)) ||
            (p.category && p.category.toLowerCase().includes(q))
        );
    }
};
