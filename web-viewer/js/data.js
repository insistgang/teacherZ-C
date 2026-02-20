/**
 * Xiaohao Cai 论文数据
 * 版本: 2.0 (优化版)
 * 更新: 修复笔记文件路径，匹配实际存在的文件
 */

const PAPERS_DATA = {
    summary: {
        total: 80,
        filled: 80,
        templates: 0,
        hasPDF: 80,
        noPDF: 0,
        hasNote: 57  // 实际有笔记的论文数量
    },
    categories: {
        "基础理论": { count: 12, filled: 12, hasPDF: 12, hasNote: 5, color: "#ef4444" },
        "变分分割": { count: 14, filled: 14, hasPDF: 14, hasNote: 10, color: "#3b82f6" },
        "深度学习": { count: 10, filled: 10, hasPDF: 10, hasNote: 7, color: "#10b981" },
        "雷达与无线电": { count: 15, filled: 15, hasPDF: 15, hasNote: 7, color: "#8b5cf6" },
        "医学图像": { count: 10, filled: 10, hasPDF: 10, hasNote: 4, color: "#f59e0b" },
        "其他": { count: 19, filled: 19, hasPDF: 19, hasNote: 24, color: "#6b7280" }
    },
    papers: [
        // ===== 基础理论 (12篇) =====
        { 
            id: "1-01", 
            title: "CNNs、RNNs与Transformer在人体动作识别中的综合综述", 
            category: "基础理论", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "CNNs_RNNs_Transformers_HAR_Survey_超精读笔记_已填充.md", 
            pdfFile: "深度学习架构综述 CNNs RNNs Transformers.pdf" 
        },
        { 
            id: "1-02", 
            title: "深度学习架构综述", 
            category: "基础理论", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "深度学习架构综述 CNNs RNNs Transformers.pdf" 
        },
        { 
            id: "1-03", 
            title: "TransNet动作识别", 
            category: "基础理论", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "TransNet_Transfer_Learning_HAR_超精读笔记_已填充.md", 
            pdfFile: "TransNet动作识别 TransNet HAR.pdf" 
        },
        { 
            id: "1-04", 
            title: "变分分割基础Mumford-Shah与ROF", 
            category: "基础理论", 
            year: "2018", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md", 
            pdfFile: "变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf" 
        },
        { 
            id: "1-05", 
            title: "两阶段分类方法", 
            category: "基础理论", 
            year: "2013", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Two_Stage_High_Dimensional_Classification_超精读笔记_已填充.md", 
            pdfFile: "两阶段分类 Two-Stage.pdf" 
        },
        { 
            id: "1-06", 
            title: "概念级可解释AI指标", 
            category: "基础理论", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Concept-Based_XAI_Metrics_超精读笔记_已填充.md", 
            pdfFile: "概念级XAI指标 Concept XAI.pdf" 
        },
        { 
            id: "1-07", 
            title: "可解释AI综述", 
            category: "基础理论", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "可解释AI综述 XAI Survey.pdf" 
        },
        { 
            id: "1-08", 
            title: "多层次可解释AI", 
            category: "基础理论", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "多层次可解释AI Multilevel XAI.pdf" 
        },
        { 
            id: "1-09", 
            title: "数据增强综述", 
            category: "基础理论", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "数据增强综述 Data Augmentation.pdf" 
        },
        { 
            id: "1-10", 
            title: "高维逆问题不确定性量化", 
            category: "基础理论", 
            year: "2018", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "High-Dimensional_Inverse_Problems_超精读笔记_已填充.md", 
            pdfFile: "高维逆问题不确定性量化 Uncertainty Quantification.pdf" 
        },
        { 
            id: "1-11", 
            title: "分割方法论总览", 
            category: "基础理论", 
            year: "2019", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md", 
            pdfFile: "分割方法论总览 SaT Overview.pdf" 
        },
        { 
            id: "1-12", 
            title: "高效变分分类", 
            category: "基础理论", 
            year: "2021", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "高效变分分类方法_Efficient_Variational_Classification_超精读笔记_已填充.md", 
            pdfFile: "高效变分分类 Efficient Variational.pdf" 
        },

        // ===== 变分分割 (14篇) =====
        { 
            id: "2-01", 
            title: "SLaT三阶段分割", 
            category: "变分分割", 
            year: "2015", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "SLaT_Three-stage_Segmentation_超精读笔记_已填充.md", 
            pdfFile: "SLaT三阶段分割 SLaT Segmentation.pdf" 
        },
        { 
            id: "2-02", 
            title: "分割与恢复联合模型", 
            category: "变分分割", 
            year: "2016", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Variational_Segmentation-Restoration_超精读笔记_已填充.md", 
            pdfFile: "分割恢复联合模型 Segmentation Restoration.pdf" 
        },
        { 
            id: "2-03", 
            title: "语义比例分割", 
            category: "变分分割", 
            year: "2017", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Semantic_Segmentation_by_Proportions_超精读笔记_已填充.md", 
            pdfFile: "语义比例分割 Semantic Proportions.pdf" 
        },
        { 
            id: "2-04", 
            title: "框架分割管状结构", 
            category: "变分分割", 
            year: "2013", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "框架分割管状结构_Framelet_Tubular_超精读笔记_已填充.md", 
            pdfFile: "框架分割管状结构 Framelet Tubular.pdf" 
        },
        { 
            id: "2-05", 
            title: "框架管状结构分割", 
            category: "变分分割", 
            year: "2021", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Tight-Frame_Vessel_Segmentation_2011_超精读笔记_已填充.md", 
            pdfFile: "框架管状结构分割 Framelet.pdf" 
        },
        { 
            id: "2-06", 
            title: "3D方向场变换", 
            category: "变分分割", 
            year: "2020", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "3D_Orientation_Field_Transform_超精读笔记_已填充.md", 
            pdfFile: "3D方向场变换 3D Orientation Field.pdf" 
        },
        { 
            id: "2-07", 
            title: "CornerPoint3D 3D检测新尺度", 
            category: "变分分割", 
            year: "2020", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "CornerPoint3D_Nearest_Corner_3D_Detection_超精读笔记_已填充.md", 
            pdfFile: "CornerPoint3D 3D检测新尺度 CornerPoint3D.pdf" 
        },
        { 
            id: "2-08", 
            title: "可见表面检测", 
            category: "变分分割", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Detect_Closer_Surfaces_3D_Detection_超精读笔记_已填充.md", 
            pdfFile: "可见表面检测 Detect Closer Surfaces.pdf" 
        },
        { 
            id: "2-09", 
            title: "球面小波分割", 
            category: "变分分割", 
            year: "2016", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md", 
            pdfFile: "球面小波分割 Wavelet Sphere.pdf" 
        },
        { 
            id: "2-10", 
            title: "CenSegNet中心体分割", 
            category: "变分分割", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "CenSegNet_中心体表型分析_超精读笔记.md", 
            pdfFile: "CenSegNet中心体 CenSegNet.pdf" 
        },
        { 
            id: "2-11", 
            title: "3D树木分割图", 
            category: "变分分割", 
            year: "2019", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "3D_Tree_Segmentation_MCGC_超精读笔记_已填充.md", 
            pdfFile: "3D树木分割图 3D Tree Segmentation.pdf" 
        },
        { 
            id: "2-12", 
            title: "3D树木分割图割", 
            category: "变分分割", 
            year: "2020", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "3D树木分割图割 3D Tree Graph Cut.pdf" 
        },
        { 
            id: "2-13", 
            title: "3D树木描绘图割", 
            category: "变分分割", 
            year: "2020", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "3D_Tree_Delineation_Graph_Cut_超精读笔记_已填充.md", 
            pdfFile: "3D树木描绘图割 3D Tree Delineation.pdf" 
        },
        { 
            id: "2-14", 
            title: "3DKMI球状正交矩", 
            category: "变分分割", 
            year: "2022", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "3DKMI Krawtchouk矩形状签名 3DKMI.pdf" 
        },

        // ===== 深度学习 (10篇) =====
        { 
            id: "3-01", 
            title: "大模型高效微调CALM", 
            category: "深度学习", 
            year: "2026", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "CALM_Culturally_Aware_Language_Model_超精读笔记_已填充.md", 
            pdfFile: "大模型高效微调CALM CALM Fine-tuning.pdf" 
        },
        { 
            id: "3-02", 
            title: "张量CUR分解LoRA", 
            category: "深度学习", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "张量CUR分解LoRA_Tensor_CUR_LoRA_超精读笔记_已填充.md", 
            pdfFile: "张量CUR分解LoRA tCURLoRA.pdf" 
        },
        { 
            id: "3-03", 
            title: "蛋白质结构网络图LL4G", 
            category: "深度学习", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "LL4G_Graph-Based_Personality_Detection_超精读笔记_已填充.md", 
            pdfFile: "蛋白质结构网络图LL4G LL4G Graph.pdf" 
        },
        { 
            id: "3-04", 
            title: "Tucker近似", 
            category: "深度学习", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "Tucker近似 Tucker Approximation.pdf" 
        },
        { 
            id: "3-05", 
            title: "低秩Tucker逼近", 
            category: "深度学习", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Practical_Sketching_Tucker_Approximation_超精读笔记_已填充.md", 
            pdfFile: "2023_2308.01480_Tensor_Train_Approximation.pdf" 
        },
        { 
            id: "3-06", 
            title: "大规模张量分解", 
            category: "深度学习", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "大规模张量分解_Large_Scale_Tensor_Decomposition_超精读笔记_已填充.md", 
            pdfFile: "双面Sketching张量 Two-Sided Sketching.pdf" 
        },
        { 
            id: "3-07", 
            title: "GAMED虚假新闻检测", 
            category: "深度学习", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "GAMED_Multimodal_Fake_News_Detection_超精读笔记_已填充.md", 
            pdfFile: "GAMED虚假新闻检测 GAMED Fake News.pdf" 
        },
        { 
            id: "3-08", 
            title: "MOGO 3D人体运动生成", 
            category: "深度学习", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "MOGO 3D人体运动生成 MOGO Motion.pdf" 
        },
        { 
            id: "3-09", 
            title: "MotionDuet运动生成", 
            category: "深度学习", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "MotionDuet_3D_Motion_Generation_超精读笔记_已填充.md", 
            pdfFile: "2025_2511.18209_MotionDuet_3D_Motion_Generation.pdf" 
        },
        { 
            id: "3-10", 
            title: "高效PEFT微调", 
            category: "深度学习", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Less_but_Better_PEFT_for_Personality_Detection_超精读笔记_已填充.md", 
            pdfFile: "高效PEFT微调 Less but Better PEFT.pdf" 
        },

        // ===== 雷达与无线电 (15篇) =====
        { 
            id: "4-01", 
            title: "Talk2Radar语言-雷达多模态", 
            category: "雷达与无线电", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Talk2Radar_Language_Radar_Multimodal_超精读笔记_已填充.md", 
            pdfFile: "Talk2Radar 雷达语言多模态 Talk2Radar.pdf" 
        },
        { 
            id: "4-02", 
            title: "无线电干涉不确定性I", 
            category: "雷达与无线电", 
            year: "2017", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Radio_Interferometric_Imaging_I_超精读笔记_已填充.md", 
            pdfFile: "无线电干涉不确定性I Radio Interferometric I.pdf" 
        },
        { 
            id: "4-03", 
            title: "无线电干涉不确定性II", 
            category: "雷达与无线电", 
            year: "2017", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Radio_Interferometric_Imaging_II_超精读笔记_已填充.md", 
            pdfFile: "无线电干涉不确定性II Radio Interferometric II.pdf" 
        },
        { 
            id: "4-04", 
            title: "在线无线电干涉成像", 
            category: "雷达与无线电", 
            year: "2017", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md", 
            pdfFile: "在线无线电干涉成像 Online Radio Imaging.pdf" 
        },
        { 
            id: "4-05", 
            title: "分布式无线电优化", 
            category: "雷达与无线电", 
            year: "2018", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "分布式无线电优化 Distributed Radio Optimization.pdf" 
        },
        { 
            id: "4-06", 
            title: "PURIFY分布式优化", 
            category: "雷达与无线电", 
            year: "2019", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "PURIFY分布式优化 PURIFY.pdf" 
        },
        { 
            id: "4-07", 
            title: "近端嵌套采样", 
            category: "雷达与无线电", 
            year: "2021", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Proximal_Nested_Sampling_超精读笔记_已填充.md", 
            pdfFile: "2023_2307.00056_Proximal_Nested_Sampling.pdf" 
        },
        { 
            id: "4-08", 
            title: "近端嵌套采样(中文)", 
            category: "雷达与无线电", 
            year: "2021", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "近端嵌套采样 Proximal Nested Sampling.pdf" 
        },
        { 
            id: "4-09", 
            title: "多功能传感器树木分类", 
            category: "雷达与无线电", 
            year: "2021", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "多功能传感器树木分类 Tree Species Classification.pdf" 
        },
        { 
            id: "4-10", 
            title: "多传感器树木制图", 
            category: "雷达与无线电", 
            year: "2022", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "多传感器树木制图 Multi-Sensor Trees.pdf" 
        },
        { 
            id: "4-11", 
            title: "RobustPCA树木分类", 
            category: "雷达与无线电", 
            year: "2020", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "RobustPCA树木分类 Robust PCA Trees.pdf" 
        },
        { 
            id: "4-12", 
            title: "ISAR卫星特征识别", 
            category: "雷达与无线电", 
            year: "2026", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "ISAR卫星特征识别 ISAR.pdf" 
        },
        { 
            id: "4-13", 
            title: "雷达工作模式识别", 
            category: "雷达与无线电", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "雷达工作模式识别 Radar Work Mode.pdf" 
        },
        { 
            id: "4-14", 
            title: "DNCNet雷达去噪", 
            category: "雷达与无线电", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "DNCNet雷达去噪 DNCNet.pdf" 
        },
        { 
            id: "4-15", 
            title: "船舶匹配遥感", 
            category: "雷达与无线电", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "船舶匹配遥感 Ship Matching.pdf" 
        },

        // ===== 医学图像 (10篇) =====
        { 
            id: "5-01", 
            title: "扩散模型脑MRI", 
            category: "医学图像", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Discrepancy-based_Diffusion_MRI_超精读笔记_已填充.md", 
            pdfFile: "2024_2405.04974_Diffusion_Brain_MRI.pdf" 
        },
        { 
            id: "5-02", 
            title: "扩散模型脑MRI(中文)", 
            category: "医学图像", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Discrepancy-based_Diffusion_MRI_超精读笔记_已填充.md", 
            pdfFile: "扩散模型脑MRI Diffusion Brain MRI.pdf" 
        },
        { 
            id: "5-03", 
            title: "医学图像小样本学习", 
            category: "医学图像", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Few-shot_Medical_Imaging_Inference_超精读笔记_已填充.md", 
            pdfFile: "医学图像小样本学习 Medical Few-Shot.pdf" 
        },
        { 
            id: "5-04", 
            title: "医学报告生成IIHT", 
            category: "医学图像", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "IIHT_Medical_Report_Generation_超精读笔记_已填充.md", 
            pdfFile: "医学报告生成IIHT Medical Report IIHT.pdf" 
        },
        { 
            id: "5-05", 
            title: "医学图像分类", 
            category: "医学图像", 
            year: "2022", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "医学图像分类 Medical Classification.pdf" 
        },
        { 
            id: "5-06", 
            title: "直肠分割放疗", 
            category: "医学图像", 
            year: "2021", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "直肠分割放疗 Deep Rectum.pdf" 
        },
        { 
            id: "5-07", 
            title: "HiFi-Mamba MRI重建", 
            category: "医学图像", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "HiFi-Mamba_MRI_Reconstruction_超精读笔记_已填充.md", 
            pdfFile: "HiFi-Mamba MRI重建 HiFi-Mamba MRI Reconstruction.pdf" 
        },
        { 
            id: "5-08", 
            title: "HiFi-MambaV2分层MRI", 
            category: "医学图像", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "HiFi-MambaV2_Hierarchical_MRI_超精读笔记_已填充.md", 
            pdfFile: "HiFi-MambaV2分层MRI HiFi-MambaV2 Hierarchical MRI.pdf" 
        },
        { 
            id: "5-09", 
            title: "非负子空间小样本学习", 
            category: "医学图像", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Non-negative_Subspace_Few-Shot_Learning_超精读笔记_已填充.md", 
            pdfFile: "非负子空间小样本学习 Non-negative Subspace.pdf" 
        },
        { 
            id: "5-10", 
            title: "情感感知人格检测", 
            category: "医学图像", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "EmoPerso_Emotion-Aware_Personality_Detection_超精读笔记_已填充.md", 
            pdfFile: "情感感知人格检测 EmoPerso Emotion-Aware.pdf" 
        },

        // ===== 其他 (19篇) =====
        { 
            id: "6-01", 
            title: "点云神经表示Neural Varifolds", 
            category: "其他", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Neural_varifolds_quantifying_point_cloud_geometry_超精读笔记_已填充.md", 
            pdfFile: "点云神经表示 Neural Varifolds.pdf" 
        },
        { 
            id: "6-02", 
            title: "跨域LiDAR检测", 
            category: "其他", 
            year: "2021", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Cross-Domain_LiDAR_Detection_超精读笔记_已填充.md", 
            pdfFile: "跨域LiDAR检测 Cross-Domain LiDAR.pdf" 
        },
        { 
            id: "6-03", 
            title: "3D生物雷达生长轨迹", 
            category: "其他", 
            year: "2022", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "3D_Growth_Trajectory_Reconstruction_超精读笔记_已填充.md", 
            pdfFile: "2025_2511.02142_3D_Growth_Trajectory_Reconstruction.pdf" 
        },
        { 
            id: "6-04", 
            title: "双层同行评审问题", 
            category: "其他", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Bilevel_Peer-Reviewing_Problem_超精读笔记_已填充.md", 
            pdfFile: "2023_2307.12248_Bilevel_Peer_Review.pdf" 
        },
        { 
            id: "6-05", 
            title: "张量列车近似", 
            category: "其他", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Tensor_Train_Approximation_超精读笔记_已填充.md", 
            pdfFile: "2023_2308.01480_Tensor_Train_Approximation.pdf" 
        },
        { 
            id: "6-06", 
            title: "平衡受保护属性", 
            category: "其他", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "Equalizing_Protected_Attributes_超精读笔记_已填充.md", 
            pdfFile: "2023_2311.14733_Equalizing_Protected_Attributes.pdf" 
        },
        { 
            id: "6-07", 
            title: "脑启发的个性检测", 
            category: "其他", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "HIPPD_Brain-Inspired_Personality_Detection_超精读笔记_已填充.md", 
            pdfFile: "2025_2510.09893_HIPPD_Brain-Inspired_Personality_Detection.pdf" 
        },
        { 
            id: "6-08", 
            title: "GRASPTrack多目标跟踪", 
            category: "其他", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "GRASPTrack_Multi-Object_Tracking_超精读笔记_已填充.md", 
            pdfFile: "2025_2508.08117_GRASPTrack_MultiObject_Tracking.pdf" 
        },
        { 
            id: "6-09", 
            title: "生物孔隙分割", 
            category: "其他", 
            year: "2019", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "生物孔隙变分分割_Bio-Pores_Segmentation_超精读笔记_已填充.md", 
            pdfFile: "生物孔隙分割 Bio-Pores.pdf" 
        },
        { 
            id: "6-10", 
            title: "叶绿体电子断层扫描", 
            category: "其他", 
            year: "2019", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "2019_Thylakoid_Electron_Tomography.pdf" 
        },
        { 
            id: "6-11", 
            title: "叶绿体电子断层扫描(中文)", 
            category: "其他", 
            year: "2019", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "叶绿体电子断层扫描 Thylakoid.pdf" 
        },
        { 
            id: "6-12", 
            title: "帽贝识别计算机视觉", 
            category: "其他", 
            year: "2023", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "2023_Limpets_Computer_Vision_Frontiers.pdf" 
        },
        { 
            id: "6-13", 
            title: "基因与壳AI分析", 
            category: "其他", 
            year: "2025", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "2025_Genes_Shells_AI_Scientific_Reports.pdf" 
        },
        { 
            id: "6-14", 
            title: "生物启发迭代学习控制", 
            category: "其他", 
            year: "2018", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "Biologically-Inspired_Iterative_Learning_Control_Design_A_Modular-Based_Approach.pdf" 
        },
        { 
            id: "6-15", 
            title: "生物启发迭代学习控制(中文)", 
            category: "其他", 
            year: "2018", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "生物启发迭代学习控制 ILC.pdf" 
        },
        { 
            id: "6-16", 
            title: "稀疏贝叶斯质量映射I", 
            category: "其他", 
            year: "2015", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "稀疏贝叶斯质量映射I Sparse Bayesian I.pdf" 
        },
        { 
            id: "6-17", 
            title: "稀疏贝叶斯质量映射II", 
            category: "其他", 
            year: "2016", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "稀疏贝叶斯质量映射II Sparse Bayesian II.pdf" 
        },
        { 
            id: "6-18", 
            title: "稀疏贝叶斯质量映射III", 
            category: "其他", 
            year: "2017", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "稀疏贝叶斯质量映射III Sparse Bayesian III.pdf" 
        },
        { 
            id: "6-19", 
            title: "自然历史馆藏错误标本", 
            category: "其他", 
            year: "2024", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "自然历史馆藏错误标本能计算机视觉方法 Mislabelled Specimens.pdf" 
        },

        // ===== 平衡神经网络搜索 (2篇) =====
        { 
            id: "7-01", 
            title: "平衡神经网络搜索I", 
            category: "其他", 
            year: "2020", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "平衡神经网络搜索I_Balanced_NAS_I_超精读笔记.md", 
            pdfFile: "平衡神经网络搜索I Balanced NAS I.pdf" 
        },
        { 
            id: "7-02", 
            title: "平衡神经网络搜索II", 
            category: "其他", 
            year: "2021", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "平衡神经网络搜索II Balanced NAS II.pdf" 
        },

        // ===== 多类ROF分割 =====
        { 
            id: "8-01", 
            title: "多类ROF分割", 
            category: "变分分割", 
            year: "2014", 
            status: "filled", 
            hasPDF: true, 
            hasNote: true,
            noteFile: "多类分割迭代ROF_Iterated_ROF_超精读笔记_已填充.md", 
            pdfFile: "多类ROF分割 Iterated ROF.pdf" 
        },

        // ===== 非参数图像配准 =====
        { 
            id: "9-01", 
            title: "非参数图像配准", 
            category: "变分分割", 
            year: "2014", 
            status: "filled", 
            hasPDF: true, 
            hasNote: false,
            noteFile: "", 
            pdfFile: "非参数图像配准 Nonparametric Registration.pdf" 
        }
    ],
    citations: [
        { source: "2-01", target: "2-02", strength: 2 },
        { source: "1-04", target: "2-01", strength: 2 },
        { source: "3-01", target: "3-02", strength: 1 },
        { source: "3-04", target: "3-06", strength: 2 },
        { source: "4-02", target: "4-04", strength: 3 },
        { source: "4-04", target: "4-05", strength: 2 },
        { source: "5-01", target: "5-03", strength: 1 },
        { source: "5-07", target: "5-08", strength: 3 },
        { source: "2-06", target: "2-11", strength: 2 },
        { source: "3-03", target: "3-06", strength: 1 },
        { source: "6-01", target: "6-02", strength: 2 }
    ]
};

// 向后兼容：导出到全局作用域
if (typeof window !== 'undefined') {
    window.PAPERS_DATA = PAPERS_DATA;
}
