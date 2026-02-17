// 论文数据 - 匹配中文命名的PDF文件
const PAPERS_DATA = {
    summary: {
        total: 49,
        filled: 49,
        templates: 0,
        hasPDF: 49,
        noPDF: 0
    },
    categories: {
        "基础理论": { count: 5, filled: 5, hasPDF: 5, color: "#ef4444" },
        "变分分割": { count: 11, filled: 11, hasPDF: 11, color: "#3b82f6" },
        "深度学习": { count: 8, filled: 8, hasPDF: 8, color: "#10b981" },
        "雷达与无线电": { count: 8, filled: 8, hasPDF: 8, color: "#8b5cf6" },
        "医学图像": { count: 6, filled: 6, hasPDF: 6, color: "#f59e0b" },
        "其他": { count: 11, filled: 11, hasPDF: 11, color: "#6b7280" }
    },
    papers: [
        // 基础理论 (5篇)
        { id: "1-01", title: "CNNs、RNNs与Transformer在人体动作识别中的综合综述", category: "基础理论", year: "2024", status: "filled", hasPDF: true, noteFile: "论文精读_[1-01]_深度学习架构综述_详细版.md", pdfFile: "深度学习架构综述 CNNs RNNs Transformers.pdf" },
        { id: "1-04", title: "变分分割基础Mumford-Shah与ROF", category: "基础理论", year: "2018", status: "filled", hasPDF: true, noteFile: "论文精读_[1-04]_变分分割基础Mumford-Shah与ROF_详细版.md", pdfFile: "变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf" },
        { id: "1-05", title: "两阶段分类方法", category: "基础理论", year: "2013", status: "filled", hasPDF: true, noteFile: "论文精读_[1-05]_两阶段分类方法_详细版.md", pdfFile: "两阶段分类方法 Two-Stage Classification.pdf" },
        { id: "1-06", title: "概念级可解释AI指标", category: "基础理论", year: "2025", status: "filled", hasPDF: true, noteFile: "论文精读_[1-06]_概念级可解释AI指标_详细版.md", pdfFile: "概念级可解释AI指标 Concept-Based XAI.pdf" },
        { id: "1-07", title: "TransNet动作识别", category: "基础理论", year: "2023", status: "filled", hasPDF: true, noteFile: "论文精读_[1-07]_TransNet动作识别_详细版.md", pdfFile: "TransNet动作识别 TransNet HAR.pdf" },
        
        // 变分分割 (11篇)
        { id: "2-03", title: "SLaT三阶段分割", category: "变分分割", year: "2015", status: "filled", hasPDF: true, noteFile: "论文精读_[2-03]_SLaT三阶段分割_详细版.md", pdfFile: "SLaT三阶段分割 SLaT Segmentation.pdf" },
        { id: "2-04", title: "分割与恢复联合模型", category: "变分分割", year: "2016", status: "filled", hasPDF: true, noteFile: "论文精读_[2-04]_分割与恢复联合模型_详细版.md", pdfFile: "分割与恢复联合模型 Segmentation Restoration.pdf" },
        { id: "2-05", title: "语义比例分割", category: "变分分割", year: "2017", status: "filled", hasPDF: true, noteFile: "论文精读_[2-05]_语义比例分割_详细版.md", pdfFile: "语义比例分割 Semantic Proportions.pdf" },
        { id: "2-08", title: "小波框架血管分割", category: "变分分割", year: "2011", status: "filled", hasPDF: true, noteFile: "论文精读_[2-08]_小波框架血管分割_详细版.md", pdfFile: "小波框架血管分割 Tight-Frame Vessel.pdf" },
        { id: "2-09", title: "框架分割管状结构", category: "变分分割", year: "2013", status: "filled", hasPDF: true, noteFile: "论文精读_[2-09]_框架分割管状结构_详细版.md", pdfFile: "框架分割管状结构 Framelet Tubular.pdf" },
        { id: "2-11", title: "CornerPoint3D 3D检测新尺度", category: "变分分割", year: "2020", status: "filled", hasPDF: true, noteFile: "论文精读_[2-11]_CornerPoint3D_3D检测新尺度_详细版.md", pdfFile: "CornerPoint3D 3D检测新尺度 CornerPoint3D.pdf" },
        { id: "2-18", title: "3D方向场变换", category: "变分分割", year: "2020", status: "filled", hasPDF: true, noteFile: "论文精读_[2-18]_3D方向场变换_详细版.md", pdfFile: "3D方向场变换 3D Orientation Field.pdf" },
        { id: "2-28", title: "医学报告生成IIHT", category: "变分分割", year: "2023", status: "filled", hasPDF: true, noteFile: "论文精读_[2-28]_医学报告生成IIHT_详细版.md", pdfFile: "医学报告生成IIHT Medical Report IIHT.pdf" },
        { id: "4-11", title: "非参数图像配准", category: "变分分割", year: "2014", status: "filled", hasPDF: true, noteFile: "论文精读_[4-11]_非参数图像配准_详细版.md", pdfFile: "非参数图像配准 Nonparametric Registration.pdf" },
        { id: "4-12", title: "球面小波分割", category: "变分分割", year: "2016", status: "filled", hasPDF: true, noteFile: "论文精读_[4-12]_球面小波分割_详细版.md", pdfFile: "球面小波分割 Wavelet Sphere.pdf" },
        { id: "2-30", title: "高效变分分类方法", category: "变分分割", year: "2021", status: "filled", hasPDF: false, noteFile: "论文精读_[2-30]_高效变分分类方法_详细版.md", pdfFile: "" },
        
        // 深度学习 (8篇)
        { id: "3-01", title: "大模型高效微调CALM", category: "深度学习", year: "2026", status: "filled", hasPDF: true, noteFile: "论文精读_[3-01]_大模型高效微调CALM_详细版.md", pdfFile: "大模型高效微调CALM CALM Fine-tuning.pdf" },
        { id: "3-02", title: "张量CUR分解LoRA", category: "深度学习", year: "2025", status: "filled", hasPDF: true, noteFile: "论文精读_[3-02]_张量CUR分解LoRA_详细版.md", pdfFile: "张量CUR分解LoRA tCURLoRA.pdf" },
        { id: "3-03", title: "蛋白质结构网络图LL4G", category: "深度学习", year: "2025", status: "filled", hasPDF: true, noteFile: "论文精读_[3-03]_蛋白质结构网络图LL4G_详细版.md", pdfFile: "蛋白质结构网络图LL4G LL4G Graph.pdf" },
        { id: "3-04", title: "低秩Tucker逼近", category: "深度学习", year: "2023", status: "filled", hasPDF: true, noteFile: "论文精读_[3-04]_低秩Tucker逼近_详细版.md", pdfFile: "低秩Tucker近似 sketching Tucker Approximation.pdf" },
        { id: "3-05", title: "大规模模型分解", category: "深度学习", year: "2024", status: "filled", hasPDF: true, noteFile: "论文精读_[3-05]_大规模模型分解_详细版.md", pdfFile: "大规模张量分解 Two-Sided Sketching.pdf" },
        { id: "3-07", title: "GAMED虚假新闻检测", category: "深度学习", year: "2024", status: "filled", hasPDF: true, noteFile: "论文精读_[3-07]_GAMED虚假新闻检测_详细版.md", pdfFile: "GAMED虚假新闻检测 GAMED Fake News.pdf" },
        { id: "3-08", title: "MOGO 3D人体运动生成", category: "深度学习", year: "2024", status: "filled", hasPDF: true, noteFile: "论文精读_[3-08]_MOGO_3D人体运动生成_详细版.md", pdfFile: "MOGO 3D人体运动生成 MOGO Motion.pdf" },
        { id: "PEFT", title: "高效PEFT微调", category: "深度学习", year: "2025", status: "filled", hasPDF: true, noteFile: "", pdfFile: "高效PEFT微调 Less but Better PEFT.pdf" },
        
        // 雷达与无线电 (8篇)
        { id: "3-06", title: "Talk2Radar语言-雷达多模态", category: "雷达与无线电", year: "2025", status: "filled", hasPDF: true, noteFile: "论文精读_[3-06]_Talk2Radar_详细版.md", pdfFile: "Talk2Radar 雷达语言多模态 Talk2Radar.pdf" },
        { id: "4-04", title: "无线电干涉不确定性I", category: "雷达与无线电", year: "2017", status: "filled", hasPDF: true, noteFile: "论文精读_[4-04]_无线电干涉不确定性I_详细版.md", pdfFile: "无线电干涉不确定性I Radio Interferometric I.pdf" },
        { id: "4-05", title: "在线无线电干涉成像", category: "雷达与无线电", year: "2017", status: "filled", hasPDF: true, noteFile: "论文精读_[4-05]_在线无线电干涉成像_详细版.md", pdfFile: "在线无线电干涉成像 Online Radio Imaging.pdf" },
        { id: "4-06", title: "分布式无线电优化", category: "雷达与无线电", year: "2018", status: "filled", hasPDF: true, noteFile: "论文精读_[4-06]_分布式无线电优化_详细版.md", pdfFile: "分布式无线电优化 Distributed Radio Optimization.pdf" },
        { id: "4-08", title: "近端嵌套采样", category: "雷达与无线电", year: "2021", status: "filled", hasPDF: true, noteFile: "论文精读_[4-08]_近端嵌套采样_详细版.md", pdfFile: "近端嵌套采样 Proximal Nested Sampling.pdf" },
        { id: "4-10", title: "多功能传感器树木分类", category: "雷达与无线电", year: "2021", status: "filled", hasPDF: true, noteFile: "论文精读_[4-10]_多功能传感器树木分类_详细版.md", pdfFile: "多功能传感器树木分类 Tree Species Classification.pdf" },
        { id: "4-03", title: "ISAR卫星特征识别", category: "雷达与无线电", year: "2026", status: "filled", hasPDF: false, noteFile: "论文精读_[4-03]_ISAR卫星特征识别_详细版.md", pdfFile: "" },
        { id: "4-09", title: "数据驱动先验验后", category: "雷达与无线电", year: "2024", status: "filled", hasPDF: false, noteFile: "论文精读_[4-09]_数据驱动先验验后_详细版.md", pdfFile: "" },
        
        // 医学图像 (6篇)
        { id: "2-21", title: "扩散模型脑MRI", category: "医学图像", year: "2024", status: "filled", hasPDF: true, noteFile: "论文精读_[2-21]_扩散模型脑MRI_详细版.md", pdfFile: "扩散模型脑MRI Diffusion Brain MRI.pdf" },
        { id: "2-25", title: "医学图像小样本学习", category: "医学图像", year: "2024", status: "filled", hasPDF: true, noteFile: "论文精读_[2-25]_医学图像小样本学习_详细版.md", pdfFile: "医学图像小样本学习 Medical Few-Shot.pdf" },
        { id: "HiFi-Mamba", title: "HiFi-Mamba MRI重建", category: "医学图像", year: "2025", status: "filled", hasPDF: true, noteFile: "", pdfFile: "HiFi-Mamba MRI重建 HiFi-Mamba MRI Reconstruction.pdf" },
        { id: "HiFi-MambaV2", title: "HiFi-MambaV2分层MRI", category: "医学图像", year: "2025", status: "filled", hasPDF: true, noteFile: "", pdfFile: "HiFi-MambaV2分层MRI HiFi-MambaV2 Hierarchical MRI.pdf" },
        { id: "go-lda", title: "非负子空间小样本学习GO-LDA", category: "医学图像", year: "2024", status: "filled", hasPDF: true, noteFile: "论文精读_非负子空间小样本学习GO-LDA_详细版.md", pdfFile: "非负子空间小样本学习 Non-negative Subspace.pdf" },
        { id: "2-28", title: "医学报告生成IIHT", category: "医学图像", year: "2023", status: "filled", hasPDF: true, noteFile: "论文精读_[2-28]_医学报告生成IIHT_详细版.md", pdfFile: "医学报告生成IIHT Medical Report IIHT.pdf" },
        
        // 其他 (11篇)
        { id: "2-12", title: "点云神经表示Neural Varifolds", category: "其他", year: "2025", status: "filled", hasPDF: true, noteFile: "论文精读_[2-12]_点云神经表示Neural_Varifolds_详细版.md", pdfFile: "点云神经表示 Neural Varifolds.pdf" },
        { id: "2-13", title: "跨域3D目标检测", category: "其他", year: "2021", status: "filled", hasPDF: true, noteFile: "论文精读_[2-13]_跨域3D目标检测_详细版.md", pdfFile: "跨域LiDAR检测 Cross-Domain LiDAR.pdf" },
        { id: "2-14", title: "3D生物雷达生长轨迹", category: "其他", year: "2022", status: "filled", hasPDF: true, noteFile: "论文精读_[2-14]_3D生物雷达生长轨迹_详细版.md", pdfFile: "3D生物雷达生长轨迹 3D Growth Trajectory.pdf" },
        { id: "2-15", title: "3D树木分割图", category: "其他", year: "2019", status: "filled", hasPDF: true, noteFile: "论文精读_[2-15]_3D树木分割图_详细版.md", pdfFile: "3D树木分割图 3D Tree Segmentation.pdf" },
        { id: "2-16", title: "3D树木描绘图割", category: "其他", year: "2020", status: "filled", hasPDF: true, noteFile: "论文精读_[2-16]_3D树木描绘图割_详细版.md", pdfFile: "3D树木描绘图割 3D Tree Delineation.pdf" },
        { id: "2-17", title: "球状正交Krawtchouk矩", category: "其他", year: "2024", status: "filled", hasPDF: false, noteFile: "论文精读_[2-17]_球状正交Krawtchouk矩_详细版.md", pdfFile: "" },
        { id: "MotionDuet", title: "MotionDuet运动生成", category: "其他", year: "2025", status: "filled", hasPDF: true, noteFile: "", pdfFile: "MotionDuet运动生成 MotionDuet 3D Motion.pdf" },
        { id: "GRASPTrack", title: "多目标跟踪GRASPTrack", category: "其他", year: "2025", status: "filled", hasPDF: true, noteFile: "", pdfFile: "多目标跟踪 GRASPTrack Multi-Object Tracking.pdf" },
        { id: "Bilevel", title: "双层同行评审问题", category: "其他", year: "2023", status: "filled", hasPDF: true, noteFile: "", pdfFile: "双层同行评审问题 Bilevel Peer-Reviewing.pdf" },
        { id: "TensorTrain", title: "张量列车近似", category: "其他", year: "2023", status: "filled", hasPDF: true, noteFile: "", pdfFile: "张量列车近似 Tensor Train Approximation.pdf" },
        { id: "Equalizing", title: "平衡受保护属性", category: "其他", year: "2023", status: "filled", hasPDF: true, noteFile: "", pdfFile: "平衡受保护属性 Equalizing Protected Attributes.pdf" },
        { id: "EmoPerso", title: "情感感知人格检测", category: "其他", year: "2025", status: "filled", hasPDF: true, noteFile: "", pdfFile: "情感感知人格检测 EmoPerso Emotion-Aware.pdf" },
        { id: "HIPPD", title: "脑启发的个性检测", category: "其他", year: "2025", status: "filled", hasPDF: true, noteFile: "", pdfFile: "脑启发的个性检测 HIPPD Brain-Inspired.pdf" },
        { id: "DetectCloser", title: "可见表面检测", category: "其他", year: "2024", status: "filled", hasPDF: true, noteFile: "论文精读_[2-06]_可见表面检测_详细版.md", pdfFile: "可见表面检测 Detect Closer Surfaces.pdf" },
        { id: "HighDim", title: "高维不确定性性能", category: "其他", year: "2018", status: "filled", hasPDF: true, noteFile: "", pdfFile: "高维不确定性性能 High-Dimensional Uncertainty.pdf" },
        { id: "Radio2", title: "无线电干涉不确定性II", category: "其他", year: "2017", status: "filled", hasPDF: true, noteFile: "", pdfFile: "无线电干涉不确定性II Radio Interferometric II.pdf" }
    ],
    citations: [
        { source: "2-03", target: "2-04", strength: 2 },
        { source: "1-04", target: "2-03", strength: 2 },
        { source: "3-01", target: "3-02", strength: 1 },
        { source: "3-04", target: "3-05", strength: 2 },
        { source: "4-04", target: "4-05", strength: 3 },
        { source: "4-05", target: "4-06", strength: 2 },
        { source: "2-21", target: "2-25", strength: 1 },
        { source: "HiFi-Mamba", target: "HiFi-MambaV2", strength: 3 }
    ]
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
            "其他": "#6b7280"
        };
        return colors[category] || "#6b7280";
    },
    
    getCategoryClass: (category) => {
        const classes = {
            "基础理论": "category-theory",
            "变分分割": "category-segmentation",
            "深度学习": "category-deep",
            "雷达与无线电": "category-signal",
            "医学图像": "category-medical",
            "其他": "category-other"
        };
        return classes[category] || "category-other";
    },
    
    getNotePath: (filename) => {
        if (!filename) return null;
        return `../${filename}`;
    },
    
    getPDFPath: (filename) => {
        if (!filename) return null;
        // 使用绝对路径，确保中文文件名正确编码
        return '/00_papers/' + encodeURIComponent(filename);
    }
};
