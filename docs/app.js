const basePath = "00_papers_first_author_xiaohao_cai_deduped/";

const tracks = {
  sat: {
    label: "SaT / 变分分割",
    short: "SaT",
    count: 3,
    color: "#28666e",
    summary: "以 smoothing/restoration 先稳定图像，再通过 thresholding、lifting 或聚类得到分割。",
    emphasis: "这条线说明他的分割方法不是单纯应用，而是用凸模型替代直接求解困难的非凸分割问题。"
  },
  rof: {
    label: "ROF / PCMS 理论",
    short: "ROF",
    count: 2,
    color: "#795c34",
    summary: "建立 ROF 恢复模型与 PCMS、Chan-Vese 分割模型之间的桥梁。",
    emphasis: "核心贡献是把“恢复 + 阈值化 = 分割”从经验套路推进到有理论支撑的范式。"
  },
  framelet: {
    label: "Framelet / 特殊几何",
    short: "FR",
    count: 3,
    color: "#3f6f4f",
    summary: "用 tight-frame、framelet、wavelet 处理血管、管状结构和球面图像等特殊几何数据。",
    emphasis: "这条线展示 SaT/阈值化思想如何迁移到细长结构、弱边缘和非欧氏图像。"
  },
  classification: {
    label: "高维图分类",
    short: "CL",
    count: 2,
    color: "#5e548e",
    summary: "把图像分割抽象成 graph-based classification，用图 Laplacian 和 graph TV 做半监督分类。",
    emphasis: "后期重点是高维数据、点云和无结构数据上的变分分类模型。"
  },
  bayes: {
    label: "Bayesian 逆问题 / UQ",
    short: "UQ",
    count: 5,
    color: "#9a5a36",
    summary: "从无线电干涉成像进入高维贝叶斯逆问题，关注重建、采样、不确定性量化和模型选择。",
    emphasis: "主链是 proximal MCMC、MAP-UQ、online RI 和 proximal nested sampling。"
  }
};

const thesis = {
  headline: "从图像分割到贝叶斯逆问题的同一套数学语言",
  body: "这 15 篇共同呈现的主方向是：变分模型 + 凸优化 + 稀疏/小波/框架表示 + 贝叶斯逆问题，用来做图像分割、恢复、分类、成像和不确定性量化。",
  oneLine: "核心套路是先把复杂非凸问题转化为更稳定的凸问题，再通过阈值化、投影、采样或近端算法得到最终结果。"
};

const papers = [
  {
    priority: 1,
    file: "分割方法论总览 SaT Overview.pdf",
    title: "An Overview of SaT Segmentation Methodology and Its Applications",
    time: "2023",
    year: 2023,
    pages: 27,
    track: "sat",
    type: "Springer Handbook",
    position: "全局地图：先解释 SaT、T-ROF、SLaT、血管、球面和 hyperspectral 等分支。",
    note: "建议先读 Introduction、SaT Methodology、T-ROF、SLaT、vascular、sphere 几节。"
  },
  {
    priority: 2,
    file: "变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf",
    title: "Linkage Between PCMS and ROF Model and Its Virtue in Image Segmentation",
    time: "2019",
    year: 2019,
    pages: 31,
    track: "rof",
    type: "arXiv / theory",
    position: "理论核心：解释 ROF minimizer 阈值化为什么能得到 PCMS/Chan-Vese 的部分极小解。",
    note: "重点搞清 Mumford-Shah、PCMS、Chan-Vese、ROF 四者关系。"
  },
  {
    priority: 3,
    file: "多类ROF分割 Iterated ROF.pdf",
    title: "Multiclass Segmentation by Iterated ROF Thresholding",
    time: "2013",
    year: 2013,
    pages: 14,
    track: "rof",
    type: "LNCS / EMMCVPR",
    position: "T-ROF 的早期算法雏形：ROF 去噪解反复阈值化，用于多类分割。",
    note: "和 Linkage 连着读，可以看到从算法动机到理论支撑的演化。"
  },
  {
    priority: 4,
    file: "分割恢复联合模型 Segmentation Restoration.pdf",
    title: "Variational Image Segmentation Model Coupled with Image Restoration Achievements",
    time: "2014",
    year: 2014,
    pages: 23,
    track: "sat",
    type: "arXiv",
    position: "把 restoration fidelity term 引入 PCMS，使模型能处理噪声、模糊、缺失像素和向量值图像。",
    note: "先看 Abstract、Introduction 和模型式子，不必一开始深究全部收敛证明。"
  },
  {
    priority: 5,
    file: "框架管状结构分割 Framelet.pdf",
    title: "Framelet-Based Algorithm for Segmentation of Tubular Structures",
    time: "2011 / 2012",
    year: 2012,
    pages: 12,
    track: "framelet",
    type: "SSVM / LNCS",
    position: "早期管状结构分割：用 framelet/tight-frame 处理血管、道路等细长结构。",
    note: "适合先读短版，快速理解管状结构分割算法怎么设计。"
  },
  {
    priority: 6,
    file: "框架分割管状结构 Framelet Tubular.pdf",
    title: "Vessel Segmentation in Medical Imaging Using a Tight-Frame Based Algorithm",
    time: "2011 preprint / about 2013",
    year: 2011,
    pages: 13,
    track: "framelet",
    type: "arXiv / extended version",
    position: "Framelet 管状结构分割扩展版：补足 MRA 血管、tight-frame 迭代、收敛性和 2D/3D 实验。",
    note: "建议放在短版后读，重点看算法步骤、收敛性和医学图像实验。"
  },
  {
    priority: 7,
    file: "SLaT三阶段分割 SLaT Segmentation.pdf",
    title: "SLaT: Smoothing, Lifting and Thresholding",
    time: "2015 preprint / 2017 line",
    year: 2015,
    pages: 19,
    track: "sat",
    type: "arXiv",
    position: "从灰度图扩展到彩色退化图像：RGB 平滑后 lifting 到 RGB + Lab，再聚类阈值化。",
    note: "关键不是 K-means，而是 Lifting：Lab 提供感知颜色信息，补充 RGB 通道相关性。"
  },
  {
    priority: 8,
    file: "球面小波分割 Wavelet Sphere.pdf",
    title: "Wavelet-Based Segmentation on the Sphere",
    time: "2016 preprint / 2019 v2 / about 2020",
    year: 2016,
    pages: 22,
    track: "framelet",
    type: "arXiv",
    position: "把 tight-frame/wavelet 分割推广到球面图像，如地球、太阳、全天图和球面视网膜图像。",
    note: "看几何扩展：球面 wavelet/curvelet、球面梯度和球面采样。"
  },
  {
    priority: 9,
    file: "两阶段分类 Two-Stage.pdf",
    title: "A Two-Stage Classification Method for High-Dimensional Data and Point Clouds",
    time: "2019",
    year: 2019,
    pages: 21,
    track: "classification",
    type: "arXiv",
    position: "把 SaT 从 image segmentation 推到 graph-based classification。",
    note: "先用 SVM 等 warm initialization，再图上凸变分平滑，最后投影为分类结果。"
  },
  {
    priority: 10,
    file: "高效变分分类 Efficient Variational.pdf",
    title: "An Efficient and Versatile Variational Method for High-Dimensional Data Classification",
    time: "2024",
    year: 2024,
    pages: 25,
    track: "classification",
    type: "Journal of Scientific Computing",
    position: "2019 两阶段分类的成熟期/期刊版，面向多类半监督分类、高维数据和点云。",
    note: "重点看 graph Laplacian、graph TV、唯一解、primal-dual 算法和实验对比。"
  },
  {
    priority: 11,
    file: "高维逆问题不确定性量化 Uncertainty Quantification.pdf",
    title: "Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation",
    time: "2019",
    year: 2019,
    pages: 5,
    track: "bayes",
    type: "EUSIPCO",
    position: "UQ 总入口：把 RI 里的 MAP-UQ 思路推广到一般高维逆问题。",
    note: "先用它理解 MAP、HPD credible region、local credible interval。"
  },
  {
    priority: 12,
    file: "无线电干涉不确定性I Radio Interferometric I.pdf",
    title: "Uncertainty Quantification for Radio Interferometric Imaging I: Proximal MCMC Methods",
    time: "2018",
    year: 2018,
    pages: 16,
    track: "bayes",
    type: "MNRAS companion I",
    position: "用 proximal MCMC 支持非光滑稀疏先验，并从完整后验样本做 UQ。",
    note: "重点看 credible intervals、HPD regions、hypothesis testing。"
  },
  {
    priority: 13,
    file: "无线电干涉不确定性II Radio Interferometric II.pdf",
    title: "Uncertainty Quantification for Radio Interferometric Imaging II: MAP Estimation",
    time: "2018",
    year: 2018,
    pages: 13,
    track: "bayes",
    type: "MNRAS companion II",
    position: "RI I 的快速版：用 MAP estimation 加概率集中理论近似 UQ，面向 SKA 级大数据。",
    note: "和 RI I 对照读，理解完整采样和快速近似之间的取舍。"
  },
  {
    priority: 14,
    file: "在线无线电干涉成像 Online Radio Imaging.pdf",
    title: "Online Radio Interferometric Imaging",
    time: "2019",
    year: 2019,
    pages: 14,
    track: "bayes",
    type: "MNRAS style preprint",
    position: "无线电干涉成像的大数据流式处理：visibilities 到达就 assimilate，然后丢弃。",
    note: "解决大规模观测里数据不能全部存下来、也不能等全部到齐再处理的问题。"
  },
  {
    priority: 15,
    file: "近端嵌套采样 Proximal Nested Sampling.pdf",
    title: "Proximal Nested Sampling for High-Dimensional Bayesian Model Selection",
    time: "2022",
    year: 2022,
    pages: 42,
    track: "bayes",
    type: "arXiv / Bayesian computation",
    position: "从“估计图像 + 不确定性”升级到“哪个 Bayesian model 更合适”。",
    note: "最后读，因为它需要 Bayesian evidence、nested sampling 和 proximal MCMC 背景。"
  }
];

const chronology = [
  { time: "2011 / 2012", priority: 5, track: "framelet", label: "Framelet-Based Algorithm for Segmentation of Tubular Structures" },
  { time: "2011 preprint / about 2013", priority: 6, track: "framelet", label: "Vessel Segmentation in Medical Imaging Using a Tight-Frame Based Algorithm" },
  { time: "2013", priority: 3, track: "rof", label: "Multiclass Segmentation by Iterated ROF Thresholding" },
  { time: "2014", priority: 4, track: "sat", label: "Variational Image Segmentation Model Coupled with Image Restoration Achievements" },
  { time: "2015 preprint / 2017 line", priority: 7, track: "sat", label: "SLaT: Smoothing, Lifting and Thresholding" },
  { time: "2016 preprint / 2019 v2 / about 2020", priority: 8, track: "framelet", label: "Wavelet-Based Segmentation on the Sphere" },
  { time: "2018", priority: 12, track: "bayes", label: "RI UQ I: Proximal MCMC Methods" },
  { time: "2018", priority: 13, track: "bayes", label: "RI UQ II: MAP Estimation" },
  { time: "2019", priority: 14, track: "bayes", label: "Online Radio Interferometric Imaging" },
  { time: "2019", priority: 2, track: "rof", label: "Linkage Between PCMS and ROF Model" },
  { time: "2019", priority: 11, track: "bayes", label: "High-Dimensional Inverse Problems UQ" },
  { time: "2019", priority: 9, track: "classification", label: "Two-Stage Classification for High-Dimensional Data" },
  { time: "2022", priority: 15, track: "bayes", label: "Proximal Nested Sampling" },
  { time: "2023", priority: 1, track: "sat", label: "SaT Segmentation Methodology Overview" },
  { time: "2024", priority: 10, track: "classification", label: "Efficient Variational High-Dimensional Classification" }
];

const readingStages = [
  {
    stage: "第一阶段",
    title: "先建立全局地图",
    focus: "先读 SaT 综述，建立 smoothing + thresholding 的整体地图，不要一开始啃证明。",
    priorities: [1]
  },
  {
    stage: "第二阶段",
    title: "补核心理论",
    focus: "连着读 ROF-PCMS 理论和 Iterated ROF，理解恢复模型如何支撑分割模型。",
    priorities: [2, 3]
  },
  {
    stage: "第三阶段",
    title: "看恢复与分割如何结合",
    focus: "理解 restoration fidelity term 为什么能让噪声、模糊、缺失像素和向量值图像更稳。",
    priorities: [4]
  },
  {
    stage: "第四阶段",
    title: "读医学血管与 framelet/tight-frame 应用",
    focus: "先短版、再长版，看管状结构、细血管、弱边缘、分叉和 2D/3D MRA 实验。",
    priorities: [5, 6]
  },
  {
    stage: "第五阶段",
    title: "读 SaT 的扩展应用",
    focus: "一篇看彩色图像 lifting，一篇看球面几何，确认 SaT/thresholding 的可迁移性。",
    priorities: [7, 8]
  },
  {
    stage: "第六阶段",
    title: "读高维分类线",
    focus: "把 image segmentation 抽象成 graph-based classification：初始化、图上平滑、投影/argmax。",
    priorities: [9, 10]
  },
  {
    stage: "第七阶段",
    title: "读无线电干涉与不确定性量化线",
    focus: "先短 UQ 总入口，再读 RI I/II、online imaging，最后读 Bayesian model selection。",
    priorities: [11, 12, 13, 14, 15]
  }
];

const noteThemes = [
  { key: "all", label: "全部" },
  { key: "sat-rof", label: "SaT/ROF 理论" },
  { key: "medical", label: "医学与管状结构" },
  { key: "extension", label: "彩色/球面扩展" },
  { key: "classification", label: "高维分类" },
  { key: "ri-uq", label: "无线电干涉与 UQ" },
  { key: "bayes-model", label: "贝叶斯模型选择" }
];

const noteMainlines = [
  {
    title: "SaT / ROF / PCMS 变分分割主线",
    papers: [1, 2, 3, 4],
    summary: "从 piecewise constant Mumford-Shah (PCMS) 的非凸分割难题出发，用 Rudin-Osher-Fatemi (ROF)、全变分 Total Variation 和阈值化建立“恢复先行、分割随后”的方法论。"
  },
  {
    title: "framelet / tight-frame 管状结构与血管分割主线",
    papers: [5, 6],
    summary: "把不确定边界像素限制在候选区间内，只在候选区域做 framelet / tight-frame 平滑，面向 2D/3D MRA 血管、道路和分叉管状结构。"
  },
  {
    title: "SLaT / spherical wavelet 几何与彩色扩展主线",
    papers: [7, 8],
    summary: "一边把 SaT 扩展到 RGB+Lab 彩色特征，一边把 Euclidean 图像上的 tight-frame 思路迁移到 spherical wavelet、directional wavelet 和 curvelet。"
  },
  {
    title: "graph variational / high-dimensional classification 高维分类主线",
    papers: [9, 10],
    summary: "把像素分割抽象成图上标签函数平滑：warm initialization、graph Laplacian、graph TV、无 simplex 约束凸模型和 binary projection。"
  },
  {
    title: "RI imaging / Bayesian UQ / proximal sampling 贝叶斯逆问题主线",
    papers: [11, 12, 13, 14, 15],
    summary: "从 radio interferometric imaging 的大规模逆问题进入 maximum a posteriori (MAP)、highest posterior density (HPD)、proximal MCMC、online imaging 和 Bayesian evidence。"
  }
];

const readingStandard = [
  "第一遍读 Abstract + Introduction，定位问题、数据对象和旧方法瓶颈。",
  "第二遍读模型和算法，抓住变量、目标函数、约束、proximal / thresholding / projection 的迭代步骤。",
  "第三遍读 theorem 或 proof sketch，判断保证的是唯一解、收敛、partial minimizer、复杂度还是 HPD 近似。",
  "第四遍读 experiments，确认实验对象、退化类型、对照方法和指标到底支撑了什么结论。",
  "读完每篇要能输出：一句话贡献、一个关键公式、一个算法流程、一个与其他论文的关系。"
];

const paperNotesV2 = [
  {
    id: "sat-overview",
    priority: 1,
    titleCn: "SaT 分割方法论总览",
    titleEn: "An Overview of SaT Segmentation Methodology and Its Applications",
    year: 2023,
    theme: "sat-rof",
    difficulty: "入门",
    prerequisites: ["Mumford-Shah", "Chan-Vese", "Total Variation", "K-means"],
    oneSentence: "先用综述搭起 SaT 分割地图。",
    coreProblem: "传统 Mumford-Shah、Chan-Vese 和 piecewise constant Mumford-Shah (PCMS) 模型通常非凸，类别数或退化类型一变，直接优化就容易慢、不稳或依赖初始化。",
    whyHard: "分割模型同时要估计区域边界和区域常数，噪声、模糊、缺失信息和彩色通道相关性会让能量地形更复杂；如果每次改变类别数 K 都重解非凸模型，交互式精调几乎不可用。",
    methodHandle: "SaT 把问题拆成 smoothing 和 thresholding。Smoothing step 先解一个凸恢复/平滑目标函数，得到比原图更稳定的中间图像；Thresholding step 再用阈值、K-means 或多通道聚类把中间图像分成 K 相。K 只在最后一步进入，所以改类别数不需要重解 smoothing 模型。",
    keyModelOrFormula: "SaT = solve convex smoothing model for g, then segment g by thresholds or K-means; K appears in thresholding, not in smoothing.",
    algorithmFlow: ["读入退化图像 f。", "选择适合噪声/模糊/缺失的凸 smoothing 模型。", "求平滑图像 g。", "按灰度、多通道特征或聚类结果做 thresholding。", "改变类别数 K 时只重做 thresholding。"],
    theoremOrGuarantee: "综述中整理的 SaT 分支强调 smoothing 子问题常有凸性、唯一解或稳定求解保证；其理论核心要回到 ROF-PCMS linkage 和 T-ROF 收敛性论文中核对。",
    experimentFocus: "综述覆盖 synthetic retina、退化彩色图像、vascular structures、spherical images、hyperspectral images 等应用；读实验时看每个分支解决的退化类型，而不是只看分割图。",
    howToRead: "先读目录、Introduction 和 SaT Methodology；第二轮只读 T-ROF、SLaT、vascular、sphere 小节；最后把每个小节映射到本 dashboard 的对应原始论文。",
    relation: { text: "这是入口论文，负责给出地图，不负责所有细节证明。", links: [2, 3, 7, 8] },
    readingQuestions: ["为什么 SaT 改类别数 K 时不用重解 smoothing？", "SaT 和直接求解 PCMS 的优化难点差别在哪里？", "SLaT、T-ROF、vascular、sphere 四个分支各解决哪一种退化或数据域？"],
    afterReadingOutput: "画出一张 SaT 方法树：smoothing 模型、thresholding 方式、适用数据和对应论文。"
  },
  {
    id: "pcms-rof-linkage",
    priority: 2,
    titleCn: "PCMS 与 ROF 的理论连接",
    titleEn: "Linkage Between PCMS and ROF Model and Its Virtue in Image Segmentation",
    year: 2019,
    theme: "sat-rof",
    difficulty: "困难",
    prerequisites: ["BV space", "Total Variation", "ROF", "Chan-Vese", "partial minimizer"],
    oneSentence: "证明 ROF 阈值化支撑 PCMS 分割。",
    coreProblem: "Rudin-Osher-Fatemi (ROF) 是图像恢复模型，piecewise constant Mumford-Shah (PCMS) / Chan-Vese 是分割模型；论文要回答二者为什么不是经验拼接，而能在变分意义下连接。",
    whyHard: "PCMS 的边界和区域常数耦合导致非凸，直接求解多相问题代价高；ROF 虽是凸恢复，但它的 minimizer 经过 thresholding 后是否仍对应分割能量的局部最优，需要严格说明。",
    methodHandle: "论文先分析 K=2 的 T-ROF 与 PCMS 关系，再把 ROF minimizer 的 level set/threshold set 与 PCMS partial minimizer 联系起来。核心操作是求 ROF minimizer u*，再取 threshold set Σ = {x : u*(x) > (m0 + m1) / 2}，从而把恢复解转成分割区域。",
    keyModelOrFormula: "ROF: TV(u) + μ/2 ∫(u - f)^2 dx; PCMS/Chan-Vese: perimeter(Σ) + λ ∫ data fitting; threshold: Σ = {x : u*(x) > (m0 + m1)/2}.",
    algorithmFlow: ["固定二相常数 m0、m1。", "求解 ROF / T-ROF 相关凸恢复子问题。", "用 (m0 + m1) / 2 构造阈值。", "把 ROF minimizer 的 level set 转为区域 Σ。", "检查 Σ 与 PCMS partial minimizer 的关系。"],
    theoremOrGuarantee: "PDF 中主定理集中在 Section 3：K=2 时，ROF minimizer 的合适阈值集可给出 PCMS/Chan-Vese 的 partial minimizer；多相 K>2 仍有类似结论，但依赖更具体的假设和阈值结构。",
    experimentFocus: "实验部分展示 T-ROF 在 noisy、blurry、information loss 等退化图像上的分割效率和质量，重点不是新数据集，而是验证理论范式带来的计算优势。",
    howToRead: "先读 Abstract 和 Section 1，明确 ROF 与 PCMS 各自角色；再精读 Section 3 的主定理和 partial minimizer；最后读 Section 4 的 T-ROF 算法和收敛分析。",
    relation: { text: "它给 T-ROF 与 SaT 提供理论合法性，并解释为什么恢复模型可以服务分割。", links: [1, 3, 4] },
    readingQuestions: ["partial minimizer 与 global minimizer 有什么差别？", "为什么 K=2 的结论最清楚，多相 K>2 需要附加条件？", "ROF 的 Total Variation 项在分割解释里对应什么几何量？"],
    afterReadingOutput: "写一页说明：ROF minimizer 为什么能通过阈值化得到 PCMS partial minimizer。"
  },
  {
    id: "iterated-rof",
    priority: 3,
    titleCn: "多类 ROF 阈值迭代分割",
    titleEn: "Multiclass Segmentation by Iterated ROF Thresholding",
    year: 2013,
    theme: "sat-rof",
    difficulty: "中等",
    prerequisites: ["ROF", "Chan-Vese", "multiphase segmentation", "threshold update"],
    oneSentence: "T-ROF 的早期算法雏形。",
    coreProblem: "多相 PCMS/Chan-Vese 直接优化困难，且灰度值接近的类别容易被合并；论文要用 ROF 阈值化避免直接求解非凸多相分割。",
    whyHard: "多类分割要同时估计多个区域和多个均值，类别间灰度差很小时，固定阈值或单次聚类容易失败；非凸模型反复优化又会放大初始化和参数敏感性。",
    methodHandle: "T-ROF 先解 Rudin-Osher-Fatemi (ROF) 恢复问题，再对 ROF 解做多阈值分割。算法反复根据当前分割区域均值 m_i 更新阈值 τ_i = 1/2(m_{i-1}+m_i)，用投影后的阈值序列继续迭代，使阈值自动适配相邻类别。",
    keyModelOrFormula: "threshold update: τ_i = 1/2(m_{i-1} + m_i); ROF/T-ROF energy uses Total Variation plus linear or quadratic data term.",
    algorithmFlow: ["用 ROF 得到平滑图像。", "按当前阈值 τ_i 划分多个类别。", "计算每个类别的均值 m_i。", "用相邻均值更新 τ_i。", "投影阈值并重复，直到阈值序列收敛。"],
    theoremOrGuarantee: "论文给出 projected T-ROF algorithm 在特定假设下阈值序列收敛的定理；K=2 时模型与 Chan-Vese 之间有等价/联系，并带有调整后的正则参数解释。",
    experimentFocus: "实验对象包括 cartoon、texture 和 medical images；重点看灰度值相近类别是否被正确分开，以及算法速度相对其他 variational segmentation 方法的差异。",
    howToRead: "先读 Abstract 和 Section 2，理解为什么 ROF 与 Chan-Vese 能接上；再读 Algorithm T-ROF 和阈值更新规则；最后看 texture/medical 实验中的失败与成功样例。",
    relation: { text: "它是 Linkage 论文的算法前身，也是 SaT 方法论中 T-ROF 分支的核心实例。", links: [1, 2, 4] },
    readingQuestions: ["为什么 τ_i 要用相邻区域均值的一半和更新？", "只解一次 ROF 与迭代更新阈值之间如何配合？", "T-ROF 在灰度值相近类别上的优势来自模型还是阈值更新？"],
    afterReadingOutput: "画出 T-ROF 阈值更新流程图，并标出 m_i、τ_i 和分割区域的循环关系。"
  },
  {
    id: "segmentation-restoration",
    priority: 4,
    titleCn: "分割与恢复耦合模型",
    titleEn: "Variational Image Segmentation Model Coupled with Image Restoration Achievements",
    year: 2014,
    theme: "sat-rof",
    difficulty: "中等偏难",
    prerequisites: ["PCMS", "image restoration", "alternating minimization", "Total Variation"],
    oneSentence: "把恢复与分割合成一个模型。",
    coreProblem: "传统 PCMS 难以稳定处理 blur、missing pixels、vector-valued images；如果先恢复再分割，恢复误差可能传递，论文改为把恢复变量直接并入分割能量。",
    whyHard: "观察图像 f 可能是由 clean image g 经过算子 A、噪声或缺失采样得到；分割变量 u_i 和区域均值 c_i 依赖 g，而 g 又需要从 f 反演，三类变量相互耦合。",
    methodHandle: "模型引入恢复变量 g，将 image restoration fidelity term Φ(f,Ag) 与 segmentation term Ψ(g,u,c) 耦合。通过 alternating minimization 依次更新 g、区域常数 c_i 和 label/indicator 函数 u_i，使恢复任务和 PCMS 分割任务在同一能量中互相约束。",
    keyModelOrFormula: "E(u,c,g)= μ Φ(f, A g) + λ Σ_i ∫(g - c_i)^2 u_i dx + Σ_i TV(u_i).",
    algorithmFlow: ["初始化 u_i、c_i 和恢复图像 g。", "固定 u_i、c_i 更新 g。", "固定 g、u_i 更新每类均值 c_i。", "固定 g、c_i 更新 u_i。", "重复 alternating minimization 直到能量或变量稳定。"],
    theoremOrGuarantee: "论文给出固定 c_i、u_i 时 g 子问题唯一解条件，并证明三变量 alternating minimization 在 mild condition 下的收敛性质。",
    experimentFocus: "实验覆盖 synthetic 和 real-world images，尤其关注 high noisy images、blurry images、missing pixels 和 vector-valued images；读实验时看 blur/missing 的对照组。",
    howToRead: "先读 Abstract + Introduction；再读模型中 f、g、A、u_i、c_i 的定义；随后读 Algorithm 1 和 Theorem 1/4；最后看模糊与缺失像素实验。",
    relation: { text: "与 SaT 一样体现 restoration helps segmentation，但这里是 joint optimization，不是两阶段阈值化。", links: [1, 2, 7] },
    readingQuestions: ["f、g、A 分别代表什么？", "为什么加入 g 能处理 blur 和 missing pixels？", "joint optimization 与 SaT 两阶段路线的风险和优势分别是什么？"],
    afterReadingOutput: "写出三变量 alternating minimization 的伪代码，并解释每一步优化的变量。"
  },
  {
    id: "framelet-tubular",
    priority: 5,
    titleCn: "Framelet 管状结构分割短版",
    titleEn: "Framelet-Based Algorithm for Segmentation of Tubular Structures",
    year: 2011,
    theme: "medical",
    difficulty: "中等",
    prerequisites: ["framelet", "tight frame", "soft-thresholding", "MRA"],
    oneSentence: "只平滑管状边界候选区。",
    coreProblem: "MRA 血管、道路和其他 tube-like structures 有细长、弱边缘、分叉、遮挡等特点；全图平滑会抹掉细节，传统 PDE/active contour 又容易被噪声和初始化影响。",
    whyHard: "管状结构的内部、外部和边界灰度不是单点可分；真正难的是灰度落在边界候选区间内的像素，既不能粗暴归类，也不能对整幅图无差别平滑。",
    methodHandle: "算法估计边界灰度区间 [α_i, β_i]，每轮把图像分成 below、inside、above 三部分，只对 inside，即可能边界区域，做 framelet denoising / smoothing 和 soft-thresholding，再收缩候选区间直到得到二值图像。",
    keyModelOrFormula: "candidate boundary Λ_i = {x : α_i < f_i(x) < β_i}; framelet denoising on Λ_i; stop when all pixels map to 0 or 1.",
    algorithmFlow: ["估计当前边界灰度区间 [α_i, β_i]。", "把像素分成背景、边界候选和血管三类。", "只在候选区域 Λ_i 做 framelet soft-thresholding。", "更新图像并收缩候选区域。", "候选区域为空时输出二值管状结构。"],
    theoremOrGuarantee: "论文给出 convergence statement：framelet-based algorithm 会在有限步收敛到二值图像；关键是候选边界区域 Λ_i 持续收缩。",
    experimentFocus: "实验为 real 2D/3D images，并在文本中明确指向 Magnetic Resonance Angiography (MRA) 血管场景；重点看细血管、分叉和弱边界是否保留。",
    howToRead: "先读 Section 2 的 tight frame / framelet 基础；重点读 Section 3 算法步骤；再读 Theorem 1 的 finite convergence 证明；最后看 2D/3D 图像实验。",
    relation: { text: "这是 vessel tight-frame 长版的短版基础，也与 SaT 共享“平滑不确定区域 + 阈值化”的思想。", links: [6, 1, 8] },
    readingQuestions: ["为什么只对 inside 候选区域做 framelet 平滑？", "候选区间 [α_i, β_i] 如何影响收敛速度和漏检？", "这个算法为什么不是标准 variational minimization？"],
    afterReadingOutput: "画出 below / inside / above 三分图像和候选区域收缩过程。"
  },
  {
    id: "tight-frame-vessel",
    priority: 6,
    titleCn: "Tight-frame 医学血管分割长版",
    titleEn: "Vessel Segmentation in Medical Imaging Using a Tight-Frame Based Algorithm",
    year: 2011,
    theme: "medical",
    difficulty: "中等偏难",
    prerequisites: ["tight-frame", "wavelet transform", "MRA", "finite convergence"],
    oneSentence: "把血管分割算法补成完整版本。",
    coreProblem: "真实 2D/3D MRA 图像中，血管细节、分叉和弱边界需要自动提取；算法既要保留细节，又要在少量迭代内稳定收敛。",
    whyHard: "医学血管图像的边界像素不一定形成清晰闭合曲线；PDE 和 active contour 方法常需要较强参数调节，且在 3D MRA 上计算压力明显。",
    methodHandle: "长版用 tight-frame 表示迭代细化可能边界区域。它初始化 Λ^(0) 为潜在边界像素，根据 μ、μ_-、μ_+ 得到 [α_i, β_i]，再只在 Λ 区域执行 tight-frame denoising / smoothing，并逐轮更新二值候选。",
    keyModelOrFormula: "Λ^(i+1) = {j : 0 < f_j^(i+1/2) < 1}; update only Λ; pixels mapped to 0 or 1 leave the candidate set.",
    algorithmFlow: ["初始化潜在边界集合 Λ^(0)。", "计算 μ、μ_-、μ_+ 并形成 [α_i, β_i]。", "按区间把像素映射到 0、候选值或 1。", "在 Λ 区域做 tight-frame 迭代。", "更新 Λ，直到得到二值血管图像。"],
    theoremOrGuarantee: "Theorem 1 证明 tight-frame algorithm 会收敛到二值图像；文本还强调通常几次迭代即可收敛，每轮复杂度可按像素规模理解为线性级别。",
    experimentFocus: "实验对象为 real 2D/3D MRA images；对照 PDE 和 variational methods，重点看是否提取更多 tubular objects 与 fine details。",
    howToRead: "先读 Introduction 中与 PDE/active contour 的差异；精读 Algorithm 1 和 Theorem 1；实验部分重点看 2D 与 3D MRA 的细节保持。",
    relation: { text: "它扩展了 Framelet 短版，并为 spherical wavelet segmentation 提供“候选边界区间 + wavelet/frame”思想来源。", links: [5, 8, 1] },
    readingQuestions: ["Λ^(i) 中的像素为什么是唯一需要继续处理的像素？", "Theorem 1 的 finite convergence 依赖什么事实？", "2D 与 3D MRA 实验中算法优势是否来自 tight-frame 还是候选区间策略？"],
    afterReadingOutput: "整理 Algorithm 1 的变量表：Λ、μ、μ_-、μ_+、α_i、β_i、f^(i+1/2)。"
  },
  {
    id: "slat-color",
    priority: 7,
    titleCn: "SLaT 彩色图像三阶段分割",
    titleEn: "SLaT: Smoothing, Lifting and Thresholding",
    year: 2015,
    theme: "extension",
    difficulty: "中等",
    prerequisites: ["SaT", "color spaces", "Lab", "convex Mumford-Shah variant"],
    oneSentence: "把 SaT 扩展到退化彩色图。",
    coreProblem: "RGB 通道高度相关，退化彩色图像在 noise、information loss、blur 下只靠 RGB 或单一颜色空间分割会不稳。",
    whyHard: "彩色分割需要同时处理通道相关性和感知颜色差异；如果对退化图像直接聚类，噪声和缺失会污染颜色特征；如果先单通道恢复，又可能丢失跨颜色空间信息。",
    methodHandle: "SLaT 采用 Smoothing、Lifting、Thresholding 三阶段。第一阶段每个 RGB channel 解凸 Mumford-Shah 变体得到平滑图像；第二阶段把平滑 RGB 变换到 Lab 等 secondary color space，并拼接成 RGB+Lab 六维特征；第三阶段对多通道特征做 thresholding / K-means。",
    keyModelOrFormula: "SLaT feature = [smooth RGB, transformed Lab]; Stage 1 convex smoothing has unique minimizer; Stage 3 chooses K.",
    algorithmFlow: ["对每个 RGB 通道解 convex smoothing 模型。", "得到平滑 RGB 图像。", "把平滑 RGB 转到 Lab 等 secondary color space。", "拼接 RGB+Lab 多通道特征。", "用 multichannel thresholding 或 K-means 分割。"],
    theoremOrGuarantee: "PDF 中 Theorem III.1 证明 Stage 1 restoration/smoothing 子问题在不同退化设定下存在唯一 minimizer；类别数 K 只在 Stage 3 进入。",
    experimentFocus: "实验为 synthetic and real-world degraded color images，退化包括 noise、information loss、blur；重点看 segmentation quality 与 CPU time，而不是只看视觉图。",
    howToRead: "先读 Abstract 和三阶段概览；精读 Stage 1 的模型与 Theorem III.1；再读 Lifting 的颜色空间设计；最后看 Stage 3 的 multichannel thresholding。",
    relation: { text: "它是 SaT 彩色扩展；与 Segmentation Restoration 都处理 vector-valued images，但一个两阶段/三阶段，一个 joint optimization。", links: [1, 4, 8] },
    readingQuestions: ["为什么 Lifting 不能简单理解成多加几个颜色特征？", "Stage 1 的唯一解定理对后续 thresholding 有什么意义？", "改变 K 时 SLaT 哪些步骤需要重算？"],
    afterReadingOutput: "画出 Smoothing -> Lifting -> Thresholding 的六维特征流程图。"
  },
  {
    id: "sphere-wavelet",
    priority: 8,
    titleCn: "球面小波图像分割",
    titleEn: "Wavelet-Based Segmentation on the Sphere",
    year: 2016,
    theme: "extension",
    difficulty: "困难",
    prerequisites: ["spherical sampling", "spherical wavelets", "curvelets", "tight-frame segmentation"],
    oneSentence: "把分割从平面搬到球面。",
    coreProblem: "地球、太阳、全天图和球面视网膜图像定义在 sphere 上，不能直接套用 Euclidean image 的梯度、采样、wavelet 或 tight-frame 算法。",
    whyHard: "球面数据没有平面网格的平移不变性；曲线结构和方向性内容需要 spherical directional wavelets 或 curvelets 才能有效表示，K-means 这类普通方法会忽略几何结构。",
    methodHandle: "论文把 tight-frame vessel segmentation 的候选边界区间思想迁移到球面，使用 spherical wavelets、directional wavelets 或 curvelets 做 soft-thresholding / denoising，再利用球面梯度定位潜在边界像素并迭代收缩区间。",
    keyModelOrFormula: "Euclidean tight-frame segmentation -> spherical wavelet frame; candidate boundary interval is refined on spherical samples.",
    algorithmFlow: ["在球面采样网格上表示图像。", "用 spherical wavelet / curvelet 做去噪或平滑。", "基于球面梯度找潜在边界像素。", "迭代更新边界灰度区间。", "输出球面上的分割区域。"],
    theoremOrGuarantee: "主要算法保证是继承 tight-frame 边界候选区间的收缩思路，并兼容 axisymmetric、directional、curvelet 等任意 spherical wavelet frame；严格证明更多依赖前作，需结合 Vessel/Tight-frame 论文读。",
    experimentFocus: "实验对象包括 Earth topographic map、light probe image、solar data sets 和 spherical retina images；重点看 directional wavelets / curvelets 对曲线结构的优势。",
    howToRead: "先读 Introduction 中 Euclidean 与 spherical 的差别；再读 spherical sampling、spherical gradient 和 wavelet transform；实验只看真实球面对象的失败/成功模式。",
    relation: { text: "它继承 Vessel/Tight-frame 的边界候选思想，同时是 SLaT 之外 SaT 方法向非欧氏几何域的扩展。", links: [5, 6, 7] },
    readingQuestions: ["为什么普通平面 wavelet 不能直接用于 spherical images？", "directional wavelets / curvelets 对曲线结构有什么帮助？", "这篇和 tight-frame vessel segmentation 的算法共同点是什么？"],
    afterReadingOutput: "列出 Euclidean image 与 spherical image 在采样、梯度、wavelet 表示上的三点差异。"
  },
  {
    id: "two-stage-classification",
    priority: 9,
    titleCn: "高维数据与点云两阶段分类",
    titleEn: "A Two-Stage Classification Method for High-Dimensional Data and Point Clouds",
    year: 2019,
    theme: "classification",
    difficulty: "中等偏难",
    prerequisites: ["graph Laplacian", "graph TV", "SVM", "semi-supervised classification"],
    oneSentence: "把 SaT 迁移到图分类。",
    coreProblem: "高维数据和点云没有规则图像网格；传统 graph-based variational classification 常受 unit simplex 约束、非凸或 NP-hard 形式影响，速度和可扩展性受限。",
    whyHard: "点云分类要在 k-NN 图上处理标签传播，既需要保留少量标签或 warm initialization 的信息，又要在图结构上平滑；多类问题中 K 个类别互相耦合会拖慢求解。",
    methodHandle: "论文先用 support vector machine (SVM) 或随机标签生成 fuzzy warm initialization，再在图上解无约束凸变分 smoothing 模型。该模型包含保真项、graph Laplacian 平滑项和 graph Total Variation，最后把 smoothed partition 投影到 binary partition。",
    keyModelOrFormula: "Σ_j [ β/2 ||u_j - û_j||^2 + α/2 u_j^T L u_j + ||∇u_j||_1 ].",
    algorithmFlow: ["构建 k-NN 图和图权重。", "用 SVM 或随机标签得到 warm initialization û。", "对每个类别独立求解凸 smoothing 子问题。", "把平滑标签函数投影到 simplex 顶点/二值划分。", "用结果作为新初始化重复 refinement。"],
    theoremOrGuarantee: "论文证明 smoothing convex model 有唯一解，并设计 primal-dual algorithm 求解；算法收敛性在 Section 4 中给出。",
    experimentFocus: "实验覆盖 benchmark high-dimensional data sets 和 unstructured point clouds；重点看第一次初始化后迭代 refinement 对 accuracy 与 computation speed 的提升。",
    howToRead: "先读 Abstract 和 Section 3 模型；重点看 graph Laplacian L、graph TV 和无约束凸模型；再读 Section 4 primal-dual；最后看 point cloud benchmark。",
    relation: { text: "它把 SaT 从 pixel segmentation 推到 graph classification，是 2024 期刊版的早期版本。", links: [1, 10, 3] },
    readingQuestions: ["为什么 K 个类别子问题可以独立求解？", "graph Laplacian 与 graph TV 在模型中分别起什么作用？", "projection 到 binary partition 会不会丢失 smoothing 得到的信息？"],
    afterReadingOutput: "写出 graph variational classification 的三步：warm initialization、convex smoothing、binary projection。"
  },
  {
    id: "efficient-variational-classification",
    priority: 10,
    titleCn: "高维数据高效变分分类期刊版",
    titleEn: "An Efficient and Versatile Variational Method for High-Dimensional Data Classification",
    year: 2024,
    theme: "classification",
    difficulty: "困难",
    prerequisites: ["graph Laplacian", "primal-dual", "convex optimization", "point cloud classification"],
    oneSentence: "高维图分类线的成熟版本。",
    coreProblem: "高维数据和非结构点云分类既要求速度，又要求精度；多类半监督分类中如果维持 simplex 约束或强耦合模型，计算会随类别和样本规模变重。",
    whyHard: "图上分类需要在稀疏标签、噪声初始化和高维相似图之间平衡。若每个类别标签函数互相约束，优化器难以并行；若平滑不足，projection 后分类边界会不稳定。",
    methodHandle: "期刊版强化 warm initialization、unconstrained convex variational smoothing、binary projection 和 iterative refinement。模型继续使用 graph Laplacian 与 graph Total Variation，但去掉 simplex 约束，使每个类别标签函数可独立求解，并可用 primal-dual algorithm 高效实现。",
    keyModelOrFormula: "min_U β/2 ||U - Û||_F^2 + α/2 Tr(U^T L U) + ||∇_w U||_1, then binary/argmax projection.",
    algorithmFlow: ["用 SVM 或 random labeling 生成 Û。", "建立 k-NN 图、graph Laplacian L 与 graph gradient。", "求解无 simplex 约束凸 smoothing 模型。", "用 argmax / binary projection 得到硬分类。", "用最新分类结果迭代 refinement。"],
    theoremOrGuarantee: "Theorem 1 证明 proposed model 有唯一解；Section 4 给出 specifically designed primal-dual algorithm 并说明其收敛性。",
    experimentFocus: "实验为 benchmark high-dimensional data sets 和 unstructured point clouds；读表格时比较 accuracy、CPU time 以及迭代 refinement 前后变化。",
    howToRead: "先对照 2019 Two-Stage 版本；再读 Section 3 模型，确认新增或更严谨之处；精读 Section 4 primal-dual；最后看 benchmark 表格。",
    relation: { text: "这是 Two-Stage Classification 的成熟表达，也把 ROF/TV/Mumford-Shah 工具箱抽象为图上点分类。", links: [9, 1, 2] },
    readingQuestions: ["无 simplex 约束为什么能提升并行性？", "唯一解定理保证的是 smoothing 子问题还是最终硬分类？", "2024 版相对 2019 版最重要的强化是什么？"],
    afterReadingOutput: "做一张 2019 Two-Stage 与 2024 Efficient Variational 的模型、算法、实验对比表。"
  },
  {
    id: "high-dimensional-uq",
    priority: 11,
    titleCn: "高维逆问题不确定性量化入口",
    titleEn: "Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation",
    year: 2019,
    theme: "ri-uq",
    difficulty: "中等偏难",
    prerequisites: ["Bayesian posterior", "MAP", "HPD", "credible interval", "convex optimization"],
    oneSentence: "MAP-UQ 的短入口。",
    coreProblem: "高维逆问题通常 ill-conditioned 或 ill-posed，只给 maximum a posteriori (MAP) 图像会掩盖像素和结构层面的不确定性；完整 MCMC 又太慢。",
    whyHard: "后验维度很高，先验常含 l1 或 sparsity-promoting 非光滑项；既要支持 non-smooth priors，又要能在图像/信号规模上给出 local credible intervals 和 HPD credible regions。",
    methodHandle: "文章先用 convex optimisation 求 MAP estimator，再用 probability concentration 近似 highest posterior density (HPD) credible region，随后通过像素或 superpixel 的 local credible intervals 可视化不确定性，并讨论 regularisation parameter μ 的自动选择。",
    keyModelOrFormula: "posterior p(x|y) ∝ p(y|x)p(x); MAP x_MAP = argmax p(x|y); approximate HPD threshold γ'_α or γ^0_α from concentration bounds.",
    algorithmFlow: ["建立线性观测模型 y = Φx + n。", "选择 analysis 或 synthesis prior。", "用 convex optimisation 求 MAP。", "用 probability concentration 近似 HPD credible region。", "计算 pixel/superpixel local credible intervals。"],
    theoremOrGuarantee: "主要保证是 HPD region 的可计算近似和可扩展 MAP-UQ 流程；它不声称替代完整后验采样，而是用 concentration theory 给出高维可用的近似。",
    experimentFocus: "实验包括 MRI brain image，并说明 RI image M31 结果类似；比较 orthonormal basis 与 SARA dictionary，关注 SNR、自动 μ 和 credible interval 误差。",
    howToRead: "先读 posterior、MAP、HPD、credible interval 定义；再读 γ'_α / γ^0_α 近似公式；最后看 MRI/M31 与 complete/over-complete dictionary 实验。",
    relation: { text: "它是 RI UQ II 的一般化短入口，也为 Proximal Nested Sampling 的贝叶斯模型比较做概念铺垫。", links: [13, 12, 15] },
    readingQuestions: ["MAP 为什么不能直接代表不确定性？", "HPD credible region 与 local credible interval 分别回答什么问题？", "自动选择 μ 对 UQ 可用性有什么影响？"],
    afterReadingOutput: "写出 MAP-UQ 流程卡：posterior、MAP、HPD region、local credible interval、实验图像。"
  },
  {
    id: "ri-uq-i",
    priority: 12,
    titleCn: "无线电干涉 UQ I：Proximal MCMC",
    titleEn: "Uncertainty Quantification for Radio Interferometric Imaging I: Proximal MCMC Methods",
    year: 2018,
    theme: "ri-uq",
    difficulty: "困难",
    prerequisites: ["radio interferometric imaging", "MCMC", "Moreau-Yosida", "sparse prior", "HPD"],
    oneSentence: "用 proximal MCMC 采 RI 后验。",
    coreProblem: "radio interferometric imaging 是高维病态逆问题，CLEAN、MEM、compressive sensing 通常给重建图像但不给可信区间；普通 MCMC 又难处理非光滑 sparse priors。",
    whyHard: "RI 观测 y = Φx + n 是不完整 Fourier 型测量；后验维度高，analysis/synthesis sparse prior 常含 l1 非光滑项，标准 Langevin 或 Metropolis 方法不能直接高效采样。",
    methodHandle: "论文建立 Bayesian posterior with sparse prior，区分 analysis 和 synthesis formulations，并引入 Moreau-Yosida unadjusted Langevin algorithm (MYULA) 与 proximal Metropolis-adjusted Langevin algorithm (Px-MALA)。用 posterior samples 计算 local credible intervals、HPD regions 和 image structure hypothesis testing。",
    keyModelOrFormula: "RI measurement: y = Φx + n; posterior p(x|y) ∝ exp(-f(x) - g(x)); proximal MCMC handles non-smooth g such as l1 priors.",
    algorithmFlow: ["建立 RI measurement operator Φ 和 sparse prior。", "写出 analysis/synthesis posterior。", "用 Moreau-Yosida envelope 平滑非光滑项。", "运行 MYULA 或 Px-MALA 采样后验。", "从样本计算 credible intervals、HPD regions、hypothesis tests。"],
    theoremOrGuarantee: "Px-MALA 带 Metropolis correction，理论上以目标 posterior 为不变分布；MYULA 更快但近似。论文的贡献是把 proximal calculus 接入 RI imaging 的非光滑稀疏后验采样。",
    experimentFocus: "实验使用 M31 galaxy、Cygnus A、W28 supernova remnant、3C288 等 RI images；重点看 posterior samples 如何转成 pixel-wise credible intervals 和结构假设检验。",
    howToRead: "先读 RI measurement model y = Φx + n；再读 analysis prior 与 synthesis prior；第三步读 Moreau-Yosida envelope、MYULA、Px-MALA；最后看 M31 等实验。",
    relation: { text: "它是 RI UQ II 的完整采样基准；缺点是慢，因此催生 MAP-based UQ 和一般高维 MAP-UQ。", links: [13, 11, 14] },
    readingQuestions: ["analysis prior 和 synthesis prior 的变量分别是什么？", "Moreau-Yosida envelope 为什么能处理 l1 非光滑先验？", "MYULA 与 Px-MALA 在速度和精确性上如何取舍？"],
    afterReadingOutput: "画出 RI UQ I 的 posterior sampling 到 HPD / credible interval / hypothesis testing 的流程。"
  },
  {
    id: "ri-uq-ii",
    priority: 13,
    titleCn: "无线电干涉 UQ II：MAP 快速版",
    titleEn: "Uncertainty Quantification for Radio Interferometric Imaging II: MAP Estimation",
    year: 2018,
    theme: "ri-uq",
    difficulty: "困难",
    prerequisites: ["MAP", "convex optimization", "HPD", "probability concentration", "RI imaging"],
    oneSentence: "用 MAP 近似替代慢采样。",
    coreProblem: "RI UQ I 的 MCMC 能恢复完整后验，但对 Square Kilometre Array (SKA) 级数据计算代价过高；需要可扩展的 UQ 方法。",
    whyHard: "大规模 RI 数据既要 sparse reconstruction 又要 uncertainty quantification。采样法随维度和数据量增长过快，而只求 MAP 又会丢失不确定性信息。",
    methodHandle: "论文用 convex optimisation 求 sparsity-promoting prior 下的 maximum a posteriori (MAP) point estimator，再利用 probability concentration 从 MAP 后处理得到近似 HPD credible regions、local credible intervals 和 structure hypothesis testing。",
    keyModelOrFormula: "MAP: x_MAP = argmin f_y(x) + g(x); approximate HPD credible region C'_α = {x: f(x)+g(x) ≤ γ'_α}.",
    algorithmFlow: ["建立 RI posterior 与 sparse prior。", "用 convex optimisation 求 MAP estimator。", "用 probability concentration 构造近似 HPD region。", "对 pixels / superpixels 求 local credible intervals。", "对图像结构做 hypothesis testing。"],
    theoremOrGuarantee: "核心保证是 MAP-based HPD approximation 的 concentration bound；论文报告该路线比 state-of-the-art MCMC 快约 10^5 倍，并支持 distributed / parallel algorithmic structures。",
    experimentFocus: "实验与 RI UQ I 对照，使用 M31、Cygnus A、W28、3C288；重点看 MAP-UQ 与 MCMC 在 credible intervals 和结构测试上的差异。",
    howToRead: "先读 Abstract 中三种 UQ 输出；再读 Section 2/3 的 MAP 模型和 HPD approximation 公式；最后与 RI UQ I 的 MCMC 结果逐项比较。",
    relation: { text: "它是 RI UQ I 的 scalable 替代方案，并与 high-dimensional UQ 短文共享 MAP-UQ 思想。", links: [12, 11, 14] },
    readingQuestions: ["MAP-UQ 如何从一个点估计恢复 credible region 信息？", "为什么这条路线能面向 SKA big-data？", "它相对 RI UQ I 丢失了哪些完整后验信息？"],
    afterReadingOutput: "做一张 RI UQ I 与 RI UQ II 对比表：采样/MAP、输出、速度、适用规模、风险。"
  },
  {
    id: "online-ri",
    priority: 14,
    titleCn: "在线无线电干涉成像",
    titleEn: "Online Radio Interferometric Imaging",
    year: 2019,
    theme: "ri-uq",
    difficulty: "中等偏难",
    prerequisites: ["RI visibilities", "forward-backward splitting", "MAP", "online optimization"],
    oneSentence: "边到数据边重建并丢弃。",
    coreProblem: "RI telescopes 会产生海量 visibilities；传统 offline imaging 必须等观测完成并存下全部数据后再重建，存储和计算都难以支撑 SKA big-data。",
    whyHard: "visibilities 是流式到达的 Fourier measurements；如果不能在线处理，就需要保存完整数据集并反复访问。现有 CLEAN、compressive sensing 等方法很难直接在 acquisition stage 同步运行。",
    methodHandle: "论文提出 online sparse regularisation methodology：把 visibilities 分成 data blocks，到达时 assimilate 到当前图像估计，用 online forward-backward algorithm 更新 reconstruction，然后 discard 已处理数据块，从而把存储需求降到当前 block 规模。",
    keyModelOrFormula: "online step: assimilate y_b, update x by online forward-backward / proximal step, discard y_b after use.",
    algorithmFlow: ["把 RI visibilities 划分为数据块。", "每个 block 到达时读入并 assimilate。", "用 online forward-backward 更新图像。", "丢弃已处理 block。", "继续处理下一个 block，直到观测流结束。"],
    theoremOrGuarantee: "Algorithm 1 产生的目标函数序列具有单调下降到极限的性质；关键工程保证是存储需求从全量 visibilities 降到单个 data block 级别。",
    experimentFocus: "实验使用 HI region of M31、Cygnus A、W28、3C288；重点比较 online 与 offline 的重建启动时间、总时间和数据存储压力。",
    howToRead: "先读 Abstract 的 assimilating and discarding visibilities；再读 Algorithm 1；重点看 proof 中 Fy(x^(i)) 单调下降；最后看 M31 等 simulation。",
    relation: { text: "它解决 RI big-data 的在线重建前置问题，可与 RI UQ II 的 MAP-UQ 结合面向 SKA 场景。", links: [13, 12, 11] },
    readingQuestions: ["online 与 offline 在数据到达时间线上有什么根本差异？", "discard visibilities 后还保留了什么信息？", "online forward-backward 与 MAP/convex optimisation 有什么关系？"],
    afterReadingOutput: "画出 RI 数据流：block arrival、assimilate、proximal update、discard、next block。"
  },
  {
    id: "proximal-nested-sampling",
    priority: 15,
    titleCn: "Proximal Nested Sampling 贝叶斯模型选择",
    titleEn: "Proximal Nested Sampling for High-Dimensional Bayesian Model Selection",
    year: 2022,
    theme: "bayes-model",
    difficulty: "很难",
    prerequisites: ["Bayesian evidence", "nested sampling", "Bayes factor", "proximal MCMC", "log-concave priors"],
    oneSentence: "从 UQ 推进到模型选择。",
    coreProblem: "Bayesian model selection 需要 model evidence / marginal likelihood p(y|M)，但高维图像逆问题中的 evidence 积分几乎不可直接计算。",
    whyHard: "模型比较不能只看重建误差；需要在没有 ground truth 的情况下比较 prior、dictionary、measurement model。高维、log-concave、非光滑 l1 / Total Variation priors 会让传统 nested sampling 的 constrained sampling 变得困难。",
    methodHandle: "论文把 nested sampling 与 proximal Markov chain Monte Carlo 结合。Nested sampling 将 evidence 积分转成 prior volume 上的一维积分；proximal MCMC 负责在高维、可能非光滑的 constrained likelihood contour 内采样，从而处理 l1 或 Total Variation priors 的 imaging model。",
    keyModelOrFormula: "evidence: p(y|M)=∫ p(y|x,M)p(x|M) dx; nested sampling rewrites evidence over prior volume ξ; model comparison uses Bayes factor.",
    algorithmFlow: ["定义候选 Bayesian imaging models。", "把 evidence 积分表示为 prior volume ξ 上的一维积分。", "用 nested sampling 逐步收缩 likelihood contour。", "在约束区域内用 proximal MCMC 采样。", "估计 evidence 并用 Bayes factor 比较模型。"],
    theoremOrGuarantee: "主要算法保证是把 proximal MCMC 的高维非光滑采样能力嵌入 nested sampling；摘要中明确可扩展到 O(10^6) 维及以上，并适配 log-concave non-smooth priors。",
    experimentFocus: "实验先用 large Gaussian models 验证 evidence 可用，再在 imaging problems 中分析 dictionary 和 measurement model 选择；读时关注模型比较，而不是单张图像重建质量。",
    howToRead: "先读 Bayesian evidence、Bayes factor 和 prior volume ξ；再读 proximal Langevin / MYULA 背景；最后看实验中如何用 evidence 选择 dictionary 或 measurement model。",
    relation: { text: "它是 RI/UQ/Bayesian inverse problem 线的上层延伸：从“图像是什么”升级为“哪个模型更可信”。", links: [11, 12, 13] },
    readingQuestions: ["Bayesian evidence 与 MAP reconstruction 回答的问题有何不同？", "nested sampling 为什么能把高维 evidence 积分变成一维 prior volume 积分？", "proximal MCMC 在非光滑 prior 的 constrained sampling 中承担什么角色？"],
    afterReadingOutput: "写出 evidence、Bayes factor、nested sampling prior volume ξ 三个概念的关系图。"
  }
];

const paperByPriority = new Map(papers.map((paper) => [paper.priority, paper]));
const noteByPriority = new Map(paperNotesV2.map((note) => [note.priority, note]));

let activeTrack = "all";
let query = "";
let activeNoteTheme = "all";
let noteQuery = "";

function pdfHref(file) {
  return encodeURI(basePath + file);
}

function byId(id) {
  return document.getElementById(id);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function themeLabel(key) {
  return noteThemes.find((theme) => theme.key === key)?.label || key;
}

function notePdf(note) {
  const paper = paperByPriority.get(note.priority);
  return note.pdf || paper?.file || "";
}

function noteSearchText(note) {
  return [
    note.titleCn,
    note.titleEn,
    note.year,
    themeLabel(note.theme),
    note.difficulty,
    note.prerequisites.join(" "),
    note.oneSentence,
    note.coreProblem,
    note.whyHard,
    note.methodHandle,
    note.keyModelOrFormula,
    note.algorithmFlow.join(" "),
    note.theoremOrGuarantee,
    note.experimentFocus,
    note.howToRead,
    note.relation.text,
    note.readingQuestions.join(" "),
    note.afterReadingOutput
  ].join(" ").toLowerCase();
}

function filteredNotes() {
  const normalized = noteQuery.trim().toLowerCase();
  return paperNotesV2.filter((note) => {
    const themeMatch = activeNoteTheme === "all" || note.theme === activeNoteTheme;
    const searchMatch = !normalized || noteSearchText(note).includes(normalized);
    return themeMatch && searchMatch;
  });
}

function filteredPapers() {
  const normalized = query.trim().toLowerCase();
  return papers.filter((paper) => {
    const trackMatch = activeTrack === "all" || paper.track === activeTrack;
    const searchMatch = !normalized || [
      paper.title,
      paper.file,
      paper.time,
      paper.year,
      tracks[paper.track].label,
      paper.position,
      paper.note
    ].join(" ").toLowerCase().includes(normalized);
    return trackMatch && searchMatch;
  });
}

function renderThesis() {
  byId("researchThesis").innerHTML = `
    <article class="thesis-card">
      <div>
        <p class="eyebrow">Synthesis</p>
        <h2>${thesis.headline}</h2>
      </div>
      <p>${thesis.body}</p>
      <strong>${thesis.oneLine}</strong>
    </article>
  `;
}

function renderMetrics() {
  const metrics = [
    { label: "去重后论文", value: "15", detail: "Xiaohao Cai 第一作者" },
    { label: "研究方向", value: "5", detail: "SaT / ROF / Framelet / 分类 / UQ" },
    { label: "精读笔记", value: "15", detail: "逐篇结构化卡片" },
    { label: "阅读阶段", value: "7", detail: "按知识依赖排序" }
  ];

  byId("metrics").innerHTML = metrics.map((metric) => `
    <article class="metric">
      <span>${metric.label}</span>
      <strong>${metric.value}</strong>
      <small>${metric.detail}</small>
    </article>
  `).join("");
}

function renderTrackOverview() {
  const maxCount = Math.max(...Object.values(tracks).map((track) => track.count));
  byId("trackOverview").innerHTML = Object.entries(tracks).map(([key, track]) => `
    <article class="track-card" style="--accent:${track.color}">
      <div class="track-top">
        <span>${track.short}</span>
        <strong>${track.count}</strong>
      </div>
      <h3>${track.label}</h3>
      <p>${track.summary}</p>
      <div class="bar"><i style="width:${track.count / maxCount * 100}%"></i></div>
    </article>
  `).join("");
}

function renderTimeline() {
  byId("timeline").innerHTML = chronology.map((item) => {
    const paper = paperByPriority.get(item.priority);
    return `
      <article class="time-node" style="--accent:${tracks[item.track].color}">
        <span>${item.time}</span>
        <h3>${item.label}</h3>
        <p>${paper.position}</p>
        <a href="${pdfHref(paper.file)}">#${paper.priority} 打开 PDF</a>
      </article>
    `;
  }).join("");
}

function renderFilters() {
  const filters = [{ key: "all", label: "全部" }].concat(
    Object.entries(tracks).map(([key, track]) => ({ key, label: track.label }))
  );

  byId("trackFilters").innerHTML = filters.map((filter) => `
    <button class="${filter.key === activeTrack ? "active" : ""}" data-filter="${filter.key}" type="button">${filter.label}</button>
  `).join("");
}

function renderRows() {
  const rows = filteredPapers();
  byId("paperRows").innerHTML = rows.map((paper) => {
    const track = tracks[paper.track];
    return `
      <tr>
        <td><span class="rank">${paper.priority}</span></td>
        <td>
          <strong>${paper.title}</strong>
          <small>${paper.position}</small>
          <small>${paper.note}</small>
        </td>
        <td>${paper.time}</td>
        <td><span class="chip" style="--accent:${track.color}">${track.label}</span></td>
        <td>${paper.pages}</td>
        <td><a class="file-link" href="${pdfHref(paper.file)}">PDF</a></td>
      </tr>
    `;
  }).join("");
}

function renderTrackDetails() {
  byId("trackDetails").innerHTML = Object.entries(tracks).map(([key, track]) => {
    const list = papers.filter((paper) => paper.track === key);
    return `
      <article class="track-detail" style="--accent:${track.color}">
        <span class="track-code">${track.short}</span>
        <h3>${track.label}</h3>
        <p>${track.summary}</p>
        <p class="emphasis">${track.emphasis}</p>
        <ul>
          ${list.map((paper) => `<li><a href="${pdfHref(paper.file)}">#${paper.priority} ${paper.title}</a></li>`).join("")}
        </ul>
      </article>
    `;
  }).join("");
}

function renderReadingList() {
  byId("readingList").innerHTML = readingStages.map((stage) => `
    <li class="reading-stage">
      <div class="stage-label">${stage.stage}</div>
      <div class="stage-body">
        <h3>${stage.title}</h3>
        <p>${stage.focus}</p>
        <div class="stage-papers">
          ${stage.priorities.map((priority) => {
            const paper = paperByPriority.get(priority);
            const track = tracks[paper.track];
            return `
              <a class="stage-paper" style="--accent:${track.color}" href="${pdfHref(paper.file)}">
                <span>#${paper.priority}</span>
                <strong>${paper.title}</strong>
                <small>${paper.time} · ${track.label}</small>
              </a>
            `;
          }).join("")}
        </div>
      </div>
    </li>
  `).join("");
}

function renderNoteMainlines() {
  byId("noteMainlines").innerHTML = noteMainlines.map((mainline) => `
    <article class="mainline-card">
      <h4>${escapeHtml(mainline.title)}</h4>
      <p>${escapeHtml(mainline.summary)}</p>
      <div class="mainline-papers">
        ${mainline.papers.map((priority) => {
          const note = noteByPriority.get(priority);
          return `<button type="button" data-scroll-note="${priority}">#${priority} ${escapeHtml(note.titleCn)}</button>`;
        }).join("")}
      </div>
    </article>
  `).join("");
}

function renderNoteReadingOrder() {
  byId("noteReadingOrder").innerHTML = paperNotesV2.map((note) => `
    <button type="button" data-scroll-note="${note.priority}">
      <span>#${note.priority}</span>
      ${escapeHtml(note.titleCn)}
    </button>
  `).join("");
}

function renderNoteReadingStandard() {
  byId("noteReadingStandard").innerHTML = readingStandard.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderThemeFilters() {
  byId("noteThemeFilters").innerHTML = noteThemes.map((theme) => `
    <button class="${theme.key === activeNoteTheme ? "active" : ""}" data-note-theme="${theme.key}" type="button">
      ${escapeHtml(theme.label)}
    </button>
  `).join("");
}

function renderRelationChips(note) {
  return note.relation.links.map((priority) => {
    const related = noteByPriority.get(priority);
    return `
      <button type="button" class="relation-chip" data-scroll-note="${priority}">
        #${priority} ${escapeHtml(related?.titleCn || `论文 ${priority}`)}
      </button>
    `;
  }).join("");
}

function renderNoteList(title, items) {
  return `
    <section class="note-detail-block">
      <strong>${escapeHtml(title)}</strong>
      <ul>
        ${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </section>
  `;
}

function renderNotes() {
  const notes = filteredNotes();
  byId("noteCount").textContent = `${notes.length} / ${paperNotesV2.length} 篇`;
  byId("noteGrid").innerHTML = notes.map((note) => {
    const paper = paperByPriority.get(note.priority);
    const track = tracks[paper.track];
    const pdf = notePdf(note);
    return `
      <article class="note-card" id="note-${note.priority}" style="--accent:${track.color}">
        <header class="note-head">
          <span>#${note.priority}</span>
          <div>
            <div class="note-meta">
              <span>${escapeHtml(String(note.year))}</span>
              <span>${escapeHtml(themeLabel(note.theme))}</span>
              <span>${escapeHtml(note.difficulty)}</span>
              <span>阅读顺序 ${note.priority} / ${paperNotesV2.length}</span>
            </div>
            <h3>${escapeHtml(note.titleCn)}</h3>
            <small>${escapeHtml(note.titleEn)} · ${paper.pages} 页</small>
          </div>
        </header>

        <div class="note-prereq">
          ${note.prerequisites.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}
        </div>

        <div class="note-summary">
          <section>
            <strong>一句话定位</strong>
            <p>${escapeHtml(note.oneSentence)}</p>
          </section>
          <section>
            <strong>核心问题</strong>
            <p>${escapeHtml(note.coreProblem)}</p>
          </section>
          <section>
            <strong>方法抓手</strong>
            <p>${escapeHtml(note.methodHandle)}</p>
          </section>
        </div>

        <div class="note-expanded" id="note-detail-${note.priority}">
          <section class="note-detail-block">
            <strong>为什么难</strong>
            <p>${escapeHtml(note.whyHard)}</p>
          </section>
          <section class="note-detail-block">
            <strong>关键模型 / 公式</strong>
            <code>${escapeHtml(note.keyModelOrFormula)}</code>
          </section>
          ${renderNoteList("算法流程", note.algorithmFlow)}
          <section class="note-detail-block">
            <strong>理论保证 / 定理结论</strong>
            <p>${escapeHtml(note.theoremOrGuarantee)}</p>
          </section>
          <section class="note-detail-block">
            <strong>实验重点</strong>
            <p>${escapeHtml(note.experimentFocus)}</p>
          </section>
          <section class="note-detail-block">
            <strong>怎么精读</strong>
            <p>${escapeHtml(note.howToRead)}</p>
          </section>
          <section class="note-detail-block">
            <strong>关联关系</strong>
            <p>${escapeHtml(note.relation.text)}</p>
            <div class="relation-chips">${renderRelationChips(note)}</div>
          </section>
          ${renderNoteList("精读问题", note.readingQuestions)}
          <section class="note-detail-block output-block">
            <strong>读后产出</strong>
            <p>${escapeHtml(note.afterReadingOutput)}</p>
          </section>
        </div>

        <div class="note-actions">
          <a href="${pdfHref(pdf)}">打开 PDF</a>
          <a href="00_papers_first_author_xiaohao_cai_deduped/agent_team_reading_report.md">完整报告</a>
          <button type="button" class="note-toggle" data-note-toggle="${note.priority}" aria-expanded="false" aria-controls="note-detail-${note.priority}">展开精读字段</button>
        </div>
      </article>
    `;
  }).join("");
}

function scrollToNote(priority) {
  switchView("notes");
  activeNoteTheme = "all";
  noteQuery = "";
  const search = byId("noteSearchInput");
  if (search) search.value = "";
  renderThemeFilters();
  renderNotes();
  requestAnimationFrame(() => {
    const card = byId(`note-${priority}`);
    if (!card) return;
    card.classList.add("expanded");
    const button = card.querySelector("[data-note-toggle]");
    if (button) {
      button.setAttribute("aria-expanded", "true");
      button.textContent = "收起精读字段";
    }
    card.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function switchView(viewId) {
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === viewId);
  });
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.view === viewId);
  });
}

function bindEvents() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => switchView(item.dataset.view));
  });

  byId("searchInput").addEventListener("input", (event) => {
    query = event.target.value;
    renderRows();
    switchView("papers");
  });

  byId("trackFilters").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    activeTrack = button.dataset.filter;
    renderFilters();
    renderRows();
  });

  byId("noteThemeFilters").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-note-theme]");
    if (!button) return;
    activeNoteTheme = button.dataset.noteTheme;
    renderThemeFilters();
    renderNotes();
  });

  byId("noteSearchInput").addEventListener("input", (event) => {
    noteQuery = event.target.value;
    renderNotes();
  });

  byId("noteGrid").addEventListener("click", (event) => {
    const toggle = event.target.closest("button[data-note-toggle]");
    if (toggle) {
      const card = byId(`note-${toggle.dataset.noteToggle}`);
      const expanded = card.classList.toggle("expanded");
      toggle.setAttribute("aria-expanded", String(expanded));
      toggle.textContent = expanded ? "收起精读字段" : "展开精读字段";
      return;
    }

    const relation = event.target.closest("button[data-scroll-note]");
    if (relation) {
      scrollToNote(relation.dataset.scrollNote);
    }
  });

  byId("noteMainlines").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-scroll-note]");
    if (button) scrollToNote(button.dataset.scrollNote);
  });

  byId("noteReadingOrder").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-scroll-note]");
    if (button) scrollToNote(button.dataset.scrollNote);
  });
}

function init() {
  renderThesis();
  renderMetrics();
  renderTrackOverview();
  renderTimeline();
  renderFilters();
  renderRows();
  renderTrackDetails();
  renderReadingList();
  renderNoteMainlines();
  renderNoteReadingOrder();
  renderNoteReadingStandard();
  renderThemeFilters();
  renderNotes();
  bindEvents();
}

document.addEventListener("DOMContentLoaded", init);
