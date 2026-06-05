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
    authors: "Xiaohao Cai, Raymond Chan, Tieyong Zeng",
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
    title: "Linkage Between Piecewise Constant Mumford-Shah Model and ROF Model and Its Virtue in Image Segmentation",
    authors: "Xiaohao Cai, Raymond Chan, Carola-Bibiane Schönlieb, Gabriele Steidl, Tieyong Zeng",
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
    authors: "Xiaohao Cai, Gabriele Steidl",
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
    authors: "Xiaohao Cai",
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
    authors: "Xiaohao Cai, Raymond H. Chan, Serena Morigi, Fiorella Sgallari",
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
    authors: "Xiaohao Cai, Raymond Chan, Serena Morigi, Fiorella Sgallari",
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
    authors: "Xiaohao Cai, Raymond Chan, Mila Nikolova, Tieyong Zeng",
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
    authors: "Xiaohao Cai, Christopher G. R. Wallis, Jennifer Y. H. Chan, Jason D. McEwen",
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
    authors: "Xiaohao Cai, Raymond Chan, Xiaoyu Xie, Tieyong Zeng",
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
    authors: "Xiaohao Cai, Raymond H. Chan, Xiaoyu Xie, Tieyong Zeng",
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
    authors: "Xiaohao Cai, Marcelo Pereyra, Jason D. McEwen",
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
    authors: "Xiaohao Cai, Marcelo Pereyra, Jason D. McEwen",
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
    authors: "Xiaohao Cai, Marcelo Pereyra, Jason D. McEwen",
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
    authors: "Xiaohao Cai, Luke Pratley, Jason D. McEwen",
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
    authors: "Xiaohao Cai, Jason D. McEwen, Marcelo Pereyra",
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
    titleEn: "Linkage Between Piecewise Constant Mumford-Shah Model and ROF Model and Its Virtue in Image Segmentation",
    year: 2019,
    theme: "sat-rof",
    difficulty: "困难",
    prerequisites: ["BV space", "Total Variation", "ROF", "Chan-Vese", "partial minimizer"],
    oneSentence: "证明 ROF 阈值化支撑 PCMS 分割。",
    coreProblem: "Rudin-Osher-Fatemi (ROF) 是图像恢复模型，piecewise constant Mumford-Shah (PCMS) / Chan-Vese 是分割模型；论文要回答二者为什么不是经验拼接，而能在变分意义下连接。",
    whyHard: "PCMS 的边界和区域常数耦合导致非凸，直接求解多相问题代价高；ROF 虽是凸恢复，但它的 minimizer 经过 thresholding 后是否仍对应分割能量的局部最优，需要严格说明。",
    methodHandle: "论文先分析 K=2 的 T-ROF 与 PCMS 关系，再把 ROF minimizer 的 level set/threshold set 与 PCMS partial minimizer 联系起来。核心操作是求 ROF minimizer u*，再取 threshold set Σ = {x : u*(x) > (m0 + m1) / 2}，从而把恢复解转成分割区域。",
    keyModelOrFormula: "ROF: TV(u) + μ/2 ∫(u - f)^2 dx; PCMS/Chan-Vese: perimeter(Σ) + λ ∫ data fitting; threshold: Σ = {x : u*(x) > (m0 + m1)/2}; K=2 linkage uses λ = μ/[2(m1-m0)].",
    algorithmFlow: ["固定或初始化二相/多相阈值。", "先求解一次 ROF 凸恢复子问题得到 u*。", "用当前阈值把 ROF minimizer 的 level set 转为区域 Σ。", "按区域均值更新阈值 τ_i = (m_{i-1}+m_i)/2。", "检查 Σ 与 PCMS partial minimizer / T-ROF solution 的关系。"],
    theoremOrGuarantee: "精读时以 Theorem 3.6 或本地 PDF 对应主定理为核心：K=2 时，ROF minimizer 的合适阈值集可给出 PCMS/Chan-Vese 的 partial minimizer；多相 K>2 仍有类似结论，但依赖更具体的假设和阈值结构。",
    experimentFocus: "实验部分展示 T-ROF 在 missing pixels、close-intensity、Gaussian noisy multiphase、MRI、stripe 和 retina vessel 等案例上的分割效率和质量，重点不是新数据集，而是验证理论范式带来的计算优势。",
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
    theoremOrGuarantee: "Theorem 1 证明 tight-frame algorithm 会有限步收敛到二值图像；文本还强调通常几次迭代即可收敛，每轮复杂度为 O(n)，n 为像素/体素规模。",
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

const noteEnhancements = {
  "sat-overview": {
    evidence: ["Abstract", "Introduction", "SaT Methodology", "Theorem 1", "T-ROF Method", "SLaT Method", "vascular / sphere application sections"],
    reportExpansion: {
      context: "这篇综述适合作为整组论文的入口，因为它不是只总结一个算法，而是把 smoothing and thresholding (SaT) 作为方法论框架来组织 T-ROF、SLaT、vascular segmentation、spherical segmentation 等分支。精读时应先把它当成索引，而不是当成证明论文。",
      technicalReading: "技术阅读重点是 SaT 的两段式分解：第一段用 convex smoothing model 得到稳定图像或特征表示，第二段用 thresholding、K-means 或 multichannel clustering 得到分割。要特别注意类别数 K 只进入 thresholding 阶段，因此改变 K 时通常不用重解 smoothing 子问题。",
      theoremReading: "综述里出现的 Theorem 1 主要用于说明 smoothing 子问题的可解性或稳定性，但真正支撑 ROF thresholding 与 PCMS partial minimizer 关系的细节，需要回到 Linkage 和 Multiclass T-ROF 两篇。这里的理论阅读目标是标出哪些结论来自综述，哪些要回原文核对。",
      experimentReading: "实验和应用阅读不要泛泛看图片，而要按数据域分类：synthetic retina 检查噪声鲁棒性，degraded color images 检查 SLaT，vascular structures 检查候选边界策略，spherical images 检查球面几何扩展。",
      relationReading: "它与 Linkage 的关系是“地图 vs 理论根基”，与 SLaT、Framelet/Tight-frame、Wavelet Sphere 的关系是“总方法论 vs 应用分支”。读完这篇后，应能把 dashboard 中 15 篇论文按五条主线放回正确位置。",
      researchValue: "这篇的研究价值在于帮助读者看到一条可迁移范式：先把退化图像或复杂数据变成稳定表示，再用低成本阈值化或投影得到结构结果。这为后续做深度特征 SaT、graph SaT 或球面分割扩展提供了清楚入口。"
    }
  },
  "pcms-rof-linkage": {
    evidence: ["Abstract", "Introduction", "Theorem 3.6", "Section 4 T-ROF Algorithm", "Experiments"],
    reportExpansion: {
      context: "这篇是整组 SaT/ROF/PCMS 论文的理论核心。它回答的是一个基础问题：Rudin-Osher-Fatemi (ROF) 本来是恢复模型，piecewise constant Mumford-Shah (PCMS) / Chan-Vese 本来是分割模型，为什么恢复解经过 thresholding 后可以解释为分割解。",
      technicalReading: "技术阅读应围绕三个对象展开：ROF minimizer u*、二相区域常数 m0/m1、threshold set Σ = {x : u*(x) > (m0 + m1)/2}。不要把阈值化看成后处理技巧，而要把它理解为 ROF level set 与 PCMS energy 的连接点。",
      theoremReading: "Theorem 3.6 或本地 PDF 对应主定理是精读核心。重点不是背定理陈述，而是理解 partial minimizer 的含义：结论保证的是在固定某些变量或特定阈值结构下的局部/部分最优，不是直接宣称全局最优。K=2 最清楚，K>2 需要额外假设。",
      experimentReading: "实验阅读要对照 Section 5 的 missing pixels、close-intensity、Gaussian noisy multiphase、MRI、stripe 和 retina vessel 例子，检查 T-ROF 是否因只解一次 ROF 恢复子问题而获得速度优势，以及分割质量是否仍能贴近 PCMS/Chan-Vese 的目标。",
      relationReading: "它为 SaT Overview 中的“恢复 + 阈值化”提供理论合法性，也解释 Multiclass T-ROF 为什么不是经验算法。它与 Segmentation Restoration 的区别是：这篇强调 two-stage thresholding 的理论连接，后者强调 joint optimization。",
      researchValue: "如果后续要做 SaT/T-ROF 的 K>2 理论、阈值稳定性或深度特征阈值化，这篇提供了最重要的数学模板：把一个看似启发式的算法步骤放回变分能量和 minimizer 关系中分析。"
    }
  },
  "iterated-rof": {
    evidence: ["Abstract", "Algorithm T-ROF", "threshold update τ_i = 1/2(m_{i-1}+m_i)", "convergence discussion", "Experiments"],
    reportExpansion: {
      context: "这篇可以看作 T-ROF 的算法原型，位置在 Linkage 之后最合适。Linkage 告诉你为什么 ROF thresholding 有理论意义，这篇告诉你多类分割时阈值如何自动更新、如何落成可运行算法。",
      technicalReading: "技术阅读的抓手是 solve ROF once 和 iterative threshold update 的配合。先用 ROF 平滑输入图像，再根据当前分割计算区域均值 m_i，用 τ_i = 1/2(m_{i-1}+m_i) 更新相邻类阈值。这样多类分割不必直接求解完整非凸 PCMS。",
      theoremReading: "理论部分应关注 projected T-ROF 在特定假设下的收敛条件，以及 K=2 时与 Chan-Vese 之间的等价或对应关系。要注意这里的收敛不是任意图像任意 K 的全局保证，而是对阈值序列和投影过程的条件性保证。",
      experimentReading: "实验阅读重点是 cartoon、texture、medical images 中灰度接近类别的分割表现。应记录哪些例子是单次阈值化失败而迭代阈值成功，以及速度优势是否来自只求解一次 ROF。",
      relationReading: "它是 SaT Overview 中 T-ROF 分支的原始算法来源，也是 Linkage 后续理论化的前身。与 Segmentation Restoration 相比，它保留 two-stage 结构；与 SLaT 相比，它处理灰度/多类阈值，而不是彩色特征 lifting。",
      researchValue: "这篇适合提炼可复现算法：输入、ROF 解、区域均值、阈值更新、停止条件都很清楚。读完后可以直接把它改成伪代码或小实验，用来观察 K、噪声、灰度间隔对分割稳定性的影响。"
    }
  },
  "segmentation-restoration": {
    evidence: ["Abstract", "model E(u,c,g)", "Algorithm 1", "convergence theorem", "Experiments: noise / blur / missing pixels / vector-valued images"],
    reportExpansion: {
      context: "这篇处在 SaT/ROF 基础之后，是因为它代表另一条路线：不是先恢复再分割，而是把恢复变量 g 和分割变量 u_i、区域常数 c_i 放进同一个能量函数中同时协调。",
      technicalReading: "技术阅读应先标清 f、g、A、u_i、c_i 的角色。f 是观测图像，g 是待恢复图像，A 是退化算子，u_i 是区域 indicator 或 label 函数，c_i 是区域常数。核心能量是 μΦ(f,Ag)+λΣ_i∫(g-c_i)^2u_i+Σ_i TV(u_i)。",
      theoremReading: "理论阅读关注 alternating minimization 的可解性和收敛性：固定两类变量后更新第三类变量，尤其是 g 子问题在什么条件下有唯一解，三变量迭代在 mild condition 下能得到怎样的稳定结论。",
      experimentReading: "实验必须按退化类型读：high noise、blur、missing pixels、vector-valued images。每类实验都应问：如果没有恢复变量 g，传统 PCMS 会在哪里失败；加入 restoration fidelity 后具体改善什么。",
      relationReading: "它与 SaT Overview 共享 restoration helps segmentation 的思想，但技术路线不同：SaT 是 two-stage，改变 K 只重做 thresholding；这篇是 joint optimization，变量耦合更强但能直接处理 A 和 Φ。",
      researchValue: "这篇给后续医学成像、遥感或缺失数据分割一个清晰入口：当退化模型 A 已知或可建模时，与其把恢复和分割割裂，不如研究一个包含 fidelity、region fitting 和 Total Variation 的联合变分模型。"
    }
  },
  "framelet-tubular": {
    evidence: ["Abstract", "Section 3 algorithm", "boundary interval [α_i, β_i]", "finite convergence theorem", "2D/3D tubular experiments"],
    reportExpansion: {
      context: "这篇是管状结构分割线的短版入口，适合先读来理解思想。目标对象不是普通区域分割，而是 MRA 血管、道路等细长结构，它们的边界弱、分叉多、噪声下容易断裂。",
      technicalReading: "技术抓手是 possible boundary gray interval [α_i, β_i]。算法每轮把像素分成 below、inside、above，只对 inside 的候选边界区域做 framelet denoising / soft-thresholding，而不是对整幅图做统一平滑。",
      theoremReading: "理论阅读重点是 finite convergence：候选边界集合在迭代中持续收缩，已经确定为 0 或 1 的像素离开候选区。要理解这个保证与传统 variational minimization 不同，它更像有限步分类和局部平滑的组合。",
      experimentReading: "实验要看 2D/3D tubular structures，尤其是细血管、弱边界、分叉处是否保留。不要只看最终二值图，还要看候选区域收缩是否可能导致漏检或断裂。",
      relationReading: "它是 Tight-frame Vessel 长版的基础，也与 SaT 有相似结构：先稳定不确定部分，再阈值化得到结构。但它更强调候选边界区间和 framelet 表示，而不是 ROF/PCMS 理论。",
      researchValue: "这篇的价值是给出一种局部处理策略：复杂图像中不是所有像素都同等困难，真正值得用 framelet 平滑的是边界候选集合。这种思想可迁移到医学点云、血管中心线和遥感线状目标。"
    }
  },
  "tight-frame-vessel": {
    evidence: ["Abstract", "Algorithm 1", "Theorem 1", "O(n) complexity statement", "2D/3D MRA experiments"],
    reportExpansion: {
      context: "这篇是管状结构分割线的长版或完整版本，补足短版中没有展开的 tight-frame 迭代、MRA 实验和有限收敛证明。读它时应把重点放在真实 2D/3D 医学血管数据。",
      technicalReading: "技术阅读围绕 Λ possible boundary set 展开。算法初始化 Λ^(0)，计算 μ、μ_-、μ_+ 并形成 [α_i,β_i]，再只在 Λ 区域进行 tight-frame smoothing。每一轮将部分像素固定为 0 或 1，剩余像素继续进入下一轮。",
      theoremReading: "Theorem 1 说明算法会有限步收敛到二值图像；文本还给出每轮复杂度 O(n) 的线性规模解释。精读时要把 n、Λ、候选像素离开机制和 finite convergence 联系起来。",
      experimentReading: "实验重点是真实 2D/3D MRA images。应观察它相对 PDE、active contour 或其他 variational methods 是否能提取更多 fine tubular details，以及 3D 场景中参数和运行时间是否稳定。",
      relationReading: "它扩展 Framelet Tubular，并直接启发 Wavelet Sphere 中的边界候选区间思想。与 SaT/ROF 线相比，它的理论核心不是 PCMS partial minimizer，而是候选集合收缩和 tight-frame 表示。",
      researchValue: "这篇适合提炼为医学图像算法模板：先找不确定边界集合，再把高成本平滑限制在局部区域，并用有限收敛和 O(n) 复杂度说明工程可行性。"
    }
  },
  "slat-color": {
    evidence: ["Abstract", "Smoothing stage", "Theorem III.1", "Algorithm 1", "Lifting to Lab / RGB+Lab", "Experiments degraded color images"],
    reportExpansion: {
      context: "SLaT 是 SaT 从灰度图到彩色图像的自然扩展。它解决的问题不是简单把 ROF 分别用于 R、G、B 三通道，而是处理 RGB 通道相关、颜色空间不足和退化彩色图像分割不稳定。",
      technicalReading: "技术路线是 Smoothing、Lifting、Thresholding。先对 RGB 每个通道做 convex smoothing，再将平滑后的 RGB 转换到 Lab，并拼成 RGB+Lab 六维特征，最后在六维特征上做 K-means 或 multichannel thresholding。",
      theoremReading: "Theorem III.1 是 smoothing stage 的唯一解或可解性核心。精读时要分清理论保证主要覆盖第一阶段的凸模型，而最后的 K-means / thresholding 更偏算法步骤。",
      experimentReading: "实验对象是 degraded color images，退化包括 noise、information loss 和 blur。阅读时要比较 RGB-only、Lab-only 和 RGB+Lab 是否真的提供互补信息，而不是只记录最终视觉效果。",
      relationReading: "它与 SaT Overview 是总分关系；与 Segmentation Restoration 都处理 vector-valued images，但路线不同：SLaT 是三阶段 feature lifting，Segmentation Restoration 是 joint energy with g。",
      researchValue: "SLaT 的价值在于把颜色空间作为可设计的中间表示。后续如果把 Lab 换成深度特征、医学多模态通道或遥感光谱通道，这篇提供了“平滑后再提升特征维度”的清晰范式。"
    }
  },
  "sphere-wavelet": {
    evidence: ["Abstract", "spherical gradient", "spherical wavelet transform", "axisymmetric/directional wavelets and curvelets", "boundary interval shrinkage", "Earth / solar / spherical retina experiments"],
    reportExpansion: {
      context: "这篇把 wavelet/frame segmentation 从 Euclidean image 推广到 spherical images。它的关键背景是地球、太阳、全天图和球面视网膜这类数据定义在球面上，不能直接套平面梯度和普通卷积。",
      technicalReading: "技术阅读应拆成三块：球面采样和 spherical gradient，spherical wavelet transform，包括 axisymmetric wavelets、directional wavelets、curvelets，以及继承自 vessel segmentation 的 boundary candidate interval shrinkage。",
      theoremReading: "这篇的理论重点更多是算法构造和几何适配，而不是单个强定理。应核对 spherical transform 与 Euclidean tight-frame 的对应关系，并标注哪些性质来自球面小波工具，哪些来自候选区间迭代。",
      experimentReading: "实验对象包括 Earth topographic map、light probe image、solar data sets、spherical retina images。读实验时要问 directional wavelets / curvelets 是否更适合曲线结构，以及球面几何是否避免了投影到平面带来的失真。",
      relationReading: "它继承 Framelet/Tight-frame Vessel 的候选边界思想，同时构成 SaT 应用中的几何扩展分支。与 SLaT 并列：SLaT 扩展特征空间，Wavelet Sphere 扩展数据定义域。",
      researchValue: "这篇适合引出 spherical CNN、球面遥感和全景医学图像分割的后续选题。其研究价值不在于替代深度模型，而在于提供可解释的球面几何和小波稀疏表示基线。"
    }
  },
  "two-stage-classification": {
    evidence: ["Abstract", "warm initialization", "graph Laplacian", "graph TV", "unconstrained convex model", "projection to binary partition", "point cloud experiments"],
    reportExpansion: {
      context: "这篇是从 image segmentation 到 graph classification 的迁移入口。它把像素区域分割抽象成图上标签函数分类，目标对象变成 high-dimensional data 和 point clouds。",
      technicalReading: "技术路线是 two-stage：先用 SVM 或随机标签做 warm initialization，再在图上解无约束凸变分模型，包含 fidelity、graph Laplacian 和 graph Total Variation (graph TV)，最后投影到 binary partition 或 simplex 顶点。",
      theoremReading: "理论阅读要关注为什么去掉 simplex constraint 后 K 个类别子问题可以独立求解，以及 convex smoothing 子问题如何保证可计算性。这里的重点不是证明分类全局最优，而是用凸模型替代 NP-hard 或强约束图分割。",
      experimentReading: "实验覆盖 benchmark high-dimensional data sets 和 unstructured point clouds。读表时要记录初始化方式、迭代 refinement、accuracy 和 CPU time，而不是只看最终分类率。",
      relationReading: "它把 SaT 的 smoothing + thresholding 迁移成 graph smoothing + projection，是 Efficient Variational Classification 的早期版本。与 SaT/ROF 的关系是思想迁移，不是模型公式一一相同。",
      researchValue: "这篇提供了可解释图分类路线，适合发展到医学点云、遥感点云和少标签半监督分类。研究入口在于图构建、graph TV 权重、初始化质量和 projection 误差。"
    }
  },
  "efficient-variational-classification": {
    evidence: ["Abstract", "Section 3 model", "graph Laplacian / graph TV", "unique solution theorem", "Section 4 primal-dual algorithm", "primal-dual convergence", "benchmarks"],
    reportExpansion: {
      context: "这是高维分类线的成熟期或期刊版，应该和 Two-Stage Classification 对照阅读。它把早期 two-stage 思想补成更完整的模型、唯一解、primal-dual algorithm 和系统 benchmark。",
      technicalReading: "技术阅读重点是 multi-class semi-supervised classification 的无约束凸模型。每个类别 j 对应一个标签函数 u_j，目标函数包含 β/2||u_j-û_j||²、α/2 u_j^T L u_j 和 ||∇u_j||_1，K 个子问题可独立求解。",
      theoremReading: "理论部分应抓住 smoothing convex model 的 unique solution 和 primal-dual algorithm convergence。要理解这些保证覆盖的是连续/松弛标签函数的优化，不等于投影后 hard labels 一定全局最优。",
      experimentReading: "实验覆盖 high-dimensional benchmark data 和 point clouds。精读时要比较 accuracy、CPU time、迭代次数、初始化方式，以及它相对 graph cut、MBO 或其他半监督方法的速度与准确率。",
      relationReading: "它是 Two-Stage Classification 的成熟表达，也把 SaT 从像素分割进一步抽象为图上分类。与 RI/UQ 线相比，它共享 convex optimization 风格，但问题对象是 graph labels 而不是 posterior imaging。",
      researchValue: "这篇适合用于后续工程化复现：模型清楚、primal-dual 流程明确、实验对象标准。它也提供了研究少标签、点云分类和图正则可解释性的入口。"
    }
  },
  "high-dimensional-uq": {
    evidence: ["Abstract", "posterior and MAP definitions", "HPD credible region approximation", "local credible intervals", "automatic regularization parameter μ", "MRI brain and M31 experiments"],
    reportExpansion: {
      context: "这篇是进入高维逆问题不确定性量化的短入口。它把 radio interferometric imaging 中的 MAP-UQ 思路抽象到更一般的 high-dimensional inverse problems，适合在 RI UQ I/II 前预读。",
      technicalReading: "技术阅读要先建立 posterior、maximum a posteriori (MAP)、highest posterior density (HPD) credible region 和 local credible interval 的概念。方法不是完整采样后验，而是先求 MAP estimator，再用 probability concentration 近似 HPD region。",
      theoremReading: "理论重点是 γ'_α 或 γ^0_α 类型的 HPD credible region approximation，以及这种近似在 log-concave posterior 下为什么能给高维可计算 UQ。要注意它是近似 UQ，不等同于完整 MCMC。",
      experimentReading: "实验包括 MRI brain image，并说明 RI image M31 结果类似。应关注自动选择 regularization parameter μ、orthonormal basis 与 SARA dictionary 的差异、credible interval 与误差的对应关系。",
      relationReading: "它与 RI UQ II 共享 MAP + concentration 的可扩展路线，与 RI UQ I 的完整 posterior sampling 形成取舍对照。它也为其他医学成像逆问题提供迁移模板。",
      researchValue: "这篇的研究价值在于把不确定性量化从“昂贵采样”变成“可扩展凸优化后处理”。后续可以把 MAP-UQ 用于 MRI、CT、PET 或遥感重建。"
    }
  },
  "ri-uq-i": {
    evidence: ["Abstract", "RI measurement y = Φx + n", "analysis/synthesis priors", "MYULA and Px-MALA", "credible intervals / HPD regions / hypothesis testing", "M31 / Cygnus A / W28 / 3C288 experiments"],
    reportExpansion: {
      context: "RI UQ I 是无线电干涉成像不确定性量化的完整采样版。它先回答“怎样从 posterior samples 得到可信区间和结构假设检验”，再让 RI UQ II 讨论如何用 MAP 近似扩展到大数据。",
      technicalReading: "技术阅读应从 RI measurement model y = Φx + n 开始，区分 analysis prior 和 synthesis prior。由于 sparse prior 常含 l1 非光滑项，论文使用 Moreau-Yosida envelope、proximal operator、MYULA 和 Px-MALA 让 MCMC 能处理非光滑后验。",
      theoremReading: "理论重点不是单一闭式定理，而是 proximal MCMC 对 non-smooth priors 的支持机制。要理解 MYULA 如何用平滑 envelope 构造 Langevin 近似，Px-MALA 如何通过 Metropolis-Hastings 校正采样误差。",
      experimentReading: "实验使用 M31 galaxy、Cygnus A、W28 supernova remnant、3C288 等 RI images。阅读时要把 posterior samples 如何转成 pixel-wise credible intervals、HPD regions 和 structure hypothesis testing 逐项记录。",
      relationReading: "它是 RI UQ II 的完整采样基准，也是 Proximal Nested Sampling 中 proximal MCMC 背景的来源之一。缺点是慢，因此引出 MAP-UQ 和 online / scalable 方法。",
      researchValue: "这篇适合作为 Bayesian imaging UQ 的标准模板：先明确定义 posterior，再用能处理非光滑 prior 的采样器，再把样本转化为可解释的不确定性产品。"
    }
  },
  "ri-uq-ii": {
    evidence: ["Abstract", "MAP estimation model", "probability concentration", "HPD credible region approximation", "local credible intervals", "structure hypothesis testing", "10^5 speed comparison"],
    reportExpansion: {
      context: "RI UQ II 是 RI UQ I 的 scalable 替代方案。它保留 UQ 的核心输出，但用 maximum a posteriori (MAP) estimation 和 probability concentration 避免完整 MCMC 的计算成本。",
      technicalReading: "技术阅读重点是先用 convex optimization 得到 x_MAP，再用 concentration inequality 构造近似 HPD credible regions，进而得到 local credible intervals 和 structure hypothesis testing。这里 MAP 不是终点，而是 UQ 近似的入口。",
      theoremReading: "理论阅读要抓住 approximate HPD credible region 的公式和假设边界。它依赖 posterior concentration，因此可以很快，但相对 RI UQ I 会丢失完整后验样本的某些细节。",
      experimentReading: "实验与 RI UQ I 对照，使用 M31、Cygnus A、W28、3C288。重点看 MAP-UQ 与 MCMC 的 credible interval 差异、结构测试是否一致，以及约 10^5 倍速度提升来自哪里。",
      relationReading: "它与 Quantifying UQ 是同一 MAP-UQ 思想的专门版与一般版；与 Online RI Imaging 一起面向 Square Kilometre Array (SKA) 级 big-data setting。",
      researchValue: "这篇的价值在于把 UQ 做成可扩展工具。后续研究可以围绕近似误差、保守性、不同 prior 下的 HPD region 质量，以及医学或遥感 inverse problem 迁移展开。"
    }
  },
  "online-ri": {
    evidence: ["Abstract", "Algorithm 1", "assimilate and discard data blocks", "online forward-backward algorithm", "storage one block", "M31 / Cygnus A / W28 / 3C288 experiments"],
    reportExpansion: {
      context: "Online RI Imaging 解决的是无线电干涉成像的数据流和存储问题。它不直接主打 UQ，而是回答 SKA 级数据下能不能边观测、边更新、边丢弃已经处理过的 visibilities。",
      technicalReading: "技术阅读应抓住 data block 思想：visibilities 分块到达，每个 block 被 assimilate 后进入 online forward-backward / proximal update，然后 discard。算法状态保留在当前图像和必要中间量中，而不是全量数据。",
      theoremReading: "理论部分关注 Algorithm 1 的目标函数序列单调下降或收敛性质，以及为什么存储需求可以从全量 visibilities 降到单个 data block 规模。这里的保证偏优化和工程可扩展性。",
      experimentReading: "实验使用 HI region of M31、Cygnus A、W28、3C288。要比较 online 与 offline 的 reconstruction quality、开始重建时间、总运行时间和内存/存储压力。",
      relationReading: "它与 RI UQ II 共同面向 big-data RI imaging：一个解决在线重建和存储，另一个解决不确定性近似。它也可作为未来 online MAP-UQ 的前置模块。",
      researchValue: "这篇的价值在于把 inverse problem 算法放回数据获取流程中考虑。后续可研究 streaming UQ、adaptive data acquisition 或在线 proximal algorithm 在其他成像系统中的应用。"
    }
  },
  "proximal-nested-sampling": {
    evidence: ["Abstract", "Bayesian evidence / marginal likelihood", "Bayes factor", "prior volume ξ", "nested sampling algorithm", "proximal MCMC constrained sampling", "O(10^6) dimension statement"],
    reportExpansion: {
      context: "Proximal Nested Sampling 是 Bayesian inverse problem 线的上层延伸。RI UQ I/II 问的是给定模型下图像和不确定性是什么，这篇进一步问：不同模型、prior 或 dictionary 中哪个更可信。",
      technicalReading: "技术阅读从 Bayesian evidence p(y|M) 和 Bayes factor 开始。Nested sampling 将高维 evidence 积分改写为 prior volume ξ 上的一维积分；proximal MCMC 则负责在 likelihood contour 约束下处理高维、log-concave、非光滑 prior。",
      theoremReading: "理论重点是 nested sampling 的 prior volume 变换和 proximal MCMC 对 constrained sampling 的适配。要理解它为什么能处理 l1 / Total Variation 等 non-smooth imaging models，以及 O(10^6) 维可扩展性意味着什么。",
      experimentReading: "实验先用 large Gaussian models 验证 evidence estimation，再在 imaging problems 中比较 dictionary 或 measurement model。读实验时不要只看重建图，而要看 Bayes factor 如何改变模型选择。",
      relationReading: "它与 RI UQ I 共享 proximal MCMC 背景，与 Quantifying UQ / RI UQ II 共享 Bayesian inverse problem 语境，但问题层级更高：从 posterior uncertainty 进入 model evidence。",
      researchValue: "这篇适合发展模型选择选题，例如比较 TV prior、wavelet prior、deep prior 或不同 measurement models。它的价值在于把“哪个模型更可信”变成可计算问题。"
    }
  }
};

paperNotesV2.forEach((note) => {
  const enhancement = noteEnhancements[note.id];
  if (enhancement) Object.assign(note, enhancement);
});

const readingReasons = {
    1: "先读综述是为了拿到整组分割论文的地图：SaT methodology、T-ROF、SLaT、vascular 和 sphere 都会在这里出现，后面读原始论文时才知道每篇负责哪一块。",
    2: "第二步读 ROF-PCMS linkage，因为它回答最关键的合法性问题：为什么 Rudin-Osher-Fatemi (ROF) minimizer 阈值化之后能服务 piecewise constant Mumford-Shah (PCMS) / Chan-Vese 分割。",
    3: "T-ROF 放在理论论文后面读，可以把抽象的 partial minimizer 结论落到算法上：solve ROF once，再用 τ_i = 1/2(m_{i-1}+m_i) 更新多类阈值。",
    4: "有了 SaT/ROF 的两阶段思路后，再读分割恢复耦合模型，才能看清它的差别：这篇不是 simple thresholding，而是把恢复变量 g、分割变量 u_i 和区域常数 c_i 放进同一个 joint optimization。",
    5: "Framelet tubular 是医学管状结构线的短版入口，先读它能快速理解 below / inside / above 三分区域和只平滑 boundary candidate region 的思想。",
    6: "Tight-frame vessel 是长版和更完整版本，应放在短版后读，重点补 Algorithm 1、Λ 区域收缩、finite convergence 和 O(n) per iteration。",
    7: "SLaT 放在基础分割线之后读，因为它继承 SaT 的 smoothing + thresholding，但增加 lifting：RGB smoothing 后转 Lab，拼成 RGB+Lab 六维特征。",
    8: "Wavelet sphere 是几何域扩展，应在 framelet/tight-frame 与 SLaT 后读。这样可以把 Euclidean image 上的候选边界思想迁移到 spherical wavelet、directional wavelet 和 curvelet。",
    9: "Two-stage classification 是从 image segmentation 到 graph classification 的桥，读它前需要先理解 SaT 的 smoothing + projection/thresholding 框架。",
    10: "Efficient variational classification 是 2019 two-stage 的成熟期版本，放在后面读可以对比新增的唯一解、primal-dual convergence、independent K subproblems 和 benchmark point clouds。",
    11: "高维逆问题 UQ 是进入 RI/UQ 线的短入口，先用它熟悉 posterior、maximum a posteriori (MAP)、highest posterior density (HPD) 和 local credible intervals。",
    12: "RI UQ I 是完整采样版，应先于快速 MAP 版阅读，因为它给出 posterior samples、MYULA、Px-MALA 和 hypothesis testing 的完整基准。",
    13: "RI UQ II 放在 RI UQ I 后读，才能看清 MAP + probability concentration 为什么是 scalable 替代，以及它相对完整 MCMC 牺牲了哪些后验信息。",
    14: "Online RI imaging 解决数据流和存储问题，放在 MAP-UQ 后读可以把 convex optimization 与 SKA big-data setting 联系起来。",
    15: "Proximal nested sampling 最后读，因为它从估计图像和量化不确定性上升到 Bayesian model selection，需要先理解 proximal MCMC、evidence、Bayes factor 和 high-dimensional inverse problems。"
  };

const layerBlocks = [
    {
      title: "第一层：分割方法论层",
      papers: [1, 2, 3, 4, 7, 5, 6, 8],
      body: "这一层围绕 SaT、ROF、PCMS、framelet 和 spherical wavelet 展开。它的核心不是某个特定数据集，而是把非凸分割问题拆成更可控的恢复、平滑、阈值化、lifting 或候选边界收缩。"
    },
    {
      title: "第二层：逆问题与不确定性层",
      papers: [12, 13, 14, 11, 15],
      body: "这一层把变分和凸优化语言迁移到 radio interferometric imaging 和 Bayesian inverse problems。重点从单张重建图像转向 posterior、HPD credible regions、local credible intervals、online processing 和 Bayesian evidence。"
    },
    {
      title: "第三层：高维分类迁移层",
      papers: [9, 10],
      body: "这一层说明 SaT 的思想可以脱离像素网格，进入 graph-based high-dimensional classification。warm initialization 对应初始标签，graph Laplacian / graph TV 对应图上 smoothing，binary projection 对应 thresholding。"
    }
  ];

const weeklyPlan = [
    {
      week: "第 1 周",
      theme: "SaT / ROF / PCMS 基础",
      papers: [1, 2, 3, 4],
      goal: "建立 smoothing + thresholding 的理论地图，搞清 ROF minimizer、PCMS partial minimizer、T-ROF 阈值更新和 joint restoration-segmentation 的区别。",
      diagrams: "画 SaT 两阶段流程图、ROF-PCMS 关系图、T-ROF 阈值更新循环图。",
      pseudo: "写出 T-ROF 的阈值更新伪代码，并写出 E(u,c,g) alternating minimization 的三变量更新顺序。",
      summary: "写 300-500 字说明：为什么恢复模型可以成为分割模型的前处理，以及 joint optimization 与 two-stage SaT 的取舍。"
    },
    {
      week: "第 2 周",
      theme: "framelet / SLaT / sphere 应用",
      papers: [5, 6, 7, 8],
      goal: "理解 SaT/thresholding 思想如何进入管状结构、真实 MRA、退化彩色图像和球面图像。",
      diagrams: "画 below / inside / above 三分区域、Λ candidate set 收缩过程、RGB+Lab lifting 流程和 spherical image 分割 pipeline。",
      pseudo: "伪代码化 tight-frame vessel Algorithm 1，并列出 spherical wavelet segmentation 的边界候选区间更新步骤。",
      summary: "写 300-500 字比较：framelet/tight-frame 与 SLaT 都在处理不确定性，但一个处理边界候选像素，一个处理颜色特征空间。"
    },
    {
      week: "第 3 周",
      theme: "high-dimensional classification",
      papers: [9, 10],
      goal: "把 image segmentation 的 SaT 抽象成 graph classification：初始化标签函数，在图上做 convex variational smoothing，再投影到 hard labels。",
      diagrams: "画 graph construction、warm initialization、K 个独立子问题、binary projection 和 iterative refinement 的完整流程。",
      pseudo: "写出 graph Laplacian + graph TV 模型的 primal-dual 迭代骨架，并标出每个类别子问题如何独立求解。",
      summary: "写 300-500 字说明：为什么去掉 simplex constraint 能提升速度，以及这种近似对分类准确率和理论保证有什么影响。"
    },
    {
      week: "第 4 周",
      theme: "RI imaging / UQ / proximal sampling / nested sampling",
      papers: [11, 12, 13, 14, 15],
      goal: "理解从完整 posterior sampling 到 scalable MAP-UQ，再到 online processing 和 Bayesian model selection 的路径。",
      diagrams: "画 RI measurement y = Φx + n、RI UQ I vs RI UQ II 对照表、online block assimilation 流程和 nested sampling prior volume ξ 示意图。",
      pseudo: "写出 MAP-UQ 的 HPD approximation 计算步骤，以及 proximal nested sampling 中 likelihood contour 收缩和 proximal MCMC 采样步骤。",
      summary: "写 300-500 字说明：Bayesian UQ 与传统 reconstruction 的问题不同，它不只问图像是什么，还问哪些像素、结构和模型选择是不确定的。"
    }
  ];

const researchTopics = [
    {
      title: "SaT 在深度特征图上的可解释分割",
      sources: [1, 2, 3, 7],
      innovation: "把 CNN 或 foundation model 的 feature map 作为 smoothing 后的表征，再用 thresholding / clustering 输出可解释区域，比较与 end-to-end segmentation head 的差别。",
      difficulty: "深度特征不一定满足 ROF/PCMS 的数学假设，阈值化后的区域语义也可能不稳定。",
      experiment: "先用退化彩色图和少量医学图像，把 RGB+Lab SLaT 与深度特征 SaT 做可视化和 mIoU/边界质量对比。"
    },
    {
      title: "Graph SaT 用于医学点云或遥感点云分类",
      sources: [9, 10, 5, 6],
      innovation: "把 warm initialization + graph variational smoothing + binary projection 用到血管中心线点云、LiDAR 或遥感点云上，强调可解释图正则项。",
      difficulty: "图构建、邻域尺度、label sparsity 和类别不平衡会直接影响 graph TV 的效果。",
      experiment: "从公开 point cloud benchmark 开始，构造 k-NN graph，比较 SVM 初始化、随机初始化和少量标签初始化下的 accuracy 与 CPU time。"
    },
    {
      title: "tight-frame / wavelet 与 neural implicit representation 结合",
      sources: [5, 6, 8],
      innovation: "用 neural implicit representation 表示连续图像或球面信号，再把 wavelet/tight-frame 的稀疏正则用于边界或曲线结构分割。",
      difficulty: "implicit model 的连续参数化与 framelet coefficient shrinkage 如何稳定结合，需要重新设计优化流程。",
      experiment: "先在 synthetic tubular structures 和 spherical retina 上比较 explicit image grid、spherical wavelet 与 implicit representation。"
    },
    {
      title: "Spherical wavelet segmentation 与 spherical CNN 对比",
      sources: [8, 1],
      innovation: "以 Earth map、solar data、spherical retina 为对象，对比可解释 spherical wavelet/curvelet segmentation 与数据驱动 spherical CNN。",
      difficulty: "数据量、标注质量和球面采样方式会影响公平对比；wavelet 方法强在可解释，CNN 强在学习能力。",
      experiment: "复用球面小波论文中的数据类型，构造相同采样和评价指标，比较 boundary quality、噪声鲁棒性和推理时间。"
    },
    {
      title: "MAP-UQ 用于其他医学成像逆问题",
      sources: [11, 12, 13],
      innovation: "把 MAP + probability concentration 的 HPD approximation 从 RI imaging 迁移到 MRI、CT 或 PET reconstruction，输出 local credible intervals。",
      difficulty: "不同成像算子和噪声模型会改变 posterior geometry，HPD 近似的保守性需要重新验证。",
      experiment: "先用 MRI brain image 模拟 undersampling，比较 MAP-UQ 与小规模 MCMC 的 credible interval 覆盖差异。"
    },
    {
      title: "Proximal nested sampling 比较 TV prior、wavelet prior 与 deep prior",
      sources: [15, 11, 13],
      innovation: "用 Bayesian evidence 和 Bayes factor 比较不同 imaging priors，而不是只看重建图像质量。",
      difficulty: "deep prior 不一定是 log-concave non-smooth prior，可能破坏 proximal nested sampling 的理论便利性。",
      experiment: "先在小规模 Gaussian/imaging toy problem 上比较 TV prior、wavelet prior，再逐步加入 learned prior。"
    },
    {
      title: "SaT/T-ROF 多相理论缺口与 K>2 近似误差分析",
      sources: [2, 3, 1],
      innovation: "围绕 K>2 的附加条件、阈值间隔和 partial minimizer 关系建立更清楚的误差或稳定性分析。",
      difficulty: "多相边界、区域常数和阈值序列相互耦合，无法直接套用二相结论。",
      experiment: "先构造 synthetic noisy multiphase images，系统改变 K、灰度间隔和噪声水平，观察 T-ROF 与 PCMS 近似解之间的差异。"
    }
  ];

const finalSummary = [
    "这组论文的研究风格偏数学建模，而不是黑箱经验方法。早期分割论文持续追问一个问题：能不能把非凸、难解、对初始化敏感的分割问题，转成更稳定的 convex smoothing、Total Variation restoration、thresholding 或 candidate-region refinement。",
    "SaT / ROF / PCMS 这条线的关键价值在于可解释性。Linkage 论文把 ROF minimizer thresholding 与 PCMS / Chan-Vese partial minimizer 接起来，T-ROF 给出可执行的阈值更新算法，Segmentation Restoration 则展示 restoration variable g 如何直接进入 joint optimization。",
    "framelet、tight-frame、SLaT 和 spherical wavelet 论文说明这套思想不是只适合灰度平面图像。它可以进入 MRA 血管、管状结构、退化彩色图像、Earth map、solar data 和 spherical retina。方法变化很大，但共同模式仍是先稳定表达，再把不确定区域或特征空间转成分割。",
    "高维分类线把像素分割抽象成 graph classification：warm initialization 提供初始标签函数，graph Laplacian 和 graph TV 做平滑，binary projection 给出 hard labels。无线电干涉与 Bayesian UQ 线则把变分/凸优化语言推进到 inverse problems、MAP、HPD、proximal MCMC、online processing 和 evidence estimation。",
    "如果要从这 15 篇里提炼自己的研究入口，最值得抓住的不是某个单独算法，而是这种研究范式：从实际成像或分类困难出发，建立可解释模型，寻找凸性、唯一解、收敛、partial minimizer、finite convergence 或 scalable UQ 这样的硬保证，再回到具体数据对象验证这些保证到底解决了什么。"
  ];

const reproScoring = {
  difficultyDimensions: [
    "algorithmComplexity：目标模型、优化算法、采样或变换实现是否复杂。",
    "dependencyComplexity：是否依赖 S2LET / SSHT / SO3、RI operators、MCMC diagnostics 等专门库。",
    "dataComplexity：是否需要真实 MRA、radio interferometry visibilities、球面数据或大规模标注数据。",
    "computeComplexity：是否需要长时间 proximal MCMC、nested sampling 或大规模图优化。",
    "verificationComplexity：是否容易用 Dice、accuracy、SNR、PSNR、HPD interval、evidence error 等指标验证。"
  ],
  effectDimensions: [
    "visualClarity：toy 图是否能直观看到 smoothing、thresholding、uncertainty 或 storage reduction。",
    "metricClarity：是否有明确数值指标，并且指标能反映论文核心机制。",
    "paperFaithfulness：minimal experiment 与论文模型或算法路线的贴近程度。",
    "teachingValue：是否适合放进 dashboard，帮助读者理解论文而不是误导为完整复现。"
  ]
};

const reproRecommendedBatches = [
  {
    title: "第一批：最适合快速出效果",
    priorities: [3, 7, 1, 2],
    reason: "T-ROF、SLaT、SaT 和 ROF-threshold demo 都能用 synthetic image 展示 smoothing + thresholding 的机制；其中 SLaT 当前 toy 指标提升很小，需要后续重新设计颜色案例。"
  },
  {
    title: "第二批：中等难度但很有价值",
    priorities: [4, 5, 6, 9, 10],
    reason: "分割恢复、管状结构和图分类能展示论文方法如何迁移到真实应用形态，但这里仍是 toy/partial，不等同于 MRA 或 benchmark paper-level reproduction。"
  },
  {
    title: "第三批：逆问题与 UQ toy",
    priorities: [11, 13, 14],
    reason: "MAP-UQ 与 online RI 可以在小型 Fourier inverse problem 上展示重建质量、credible interval 和峰值存储差异。"
  },
  {
    title: "第四批：只做 toy 或长期任务",
    priorities: [12, 8, 15],
    reason: "RI proximal MCMC、spherical wavelet stack 和 proximal nested sampling 依赖重、验证难，短期只适合教学 toy。"
  }
];

const reproDetails = {
  1: {
    reproductionLevel: "toy-to-partial",
    difficultyScore: 3,
    difficultyLabel: "中",
    effectScore: 4,
    effectLabel: "很明显",
    fullReproductionFeasibility: "中等可行。论文是 overview，本身不要求单一实验；paper-level 复现应覆盖 T-ROF、SLaT、vascular、sphere 等多个分支。",
    minimalExperiment: "noisy/blurred synthetic multiphase image，先做 TV/ROF-like smoothing，再用 K-means thresholding 得到分割。",
    expectedOutcome: "smoothing 后再 thresholding 比直接 K-means 更稳，toy 结果显示 synthetic multiphase accuracy 从 0.6590 提升到 0.9799。",
    metrics: ["direct_accuracy", "sat_accuracy", "accuracy_gain", "runtime_seconds"],
    dependencies: ["numpy", "scipy", "matplotlib"],
    dataRequirement: "synthetic 4-phase degraded image；不需要下载真实数据。",
    computeRequirement: "CPU，约 1 秒内。",
    implementationRisk: "TV/ROF-like smoothing 是教学近似，不覆盖 overview 中所有 SaT 应用分支。",
    verificationPlan: "比较 direct K-means 与 SaT toy segmentation 的 pixel accuracy，并检查输出图是否展示平滑后类别更稳定。",
    resultStatus: "completed",
    experimentId: "sat_rof_trof",
    runtimeSeconds: 0.4396,
    runMetrics: { direct_accuracy: 0.659, sat_accuracy: 0.9799, accuracy_gain: 0.321 },
    resultFiles: ["assets/repro/sat_demo.png"],
    fidelityWarning: "Uses Gaussian proxy smoothing, not an exact convex ROF/TV minimizer.",
    notes: "Gaussian smoothing is used as a lightweight proxy for convex ROF/TV smoothing on a synthetic toy image."
  },
  2: {
    reproductionLevel: "toy-to-partial",
    difficultyScore: 3,
    difficultyLabel: "中",
    effectScore: 4,
    effectLabel: "很明显",
    fullReproductionFeasibility: "理论 full reproduction 需要重读并复核 Theorem 3.6 的证明；代码只能演示 ROF minimizer thresholding 的现象，不能替代定理证明。",
    minimalExperiment: "synthetic two-phase image，求 ROF/TV-like denoising result u*，按 (m0+m1)/2 阈值化，并与 ground truth 比较。",
    expectedOutcome: "展示 ROF minimizer thresholding 的分割效果，并记录 PCMS-like energy；toy Dice 达到 0.9960。",
    metrics: ["direct_dice", "rof_threshold_dice", "pcms_like_energy", "runtime_seconds"],
    dependencies: ["numpy", "scipy", "matplotlib"],
    dataRequirement: "synthetic two-phase image。",
    computeRequirement: "CPU，约 1 秒内。",
    implementationRisk: "实验只能说明现象，不能声称复现 PCMS partial minimizer 定理。",
    verificationPlan: "对比 noisy direct threshold 与 ROF-threshold Dice，并记录 perimeter + data fitting 的 PCMS-like energy。",
    resultStatus: "completed",
    experimentId: "sat_rof_trof",
    runtimeSeconds: 0.4396,
    runMetrics: { direct_dice: 0.8989, rof_threshold_dice: 0.996, pcms_like_energy: 204.0 },
    resultFiles: ["assets/repro/sat_demo.png"],
    fidelityWarning: "Uses proxy smoothing; does not solve the exact ROF model.",
    notes: "This synthetic toy demonstrates thresholding after proxy smoothing, but does not solve the exact ROF model or prove Theorem 3.6."
  },
  3: {
    reproductionLevel: "toy-to-partial",
    difficultyScore: 3,
    difficultyLabel: "中",
    effectScore: 5,
    effectLabel: "很明显",
    fullReproductionFeasibility: "中等可行。接近论文算法需要更严格的 ROF solver 与多组 close-gray-value 实验；当前实现保留 solve once + iterative threshold update 的核心。",
    minimalExperiment: "close-gray-value multiphase synthetic image，先解一次 ROF/TV-like smoothing，再迭代更新 tau_i = 1/2(m_{i-1}+m_i)。",
    expectedOutcome: "比 direct K-means/no smoothing 更能分出接近灰度类别；toy accuracy 从 0.6590 提升到 0.9799，阈值迭代 3 次。",
    metrics: ["raw_kmeans_accuracy", "trof_accuracy", "threshold_iterations", "runtime_seconds"],
    dependencies: ["numpy", "scipy", "matplotlib"],
    dataRequirement: "synthetic close-gray-value 4-phase image。",
    computeRequirement: "CPU，约 1 秒内。",
    implementationRisk: "使用近似 TV smoothing，不等同于论文中的完整 ROF 数值实现和收敛条件验证。",
    verificationPlan: "记录每轮阈值变化、最终 pixel accuracy，并保存 threshold history 图。",
    resultStatus: "completed",
    experimentId: "sat_rof_trof",
    runtimeSeconds: 0.4396,
    runMetrics: { raw_kmeans_accuracy: 0.659, trof_accuracy: 0.9799, threshold_iterations: 3 },
    resultFiles: ["assets/repro/sat_demo.png", "assets/repro/trof_thresholds.png"],
    fidelityWarning: "Uses proxy smoothing before threshold updates; strict T-ROF should solve ROF once.",
    notes: "This synthetic toy implements the threshold update tau_i = 1/2(m_{i-1}+m_i) after proxy smoothing; strict T-ROF should solve ROF once."
  },
  4: {
    reproductionLevel: "toy",
    difficultyScore: 4,
    difficultyLabel: "高",
    effectScore: 4,
    effectLabel: "很明显",
    fullReproductionFeasibility: "偏难。full reproduction 需要实现论文中的 Φ(f,Ag)、segmentation variables u_i、restoration variable g 和 convergence setting。",
    minimalExperiment: "blurred/noisy/missing synthetic image，做 alternating minimization toy：更新 g、class means c_i 与 labels u_i。",
    expectedOutcome: "joint restoration-segmentation 比只在 degraded image 上直接分割更稳；toy accuracy 从 0.5332 提升到 0.9604。",
    metrics: ["direct_accuracy", "joint_toy_accuracy", "accuracy_gain", "alternating_iterations"],
    dependencies: ["numpy", "scipy", "matplotlib"],
    dataRequirement: "synthetic blurred/noisy/missing image。",
    computeRequirement: "CPU，约 1 秒内。",
    implementationRisk: "toy AM 不覆盖论文的全部 fidelity term、vector-valued image 和收敛证明。",
    verificationPlan: "比较 degraded direct segmentation 与 AM toy segmentation 的 accuracy，并保存恢复图、分割图和 ground truth。",
    resultStatus: "completed",
    experimentId: "segmentation_restoration",
    runtimeSeconds: 0.1208,
    runMetrics: { direct_accuracy: 0.5332, joint_toy_accuracy: 0.9604, accuracy_gain: 0.4272, alternating_iterations: 8 },
    resultFiles: ["assets/repro/segmentation_restoration_toy.png"],
    notes: "Toy alternating restoration-segmentation over g, class means and labels; not full variational AM proof reproduction."
  },
  5: {
    reproductionLevel: "toy",
    difficultyScore: 4,
    difficultyLabel: "高",
    effectScore: 4,
    effectLabel: "很明显",
    fullReproductionFeasibility: "偏难。full reproduction 需要 proper framelet transform、2D/3D tubular experiments 和论文参数设置。",
    minimalExperiment: "synthetic tube/vessel mask with noise，构造 boundary interval [alpha,beta]，只在 uncertain region 做 wavelet/TV-like smoothing。",
    expectedOutcome: "uncertainty region shrinks and binary tube mask emerges；toy Dice 0.9981，Lambda 从 651 收缩到 2。",
    metrics: ["dice", "iou", "lambda_initial", "lambda_final", "iterations"],
    dependencies: ["numpy", "scipy", "scikit-image", "matplotlib"],
    dataRequirement: "synthetic 2D tube network；full reproduction 需要论文使用的 2D/3D tubular structures。",
    computeRequirement: "CPU，约 1 秒内。",
    implementationRisk: "这里用 Gaussian smoothing 近似 framelet smoothing，不是严格 framelet implementation。",
    verificationPlan: "记录 Lambda size per iteration、Dice/IoU，并检查候选区域是否随迭代收缩。",
    resultStatus: "completed",
    experimentId: "tubular_tight_frame",
    runtimeSeconds: 0.0784,
    runMetrics: { dice: 0.9981, iou: 0.9962, lambda_initial: 651, lambda_final: 2, iterations: 12 },
    resultFiles: ["assets/repro/tubular_lambda_shrinkage.png"],
    notes: "Approximate toy reproduction: Gaussian smoothing stands in for framelet smoothing inside uncertain boundary interval. Dice is measured on a simple synthetic 2D vessel toy; it does not represent real 2D/3D MRA paper-level performance."
  },
  6: {
    reproductionLevel: "toy",
    difficultyScore: 4,
    difficultyLabel: "高",
    effectScore: 4,
    effectLabel: "很明显",
    fullReproductionFeasibility: "偏难到高。full reproduction requires real 2D/3D MRA data and proper tight-frame/DCWT implementation。",
    minimalExperiment: "synthetic 2D vessel network，构造 Lambda boundary set shrinkage，记录 iterations、Dice 和 IoU。",
    expectedOutcome: "finite shrinkage of Lambda, convergence to binary mask；toy Dice 0.9981，12 次迭代后 Lambda 只剩 2 个像素。",
    metrics: ["dice", "iou", "lambda_initial", "lambda_final", "iterations"],
    dependencies: ["numpy", "scipy", "scikit-image", "matplotlib"],
    dataRequirement: "toy 用 synthetic vessel network；full reproduction 需要真实 2D/3D MRA 图像。",
    computeRequirement: "toy 为 CPU 秒级；3D MRA 与 tight-frame transform 会显著增加内存和时间。",
    implementationRisk: "缺少论文级 tight-frame/DCWT，当前只复现 Lambda 收缩逻辑和有限收敛现象。",
    verificationPlan: "检查 Lambda size 单调收缩、最终二值图与 ground truth 的 Dice/IoU。",
    resultStatus: "completed",
    experimentId: "tubular_tight_frame",
    runtimeSeconds: 0.0784,
    runMetrics: { dice: 0.9981, iou: 0.9962, lambda_initial: 651, lambda_final: 2, iterations: 12 },
    resultFiles: ["assets/repro/tubular_lambda_shrinkage.png"],
    notes: "Approximate toy reproduction: Lambda boundary set shrinkage and finite convergence pattern on synthetic 2D vessel network. Dice is measured on a simple synthetic 2D vessel toy; it does not represent real 2D/3D MRA paper-level performance."
  },
  7: {
    reproductionLevel: "partial",
    difficultyScore: 3,
    difficultyLabel: "中",
    effectScore: 3,
    effectLabel: "明显",
    fullReproductionFeasibility: "中等可行。paper-level 需要退化彩色图像组、准确 Lab conversion 和论文对比基线；toy 可展示 RGB+Lab lifting。",
    minimalExperiment: "degraded RGB synthetic image，做 channel smoothing，构造 RGB + Lab-like luminance/chroma 六维特征，再 K-means。",
    expectedOutcome: "RGB+Lab segmentation should be more robust than RGB-only on degraded color image；本 toy gain 为 0.0053，视觉对比比数值提升更明显。",
    metrics: ["rgb_only_accuracy", "rgb_lab_accuracy", "accuracy_gain", "runtime_seconds"],
    dependencies: ["numpy", "scipy", "matplotlib"],
    dataRequirement: "synthetic degraded color image；不下载真实数据。",
    computeRequirement: "CPU，约 1 秒内。",
    implementationRisk: "为避免额外依赖，Lab 使用 Lab-like luminance/chroma toy transform，不是严格 color science 复现。",
    verificationPlan: "比较 RGB-only 与 RGB+Lab-like 的 pixel accuracy，并检查输出图中颜色区域边界是否更稳定。",
    resultStatus: "completed",
    experimentId: "slat_color",
    runtimeSeconds: 0.0881,
    runMetrics: { rgb_only_accuracy: 0.7092, rgb_lab_accuracy: 0.7145, accuracy_gain: 0.0053 },
    resultFiles: ["assets/repro/slat_rgb_vs_rgblab.png"],
    notes: "Toy SLaT: channel smoothing, RGB plus Lab-like luminance/chroma lifting, K-means on synthetic degraded color image. Current toy shows only a small metric gain; a better synthetic color case is needed to highlight Lab lifting."
  },
  8: {
    reproductionLevel: "toy",
    difficultyScore: 5,
    difficultyLabel: "极高",
    effectScore: 3,
    effectLabel: "明显",
    fullReproductionFeasibility: "高依赖长期任务。full reproduction requires S2LET/SSHT/SO3 spherical wavelet stack, spherical datasets and exact sampling conventions。",
    minimalExperiment: "synthetic equirectangular sphere-like image with bands/curves，approximate spherical gradient and boundary interval shrinkage。",
    expectedOutcome: "toy sphere segmentation works, but full reproduction requires specialized spherical wavelet libraries；toy Dice 0.8418。",
    metrics: ["dice", "gradient_threshold_quantile", "runtime_seconds"],
    dependencies: ["numpy", "scipy", "matplotlib"],
    dataRequirement: "toy equirectangular image；paper-level 需要 Earth map、light probe、solar data、spherical retina 等。",
    computeRequirement: "toy 为 CPU 秒级；paper-level spherical wavelet transforms 依赖专门库和更多调参。",
    implementationRisk: "equirectangular approximation 无法替代 axisymmetric/directional wavelets 或 curvelets。",
    verificationPlan: "记录 Dice 和 spherical-gradient threshold，图中标注这是 approximation toy。",
    resultStatus: "completed",
    experimentId: "sphere_wavelet_toy",
    runtimeSeconds: 0.0714,
    runMetrics: { dice: 0.8418, gradient_threshold_quantile: 0.93 },
    resultFiles: ["assets/repro/sphere_wavelet_toy.png"],
    notes: "Approximate sphere toy: equirectangular smoothing plus spherical-gradient correction; no S2LET/SSHT/SO3 stack."
  },
  9: {
    reproductionLevel: "partial",
    difficultyScore: 4,
    difficultyLabel: "高",
    effectScore: 4,
    effectLabel: "很明显",
    fullReproductionFeasibility: "偏难。paper-level 需要 benchmark high-dimensional data、graph TV solver 和系统对比；toy 能展示 warm init + graph smoothing + projection。",
    minimalExperiment: "synthetic moons/blobs，warm initialization，kNN graph，graph Laplacian smoothing，argmax projection。",
    expectedOutcome: "smoothing + projection improves warm initialization；toy accuracy 从 0.8000 提升到 0.8139。",
    metrics: ["initial_accuracy", "smoothed_accuracy", "accuracy_gain", "iterations"],
    dependencies: ["numpy", "scipy", "matplotlib"],
    dataRequirement: "synthetic 2D classification data；不下载 benchmark。",
    computeRequirement: "CPU，约 1 秒内。",
    implementationRisk: "toy 使用 Laplacian smoothing，不是完整 graph TV convex model。",
    verificationPlan: "比较 warm init 与 smoothing 后 accuracy，并保存 before/after decision colors。",
    resultStatus: "completed",
    experimentId: "graph_classification",
    runtimeSeconds: 0.0898,
    runMetrics: { initial_accuracy: 0.8, smoothed_accuracy: 0.8139, accuracy_gain: 0.0139, iterations: 18 },
    resultFiles: ["assets/repro/graph_classification_before_after.png"],
    notes: "Toy graph classification: centroid warm initialization, kNN graph smoothing, argmax projection."
  },
  10: {
    reproductionLevel: "partial",
    difficultyScore: 4,
    difficultyLabel: "高",
    effectScore: 4,
    effectLabel: "很明显",
    fullReproductionFeasibility: "偏难。full reproduction 需要论文 benchmark、primal-dual solver、graph TV 和多类大规模对比。",
    minimalExperiment: "reuse graph classification experiment，增加 repeated smoothing iterations，展示 K label functions 独立更新再 projection。",
    expectedOutcome: "repeated convex smoothing improves or stabilizes accuracy；toy gain 0.0139。",
    metrics: ["initial_accuracy", "smoothed_accuracy", "accuracy_gain", "iterations"],
    dependencies: ["numpy", "scipy", "matplotlib"],
    dataRequirement: "synthetic graph classification data；paper-level 需要 high-dimensional data and point clouds benchmarks。",
    computeRequirement: "toy 为 CPU 秒级；benchmark graph construction 和 primal-dual iteration 更重。",
    implementationRisk: "当前没有实现完整 graph TV primal-dual convergence，只演示 variational smoothing 机制。",
    verificationPlan: "记录 repeated iteration 的最终 accuracy 和运行时间，页面上明确 partial reproduction。",
    resultStatus: "completed",
    experimentId: "graph_classification",
    runtimeSeconds: 0.0898,
    runMetrics: { initial_accuracy: 0.8, smoothed_accuracy: 0.8139, accuracy_gain: 0.0139, iterations: 18 },
    resultFiles: ["assets/repro/graph_classification_before_after.png"],
    notes: "Toy repeated graph smoothing: demonstrates independent label-function update idea without full graph TV primal-dual solver."
  },
  11: {
    reproductionLevel: "toy",
    difficultyScore: 4,
    difficultyLabel: "高",
    effectScore: 4,
    effectLabel: "很明显",
    fullReproductionFeasibility: "偏难。full reproduction 需要 MRI/M31 设定、exact MAP objective、dictionary/prior 和 HPD calibration。",
    minimalExperiment: "small Fourier undersampling inverse problem，求 MAP estimate，构造 approximate HPD threshold gamma'_alpha 和 local credible interval length map。",
    expectedOutcome: "produce MAP reconstruction and uncertainty interval length map；toy MAP PSNR 18.7123，mean interval length 0.1739。",
    metrics: ["map_psnr", "map_snr", "gamma_alpha_toy", "mean_interval_length"],
    dependencies: ["numpy", "scipy", "scikit-image", "matplotlib"],
    dataRequirement: "synthetic small image and Fourier mask；不下载 M31/MRI 数据。",
    computeRequirement: "CPU，约 1 秒内。",
    implementationRisk: "HPD formula 是教学近似，不校准真实 posterior coverage。",
    verificationPlan: "记录 MAP PSNR/SNR、gamma alpha toy 值和 interval length map。",
    resultStatus: "completed",
    experimentId: "map_uq_toy",
    runtimeSeconds: 0.075,
    runMetrics: { map_psnr: 18.7123, map_snr: 9.6004, map_runtime_seconds: 0.0016, mcmc_runtime_seconds: 0.004, gamma_alpha_toy: 939.9229, mean_interval_length: 0.1739 },
    resultFiles: ["assets/repro/map_uq_reconstruction_uncertainty.png"],
    notes: "Toy MAP-UQ: small Fourier undersampling inverse problem with approximate HPD and local interval map. Toy runtime comparison is not comparable to the paper's large-scale 10^5 speedup claim."
  },
  12: {
    reproductionLevel: "toy",
    difficultyScore: 5,
    difficultyLabel: "极高",
    effectScore: 3,
    effectLabel: "明显",
    fullReproductionFeasibility: "极难。full reproduction requires RI operators, large sampling, sparse analysis/synthesis priors, MYULA/Px-MALA tuning and careful MCMC diagnostics。",
    minimalExperiment: "very small 32x32 Fourier inverse problem，MYULA-like proximal sampling with simple prior，posterior samples and credible intervals。",
    expectedOutcome: "credible intervals computable but slower than MAP in principle；本 toy 只验证 interval map 可生成，不做 RI 级 diagnostics。",
    metrics: ["map_psnr", "mcmc_runtime_seconds", "mean_interval_length", "gamma_alpha_toy"],
    dependencies: ["numpy", "scipy", "scikit-image", "matplotlib"],
    dataRequirement: "toy Fourier measurements；full reproduction 需要 RI visibility data 和 measurement operator。",
    computeRequirement: "toy 为 CPU 秒级；full MCMC 需要大量样本和诊断，普通笔记本不适合完整跑。",
    implementationRisk: "MYULA-like sampler 是极简教学版，不等于论文级 proximal MCMC。",
    verificationPlan: "记录 toy sampler runtime、interval map，并在页面明确 no RI operator / no MCMC diagnostics。",
    resultStatus: "completed",
    experimentId: "map_uq_toy",
    runtimeSeconds: 0.075,
    runMetrics: { map_psnr: 18.7123, map_snr: 9.6004, map_runtime_seconds: 0.0016, mcmc_runtime_seconds: 0.004, gamma_alpha_toy: 939.9229, mean_interval_length: 0.1739 },
    resultFiles: ["assets/repro/map_uq_reconstruction_uncertainty.png"],
    notes: "Toy proximal-MCMC-style sampling on a 32x32 Fourier inverse problem; no RI operator or MCMC diagnostics. Toy runtime comparison is not comparable to the paper's large-scale 10^5 speedup claim."
  },
  13: {
    reproductionLevel: "toy",
    difficultyScore: 4,
    difficultyLabel: "高",
    effectScore: 5,
    effectLabel: "很明显",
    fullReproductionFeasibility: "偏难。full reproduction 需要 RI measurement operator、大图像 MAP solver 和与 MCMC 的系统对比；toy 可展示 MAP-UQ 快速路线。",
    minimalExperiment: "same Fourier inverse problem as RI UQ I toy，solve MAP by forward-backward/proximal gradient，build HPD approximation and local intervals。",
    expectedOutcome: "much faster than toy sampling while giving comparable uncertainty pattern；toy MAP runtime 0.0016 秒。",
    metrics: ["map_runtime_seconds", "mcmc_runtime_seconds", "map_psnr", "mean_interval_length"],
    dependencies: ["numpy", "scipy", "scikit-image", "matplotlib"],
    dataRequirement: "toy Fourier inverse problem；paper-level 需要 M31/Cygnus A/W28/3C288 等 RI images。",
    computeRequirement: "toy 为 CPU 秒级；paper-level 需高效 convex optimization。",
    implementationRisk: "没有复现 10^5 倍速度比较，只在 toy 中记录 MAP 与小采样的时间差。",
    verificationPlan: "保存 MAP reconstruction + uncertainty map，并记录 MAP/MCMC toy runtimes。",
    resultStatus: "completed",
    experimentId: "map_uq_toy",
    runtimeSeconds: 0.075,
    runMetrics: { map_psnr: 18.7123, map_snr: 9.6004, map_runtime_seconds: 0.0016, mcmc_runtime_seconds: 0.004, gamma_alpha_toy: 939.9229, mean_interval_length: 0.1739 },
    resultFiles: ["assets/repro/map_uq_reconstruction_uncertainty.png"],
    notes: "Toy MAP-UQ is faster than the toy sampler and gives a similar uncertainty pattern; not a paper-level SKA experiment. Toy runtime comparison is not comparable to the paper's large-scale 10^5 speedup claim."
  },
  14: {
    reproductionLevel: "toy",
    difficultyScore: 4,
    difficultyLabel: "高",
    effectScore: 4,
    effectLabel: "很明显",
    fullReproductionFeasibility: "偏难。full reproduction 需要真实 RI visibilities、online forward-backward implementation 和内存/时间基准。",
    minimalExperiment: "split Fourier measurements into blocks，online forward-backward updates，discard blocks，compare offline reconstruction and peak storage。",
    expectedOutcome: "similar reconstruction quality to offline toy baseline with lower peak stored measurements；toy offline/online PSNR 均 12.3359，peak storage 585 vs 98。",
    metrics: ["offline_psnr", "online_psnr", "peak_stored_measurements_offline", "peak_stored_measurements_online"],
    dependencies: ["numpy", "scipy", "scikit-image", "matplotlib"],
    dataRequirement: "toy Fourier mask blocks；paper-level 需要 RI visibility blocks。",
    computeRequirement: "toy 为 CPU 秒级；SKA big-data setting 需要流式算子和内存监控。",
    implementationRisk: "online discard 是概念级 toy，不覆盖真实 RI operator cache 和 distributed setting。",
    verificationPlan: "比较 offline/online PSNR/SNR 与 peak stored measurements，并输出 storage-quality 图。",
    resultStatus: "completed",
    experimentId: "online_ri_toy",
    runtimeSeconds: 0.0665,
    runMetrics: { offline_psnr: 12.3359, online_psnr: 12.3359, offline_snr: 2.6069, online_snr: 2.6069, peak_stored_measurements_offline: 585, peak_stored_measurements_online: 98 },
    resultFiles: ["assets/repro/online_ri_storage_quality.png"],
    notes: "Toy online RI: split Fourier measurements into blocks, assimilate each block, then discard it conceptually."
  },
  15: {
    reproductionLevel: "toy",
    difficultyScore: 5,
    difficultyLabel: "极高",
    effectScore: 3,
    effectLabel: "明显",
    fullReproductionFeasibility: "极难。full reproduction requires constrained proximal MCMC, evidence validation and high-dimensional imaging benchmarks up to O(10^6) dimension。",
    minimalExperiment: "low-dimensional Gaussian evidence toy with known/reference evidence，run simple nested sampling estimator。",
    expectedOutcome: "evidence estimate works on toy, but high-dimensional proximal nested sampling remains high-complexity；toy absolute log-evidence error 2.4676。",
    metrics: ["estimated_log_evidence", "reference_log_evidence", "absolute_log_error", "iterations"],
    dependencies: ["numpy", "matplotlib"],
    dataRequirement: "2D Gaussian likelihood under uniform prior；paper-level 需要 high-dimensional imaging model comparison。",
    computeRequirement: "toy 为 CPU 秒级；paper-level nested sampling 需要大量 constrained samples。",
    implementationRisk: "没有实现 proximal constrained sampler，只复现 nested sampling evidence trace 的教学核心。",
    verificationPlan: "比较 estimated/reference log evidence，记录 absolute error，并保存 evidence trace。",
    resultStatus: "completed",
    experimentId: "nested_sampling_toy",
    runtimeSeconds: 0.0591,
    runMetrics: { estimated_log_evidence: -5.5996, reference_log_evidence: -3.1319, absolute_log_error: 2.4676, live_points: 80, iterations: 180 },
    resultFiles: ["assets/repro/nested_sampling_evidence_trace.png"],
    resultQuality: "rough illustrative",
    warning: "large evidence error; toy only",
    notes: "Toy nested sampling on a 2D Gaussian likelihood under a uniform prior; not proximal constrained MCMC. Completed with large error; use as nested sampling mechanism demo only."
  }
};

const reproAssessments = paperNotesV2.map((note) => {
  const paper = papers.find((item) => item.priority === note.priority);
  const detail = reproDetails[note.priority];
  const reproductionTruthLevel = detail.reproductionTruthLevel
    || (detail.reproductionLevel === "paper-level" ? "paper-level-completed"
      : detail.reproductionLevel.includes("partial") ? "partial-completed"
        : detail.reproductionLevel === "assessment-only" ? "assessment-only" : "toy-completed");
  return {
    id: note.id,
    priority: note.priority,
    titleCn: note.titleCn,
    titleEn: note.titleEn,
    pdf: note.pdf || paper?.file || "",
    theme: note.theme,
    reproductionTruthLevel,
    ...detail
  };
});

const siteMeta = {
  scope: "15-paper-scope",
  lastUpdated: "2026-06-04"
};

window.ZX_READING_DATA = {
  basePath,
  tracks,
  thesis,
  papers,
  chronology,
  readingStages,
  noteThemes,
  noteMainlines,
  readingStandard,
  paperNotesV2,
  readingReasons,
  layerBlocks,
  weeklyPlan,
  researchTopics,
  finalSummary,
  reproScoring,
  reproRecommendedBatches,
  reproAssessments,
  siteMeta
};
