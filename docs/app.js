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

const readingNotes = [
  {
    priority: 1,
    problem: "作为入口论文，它回答 SaT 方法到底覆盖哪些分支，以及这些分支如何从灰度分割扩展到彩色、血管、球面和高光谱图像。",
    method: "主线是 smoothing + thresholding：先用更稳定的恢复/平滑模型处理退化图像，再用阈值化、lifting 或聚类得到分割。",
    readFor: "不要当作单篇原始方法论文读，重点画方法树：T-ROF、SLaT、framelet、sphere 这些节点分别解决什么问题。",
    relation: "先读它建立地图，再回到 Linkage 与 Iterated ROF 补理论，之后读 SLaT、Framelet 和 Wavelet Sphere 看扩展应用。"
  },
  {
    priority: 2,
    problem: "核心问题是 ROF 图像恢复模型和 PCMS/Chan-Vese 分割模型之间是否有可证明的联系。",
    method: "论文证明在二相分割中，对 ROF minimizer 做合适阈值化，可以得到 PCMS/Chan-Vese 类模型的部分极小解。",
    readFor: "重点盯住 theorem 的假设、阈值范围、partial minimizer 的含义，以及为什么这能支撑 SaT/T-ROF 的合理性。",
    relation: "这是整组 SaT/ROF 论文的理论支柱；Iterated ROF 是算法前身，SaT Overview 是后来的方法论总结。"
  },
  {
    priority: 3,
    problem: "传统多类分割直接求 Chan-Vese/Mumford-Shah 类模型困难，论文尝试用 ROF 去噪加阈值迭代来绕开非凸求解。",
    method: "每轮解 ROF 类恢复问题，再根据当前灰度/类别结构更新阈值，实现 multiclass segmentation。",
    readFor: "重点看算法流程、阈值更新、多类扩展方式，以及它和 Chan-Vese/Mumford-Shah 能量的关系。",
    relation: "它是 Linkage 论文的历史源头；先读算法，再读 2019 理论长文，会更容易理解证明为什么重要。"
  },
  {
    priority: 4,
    problem: "噪声、模糊、缺失像素和向量值图像会让传统分割模型不稳定，论文把恢复信息直接耦合进分割模型。",
    method: "在 PCMS 分割框架里引入 restoration fidelity term，让分割和恢复在同一个变分模型中相互约束。",
    readFor: "先看模型各项分别对应什么退化情形，再看它和两阶段 SaT 路线的差异：联合优化 vs 先恢复后阈值。",
    relation: "它连接 restoration 和 segmentation；和 SaT 一起看，能形成“联合模型”和“两阶段模型”的对照。"
  },
  {
    priority: 5,
    problem: "血管、道路等管状结构细长、弱边缘、分叉多，常规边缘或区域模型容易断裂。",
    method: "先定位可能边界灰度区间并三分图像，再用 framelet/tight-frame 迭代收缩和平滑边界候选区域。",
    readFor: "把它当短版算法读：输入如何预处理、候选边界怎么定义、framelet 迭代如何逐步收敛到二值管状结构。",
    relation: "这是管状结构分割的会议短版；读完后接 Vessel Segmentation 长版看完整 tight-frame 表述和实验。"
  },
  {
    priority: 6,
    problem: "医学 MRA 血管分割需要保留细血管、分叉和 3D 结构，同时抵抗噪声和弱边界。",
    method: "用方向选择性 tight-frame / framelet 表示做迭代分割，配合边界区域收缩和 2D/3D 实验验证。",
    readFor: "重点看 tight-frame 的角色、迭代步骤、收敛性说明，以及 2D/3D MRA 上哪些结果比短版更完整。",
    relation: "它是 Framelet-Based Tubular 的扩展版；两篇合读可以看到从会议算法到完整医学分割论文的扩展。"
  },
  {
    priority: 7,
    problem: "彩色退化图像分割不能只看 RGB，通道相关性和感知颜色差异都会影响分割质量。",
    method: "SLaT 增加 Lifting：RGB 平滑后转换到 Lab，并把 RGB + Lab 特征合起来做聚类/阈值化。",
    readFor: "重点理解 Lifting 为什么存在：Lab 不是装饰特征，而是补充 RGB 的感知颜色表达。",
    relation: "它是 SaT 从灰度/多类分割走向彩色图像的关键扩展；读完再看 Overview 中对 SLaT 的归纳。"
  },
  {
    priority: 8,
    problem: "球面图像不在平面欧氏网格上，平面 TV、wavelet 和采样方式不能直接搬用。",
    method: "把 wavelet/curvelet 分割思想推广到球面，处理球面梯度、球面采样和非欧氏几何上的稀疏表示。",
    readFor: "重点看几何部分：球面上的导数、采样和变换如何定义，而不是只看实验图像。",
    relation: "它和 framelet 管状结构同属特殊几何/特殊表示分割，说明 SaT/thresholding 思想可以离开平面图像。"
  },
  {
    priority: 9,
    problem: "高维数据和点云没有规则图像网格，但仍然可以被看成图上的分类/分割问题。",
    method: "先用 SVM 或随机标签做 warm initialization，再在 k-NN 图上解凸变分平滑模型，最后投影成硬分类。",
    readFor: "重点看 SaT 思想如何从 pixel segmentation 迁移到 graph classification：初始化、图构造、平滑、投影。",
    relation: "它是 2024 Efficient Variational Classification 的前身；两篇连读能看到方法从两阶段思路走向成熟期刊版。"
  },
  {
    priority: 10,
    problem: "多类半监督高维分类需要可扩展、可并行、能处理点云和无结构数据的变分方法。",
    method: "用无 simplex 约束的凸模型，每个类别标签函数独立求解，结合 graph Laplacian、graph TV、primal-dual 和 argmax 投影。",
    readFor: "重点看无 simplex 约束带来的并行性、唯一解证明、primal-dual 算法，以及实验如何证明效率和泛化性。",
    relation: "这是高维分类线的成熟版本，也说明早期 ROF/Mumford-Shah/TV 工具箱已经抽象到 graph-based classification。"
  },
  {
    priority: 11,
    problem: "高维逆问题中只给一个重建图像不够，还需要知道哪些区域可信、哪些结构不确定。",
    method: "用凸优化得到 MAP，再通过概率集中不等式构造 HPD credible regions 和 local credible intervals。",
    readFor: "把它当 UQ 入口读：先掌握 MAP、HPD 区域、局部可信区间和正则参数估计，再读无线电干涉两篇 companion。",
    relation: "它把 RI MAP-UQ 思路推广到一般高维逆问题，是 RI II 与 Proximal Nested Sampling 之间的概念桥。"
  },
  {
    priority: 12,
    problem: "无线电干涉成像后验含稀疏、非光滑先验，传统 MCMC 很难直接采样。",
    method: "用 proximal MCMC 对完整后验采样，然后做 pixel-wise credible intervals、HPD regions 和结构假设检验。",
    readFor: "重点看它能给出完整后验信息，也要注意计算代价；这正是 RI II 做 MAP 快速近似的动机。",
    relation: "它是完整采样基线；和 RI II 对照读能理解“准确后验采样”和“可扩展近似 UQ”的取舍。"
  },
  {
    priority: 13,
    problem: "RI I 的 MCMC 后验采样对 SKA 级大数据太贵，需要一个能落地的快速 UQ 方案。",
    method: "用 MAP estimation 加 probability concentration 构造近似可信区域和局部可信区间。",
    readFor: "重点看 MAP-UQ 的近似逻辑、credible region 如何从优化解得到、以及它相对 RI I 损失和获得了什么。",
    relation: "这是无线电/UQ 线最关键的实践论文；High-Dimensional Inverse Problems UQ 是它的一般化版本。"
  },
  {
    priority: 14,
    problem: "无线电观测 visibility 数据是流式到达且规模巨大，不能全部保存后再离线重建。",
    method: "数据块到达时 assimilate，完成更新后 discard，通过 online optimization 降低存储和计算压力。",
    readFor: "重点看数据流设计、到达即处理的优化流程，以及它如何服务后续 RI-UQ 的大规模场景。",
    relation: "它更像 RI-UQ 的工程前置条件：先解决数据规模和在线重建，再讨论重建结果的不确定性。"
  },
  {
    priority: 15,
    problem: "UQ 问给定模型下结果有多可信，模型选择则进一步问哪个 Bayesian model 更合适。",
    method: "结合 proximal MCMC 和 nested sampling 来估计 Bayesian evidence，用于高维模型比较。",
    readFor: "最后读，重点看 evidence、nested sampling、likelihood contour 和 proximal operator 如何组合。",
    relation: "它是无线电/UQ 线的上层扩展：从重建和不确定性推进到 Bayesian model selection。"
  }
];

const paperByPriority = new Map(papers.map((paper) => [paper.priority, paper]));

let activeTrack = "all";
let query = "";

function pdfHref(file) {
  return encodeURI(basePath + file);
}

function byId(id) {
  return document.getElementById(id);
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

function renderNotes() {
  byId("noteGrid").innerHTML = readingNotes.map((note) => {
    const paper = paperByPriority.get(note.priority);
    const track = tracks[paper.track];
    return `
      <article class="note-card" style="--accent:${track.color}">
        <header class="note-head">
          <span>#${paper.priority}</span>
          <div>
            <h3>${paper.title}</h3>
            <small>${paper.time} · ${track.label} · ${paper.pages} 页</small>
          </div>
        </header>
        <div class="note-sections">
          <section>
            <strong>核心问题</strong>
            <p>${note.problem}</p>
          </section>
          <section>
            <strong>方法抓手</strong>
            <p>${note.method}</p>
          </section>
          <section>
            <strong>精读重点</strong>
            <p>${note.readFor}</p>
          </section>
          <section>
            <strong>关联关系</strong>
            <p>${note.relation}</p>
          </section>
        </div>
        <div class="note-actions">
          <a href="${pdfHref(paper.file)}">打开 PDF</a>
          <a href="00_papers_first_author_xiaohao_cai_deduped/agent_team_reading_report.md">完整报告</a>
        </div>
      </article>
    `;
  }).join("");
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
  renderNotes();
  bindEvents();
}

document.addEventListener("DOMContentLoaded", init);
