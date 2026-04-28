(function () {
  const data = window.ZX_READING_DATA;
  if (!data) {
    console.error("ZX_READING_DATA is missing. Load app.js before report.js.");
    return;
  }

  const {
    basePath,
    tracks,
    papers,
    chronology,
    readingStages,
    noteMainlines,
    paperNotesV2
  } = data;

  const paperByPriority = new Map(papers.map((paper) => [paper.priority, paper]));
  const noteByPriority = new Map(paperNotesV2.map((note) => [note.priority, note]));

  const themeLabels = {
    "sat-rof": "SaT/ROF 理论",
    medical: "医学与管状结构",
    extension: "彩色/球面扩展",
    classification: "高维分类",
    "ri-uq": "无线电干涉与 UQ",
    "bayes-model": "贝叶斯模型选择"
  };

  const tocItems = [
    ["report-positioning", "0. 报告定位"],
    ["report-direction", "1. 总体研究方向判断"],
    ["report-timeline-section", "2. 论文发表时间线"],
    ["report-order-section", "3. 推荐阅读顺序"],
    ["report-layers", "4. 三层研究脉络图"],
    ["report-papers-section", "5. 逐篇精读报告"],
    ["report-dependencies-section", "6. 依赖关系"],
    ["report-plan-section", "7. 四周阅读计划"],
    ["report-topics-section", "8. 选题入口"],
    ["report-summary-section", "9. 最终总结"]
  ];

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

  function pdfHref(file) {
    return encodeURI(basePath + file);
  }

  function paperFor(note) {
    return paperByPriority.get(note.priority);
  }

  function themeLabel(key) {
    return themeLabels[key] || key;
  }

  function stageFor(priority) {
    const stage = readingStages.find((item) => item.priorities.includes(priority));
    return stage ? `${stage.stage}：${stage.title}` : `阅读顺序 #${priority}`;
  }

  function list(items) {
    return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
  }

  function numbered(items) {
    return `<ol>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ol>`;
  }

  function paperChip(priority) {
    const note = noteByPriority.get(priority);
    if (!note) return `<span class="report-chip">#${priority}</span>`;
    return `<a class="report-chip" href="#paper-${escapeHtml(note.id)}">#${priority} ${escapeHtml(note.titleCn)}</a>`;
  }

  function renderToc() {
    const sections = tocItems.map(([id, label]) => `<a href="#${id}">${escapeHtml(label)}</a>`).join("");
    const papersList = paperNotesV2.map((note) => (
      `<a href="#paper-${escapeHtml(note.id)}">5.${note.priority} ${escapeHtml(note.titleCn)}</a>`
    )).join("");
    byId("reportToc").innerHTML = sections + `<div class="toc-paper-list">${papersList}</div>`;
  }

  function renderMainlines() {
    byId("reportMainlines").innerHTML = noteMainlines.map((mainline) => `
      <article class="mainline-card">
        <h3>${escapeHtml(mainline.title)}</h3>
        <p>${escapeHtml(mainline.summary)}</p>
        <div class="mainline-papers">
          ${mainline.papers.map(paperChip).join("")}
        </div>
      </article>
    `).join("");
  }

  function renderTimeline() {
    byId("reportTimeline").innerHTML = chronology.map((item) => {
      const paper = paperByPriority.get(item.priority);
      const note = noteByPriority.get(item.priority);
      const track = tracks[item.track];
      return `
        <tr>
          <td>${escapeHtml(item.time)}</td>
          <td>
            <strong>${escapeHtml(note?.titleCn || item.label)}</strong>
            <small>${escapeHtml(paper.title)}</small>
          </td>
          <td><span class="chip" style="--accent:${track.color}">${escapeHtml(track.label)}</span></td>
          <td>${escapeHtml(paper.position)}</td>
        </tr>
      `;
    }).join("");
  }

  function renderReadingOrder() {
    byId("reportReadingOrder").innerHTML = paperNotesV2.map((note) => {
      const paper = paperFor(note);
      return `
        <li class="report-order-item">
          <div class="order-rank">#${note.priority}</div>
          <div>
            <h3>${escapeHtml(note.titleCn)}</h3>
            <p><strong>${escapeHtml(note.titleEn)}</strong></p>
            <p>${escapeHtml(readingReasons[note.priority] || note.howToRead)}</p>
            <div class="report-actions-row">
              <a href="#paper-${escapeHtml(note.id)}">跳到精读报告</a>
              <a href="${pdfHref(paper.file)}">打开 PDF</a>
            </div>
          </div>
        </li>
      `;
    }).join("");
  }

  function renderLayers() {
    const tree = `分割方法论层
 ├─ SaT Overview
 ├─ Linkage Between PCMS and ROF
 │   └─ Multiclass T-ROF
 ├─ Segmentation Restoration
 ├─ SLaT
 └─ Framelet / Tight-frame Vessel
     └─ Wavelet Sphere

逆问题与不确定性层
 ├─ RI UQ I: Proximal MCMC
 ├─ RI UQ II: MAP Estimation
 ├─ Quantifying UQ
 ├─ Online RI Imaging
 └─ Proximal Nested Sampling

高维分类迁移层
 ├─ Two-Stage Classification
 └─ Efficient Variational Classification`;

    byId("reportLayers").innerHTML = `
      <div class="layer-grid">
        ${layerBlocks.map((block) => `
          <article class="layer-card">
            <h3>${escapeHtml(block.title)}</h3>
            <p>${escapeHtml(block.body)}</p>
            <div class="relation-chips">${block.papers.map(paperChip).join("")}</div>
          </article>
        `).join("")}
      </div>
      <pre class="report-tree">${escapeHtml(tree)}</pre>
      <p>三层之间的关系可以这样理解：分割方法论层提供了 smoothing + thresholding、Total Variation、frame/wavelet sparse representation 和候选边界收缩等基础工具；高维分类层把这些工具从像素区域迁移到 graph label function；逆问题与不确定性层则把同样的 convex optimisation 和 sparse prior 语言用于 radio interferometric imaging、Bayesian posterior、MAP estimation 和 proximal sampling。</p>
    `;
  }

  function renderPaper(note) {
    const paper = paperFor(note);
    const track = tracks[paper.track];
    const relations = note.relation.links.map(paperChip).join("");
    const pdf = pdfHref(paper.file);
    const dashboardHref = `index.html?note=${encodeURIComponent(note.id)}#notes`;
    return `
      <article class="report-paper" id="paper-${escapeHtml(note.id)}" style="--accent:${track.color}">
        <header class="report-paper-head">
          <div>
            <p class="eyebrow">5.${note.priority}</p>
            <h3>${escapeHtml(note.titleCn)}</h3>
            <small>${escapeHtml(note.titleEn)}</small>
          </div>
          <div class="report-paper-actions">
            <a class="report-button primary" href="${pdf}">打开 PDF</a>
            <a class="report-button" href="${dashboardHref}">回到精读笔记</a>
          </div>
        </header>

        <div class="report-meta-row">
          <span>${escapeHtml(String(note.year))}</span>
          <span>${escapeHtml(themeLabel(note.theme))}</span>
          <span>${escapeHtml(note.difficulty)}</span>
          <span>${escapeHtml(stageFor(note.priority))}</span>
          <span>${paper.pages} 页</span>
        </div>

        <section class="report-subsection">
          <h4>5.${note.priority}.1 一句话定位</h4>
          <p>${escapeHtml(note.oneSentence)}</p>
          <p>在 15 篇论文中，它的阅读位置是 <strong>#${note.priority}</strong>。它不只是一个孤立应用，而是服务于 ${escapeHtml(themeLabel(note.theme))} 这条主线，并与相邻论文共同构成从模型、算法到实验对象的递进关系。</p>
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.2 核心问题</h4>
          <p>${escapeHtml(note.coreProblem)}</p>
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.3 为什么难</h4>
          <p>${escapeHtml(note.whyHard)}</p>
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.4 方法抓手</h4>
          <p>${escapeHtml(note.methodHandle)}</p>
          <p>精读时不要只记住“效果”这样的结论，而要把方法拆成变量、目标函数、迭代步骤和输出对象：这篇的核心输出应能落到公式、流程图或伪代码上。</p>
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.5 关键模型/公式</h4>
          <pre class="report-formula">${escapeHtml(note.keyModelOrFormula)}</pre>
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.6 算法流程</h4>
          ${numbered(note.algorithmFlow)}
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.7 理论保证</h4>
          <p>${escapeHtml(note.theoremOrGuarantee)}</p>
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.8 实验重点</h4>
          <p>${escapeHtml(note.experimentFocus)}</p>
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.9 和其他论文的关系</h4>
          <p>${escapeHtml(note.relation.text)}</p>
          <div class="relation-chips">${relations}</div>
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.10 精读问题</h4>
          ${list(note.readingQuestions)}
        </section>

        <section class="report-subsection output-block">
          <h4>5.${note.priority}.11 读后产出</h4>
          <p>${escapeHtml(note.afterReadingOutput)}</p>
          <p><strong>为什么这样读：</strong>${escapeHtml(note.howToRead)}</p>
        </section>
      </article>
    `;
  }

  function renderPapers() {
    byId("reportPapers").innerHTML = paperNotesV2.map(renderPaper).join("");
  }

  function renderDependencies() {
    const dependencyTree = `SaT Overview
 ├─ Linkage Between PCMS and ROF
 │   └─ Multiclass T-ROF
 ├─ Segmentation Restoration
 ├─ SLaT
 ├─ Framelet / Tight-frame Vessel
 │   └─ Wavelet Sphere
 └─ Two-Stage Classification
     └─ Efficient Variational Classification

RI UQ I
 ├─ RI UQ II
 ├─ Quantifying UQ
 ├─ Online RI Imaging
 └─ Proximal Nested Sampling`;

    byId("reportDependencies").innerHTML = `
      <pre class="report-tree">${escapeHtml(dependencyTree)}</pre>
      <p>第一棵树从 SaT Overview 出发，是因为它在方法论上覆盖 T-ROF、SLaT、vascular 和 sphere。Linkage 负责解释 ROF 与 PCMS/Chan-Vese 的理论连接，Multiclass T-ROF 是算法前身，Segmentation Restoration 则展示 joint optimization 路线。Framelet/Tight-frame vessel 与 Wavelet Sphere 的连接来自候选边界区间和 wavelet/frame 表示，高维分类线则把 thresholding/projection 思想从 image segmentation 迁移到 graph classification。</p>
      <p>第二棵树从 RI UQ I 出发，是因为完整 posterior sampling 给了 UQ 的基准：RI UQ II 用 MAP estimation 和 probability concentration 追求可扩展性，Quantifying UQ 把 MAP-UQ 思想推广到一般高维逆问题，Online RI Imaging 解决数据块流式处理，Proximal Nested Sampling 则把问题推进到 Bayesian evidence 和模型选择。</p>
    `;
  }

  function renderPlan() {
    byId("reportPlan").innerHTML = weeklyPlan.map((week) => `
      <article class="week-card">
        <span>${escapeHtml(week.week)}</span>
        <h3>${escapeHtml(week.theme)}</h3>
        <p><strong>必读论文：</strong>${week.papers.map(paperChip).join(" ")}</p>
        <p><strong>目标：</strong>${escapeHtml(week.goal)}</p>
        <p><strong>要画的图：</strong>${escapeHtml(week.diagrams)}</p>
        <p><strong>要复现或伪代码化的算法：</strong>${escapeHtml(week.pseudo)}</p>
        <p><strong>300-500 字总结：</strong>${escapeHtml(week.summary)}</p>
      </article>
    `).join("");
  }

  function renderTopics() {
    byId("reportTopics").innerHTML = researchTopics.map((topic) => `
      <article class="topic-card">
        <h3>${escapeHtml(topic.title)}</h3>
        <p><strong>来自论文：</strong>${topic.sources.map(paperChip).join(" ")}</p>
        <p><strong>可以创新的地方：</strong>${escapeHtml(topic.innovation)}</p>
        <p><strong>难点：</strong>${escapeHtml(topic.difficulty)}</p>
        <p><strong>初步实验：</strong>${escapeHtml(topic.experiment)}</p>
      </article>
    `).join("");
  }

  function renderSummary() {
    byId("reportSummary").innerHTML = finalSummary.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join("");
  }

  function renderReport() {
    renderToc();
    renderMainlines();
    renderTimeline();
    renderReadingOrder();
    renderLayers();
    renderPapers();
    renderDependencies();
    renderPlan();
    renderTopics();
    renderSummary();
  }

  document.addEventListener("DOMContentLoaded", renderReport);
})();
