const basePath = "00_papers_first_author_xiaohao_cai_deduped/";

const tracks = {
  variational: {
    label: "变分分割 / SaT",
    short: "SaT",
    count: 8,
    color: "#28666e",
    summary: "用恢复、凸优化、TV/ROF/Mumford-Shah、framelet 替代直接非凸分割，再用阈值、投影或聚类输出分割结果。",
    emphasis: "理论核心是 PCMS 与 ROF 的联系；应用核心是 SLaT、T-ROF 和 framelet 管状结构分割。"
  },
  inverse: {
    label: "无线电 / UQ / 采样",
    short: "UQ",
    count: 6,
    color: "#8f5b2e",
    summary: "从无线电干涉重建进入高维贝叶斯逆问题，逐步补齐可扩展重建、不确定性量化和模型选择。",
    emphasis: "主链是 Online RI -> proximal MCMC UQ -> MAP UQ -> 一般逆问题 UQ -> proximal nested sampling。"
  },
  classification: {
    label: "高维图分类",
    short: "CL",
    count: 1,
    color: "#5e548e",
    summary: "把 SaT/ROF/Mumford-Shah 的变分分割思想迁移到高维数据和点云半监督分类。",
    emphasis: "关键设计是无 simplex 约束的凸模型、类别标签函数独立求解和 argmax 投影。"
  }
};

const papers = [
  {
    priority: 1,
    file: "分割方法论总览 SaT Overview.pdf",
    title: "An Overview of SaT Segmentation Methodology and Its Applications in Image Processing",
    year: 2023,
    pages: 27,
    track: "variational",
    type: "Springer Handbook",
    note: "先建立 SaT 方法地图；这是综述章节，不是原始方法论文。"
  },
  {
    priority: 2,
    file: "变分分割基础Mumford-Shah与ROF Mumford-Shah ROF.pdf",
    title: "Linkage Between Piecewise Constant Mumford-Shah Model and ROF Model and Its Virtue in Image Segmentation",
    year: 2019,
    pages: 31,
    track: "variational",
    type: "arXiv",
    note: "理论核心：解释为什么 ROF 恢复加阈值能求解 PCMS/Chan-Vese 类分割。"
  },
  {
    priority: 3,
    file: "多类ROF分割 Iterated ROF.pdf",
    title: "Multiclass Segmentation by Iterated ROF Thresholding",
    year: 2013,
    pages: 14,
    track: "variational",
    type: "LNCS / EMMCVPR",
    note: "T-ROF 早期会议版，是 2019 理论长文的直接前身。"
  },
  {
    priority: 4,
    file: "SLaT三阶段分割 SLaT Segmentation.pdf",
    title: "A Three-stage Approach for Segmenting Degraded Color Images: Smoothing, Lifting and Thresholding",
    year: 2015,
    pages: 19,
    track: "variational",
    type: "arXiv",
    note: "Smoothing-Lifting-Thresholding，把 SaT 推到退化彩色图像。"
  },
  {
    priority: 5,
    file: "分割恢复联合模型 Segmentation Restoration.pdf",
    title: "Variational Image Segmentation Model Coupled with Image Restoration Achievements",
    year: 2014,
    pages: 23,
    track: "variational",
    type: "arXiv",
    note: "联合恢复与分割路线，对比 SaT 的解耦路线。"
  },
  {
    priority: 6,
    file: "两阶段分类 Two-Stage.pdf",
    title: "A Two-Stage Classification Method for High-Dimensional Data and Point Clouds",
    year: 2019,
    pages: 21,
    track: "variational",
    type: "arXiv",
    note: "SaT 方法论迁移到图上的半监督分类。"
  },
  {
    priority: 7,
    file: "高效变分分类 Efficient Variational.pdf",
    title: "An Efficient and Versatile Variational Method for High-Dimensional Data Classification",
    year: 2024,
    pages: 25,
    track: "classification",
    type: "Journal of Scientific Computing",
    note: "2024 期刊版，强调无 simplex 约束、并行标签函数和 one-class 扩展。"
  },
  {
    priority: 8,
    file: "在线无线电干涉成像 Online Radio Imaging.pdf",
    title: "Online radio interferometric imaging: assimilating and discarding visibilities on arrival",
    year: 2017,
    pages: 14,
    track: "inverse",
    type: "MNRAS style preprint",
    note: "RI 大数据流式重建入口：边观测、边重建、边丢弃 visibility blocks。"
  },
  {
    priority: 9,
    file: "无线电干涉不确定性I Radio Interferometric I.pdf",
    title: "Uncertainty quantification for radio interferometric imaging: I. proximal MCMC methods",
    year: 2018,
    pages: 16,
    track: "inverse",
    type: "MNRAS companion I",
    note: "用 proximal MCMC 处理非光滑稀疏后验，给出 credible intervals、HPD regions 和结构检验。"
  },
  {
    priority: 10,
    file: "无线电干涉不确定性II Radio Interferometric II.pdf",
    title: "Uncertainty quantification for radio interferometric imaging: II. MAP estimation",
    year: 2018,
    pages: 13,
    track: "inverse",
    type: "MNRAS companion II",
    note: "核心落地论文：MAP + probability concentration 近似 UQ，替代昂贵 MCMC。"
  },
  {
    priority: 11,
    file: "高维逆问题不确定性量化 Uncertainty Quantification.pdf",
    title: "Quantifying Uncertainty in High Dimensional Inverse Problems by Convex Optimisation",
    year: 2019,
    pages: 5,
    track: "inverse",
    type: "EUSIPCO",
    note: "把 RI MAP-UQ 泛化到一般高维逆问题，并加入自动正则参数估计。"
  },
  {
    priority: 12,
    file: "近端嵌套采样 Proximal Nested Sampling.pdf",
    title: "Proximal nested sampling for high-dimensional Bayesian model selection",
    year: 2022,
    pages: 42,
    track: "inverse",
    type: "arXiv / Bayesian computation",
    note: "从 UQ 推进到 Bayesian evidence 和模型选择，是 UQ 主链的升级。"
  },
  {
    priority: 13,
    file: "球面小波分割 Wavelet Sphere.pdf",
    title: "Wavelet-based segmentation on the sphere",
    year: 2016,
    pages: 22,
    track: "inverse",
    type: "arXiv",
    note: "球面/小波/几何信号处理支线，与 RI-UQ 共享稀疏表示和非欧氏数据处理。"
  },
  {
    priority: 14,
    file: "框架管状结构分割 Framelet.pdf",
    title: "Framelet-Based Algorithm for Segmentation of Tubular Structures",
    year: 2012,
    pages: 12,
    track: "variational",
    type: "SSVM / LNCS",
    note: "早期会议版，体现 framelet 管状结构分割的源头。"
  },
  {
    priority: 15,
    file: "框架分割管状结构 Framelet Tubular.pdf",
    title: "Vessel Segmentation in Medical Imaging Using a Tight-Frame Based Algorithm",
    year: 2011,
    pages: 13,
    track: "variational",
    type: "arXiv",
    note: "扩展版，引入方向选择性 tight-frame 和更复杂 3D MRA 实验。"
  }
];

const timeline = [
  { year: "2011-2012", label: "Framelet / tight-frame 管状结构", track: "variational" },
  { year: "2013-2015", label: "T-ROF、联合恢复分割、SLaT", track: "variational" },
  { year: "2017-2019", label: "RI 在线重建、proximal MCMC UQ、MAP UQ", track: "inverse" },
  { year: "2019-2024", label: "SaT 到图分类，再到高维变分分类期刊版", track: "classification" },
  { year: "2022", label: "Proximal nested sampling 做高维模型选择", track: "inverse" }
];

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
      paper.year,
      tracks[paper.track].label,
      paper.note
    ].join(" ").toLowerCase().includes(normalized);
    return trackMatch && searchMatch;
  });
}

function renderMetrics() {
  const metrics = [
    { label: "去重后论文", value: "15", detail: "第一作者 PDF" },
    { label: "研究主线", value: "3", detail: "SaT / UQ / 分类" },
    { label: "删除旧重复", value: "1", detail: "错命名近重复" },
    { label: "阅读报告", value: "3", detail: "Agent 分组完成" }
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
  byId("trackOverview").innerHTML = Object.entries(tracks).map(([key, track]) => `
    <article class="track-card" style="--accent:${track.color}">
      <div class="track-top">
        <span>${track.short}</span>
        <strong>${track.count}</strong>
      </div>
      <h3>${track.label}</h3>
      <p>${track.summary}</p>
      <div class="bar"><i style="width:${track.count / 8 * 100}%"></i></div>
    </article>
  `).join("");
}

function renderTimeline() {
  byId("timeline").innerHTML = timeline.map((item) => `
    <article class="time-node" style="--accent:${tracks[item.track].color}">
      <span>${item.year}</span>
      <p>${item.label}</p>
    </article>
  `).join("");
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
          <small>${paper.note}</small>
        </td>
        <td>${paper.year}</td>
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
          ${list.slice(0, 5).map((paper) => `<li>${paper.title}</li>`).join("")}
        </ul>
      </article>
    `;
  }).join("");
}

function renderReadingList() {
  byId("readingList").innerHTML = papers.map((paper) => `
    <li style="--accent:${tracks[paper.track].color}">
      <div class="read-rank">${paper.priority}</div>
      <div>
        <h3>${paper.title}</h3>
        <p>${paper.note}</p>
        <span>${paper.year} · ${paper.type} · ${paper.pages} 页</span>
      </div>
      <a href="${pdfHref(paper.file)}">打开 PDF</a>
    </li>
  `).join("");
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
  renderMetrics();
  renderTrackOverview();
  renderTimeline();
  renderFilters();
  renderRows();
  renderTrackDetails();
  renderReadingList();
  bindEvents();
}

document.addEventListener("DOMContentLoaded", init);
