(function () {
  const data = window.ZX_READING_DATA;
  const shared = window.ZX_SHARED;
  if (!data || !shared) {
    console.error("ZX report dependencies are missing. Load shared.js and reading-data.js before report.js.");
    return;
  }

  const {
    tracks,
    papers,
    chronology,
    readingStages,
    noteMainlines,
    paperNotesV2,
    readingReasons,
    layerBlocks,
    weeklyPlan,
    researchTopics,
    finalSummary,
    siteMeta
  } = data;

  const {
    byId,
    escapeHtml,
    pdfHref,
    themeLabel,
    createPaperMaps,
    asList,
    asNumberedList
  } = shared;

  const { paperByPriority, noteByPriority } = createPaperMaps(data);

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

  function paperFor(note) {
    return paperByPriority.get(note.priority);
  }

  function stageFor(priority) {
    const stage = readingStages.find((item) => item.priorities.includes(priority));
    return stage ? `${stage.stage}：${stage.title}` : `阅读顺序 #${priority}`;
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
              <a href="${pdfHref(paper.file)}" target="_blank" rel="noopener">打开 PDF</a>
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

  function renderReportExpansion(note) {
    const expansion = note.reportExpansion || {};
    const rows = [
      ["背景定位", expansion.context],
      ["技术阅读", expansion.technicalReading],
      ["理论阅读", expansion.theoremReading],
      ["实验阅读", expansion.experimentReading],
      ["关系阅读", expansion.relationReading],
      ["研究价值", expansion.researchValue]
    ].filter(([, value]) => value);

    return `
      <section class="report-subsection report-expansion">
        <h4>报告专用精读展开</h4>
        ${rows.map(([label, value]) => `
          <p><strong>${escapeHtml(label)}：</strong>${escapeHtml(value)}</p>
        `).join("")}
      </section>
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
            <a class="report-button primary" href="${pdf}" target="_blank" rel="noopener">打开 PDF</a>
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

        ${renderReportExpansion(note)}

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
          ${asNumberedList(note.algorithmFlow)}
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
          <h4>5.${note.priority}.9 证据定位</h4>
          ${asList(note.evidence || ["待核对"])}
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.10 和其他论文的关系</h4>
          <p>${escapeHtml(note.relation.text)}</p>
          <div class="relation-chips">${relations}</div>
        </section>

        <section class="report-subsection">
          <h4>5.${note.priority}.11 精读问题</h4>
          ${asList(note.readingQuestions)}
        </section>

        <section class="report-subsection output-block">
          <h4>5.${note.priority}.12 读后产出</h4>
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

  function renderSiteVersion() {
    const target = byId("reportVersion");
    if (!target || !siteMeta) return;
    target.textContent = `commit: ${siteMeta.commit} · last updated: ${siteMeta.lastUpdated}`;
  }

  function renderReport() {
    renderSiteVersion();
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
