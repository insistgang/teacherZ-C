(function () {
  const data = window.ZX_READING_DATA;
  const shared = window.ZX_SHARED;
  if (!data || !shared) {
    console.error("ZX dashboard dependencies are missing. Load shared.js and reading-data.js before dashboard.js.");
    return;
  }

  const {
    tracks,
    thesis,
    papers,
    chronology,
    readingStages,
    noteThemes,
    noteMainlines,
    readingStandard,
    paperNotesV2,
    reproScoring,
    reproRecommendedBatches,
    reproAssessments,
    siteMeta
  } = data;
  const { byId, escapeHtml, pdfHref, themeLabel, notePdf, createPaperMaps } = shared;
  const { paperByPriority, noteByPriority } = createPaperMaps(data);

let activeTrack = "all";
let query = "";
let activeNoteTheme = "all";
let noteQuery = "";
let activeReproLevel = "all";
let reproQuery = "";






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
    note.evidence?.join(" "),
    Object.values(note.reportExpansion || {}).join(" "),
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

function reproSearchText(item) {
  return [
    item.titleCn,
    item.titleEn,
    item.reproductionLevel,
    item.difficultyLabel,
    item.effectLabel,
    item.fullReproductionFeasibility,
    item.minimalExperiment,
    item.expectedOutcome,
    item.metrics?.join(" "),
    item.dependencies?.join(" "),
    item.dataRequirement,
    item.computeRequirement,
    item.implementationRisk,
    item.verificationPlan,
    item.resultStatus,
    item.notes,
    Object.keys(item.runMetrics || {}).join(" "),
    Object.values(item.runMetrics || {}).join(" ")
  ].join(" ").toLowerCase();
}

function filteredReproAssessments() {
  const normalized = reproQuery.trim().toLowerCase();
  return reproAssessments.filter((item) => {
    const levelMatch = activeReproLevel === "all" || item.reproductionLevel === activeReproLevel;
    const searchMatch = !normalized || reproSearchText(item).includes(normalized);
    return levelMatch && searchMatch;
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

function renderSiteVersion() {
  const target = byId("dashboardVersion");
  if (!target || !siteMeta) return;
  target.textContent = `commit: ${siteMeta.commit} · last updated: ${siteMeta.lastUpdated}`;
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
        <a href="${pdfHref(paper.file)}" target="_blank" rel="noopener">#${paper.priority} 打开 PDF</a>
      </article>
    `;
  }).join("");
}

function renderTrackFilters() {
  const filters = [{ key: "all", label: "全部" }].concat(
    Object.entries(tracks).map(([key, track]) => ({ key, label: track.label }))
  );

  byId("trackFilters").innerHTML = filters.map((filter) => `
    <button class="${filter.key === activeTrack ? "active" : ""}" data-filter="${filter.key}" type="button">${filter.label}</button>
  `).join("");
}

function renderPaperRows() {
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
        <td><a class="file-link" href="${pdfHref(paper.file)}" target="_blank" rel="noopener">PDF</a></td>
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
          ${list.map((paper) => `<li><a href="${pdfHref(paper.file)}" target="_blank" rel="noopener">#${paper.priority} ${paper.title}</a></li>`).join("")}
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
              <a class="stage-paper" style="--accent:${track.color}" href="${pdfHref(paper.file)}" target="_blank" rel="noopener">
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
          ${renderNoteList("证据定位", note.evidence || ["待核对"])}
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
          <a href="${pdfHref(pdf)}" target="_blank" rel="noopener">打开 PDF</a>
          <a href="reading_report.html#paper-${escapeHtml(note.id)}">完整研究报告</a>
          <button type="button" class="note-toggle" data-note-toggle="${note.priority}" aria-expanded="false" aria-controls="note-detail-${note.priority}">展开精读字段</button>
        </div>
      </article>
    `;
  }).join("");
}

function renderScoreDots(score, label) {
  const dots = Array.from({ length: 5 }, (_, index) => (
    `<span class="${index < score ? "on" : ""}"></span>`
  )).join("");
  return `<div class="score-dots" aria-label="${escapeHtml(label)} ${score} / 5">${dots}<strong>${score}/5</strong></div>`;
}

function renderMetricPairs(metrics) {
  const entries = Object.entries(metrics || {});
  if (!entries.length) return "<p>暂无运行指标。</p>";
  return `
    <dl class="metric-pairs">
      ${entries.map(([key, value]) => `
        <div>
          <dt>${escapeHtml(key)}</dt>
          <dd>${escapeHtml(value)}</dd>
        </div>
      `).join("")}
    </dl>
  `;
}

function renderResultFiles(item) {
  if (!item.resultFiles?.length) return "<p>暂无结果文件。</p>";
  return `
    <div class="result-file-grid">
      ${item.resultFiles.map((file) => `
        <a href="${escapeHtml(file)}" target="_blank" rel="noopener">
          ${/\.(png|jpg|jpeg|webp)$/i.test(file) ? `<img src="${escapeHtml(file)}" alt="${escapeHtml(item.titleCn)} 复现结果图">` : ""}
          <span>${escapeHtml(file.replace("assets/repro/", ""))}</span>
        </a>
      `).join("")}
    </div>
  `;
}

function renderReproSummary() {
  if (!byId("reproSummary")) return;
  const completed = reproAssessments.filter((item) => item.resultStatus === "completed").length;
  const skipped = reproAssessments.filter((item) => item.resultStatus === "skipped").length;
  const partial = reproAssessments.filter((item) => item.reproductionLevel === "partial").length;
  const toy = reproAssessments.filter((item) => item.reproductionLevel === "toy").length;
  const hardest = [...reproAssessments].sort((a, b) => b.difficultyScore - a.difficultyScore || a.priority - b.priority).slice(0, 3);
  const clearest = [...reproAssessments].sort((a, b) => b.effectScore - a.effectScore || a.difficultyScore - b.difficultyScore).slice(0, 3);

  const cards = [
    { label: "评估论文", value: reproAssessments.length, detail: "与精读卡片一一对应" },
    { label: "已运行", value: completed, detail: skipped ? `${skipped} 个 skipped` : "当前 0 个 skipped" },
    { label: "partial / toy", value: `${partial} / ${toy}`, detail: "没有伪装 paper-level full reproduction" },
    { label: "最难 full reproduction", value: hardest.map((item) => `#${item.priority}`).join(" "), detail: hardest.map((item) => item.titleCn).join(" · ") },
    { label: "演示最清楚", value: clearest.map((item) => `#${item.priority}`).join(" "), detail: clearest.map((item) => item.titleCn).join(" · ") }
  ];

  byId("reproSummary").innerHTML = cards.map((card) => `
    <article class="metric repro-metric">
      <span>${escapeHtml(card.label)}</span>
      <strong>${escapeHtml(card.value)}</strong>
      <small>${escapeHtml(card.detail)}</small>
    </article>
  `).join("");
}

function renderReproScoring() {
  const difficulty = byId("reproDifficultyScoring");
  const effect = byId("reproEffectScoring");
  if (difficulty) difficulty.innerHTML = (reproScoring?.difficultyDimensions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  if (effect) effect.innerHTML = (reproScoring?.effectDimensions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
}

function renderReproBatches() {
  const target = byId("reproBatches");
  if (!target) return;
  target.innerHTML = reproRecommendedBatches.map((batch, index) => `
    <article class="repro-batch">
      <button type="button" data-repro-batch="${index}">
        <span>${String(index + 1).padStart(2, "0")}</span>
        ${escapeHtml(batch.title)}
      </button>
      <p>${escapeHtml(batch.reason)}</p>
      <div class="relation-chips">
        ${batch.priorities.map((priority) => {
          const item = reproAssessments.find((entry) => entry.priority === priority);
          return `<button type="button" class="relation-chip" data-scroll-repro="${priority}">#${priority} ${escapeHtml(item?.titleCn || `论文 ${priority}`)}</button>`;
        }).join("")}
      </div>
    </article>
  `).join("");
}

function renderReproLevelFilters() {
  const target = byId("reproLevelFilters");
  if (!target) return;
  const levels = [{ key: "all", label: "全部" }].concat(
    [...new Set(reproAssessments.map((item) => item.reproductionLevel))].map((level) => ({ key: level, label: level }))
  );
  target.innerHTML = levels.map((level) => `
    <button class="${level.key === activeReproLevel ? "active" : ""}" data-repro-level="${escapeHtml(level.key)}" type="button">
      ${escapeHtml(level.label)}
    </button>
  `).join("");
}

function renderReproCards() {
  const target = byId("reproCards");
  if (!target) return;
  const items = filteredReproAssessments();
  const count = byId("reproCount");
  if (count) count.textContent = `${items.length} / ${reproAssessments.length} 篇`;
  target.innerHTML = items.map((item) => {
    const paper = paperByPriority.get(item.priority);
    const track = tracks[paper.track];
    const pdf = notePdf(item);
    return `
      <article class="repro-card" id="repro-card-${item.priority}" style="--accent:${track.color}">
        <header class="repro-card-head">
          <div>
            <div class="note-meta">
              <span>#${item.priority}</span>
              <span>${escapeHtml(item.reproductionLevel)}</span>
              <span>${escapeHtml(item.resultStatus)}</span>
            </div>
            <h3>${escapeHtml(item.titleCn)}</h3>
            <small>${escapeHtml(item.titleEn)}</small>
          </div>
          <span class="status-pill ${escapeHtml(item.resultStatus)}">${escapeHtml(item.resultStatus)}</span>
        </header>

        <div class="repro-score-grid">
          <section>
            <strong>复现难度：${escapeHtml(item.difficultyLabel)}</strong>
            ${renderScoreDots(item.difficultyScore, "复现难度")}
          </section>
          <section>
            <strong>展示效果：${escapeHtml(item.effectLabel)}</strong>
            ${renderScoreDots(item.effectScore, "展示效果")}
          </section>
        </div>

        <div class="note-summary repro-summary">
          <section>
            <strong>最小可复现实验</strong>
            <p>${escapeHtml(item.minimalExperiment)}</p>
          </section>
          <section>
            <strong>预期 / 实际结果</strong>
            <p>${escapeHtml(item.expectedOutcome)}</p>
          </section>
          <section>
            <strong>Full reproduction 判断</strong>
            <p>${escapeHtml(item.fullReproductionFeasibility)}</p>
          </section>
        </div>

        <div class="note-expanded repro-expanded" id="repro-detail-${item.priority}">
          <section class="note-detail-block">
            <strong>依赖与数据</strong>
            <p>${escapeHtml(item.dependencies.join(" / "))}</p>
            <p>${escapeHtml(item.dataRequirement)}</p>
          </section>
          <section class="note-detail-block">
            <strong>算力与风险</strong>
            <p>${escapeHtml(item.computeRequirement)}</p>
            <p>${escapeHtml(item.implementationRisk)}</p>
          </section>
          <section class="note-detail-block">
            <strong>验证计划</strong>
            <p>${escapeHtml(item.verificationPlan)}</p>
          </section>
          <section class="note-detail-block">
            <strong>指标字段</strong>
            <p>${escapeHtml(item.metrics.join(" / "))}</p>
          </section>
          <section class="note-detail-block">
            <strong>实际运行指标</strong>
            ${renderMetricPairs(item.runMetrics)}
          </section>
          <section class="note-detail-block">
            <strong>结果文件</strong>
            ${renderResultFiles(item)}
          </section>
          <section class="note-detail-block output-block">
            <strong>说明</strong>
            <p>${escapeHtml(item.notes)}</p>
          </section>
        </div>

        <div class="note-actions">
          <a href="${pdfHref(pdf)}" target="_blank" rel="noopener">打开 PDF</a>
          <a href="reading_report.html#paper-${escapeHtml(item.id)}">查看完整报告</a>
          <a href="reproduction_report.html#repro-${escapeHtml(item.id)}">查看复现结果</a>
          <button type="button" class="note-toggle" data-repro-toggle="${item.priority}" aria-expanded="false" aria-controls="repro-detail-${item.priority}">展开复现字段</button>
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

function scrollToRepro(priority) {
  switchView("repro");
  activeReproLevel = "all";
  reproQuery = "";
  const search = byId("reproSearchInput");
  if (search) search.value = "";
  renderReproLevelFilters();
  renderReproCards();
  requestAnimationFrame(() => {
    const card = byId(`repro-card-${priority}`);
    if (!card) return;
    card.classList.add("expanded");
    const button = card.querySelector("[data-repro-toggle]");
    if (button) {
      button.setAttribute("aria-expanded", "true");
      button.textContent = "收起复现字段";
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

function bindDashboardEvents() {
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.addEventListener("click", () => switchView(item.dataset.view));
  });

  byId("searchInput").addEventListener("input", (event) => {
    query = event.target.value;
    renderPaperRows();
    switchView("papers");
  });

  byId("trackFilters").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-filter]");
    if (!button) return;
    activeTrack = button.dataset.filter;
    renderTrackFilters();
    renderPaperRows();
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

  byId("reproLevelFilters").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-repro-level]");
    if (!button) return;
    activeReproLevel = button.dataset.reproLevel;
    renderReproLevelFilters();
    renderReproCards();
  });

  byId("reproSearchInput").addEventListener("input", (event) => {
    reproQuery = event.target.value;
    renderReproCards();
  });

  byId("reproCards").addEventListener("click", (event) => {
    const toggle = event.target.closest("button[data-repro-toggle]");
    if (toggle) {
      const card = byId(`repro-card-${toggle.dataset.reproToggle}`);
      const expanded = card.classList.toggle("expanded");
      toggle.setAttribute("aria-expanded", String(expanded));
      toggle.textContent = expanded ? "收起复现字段" : "展开复现字段";
    }
  });

  byId("reproBatches").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-scroll-repro]");
    if (button) scrollToRepro(button.dataset.scrollRepro);
  });
}

function handleNoteHashOrQuery() {
  const params = new URLSearchParams(window.location.search);
  const noteId = params.get("note");
  if (noteId) {
    const note = paperNotesV2.find((item) => item.id === noteId);
    if (note) {
      scrollToNote(note.priority);
      return;
    }
  }

  if (window.location.hash === "#notes") {
    switchView("notes");
  }

  if (window.location.hash === "#repro") {
    switchView("repro");
  }
}

function init() {
  renderThesis();
  renderMetrics();
  renderSiteVersion();
  renderTrackOverview();
  renderTimeline();
  renderTrackFilters();
  renderPaperRows();
  renderTrackDetails();
  renderReadingList();
  renderNoteMainlines();
  renderNoteReadingOrder();
  renderNoteReadingStandard();
  renderThemeFilters();
  renderNotes();
  renderReproSummary();
  renderReproScoring();
  renderReproBatches();
  renderReproLevelFilters();
  renderReproCards();
  bindDashboardEvents();
  handleNoteHashOrQuery();
}

document.addEventListener("DOMContentLoaded", () => {
  if (document.getElementById("metrics")) {
    init();
  }
});

})();
