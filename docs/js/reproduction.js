(function () {
  const data = window.ZX_READING_DATA;
  const shared = window.ZX_SHARED;
  if (!data || !shared) {
    console.error("ZX reproduction report dependencies are missing. Load shared.js and reading-data.js before reproduction.js.");
    return;
  }

  const { tracks, papers, reproScoring, reproRecommendedBatches, reproAssessments, siteMeta } = data;
  const { byId, escapeHtml, pdfHref, createPaperMaps, notePdf, scoreDots, metricPairs, resultFiles } = shared;
  const { paperByPriority } = createPaperMaps(data);

  function metricSummary(metrics, maxItems = 4) {
    const entries = Object.entries(metrics || {}).slice(0, maxItems);
    if (!entries.length) return "暂无指标";
    return entries.map(([key, value]) => `${key}: ${value}`).join(" · ");
  }

  function renderVersion() {
    const target = byId("reproReportVersion");
    if (!target || !siteMeta) return;
    target.textContent = `scope: ${siteMeta.scope || siteMeta.commit} · last updated: ${siteMeta.lastUpdated}`;
  }

  function renderToc() {
    const entries = [
      ["repro-positioning", "0. 报告定位"],
      ["repro-overview", "1. 总体结论"],
      ["repro-scoring", "2. 评分系统"],
      ["repro-ranking", "3. 难度与优先级"],
      ["repro-order", "4. 推荐复现顺序"],
      ["repro-results", "5. 实际运行结果"],
      ["repro-figures", "6. 结果图"],
      ["repro-paper-sections", "7. 逐篇评估"],
      ["repro-next", "8. 下一步路线"]
    ];
    byId("reproReportToc").innerHTML = entries.map(([id, label]) => `<a href="#${id}">${escapeHtml(label)}</a>`).join("");
  }

  function renderSummary() {
    const completed = reproAssessments.filter((item) => item.resultStatus === "completed").length;
    const skipped = reproAssessments.filter((item) => item.resultStatus === "skipped").length;
    const failed = reproAssessments.filter((item) => item.resultStatus === "failed").length;
    const paperLevel = reproAssessments.filter((item) => item.reproductionTruthLevel === "paper-level-completed").length;
    const avgDifficulty = reproAssessments.reduce((sum, item) => sum + item.difficultyScore, 0) / reproAssessments.length;
    const avgEffect = reproAssessments.reduce((sum, item) => sum + item.effectScore, 0) / reproAssessments.length;
    const figures = new Set(reproAssessments.flatMap((item) => item.resultFiles || [])).size;
    const cards = [
      { label: "评估对象", value: "15", detail: "Xiaohao Cai 第一作者论文" },
      { label: "运行状态", value: `${completed} completed`, detail: `${skipped} skipped · ${failed} failed` },
      { label: "paper-level", value: `${paperLevel} / ${reproAssessments.length}`, detail: "当前无论文级完整复现" },
      { label: "平均难度", value: avgDifficulty.toFixed(1), detail: "1 低到 5 极高" },
      { label: "平均展示效果", value: avgEffect.toFixed(1), detail: "1 弱到 5 很明显" },
      { label: "结果图", value: figures, detail: "位于 docs/assets/repro/" }
    ];
    byId("reproReportSummary").innerHTML = cards.map((card) => `
      <article class="metric repro-metric">
        <span>${escapeHtml(card.label)}</span>
        <strong>${escapeHtml(card.value)}</strong>
        <small>${escapeHtml(card.detail)}</small>
      </article>
    `).join("");
  }

  function renderTruthAlert() {
    const target = byId("reproReportTruthAlert");
    if (!target) return;
    const paperLevel = reproAssessments.filter((item) => item.reproductionTruthLevel === "paper-level-completed").length;
    target.innerHTML = `
      <strong>复现真实性提示</strong>
      <p>completed 表示 toy/partial 实验跑通，不表示论文级完整复现。当前 paper-level reproduction: ${paperLevel} / ${reproAssessments.length}。</p>
      <div class="truth-legend">
        <span>completed = script ran successfully</span>
        <span>toy-completed = synthetic minimal demo only</span>
        <span>partial-completed = partial algorithmic route demo</span>
        <span>paper-level-completed = close to paper experiments</span>
        <span>current paper-level-completed: ${paperLevel}</span>
      </div>
    `;
  }

  function renderScoring() {
    byId("reproReportDifficultyScoring").innerHTML = (reproScoring?.difficultyDimensions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
    byId("reproReportEffectScoring").innerHTML = (reproScoring?.effectDimensions || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  }

  function renderRankings() {
    const hardest = [...reproAssessments].sort((a, b) => b.difficultyScore - a.difficultyScore || a.priority - b.priority).slice(0, 5);
    const quickWins = [...reproAssessments].sort((a, b) => (a.difficultyScore - b.difficultyScore) || (b.effectScore - a.effectScore)).slice(0, 5);
    const unsuitable = [...reproAssessments].sort((a, b) => b.difficultyScore - a.difficultyScore || b.implementationRisk.length - a.implementationRisk.length).slice(0, 5);

    const blocks = [
      { title: "难度最高的 5 篇", items: hardest, detail: "这些论文通常依赖真实数据、专门算子、MCMC / nested sampling 或球面小波栈。" },
      { title: "最适合先复现的 5 篇", items: quickWins, detail: "这些实验能较快得到可视化结果，适合作为复现入口。" },
      { title: "最不适合短期 full reproduction 的 5 篇", items: unsuitable, detail: "短期建议只做 toy，长期再补真实数据、专门库和严格验证。" }
    ];

    byId("reproReportRankings").innerHTML = blocks.map((block) => `
      <article class="workbench-block">
        <h3>${escapeHtml(block.title)}</h3>
        <p>${escapeHtml(block.detail)}</p>
        <ol class="ranked-list">
          ${block.items.map((item) => `
            <li>
              <a href="#repro-${escapeHtml(item.id)}">#${item.priority} ${escapeHtml(item.titleCn)}</a>
              <span>难度 ${item.difficultyScore}/5 · 效果 ${item.effectScore}/5 · ${escapeHtml(item.reproductionLevel)}</span>
            </li>
          `).join("")}
        </ol>
      </article>
    `).join("");
  }

  function renderBatches() {
    byId("reproReportBatches").innerHTML = reproRecommendedBatches.map((batch, index) => `
      <article class="repro-batch">
        <h3><span>${String(index + 1).padStart(2, "0")}</span>${escapeHtml(batch.title)}</h3>
        <p>${escapeHtml(batch.reason)}</p>
        <div class="relation-chips">
          ${batch.priorities.map((priority) => {
            const item = reproAssessments.find((entry) => entry.priority === priority);
            return `<a class="relation-chip" href="#repro-${escapeHtml(item.id)}">#${priority} ${escapeHtml(item.titleCn)}</a>`;
          }).join("")}
        </div>
      </article>
    `).join("");
  }

  function renderResultsTable() {
    byId("reproReportResultsTable").innerHTML = reproAssessments.map((item) => `
      <tr>
        <td><span class="rank">${item.priority}</span></td>
        <td>
          <strong>${escapeHtml(item.titleCn)}</strong>
          <small>${escapeHtml(item.titleEn)}</small>
        </td>
        <td>${escapeHtml(item.reproductionLevel)}</td>
        <td>${escapeHtml(item.reproductionTruthLevel)}</td>
        <td><span class="status-pill ${escapeHtml(item.resultStatus)}">${escapeHtml(item.resultStatus)}</span></td>
        <td>${escapeHtml(item.runtimeSeconds ?? "n/a")}s</td>
        <td>${escapeHtml(metricSummary(item.runMetrics))}</td>
      </tr>
    `).join("");
  }

  function renderFigures() {
    const seen = new Set();
    const figures = [];
    reproAssessments.forEach((item) => {
      (item.resultFiles || []).forEach((file) => {
        if (seen.has(file) || !/\.(png|jpg|jpeg|webp)$/i.test(file)) return;
        seen.add(file);
        figures.push({ file, item });
      });
    });

    byId("reproReportFigures").innerHTML = figures.map(({ file, item }) => `
      <a href="${escapeHtml(file)}" target="_blank" rel="noopener">
        <img src="${escapeHtml(file)}" alt="${escapeHtml(item.titleCn)} 复现结果图">
        <span>#${item.priority} ${escapeHtml(item.titleCn)}</span>
      </a>
    `).join("");
  }

  function renderPaperSections() {
    byId("reproReportCards").innerHTML = reproAssessments.map((item) => {
      const paper = paperByPriority.get(item.priority);
      const track = tracks[paper.track];
      const pdf = notePdf(item);
      return `
        <article class="report-paper repro-paper" id="repro-${escapeHtml(item.id)}" style="--accent:${track.color}">
          <header class="report-paper-head">
            <div>
              <p class="eyebrow">Reproduction #${item.priority}</p>
              <h3>${escapeHtml(item.titleCn)}</h3>
              <small>${escapeHtml(item.titleEn)}</small>
            </div>
            <div class="report-paper-actions">
              <a href="${pdfHref(pdf)}" target="_blank" rel="noopener">打开 PDF</a>
              <a href="reading_report.html#paper-${escapeHtml(item.id)}">回到精读报告</a>
              <a href="index.html#repro">回到复现页签</a>
            </div>
          </header>

          <div class="report-meta-row">
            <span>${escapeHtml(item.reproductionLevel)}</span>
            <span>${escapeHtml(item.reproductionTruthLevel)}</span>
            <span>难度 ${item.difficultyScore}/5 · ${escapeHtml(item.difficultyLabel)}</span>
            <span>效果 ${item.effectScore}/5 · ${escapeHtml(item.effectLabel)}</span>
            <span>${escapeHtml(item.resultStatus)}</span>
          </div>

          <section class="report-subsection">
            <h4>复现难度与效果判断</h4>
            <div class="repro-score-grid">
              <section><strong>复现难度</strong>${scoreDots(item.difficultyScore, "复现难度")}</section>
              <section><strong>展示效果</strong>${scoreDots(item.effectScore, "展示效果")}</section>
            </div>
            <p>${escapeHtml(item.fullReproductionFeasibility)}</p>
          </section>

          <section class="report-subsection">
            <h4>最小可复现实验</h4>
            <p>${escapeHtml(item.minimalExperiment)}</p>
            <p>${escapeHtml(item.expectedOutcome)}</p>
          </section>

          <section class="report-subsection">
            <h4>依赖、数据与算力</h4>
            <p><strong>依赖：</strong>${escapeHtml(item.dependencies.join(" / "))}</p>
            <p><strong>数据：</strong>${escapeHtml(item.dataRequirement)}</p>
            <p><strong>算力：</strong>${escapeHtml(item.computeRequirement)}</p>
          </section>

          <section class="report-subsection">
            <h4>实现风险与验证计划</h4>
            <p>${escapeHtml(item.implementationRisk)}</p>
            <p>${escapeHtml(item.verificationPlan)}</p>
          </section>

          <section class="report-subsection">
            <h4>实际运行指标</h4>
            ${metricPairs(item.runMetrics)}
          </section>

          <section class="report-subsection">
            <h4>结果质量与 warning</h4>
            <p><strong>truth level：</strong>${escapeHtml(item.reproductionTruthLevel)}</p>
            ${item.fidelityWarning ? `<p>${escapeHtml(item.fidelityWarning)}</p>` : ""}
            ${item.resultQuality ? `<p><strong>resultQuality：</strong>${escapeHtml(item.resultQuality)}</p>` : ""}
            ${item.warning ? `<p class="warning-text">${escapeHtml(item.warning)}</p>` : ""}
          </section>

          <section class="report-subsection">
            <h4>结果文件</h4>
            ${resultFiles(item)}
          </section>

          <section class="report-subsection">
            <h4>诚实标注</h4>
            <p>${escapeHtml(item.notes)}</p>
          </section>
        </article>
      `;
    }).join("");
  }

  function renderNextSteps() {
    const paragraphs = [
      "第一步应把效果最清楚的 SaT / T-ROF / SLaT toy 变成可反复运行的小 benchmark：固定噪声、模糊、灰度间隔和类别数，系统记录 accuracy、Dice 与运行时间。",
      "重新设计 SLaT toy，使 RGB 通道高度相关但 Lab/chroma 差异明显，以更好展示 lifting 的价值。当前 RGB+Lab-like 只带来 0.0053 的 accuracy gain，不能支撑很强的效果结论。",
      "第二步可以补 framelet/tight-frame 与 graph classification 的更忠实实现：前者替换 Gaussian fallback 为真正 framelet/tight-frame shrinkage，后者补 graph TV proximal 或 primal-dual solver。",
      "第三步再进入 inverse problem：把 MAP-UQ toy 扩展到更真实的 Fourier / MRI undersampling，再做小规模 posterior sampling 校验 HPD approximation 的保守性。",
      "最后才建议推进 spherical wavelet 和 proximal nested sampling 的 full reproduction，因为它们依赖专门库、采样诊断和高维模型选择验证，短期最容易把 toy 结果误读成论文级结果。"
    ];
    byId("reproReportNextSteps").innerHTML = paragraphs.map((paragraph) => `<p>${escapeHtml(paragraph)}</p>`).join("");
  }

  function init() {
    renderVersion();
    renderToc();
    renderSummary();
    renderTruthAlert();
    renderScoring();
    renderRankings();
    renderBatches();
    renderResultsTable();
    renderFigures();
    renderPaperSections();
    renderNextSteps();
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (document.getElementById("reproReportSummary")) init();
  });
})();
