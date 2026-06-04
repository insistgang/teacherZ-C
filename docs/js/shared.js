(function () {
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

  function data() {
    return window.ZX_READING_DATA || {};
  }

  function pdfHref(file, sourceData = data()) {
    return encodeURI((sourceData.basePath || "") + file);
  }

  function themeLabel(key, sourceData = data()) {
    return (sourceData.noteThemes || []).find((theme) => theme.key === key)?.label || key;
  }

  function createPaperMaps(sourceData = data()) {
    const paperByPriority = new Map((sourceData.papers || []).map((paper) => [paper.priority, paper]));
    const noteByPriority = new Map((sourceData.paperNotesV2 || []).map((note) => [note.priority, note]));
    const noteById = new Map((sourceData.paperNotesV2 || []).map((note) => [note.id, note]));
    return { paperByPriority, noteByPriority, noteById };
  }

  function notePdf(note, sourceData = data()) {
    const { paperByPriority } = createPaperMaps(sourceData);
    const paper = paperByPriority.get(note.priority);
    return note.pdf || paper?.file || "";
  }

  function asList(items, className = "") {
    const classAttr = className ? ' class="' + escapeHtml(className) + '"' : "";
    return "<ul" + classAttr + ">" + items.map((item) => "<li>" + escapeHtml(item) + "</li>").join("") + "</ul>";
  }

  function asNumberedList(items, className = "") {
    const classAttr = className ? ' class="' + escapeHtml(className) + '"' : "";
    return "<ol" + classAttr + ">" + items.map((item) => "<li>" + escapeHtml(item) + "</li>").join("") + "</ol>";
  }

  function scoreDots(score, label) {
    const dots = Array.from({ length: 5 }, (_, index) => (
      `<span class="${index < score ? "on" : ""}"></span>`
    )).join("");
    return `<div class="score-dots" aria-label="${escapeHtml(label)} ${score} / 5">${dots}<strong>${score}/5</strong></div>`;
  }

  function metricPairs(metrics) {
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

  function resultFiles(item) {
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

  window.ZX_SHARED = {
    byId,
    escapeHtml,
    pdfHref,
    themeLabel,
    createPaperMaps,
    notePdf,
    asList,
    asNumberedList,
    scoreDots,
    metricPairs,
    resultFiles
  };
})();
