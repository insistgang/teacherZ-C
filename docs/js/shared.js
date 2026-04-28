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

  window.ZX_SHARED = {
    byId,
    escapeHtml,
    pdfHref,
    themeLabel,
    createPaperMaps,
    notePdf,
    asList,
    asNumberedList
  };
})();
