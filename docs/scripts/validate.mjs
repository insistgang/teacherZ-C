import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const repoRoot = path.resolve(process.cwd());
const docsDir = path.join(repoRoot, "docs");
const jsDir = path.join(docsDir, "js");

const failures = [];

function check(condition, message) {
  if (!condition) failures.push(message);
}

function readText(file) {
  return fs.readFileSync(file, "utf8");
}

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) return walk(fullPath);
    return [fullPath];
  });
}

const context = { window: {}, console };
vm.runInNewContext(readText(path.join(jsDir, "reading-data.js")), context, {
  filename: "reading-data.js"
});

const data = context.window.ZX_READING_DATA;
check(Boolean(data), "window.ZX_READING_DATA 未导出");

if (data) {
  const { papers, paperNotesV2 } = data;
  check(Array.isArray(papers) && papers.length === 15, "papers 长度不是 15");
  check(Array.isArray(paperNotesV2) && paperNotesV2.length === 15, "paperNotesV2 长度不是 15");

  const priorities = papers.map((paper) => paper.priority);
  check(new Set(priorities).size === priorities.length, "paper priority 不唯一");

  const paperPrioritySet = new Set(priorities);
  const notePriorities = paperNotesV2.map((note) => note.priority);
  check(notePriorities.every((priority) => paperPrioritySet.has(priority)), "note.priority 存在无法匹配的 paper.priority");

  const noteIds = paperNotesV2.map((note) => note.id);
  check(new Set(noteIds).size === noteIds.length, "note.id 不唯一");

  const notePrioritySet = new Set(notePriorities);
  paperNotesV2.forEach((note) => {
    const missingRelation = (note.relation?.links || []).filter((priority) => !notePrioritySet.has(priority));
    check(missingRelation.length === 0, `${note.id} 的 relation.links 存在无效目标：${missingRelation.join(", ")}`);
    check(note.reportExpansion && typeof note.reportExpansion === "object", `${note.id} 缺少 reportExpansion`);
    check(Array.isArray(note.evidence) && note.evidence.length > 0, `${note.id} 缺少 evidence`);
    check(Array.isArray(note.readingQuestions) && note.readingQuestions.length >= 3, `${note.id} readingQuestions 少于 3 个`);
  });

  papers.forEach((paper) => {
    const pdfPath = path.join(docsDir, data.basePath, paper.file);
    check(fs.existsSync(pdfPath), `PDF 不存在：${paper.file}`);
  });
}

const indexHtml = readText(path.join(docsDir, "index.html"));
const reportHtml = readText(path.join(docsDir, "reading_report.html"));
const oldReportName = ["agent_team_reading_report", ".md"].join("");

check(!indexHtml.includes(oldReportName), "index.html 仍直接指向旧 Markdown 完整报告");
check(indexHtml.includes('src="js/shared.js"'), "index.html 未加载 js/shared.js");
check(indexHtml.includes('src="js/reading-data.js"'), "index.html 未加载 js/reading-data.js");
check(indexHtml.includes('src="js/dashboard.js"'), "index.html 未加载 js/dashboard.js");
check(reportHtml.includes('src="js/shared.js"'), "reading_report.html 未加载 js/shared.js");
check(reportHtml.includes('src="js/reading-data.js"'), "reading_report.html 未加载 js/reading-data.js");
check(reportHtml.includes('src="js/report.js"'), "reading_report.html 未加载 js/report.js");

const forbiddenRefs = [
  { label: "standalone legacy data script", test: (text) => /(^|[/"'\s])data\.js($|[?"'\s])/.test(text) },
  { label: "legacy papers global", test: (text) => text.includes(["PAPERS", "_DATA"].join("")) },
  { label: "legacy paper directory", test: (text) => text.includes(["00_papers", "/"].join("")) },
  { label: "legacy notes directory", test: (text) => text.includes(["docs", "/notes"].join("")) },
  { label: "legacy all directory", test: (text) => text.includes(["docs", "/all"].join("")) }
];

const textExts = new Set([".html", ".js", ".css", ".md", ".mjs", ".tsv", ".txt", ".csv"]);
const textFiles = walk(docsDir).filter((file) => textExts.has(path.extname(file)));
textFiles.forEach((file) => {
  const rel = path.relative(repoRoot, file).replaceAll(path.sep, "/");
  const text = readText(file);
  forbiddenRefs.forEach((ref) => {
    check(!ref.test(text), `${rel} 包含旧引用：${ref.label}`);
  });
});

if (failures.length) {
  console.error("validate failed:");
  failures.forEach((failure) => console.error(`- ${failure}`));
  process.exit(1);
}

console.log("validate passed: papers=15, notes=15, PDFs ok, links ok, old refs clean");
