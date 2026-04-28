import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { execFileSync } from "node:child_process";

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
  const { papers, paperNotesV2, reproAssessments } = data;
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

  check(Array.isArray(reproAssessments) && reproAssessments.length === 15, "reproAssessments 长度不是 15");
  if (Array.isArray(reproAssessments)) {
    const noteIdSet = new Set(noteIds);
    const reproIds = reproAssessments.map((item) => item.id);
    check(new Set(reproIds).size === reproIds.length, "reproAssessment.id 不唯一");

    reproAssessments.forEach((item) => {
      check(noteIdSet.has(item.id), `${item.id} 无法匹配 paperNotesV2.id`);
      check(Number.isInteger(item.difficultyScore) && item.difficultyScore >= 1 && item.difficultyScore <= 5, `${item.id} difficultyScore 不在 1-5`);
      check(Number.isInteger(item.effectScore) && item.effectScore >= 1 && item.effectScore <= 5, `${item.id} effectScore 不在 1-5`);
      check(typeof item.minimalExperiment === "string" && item.minimalExperiment.trim().length > 0, `${item.id} minimalExperiment 为空`);
      check(Array.isArray(item.metrics), `${item.id} metrics 不是数组`);
      check(item.resultStatus, `${item.id} 缺少 resultStatus`);
      if (item.resultStatus === "completed") {
        check(Array.isArray(item.resultFiles) && item.resultFiles.length > 0, `${item.id} completed 但没有 resultFiles`);
        (item.resultFiles || []).forEach((file) => {
          check(fs.existsSync(path.join(docsDir, file)), `${item.id} resultFile 不存在：${file}`);
        });
      }
      if (item.resultStatus === "skipped") {
        check(Boolean(item.skipped_reason || item.skippedReason), `${item.id} skipped 但缺少 skipped_reason`);
      }
    });
  }
}

const indexHtml = readText(path.join(docsDir, "index.html"));
const reportHtml = readText(path.join(docsDir, "reading_report.html"));
const reproductionReportPath = path.join(docsDir, "reproduction_report.html");
const reproductionReportHtml = fs.existsSync(reproductionReportPath) ? readText(reproductionReportPath) : "";
const styleCss = readText(path.join(docsDir, "style.css"));
const dashboardJs = readText(path.join(jsDir, "dashboard.js"));
const oldReportName = ["agent_team_reading_report", ".md"].join("");

check(!indexHtml.includes(oldReportName), "index.html 仍直接指向旧 Markdown 完整报告");
check(fs.existsSync(reproductionReportPath), "docs/reproduction_report.html 不存在");
check(indexHtml.includes("复现评估"), "index.html 缺少“复现评估”入口");
check(indexHtml.includes('src="js/shared.js"'), "index.html 未加载 js/shared.js");
check(indexHtml.includes('src="js/reading-data.js"'), "index.html 未加载 js/reading-data.js");
check(indexHtml.includes('src="js/dashboard.js"'), "index.html 未加载 js/dashboard.js");
check(reportHtml.includes('src="js/shared.js"'), "reading_report.html 未加载 js/shared.js");
check(reportHtml.includes('src="js/reading-data.js"'), "reading_report.html 未加载 js/reading-data.js");
check(reportHtml.includes('src="js/report.js"'), "reading_report.html 未加载 js/report.js");
check(reproductionReportHtml.includes('src="js/shared.js"'), "reproduction_report.html 未加载 js/shared.js");
check(reproductionReportHtml.includes('src="js/reading-data.js"'), "reproduction_report.html 未加载 js/reading-data.js");
check(reproductionReportHtml.includes('src="js/reproduction.js"'), "reproduction_report.html 未加载 js/reproduction.js");
check(!/src=["']app\.js["']/.test(indexHtml + reportHtml + reproductionReportHtml), "HTML 仍加载根目录 app.js");
check(!/src=["']report\.js["']/.test(indexHtml + reportHtml + reproductionReportHtml), "HTML 仍加载根目录 report.js");
check(fs.existsSync(path.join(repoRoot, "reproduce", "results", "repro_results.json")), "reproduce/results/repro_results.json 不存在");
check(fs.existsSync(path.join(repoRoot, "reproduce", "results", "repro_results.csv")), "reproduce/results/repro_results.csv 不存在");

check(styleCss.includes("@media (max-width: 1100px)"), "style.css 缺少 @media (max-width: 1100px)");
check(styleCss.includes("@media (max-width: 900px)"), "style.css 缺少 @media (max-width: 900px)");
check(styleCss.includes("@media (max-width: 640px)"), "style.css 缺少 @media (max-width: 640px)");
if (styleCss.includes("color-mix(")) {
  check(styleCss.includes("@supports not"), "style.css 使用 color-mix 但缺少 @supports not fallback");
}
check(styleCss.includes("focus-visible"), "style.css 缺少 focus-visible 样式");
check(dashboardJs.includes("aria-expanded"), "dashboard 展开按钮缺少 aria-expanded");
check(dashboardJs.includes("aria-controls"), "dashboard 展开按钮缺少 aria-controls");
check(/setAttribute\(["']aria-expanded["']/.test(dashboardJs), "dashboard 展开/收起未同步 aria-expanded");

const trackedLegacyScripts = execFileSync("git", ["ls-files", "docs/app.js", "docs/report.js"], {
  cwd: repoRoot,
  encoding: "utf8"
}).trim();
check(!trackedLegacyScripts, `根目录 legacy 脚本仍被 git 跟踪：${trackedLegacyScripts}`);
check(!fs.existsSync(path.join(docsDir, "app.js")), "docs/app.js 仍存在");
check(!fs.existsSync(path.join(docsDir, "report.js")), "docs/report.js 仍存在");

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

console.log("validate passed: papers=15, notes=15, reproAssessments=15, PDFs ok, links ok, old refs clean");
