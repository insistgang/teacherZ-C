import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { execFileSync } from "node:child_process";

const repoRoot = path.resolve(process.cwd());
const docsDir = path.join(repoRoot, "docs");
const jsDir = path.join(docsDir, "js");
const notesDir = path.join(repoRoot, "xiaohao_cai_ultimate_notes");
const reproAssetResultsPath = path.join(docsDir, "assets", "repro", "repro_results.json");

const expectedNoteFiles = [
  "Framelet_Based_Tubular_Structures_超精读笔记_已填充.md",
  "High-Dimensional_Inverse_Problems_UQ_超精读笔记_已填充.md",
  "Mumford-Shah_and_ROF_Linkage_超精读笔记_已填充.md",
  "Multiclass_Segmentation_Iterated_ROF_超精读笔记_已填充.md",
  "Online_Radio_Interferometric_Imaging_超精读笔记_已填充.md",
  "Proximal_Nested_Sampling_超精读笔记_已填充.md",
  "Radio_Interferometric_Imaging_I_超精读笔记_已填充.md",
  "Radio_Interferometric_Imaging_II_超精读笔记_已填充.md",
  "SLaT_Three-stage_Segmentation_超精读笔记_已填充.md",
  "Tight_Frame_Vessel_Segmentation_超精读笔记_已填充.md",
  "Two-Stage_Classification_Point_Clouds_超精读笔记_已填充.md",
  "Variational_Segmentation-Restoration_超精读笔记_已填充.md",
  "Wavelet_Segmentation_on_Sphere_超精读笔记_已填充.md",
  "分割方法论总览_SaT_Segmentation_Overview_超精读笔记_已填充.md",
  "高效变分分类方法_Efficient_Variational_Classification_超精读笔记_已填充.md"
];

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
  check(expectedNoteFiles.length === 15, "独立 Markdown 笔记文件口径不是 15");
  expectedNoteFiles.forEach((file) => {
    check(fs.existsSync(path.join(notesDir, file)), `精读笔记文件不存在：${file}`);
  });
  const actualNoteFiles = fs.readdirSync(notesDir).filter((file) => file.endsWith(".md")).sort();
  const sortedExpectedNoteFiles = [...expectedNoteFiles].sort();
  check(actualNoteFiles.length === sortedExpectedNoteFiles.length, `独立 Markdown 笔记文件数量不是 15，实际为 ${actualNoteFiles.length}`);
  check(actualNoteFiles.every((file, index) => file === sortedExpectedNoteFiles[index]), "独立 Markdown 笔记目录存在非 15 篇口径文件");
  expectedNoteFiles.forEach((file) => {
    const noteText = readText(path.join(notesDir, file));
    check(noteText.includes("第一作者核验"), `精读笔记缺少第一作者核验字段：${file}`);
  });

  const priorities = papers.map((paper) => paper.priority);
  check(new Set(priorities).size === priorities.length, "paper priority 不唯一");
  check(priorities.every((priority) => Number.isInteger(priority) && priority >= 1 && priority <= 15), "paper priority 不在 1-15");
  papers.forEach((paper) => {
    check(typeof paper.authors === "string" && paper.authors.startsWith("Xiaohao Cai"), `${paper.title} 作者顺序未以 Xiaohao Cai 开头`);
  });

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
    const allowedTruthLevels = new Set(["toy-completed", "partial-completed", "paper-level-completed", "assessment-only"]);
    const paperLevelCount = reproAssessments.filter((item) => item.reproductionTruthLevel === "paper-level-completed").length;
    const resultsPath = path.join(repoRoot, "reproduce", "results", "repro_results.json");
    const runResults = fs.existsSync(resultsPath)
      ? JSON.parse(readText(resultsPath))
      : (fs.existsSync(reproAssetResultsPath) ? JSON.parse(readText(reproAssetResultsPath)) : []);
    const runResultById = new Map(runResults.map((item) => [item.id, item]));
    const satSource = readText(path.join(repoRoot, "reproduce", "experiments", "sat_rof_trof.py"));
    check(new Set(reproIds).size === reproIds.length, "reproAssessment.id 不唯一");
    check(process.env.ALLOW_PAPER_LEVEL === "1" || paperLevelCount === 0, `paper-level-completed 当前必须为 0，实际为 ${paperLevelCount}`);
    check(Array.isArray(runResults) && runResults.length === 15, "静态复现实验结果不是 15 条");
    check(runResults.every((item) => noteIdSet.has(item.id)), "静态复现实验结果存在 15 篇之外的 id");

    reproAssessments.forEach((item) => {
      check(noteIdSet.has(item.id), `${item.id} 无法匹配 paperNotesV2.id`);
      check(allowedTruthLevels.has(item.reproductionTruthLevel), `${item.id} reproductionTruthLevel 无效或缺失`);
      check(Number.isInteger(item.difficultyScore) && item.difficultyScore >= 1 && item.difficultyScore <= 5, `${item.id} difficultyScore 不在 1-5`);
      check(Number.isInteger(item.effectScore) && item.effectScore >= 1 && item.effectScore <= 5, `${item.id} effectScore 不在 1-5`);
      check(typeof item.minimalExperiment === "string" && item.minimalExperiment.trim().length > 0, `${item.id} minimalExperiment 为空`);
      check(Array.isArray(item.metrics), `${item.id} metrics 不是数组`);
      check(item.resultStatus, `${item.id} 缺少 resultStatus`);
      const notesLower = String(item.notes || "").toLowerCase();
      if (String(item.reproductionTruthLevel || "").includes("toy")) {
        check(notesLower.includes("toy") || notesLower.includes("synthetic"), `${item.id} truthLevel 含 toy 但 notes 未说明 toy/synthetic`);
      }
      if (item.resultStatus === "completed") {
        check(Array.isArray(item.resultFiles) && item.resultFiles.length > 0, `${item.id} completed 但没有 resultFiles`);
        check(Boolean(item.resultQuality || item.notes), `${item.id} completed 但缺少 resultQuality 或 notes`);
        (item.resultFiles || []).forEach((file) => {
          check(fs.existsSync(path.join(docsDir, file)), `${item.id} resultFile 不存在：${file}`);
        });
      }
      if (item.resultStatus === "skipped") {
        check(Boolean(item.skipped_reason || item.skippedReason), `${item.id} skipped 但缺少 skipped_reason`);
      }
      if (item.id === "proximal-nested-sampling") {
        const absoluteLogError = Number(item.runMetrics?.absolute_log_error ?? runResultById.get(item.id)?.metrics?.absolute_log_error ?? 0);
        if (absoluteLogError > 1) {
          check(Boolean(item.warning), "nested_sampling_toy absolute_log_error > 1 但缺少 warning");
        }
      }
      if (item.id === "slat-color") {
        const accuracyGain = Number(item.runMetrics?.accuracy_gain ?? runResultById.get(item.id)?.metrics?.accuracy_gain ?? 0);
        if (accuracyGain < 0.02) {
          check(item.effectScore !== 5, "SLaT accuracy_gain < 0.02 时 effectScore 不能为 5");
        }
      }
      if (["sat-overview", "pcms-rof-linkage", "iterated-rof"].includes(item.id) && satSource.includes("gaussian_filter")) {
        check(notesLower.includes("proxy smoothing") || (notesLower.includes("gaussian") && notesLower.includes("proxy")), `${item.id} 使用 gaussian_filter 但 notes 未说明 proxy smoothing / Gaussian proxy`);
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
const runAllSource = readText(path.join(repoRoot, "reproduce", "run_all.py"));
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
check(fs.existsSync(reproAssetResultsPath), "docs/assets/repro/repro_results.json 不存在");

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
check(!runAllSource.includes("variational_classification"), "run_all.py 仍包含会产生重复分类结果的 variational_classification runner");
check((runAllSource.match(/^\s+\w+\.run,/gm) || []).length === 9, "run_all.py runner 数量不是 9");

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

console.log("validate passed: papers=15, firstAuthorPapers=15, firstAuthorNotes=15, structuredNotes=15, markdownNotes=15, reproAssessments=15, PDFs ok, links ok, old refs clean");
