import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";

const repoRoot = path.resolve(process.cwd());
const readingDataPath = path.join(repoRoot, "docs", "js", "reading-data.js");
const defaultResultsPath = path.join(repoRoot, "reproduce", "results", "repro_results.json");
const assetResultsPath = path.join(repoRoot, "docs", "assets", "repro", "repro_results.json");

const args = new Set(process.argv.slice(2));
const checkOnly = args.has("--check");
const quiet = args.has("--quiet");

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function loadDashboardData() {
  const context = { window: {}, console };
  vm.runInNewContext(fs.readFileSync(readingDataPath, "utf8"), context, {
    filename: readingDataPath
  });
  return context.window.ZX_READING_DATA;
}

function normalizeJson(value) {
  if (Array.isArray(value)) return value.map(normalizeJson);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
        .map(([key, nestedValue]) => [key, normalizeJson(nestedValue)])
    );
  }
  return value ?? null;
}

function sameJson(left, right) {
  return JSON.stringify(normalizeJson(left)) === JSON.stringify(normalizeJson(right));
}

function normalizeFiles(files) {
  return [...(files || [])].sort();
}

function compareDashboardToResults(data, runResults) {
  const differences = [];
  const resultById = new Map(runResults.map((item) => [item.id, item]));

  for (const item of data.reproAssessments || []) {
    const result = resultById.get(item.id);
    if (!result) {
      differences.push(`${item.id}: missing run result`);
      continue;
    }

    const checks = [
      ["experimentId", item.experimentId, result.experiment_id],
      ["reproductionLevel", item.reproductionLevel, result.reproductionLevel],
      ["resultStatus", item.resultStatus, result.status],
      ["runtimeSeconds", item.runtimeSeconds, result.runtime_seconds],
      ["runMetrics", item.runMetrics || {}, result.metrics || {}],
      ["resultFiles", normalizeFiles(item.resultFiles), normalizeFiles(result.resultFiles)],
      ["notes", item.notes || "", result.notes || ""]
    ];

    if (result.status === "skipped") {
      checks.push([
        "skippedReason",
        item.skippedReason || item.skipped_reason || "",
        result.skipped_reason || ""
      ]);
    }

    for (const field of ["resultQuality", "warning"]) {
      if (result[field] !== undefined) {
        checks.push([field, item[field], result[field]]);
      }
    }

    checks.forEach(([field, dashboardValue, resultValue]) => {
      if (!sameJson(dashboardValue, resultValue)) {
        differences.push(`${item.id}.${field}: dashboard=${JSON.stringify(dashboardValue)} result=${JSON.stringify(resultValue)}`);
      }
    });
  }

  const dashboardIds = new Set((data.reproAssessments || []).map((item) => item.id));
  runResults.forEach((result) => {
    if (!dashboardIds.has(result.id)) differences.push(`${result.id}: run result has no dashboard assessment`);
  });

  return differences;
}

const resultsPath = fs.existsSync(defaultResultsPath) ? defaultResultsPath : assetResultsPath;
const data = loadDashboardData();
const runResults = readJson(resultsPath);
const differences = compareDashboardToResults(data, runResults);

if (!quiet) {
  console.log(`checked ${runResults.length} run results against docs/js/reading-data.js`);
}

if (differences.length) {
  if (!quiet) {
    console.error("dashboard repro fields are out of sync:");
    differences.forEach((item) => console.error(`- ${item}`));
  }
  if (checkOnly) process.exit(1);
}

if (!quiet && !differences.length) {
  console.log("dashboard repro fields match latest run results");
}
