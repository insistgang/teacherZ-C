export function reproductionPromotionCountFailures(reproAssessments, env = process.env, options = {}) {
  const label = options.label || "dashboard";
  if (!Array.isArray(reproAssessments)) return [`${label} reproAssessments 不是数组，无法检查复现等级晋升`];

  const paperLikeCount = reproAssessments.filter((item) => item.reproductionLevel === "paper-like").length;
  const paperLevelLevelCount = reproAssessments.filter((item) => item.reproductionLevel === "paper-level").length;
  const paperLevelTruthCount = reproAssessments.filter(
    (item) => item.reproductionTruthLevel === "paper-level-completed"
  ).length;

  const failures = [];
  if (env.ALLOW_PAPER_LIKE !== "1" && paperLikeCount !== 0) {
    failures.push(`${label} paper-like 当前必须为 0，实际为 ${paperLikeCount}`);
  }
  if (env.ALLOW_PAPER_LEVEL !== "1" && paperLevelTruthCount !== 0) {
    failures.push(`${label} paper-level-completed 当前必须为 0，实际为 ${paperLevelTruthCount}`);
  }
  if (env.ALLOW_PAPER_LEVEL !== "1" && paperLevelLevelCount !== 0) {
    failures.push(`${label} paper-level 当前必须为 0，实际为 ${paperLevelLevelCount}`);
  }
  return failures;
}
