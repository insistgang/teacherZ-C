import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { reproductionPromotionCountFailures } from "../../docs/scripts/repro-promotion-guards.mjs";

const baseAssessment = {
  id: "iterated-rof",
  reproductionLevel: "partial",
  reproductionTruthLevel: "partial-completed"
};

test("promotion guard rejects paper-like levels unless explicitly allowed", () => {
  const failures = reproductionPromotionCountFailures(
    [
      baseAssessment,
      {
        ...baseAssessment,
        id: "future-paper-like",
        reproductionLevel: "paper-like",
        reproductionTruthLevel: "partial-completed"
      }
    ],
    {}
  );

  assert.ok(failures.some((failure) => failure.includes("dashboard paper-like 当前必须为 0")));
  assert.deepEqual(
    reproductionPromotionCountFailures(
      [
        {
          ...baseAssessment,
          reproductionLevel: "paper-like",
          reproductionTruthLevel: "partial-completed"
        }
      ],
      { ALLOW_PAPER_LIKE: "1" }
    ),
    []
  );
});

test("promotion guard rejects paper-level levels unless explicitly allowed", () => {
  const failures = reproductionPromotionCountFailures(
    [
      {
        ...baseAssessment,
        reproductionLevel: "paper-level",
        reproductionTruthLevel: "paper-level-completed"
      }
    ],
    {}
  );

  assert.ok(failures.some((failure) => failure.includes("dashboard paper-level-completed 当前必须为 0")));
  assert.ok(failures.some((failure) => failure.includes("dashboard paper-level 当前必须为 0")));
  assert.deepEqual(
    reproductionPromotionCountFailures(
      [
        {
          ...baseAssessment,
          reproductionLevel: "paper-level",
          reproductionTruthLevel: "paper-level-completed"
        }
      ],
      { ALLOW_PAPER_LEVEL: "1" }
    ),
    []
  );
});

test("promotion guard labels run-result promotion failures separately", () => {
  const failures = reproductionPromotionCountFailures(
    [
      {
        id: "iterated-rof",
        reproductionLevel: "paper-like"
      }
    ],
    {},
    { label: "run result" }
  );

  assert.ok(failures.some((failure) => failure.includes("run result paper-like 当前必须为 0")));
});

test("validate CLI rejects forged paper-like run results even when promotion count is allowed", () => {
  const sourceResults = JSON.parse(fs.readFileSync("docs/assets/repro/repro_results.json", "utf8"));
  // Forge a SHALLOW paper-like: declare paper-like but strip the data-backed gate/verification
  // evidence (iterated-rof may already be a legitimately promoted paper-like in the source).
  const forgedResults = sourceResults.map((item) => {
    if (item.id !== "iterated-rof") return item;
    const { paper_like_gate, paper_like_verification, ...rest } = item;
    return { ...rest, reproductionLevel: "paper-like" };
  });
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "teacherz-validate-forged-repro-"));
  const fixturePath = path.join(tempDir, "repro_results.json");
  fs.writeFileSync(fixturePath, `${JSON.stringify(forgedResults, null, 2)}\n`);

  try {
    const rejected = spawnSync(process.execPath, ["docs/scripts/validate.mjs"], {
      cwd: process.cwd(),
      encoding: "utf8",
      env: {
        ...process.env,
        TEACHERZ_VALIDATE_REPRO_RESULTS_PATH: fixturePath
      }
    });
    assert.notEqual(rejected.status, 0);
    assert.match(`${rejected.stdout}\n${rejected.stderr}`, /run result paper-like 当前必须为 0/);

    const allowedCountOnly = spawnSync(process.execPath, ["docs/scripts/validate.mjs"], {
      cwd: process.cwd(),
      encoding: "utf8",
      env: {
        ...process.env,
        ALLOW_PAPER_LIKE: "1",
        TEACHERZ_VALIDATE_REPRO_RESULTS_PATH: fixturePath
      }
    });
    assert.notEqual(allowedCountOnly.status, 0);
    assert.match(
      `${allowedCountOnly.stdout}\n${allowedCountOnly.stderr}`,
      /paper-like result requires a complete recomputed paper_like_gate checklist/
    );
  } finally {
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
});
