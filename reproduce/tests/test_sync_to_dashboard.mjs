import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import zlib from "node:zlib";

const testRepoRoot = fs.mkdtempSync(path.join(os.tmpdir(), "teacherz-sync-repo-"));
process.env.REPRO_SYNC_REPO_ROOT = testRepoRoot;
process.env.REPRO_SYNC_ALLOW_REPO_ROOT_OVERRIDE = "1";

function copyRepoFixture(relativePath) {
  const sourcePath = path.join(process.cwd(), relativePath);
  const targetPath = path.join(testRepoRoot, relativePath);
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.copyFileSync(sourcePath, targetPath);
}

copyRepoFixture(path.join("docs", "js", "reading-data.js"));
copyRepoFixture(path.join("reproduce", "results", "repro_results.json"));
copyRepoFixture(path.join("docs", "assets", "repro", "repro_results.json"));
copyRepoFixture(path.join("reproduce", "paper_like", "iterated_rof_dataset_sources.json"));

test.after(() => {
  fs.rmSync(testRepoRoot, { recursive: true, force: true });
});

const {
  compareDashboardToResults,
  compareResultAssetSnapshot,
  expectedTruthLevel,
  iteratedRofSourceRegistryDifferences,
  syncSnapshotDifferences,
  validateDashboardCandidate,
  validateDashboardCandidateShape
} = await import(
  `${pathToFileURL(path.join(process.cwd(), "reproduce", "sync_to_dashboard.mjs")).href}?testRepo=${path.basename(testRepoRoot)}`
);

const testReproduceResultsRoot = path.join(testRepoRoot, "reproduce", "results");

const baseResult = {
  priority: 3,
  id: "iterated-rof",
  experiment_id: "sat_rof_trof",
  reproductionLevel: "partial",
  status: "completed",
  runtime_seconds: 0.5,
  metrics: { accuracy: 0.9 },
  resultFiles: ["assets/repro/sat_demo.png"],
  notes: "Partial reproduction on local runner evidence."
};

const baseAssessment = {
  priority: 3,
  id: "iterated-rof",
  experimentId: "sat_rof_trof",
  reproductionLevel: "partial",
  reproductionTruthLevel: "partial-completed",
  resultStatus: "completed",
  runtimeSeconds: 0.7,
  runMetrics: { accuracy: 0.9 },
  resultFiles: ["assets/repro/sat_demo.png"],
  notes: "Partial reproduction on local runner evidence."
};

const requiredPaperLikeFamilies = ["cartoon", "texture", "medical"];
const paperLikeSourceIdByFamily = {
  cartoon: "bsds500",
  texture: "prague-texture",
  medical: "brainweb"
};
const paperLikeSourceUrlByFamily = {
  cartoon: "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/",
  texture: "https://mosaic.utia.cas.cz/",
  medical: "https://brainweb.bic.mni.mcgill.ca/brainweb/"
};
const paperLikeSourceAudit = {
  downloaded_at: "2026-06-09",
  source_artifact_sha256: "1".repeat(64),
  license_snapshot_sha256: "2".repeat(64),
  conversion_notes: "Converted reviewed local source files into canonical PNG image/mask pairs.",
  local_file_mapping_reviewed: true
};

function writeTempSourceRegistry(mutator) {
  const sourceRegistryPath = path.join(testRepoRoot, "reproduce", "paper_like", "iterated_rof_dataset_sources.json");
  const registry = JSON.parse(fs.readFileSync(sourceRegistryPath, "utf8"));
  mutator(registry);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "teacherz-source-registry-"));
  const registryPath = path.join(directory, "iterated_rof_dataset_sources.json");
  fs.writeFileSync(registryPath, `${JSON.stringify(registry, null, 2)}\n`);
  return {
    path: registryPath,
    cleanup: () => fs.rmSync(directory, { recursive: true, force: true })
  };
}

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data = Buffer.alloc(0)) {
  const typeBuffer = Buffer.from(type, "ascii");
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([typeBuffer, data])));
  return Buffer.concat([length, typeBuffer, data, crc]);
}

function pngFixture(width, height, pixelValue = null) {
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // truecolor
  const rows = [];
  for (let y = 0; y < height; y += 1) {
    const row = Buffer.alloc(1 + width * 3);
    row[0] = 0;
    for (let x = 0; x < width; x += 1) {
      const offset = 1 + x * 3;
      if (pixelValue === null) {
        row[offset] = (x * 255) / Math.max(1, width - 1);
        row[offset + 1] = (y * 255) / Math.max(1, height - 1);
        row[offset + 2] = 127;
      } else {
        row[offset] = pixelValue;
        row[offset + 1] = pixelValue;
        row[offset + 2] = pixelValue;
      }
    }
    rows.push(row);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", zlib.deflateSync(Buffer.concat(rows))),
    pngChunk("IEND")
  ]);
}

const paperLikePngFixture = pngFixture(32, 32);
const blankPaperLikePngFixture = pngFixture(32, 32, 0);
const tinyPngFixture = pngFixture(1, 1);

function jpegFixture(width, height) {
  const sof0 = Buffer.from([
    0xff, 0xc0,
    0x00, 0x11,
    0x08,
    (height >> 8) & 0xff, height & 0xff,
    (width >> 8) & 0xff, width & 0xff,
    0x03,
    0x01, 0x11, 0x00,
    0x02, 0x11, 0x00,
    0x03, 0x11, 0x00
  ]);
  return Buffer.concat([
    Buffer.from([0xff, 0xd8]),
    sof0,
    Buffer.from([0xff, 0xd9])
  ]);
}

function bmpFixture(width, height) {
  const rowSize = Math.ceil((width * 3) / 4) * 4;
  const imageSize = rowSize * height;
  const buffer = Buffer.alloc(54 + imageSize);
  buffer.write("BM", 0, "ascii");
  buffer.writeUInt32LE(buffer.length, 2);
  buffer.writeUInt32LE(54, 10);
  buffer.writeUInt32LE(40, 14);
  buffer.writeInt32LE(width, 18);
  buffer.writeInt32LE(height, 22);
  buffer.writeUInt16LE(1, 26);
  buffer.writeUInt16LE(24, 28);
  buffer.writeUInt32LE(0, 30);
  buffer.writeUInt32LE(imageSize, 34);
  return buffer;
}

const tinyJpegFixture = jpegFixture(1, 1);
const tinyBmpFixture = bmpFixture(1, 1);

function sha256Content(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function paperLikeFixtureContent(family, kind) {
  if (["image", "mask", "figure"].includes(kind)) return paperLikePngFixture;
  return Buffer.from(
    [
      `teacherZ Iterated ROF ${kind} review record for ${family}.`,
      `source_url=${paperLikeSourceUrlByFamily[family] || "registered-source-url"}`,
      "review_date=2026-06-09",
      "reviewer_note=Local-only artifact retained to document source page, license terms page, and conversion mapping for this data family.",
      "conversion_note=Images and masks are mapped by same relative path after source review; raw archives remain outside the repository unless redistribution is approved.",
      ""
    ].join("\n")
  );
}

function paperLikeFixtureFileEvidence(family, kind) {
  const content = paperLikeFixtureContent(family, kind);
  return {
    size_bytes: Buffer.byteLength(content),
    sha256: sha256Content(content)
  };
}

function fingerprintFromRecords(records) {
  const digest = crypto.createHash("sha256");
  const sortedRecords = [...records].sort((left, right) => {
    const leftKey = [left.family, left.path, left.kind === "image" ? "0" : "1"].join("\t");
    const rightKey = [right.family, right.path, right.kind === "image" ? "0" : "1"].join("\t");
    return leftKey.localeCompare(rightKey);
  });
  for (const record of sortedRecords) {
    digest.update(`${record.kind}\t${record.family}\t${record.path}\t${record.sha256}\n`);
  }
  return {
    algorithm: "sha256",
    file_count: sortedRecords.length,
    image_count: sortedRecords.filter((record) => record.kind === "image").length,
    mask_count: sortedRecords.filter((record) => record.kind === "mask").length,
    sha256: digest.digest("hex")
  };
}

function updateSummaryFingerprint(summary) {
  summary.dataset_fingerprint = fingerprintFromRecords(
    summary.images.flatMap((item) => [
      {
        kind: "image",
        family: item.family,
        path: item.source_claim.image,
        sha256: item.image_file.sha256
      },
      {
        kind: "mask",
        family: item.family,
        path: item.source_claim.mask,
        sha256: item.mask_file.sha256
      }
    ])
  );
  summary.paper_like_gate.evidence_summary.dataset_fingerprint = summary.dataset_fingerprint;
}

function updateFigureEvidenceSidecar(image) {
  image.figure_evidence = {
    ...image.figure_evidence,
    image_sha256: image.image_file.sha256,
    mask_sha256: image.mask_file.sha256,
    figure_sha256: image.figure_file.sha256,
    figure_size_bytes: image.figure_file.size_bytes
  };
  fs.writeFileSync(image.figure_evidence_path, `${JSON.stringify(image.figure_evidence, null, 2)}\n`);
  const sidecarBytes = fs.readFileSync(image.figure_evidence_path);
  image.figure_evidence_file = {
    path: image.figure_evidence_path,
    size_bytes: sidecarBytes.length,
    sha256: sha256Content(sidecarBytes)
  };
}

const completePaperLikeDatasetFingerprint = fingerprintFromRecords(
  requiredPaperLikeFamilies.flatMap((family) => [
    {
      kind: "image",
      family,
      path: "sample.png",
      sha256: paperLikeFixtureFileEvidence(family, "image").sha256
    },
    {
      kind: "mask",
      family,
      path: "sample.png",
      sha256: paperLikeFixtureFileEvidence(family, "mask").sha256
    }
  ])
);

const completePaperLikeGate = {
  passed: true,
  dashboard_level: "paper-like",
  reasons: [],
  checked_requirements: ["all data families have completed quantitative local runner outputs"],
  checklist: [
    { id: "canonical_data_root", passed: true, reasons: [] },
    { id: "readiness_clean", passed: true, reasons: [] },
    { id: "runner_outputs", passed: true, reasons: [] },
    { id: "output_evidence", passed: true, reasons: [] }
  ],
  evidence_summary: {
    schema_version: 1,
    gate_id: "iterated_rof_paper_like_v1",
    dataset_fingerprint: completePaperLikeDatasetFingerprint,
    image_count: 3,
    completed_image_count: 3,
    quantitative_image_count: 3,
    required_families: requiredPaperLikeFamilies,
    completed_families: requiredPaperLikeFamilies,
    quantitative_families: requiredPaperLikeFamilies,
    source_claim_count: 3,
    figure_evidence_count: 3
  }
};

const completePaperLikeVerification = {
  schema_version: 1,
  generated_by: "iterated_rof_paper_like.dashboard_candidate_v1",
  recomputed_gate: true,
  can_promote: true,
  promotion_shape_blocker_count: 0,
  gate_id: "iterated_rof_paper_like_v1",
  dataset_fingerprint: completePaperLikeGate.evidence_summary.dataset_fingerprint
};

function writeFixtureFile(filePath, content) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content);
  const buffer = fs.readFileSync(filePath);
  return {
    path: filePath,
    size_bytes: buffer.length,
    sha256: crypto.createHash("sha256").update(buffer).digest("hex")
  };
}

function writePaperLikeSummaryFixture(gate = completePaperLikeGate, options = {}) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "teacherz-paper-like-summary-"));
  const token = path.basename(directory);
  const sourceSummaryInRepo = options.sourceSummaryInRepo !== false;
  const summaryPath = sourceSummaryInRepo
    ? path.join(
      testRepoRoot,
      "reproduce",
      "results",
      `iterated_rof_paper_like_summary.${path.basename(directory)}.json`
    )
    : path.join(directory, "summary.json");
  const includeRunnerEvidence = options.includeRunnerEvidence !== false;
  const canonicalLocalEvidence = options.canonicalLocalEvidence !== false;
  const dataRoot = canonicalLocalEvidence
    ? path.join(testRepoRoot, "reproduce", "data", "iterated_rof")
    : path.join(directory, "data");
  const figureRoot = canonicalLocalEvidence
    ? path.join(testRepoRoot, "reproduce", "results", "figures", "iterated_rof_paper_like")
    : path.join(directory, "figures");
  const manifestPath = path.join(dataRoot, "dataset_manifest.json");
  const managedFiles = new Map();
  const managedDirs = new Set();
  function rememberDir(dir) {
    let current = dir;
    while (current && current.startsWith(testRepoRoot) && current !== testRepoRoot) {
      managedDirs.add(current);
      current = path.dirname(current);
    }
  }
  function trackManagedFile(filePath) {
    if (!managedFiles.has(filePath)) {
      managedFiles.set(filePath, fs.existsSync(filePath) ? fs.readFileSync(filePath) : null);
      rememberDir(path.dirname(filePath));
    }
  }
  function writeManagedFixtureFile(filePath, content, track = canonicalLocalEvidence) {
    if (track) {
      trackManagedFile(filePath);
    }
    return writeFixtureFile(filePath, content);
  }
  const images = includeRunnerEvidence
    ? requiredPaperLikeFamilies.map((family) => {
      const imagePath = path.join(dataRoot, family, "images", "sample.png");
      const maskPath = path.join(dataRoot, family, "masks", "sample.png");
      const figurePath = path.join(figureRoot, token, `${family}.png`);
      const resultFile = `assets/repro/iterated_rof_paper_like/${token}/${family}.png`;
      const imageFile = writeManagedFixtureFile(imagePath, paperLikeFixtureContent(family, "image"));
      const maskFile = writeManagedFixtureFile(maskPath, paperLikeFixtureContent(family, "mask"));
      const figureFile = writeManagedFixtureFile(figurePath, paperLikeFixtureContent(family, "figure"));
      const resultFileEvidence = writeManagedFixtureFile(
        path.join(testRepoRoot, "docs", resultFile),
        paperLikeFixtureContent(family, "figure")
      );
      const sourceArtifactFile = writeManagedFixtureFile(
        path.join(dataRoot, family, "audit", token, "source-artifact.txt"),
        paperLikeFixtureContent(family, "source-artifact")
      );
      const licenseSnapshotFile = writeManagedFixtureFile(
        path.join(dataRoot, family, "audit", token, "license-snapshot.txt"),
        paperLikeFixtureContent(family, "license-snapshot")
      );
      const sourceAudit = {
        ...paperLikeSourceAudit,
        source_url: paperLikeSourceUrlByFamily[family],
        source_artifact_path: sourceArtifactFile.path,
        source_artifact_sha256: sourceArtifactFile.sha256,
        license_snapshot_path: licenseSnapshotFile.path,
        license_snapshot_sha256: licenseSnapshotFile.sha256
      };
      const figureEvidencePath = `${figurePath}.evidence.json`;
      const figurePanels = ["input", "mask", "ROF", "T-ROF", "raw K-means", "multi-Otsu", "T-ROF error", "T-ROF vs Otsu"];
      const figureEvidence = {
        schema_version: 1,
        paper_id: "iterated-rof",
        generator: "iterated_rof_paper_like.figure_grid_v1",
        family,
        image_path: imagePath,
        mask_path: maskPath,
        figure_path: figurePath,
        qualitative_only: false,
        image_sha256: imageFile.sha256,
        mask_sha256: maskFile.sha256,
        figure_sha256: figureFile.sha256,
        figure_size_bytes: figureFile.size_bytes,
        figure_panels: figurePanels,
        solver: "sat_rof_trof.rof_chambolle_pock + sat_rof_trof.run_trof_thresholds",
        parameters: { mu: 8, rof_n_iter: 8, trof_max_iter: 4 },
        thresholds: [0.25, 0.5, 0.75],
        n_classes: 4,
        metrics: { clustering_accuracy: 1 },
        baselines: {
          raw_kmeans: {},
          multi_otsu: { thresholds: [0.25, 0.5, 0.75] }
        }
      };
      writeManagedFixtureFile(figureEvidencePath, `${JSON.stringify(figureEvidence, null, 2)}\n`);
      const figureEvidenceFile = {
        path: figureEvidencePath,
        ...paperLikeFixtureFileEvidence(family, "figure-evidence")
      };
      const actualSidecar = fs.readFileSync(figureEvidencePath);
      figureEvidenceFile.size_bytes = actualSidecar.length;
      figureEvidenceFile.sha256 = crypto.createHash("sha256").update(actualSidecar).digest("hex");
      return {
        family,
        status: "completed",
        qualitative_only: false,
        image_path: imagePath,
        mask_path: maskPath,
        metrics: { clustering_accuracy: 1 },
        baselines: {
          raw_kmeans: { metrics: { clustering_accuracy: 1 } },
          multi_otsu: { thresholds: [0.25, 0.5, 0.75], metrics: { clustering_accuracy: 1 } }
        },
        solver: "sat_rof_trof.rof_chambolle_pock + sat_rof_trof.run_trof_thresholds",
        n_classes: 4,
        thresholds: [0.25, 0.5, 0.75],
        threshold_iterations: 3,
        rof_iterations: 8,
        rof_final_residual: 0.001,
        parameters: { mu: 8, rof_n_iter: 8, trof_max_iter: 4 },
        image_file: imageFile,
        mask_file: maskFile,
        figure_path: figurePath,
        figure_file: figureFile,
        result_file: resultFile,
        result_file_evidence: resultFileEvidence,
        figure_panels: figurePanels,
        figure_evidence_path: figureEvidencePath,
        figure_evidence: figureEvidence,
        figure_evidence_file: figureEvidenceFile,
        source_claim: {
          manifest_status: "present",
          manifest_path: manifestPath,
          claim_scope: "file",
          image: "sample.png",
          mask: "sample.png",
          source_id: paperLikeSourceIdByFamily[family],
          source_name: `${family} source`,
          license_reviewed: true,
          license_note: "reviewed dataset license",
          citation: "recorded dataset citation",
          provenance_reviewed: true,
          provenance_note: "recorded dataset provenance",
          synthetic_fixture: false,
          source_audit: sourceAudit,
          sha256: imageFile.sha256,
          mask_sha256: maskFile.sha256
        }
      };
    })
    : [];
  if (includeRunnerEvidence && canonicalLocalEvidence) {
    const manifest = {
      families: Object.fromEntries(requiredPaperLikeFamilies.map((family) => {
        const image = images.find((item) => item.family === family);
        return [
          family,
          {
            source_id: image.source_claim.source_id,
            source_name: image.source_claim.source_name,
            license_reviewed: true,
            license_note: image.source_claim.license_note,
            citation: image.source_claim.citation,
            provenance_reviewed: true,
            provenance_note: image.source_claim.provenance_note,
            synthetic_fixture: false,
            source_audit: image.source_claim.source_audit,
            files: [
              {
                image: "sample.png",
                sha256: image.image_file.sha256,
                mask: "sample.png",
                mask_sha256: image.mask_file.sha256
              }
            ]
          }
        ];
      }))
    };
    writeManagedFixtureFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  }
  const familySummaries = includeRunnerEvidence
    ? requiredPaperLikeFamilies.map((family) => ({
      family,
      status: "completed_quantitative",
      image_count: 1,
      mask_count: 1,
      matched_mask_count: 1,
      completed_image_count: 1,
      failed_image_count: 0,
      quantitative_image_count: 1,
      qualitative_image_count: 0,
      metrics_mean: { clustering_accuracy: 1 },
      baseline_metrics_mean: {
        raw_kmeans: { clustering_accuracy: 1 },
        multi_otsu: { clustering_accuracy: 1 }
      },
      figure_paths: [images.find((item) => item.family === family).figure_path],
      source_claims: [images.find((item) => item.family === family).source_claim],
      errors: []
    }))
    : [];
  const summary = {
    status: "completed_local_runner",
    readiness_status: "ready_for_paper_like_runner",
    paper_id: "iterated-rof",
    experiment_id: "iterated_rof_paper_like",
    paper_like_gate: gate,
    dataset_fingerprint: gate.evidence_summary.dataset_fingerprint,
    image_count: images.length || gate.evidence_summary.image_count,
    completed_image_count: images.length || gate.evidence_summary.completed_image_count,
    quantitative_image_count: images.length || gate.evidence_summary.quantitative_image_count,
    images,
    family_summaries: familySummaries,
    local_dataset_manifest: {
      status: includeRunnerEvidence ? "present" : "missing",
      path: manifestPath
    },
    run_protocol: {
      protocol_id: "iterated_rof_trof_local_data_v1",
      solver: "sat_rof_trof.rof_chambolle_pock + sat_rof_trof.run_trof_thresholds"
    }
  };
  const text = `${JSON.stringify(summary, null, 2)}\n`;
  writeManagedFixtureFile(summaryPath, text, sourceSummaryInRepo);
  return {
    path: summaryPath,
    sha256: crypto.createHash("sha256").update(text).digest("hex"),
    cleanup: () => {
      fs.rmSync(directory, { recursive: true, force: true });
      for (const [filePath, original] of [...managedFiles.entries()].reverse()) {
        fs.rmSync(filePath, { force: true });
        if (original === null) {
          continue;
        } else {
          fs.mkdirSync(path.dirname(filePath), { recursive: true });
          fs.writeFileSync(filePath, original);
        }
      }
      for (const dir of [...managedDirs].sort((left, right) => right.length - left.length)) {
        try {
          fs.rmdirSync(dir);
        } catch {
          // Directory is not empty or no longer exists; leave unrelated files alone.
        }
      }
    }
  };
}

function verificationWithSummary(source) {
  return {
    ...completePaperLikeVerification,
    source_summary_path: source.path,
    source_summary_sha256: source.sha256
  };
}

function resultFilesWithSummary(source) {
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  return (summary.images || []).map((image) => image.result_file).filter(Boolean);
}

function completeCandidate(verification) {
  let resultFiles = baseResult.resultFiles;
  if (verification?.source_summary_path && fs.existsSync(verification.source_summary_path)) {
    const summary = JSON.parse(fs.readFileSync(verification.source_summary_path, "utf8"));
    resultFiles = (summary.images || []).map((image) => image.result_file).filter(Boolean);
  }
  const dashboardDetailPatch = {
    ...baseAssessment,
    reproductionLevel: "paper-like",
    resultFiles,
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const runResultPatch = {
    ...baseResult,
    reproductionLevel: "paper-like",
    resultFiles,
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  return {
    paper_id: "iterated-rof",
    priority: 3,
    can_promote: true,
    reproductionLevel: "paper-like",
    paperLikeGate: completePaperLikeGate,
    candidateDetails: dashboardDetailPatch,
    dashboardDetailPatch,
    runResultPatch
  };
}

function completePaperLevelGate() {
  return {
    passed: true,
    dashboard_level: "paper-level",
    reasons: [],
    checked_requirements: ["original or equivalent paper protocol reproduced"],
    checklist: [{ id: "paper_protocol", passed: true, reasons: [] }],
    evidence_summary: {
      schema_version: 1,
      gate_id: "self-claimed-paper-level",
      paper_level_protocol: true,
      original_or_equivalent_data: true,
      paper_tables_reproduced: true,
      protocol_id: "paper-original-protocol-v1",
      dataset_ids: ["original-or-equivalent-audited-data"],
      table_ids: ["table-1"],
      baseline_ids: ["paper-baseline"],
      parameter_record_count: 1,
      independent_artifact_count: 1
    }
  };
}

function completePaperLevelCandidate() {
  const gate = completePaperLevelGate();
  const verification = {
    schema_version: 1,
    generated_by: "paper_level.independent_verifier_v1",
    recomputed_gate: true,
    can_promote: true,
    gate_id: gate.evidence_summary.gate_id
  };
  const dashboardDetailPatch = {
    ...baseAssessment,
    reproductionLevel: "paper-level",
    reproductionTruthLevel: "paper-level-completed",
    paper_level_gate: gate,
    paper_level_verification: verification
  };
  const runResultPatch = {
    ...baseResult,
    reproductionLevel: "paper-level",
    paper_level_gate: gate,
    paper_level_verification: verification
  };
  return {
    paper_id: "iterated-rof",
    priority: 3,
    can_promote: true,
    reproductionLevel: "paper-level",
    candidateDetails: dashboardDetailPatch,
    dashboardDetailPatch,
    runResultPatch
  };
}

function secondResultFixture() {
  return {
    ...baseResult,
    priority: 4,
    id: "joint-restoration",
    experiment_id: "joint_restoration_segmentation"
  };
}

function secondAssessmentFixture() {
  return {
    ...baseAssessment,
    priority: 4,
    id: "joint-restoration",
    experimentId: "joint_restoration_segmentation"
  };
}

function runSyncCli(args, env = {}) {
  return spawnSync(process.execPath, ["reproduce/sync_to_dashboard.mjs", ...args], {
    cwd: process.cwd(),
    encoding: "utf8",
    env: { ...process.env, REPRO_SYNC_REPO_ROOT: testRepoRoot, ...env }
  });
}

test("sync CLI rejects repo-root override unless explicitly allowed", () => {
  const env = { ...process.env, REPRO_SYNC_REPO_ROOT: testRepoRoot };
  delete env.REPRO_SYNC_ALLOW_REPO_ROOT_OVERRIDE;

  const result = spawnSync(process.execPath, ["reproduce/sync_to_dashboard.mjs", "--check"], {
    cwd: process.cwd(),
    encoding: "utf8",
    env
  });

  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /REPRO_SYNC_REPO_ROOT requires REPRO_SYNC_ALLOW_REPO_ROOT_OVERRIDE=1/);
});

test("expectedTruthLevel derives truth from run result reproductionLevel", () => {
  assert.equal(expectedTruthLevel({ reproductionLevel: "toy" }), "toy-completed");
  assert.equal(expectedTruthLevel({ reproductionLevel: "toy-to-partial" }), "partial-completed");
  assert.equal(expectedTruthLevel({ reproductionLevel: "partial" }), "partial-completed");
  assert.equal(expectedTruthLevel({ reproductionLevel: "paper-like" }), "partial-completed");
  assert.equal(expectedTruthLevel({ reproductionLevel: "paper-level" }), "paper-level-completed");
  assert.equal(expectedTruthLevel({ reproductionLevel: "assessment-only" }), "assessment-only");
  assert.equal(expectedTruthLevel({ reproductionLevel: "paper-level", status: "failed" }), "assessment-only");
  assert.equal(expectedTruthLevel({ reproductionLevel: "paper-like", status: "skipped" }), "assessment-only");
  assert.equal(
    expectedTruthLevel({
      reproductionLevel: "partial",
      reproductionTruthLevel: "paper-level-completed"
    }),
    "partial-completed"
  );
});

test("compareDashboardToResults accepts matching dashboard truth level", () => {
  assert.deepEqual(compareDashboardToResults({ reproAssessments: [baseAssessment] }, [baseResult]), []);
});

test("compareDashboardToResults rejects missing or mismatched run-result priority", () => {
  const missingPriority = { ...baseResult };
  delete missingPriority.priority;

  assert.ok(
    compareDashboardToResults({ reproAssessments: [baseAssessment] }, [missingPriority])
      .some((item) => item.includes("priority"))
  );
  assert.ok(
    compareDashboardToResults({ reproAssessments: [baseAssessment] }, [{ ...baseResult, priority: 99 }])
      .some((item) => item.includes("priority"))
  );
});

test("compareDashboardToResults rejects duplicate dashboard and run-result ids", () => {
  const differences = compareDashboardToResults(
    { reproAssessments: [baseAssessment, { ...baseAssessment, priority: 4 }] },
    [baseResult, { ...baseResult, priority: 4 }]
  );

  assert.ok(differences.some((item) => item.includes("duplicate dashboard assessment id")));
  assert.ok(differences.some((item) => item.includes("duplicate run result id")));
});

test("compareDashboardToResults rejects run-result order drift", () => {
  const secondAssessment = {
    ...baseAssessment,
    priority: 4,
    id: "joint-restoration",
    experimentId: "joint_restoration_segmentation"
  };
  const secondResult = {
    ...baseResult,
    priority: 4,
    id: "joint-restoration",
    experiment_id: "joint_restoration_segmentation"
  };

  const differences = compareDashboardToResults(
    { reproAssessments: [baseAssessment, secondAssessment] },
    [secondResult, baseResult]
  );

  assert.ok(differences.some((item) => item.includes("order")));
});

test("compareDashboardToResults rejects manual paper-level truth override", () => {
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionTruthLevel: "paper-level-completed"
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [baseResult]);

  assert.ok(differences.some((item) => item.includes("reproductionTruthLevel")));
});

test("compareDashboardToResults rejects run-result truth override", () => {
  const result = {
    ...baseResult,
    reproductionTruthLevel: "paper-level-completed"
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionTruthLevel: "paper-level-completed"
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [result]);

  assert.ok(differences.some((item) => item.includes("run result truth override")));
});

test("compareDashboardToResults requires paper-like gate when result claims paper-like", () => {
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like"
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like"
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [result]);

  assert.ok(differences.some((item) => item.includes("paperLikeGate")));
});

test("compareDashboardToResults rejects shallow paper-like gate", () => {
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: { passed: true }
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: { passed: true }
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [result]);

  assert.ok(differences.some((item) => item.includes("paperLikeGate")));
});

test("compareDashboardToResults rejects complete-looking paper-like gate without evidence summary", () => {
  const paperLikeGate = {
    passed: true,
    dashboard_level: "paper-like",
    reasons: [],
    checked_requirements: ["all data families have completed quantitative local runner outputs"],
    checklist: [{ id: "runner_outputs", passed: true, reasons: [] }]
  };
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: paperLikeGate
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: paperLikeGate
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [result]);

  assert.ok(differences.some((item) => item.includes("complete recomputed paper_like_gate")));
});

test("compareDashboardToResults accepts evidence-backed paper-like gate shape", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const resultFiles = resultFilesWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    resultFiles,
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        resultFiles,
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    assert.deepEqual(compareDashboardToResults(dashboard, [result]), []);
  } finally {
    source.cleanup();
  }
});

test("paper-like summary fixture writes promotion evidence under the temp sync repo", () => {
  const source = writePaperLikeSummaryFixture();
  try {
    const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
    assert.ok(source.path.startsWith(testRepoRoot));
    assert.ok(summary.local_dataset_manifest.path.startsWith(testRepoRoot));
    for (const image of summary.images) {
      assert.ok(image.image_path.startsWith(testRepoRoot));
      assert.ok(image.mask_path.startsWith(testRepoRoot));
      assert.ok(image.figure_path.startsWith(testRepoRoot));
      assert.ok(image.source_claim.source_audit.source_artifact_path.startsWith(testRepoRoot));
      assert.ok(image.source_claim.source_audit.license_snapshot_path.startsWith(testRepoRoot));
    }
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults accepts file-level source audit overrides in canonical manifest", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  const token = path.basename(source.path, ".json");
  const overrideArtifactPath = path.join(
    testRepoRoot,
    "reproduce",
    "data",
    "iterated_rof",
    auditedImage.family,
    "audit",
    `${token}-file-source-artifact.txt`
  );
  const overrideLicensePath = path.join(
    testRepoRoot,
    "reproduce",
    "data",
    "iterated_rof",
    auditedImage.family,
    "audit",
    `${token}-file-license-snapshot.txt`
  );
  fs.mkdirSync(path.dirname(overrideArtifactPath), { recursive: true });
  fs.writeFileSync(overrideArtifactPath, paperLikeFixtureContent(auditedImage.family, "file-source-artifact"));
  fs.writeFileSync(overrideLicensePath, paperLikeFixtureContent(auditedImage.family, "file-license-snapshot"));
  const overrideAudit = {
    ...auditedImage.source_claim.source_audit,
    source_artifact_path: overrideArtifactPath,
    source_artifact_sha256: sha256Content(fs.readFileSync(overrideArtifactPath)),
    license_snapshot_path: overrideLicensePath,
    license_snapshot_sha256: sha256Content(fs.readFileSync(overrideLicensePath))
  };
  auditedImage.source_claim.source_audit = overrideAudit;
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0].source_audit = overrideAudit;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].files[0].source_audit = overrideAudit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const resultFiles = resultFilesWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    resultFiles,
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        resultFiles,
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    assert.deepEqual(compareDashboardToResults(dashboard, [result]), []);
  } finally {
    fs.rmSync(overrideArtifactPath, { force: true });
    fs.rmSync(overrideLicensePath, { force: true });
    source.cleanup();
  }
});

test("compareDashboardToResults rejects summaries that ignore file-level source audit overrides", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  const badOverrideAudit = {
    ...auditedImage.source_claim.source_audit,
    source_artifact_path: path.join(
      testRepoRoot,
      "reproduce",
      "data",
      "iterated_rof",
      auditedImage.family,
      "audit",
      "missing-file-level-source-artifact.txt"
    ),
    source_artifact_sha256: "f".repeat(64)
  };
  manifest.families[auditedImage.family].files[0].source_audit = badOverrideAudit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit does not match canonical local dataset_manifest")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects paper-like source summary artifacts outside repo results", () => {
  const source = writePaperLikeSummaryFixture(completePaperLikeGate, { sourceSummaryInRepo: false });
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source summary artifact must be under reproduce/results")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects paper-like summary artifact without runner image evidence", () => {
  const source = writePaperLikeSummaryFixture(completePaperLikeGate, { includeRunnerEvidence: false });
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("completed quantitative image evidence rows")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects paper-like summary artifact without canonical local manifest", () => {
  const source = writePaperLikeSummaryFixture(completePaperLikeGate, { canonicalLocalEvidence: false });
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("canonical local dataset_manifest")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects paper-like source summary with non-image file evidence", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const forgedImage = summary.images[0];
  const textContent = Buffer.from("not an image despite matching sha\n");
  fs.writeFileSync(forgedImage.image_path, textContent);
  const textEvidence = {
    path: forgedImage.image_path,
    size_bytes: textContent.length,
    sha256: sha256Content(textContent)
  };
  forgedImage.image_file = textEvidence;
  forgedImage.source_claim.sha256 = textEvidence.sha256;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  const fileClaim = manifest.families[forgedImage.family].files.find((item) => item.image === forgedImage.source_claim.image);
  fileClaim.sha256 = textEvidence.sha256;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  summary.dataset_fingerprint = fingerprintFromRecords(
    summary.images.flatMap((item) => [
      {
        kind: "image",
        family: item.family,
        path: item.source_claim.image,
        sha256: item.image_file.sha256
      },
      {
        kind: "mask",
        family: item.family,
        path: item.source_claim.mask,
        sha256: item.mask_file.sha256
      }
    ])
  );
  summary.paper_like_gate.evidence_summary.dataset_fingerprint = summary.dataset_fingerprint;
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = {
    ...completePaperLikeVerification,
    dataset_fingerprint: summary.dataset_fingerprint,
    source_summary_path: source.path,
    source_summary_sha256: source.sha256
  };
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: summary.paper_like_gate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: summary.paper_like_gate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("not a supported image file")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects paper-like source summary with tiny image evidence", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const tinyImage = summary.images[0];
  fs.writeFileSync(tinyImage.image_path, tinyPngFixture);
  const tinyEvidence = {
    path: tinyImage.image_path,
    size_bytes: tinyPngFixture.length,
    sha256: sha256Content(tinyPngFixture)
  };
  tinyImage.image_file = tinyEvidence;
  tinyImage.source_claim.sha256 = tinyEvidence.sha256;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  const fileClaim = manifest.families[tinyImage.family].files.find((item) => item.image === tinyImage.source_claim.image);
  fileClaim.sha256 = tinyEvidence.sha256;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  summary.dataset_fingerprint = fingerprintFromRecords(
    summary.images.flatMap((item) => [
      {
        kind: "image",
        family: item.family,
        path: item.source_claim.image,
        sha256: item.image_file.sha256
      },
      {
        kind: "mask",
        family: item.family,
        path: item.source_claim.mask,
        sha256: item.mask_file.sha256
      }
    ])
  );
  summary.paper_like_gate.evidence_summary.dataset_fingerprint = summary.dataset_fingerprint;
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = {
    ...completePaperLikeVerification,
    dataset_fingerprint: summary.dataset_fingerprint,
    source_summary_path: source.path,
    source_summary_sha256: source.sha256
  };
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: summary.paper_like_gate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: summary.paper_like_gate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("too small for paper-like evidence")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects tiny non-PNG paper-like image evidence", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const tinyImage = summary.images[0];
  const oldImagePath = tinyImage.image_path;
  const oldMaskPath = tinyImage.mask_path;
  const jpegPath = oldImagePath.replace(/sample\.png$/, "sample.jpg");
  const bmpPath = oldMaskPath.replace(/sample\.png$/, "sample.bmp");
  fs.writeFileSync(jpegPath, tinyJpegFixture);
  fs.writeFileSync(bmpPath, tinyBmpFixture);
  const tinyImageEvidence = {
    path: jpegPath,
    size_bytes: tinyJpegFixture.length,
    sha256: sha256Content(tinyJpegFixture)
  };
  const tinyMaskEvidence = {
    path: bmpPath,
    size_bytes: tinyBmpFixture.length,
    sha256: sha256Content(tinyBmpFixture)
  };
  tinyImage.image_path = jpegPath;
  tinyImage.mask_path = bmpPath;
  tinyImage.image_file = tinyImageEvidence;
  tinyImage.mask_file = tinyMaskEvidence;
  tinyImage.source_claim.image = "sample.jpg";
  tinyImage.source_claim.mask = "sample.bmp";
  tinyImage.source_claim.sha256 = tinyImageEvidence.sha256;
  tinyImage.source_claim.mask_sha256 = tinyMaskEvidence.sha256;
  const familySummary = summary.family_summaries.find((item) => item.family === tinyImage.family);
  familySummary.source_claims[0] = tinyImage.source_claim;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[tinyImage.family].files = [
    {
      image: "sample.jpg",
      sha256: tinyImageEvidence.sha256,
      mask: "sample.bmp",
      mask_sha256: tinyMaskEvidence.sha256
    }
  ];
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  summary.dataset_fingerprint = fingerprintFromRecords(
    summary.images.flatMap((item) => [
      {
        kind: "image",
        family: item.family,
        path: item.source_claim.image,
        sha256: item.image_file.sha256
      },
      {
        kind: "mask",
        family: item.family,
        path: item.source_claim.mask,
        sha256: item.mask_file.sha256
      }
    ])
  );
  summary.paper_like_gate.evidence_summary.dataset_fingerprint = summary.dataset_fingerprint;
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = {
    ...completePaperLikeVerification,
    dataset_fingerprint: summary.dataset_fingerprint,
    source_summary_path: source.path,
    source_summary_sha256: source.sha256
  };
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: summary.paper_like_gate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: summary.paper_like_gate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("too small for paper-like evidence")));
  } finally {
    fs.rmSync(jpegPath, { force: true });
    fs.rmSync(bmpPath, { force: true });
    source.cleanup();
  }
});

test("compareDashboardToResults rejects blank image, single-label mask, and blank figure evidence", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const blankImage = summary.images[0];
  fs.writeFileSync(blankImage.image_path, blankPaperLikePngFixture);
  fs.writeFileSync(blankImage.mask_path, blankPaperLikePngFixture);
  fs.writeFileSync(blankImage.figure_path, blankPaperLikePngFixture);
  blankImage.image_file = {
    path: blankImage.image_path,
    size_bytes: blankPaperLikePngFixture.length,
    sha256: sha256Content(blankPaperLikePngFixture)
  };
  blankImage.mask_file = {
    path: blankImage.mask_path,
    size_bytes: blankPaperLikePngFixture.length,
    sha256: sha256Content(blankPaperLikePngFixture)
  };
  blankImage.figure_file = {
    path: blankImage.figure_path,
    size_bytes: blankPaperLikePngFixture.length,
    sha256: sha256Content(blankPaperLikePngFixture)
  };
  blankImage.source_claim.sha256 = blankImage.image_file.sha256;
  blankImage.source_claim.mask_sha256 = blankImage.mask_file.sha256;
  updateFigureEvidenceSidecar(blankImage);

  const familySummary = summary.family_summaries.find((item) => item.family === blankImage.family);
  familySummary.source_claims[0] = blankImage.source_claim;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  const fileClaim = manifest.families[blankImage.family].files.find((item) => item.image === blankImage.source_claim.image);
  fileClaim.sha256 = blankImage.image_file.sha256;
  fileClaim.mask_sha256 = blankImage.mask_file.sha256;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  updateSummaryFingerprint(summary);
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = {
    ...completePaperLikeVerification,
    dataset_fingerprint: summary.dataset_fingerprint,
    source_summary_path: source.path,
    source_summary_sha256: source.sha256
  };
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: summary.paper_like_gate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: summary.paper_like_gate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("input image is visually blank")));
    assert.ok(differences.some((item) => item.includes("mask has fewer than two labels")));
    assert.ok(differences.some((item) => item.includes("figure file is visually blank")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects source summary masks with mismatched image shape", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const mismatchedImage = summary.images[0];
  fs.writeFileSync(mismatchedImage.mask_path, pngFixture(32, 48));
  const maskBuffer = fs.readFileSync(mismatchedImage.mask_path);
  mismatchedImage.mask_file = {
    path: mismatchedImage.mask_path,
    size_bytes: maskBuffer.length,
    sha256: sha256Content(maskBuffer)
  };
  mismatchedImage.source_claim.mask_sha256 = mismatchedImage.mask_file.sha256;
  mismatchedImage.figure_evidence.mask_sha256 = mismatchedImage.mask_file.sha256;
  fs.writeFileSync(
    mismatchedImage.figure_evidence_path,
    `${JSON.stringify(mismatchedImage.figure_evidence, null, 2)}\n`
  );
  const sidecarBuffer = fs.readFileSync(mismatchedImage.figure_evidence_path);
  mismatchedImage.figure_evidence_file = {
    path: mismatchedImage.figure_evidence_path,
    size_bytes: sidecarBuffer.length,
    sha256: sha256Content(sidecarBuffer)
  };
  const familySummary = summary.family_summaries.find((item) => item.family === mismatchedImage.family);
  familySummary.source_claims[0] = mismatchedImage.source_claim;
  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[mismatchedImage.family].files[0].mask_sha256 = mismatchedImage.mask_file.sha256;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);
  summary.dataset_fingerprint = fingerprintFromRecords(
    summary.images.flatMap((item) => [
      {
        kind: "image",
        family: item.family,
        path: item.source_claim.image,
        sha256: item.image_file.sha256
      },
      {
        kind: "mask",
        family: item.family,
        path: item.source_claim.mask,
        sha256: item.mask_file.sha256
      }
    ])
  );
  summary.paper_like_gate.evidence_summary.dataset_fingerprint = summary.dataset_fingerprint;
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = {
    ...completePaperLikeVerification,
    dataset_fingerprint: summary.dataset_fingerprint,
    source_summary_path: source.path,
    source_summary_sha256: source.sha256
  };
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: summary.paper_like_gate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: summary.paper_like_gate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("mask shape does not match image shape")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects tiny fixture source audit artifacts", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  const audit = auditedImage.source_claim.source_audit;
  fs.writeFileSync(audit.source_artifact_path, "cartoon reviewed source artifact\n");
  audit.source_artifact_sha256 = sha256Content(fs.readFileSync(audit.source_artifact_path));
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0] = auditedImage.source_claim;
  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit source_artifact artifact is too small")));
    assert.ok(differences.some((item) => item.includes("source audit source_artifact artifact contains fixture/placeholder text")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects unstructured source audit artifacts", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  const audit = auditedImage.source_claim.source_audit;
  const unstructuredArtifact = [
    "teacherZ Iterated ROF local archive note.",
    "An operator kept this narrative after checking a local archive.",
    "It is intentionally long enough to exceed the minimum byte threshold.",
    "It omits concrete web locator, calendar token, operator annotation, and pair trace.",
    "Additional neutral narrative text keeps this from being a tiny text failure.",
    ""
  ].join("\n");
  fs.writeFileSync(audit.source_artifact_path, unstructuredArtifact);
  audit.source_artifact_sha256 = sha256Content(fs.readFileSync(audit.source_artifact_path));
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0] = auditedImage.source_claim;
  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit source_artifact artifact is missing structured review evidence")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects source audit artifacts missing manifest source URL", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  const audit = auditedImage.source_claim.source_audit;
  const wrongUrlArtifact = [
    "teacherZ Iterated ROF source artifact review record for cartoon.",
    "source_url=https://example.invalid/not-the-registered-source",
    "review_date=2026-06-09",
    "reviewer_note=Local source page and source package were reviewed for this family.",
    "conversion_note=Images and masks are mapped by same relative path after source review.",
    ""
  ].join("\n");
  fs.writeFileSync(audit.source_artifact_path, wrongUrlArtifact);
  audit.source_artifact_sha256 = sha256Content(fs.readFileSync(audit.source_artifact_path));
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0] = auditedImage.source_claim;
  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit source_artifact artifact is missing structured review evidence")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects source audit artifacts with invalid review dates", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  const audit = auditedImage.source_claim.source_audit;
  const invalidDateArtifact = [
    "teacherZ Iterated ROF source artifact review record for cartoon.",
    `source_url=${audit.source_url}`,
    "review_date=2026-99-99",
    "reviewer_note=Local source page and source package were reviewed for this family.",
    "conversion_note=Images and masks are mapped by same relative path after source review.",
    ""
  ].join("\n");
  fs.writeFileSync(audit.source_artifact_path, invalidDateArtifact);
  audit.source_artifact_sha256 = sha256Content(fs.readFileSync(audit.source_artifact_path));
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0] = auditedImage.source_claim;
  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit source_artifact artifact is missing structured review evidence")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects narrative-only source audit artifact tokens", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  const audit = auditedImage.source_claim.source_audit;
  const narrativeOnlyArtifact = [
    "teacherZ Iterated ROF source artifact review record for cartoon.",
    `source_url=${audit.source_url}`,
    "review_date=2026-06-09",
    "This narrative says source review was discussed, but no concrete note field was recorded.",
    "It also mentions conversion and mapping as unchecked topics, without a concrete mapping field.",
    "Additional neutral text keeps this artifact over the minimum byte threshold.",
    ""
  ].join("\n");
  fs.writeFileSync(audit.source_artifact_path, narrativeOnlyArtifact);
  audit.source_artifact_sha256 = sha256Content(fs.readFileSync(audit.source_artifact_path));
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0] = auditedImage.source_claim;
  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit source_artifact artifact is missing structured review evidence")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects source summary when canonical manifest text is incomplete", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families.cartoon.citation = "";
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("canonical local dataset_manifest source text is incomplete")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects source claim paths that drift from canonical local file paths", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const forgedImage = summary.images[0];
  forgedImage.source_claim.image = "forged.png";
  forgedImage.source_claim.mask = "forged.png";
  const familySummary = summary.family_summaries.find((item) => item.family === forgedImage.family);
  familySummary.source_claims[0] = forgedImage.source_claim;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[forgedImage.family].files = [
    {
      image: "forged.png",
      sha256: forgedImage.image_file.sha256,
      mask: "forged.png",
      mask_sha256: forgedImage.mask_file.sha256
    }
  ];
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  summary.dataset_fingerprint = fingerprintFromRecords(
    summary.images.flatMap((item) => [
      {
        kind: "image",
        family: item.family,
        path: item.source_claim.image,
        sha256: item.image_file.sha256
      },
      {
        kind: "mask",
        family: item.family,
        path: item.source_claim.mask,
        sha256: item.mask_file.sha256
      }
    ])
  );
  summary.paper_like_gate.evidence_summary.dataset_fingerprint = summary.dataset_fingerprint;

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = {
    ...completePaperLikeVerification,
    dataset_fingerprint: summary.dataset_fingerprint,
    source_summary_path: source.path,
    source_summary_sha256: source.sha256
  };
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: summary.paper_like_gate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: summary.paper_like_gate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source claim image path does not match")));
    assert.ok(differences.some((item) => item.includes("source claim mask path does not match")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects canonical data image symlink escapes", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const escapedImage = summary.images[0];
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "teacherz-symlink-escape-"));
  const outsideImagePath = path.join(directory, "sample.png");

  try {
    fs.writeFileSync(outsideImagePath, paperLikeFixtureContent(escapedImage.family, "image"));
    fs.rmSync(escapedImage.image_path, { force: true });
    fs.symlinkSync(outsideImagePath, escapedImage.image_path);

    const verification = verificationWithSummary(source);
    const result = {
      ...baseResult,
      reproductionLevel: "paper-like",
      paper_like_gate: completePaperLikeGate,
      paper_like_verification: verification
    };
    const dashboard = {
      reproAssessments: [
        {
          ...baseAssessment,
          reproductionLevel: "paper-like",
          paper_like_gate: completePaperLikeGate,
          paper_like_verification: verification
        }
      ]
    };

    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source summary image path is outside canonical local Iterated ROF data root")));
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
    source.cleanup();
  }
});

test("compareDashboardToResults rejects figure evidence sidecar payload drift", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const forgedImage = summary.images[0];
  const forgedSidecar = {
    schema_version: 1,
    generator: "forged.figure.grid",
    image_path: forgedImage.image_path,
    figure_path: forgedImage.figure_path,
    panels: ["input"]
  };
  fs.writeFileSync(forgedImage.figure_evidence_path, `${JSON.stringify(forgedSidecar, null, 2)}\n`);
  const sidecarBytes = fs.readFileSync(forgedImage.figure_evidence_path);
  forgedImage.figure_evidence_file = {
    path: forgedImage.figure_evidence_path,
    size_bytes: sidecarBytes.length,
    sha256: sha256Content(sidecarBytes)
  };

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("figure evidence sidecar does not match report")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects paper-like source summary without structured source audit", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  delete auditedImage.source_claim.source_audit;
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  delete familySummary.source_claims[0].source_audit;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  delete manifest.families[auditedImage.family].source_audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source summary source audit must be an object")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects paper-like source summary with invalid source audit date", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  auditedImage.source_claim.source_audit = {
    ...auditedImage.source_claim.source_audit,
    downloaded_at: "2026-99-99"
  };
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0].source_audit = auditedImage.source_claim.source_audit;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = auditedImage.source_claim.source_audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("downloaded_at must use a valid YYYY-MM-DD date")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects source audit artifact sha mismatches", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  auditedImage.source_claim.source_audit = {
    ...auditedImage.source_claim.source_audit,
    source_artifact_sha256: "f".repeat(64)
  };
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0].source_audit = auditedImage.source_claim.source_audit;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = auditedImage.source_claim.source_audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit source_artifact sha256 mismatch")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects source audit artifacts outside canonical data root", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "teacherz-source-audit-escape-"));
  const outsideArtifact = path.join(directory, "source-artifact.txt");
  fs.writeFileSync(outsideArtifact, paperLikeFixtureContent(auditedImage.family, "source-artifact"));
  const outsideEvidence = {
    path: outsideArtifact,
    sha256: sha256Content(fs.readFileSync(outsideArtifact))
  };
  auditedImage.source_claim.source_audit = {
    ...auditedImage.source_claim.source_audit,
    source_artifact_path: outsideEvidence.path,
    source_artifact_sha256: outsideEvidence.sha256
  };
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0].source_audit = auditedImage.source_claim.source_audit;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = auditedImage.source_claim.source_audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit source_artifact path is outside canonical local Iterated ROF data root")));
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
    source.cleanup();
  }
});

test("compareDashboardToResults rejects source audit artifacts outside family audit root", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images.find((item) => item.family === "cartoon");
  const wrongFamilyArtifact = path.join(
    testRepoRoot,
    "reproduce",
    "data",
    "iterated_rof",
    "texture",
    "audit",
    "cartoon-source-artifact.txt"
  );
  fs.mkdirSync(path.dirname(wrongFamilyArtifact), { recursive: true });
  fs.writeFileSync(wrongFamilyArtifact, paperLikeFixtureContent(auditedImage.family, "source-artifact"));
  const wrongFamilyEvidence = {
    path: wrongFamilyArtifact,
    sha256: sha256Content(fs.readFileSync(wrongFamilyArtifact))
  };
  auditedImage.source_claim.source_audit = {
    ...auditedImage.source_claim.source_audit,
    source_artifact_path: wrongFamilyEvidence.path,
    source_artifact_sha256: wrongFamilyEvidence.sha256
  };
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0].source_audit = auditedImage.source_claim.source_audit;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = auditedImage.source_claim.source_audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit source_artifact path is outside canonical local Iterated ROF family audit root")));
  } finally {
    fs.rmSync(wrongFamilyArtifact, { force: true });
    source.cleanup();
  }
});

test("compareDashboardToResults rejects source audit URLs outside the source registry", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const auditedImage = summary.images[0];
  auditedImage.source_claim.source_audit = {
    ...auditedImage.source_claim.source_audit,
    source_url: "https://example.invalid/not-the-registered-source"
  };
  const familySummary = summary.family_summaries.find((item) => item.family === auditedImage.family);
  familySummary.source_claims[0].source_audit = auditedImage.source_claim.source_audit;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[auditedImage.family].source_audit = auditedImage.source_claim.source_audit;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source audit source_url is not registered")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects paper-like source_id outside registry even when manifest and summary agree", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const forgedImage = summary.images[0];
  const unknownSourceId = `${forgedImage.family}-unregistered-source`;
  forgedImage.source_claim.source_id = unknownSourceId;
  const familySummary = summary.family_summaries.find((item) => item.family === forgedImage.family);
  familySummary.source_claims[0].source_id = unknownSourceId;

  const manifest = JSON.parse(fs.readFileSync(summary.local_dataset_manifest.path, "utf8"));
  manifest.families[forgedImage.family].source_id = unknownSourceId;
  fs.writeFileSync(summary.local_dataset_manifest.path, `${JSON.stringify(manifest, null, 2)}\n`);

  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result]);
    assert.ok(differences.some((item) => item.includes("source_id is not in source registry")));
  } finally {
    source.cleanup();
  }
});

test("compareDashboardToResults rejects paper-like source registry entries with mismatched target family", () => {
  const registry = writeTempSourceRegistry((data) => {
    data.cartoon[0].target_family = "texture";
  });
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const resultFiles = resultFilesWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    resultFiles,
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        resultFiles,
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = compareDashboardToResults(dashboard, [result], {
      iteratedRofSourceRegistryPath: registry.path
    });
    assert.ok(differences.some((item) => item.includes("source registry target_family mismatch")));
  } finally {
    registry.cleanup();
    source.cleanup();
  }
});

test("syncSnapshotDifferences keeps canonical Iterated ROF source registry validation when override is clean", () => {
  const canonicalRegistry = writeTempSourceRegistry((data) => {
    data.cartoon[0].target_family = "texture";
  });
  const overrideRegistry = writeTempSourceRegistry(() => {});

  try {
    const differences = syncSnapshotDifferences(
      { reproAssessments: [baseAssessment] },
      [baseResult],
      null,
      { ITERATED_ROF_SOURCE_REGISTRY_PATH: overrideRegistry.path },
      { iteratedRofCanonicalSourceRegistryPath: canonicalRegistry.path }
    );
    assert.ok(differences.some((item) => item.includes("iterated-rof.sourceRegistry: source registry target_family mismatch")));
  } finally {
    canonicalRegistry.cleanup();
    overrideRegistry.cleanup();
  }
});

test("iteratedRofSourceRegistryDifferences labels override registry failures separately", () => {
  const registry = writeTempSourceRegistry((data) => {
    data.cartoon[0].target_family = "texture";
  });

  try {
    const differences = iteratedRofSourceRegistryDifferences(
      { ITERATED_ROF_SOURCE_REGISTRY_PATH: registry.path }
    );
    assert.ok(differences.some((item) => item.includes("iterated-rof.sourceRegistry.override: source registry target_family mismatch")));
  } finally {
    registry.cleanup();
  }
});

test("syncSnapshotDifferences validates the Iterated ROF source registry without paper-like results", () => {
  const registry = writeTempSourceRegistry((data) => {
    data.cartoon[0].target_family = "texture";
  });

  try {
    const differences = syncSnapshotDifferences(
      { reproAssessments: [baseAssessment] },
      [baseResult],
      null,
      { ITERATED_ROF_SOURCE_REGISTRY_PATH: registry.path }
    );
    assert.ok(differences.some((item) => item.includes("iterated-rof.sourceRegistry.override: source registry target_family mismatch")));
  } finally {
    registry.cleanup();
  }
});

test("syncSnapshotDifferences validates Iterated ROF source registry schema without paper-like results", () => {
  const registry = writeTempSourceRegistry((data) => {
    delete data.cartoon[0].download_url;
    data.texture[0].priority = "1";
  });

  try {
    const differences = syncSnapshotDifferences(
      { reproAssessments: [baseAssessment] },
      [baseResult],
      null,
      { ITERATED_ROF_SOURCE_REGISTRY_PATH: registry.path }
    );
    assert.ok(differences.some((item) => item.includes("iterated-rof.sourceRegistry.override: Iterated ROF source registry missing download_url")));
    assert.ok(differences.some((item) => item.includes("iterated-rof.sourceRegistry.override: Iterated ROF source registry priority must be an integer")));
  } finally {
    registry.cleanup();
  }
});

test("sync CLI check validates the Iterated ROF source registry override", () => {
  const registry = writeTempSourceRegistry((data) => {
    data.cartoon[0].target_family = "texture";
  });

  try {
    const result = spawnSync(
      process.execPath,
      ["reproduce/sync_to_dashboard.mjs", "--check"],
      {
        cwd: process.cwd(),
        env: {
          ...process.env,
          ITERATED_ROF_SOURCE_REGISTRY_PATH: registry.path
        },
        encoding: "utf8"
      }
    );
    assert.equal(result.status, 1);
    assert.match(result.stderr, /iterated-rof\.sourceRegistry\.override/);
    assert.match(result.stderr, /source registry target_family mismatch/);
  } finally {
    registry.cleanup();
  }
});

test("syncSnapshotDifferences applies zero-promotion guard to normal synced snapshots", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const resultFiles = resultFilesWithSummary(source);
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    resultFiles,
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: verification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        resultFiles,
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: verification
      }
    ]
  };

  try {
    const differences = syncSnapshotDifferences(dashboard, [result], null, {});
    assert.ok(differences.some((item) => item.includes("dashboard paper-like 当前必须为 0")));
    assert.ok(differences.some((item) => item.includes("run result paper-like 当前必须为 0")));
    assert.deepEqual(syncSnapshotDifferences(dashboard, [result], null, { ALLOW_PAPER_LIKE: "1" }), []);
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidateShape accepts matching generated candidate patches", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);

  try {
    assert.deepEqual(validateDashboardCandidateShape(candidate), []);
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidate rejects promotion validation without current snapshots", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);

  try {
    const differences = validateDashboardCandidate(candidate);
    assert.ok(differences.some((item) => item.includes("current dashboard, run-result, and asset snapshots are required")));
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidate rejects promotion validation without current asset snapshot", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);
  const currentDashboard = {
    reproAssessments: [baseAssessment, secondAssessmentFixture()]
  };
  const currentRunResults = [baseResult, secondResultFixture()];

  try {
    const differences = validateDashboardCandidate(candidate, currentDashboard, currentRunResults, null, {
      env: { ALLOW_PAPER_LIKE: "1" }
    });
    assert.ok(differences.some((item) => item.includes("current dashboard, run-result, and asset snapshots are required")));
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidateShape rejects paper-level self-claimed promotion without independent artifact", () => {
  const differences = validateDashboardCandidateShape(completePaperLevelCandidate());

  assert.ok(differences.some((item) => item.includes("paper-level result requires independent verification artifact")));
});

test("validateDashboardCandidateShape rejects thin paper-level independent artifact", () => {
  const directory = fs.mkdtempSync(path.join(testReproduceResultsRoot, "paper-level-thin-artifact-"));
  const artifactPath = path.join(directory, "artifact.json");
  const candidate = completePaperLevelCandidate();
  const gate = candidate.runResultPatch.paper_level_gate;
  const artifact = {
    schema_version: 1,
    generated_by: "paper_level.independent_verifier_v1",
    recomputed_gate: true,
    verifier_independent: true,
    can_promote: true,
    gate_id: gate.evidence_summary.gate_id,
    evidence_summary: gate.evidence_summary,
    paper_level_gate: gate
  };
  const artifactText = `${JSON.stringify(artifact, null, 2)}\n`;
  fs.writeFileSync(artifactPath, artifactText);
  const verification = {
    ...candidate.runResultPatch.paper_level_verification,
    source_artifact_path: artifactPath,
    source_artifact_sha256: sha256Content(artifactText)
  };
  candidate.runResultPatch.paper_level_verification = verification;
  candidate.dashboardDetailPatch.paper_level_verification = verification;
  candidate.candidateDetails = candidate.dashboardDetailPatch;

  try {
    const differences = validateDashboardCandidateShape(candidate);
    assert.ok(differences.some((item) => item.includes("requires non-empty table comparisons")));
    assert.ok(differences.some((item) => item.includes("requires non-empty baseline comparisons")));
    assert.ok(differences.some((item) => item.includes("requires non-empty parameter records")));
    assert.ok(differences.some((item) => item.includes("requires non-empty data source audits")));
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("validateDashboardCandidateShape rejects paper-level artifact rows without audited artifact refs", () => {
  const directory = fs.mkdtempSync(path.join(testReproduceResultsRoot, "paper-level-shallow-rows-"));
  const artifactPath = path.join(directory, "artifact.json");
  const candidate = completePaperLevelCandidate();
  const gate = candidate.runResultPatch.paper_level_gate;
  const artifact = {
    schema_version: 1,
    generated_by: "paper_level.independent_verifier_v1",
    recomputed_gate: true,
    verifier_independent: true,
    can_promote: true,
    gate_id: gate.evidence_summary.gate_id,
    evidence_summary: gate.evidence_summary,
    paper_level_gate: gate,
    table_comparisons: [{ table_id: "table-1" }],
    baseline_comparisons: [{ baseline_id: "paper-baseline" }],
    parameter_records: [{ parameter_id: "lambda" }],
    data_source_audits: [{ dataset_id: "original-or-equivalent-audited-data" }]
  };
  const artifactText = `${JSON.stringify(artifact, null, 2)}\n`;
  fs.writeFileSync(artifactPath, artifactText);
  const verification = {
    ...candidate.runResultPatch.paper_level_verification,
    source_artifact_path: artifactPath,
    source_artifact_sha256: sha256Content(artifactText)
  };
  candidate.runResultPatch.paper_level_verification = verification;
  candidate.dashboardDetailPatch.paper_level_verification = verification;
  candidate.candidateDetails = candidate.dashboardDetailPatch;

  try {
    const differences = validateDashboardCandidateShape(candidate);
    assert.ok(differences.some((item) => item.includes("table comparison row 0 requires audited artifact path")));
    assert.ok(differences.some((item) => item.includes("baseline comparison row 0 requires audited artifact path")));
    assert.ok(differences.some((item) => item.includes("parameter record row 0 requires audited artifact path")));
    assert.ok(differences.some((item) => item.includes("data source audit row 0 requires audited artifact path")));
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("validateDashboardCandidateShape rejects paper-level placeholder row artifacts", () => {
  const directory = fs.mkdtempSync(path.join(testReproduceResultsRoot, "paper-level-placeholder-rows-"));
  const artifactPath = path.join(directory, "artifact.json");
  const candidate = completePaperLevelCandidate();
  const gate = candidate.runResultPatch.paper_level_gate;
  const writeRowArtifact = (name) => {
    const artifactFile = path.join(directory, `${name}.txt`);
    const text = "placeholder audit artifact\n";
    fs.writeFileSync(artifactFile, text);
    return {
      artifact_path: artifactFile,
      artifact_sha256: sha256Content(text)
    };
  };
  const artifact = {
    schema_version: 1,
    generated_by: "paper_level.independent_verifier_v1",
    recomputed_gate: true,
    verifier_independent: true,
    can_promote: true,
    gate_id: gate.evidence_summary.gate_id,
    evidence_summary: gate.evidence_summary,
    paper_level_gate: gate,
    table_comparisons: [{ table_id: "table-1", ...writeRowArtifact("table") }],
    baseline_comparisons: [{ baseline_id: "paper-baseline", ...writeRowArtifact("baseline") }],
    parameter_records: [{ parameter_id: "lambda", ...writeRowArtifact("parameter") }],
    data_source_audits: [
      {
        dataset_id: "original-or-equivalent-audited-data",
        source_id: "original-source",
        license_reviewed: true,
        provenance_reviewed: true,
        ...writeRowArtifact("data-source")
      }
    ]
  };
  const artifactText = `${JSON.stringify(artifact, null, 2)}\n`;
  fs.writeFileSync(artifactPath, artifactText);
  const verification = {
    ...candidate.runResultPatch.paper_level_verification,
    source_artifact_path: artifactPath,
    source_artifact_sha256: sha256Content(artifactText)
  };
  candidate.runResultPatch.paper_level_verification = verification;
  candidate.dashboardDetailPatch.paper_level_verification = verification;
  candidate.candidateDetails = candidate.dashboardDetailPatch;

  try {
    const differences = validateDashboardCandidateShape(candidate);
    assert.ok(differences.some((item) => item.includes("table comparison row 0 audited artifact is too small")));
    assert.ok(differences.some((item) => item.includes("table comparison row 0 audited artifact contains fixture/placeholder text")));
    assert.ok(differences.some((item) => item.includes("baseline comparison row 0 audited artifact is too small")));
    assert.ok(differences.some((item) => item.includes("parameter record row 0 audited artifact is too small")));
    assert.ok(differences.some((item) => item.includes("data source audit row 0 audited artifact is too small")));
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("validateDashboardCandidateShape rejects blocked candidate without patches", () => {
  const differences = validateDashboardCandidateShape({
    paper_id: "iterated-rof",
    priority: 3,
    can_promote: false,
    reproductionLevel: "partial",
    candidateDetails: {}
  });

  assert.ok(differences.some((item) => item.includes("can_promote=true")));
  assert.ok(differences.some((item) => item.includes("runResultPatch")));
  assert.ok(differences.some((item) => item.includes("dashboardDetailPatch")));
});

test("validateDashboardCandidateShape rejects candidate metadata drift from patches", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);
  candidate.paper_id = "wrong-paper";
  candidate.priority = 99;
  candidate.paperLikeGate = { ...completePaperLikeGate, checked_requirements: ["drifted"] };
  candidate.candidateDetails = { ...candidate.dashboardDetailPatch, notes: "drifted notes" };

  try {
    const differences = validateDashboardCandidateShape(candidate);
    assert.ok(differences.some((item) => item.includes("paper_id")));
    assert.ok(differences.some((item) => item.includes("priority")));
    assert.ok(differences.some((item) => item.includes("paperLikeGate")));
    assert.ok(differences.some((item) => item.includes("candidateDetails")));
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidateShape rejects paper-like resultFiles not bound to source summary figures", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);
  candidate.dashboardDetailPatch = {
    ...candidate.dashboardDetailPatch,
    resultFiles: ["assets/repro/sat_demo.png"]
  };
  candidate.candidateDetails = candidate.dashboardDetailPatch;
  candidate.runResultPatch = {
    ...candidate.runResultPatch,
    resultFiles: ["assets/repro/sat_demo.png"]
  };

  try {
    const differences = validateDashboardCandidateShape(candidate);
    assert.ok(
      differences.some((item) => item.includes("paper-like resultFiles must match source summary figure result files"))
    );
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidateShape rejects paper-like gate evidence count drift from source summary rows", () => {
  const driftedGate = JSON.parse(JSON.stringify(completePaperLikeGate));
  driftedGate.evidence_summary.source_claim_count = 99;
  driftedGate.evidence_summary.figure_evidence_count = 99;
  const source = writePaperLikeSummaryFixture(driftedGate);
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);
  candidate.paperLikeGate = driftedGate;
  candidate.dashboardDetailPatch = {
    ...candidate.dashboardDetailPatch,
    paper_like_gate: driftedGate
  };
  candidate.runResultPatch = {
    ...candidate.runResultPatch,
    paper_like_gate: driftedGate
  };
  candidate.candidateDetails = candidate.dashboardDetailPatch;

  try {
    const differences = validateDashboardCandidateShape(candidate);
    assert.ok(
      differences.some((item) => item.includes("source_claim_count does not match source summary quantitative image rows"))
    );
    assert.ok(
      differences.some((item) => item.includes("figure_evidence_count does not match source summary quantitative image rows"))
    );
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidateShape rejects paper-like family summary drift from source summary images", () => {
  const source = writePaperLikeSummaryFixture();
  const summary = JSON.parse(fs.readFileSync(source.path, "utf8"));
  const cartoonSummary = summary.family_summaries.find((item) => item.family === "cartoon");
  cartoonSummary.figure_paths = [summary.images.find((item) => item.family === "texture").figure_path];
  cartoonSummary.source_claims = [summary.images.find((item) => item.family === "medical").source_claim];
  const summaryText = `${JSON.stringify(summary, null, 2)}\n`;
  fs.writeFileSync(source.path, summaryText);
  source.sha256 = sha256Content(summaryText);
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);

  try {
    const differences = validateDashboardCandidateShape(candidate);
    assert.ok(
      differences.some((item) => item.includes("family_summaries figure_paths do not match image evidence rows for: cartoon"))
    );
    assert.ok(
      differences.some((item) => item.includes("family_summaries source_claims do not match image evidence rows for: cartoon"))
    );
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidate accepts candidate integrated into current snapshots", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);
  const currentDashboard = {
    reproAssessments: [baseAssessment, secondAssessmentFixture()]
  };
  const currentRunResults = [baseResult, secondResultFixture()];

  try {
    assert.deepEqual(
      validateDashboardCandidate(candidate, currentDashboard, currentRunResults, currentRunResults, {
        env: { ALLOW_PAPER_LIKE: "1" }
      }),
      []
    );
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidate applies default promotion count guard to current snapshot overlays", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);
  const currentDashboard = {
    reproAssessments: [baseAssessment, secondAssessmentFixture()]
  };
  const currentRunResults = [baseResult, secondResultFixture()];

  try {
    const differences = validateDashboardCandidate(candidate, currentDashboard, currentRunResults, currentRunResults);
    assert.ok(differences.some((item) => item.includes("candidate.current.dashboard paper-like 当前必须为 0")));
    assert.ok(differences.some((item) => item.includes("candidate.current.run result paper-like 当前必须为 0")));
  } finally {
    source.cleanup();
  }
});

test("validateDashboardCandidate rejects candidate priority drift from current snapshots", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const candidate = completeCandidate(verification);
  candidate.priority = 99;
  candidate.runResultPatch = { ...candidate.runResultPatch, priority: 99 };
  candidate.dashboardDetailPatch = { ...candidate.dashboardDetailPatch, priority: 99 };
  candidate.candidateDetails = candidate.dashboardDetailPatch;
  const currentDashboard = {
    reproAssessments: [baseAssessment, secondAssessmentFixture()]
  };
  const currentRunResults = [baseResult, secondResultFixture()];

  try {
    const differences = validateDashboardCandidate(candidate, currentDashboard, currentRunResults, currentRunResults);
    assert.ok(differences.some((item) => item.includes("current dashboard priority")));
    assert.ok(differences.some((item) => item.includes("current run result priority")));
  } finally {
    source.cleanup();
  }
});

test("sync CLI rejects paper-like candidate overlays unless explicitly allowed", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "teacherz-candidate-cli-"));
  const candidatePath = path.join(directory, "candidate.json");
  fs.writeFileSync(candidatePath, `${JSON.stringify(completeCandidate(verification), null, 2)}\n`);

  try {
    const completed = runSyncCli(["--candidate", candidatePath, "--check", "--quiet"]);
    assert.notEqual(completed.status, 0);
  } finally {
    source.cleanup();
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("sync CLI validates promotable dashboard candidate files", () => {
  const source = writePaperLikeSummaryFixture();
  const verification = verificationWithSummary(source);
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "teacherz-candidate-cli-"));
  const candidatePath = path.join(directory, "candidate.json");
  fs.writeFileSync(candidatePath, `${JSON.stringify(completeCandidate(verification), null, 2)}\n`);

  try {
    const completed = runSyncCli(
      ["--candidate", candidatePath, "--check", "--quiet"],
      { ALLOW_PAPER_LIKE: "1" }
    );
    assert.equal(completed.status, 0, completed.stderr);
  } finally {
    source.cleanup();
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("sync CLI rejects blocked dashboard candidate files under --check", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "teacherz-candidate-cli-"));
  const candidatePath = path.join(directory, "blocked_candidate.json");
  fs.writeFileSync(candidatePath, `${JSON.stringify({
    paper_id: "iterated-rof",
    priority: 3,
    can_promote: false,
    reproductionLevel: "partial",
    candidateDetails: {}
  }, null, 2)}\n`);

  try {
    const completed = runSyncCli(["--candidate", candidatePath, "--check", "--quiet"]);
    assert.notEqual(completed.status, 0);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

test("sync CLI rejects missing dashboard candidate path", () => {
  const completed = runSyncCli(["--candidate", "--check"]);

  assert.notEqual(completed.status, 0);
  assert.match(completed.stderr, /missing --candidate path/);
});

test("compareDashboardToResults rejects paper-like verification without summary artifact", () => {
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate,
    paper_like_verification: completePaperLikeVerification
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate,
        paper_like_verification: completePaperLikeVerification
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [result]);

  assert.ok(differences.some((item) => item.includes("source summary")));
});

test("compareDashboardToResults rejects paper-like gate without runner promotion verification", () => {
  const result = {
    ...baseResult,
    reproductionLevel: "paper-like",
    paper_like_gate: completePaperLikeGate
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-like",
        paper_like_gate: completePaperLikeGate
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [result]);

  assert.ok(differences.some((item) => item.includes("paperLikeVerification")));
});

test("compareResultAssetSnapshot rejects stale static repro asset JSON", () => {
  const differences = compareResultAssetSnapshot(
    [baseResult],
    [
      {
        ...baseResult,
        notes: "Stale static asset notes."
      }
    ]
  );

  assert.ok(differences.some((item) => item.includes("docs/assets/repro/repro_results.json")));
});

test("compareResultAssetSnapshot rejects static repro asset order drift", () => {
  const secondResult = secondResultFixture();
  const differences = compareResultAssetSnapshot(
    [baseResult, secondResult],
    [secondResult, baseResult]
  );

  assert.ok(differences.some((item) => item.includes("asset result order")));
});

test("compareResultAssetSnapshot rejects duplicate static repro asset ids", () => {
  const differences = compareResultAssetSnapshot(
    [baseResult, { ...baseResult, priority: 4 }],
    [baseResult, { ...baseResult, priority: 4 }]
  );

  assert.ok(differences.some((item) => item.includes("duplicate run result id")));
  assert.ok(differences.some((item) => item.includes("duplicate asset result id")));
});

test("compareResultAssetSnapshot accepts matching static repro asset JSON", () => {
  assert.deepEqual(compareResultAssetSnapshot([baseResult], [baseResult]), []);
});

test("compareDashboardToResults rejects paper-level with only paper-like gate", () => {
  const result = {
    ...baseResult,
    reproductionLevel: "paper-level",
    paper_like_gate: { passed: true }
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-level",
        reproductionTruthLevel: "paper-level-completed",
        paper_like_gate: { passed: true }
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [result]);

  assert.ok(differences.some((item) => item.includes("paperLevelGate")));
});

test("compareDashboardToResults rejects paper-level gate without paper-level evidence summary", () => {
  const paperLevelGate = {
    passed: true,
    dashboard_level: "paper-level",
    reasons: [],
    checked_requirements: ["original or equivalent paper protocol reproduced"],
    checklist: [{ id: "paper_protocol", passed: true, reasons: [] }]
  };
  const result = {
    ...baseResult,
    reproductionLevel: "paper-level",
    paper_level_gate: paperLevelGate
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-level",
        reproductionTruthLevel: "paper-level-completed",
        paper_level_gate: paperLevelGate
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [result]);

  assert.ok(differences.some((item) => item.includes("complete independent paper_level_gate")));
});

test("compareDashboardToResults rejects paper-level gate without independent verification", () => {
  const paperLevelGate = {
    passed: true,
    dashboard_level: "paper-level",
    reasons: [],
    checked_requirements: ["original or equivalent paper protocol reproduced"],
    checklist: [{ id: "paper_protocol", passed: true, reasons: [] }],
    evidence_summary: {
      schema_version: 1,
      paper_level_protocol: true,
      original_or_equivalent_data: true,
      paper_tables_reproduced: true
    }
  };
  const result = {
    ...baseResult,
    reproductionLevel: "paper-level",
    paper_level_gate: paperLevelGate
  };
  const dashboard = {
    reproAssessments: [
      {
        ...baseAssessment,
        reproductionLevel: "paper-level",
        reproductionTruthLevel: "paper-level-completed",
        paper_level_gate: paperLevelGate
      }
    ]
  };

  const differences = compareDashboardToResults(dashboard, [result]);

  assert.ok(differences.some((item) => item.includes("paperLevelVerification")));
});

test("reading data derives reproduction truth after spreading detail fields", () => {
  const readingDataPath = path.join(process.cwd(), "docs", "js", "reading-data.js");
  const source = fs.readFileSync(readingDataPath, "utf8");
  const assessmentBlock = source.match(/const reproAssessments = paperNotesV2\.map[\s\S]*?\n\}\);/);
  const assessmentReturn = source.match(/return \{[\s\S]*?id: note\.id,[\s\S]*?\n  \};/);

  assert.ok(assessmentBlock, "expected reproAssessments map block");
  assert.ok(assessmentReturn, "expected reproAssessments return object");
  const spreadIndex = assessmentReturn[0].indexOf("...detail");
  const truthIndex = assessmentReturn[0].indexOf("reproductionTruthLevel");

  assert.ok(spreadIndex >= 0, "expected detail spread in reproAssessments return object");
  assert.ok(truthIndex >= 0, "expected derived reproductionTruthLevel in reproAssessments return object");
  assert.ok(
    spreadIndex < truthIndex,
    "derived reproductionTruthLevel must be assigned after ...detail so manual detail fields cannot override it"
  );
  assert.match(
    assessmentBlock[0],
    /detail\.resultStatus[\s\S]*!== "completed"[\s\S]*"assessment-only"/,
    "derived reproductionTruthLevel must treat non-completed detail results as assessment-only"
  );
});
