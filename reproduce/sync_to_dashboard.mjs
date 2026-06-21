import fs from "node:fs";
import crypto from "node:crypto";
import path from "node:path";
import { pathToFileURL } from "node:url";
import vm from "node:vm";
import zlib from "node:zlib";
import { reproductionPromotionCountFailures } from "../docs/scripts/repro-promotion-guards.mjs";

const repoRootOverride = process.env.REPRO_SYNC_REPO_ROOT;
const repoRootOverrideError = repoRootOverride && process.env.REPRO_SYNC_ALLOW_REPO_ROOT_OVERRIDE !== "1"
  ? "REPRO_SYNC_REPO_ROOT requires REPRO_SYNC_ALLOW_REPO_ROOT_OVERRIDE=1; this override is for isolated test fixtures only"
  : "";
const repoRoot = path.resolve(repoRootOverrideError ? process.cwd() : (repoRootOverride || process.cwd()));
const readingDataPath = path.join(repoRoot, "docs", "js", "reading-data.js");
const defaultResultsPath = path.join(repoRoot, "reproduce", "results", "repro_results.json");
const reproduceResultsRoot = path.join(repoRoot, "reproduce", "results");
const assetResultsPath = path.join(repoRoot, "docs", "assets", "repro", "repro_results.json");
const docsReproAssetRoot = path.join(repoRoot, "docs", "assets", "repro");
const iteratedRofDataRoot = path.join(repoRoot, "reproduce", "data", "iterated_rof");
const iteratedRofManifestPath = path.join(iteratedRofDataRoot, "dataset_manifest.json");
const iteratedRofFigureRoot = path.join(repoRoot, "reproduce", "results", "figures", "iterated_rof_paper_like");
const iteratedRofSourceRegistryPath = path.join(repoRoot, "reproduce", "paper_like", "iterated_rof_dataset_sources.json");
const minPaperLikeImageSide = 32;
const minPaperLikeImageLevels = 8;
const minSourceAuditArtifactBytes = 128;
const minPaperLevelArtifactBytes = 128;
const iteratedRofFigureEvidenceGenerator = "iterated_rof_paper_like.figure_grid_v1";
const sourceAuditArtifactPlaceholderPatterns = [
  "test fixture",
  "tiny local test",
  "fixture artifact",
  "fixture evidence",
  "placeholder",
  "reviewed source artifact",
  "reviewed license snapshot"
];
const sourceAuditArtifactReviewNotePattern = /\b(?:reviewer[_ ]note|review[_ ]note|source[_ ]review(?:[_ ]note)?)\s*[:=]\s*\S/;
const sourceAuditArtifactMappingPattern = /\b(?:conversion[_ ]notes?|mapping[_ ]note|local[_ ]file[_ ]mapping|file[_ ]mapping)\s*[:=]\s*\S/;

const rawArgs = process.argv.slice(2);
const args = new Set(rawArgs);
const checkOnly = args.has("--check");
const quiet = args.has("--quiet");
const candidatePath = argValue("--candidate");

function argValue(flag) {
  const index = rawArgs.indexOf(flag);
  if (index < 0) return null;
  const value = rawArgs[index + 1] || null;
  if (!value || value.startsWith("--")) return null;
  return value;
}

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

function duplicateIdDifferences(items, label) {
  if (!Array.isArray(items)) return [];
  const seen = new Set();
  const duplicates = new Set();
  for (const item of items) {
    const id = item?.id;
    if (id === undefined || id === null) continue;
    if (seen.has(id)) duplicates.add(id);
    seen.add(id);
  }
  return [...duplicates].sort().map((id) => `${label}: duplicate ${label} id ${JSON.stringify(id)}`);
}

function isRuntimeSecondsMetricKey(key) {
  return key.includes("runtime_seconds") || key.endsWith("_runtime_seconds");
}

function withoutRuntimeSecondsMetrics(value) {
  if (Array.isArray(value)) return value.map(withoutRuntimeSecondsMetrics);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => !isRuntimeSecondsMetricKey(key))
        .map(([key, nestedValue]) => [key, withoutRuntimeSecondsMetrics(nestedValue)])
    );
  }
  return value;
}

function isRuntimeSecondsValue(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function compareRuntimeSecondsShape(id, dashboardValue, resultValue, differences) {
  // Wall-clock runtime drifts between reruns and machines. Keep the schema strict,
  // but do not require exact runtimeSeconds/runtime_seconds equality.
  if (dashboardValue === undefined && resultValue === undefined) return;
  if (dashboardValue === undefined || resultValue === undefined) {
    differences.push(`${id}.runtimeSeconds: dashboard=${JSON.stringify(dashboardValue)} result=${JSON.stringify(resultValue)}`);
    return;
  }
  if (!isRuntimeSecondsValue(dashboardValue) || !isRuntimeSecondsValue(resultValue)) {
    differences.push(`${id}.runtimeSeconds: expected numeric runtime values, dashboard=${JSON.stringify(dashboardValue)} result=${JSON.stringify(resultValue)}`);
  }
}

const requiredPaperLikeFamilies = ["cartoon", "texture", "medical"];
const requiredSourceRegistryTextFields = [
  "source_id",
  "name",
  "url",
  "download_url",
  "target_family",
  "fit",
  "local_layout",
  "download_policy",
  "license_note",
  "paper_like_role"
];

function includesAllRequired(values, requiredValues) {
  if (!Array.isArray(values)) return false;
  const present = new Set(values);
  return requiredValues.every((value) => present.has(value));
}

function hasDatasetFingerprintShape(fingerprint) {
  return Boolean(
    fingerprint
    && typeof fingerprint === "object"
    && !Array.isArray(fingerprint)
    && fingerprint.algorithm === "sha256"
    && typeof fingerprint.file_count === "number"
    && fingerprint.file_count > 0
    && typeof fingerprint.sha256 === "string"
    && /^[a-f0-9]{64}$/i.test(fingerprint.sha256)
  );
}

function hasPaperLikeEvidenceSummaryShape(summary) {
  return Boolean(
    summary
    && typeof summary === "object"
    && !Array.isArray(summary)
    && summary.schema_version === 1
    && summary.gate_id === "iterated_rof_paper_like_v1"
    && hasDatasetFingerprintShape(summary.dataset_fingerprint)
    && summary.image_count >= requiredPaperLikeFamilies.length
    && summary.completed_image_count >= requiredPaperLikeFamilies.length
    && summary.quantitative_image_count >= requiredPaperLikeFamilies.length
    && includesAllRequired(summary.required_families, requiredPaperLikeFamilies)
    && includesAllRequired(summary.completed_families, requiredPaperLikeFamilies)
    && includesAllRequired(summary.quantitative_families, requiredPaperLikeFamilies)
    && summary.source_claim_count >= requiredPaperLikeFamilies.length
    && summary.figure_evidence_count >= requiredPaperLikeFamilies.length
  );
}

function hasCompletePaperLikeGateShape(gate) {
  if (!gate || typeof gate !== "object" || Array.isArray(gate)) return false;
  if (gate.passed !== true) return false;
  if (gate.dashboard_level !== "paper-like" && gate.dashboardLevel !== "paper-like") return false;
  if (!Array.isArray(gate.reasons) || gate.reasons.length !== 0) return false;
  if (!Array.isArray(gate.checked_requirements) || gate.checked_requirements.length === 0) return false;
  if (!Array.isArray(gate.checklist) || gate.checklist.length === 0) return false;
  if (!gate.checklist.every((item) => item && typeof item === "object" && item.passed === true)) return false;
  return hasPaperLikeEvidenceSummaryShape(gate.evidence_summary || gate.evidenceSummary);
}

function hasCompletePaperLikeVerificationShape(verification, gate) {
  const evidence = gate?.evidence_summary || gate?.evidenceSummary;
  return Boolean(
    verification
    && typeof verification === "object"
    && !Array.isArray(verification)
    && verification.schema_version === 1
    && verification.generated_by === "iterated_rof_paper_like.dashboard_candidate_v1"
    && verification.recomputed_gate === true
    && verification.can_promote === true
    && verification.promotion_shape_blocker_count === 0
    && verification.gate_id === "iterated_rof_paper_like_v1"
    && sameJson(verification.dataset_fingerprint, evidence?.dataset_fingerprint)
  );
}

function isSha256(value) {
  return typeof value === "string" && /^[a-f0-9]{64}$/i.test(value);
}

function resolveLocalArtifactPath(value) {
  if (typeof value !== "string" || !value.trim()) return null;
  return path.isAbsolute(value) ? value : path.resolve(repoRoot, value);
}

function isPathUnder(child, parent) {
  if (!child || !parent) return false;
  let childPath;
  let parentPath;
  try {
    childPath = fs.realpathSync.native(child);
    parentPath = fs.realpathSync.native(parent);
  } catch {
    return false;
  }
  const relative = path.relative(parentPath, childPath);
  return relative === "" || (relative && !relative.startsWith("..") && !path.isAbsolute(relative));
}

function resolveDocsResultFile(file) {
  if (typeof file !== "string" || !file.trim()) return null;
  if (file !== file.trim() || file.includes("\\")) return null;
  if (path.posix.isAbsolute(file) || path.posix.normalize(file) !== file) return null;
  if (file.startsWith("../") || file.includes("/../")) return null;
  if (!file.startsWith("assets/repro/")) return null;
  return path.resolve(repoRoot, "docs", file);
}

function resultFilePathDifferences(id, files) {
  const differences = [];
  if (!Array.isArray(files) || files.length === 0) {
    differences.push(`${id}.resultFiles: expected non-empty resultFiles array`);
    return differences;
  }
  const seen = new Set();
  files.forEach((file, index) => {
    const resolved = resolveDocsResultFile(file);
    if (!resolved) {
      differences.push(`${id}.resultFiles[${index}]: resultFile must be a relative docs assets/repro path without traversal`);
      return;
    }
    if (seen.has(file)) {
      differences.push(`${id}.resultFiles[${index}]: duplicate resultFile ${JSON.stringify(file)}`);
    }
    seen.add(file);
    if (!isPathUnder(path.dirname(resolved), docsReproAssetRoot)) {
      differences.push(`${id}.resultFiles[${index}]: resultFile must stay under docs/assets/repro`);
    }
  });
  return differences;
}

function sha256File(filePath) {
  return crypto.createHash("sha256").update(fs.readFileSync(filePath)).digest("hex");
}

function hasSupportedImageSignature(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const ext = path.extname(filePath).toLowerCase();
    if (ext === ".png") {
      return buffer.length >= 8 && buffer.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
    }
    if (ext === ".jpg" || ext === ".jpeg") {
      return buffer.length >= 3 && buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[buffer.length - 2] === 0xff && buffer[buffer.length - 1] === 0xd9;
    }
    if (ext === ".tif" || ext === ".tiff") {
      return buffer.length >= 4 && (
        (buffer[0] === 0x49 && buffer[1] === 0x49 && buffer[2] === 0x2a && buffer[3] === 0x00)
        || (buffer[0] === 0x4d && buffer[1] === 0x4d && buffer[2] === 0x00 && buffer[3] === 0x2a)
      );
    }
    if (ext === ".bmp") {
      return buffer.length >= 2 && buffer[0] === 0x42 && buffer[1] === 0x4d;
    }
    return false;
  } catch {
    return false;
  }
}

function imageDimensions(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const ext = path.extname(filePath).toLowerCase();
    if (
      ext === ".png"
      && buffer.length >= 24
      && buffer.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))
      && buffer.subarray(12, 16).toString("ascii") === "IHDR"
    ) {
      return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
    }
    if (
      (ext === ".jpg" || ext === ".jpeg")
      && buffer.length >= 4
      && buffer[0] === 0xff
      && buffer[1] === 0xd8
    ) {
      let offset = 2;
      while (offset + 4 <= buffer.length) {
        if (buffer[offset] !== 0xff) {
          offset += 1;
          continue;
        }
        const marker = buffer[offset + 1];
        offset += 2;
        if (marker === 0xd9) break;
        if (marker === 0xd8 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
        if (offset + 2 > buffer.length) break;
        const segmentLength = buffer.readUInt16BE(offset);
        if (segmentLength < 2 || offset + segmentLength > buffer.length) break;
        const isStartOfFrame = (
          (marker >= 0xc0 && marker <= 0xc3)
          || (marker >= 0xc5 && marker <= 0xc7)
          || (marker >= 0xc9 && marker <= 0xcb)
          || (marker >= 0xcd && marker <= 0xcf)
        );
        if (isStartOfFrame && segmentLength >= 7) {
          return {
            width: buffer.readUInt16BE(offset + 5),
            height: buffer.readUInt16BE(offset + 3)
          };
        }
        offset += segmentLength;
      }
    }
    if (ext === ".bmp" && buffer.length >= 26 && buffer[0] === 0x42 && buffer[1] === 0x4d) {
      const dibHeaderSize = buffer.readUInt32LE(14);
      if (dibHeaderSize >= 40 && buffer.length >= 26) {
        return {
          width: Math.abs(buffer.readInt32LE(18)),
          height: Math.abs(buffer.readInt32LE(22))
        };
      }
      if (dibHeaderSize === 12 && buffer.length >= 26) {
        return {
          width: buffer.readUInt16LE(18),
          height: buffer.readUInt16LE(20)
        };
      }
    }
    if (
      (ext === ".tif" || ext === ".tiff")
      && buffer.length >= 8
      && (
        (buffer[0] === 0x49 && buffer[1] === 0x49 && buffer[2] === 0x2a && buffer[3] === 0x00)
        || (buffer[0] === 0x4d && buffer[1] === 0x4d && buffer[2] === 0x00 && buffer[3] === 0x2a)
      )
    ) {
      const littleEndian = buffer[0] === 0x49;
      const readUInt16 = (offset) => (littleEndian ? buffer.readUInt16LE(offset) : buffer.readUInt16BE(offset));
      const readUInt32 = (offset) => (littleEndian ? buffer.readUInt32LE(offset) : buffer.readUInt32BE(offset));
      const ifdOffset = readUInt32(4);
      if (ifdOffset + 2 <= buffer.length) {
        const entryCount = readUInt16(ifdOffset);
        let width = null;
        let height = null;
        for (let index = 0; index < entryCount; index += 1) {
          const entryOffset = ifdOffset + 2 + index * 12;
          if (entryOffset + 12 > buffer.length) break;
          const tag = readUInt16(entryOffset);
          const type = readUInt16(entryOffset + 2);
          const count = readUInt32(entryOffset + 4);
          if ((tag === 256 || tag === 257) && count >= 1) {
            let value = null;
            if (type === 3) value = readUInt16(entryOffset + 8);
            if (type === 4) value = readUInt32(entryOffset + 8);
            if (tag === 256) width = value;
            if (tag === 257) height = value;
          }
        }
        if (width !== null && height !== null) return { width, height };
      }
    }
  } catch {
    return null;
  }
  return null;
}

function pngPixelStats(buffer) {
  if (
    buffer.length < 33
    || !buffer.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))
  ) {
    return null;
  }
  let offset = 8;
  let width = null;
  let height = null;
  let bitDepth = null;
  let colorType = null;
  const idatChunks = [];
  while (offset + 12 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.subarray(offset + 4, offset + 8).toString("ascii");
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (dataEnd + 4 > buffer.length) return null;
    const data = buffer.subarray(dataStart, dataEnd);
    if (type === "IHDR") {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colorType = data[9];
      if (data[10] !== 0 || data[11] !== 0 || data[12] !== 0) return null;
    }
    if (type === "IDAT") idatChunks.push(data);
    if (type === "IEND") break;
    offset = dataEnd + 4;
  }
  const samplesByColorType = new Map([
    [0, 1],
    [2, 3],
    [4, 2],
    [6, 4]
  ]);
  const samplesPerPixel = samplesByColorType.get(colorType);
  if (!width || !height || bitDepth !== 8 || !samplesPerPixel || idatChunks.length === 0) return null;
  const inflated = zlib.inflateSync(Buffer.concat(idatChunks));
  const rowBytes = width * samplesPerPixel;
  const expectedBytes = (rowBytes + 1) * height;
  if (inflated.length < expectedBytes) return null;
  const previous = Buffer.alloc(rowBytes);
  const current = Buffer.alloc(rowBytes);
  const levels = new Set();
  let minimum = Infinity;
  let maximum = -Infinity;
  let inputOffset = 0;
  const paethPredictor = (left, up, upperLeft) => {
    const estimate = left + up - upperLeft;
    const leftDistance = Math.abs(estimate - left);
    const upDistance = Math.abs(estimate - up);
    const upperLeftDistance = Math.abs(estimate - upperLeft);
    if (leftDistance <= upDistance && leftDistance <= upperLeftDistance) return left;
    if (upDistance <= upperLeftDistance) return up;
    return upperLeft;
  };
  for (let y = 0; y < height; y += 1) {
    const filterType = inflated[inputOffset];
    inputOffset += 1;
    for (let x = 0; x < rowBytes; x += 1) {
      const raw = inflated[inputOffset + x];
      const left = x >= samplesPerPixel ? current[x - samplesPerPixel] : 0;
      const up = previous[x];
      const upperLeft = x >= samplesPerPixel ? previous[x - samplesPerPixel] : 0;
      if (filterType === 0) current[x] = raw;
      else if (filterType === 1) current[x] = (raw + left) & 0xff;
      else if (filterType === 2) current[x] = (raw + up) & 0xff;
      else if (filterType === 3) current[x] = (raw + Math.floor((left + up) / 2)) & 0xff;
      else if (filterType === 4) current[x] = (raw + paethPredictor(left, up, upperLeft)) & 0xff;
      else return null;
    }
    for (let x = 0; x < width; x += 1) {
      const pixelOffset = x * samplesPerPixel;
      let value = current[pixelOffset];
      if (colorType === 2 || colorType === 6) {
        value = Math.round(
          0.299 * current[pixelOffset]
          + 0.587 * current[pixelOffset + 1]
          + 0.114 * current[pixelOffset + 2]
        );
      }
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      levels.add(value);
    }
    current.copy(previous);
    inputOffset += rowBytes;
  }
  return {
    width,
    height,
    min: minimum,
    max: maximum,
    uniqueLevels: levels.size
  };
}

function bmpPixelStats(buffer) {
  if (buffer.length < 54 || buffer[0] !== 0x42 || buffer[1] !== 0x4d) return null;
  const pixelOffset = buffer.readUInt32LE(10);
  const dibHeaderSize = buffer.readUInt32LE(14);
  if (dibHeaderSize < 40) return null;
  const width = Math.abs(buffer.readInt32LE(18));
  const rawHeight = buffer.readInt32LE(22);
  const height = Math.abs(rawHeight);
  const planes = buffer.readUInt16LE(26);
  const bitDepth = buffer.readUInt16LE(28);
  const compression = buffer.readUInt32LE(30);
  if (!width || !height || planes !== 1 || bitDepth !== 24 || compression !== 0) return null;
  const rowBytes = Math.ceil((width * 3) / 4) * 4;
  if (pixelOffset + rowBytes * height > buffer.length) return null;
  const levels = new Set();
  let minimum = Infinity;
  let maximum = -Infinity;
  for (let y = 0; y < height; y += 1) {
    const rowOffset = pixelOffset + y * rowBytes;
    for (let x = 0; x < width; x += 1) {
      const offset = rowOffset + x * 3;
      const value = Math.round(
        0.114 * buffer[offset]
        + 0.587 * buffer[offset + 1]
        + 0.299 * buffer[offset + 2]
      );
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      levels.add(value);
    }
  }
  return {
    width,
    height,
    min: minimum,
    max: maximum,
    uniqueLevels: levels.size
  };
}

function imagePixelStats(filePath) {
  try {
    const buffer = fs.readFileSync(filePath);
    const ext = path.extname(filePath).toLowerCase();
    if (ext === ".png") return pngPixelStats(buffer);
    if (ext === ".bmp") return bmpPixelStats(buffer);
  } catch {
    return null;
  }
  return null;
}

function isPlainObject(value) {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function loadIteratedRofSourceRegistry(id, options = {}) {
  const registryPath = options.registryPath || iteratedRofSourceRegistryPath;
  const differenceLabel = options.differenceLabel || `${id}.paperLikeVerification`;
  const differences = [];
  let registry = null;
  try {
    registry = readJson(registryPath);
  } catch (error) {
    return {
      sourceIdsByFamily: new Map(),
      differences: [`${differenceLabel}: Iterated ROF source registry is not readable: ${error.message}`]
    };
  }
  if (!isPlainObject(registry)) {
    return {
      sourceIdsByFamily: new Map(),
      differences: [`${differenceLabel}: Iterated ROF source registry must be an object keyed by family`]
    };
  }
  const sourceIdsByFamily = new Map();
  const sourceUrlsByFamilyAndId = new Map();
  const seenSourceIds = new Map();
  for (const family of Object.keys(registry)) {
    if (!requiredPaperLikeFamilies.includes(family)) {
      differences.push(`${differenceLabel}: Iterated ROF source registry has unknown family ${family}`);
    }
  }
  for (const family of requiredPaperLikeFamilies) {
    const entries = registry[family];
    if (!Array.isArray(entries) || entries.length === 0) {
      differences.push(`${differenceLabel}: Iterated ROF source registry has no source entries for family ${family}`);
      sourceIdsByFamily.set(family, new Set());
      continue;
    }
    const sourceIds = [];
    for (const [index, entry] of entries.entries()) {
      const label = `${family}[${index}]`;
      if (!isPlainObject(entry)) {
        differences.push(`${differenceLabel}: Iterated ROF source registry entry is not an object for ${label}`);
        continue;
      }
      for (const field of requiredSourceRegistryTextFields) {
        if (typeof entry[field] !== "string" || !entry[field].trim()) {
          differences.push(`${differenceLabel}: Iterated ROF source registry missing ${field} for ${label}`);
        }
      }
      if (entry.target_family !== family) {
        differences.push(`${differenceLabel}: source registry target_family mismatch for ${label}`);
      }
      if (!Number.isInteger(entry.priority)) {
        differences.push(`${differenceLabel}: Iterated ROF source registry priority must be an integer for ${label}`);
      }
      const sourceId = typeof entry.source_id === "string" ? entry.source_id.trim() : "";
      if (sourceId) {
        if (seenSourceIds.has(sourceId)) {
          differences.push(`${differenceLabel}: Iterated ROF source registry duplicate source_id ${sourceId}`);
        } else {
          seenSourceIds.set(sourceId, label);
        }
        sourceIds.push(sourceId);
        const urls = [entry.url, entry.download_url]
          .filter((value) => typeof value === "string" && value.trim())
          .map((value) => value.trim());
        sourceUrlsByFamilyAndId.set(`${family}\u0000${sourceId}`, new Set(urls));
      }
    }
    sourceIdsByFamily.set(family, new Set(sourceIds));
  }
  return { sourceIdsByFamily, sourceUrlsByFamilyAndId, differences };
}

function iteratedRofSourceRegistryOverridePathFromEnv(env = process.env) {
  const overridePath = typeof env?.ITERATED_ROF_SOURCE_REGISTRY_PATH === "string"
    ? env.ITERATED_ROF_SOURCE_REGISTRY_PATH.trim()
    : "";
  return overridePath ? resolveLocalArtifactPath(overridePath) : null;
}

function sameResolvedPath(left, right) {
  return Boolean(left && right && path.resolve(left) === path.resolve(right));
}

function iteratedRofSourceRegistryDifferences(env = process.env, options = {}) {
  const canonicalRegistryPath = options.iteratedRofCanonicalSourceRegistryPath || iteratedRofSourceRegistryPath;
  const differences = loadIteratedRofSourceRegistry("iterated-rof", {
    registryPath: canonicalRegistryPath,
    differenceLabel: "iterated-rof.sourceRegistry"
  }).differences;
  const overrideRegistryPath = iteratedRofSourceRegistryOverridePathFromEnv(env);
  if (overrideRegistryPath && !sameResolvedPath(overrideRegistryPath, canonicalRegistryPath)) {
    differences.push(
      ...loadIteratedRofSourceRegistry("iterated-rof", {
        registryPath: overrideRegistryPath,
        differenceLabel: "iterated-rof.sourceRegistry.override"
      }).differences
    );
  }
  return differences;
}

function nestedMetric(object, pathParts) {
  let current = object;
  for (const part of pathParts) {
    if (!isPlainObject(current)) return undefined;
    current = current[part];
  }
  return current;
}

function hasNumericMetric(object, pathParts) {
  return isFiniteNumber(nestedMetric(object, pathParts));
}

function isIsoDate(value) {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const [year, month, day] = value.split("-").map((part) => Number.parseInt(part, 10));
  if (year < 1) return false;
  const parsed = new Date(Date.UTC(year, month - 1, day));
  parsed.setUTCFullYear(year);
  return (
    parsed.getUTCFullYear() === year
    && parsed.getUTCMonth() === month - 1
    && parsed.getUTCDate() === day
  );
}

function sourceAuditDifferences(id, label, audit, family) {
  const differences = [];
  if (!isPlainObject(audit)) {
    return [`${id}.paperLikeVerification: source summary source audit must be an object for: ${label}`];
  }
  for (const field of ["source_url", "downloaded_at", "conversion_notes"]) {
    if (typeof audit[field] !== "string" || !audit[field].trim()) {
      differences.push(`${id}.paperLikeVerification: source summary source audit missing ${field} for: ${label}`);
    }
  }
  if (audit.downloaded_at && !isIsoDate(audit.downloaded_at)) {
    differences.push(`${id}.paperLikeVerification: source summary source audit downloaded_at must use a valid YYYY-MM-DD date for: ${label}`);
  }
  for (const field of ["source_artifact_path", "license_snapshot_path"]) {
    if (typeof audit[field] !== "string" || !audit[field].trim()) {
      differences.push(`${id}.paperLikeVerification: source summary source audit missing ${field} for: ${label}`);
    }
  }
  differences.push(
    ...sourceAuditArtifactDifferences(id, label, audit, "source_artifact_path", "source_artifact_sha256", "source_artifact", family)
  );
  differences.push(
    ...sourceAuditArtifactDifferences(id, label, audit, "license_snapshot_path", "license_snapshot_sha256", "license_snapshot", family)
  );
  if (audit.local_file_mapping_reviewed !== true) {
    differences.push(`${id}.paperLikeVerification: source summary source audit local_file_mapping_reviewed must be true for: ${label}`);
  }
  return differences;
}

function sourceAuditArtifactDifferences(id, label, audit, pathField, shaField, artifactLabel, family) {
  const differences = [];
  const artifactPath = resolveLocalArtifactPath(audit?.[pathField]);
  const expectedSha = audit?.[shaField];
  if (!isSha256(expectedSha)) {
    differences.push(`${id}.paperLikeVerification: source summary source audit missing ${shaField} for: ${label}`);
    return differences;
  }
  if (!artifactPath) return differences;
  if (!fs.existsSync(artifactPath)) {
    differences.push(`${id}.paperLikeVerification: source summary source audit ${artifactLabel} file is missing for: ${label}`);
    return differences;
  }
  const familyAuditRoot = requiredPaperLikeFamilies.includes(family)
    ? path.join(iteratedRofDataRoot, family, "audit")
    : null;
  if (!isPathUnder(artifactPath, iteratedRofDataRoot)) {
    differences.push(`${id}.paperLikeVerification: source summary source audit ${artifactLabel} path is outside canonical local Iterated ROF data root for: ${label}`);
    return differences;
  }
  if (!familyAuditRoot || !isPathUnder(artifactPath, familyAuditRoot)) {
    differences.push(`${id}.paperLikeVerification: source summary source audit ${artifactLabel} path is outside canonical local Iterated ROF family audit root for: ${label}`);
    return differences;
  }
  try {
    const stat = fs.statSync(artifactPath);
    const actualSha = sha256File(artifactPath);
    if (actualSha !== expectedSha) {
      differences.push(`${id}.paperLikeVerification: source summary source audit ${artifactLabel} sha256 mismatch for: ${label}`);
    }
    const tooSmall = stat.size < minSourceAuditArtifactBytes;
    if (tooSmall) {
      differences.push(`${id}.paperLikeVerification: source summary source audit ${artifactLabel} artifact is too small to support review evidence for: ${label}`);
    }
    const text = fs.readFileSync(artifactPath).subarray(0, 8192).toString("utf8").toLowerCase();
    const hasPlaceholder = sourceAuditArtifactPlaceholderPatterns.some((pattern) => text.includes(pattern));
    if (hasPlaceholder) {
      differences.push(`${id}.paperLikeVerification: source summary source audit ${artifactLabel} artifact contains fixture/placeholder text for: ${label}`);
    }
    if (!tooSmall && !hasPlaceholder && sourceAuditArtifactStructureIssues(text, audit?.source_url).length > 0) {
      differences.push(`${id}.paperLikeVerification: source summary source audit ${artifactLabel} artifact is missing structured review evidence for: ${label}`);
    }
  } catch (error) {
    differences.push(`${id}.paperLikeVerification: source summary source audit ${artifactLabel} file is not readable for: ${label}: ${error.message}`);
  }
  return differences;
}

function sourceAuditArtifactStructureIssues(text, expectedSourceUrl) {
  const issues = [];
  const hasSourceUrl = /https?:\/\/\S+/.test(text) || /\bsource_url\s*=\s*\S+/.test(text);
  const normalizedExpectedSourceUrl = typeof expectedSourceUrl === "string" ? expectedSourceUrl.trim().toLowerCase() : "";
  if (!hasSourceUrl) {
    issues.push("missing_source_url");
  } else if (normalizedExpectedSourceUrl && !text.includes(normalizedExpectedSourceUrl)) {
    issues.push("missing_manifest_source_url");
  }
  const dateTokens = text.match(/\b\d{4}-\d{2}-\d{2}\b/g) || [];
  if (dateTokens.length === 0) {
    issues.push("missing_review_date");
  } else if (!dateTokens.some((token) => isIsoDate(token))) {
    issues.push("invalid_review_date");
  }
  if (!sourceAuditArtifactReviewNotePattern.test(text)) {
    issues.push("missing_review_note");
  }
  if (!sourceAuditArtifactMappingPattern.test(text)) {
    issues.push("missing_conversion_or_mapping_note");
  }
  return issues;
}

function sourceAuditUrlDifferences(id, label, audit, family, sourceId, sourceRegistry) {
  const sourceUrl = typeof audit?.source_url === "string" ? audit.source_url.trim() : "";
  if (!sourceUrl || !family || !sourceId) return [];
  const registeredUrls = sourceRegistry?.sourceUrlsByFamilyAndId?.get(`${family}\u0000${sourceId}`);
  if (registeredUrls?.size && !registeredUrls.has(sourceUrl)) {
    return [`${id}.paperLikeVerification: source summary source audit source_url is not registered for source_id ${sourceId} for: ${label}`];
  }
  return [];
}

function fileEvidenceDifferences(id, label, pathValue, evidence) {
  const differences = [];
  const artifactPath = resolveLocalArtifactPath(pathValue);
  if (!artifactPath) {
    return [`${id}.paperLikeVerification: source summary ${label} path is missing`];
  }
  if (!isPlainObject(evidence) || !isSha256(evidence.sha256) || !isFiniteNumber(evidence.size_bytes) || evidence.size_bytes <= 0) {
    differences.push(`${id}.paperLikeVerification: source summary ${label} file evidence is incomplete`);
  }
  if (!fs.existsSync(artifactPath)) {
    differences.push(`${id}.paperLikeVerification: source summary ${label} file is missing: ${pathValue}`);
    return differences;
  }
  try {
    const stat = fs.statSync(artifactPath);
    const actualSha = sha256File(artifactPath);
    if (isPlainObject(evidence) && evidence.sha256 && actualSha !== evidence.sha256) {
      differences.push(`${id}.paperLikeVerification: source summary ${label} sha256 mismatch`);
    }
    if (isPlainObject(evidence) && isFiniteNumber(evidence.size_bytes) && stat.size !== evidence.size_bytes) {
      differences.push(`${id}.paperLikeVerification: source summary ${label} size mismatch`);
    }
  } catch (error) {
    differences.push(`${id}.paperLikeVerification: source summary ${label} file is not readable: ${error.message}`);
  }
  return differences;
}

function imageArtifactDifferences(id, label, pathValue, evidence, options = {}) {
  const differences = fileEvidenceDifferences(id, label, pathValue, evidence);
  const artifactPath = resolveLocalArtifactPath(pathValue);
  if (artifactPath && fs.existsSync(artifactPath) && !hasSupportedImageSignature(artifactPath)) {
    differences.push(`${id}.paperLikeVerification: source summary ${label} is not a supported image file`);
  }
  if (options.requirePaperLikeSize && artifactPath && fs.existsSync(artifactPath) && hasSupportedImageSignature(artifactPath)) {
    const dimensions = imageDimensions(artifactPath);
    if (!dimensions) {
      differences.push(`${id}.paperLikeVerification: source summary ${label} dimensions are not readable for paper-like evidence`);
    }
    if (
      dimensions
      && (dimensions.width < minPaperLikeImageSide || dimensions.height < minPaperLikeImageSide)
    ) {
      differences.push(`${id}.paperLikeVerification: source summary ${label} is too small for paper-like evidence`);
    }
  }
  if (options.contentKind && artifactPath && fs.existsSync(artifactPath) && hasSupportedImageSignature(artifactPath)) {
    const ext = path.extname(artifactPath).toLowerCase();
    if (ext === ".png" || ext === ".bmp") {
      const stats = imagePixelStats(artifactPath);
      if (!stats) {
        differences.push(`${id}.paperLikeVerification: source summary ${label} content is not decodable for paper-like evidence`);
      } else if (options.contentKind === "input") {
        if (stats.min === stats.max) {
          differences.push(`${id}.paperLikeVerification: source summary input image is visually blank for paper-like evidence for: ${label}`);
        }
        if (stats.uniqueLevels < minPaperLikeImageLevels) {
          differences.push(`${id}.paperLikeVerification: source summary input image has too few gray levels for paper-like evidence for: ${label}`);
        }
      } else if (options.contentKind === "mask") {
        if (stats.uniqueLevels < 2) {
          differences.push(`${id}.paperLikeVerification: source summary mask has fewer than two labels for paper-like evidence for: ${label}`);
        }
      } else if (options.contentKind === "figure" && stats.min === stats.max) {
        differences.push(`${id}.paperLikeVerification: source summary figure file is visually blank for: ${label}`);
      }
    }
  }
  return differences;
}

function maskShapeDifferences(id, label, imagePathValue, maskPathValue) {
  const imagePath = resolveLocalArtifactPath(imagePathValue);
  const maskPath = resolveLocalArtifactPath(maskPathValue);
  if (
    !imagePath
    || !maskPath
    || !fs.existsSync(imagePath)
    || !fs.existsSync(maskPath)
    || !hasSupportedImageSignature(imagePath)
    || !hasSupportedImageSignature(maskPath)
  ) {
    return [];
  }
  const imageShape = imageDimensions(imagePath);
  const maskShape = imageDimensions(maskPath);
  if (!imageShape || !maskShape) return [];
  if (imageShape.width !== maskShape.width || imageShape.height !== maskShape.height) {
    return [
      `${id}.paperLikeVerification: source summary mask shape does not match image shape for: ${label}`
    ];
  }
  return [];
}

function canonicalFamilyRelativePath(item, pathField, subdirName) {
  const artifactPath = resolveLocalArtifactPath(item?.[pathField]);
  const family = item?.family;
  if (artifactPath && requiredPaperLikeFamilies.includes(family)) {
    const familyRoot = path.join(iteratedRofDataRoot, family, subdirName);
    if (isPathUnder(artifactPath, familyRoot)) {
      return path.relative(familyRoot, artifactPath).split(path.sep).join("/");
    }
  }
  const rawPath = String(item?.[pathField] || "");
  const marker = `/${subdirName}/`;
  return rawPath.includes(marker) ? rawPath.slice(rawPath.lastIndexOf(marker) + marker.length) : path.basename(rawPath);
}

function imageRelativePath(item) {
  return canonicalFamilyRelativePath(item, "image_path", "images");
}

function maskRelativePath(item) {
  return canonicalFamilyRelativePath(item, "mask_path", "masks");
}

function figureBaselineEvidence(baselines) {
  if (!isPlainObject(baselines)) return {};
  return Object.fromEntries(
    Object.entries(baselines)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([name, baseline]) => {
        const evidence = {};
        if (isPlainObject(baseline)) {
          for (const key of ["method", "thresholds"]) {
            if (key in baseline) evidence[key] = baseline[key];
          }
        }
        return [name, evidence];
      })
  );
}

function expectedFigureEvidencePayload(item) {
  const figureFile = isPlainObject(item?.figure_file) ? item.figure_file : {};
  const imageFile = isPlainObject(item?.image_file) ? item.image_file : {};
  const maskFile = isPlainObject(item?.mask_file) ? item.mask_file : {};
  return {
    schema_version: 1,
    paper_id: "iterated-rof",
    generator: iteratedRofFigureEvidenceGenerator,
    family: item?.family,
    image_path: item?.image_path,
    mask_path: item?.mask_path || "",
    figure_path: item?.figure_path,
    qualitative_only: Boolean(item?.qualitative_only),
    image_sha256: imageFile.sha256 || "",
    mask_sha256: item?.mask_path ? maskFile.sha256 || "" : "",
    figure_sha256: figureFile.sha256 || "",
    figure_size_bytes: figureFile.size_bytes || "",
    figure_panels: Array.isArray(item?.figure_panels) ? [...item.figure_panels] : [],
    solver: item?.solver || "",
    parameters: isPlainObject(item?.parameters) ? item.parameters : {},
    thresholds: Array.isArray(item?.thresholds) ? [...item.thresholds] : [],
    n_classes: item?.n_classes ?? "",
    metrics: isPlainObject(item?.metrics) ? item.metrics : {},
    baselines: figureBaselineEvidence(item?.baselines)
  };
}

function figureEvidencePayloadDifferences(id, label, item) {
  const differences = [];
  const sidecarPath = resolveLocalArtifactPath(item?.figure_evidence_path);
  let diskPayload = null;
  if (!sidecarPath || !fs.existsSync(sidecarPath)) return differences;
  try {
    diskPayload = readJson(sidecarPath);
  } catch (error) {
    differences.push(`${id}.paperLikeVerification: source summary figure evidence sidecar is not readable JSON for: ${label}: ${error.message}`);
    return differences;
  }
  const expectedPayload = expectedFigureEvidencePayload(item);
  if (!sameJson(diskPayload, item?.figure_evidence) || !sameJson(item?.figure_evidence, expectedPayload)) {
    differences.push(`${id}.paperLikeVerification: source summary figure evidence sidecar does not match report for: ${label}`);
  }
  return differences;
}

function datasetFingerprintFromSummaryImages(images) {
  const records = [];
  for (const item of images) {
    if (!isPlainObject(item)) continue;
    if (isSha256(item.image_file?.sha256)) {
      records.push({
        kind: "image",
        family: item.family,
        path: imageRelativePath(item),
        sha256: item.image_file.sha256
      });
    }
    if (item.mask_path && isSha256(item.mask_file?.sha256)) {
      records.push({
        kind: "mask",
        family: item.family,
        path: maskRelativePath(item),
        sha256: item.mask_file.sha256
      });
    }
  }
  const sortedRecords = records.sort((left, right) => {
    const leftKey = [left.family, left.path, left.kind === "image" ? 0 : 1].join("\t");
    const rightKey = [right.family, right.path, right.kind === "image" ? 0 : 1].join("\t");
    return leftKey.localeCompare(rightKey);
  });
  const digest = crypto.createHash("sha256");
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

function isCompletedQuantitativeSummaryImage(item) {
  return Boolean(
    isPlainObject(item)
    && item.status === "completed"
    && item.qualitative_only === false
    && typeof item.mask_path === "string"
    && item.mask_path.trim()
  );
}

const manifestReviewClaimFields = [
  "source_id",
  "source_name",
  "license_reviewed",
  "license_note",
  "citation",
  "provenance_reviewed",
  "provenance_note",
  "synthetic_fixture",
  "notes",
  "source_audit"
];

function effectiveManifestReviewClaim(familyManifest, fileClaim) {
  const claim = {};
  for (const field of manifestReviewClaimFields) {
    if (field in familyManifest) claim[field] = familyManifest[field];
  }
  if (isPlainObject(fileClaim)) {
    for (const field of manifestReviewClaimFields) {
      if (field in fileClaim) claim[field] = fileClaim[field];
    }
  }
  return claim;
}

function paperLikeManifestClaimDifferences(id, item, manifest, sourceRegistry) {
  const differences = [];
  const label = item?.image_path || "<unknown image>";
  const sourceClaim = isPlainObject(item?.source_claim) ? item.source_claim : {};
  const familyManifest = isPlainObject(manifest?.families?.[item?.family]) ? manifest.families[item.family] : null;
  if (!familyManifest) {
    differences.push(`${id}.paperLikeVerification: source summary source claim is missing from canonical local dataset_manifest for: ${label}`);
    return differences;
  }
  const fileClaims = Array.isArray(familyManifest.files) ? familyManifest.files : [];
  const fileClaim = fileClaims.find((claim) => isPlainObject(claim) && claim.image === sourceClaim.image);
  const manifestClaim = effectiveManifestReviewClaim(familyManifest, fileClaim);
  if (
    manifestClaim.source_id !== sourceClaim.source_id
    || manifestClaim.license_reviewed !== true
    || manifestClaim.provenance_reviewed !== true
    || manifestClaim.synthetic_fixture !== false
  ) {
    differences.push(`${id}.paperLikeVerification: source summary source claim does not match reviewed canonical manifest family claim for: ${label}`);
  }
  for (const field of ["citation", "license_note", "provenance_note"]) {
    const manifestValue = typeof manifestClaim[field] === "string" ? manifestClaim[field].trim() : "";
    if (!manifestValue) {
      differences.push(`${id}.paperLikeVerification: canonical local dataset_manifest source text is incomplete for: ${label}`);
    } else if (manifestValue !== sourceClaim[field]) {
      differences.push(`${id}.paperLikeVerification: source summary source claim text does not match canonical local dataset_manifest for: ${label}`);
    }
  }
  if (!sameJson(manifestClaim.source_audit || null, sourceClaim.source_audit || null)) {
    differences.push(`${id}.paperLikeVerification: source summary source audit does not match canonical local dataset_manifest for: ${label}`);
  }
  const registeredSourceIds = sourceRegistry?.sourceIdsByFamily?.get(item?.family);
  if (!registeredSourceIds || !registeredSourceIds.has(sourceClaim.source_id)) {
    differences.push(`${id}.paperLikeVerification: source summary source_id is not in source registry for: ${label}`);
  }
  differences.push(
    ...sourceAuditUrlDifferences(
      id,
      label,
      sourceClaim.source_audit,
      item?.family,
      sourceClaim.source_id,
      sourceRegistry
    )
  );
  if (!fileClaim) {
    differences.push(`${id}.paperLikeVerification: source summary source claim image is not listed in canonical local dataset_manifest for: ${label}`);
    return differences;
  }
  if (
    fileClaim.sha256 !== item?.image_file?.sha256
    || fileClaim.mask !== sourceClaim.mask
    || fileClaim.mask_sha256 !== item?.mask_file?.sha256
  ) {
    differences.push(`${id}.paperLikeVerification: source summary source claim file hashes do not match canonical local dataset_manifest for: ${label}`);
  }
  return differences;
}

function paperLikeCanonicalPathDifferences(id, item) {
  const differences = [];
  const label = item?.image_path || "<unknown image>";
  const family = item?.family;
  const imagePath = resolveLocalArtifactPath(item?.image_path);
  const maskPath = resolveLocalArtifactPath(item?.mask_path);
  const figurePath = resolveLocalArtifactPath(item?.figure_path);
  const sidecarPath = resolveLocalArtifactPath(item?.figure_evidence_path);
  if (!requiredPaperLikeFamilies.includes(family)) return differences;
  if (!isPathUnder(imagePath, path.join(iteratedRofDataRoot, family, "images"))) {
    differences.push(`${id}.paperLikeVerification: source summary image path is outside canonical local Iterated ROF data root for: ${label}`);
  }
  if (!isPathUnder(maskPath, path.join(iteratedRofDataRoot, family, "masks"))) {
    differences.push(`${id}.paperLikeVerification: source summary mask path is outside canonical local Iterated ROF data root for: ${label}`);
  }
  if (!isPathUnder(figurePath, iteratedRofFigureRoot)) {
    differences.push(`${id}.paperLikeVerification: source summary figure path is outside canonical local Iterated ROF figure root for: ${label}`);
  }
  if (!sidecarPath || !figurePath || path.resolve(sidecarPath) !== `${path.resolve(figurePath)}.evidence.json`) {
    differences.push(`${id}.paperLikeVerification: source summary figure evidence sidecar must sit next to the canonical figure for: ${label}`);
  }
  return differences;
}

function paperLikeSummaryImageDifferences(id, item, manifest = null, sourceRegistry = null) {
  const differences = [];
  const label = item?.image_path || "<unknown image>";
  const sourceClaim = isPlainObject(item?.source_claim) ? item.source_claim : {};
  if (!requiredPaperLikeFamilies.includes(item?.family)) {
    differences.push(`${id}.paperLikeVerification: source summary image has unknown family for: ${label}`);
  }
  if (!item?.image_path) differences.push(`${id}.paperLikeVerification: source summary image_path is missing for: ${label}`);
  if (!item?.mask_path) differences.push(`${id}.paperLikeVerification: source summary mask_path is missing for: ${label}`);
  if (item?.qualitative_only !== false) {
    differences.push(`${id}.paperLikeVerification: source summary quantitative image must set qualitative_only=false for: ${label}`);
  }
  if (!hasNumericMetric(item, ["metrics", "clustering_accuracy"])) {
    differences.push(`${id}.paperLikeVerification: source summary T-ROF clustering_accuracy is missing for: ${label}`);
  }
  if (!hasNumericMetric(item, ["baselines", "raw_kmeans", "metrics", "clustering_accuracy"])) {
    differences.push(`${id}.paperLikeVerification: source summary raw_kmeans clustering_accuracy is missing for: ${label}`);
  }
  if (!hasNumericMetric(item, ["baselines", "multi_otsu", "metrics", "clustering_accuracy"])) {
    differences.push(`${id}.paperLikeVerification: source summary multi_otsu clustering_accuracy is missing for: ${label}`);
  }
  if (item?.solver !== "sat_rof_trof.rof_chambolle_pock + sat_rof_trof.run_trof_thresholds") {
    differences.push(`${id}.paperLikeVerification: source summary solver evidence is missing for: ${label}`);
  }
  if (!Array.isArray(item?.thresholds) || item.thresholds.length === 0 || !item.thresholds.every(isFiniteNumber)) {
    differences.push(`${id}.paperLikeVerification: source summary threshold evidence is missing for: ${label}`);
  }
  if (!isFiniteNumber(item?.threshold_iterations) || item.threshold_iterations <= 0) {
    differences.push(`${id}.paperLikeVerification: source summary threshold iteration evidence is missing for: ${label}`);
  }
  if (!isFiniteNumber(item?.rof_iterations) || item.rof_iterations <= 0) {
    differences.push(`${id}.paperLikeVerification: source summary ROF iteration evidence is missing for: ${label}`);
  }
  if (!isFiniteNumber(item?.rof_final_residual) || item.rof_final_residual < 0) {
    differences.push(`${id}.paperLikeVerification: source summary ROF residual evidence is missing for: ${label}`);
  }
  if (sourceClaim.manifest_status !== "present" || sourceClaim.claim_scope !== "file") {
    differences.push(`${id}.paperLikeVerification: source summary file-level source claim is missing for: ${label}`);
  }
  if (sourceClaim.license_reviewed !== true || sourceClaim.provenance_reviewed !== true || sourceClaim.synthetic_fixture !== false) {
    differences.push(`${id}.paperLikeVerification: source summary reviewed source claim is incomplete for: ${label}`);
  }
  if (!sourceClaim.source_id || !sourceClaim.citation || !sourceClaim.license_note || !sourceClaim.provenance_note) {
    differences.push(`${id}.paperLikeVerification: source summary source claim text is incomplete for: ${label}`);
  }
  differences.push(...sourceAuditDifferences(id, label, sourceClaim.source_audit, item?.family));
  if (sourceClaim.image !== imageRelativePath(item)) {
    differences.push(`${id}.paperLikeVerification: source summary source claim image path does not match for: ${label}`);
  }
  if (sourceClaim.mask !== maskRelativePath(item)) {
    differences.push(`${id}.paperLikeVerification: source summary source claim mask path does not match for: ${label}`);
  }
  if (sourceClaim.sha256 !== item?.image_file?.sha256) {
    differences.push(`${id}.paperLikeVerification: source summary source claim image sha256 does not match for: ${label}`);
  }
  if (sourceClaim.mask_sha256 !== item?.mask_file?.sha256) {
    differences.push(`${id}.paperLikeVerification: source summary source claim mask sha256 does not match for: ${label}`);
  }
  differences.push(...paperLikeCanonicalPathDifferences(id, item));
  if (manifest) differences.push(...paperLikeManifestClaimDifferences(id, item, manifest, sourceRegistry));
  differences.push(
    ...imageArtifactDifferences(id, `image ${label}`, item?.image_path, item?.image_file, {
      requirePaperLikeSize: true,
      contentKind: "input"
    })
  );
  differences.push(
    ...imageArtifactDifferences(id, `mask ${label}`, item?.mask_path, item?.mask_file, {
      requirePaperLikeSize: true,
      contentKind: "mask"
    })
  );
  differences.push(...maskShapeDifferences(id, label, item?.image_path, item?.mask_path));
  differences.push(
    ...imageArtifactDifferences(id, `figure ${label}`, item?.figure_path, item?.figure_file, {
      contentKind: "figure"
    })
  );
  if (!isPlainObject(item?.figure_evidence) || !item?.figure_evidence_path) {
    differences.push(`${id}.paperLikeVerification: source summary figure evidence sidecar is missing for: ${label}`);
  } else {
    differences.push(
      ...fileEvidenceDifferences(id, `figure evidence ${label}`, item.figure_evidence_path, item.figure_evidence_file)
    );
    differences.push(...figureEvidencePayloadDifferences(id, label, item));
  }
  return differences;
}

function sortedStrings(values) {
  return [...values].sort((left, right) => String(left).localeCompare(String(right)));
}

function sourceClaimSortKey(claim) {
  if (!isPlainObject(claim)) return "";
  return [
    claim.image || "",
    claim.mask || "",
    claim.source_id || "",
    claim.sha256 || "",
    claim.mask_sha256 || ""
  ].join("\t");
}

function sortedSourceClaims(claims) {
  return [...claims].sort((left, right) => sourceClaimSortKey(left).localeCompare(sourceClaimSortKey(right)));
}

function paperLikeFamilySummaryConsistencyDifferences(id, familySummary, images) {
  const family = familySummary?.family || "<unknown family>";
  const differences = [];
  if (!isPlainObject(familySummary)) {
    differences.push(`${id}.paperLikeVerification: source summary family_summaries entry is not an object for: ${family}`);
    return differences;
  }
  const familyImages = images.filter((item) => isPlainObject(item) && item.family === family && item.status === "completed");
  const expectedFigurePaths = sortedStrings(
    familyImages
      .map((item) => item.figure_path)
      .filter((value) => typeof value === "string" && value.trim())
  );
  const actualFigurePaths = Array.isArray(familySummary.figure_paths)
    ? sortedStrings(familySummary.figure_paths)
    : null;
  if (!sameJson(actualFigurePaths, expectedFigurePaths)) {
    differences.push(`${id}.paperLikeVerification: source summary family_summaries figure_paths do not match image evidence rows for: ${family}`);
  }

  const expectedSourceClaims = sortedSourceClaims(
    familyImages
      .map((item) => item.source_claim)
      .filter(isPlainObject)
  );
  const actualSourceClaims = Array.isArray(familySummary.source_claims)
    ? sortedSourceClaims(familySummary.source_claims)
    : null;
  if (!sameJson(actualSourceClaims, expectedSourceClaims)) {
    differences.push(`${id}.paperLikeVerification: source summary family_summaries source_claims do not match image evidence rows for: ${family}`);
  }
  return differences;
}

function paperLikeSourceSummaryDifferences(id, sourceSummary, gate, options = {}) {
  const differences = [];
  const gateEvidence = gate?.evidence_summary || gate?.evidenceSummary || {};
  const images = Array.isArray(sourceSummary.images) ? sourceSummary.images : [];
  const sourceRegistryPath = options.iteratedRofSourceRegistryPath || options.sourceRegistryPath;
  const sourceRegistry = loadIteratedRofSourceRegistry(
    id,
    sourceRegistryPath ? { registryPath: sourceRegistryPath } : {}
  );
  differences.push(...sourceRegistry.differences);
  const manifestPath = resolveLocalArtifactPath(sourceSummary.local_dataset_manifest?.path);
  let manifest = null;
  if (!manifestPath || path.resolve(manifestPath) !== path.resolve(iteratedRofManifestPath)) {
    differences.push(`${id}.paperLikeVerification: source summary must reference canonical local dataset_manifest at reproduce/data/iterated_rof/dataset_manifest.json`);
  } else if (!fs.existsSync(manifestPath)) {
    differences.push(`${id}.paperLikeVerification: canonical local dataset_manifest is missing`);
  } else {
    try {
      manifest = readJson(manifestPath);
    } catch (error) {
      differences.push(`${id}.paperLikeVerification: canonical local dataset_manifest is not readable JSON: ${error.message}`);
    }
  }
  const completedImages = images.filter((item) => isPlainObject(item) && item.status === "completed");
  const quantitativeImages = images.filter(isCompletedQuantitativeSummaryImage);
  if (
    quantitativeImages.length < requiredPaperLikeFamilies.length
    || !includesAllRequired(quantitativeImages.map((item) => item.family), requiredPaperLikeFamilies)
  ) {
    differences.push(`${id}.paperLikeVerification: source summary must include completed quantitative image evidence rows with masks for all required families`);
  }
  if (sourceSummary.image_count !== images.length) {
    differences.push(`${id}.paperLikeVerification: source summary image_count does not match image rows`);
  }
  if (sourceSummary.completed_image_count !== completedImages.length) {
    differences.push(`${id}.paperLikeVerification: source summary completed_image_count does not match image rows`);
  }
  if (sourceSummary.quantitative_image_count !== quantitativeImages.length) {
    differences.push(`${id}.paperLikeVerification: source summary quantitative_image_count does not match image rows`);
  }
  if (gateEvidence.image_count !== images.length || gateEvidence.completed_image_count !== completedImages.length || gateEvidence.quantitative_image_count !== quantitativeImages.length) {
    differences.push(`${id}.paperLikeVerification: source summary image counts do not match gate evidence`);
  }
  if (gateEvidence.source_claim_count !== quantitativeImages.length) {
    differences.push(`${id}.paperLikeVerification: source summary source_claim_count does not match source summary quantitative image rows`);
  }
  if (gateEvidence.figure_evidence_count !== quantitativeImages.length) {
    differences.push(`${id}.paperLikeVerification: source summary figure_evidence_count does not match source summary quantitative image rows`);
  }
  if (sourceSummary.local_dataset_manifest?.status !== "present") {
    differences.push(`${id}.paperLikeVerification: source summary must include present local_dataset_manifest`);
  }
  if (sourceSummary.run_protocol?.protocol_id !== "iterated_rof_trof_local_data_v1") {
    differences.push(`${id}.paperLikeVerification: source summary must include Iterated ROF run_protocol`);
  }
  if (!Array.isArray(sourceSummary.family_summaries) || !includesAllRequired(sourceSummary.family_summaries.map((item) => item.family), requiredPaperLikeFamilies)) {
    differences.push(`${id}.paperLikeVerification: source summary must include family_summaries for all required families`);
  } else {
    for (const familySummary of sourceSummary.family_summaries) {
      differences.push(...paperLikeFamilySummaryConsistencyDifferences(id, familySummary, images));
    }
  }
  const computedFingerprint = datasetFingerprintFromSummaryImages(images);
  if (!sameJson(computedFingerprint, sourceSummary.dataset_fingerprint)) {
    differences.push(`${id}.paperLikeVerification: source summary dataset_fingerprint does not match image/mask evidence rows`);
  }
  if (!sameJson(computedFingerprint, gateEvidence.dataset_fingerprint)) {
    differences.push(`${id}.paperLikeVerification: source summary image/mask evidence does not match gate dataset_fingerprint`);
  }
  for (const item of quantitativeImages) {
    differences.push(...paperLikeSummaryImageDifferences(id, item, manifest, sourceRegistry));
  }
  return differences;
}

function paperLikeResultFileForFigure(figurePathValue) {
  const figurePath = resolveLocalArtifactPath(figurePathValue);
  if (!figurePath || !isPathUnder(figurePath, iteratedRofFigureRoot)) return null;
  const relative = path.relative(path.resolve(iteratedRofFigureRoot), path.resolve(figurePath)).split(path.sep).join("/");
  if (!relative || relative.startsWith("../") || relative.includes("/../") || path.posix.isAbsolute(relative)) return null;
  return `assets/repro/iterated_rof_paper_like/${relative}`;
}

function paperLikeResultFileDifferences(id, resultFiles, sourceSummary) {
  const differences = resultFilePathDifferences(id, resultFiles);
  const quantitativeImages = (Array.isArray(sourceSummary.images) ? sourceSummary.images : [])
    .filter(isCompletedQuantitativeSummaryImage);
  const expectedByResultFile = new Map();
  for (const item of quantitativeImages) {
    const derivedResultFile = paperLikeResultFileForFigure(item.figure_path);
    if (!derivedResultFile) {
      differences.push(`${id}.resultFiles: source summary figure path cannot be mapped to docs/assets/repro for: ${item?.image_path || "<unknown image>"}`);
      continue;
    }
    if (item.result_file && item.result_file !== derivedResultFile) {
      differences.push(`${id}.resultFiles: source summary result_file does not match figure path for: ${item?.image_path || "<unknown image>"}`);
    }
    const expectedSha = item.figure_file?.sha256;
    expectedByResultFile.set(derivedResultFile, expectedSha);
  }

  const expectedFiles = [...expectedByResultFile.keys()].sort();
  const actualFiles = Array.isArray(resultFiles) ? [...resultFiles].sort() : [];
  if (!sameJson(actualFiles, expectedFiles)) {
    differences.push(`${id}.resultFiles: paper-like resultFiles must match source summary figure result files`);
  }

  for (const [file, expectedSha] of expectedByResultFile.entries()) {
    const resolved = resolveDocsResultFile(file);
    if (!resolved || !isPathUnder(path.dirname(resolved), docsReproAssetRoot)) continue;
    if (!fs.existsSync(resolved)) {
      differences.push(`${id}.resultFiles: paper-like static resultFile is missing: ${file}`);
      continue;
    }
    if (!isSha256(expectedSha)) {
      differences.push(`${id}.resultFiles: source summary figure sha256 is missing for resultFile ${file}`);
      continue;
    }
    try {
      if (sha256File(resolved) !== expectedSha) {
        differences.push(`${id}.resultFiles: paper-like static resultFile sha256 does not match source summary figure evidence: ${file}`);
      }
    } catch (error) {
      differences.push(`${id}.resultFiles: paper-like static resultFile is not readable: ${file}: ${error.message}`);
    }
  }
  return differences;
}

function paperLikeSummaryArtifactDifferences(id, verification, gate, options = {}) {
  const differences = [];
  const sourcePathValue = verification?.source_summary_path || verification?.sourceSummaryPath;
  const expectedSha = verification?.source_summary_sha256 || verification?.sourceSummarySha256;
  const sourcePath = resolveLocalArtifactPath(sourcePathValue);

  if (!sourcePath) {
    differences.push(`${id}.paperLikeVerification: paper-like result requires source summary artifact path`);
    return differences;
  }
  if (!isPathUnder(sourcePath, reproduceResultsRoot)) {
    differences.push(`${id}.paperLikeVerification: source summary artifact must be under reproduce/results`);
  }
  if (!isSha256(expectedSha)) {
    differences.push(`${id}.paperLikeVerification: paper-like result requires source summary sha256`);
    return differences;
  }
  if (!fs.existsSync(sourcePath)) {
    differences.push(`${id}.paperLikeVerification: source summary artifact is missing: ${sourcePathValue}`);
    return differences;
  }

  let sourceSummary;
  try {
    const actualSha = sha256File(sourcePath);
    if (actualSha !== expectedSha) {
      differences.push(`${id}.paperLikeVerification: source summary sha256 mismatch`);
    }
    sourceSummary = readJson(sourcePath);
  } catch (error) {
    differences.push(`${id}.paperLikeVerification: source summary artifact is not readable JSON: ${error.message}`);
    return differences;
  }

  const sourceGate = sourceSummary.paper_like_gate || sourceSummary.paperLikeGate;
  const sourceFingerprint = sourceSummary.dataset_fingerprint || sourceSummary.datasetFingerprint;
  const gateEvidence = gate?.evidence_summary || gate?.evidenceSummary;
  if (sourceSummary.status !== "completed_local_runner") {
    differences.push(`${id}.paperLikeVerification: source summary must have completed_local_runner status`);
  }
  if (sourceSummary.readiness_status !== "ready_for_paper_like_runner") {
    differences.push(`${id}.paperLikeVerification: source summary must be ready_for_paper_like_runner`);
  }
  if (!hasCompletePaperLikeGateShape(sourceGate)) {
    differences.push(`${id}.paperLikeVerification: source summary must contain complete paper_like_gate evidence`);
  }
  if (!sameJson(sourceGate, gate)) {
    differences.push(`${id}.paperLikeVerification: source summary paper_like_gate does not match run result gate`);
  }
  if (!sameJson(sourceFingerprint, gateEvidence?.dataset_fingerprint)) {
    differences.push(`${id}.paperLikeVerification: source summary dataset_fingerprint does not match gate evidence`);
  }
  differences.push(...paperLikeSourceSummaryDifferences(id, sourceSummary, gate, options));
  differences.push(...paperLikeResultFileDifferences(id, options.resultFiles, sourceSummary));

  return differences;
}

function hasCompletePaperLevelEvidenceSummary(evidence) {
  return Boolean(
    evidence
    && typeof evidence === "object"
    && !Array.isArray(evidence)
    && evidence.schema_version === 1
    && typeof evidence.gate_id === "string"
    && evidence.gate_id.trim()
    && evidence.paper_level_protocol === true
    && evidence.original_or_equivalent_data === true
    && evidence.paper_tables_reproduced === true
    && typeof evidence.protocol_id === "string"
    && evidence.protocol_id.trim()
    && Array.isArray(evidence.dataset_ids)
    && evidence.dataset_ids.length > 0
    && evidence.dataset_ids.every((item) => typeof item === "string" && item.trim())
    && Array.isArray(evidence.table_ids)
    && evidence.table_ids.length > 0
    && evidence.table_ids.every((item) => typeof item === "string" && item.trim())
    && Array.isArray(evidence.baseline_ids)
    && evidence.baseline_ids.length > 0
    && evidence.baseline_ids.every((item) => typeof item === "string" && item.trim())
    && Number.isInteger(evidence.parameter_record_count)
    && evidence.parameter_record_count > 0
    && Number.isInteger(evidence.independent_artifact_count)
    && evidence.independent_artifact_count > 0
  );
}

function hasCompletePaperLevelGateShape(gate) {
  if (!gate || typeof gate !== "object" || Array.isArray(gate)) return false;
  if (gate.passed !== true) return false;
  if (gate.dashboard_level !== "paper-level" && gate.dashboardLevel !== "paper-level") return false;
  if (!Array.isArray(gate.reasons) || gate.reasons.length !== 0) return false;
  if (!Array.isArray(gate.checked_requirements) || gate.checked_requirements.length === 0) return false;
  if (!Array.isArray(gate.checklist) || gate.checklist.length === 0) return false;
  if (!gate.checklist.every((item) => item && typeof item === "object" && item.passed === true)) return false;
  const evidence = gate.evidence_summary || gate.evidenceSummary;
  return hasCompletePaperLevelEvidenceSummary(evidence);
}

function hasCompletePaperLevelVerificationShape(verification, gate) {
  const evidence = gate?.evidence_summary || gate?.evidenceSummary;
  return Boolean(
    verification
    && typeof verification === "object"
    && !Array.isArray(verification)
    && verification.schema_version === 1
    && verification.generated_by === "paper_level.independent_verifier_v1"
    && verification.recomputed_gate === true
    && verification.can_promote === true
    && typeof verification.gate_id === "string"
    && verification.gate_id === evidence?.gate_id
    && typeof (verification.source_artifact_path || verification.sourceArtifactPath) === "string"
    && isSha256(verification.source_artifact_sha256 || verification.sourceArtifactSha256)
  );
}

function firstPresentString(object, fields) {
  for (const field of fields) {
    const value = object?.[field];
    if (typeof value === "string" && value.trim()) return value;
  }
  return "";
}

function paperLevelRowEvidenceDifferences(id, row, label, index, options = {}) {
  const differences = [];
  if (!row || typeof row !== "object" || Array.isArray(row)) {
    return [`${id}.paperLevelVerification: ${label} row ${index} must be an object`];
  }
  for (const field of options.requiredStringFields || []) {
    if (!firstPresentString(row, [field])) {
      differences.push(`${id}.paperLevelVerification: ${label} row ${index} requires ${field}`);
    }
  }
  for (const field of options.requiredTrueFields || []) {
    if (row[field] !== true) {
      differences.push(`${id}.paperLevelVerification: ${label} row ${index} requires ${field}=true`);
    }
  }

  const artifactPathValue = firstPresentString(row, [
    "artifact_path",
    "artifactPath",
    "comparison_artifact_path",
    "comparisonArtifactPath",
    "audit_artifact_path",
    "auditArtifactPath",
    ...(options.pathFields || [])
  ]);
  const artifactSha = firstPresentString(row, [
    "artifact_sha256",
    "artifactSha256",
    "comparison_artifact_sha256",
    "comparisonArtifactSha256",
    "audit_artifact_sha256",
    "auditArtifactSha256",
    ...(options.shaFields || [])
  ]);
  const artifactPath = resolveLocalArtifactPath(artifactPathValue);
  if (!artifactPath) {
    differences.push(`${id}.paperLevelVerification: ${label} row ${index} requires audited artifact path`);
  } else if (!isPathUnder(artifactPath, reproduceResultsRoot)) {
    differences.push(`${id}.paperLevelVerification: ${label} row ${index} audited artifact must be under reproduce/results`);
  } else if (!fs.existsSync(artifactPath)) {
    differences.push(`${id}.paperLevelVerification: ${label} row ${index} audited artifact is missing: ${artifactPathValue}`);
  }
  if (!isSha256(artifactSha)) {
    differences.push(`${id}.paperLevelVerification: ${label} row ${index} requires audited artifact sha256`);
  } else if (artifactPath && fs.existsSync(artifactPath) && isPathUnder(artifactPath, reproduceResultsRoot)) {
    try {
      const stat = fs.statSync(artifactPath);
      if (sha256File(artifactPath) !== artifactSha) {
        differences.push(`${id}.paperLevelVerification: ${label} row ${index} audited artifact sha256 mismatch`);
      }
      if (stat.size < minPaperLevelArtifactBytes) {
        differences.push(`${id}.paperLevelVerification: ${label} row ${index} audited artifact is too small to support paper-level evidence`);
      }
      const text = fs.readFileSync(artifactPath).subarray(0, 8192).toString("utf8").toLowerCase();
      if (sourceAuditArtifactPlaceholderPatterns.some((pattern) => text.includes(pattern))) {
        differences.push(`${id}.paperLevelVerification: ${label} row ${index} audited artifact contains fixture/placeholder text`);
      }
    } catch (error) {
      differences.push(`${id}.paperLevelVerification: ${label} row ${index} audited artifact is not readable: ${error.message}`);
    }
  }
  return differences;
}

function paperLevelVerificationArtifactDifferences(id, verification, gate) {
  const differences = [];
  const sourcePathValue = verification?.source_artifact_path || verification?.sourceArtifactPath;
  const expectedSha = verification?.source_artifact_sha256 || verification?.sourceArtifactSha256;
  const sourcePath = resolveLocalArtifactPath(sourcePathValue);
  if (!sourcePath) {
    differences.push(`${id}.paperLevelVerification: paper-level result requires independent verification artifact path`);
    return differences;
  }
  if (!isPathUnder(sourcePath, reproduceResultsRoot)) {
    differences.push(`${id}.paperLevelVerification: independent verification artifact must be under reproduce/results`);
  }
  if (!isSha256(expectedSha)) {
    differences.push(`${id}.paperLevelVerification: paper-level result requires independent verification artifact sha256`);
    return differences;
  }
  if (!fs.existsSync(sourcePath)) {
    differences.push(`${id}.paperLevelVerification: independent verification artifact is missing: ${sourcePathValue}`);
    return differences;
  }
  try {
    const actualSha = sha256File(sourcePath);
    if (actualSha !== expectedSha) {
      differences.push(`${id}.paperLevelVerification: independent verification artifact sha256 mismatch`);
    }
    const artifact = readJson(sourcePath);
    if (artifact.generated_by !== "paper_level.independent_verifier_v1" || artifact.can_promote !== true) {
      differences.push(`${id}.paperLevelVerification: independent verification artifact must be generated by paper_level.independent_verifier_v1 and can_promote=true`);
    }
    if (artifact.schema_version !== 1 || artifact.recomputed_gate !== true || artifact.verifier_independent !== true) {
      differences.push(`${id}.paperLevelVerification: independent verification artifact must include schema_version=1, recomputed_gate=true, and verifier_independent=true`);
    }
    const gateEvidence = gate?.evidence_summary || gate?.evidenceSummary;
    const artifactEvidence = artifact.evidence_summary || artifact.evidenceSummary;
    if (!hasCompletePaperLevelEvidenceSummary(artifactEvidence)) {
      differences.push(`${id}.paperLevelVerification: independent verification artifact evidence_summary is incomplete`);
    }
    if (!sameJson(artifactEvidence, gateEvidence)) {
      differences.push(`${id}.paperLevelVerification: independent verification artifact evidence_summary does not match run result gate`);
    }
    for (const [field, label] of [
      ["table_comparisons", "table comparisons"],
      ["baseline_comparisons", "baseline comparisons"],
      ["parameter_records", "parameter records"],
      ["data_source_audits", "data source audits"]
    ]) {
      if (!Array.isArray(artifact[field]) || artifact[field].length === 0) {
        differences.push(`${id}.paperLevelVerification: independent verification artifact requires non-empty ${label}`);
      }
    }
    for (const [field, label, options] of [
      ["table_comparisons", "table comparison", { requiredStringFields: ["table_id"] }],
      ["baseline_comparisons", "baseline comparison", { requiredStringFields: ["baseline_id"] }],
      ["parameter_records", "parameter record", { requiredStringFields: ["parameter_id"] }],
      [
        "data_source_audits",
        "data source audit",
        {
          requiredStringFields: ["dataset_id", "source_id"],
          requiredTrueFields: ["license_reviewed", "provenance_reviewed"]
        }
      ]
    ]) {
      if (Array.isArray(artifact[field])) {
        artifact[field].forEach((row, index) => {
          differences.push(...paperLevelRowEvidenceDifferences(id, row, label, index, options));
        });
      }
    }
    const artifactGate = artifact.paper_level_gate || artifact.paperLevelGate;
    if (!sameJson(artifactGate, gate)) {
      differences.push(`${id}.paperLevelVerification: independent verification artifact gate does not match run result gate`);
    }
  } catch (error) {
    differences.push(`${id}.paperLevelVerification: independent verification artifact is not readable JSON: ${error.message}`);
  }
  return differences;
}

function expectedTruthLevel(result) {
  if (result.status && result.status !== "completed") return "assessment-only";
  const level = result.reproductionLevel || "";
  if (level === "paper-level") return "paper-level-completed";
  if (level === "assessment-only") return "assessment-only";
  if (level.includes("partial") || level === "paper-like") return "partial-completed";
  return "toy-completed";
}

function compareDashboardToResults(data, runResults, options = {}) {
  const differences = [];
  differences.push(
    ...duplicateIdDifferences(data.reproAssessments || [], "dashboard assessment"),
    ...duplicateIdDifferences(runResults, "run result")
  );
  const resultById = new Map(runResults.map((item) => [item.id, item]));

  for (const item of data.reproAssessments || []) {
    const result = resultById.get(item.id);
    if (!result) {
      differences.push(`${item.id}: missing run result`);
      continue;
    }

    const derivedTruthLevel = expectedTruthLevel(result);
    if (
      result.reproductionTruthLevel !== undefined
      && !sameJson(result.reproductionTruthLevel, derivedTruthLevel)
    ) {
      differences.push(
        `${item.id}.reproductionTruthLevel: run result truth override=${JSON.stringify(result.reproductionTruthLevel)} derived=${JSON.stringify(derivedTruthLevel)}`
      );
    }

    const checks = [
      ["priority", item.priority, result.priority],
      ["experimentId", item.experimentId, result.experiment_id],
      ["reproductionLevel", item.reproductionLevel, result.reproductionLevel],
      ["reproductionTruthLevel", item.reproductionTruthLevel, derivedTruthLevel],
      ["resultStatus", item.resultStatus, result.status],
      ["runMetrics", withoutRuntimeSecondsMetrics(item.runMetrics || {}), withoutRuntimeSecondsMetrics(result.metrics || {})],
      ["resultFiles", normalizeFiles(item.resultFiles), normalizeFiles(result.resultFiles)],
      ["notes", item.notes || "", result.notes || ""]
    ];

    compareRuntimeSecondsShape(item.id, item.runtimeSeconds, result.runtime_seconds, differences);

    if (result.status === "skipped") {
      checks.push([
        "skippedReason",
        item.skippedReason || item.skipped_reason || "",
        result.skipped_reason || ""
      ]);
    }

    for (const field of ["resultQuality", "warning", "fidelityWarning"]) {
      if (result[field] !== undefined) {
        checks.push([field, item[field], result[field]]);
      }
    }

    const resultGate = result.paper_like_gate || result.paperLikeGate;
    if (resultGate !== undefined) {
      checks.push(["paperLikeGate", item.paper_like_gate || item.paperLikeGate || null, resultGate]);
    }
    if (result.reproductionLevel === "paper-like" && resultGate?.passed !== true) {
      differences.push(`${item.id}.paperLikeGate: ${result.reproductionLevel} result requires paper_like_gate.passed=true`);
    }
    if (result.reproductionLevel === "paper-like" && !hasCompletePaperLikeGateShape(resultGate)) {
      differences.push(`${item.id}.paperLikeGate: paper-like result requires a complete recomputed paper_like_gate checklist`);
    }
    const resultPaperLikeVerification = result.paper_like_verification || result.paperLikeVerification;
    if (resultPaperLikeVerification !== undefined) {
      checks.push([
        "paperLikeVerification",
        item.paper_like_verification || item.paperLikeVerification || null,
        resultPaperLikeVerification
      ]);
    }
    if (
      result.reproductionLevel === "paper-like"
      && !hasCompletePaperLikeVerificationShape(resultPaperLikeVerification, resultGate)
    ) {
      differences.push(`${item.id}.paperLikeVerification: paper-like result requires runner-generated promotion verification`);
    }
    differences.push(...resultFilePathDifferences(item.id, result.resultFiles));
    if (result.reproductionLevel === "paper-like") {
      differences.push(
        ...paperLikeSummaryArtifactDifferences(
          item.id,
          resultPaperLikeVerification,
          resultGate,
          { ...options, resultFiles: result.resultFiles }
        )
      );
    }
    const resultPaperLevelGate = result.paper_level_gate || result.paperLevelGate;
    if (resultPaperLevelGate !== undefined) {
      checks.push(["paperLevelGate", item.paper_level_gate || item.paperLevelGate || null, resultPaperLevelGate]);
    }
    if (result.reproductionLevel === "paper-level" && resultPaperLevelGate?.passed !== true) {
      differences.push(`${item.id}.paperLevelGate: paper-level result requires paper_level_gate.passed=true`);
    }
    if (result.reproductionLevel === "paper-level" && !hasCompletePaperLevelGateShape(resultPaperLevelGate)) {
      differences.push(`${item.id}.paperLevelGate: paper-level result requires a complete independent paper_level_gate evidence summary`);
    }
    const resultPaperLevelVerification = result.paper_level_verification || result.paperLevelVerification;
    if (resultPaperLevelVerification !== undefined) {
      checks.push([
        "paperLevelVerification",
        item.paper_level_verification || item.paperLevelVerification || null,
        resultPaperLevelVerification
      ]);
    }
    if (
      result.reproductionLevel === "paper-level"
      && !hasCompletePaperLevelVerificationShape(resultPaperLevelVerification, resultPaperLevelGate)
    ) {
      differences.push(`${item.id}.paperLevelVerification: paper-level result requires independent paper-level promotion verification`);
    }
    if (result.reproductionLevel === "paper-level") {
      differences.push(...paperLevelVerificationArtifactDifferences(item.id, resultPaperLevelVerification, resultPaperLevelGate));
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

  const dashboardOrder = (data.reproAssessments || []).map((item) => item.id);
  const resultOrder = runResults.map((item) => item.id);
  if (!sameJson(resultOrder, dashboardOrder)) {
    differences.push(
      `run result order: dashboard=${JSON.stringify(dashboardOrder)} result=${JSON.stringify(resultOrder)}`
    );
  }

  return differences;
}

function compareResultAssetSnapshot(runResults, assetResults) {
  if (!Array.isArray(runResults) || !Array.isArray(assetResults)) {
    return ["docs/assets/repro/repro_results.json: run results and asset results must both be arrays"];
  }
  const duplicateDifferences = [
    ...duplicateIdDifferences(runResults, "run result"),
    ...duplicateIdDifferences(assetResults, "asset result")
  ].map((item) => `docs/assets/repro/repro_results.json: ${item}`);
  if (runResults.length !== assetResults.length) {
    return [
      ...duplicateDifferences,
      `docs/assets/repro/repro_results.json: length mismatch asset=${assetResults.length} run=${runResults.length}`
    ];
  }

  const differences = [...duplicateDifferences];
  const runOrder = runResults.map((item) => item.id);
  const assetOrder = assetResults.map((item) => item.id);
  if (!sameJson(assetOrder, runOrder)) {
    differences.push(
      `docs/assets/repro/repro_results.json: asset result order=${JSON.stringify(assetOrder)} run order=${JSON.stringify(runOrder)}`
    );
  }

  const assetById = new Map(assetResults.map((item) => [item.id, item]));
  for (const result of runResults) {
    const asset = assetById.get(result.id);
    if (!asset) {
      differences.push(`docs/assets/repro/repro_results.json: missing asset result for ${result.id}`);
      continue;
    }
    if (!sameJson(asset, result)) {
      differences.push(`docs/assets/repro/repro_results.json: asset result differs from run result for ${result.id}`);
    }
  }

  const runIds = new Set(runResults.map((item) => item.id));
  for (const asset of assetResults) {
    if (!runIds.has(asset.id)) {
      differences.push(`docs/assets/repro/repro_results.json: stale extra asset result for ${asset.id}`);
    }
  }

  return differences;
}

function syncSnapshotDifferences(data, runResults, assetResults = null, env = process.env, options = {}) {
  const differences = compareDashboardToResults(data, runResults);
  differences.push(
    ...iteratedRofSourceRegistryDifferences(env, options),
    ...reproductionPromotionCountFailures(data.reproAssessments || [], env, { label: "dashboard" }),
    ...reproductionPromotionCountFailures(runResults, env, { label: "run result" })
  );
  if (assetResults) {
    differences.push(
      ...compareResultAssetSnapshot(runResults, assetResults),
      ...reproductionPromotionCountFailures(assetResults, env, { label: "asset result" })
    );
  }
  return differences;
}

function findById(items, id) {
  return Array.isArray(items) ? items.find((item) => item?.id === id) : undefined;
}

function replaceById(items, id, replacement) {
  return items.map((item) => (item?.id === id ? replacement : item));
}

function validateCandidateAgainstCurrentSnapshots(candidate, currentDashboardData, currentRunResults, currentAssetResults, env = process.env) {
  const differences = [];
  if (!currentDashboardData && !currentRunResults && !currentAssetResults) return differences;

  const runResult = candidate.runResultPatch;
  const id = runResult.id || candidate.paper_id;
  let patchedDashboardData = null;
  let patchedRunResults = null;
  let patchedAssetResults = null;

  if (currentDashboardData) {
    const currentAssessments = currentDashboardData.reproAssessments || [];
    const currentAssessment = findById(currentAssessments, id);
    if (!currentAssessment) {
      differences.push(`candidate.currentDashboard: no current dashboard assessment for ${id}`);
    } else {
      if (!sameJson(currentAssessment.priority, candidate.priority)) {
        differences.push(
          `candidate.currentDashboard: current dashboard priority=${JSON.stringify(currentAssessment.priority)} candidate.priority=${JSON.stringify(candidate.priority)}`
        );
      }
      patchedDashboardData = {
        ...currentDashboardData,
        reproAssessments: replaceById(
          currentAssessments,
          id,
          {
            ...currentAssessment,
            ...candidate.dashboardDetailPatch,
            id: currentAssessment.id,
            priority: currentAssessment.priority
          }
        )
      };
    }
  }

  if (currentRunResults) {
    const currentRunResult = findById(currentRunResults, id);
    if (!currentRunResult) {
      differences.push(`candidate.currentRunResults: no current run result for ${id}`);
    } else {
      if (!sameJson(currentRunResult.priority, candidate.priority)) {
        differences.push(
          `candidate.currentRunResults: current run result priority=${JSON.stringify(currentRunResult.priority)} candidate.priority=${JSON.stringify(candidate.priority)}`
        );
      }
      patchedRunResults = replaceById(currentRunResults, id, runResult);
    }
  }

  if (currentAssetResults) {
    const currentAssetResult = findById(currentAssetResults, id);
    if (!currentAssetResult) {
      differences.push(`candidate.currentAssetResults: no current asset result for ${id}`);
    } else {
      patchedAssetResults = replaceById(currentAssetResults, id, runResult);
    }
  }

  if (patchedDashboardData && patchedRunResults) {
    differences.push(
      ...compareDashboardToResults(patchedDashboardData, patchedRunResults)
        .map((item) => `candidate.current.${item}`)
    );
  }
  if (patchedDashboardData) {
    differences.push(
      ...reproductionPromotionCountFailures(patchedDashboardData.reproAssessments || [], env, { label: "candidate.current.dashboard" })
    );
  }
  if (patchedRunResults) {
    differences.push(
      ...reproductionPromotionCountFailures(patchedRunResults, env, { label: "candidate.current.run result" })
    );
  }
  if (patchedAssetResults) {
    differences.push(
      ...reproductionPromotionCountFailures(patchedAssetResults, env, { label: "candidate.current.asset result" })
    );
  }
  if (patchedRunResults && patchedAssetResults) {
    differences.push(
      ...compareResultAssetSnapshot(patchedRunResults, patchedAssetResults)
        .map((item) => `candidate.current.${item}`)
    );
  }
  return differences;
}

function validateDashboardCandidateShape(candidate) {
  const differences = [];
  if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) {
    return ["candidate: expected object"];
  }
  if (candidate.can_promote !== true) {
    differences.push("candidate.can_promote: expected can_promote=true before dashboard validation");
  }
  if (!candidate.runResultPatch || typeof candidate.runResultPatch !== "object") {
    differences.push("candidate.runResultPatch: missing runResultPatch");
  }
  if (!candidate.dashboardDetailPatch || typeof candidate.dashboardDetailPatch !== "object") {
    differences.push("candidate.dashboardDetailPatch: missing dashboardDetailPatch");
  }
  if (differences.length) return differences;

  const runResult = candidate.runResultPatch;
  const resultGate = runResult.paper_like_gate || runResult.paperLikeGate;
  const dashboardGate = candidate.dashboardDetailPatch.paper_like_gate || candidate.dashboardDetailPatch.paperLikeGate;
  if (!sameJson(candidate.paper_id, runResult.id)) {
    differences.push(`candidate.paper_id: candidate=${JSON.stringify(candidate.paper_id)} runResultPatch.id=${JSON.stringify(runResult.id)}`);
  }
  if (!sameJson(candidate.priority, runResult.priority)) {
    differences.push(`candidate.priority: candidate=${JSON.stringify(candidate.priority)} runResultPatch.priority=${JSON.stringify(runResult.priority)}`);
  }
  if (!sameJson(candidate.paperLikeGate || candidate.paper_like_gate || null, resultGate || null)) {
    differences.push("candidate.paperLikeGate: top-level paperLikeGate must match runResultPatch gate");
  }
  if (!sameJson(dashboardGate || null, resultGate || null)) {
    differences.push("candidate.paperLikeGate: dashboardDetailPatch gate must match runResultPatch gate");
  }
  if (!sameJson(candidate.candidateDetails || null, candidate.dashboardDetailPatch || null)) {
    differences.push("candidate.candidateDetails: candidateDetails must match dashboardDetailPatch");
  }

  const dashboardAssessment = {
    id: runResult.id || candidate.paper_id,
    priority: runResult.priority ?? candidate.priority,
    ...candidate.dashboardDetailPatch
  };
  const syncDifferences = compareDashboardToResults(
    { reproAssessments: [dashboardAssessment] },
    [runResult]
  );
  differences.push(...syncDifferences.map((item) => `candidate.${item}`));
  return differences;
}

function validateDashboardCandidate(candidate, currentDashboardData = null, currentRunResults = null, currentAssetResults = null, options = {}) {
  const differences = validateDashboardCandidateShape(candidate);
  const env = options.env || process.env;
  if (!currentDashboardData || !currentRunResults || !currentAssetResults) {
    differences.push("candidate.current: current dashboard, run-result, and asset snapshots are required for promotion validation");
    return differences;
  }
  if (differences.length) return differences;
  differences.push(
    ...validateCandidateAgainstCurrentSnapshots(candidate, currentDashboardData, currentRunResults, currentAssetResults, env)
  );
  return differences;
}

const resultsPath = fs.existsSync(defaultResultsPath) ? defaultResultsPath : assetResultsPath;
function main() {
  if (repoRootOverrideError) {
    console.error(repoRootOverrideError);
    process.exit(1);
  }

  if (args.has("--candidate") && !candidatePath) {
    console.error("missing --candidate path");
    process.exit(1);
  }

  if (candidatePath) {
    const candidate = readJson(resolveLocalArtifactPath(candidatePath) || candidatePath);
    const data = loadDashboardData();
    const runResults = readJson(resultsPath);
    const assetResults = fs.existsSync(defaultResultsPath) && fs.existsSync(assetResultsPath)
      ? readJson(assetResultsPath)
      : null;
    const differences = validateDashboardCandidate(candidate, data, runResults, assetResults);
    if (!quiet) {
      console.log(`checked dashboard candidate ${candidatePath}`);
    }
    if (differences.length) {
      if (!quiet) {
        console.error("dashboard candidate is not promotable:");
        differences.forEach((item) => console.error(`- ${item}`));
      }
      if (checkOnly) process.exit(1);
    } else if (!quiet) {
      console.log("dashboard candidate patches pass sync validation");
    }
    return;
  }

  const data = loadDashboardData();
  const runResults = readJson(resultsPath);
  const assetResults = fs.existsSync(defaultResultsPath) && fs.existsSync(assetResultsPath)
    ? readJson(assetResultsPath)
    : null;
  const differences = syncSnapshotDifferences(data, runResults, assetResults);

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
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}

export {
  compareDashboardToResults,
  compareResultAssetSnapshot,
  expectedTruthLevel,
  iteratedRofSourceRegistryDifferences,
  syncSnapshotDifferences,
  validateDashboardCandidate,
  validateDashboardCandidateShape
};
