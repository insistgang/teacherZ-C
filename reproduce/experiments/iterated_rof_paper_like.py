import argparse
import csv
import hashlib
import json
import math
import re
import shutil
from datetime import date
from pathlib import Path, PurePosixPath

from common import REPO_ROOT, RESULTS_DIR, SEED, clustering_accuracy, dice_score, require_modules, simple_kmeans, timer
import sat_rof_trof


DATA_ROOT = REPO_ROOT / "reproduce" / "data" / "iterated_rof"
REPORT_PATH = RESULTS_DIR / "iterated_rof_paper_like_readiness.json"
RUN_SUMMARY_PATH = RESULTS_DIR / "iterated_rof_paper_like_summary.json"
FAMILY_SUMMARY_CSV_PATH = RESULTS_DIR / "iterated_rof_paper_like_family_summary.csv"
IMAGE_EVIDENCE_CSV_PATH = RESULTS_DIR / "iterated_rof_paper_like_image_evidence.csv"
FIGURE_DIR = RESULTS_DIR / "figures" / "iterated_rof_paper_like"
DASHBOARD_REPRO_ASSET_ROOT = REPO_ROOT / "docs" / "assets" / "repro"
SUMMARY_VERIFICATION_PATH = RESULTS_DIR / "iterated_rof_paper_like_summary_verification.json"
SOURCE_MANIFEST_PATH = REPO_ROOT / "reproduce" / "paper_like" / "iterated_rof_dataset_sources.json"
LOCAL_DATASET_MANIFEST_NAME = "dataset_manifest.json"
DATASET_MANIFEST_TEMPLATE_PATH = DATA_ROOT / "dataset_manifest.template.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FIGURE_EVIDENCE_GENERATOR = "iterated_rof_paper_like.figure_grid_v1"
PAPER_LIKE_GATE_ID = "iterated_rof_paper_like_v1"
PAPER_LIKE_PROMOTION_VERIFICATION_GENERATOR = "iterated_rof_paper_like.dashboard_candidate_v1"
DASHBOARD_STATIC_ASSET_GENERATOR = "iterated_rof_paper_like.dashboard_static_assets_v1"
ITERATED_ROF_DASHBOARD_PRIORITY = 3
PAPER_LIKE_FIDELITY_WARNING = (
    "Paper-like candidate uses local reviewed cartoon/texture/medical data and the Iterated ROF "
    "thresholding runner; it is not paper-level until original or equivalently audited paper tables, "
    "baselines, and protocol details are independently reproduced."
)
MIN_PAPER_LIKE_IMAGE_SIDE = 32
MIN_PAPER_LIKE_IMAGE_LEVELS = 8
MIN_SOURCE_AUDIT_ARTIFACT_BYTES = 128

DATA_FAMILIES = {
    "cartoon": "cartoon or smooth-region image used to test missing-pixel / piecewise-smooth behavior",
    "texture": "close-intensity texture or stripe image used to test texture separation",
    "medical": "medical grayscale image, preferably MRI-like, used to test medical segmentation behavior",
}


def _find_images(directory):
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and _path_is_under(path, directory)
    )


def _count_images(directory):
    return len(_find_images(directory))


def _display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_source_manifest(path=SOURCE_MANIFEST_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


SOURCE_REGISTRY_REQUIRED_TEXT_FIELDS = (
    "source_id",
    "name",
    "url",
    "download_url",
    "target_family",
    "fit",
    "local_layout",
    "download_policy",
    "license_note",
    "paper_like_role",
)


def _source_manifest_schema_blockers(source_manifest):
    blockers = []
    if not isinstance(source_manifest, dict):
        return ["Source registry must be an object keyed by family"]

    expected_families = set(DATA_FAMILIES)
    present_families = set(source_manifest)
    for family in sorted(expected_families - present_families):
        blockers.append(f"Source registry missing family: {family}")
    for family in sorted(present_families - expected_families):
        blockers.append(f"Source registry has unknown family: {family}")

    seen_source_ids = {}
    for family in sorted(expected_families & present_families):
        entries = source_manifest.get(family)
        if not isinstance(entries, list) or not entries:
            blockers.append(f"Source registry missing source entries for family: {family}")
            continue
        for index, source in enumerate(entries):
            label = f"{family}[{index}]"
            if not isinstance(source, dict):
                blockers.append(f"Source registry entry is not an object: {label}")
                continue
            for field in SOURCE_REGISTRY_REQUIRED_TEXT_FIELDS:
                if not str(source.get(field, "")).strip():
                    blockers.append(f"Source registry missing {field}: {label}")
            source_id = str(source.get("source_id", "")).strip()
            if source_id:
                if source_id in seen_source_ids:
                    blockers.append(
                        f"Source registry duplicate source_id: {source_id} in {label} and {seen_source_ids[source_id]}"
                    )
                else:
                    seen_source_ids[source_id] = label
            if source.get("target_family") != family:
                blockers.append(f"Source registry target_family mismatch for: {label}")
            priority = source.get("priority")
            if not isinstance(priority, int) or isinstance(priority, bool):
                blockers.append(f"Source registry priority must be an integer for: {label}")
    return blockers


def prepare_data_layout(root=DATA_ROOT, template_path=DATASET_MANIFEST_TEMPLATE_PATH):
    root = Path(root)
    directories = []
    for family in DATA_FAMILIES:
        for kind in ["images", "masks", "audit"]:
            directory = root / family / kind
            status = "exists" if directory.exists() else "created"
            directory.mkdir(parents=True, exist_ok=True)
            directories.append(
                {
                    "family": family,
                    "kind": kind,
                    "path": _display_path(directory),
                    "status": status,
                }
            )

    manifest_path = root / LOCAL_DATASET_MANIFEST_NAME
    template_path = Path(template_path)
    if manifest_path.exists():
        manifest = {
            "status": "already_exists",
            "path": _display_path(manifest_path),
        }
    elif template_path.exists():
        manifest_path.write_text(template_path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest = {
            "status": "created_from_template",
            "path": _display_path(manifest_path),
            "template_path": _display_path(template_path),
        }
    else:
        manifest = {
            "status": "template_missing",
            "path": _display_path(manifest_path),
            "template_path": _display_path(template_path),
        }

    return {
        "data_root": _display_path(root),
        "directories": directories,
        "manifest": manifest,
        "downloaded_data": False,
    }


def _data_drop_files(kind_root, allowed_extensions=None):
    kind_root = Path(kind_root)
    files = []
    skipped = []
    if not kind_root.exists():
        return files, skipped
    for path in sorted(kind_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(kind_root).as_posix()
        if not _path_is_under(path, kind_root):
            skipped.append(
                {
                    "path": _display_path(path),
                    "relative_path": relative,
                    "reason": "path_escape",
                }
            )
            continue
        if allowed_extensions is not None and path.suffix.lower() not in allowed_extensions:
            skipped.append(
                {
                    "path": _display_path(path),
                    "relative_path": relative,
                    "reason": "unsupported_extension",
                }
            )
            continue
        files.append(path)
    return files, skipped


def _review_data_drop_kind(drop_root, data_root, family, kind, allowed_extensions=None):
    source_root = Path(drop_root) / family / kind
    target_root = Path(data_root) / family / kind
    source_files, skipped_files = _data_drop_files(source_root, allowed_extensions=allowed_extensions)
    file_reports = []
    copyable = 0
    current = 0
    conflicts = 0
    for source_path in source_files:
        relative_path = source_path.relative_to(source_root).as_posix()
        target_path = target_root / relative_path
        source_evidence = _file_evidence(source_path)
        report = {
            "relative_path": relative_path,
            "source_path": _display_path(source_path),
            "target_path": _display_path(target_path),
            "source_sha256": source_evidence["sha256"],
            "source_size_bytes": source_evidence["size_bytes"],
        }
        if not _path_is_under(target_path, target_root):
            report["status"] = "unsafe_target_path"
            skipped_files.append({**report, "reason": "unsafe_target_path"})
            file_reports.append(report)
            continue
        if target_path.exists():
            if not target_path.is_file():
                report["status"] = "conflict"
                report["reason"] = "target_exists_not_file"
                conflicts += 1
            else:
                target_evidence = _file_evidence(target_path)
                report["target_sha256"] = target_evidence["sha256"]
                report["target_size_bytes"] = target_evidence["size_bytes"]
                if target_evidence["sha256"] == source_evidence["sha256"]:
                    report["status"] = "current"
                    current += 1
                else:
                    report["status"] = "conflict"
                    report["reason"] = "target_sha256_mismatch"
                    conflicts += 1
            file_reports.append(report)
            continue
        report["status"] = "would_copy"
        copyable += 1
        file_reports.append(report)
    return {
        "kind": kind,
        "source_path": _display_path(source_root),
        "target_path": _display_path(target_root),
        "copyable_file_count": copyable,
        "current_file_count": current,
        "conflict_file_count": conflicts,
        "skipped_file_count": len(skipped_files),
        "files": file_reports,
        "skipped_files": skipped_files,
    }


def _ingest_data_drop_kind(drop_root, data_root, family, kind, allowed_extensions=None):
    target_root = Path(data_root) / family / kind
    target_root.mkdir(parents=True, exist_ok=True)
    review = _review_data_drop_kind(drop_root, data_root, family, kind, allowed_extensions=allowed_extensions)
    copied = 0
    for report in review["files"]:
        if report.get("status") != "would_copy":
            continue
        source_path = _resolve_report_path(report["source_path"])
        target_path = _resolve_report_path(report["target_path"])
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        target_evidence = _file_evidence(target_path)
        report["status"] = "copied"
        report["target_sha256"] = target_evidence["sha256"]
        report["target_size_bytes"] = target_evidence["size_bytes"]
        copied += 1
    review["copied_file_count"] = copied
    review.pop("copyable_file_count", None)
    return review


def review_data_drop(drop_root, data_root=DATA_ROOT):
    drop_root = Path(drop_root)
    data_root = Path(data_root)
    family_reports = []
    totals = {
        "copyable_file_count": 0,
        "current_file_count": 0,
        "conflict_file_count": 0,
        "skipped_file_count": 0,
    }
    for family in DATA_FAMILIES:
        kinds = [
            _review_data_drop_kind(drop_root, data_root, family, "images", allowed_extensions=IMAGE_EXTENSIONS),
            _review_data_drop_kind(drop_root, data_root, family, "masks", allowed_extensions=IMAGE_EXTENSIONS),
            _review_data_drop_kind(drop_root, data_root, family, "audit", allowed_extensions=None),
        ]
        for kind_report in kinds:
            for key in totals:
                totals[key] += kind_report.get(key, 0)
        family_reports.append(
            {
                "family": family,
                "kinds": kinds,
            }
        )
    if totals["conflict_file_count"]:
        status = "conflict"
    elif totals["copyable_file_count"]:
        status = "would_ingest"
    elif totals["current_file_count"]:
        status = "current"
    else:
        status = "empty"
    return {
        "status": status,
        "source_root": _display_path(drop_root),
        "data_root": _display_path(data_root),
        "downloaded_data": False,
        "would_write": False,
        **totals,
        "families": family_reports,
    }


def write_data_drop_review(drop_root, data_root, path):
    review = review_data_drop(drop_root, data_root)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return review


def ingest_data_drop(drop_root, data_root=DATA_ROOT):
    drop_root = Path(drop_root)
    data_root = Path(data_root)
    preparation = prepare_data_layout(data_root)
    family_reports = []
    totals = {
        "copied_file_count": 0,
        "current_file_count": 0,
        "conflict_file_count": 0,
        "skipped_file_count": 0,
    }
    for family in DATA_FAMILIES:
        kinds = [
            _ingest_data_drop_kind(drop_root, data_root, family, "images", allowed_extensions=IMAGE_EXTENSIONS),
            _ingest_data_drop_kind(drop_root, data_root, family, "masks", allowed_extensions=IMAGE_EXTENSIONS),
            _ingest_data_drop_kind(drop_root, data_root, family, "audit", allowed_extensions=None),
        ]
        for kind_report in kinds:
            for key in totals:
                totals[key] += kind_report.get(key, 0)
        family_reports.append(
            {
                "family": family,
                "kinds": kinds,
            }
        )
    manifest_file_claim_refresh = refresh_manifest_file_claims(data_root)
    if totals["conflict_file_count"]:
        status = "conflict"
    elif totals["copied_file_count"]:
        status = "ingested"
    elif totals["current_file_count"]:
        status = "current"
    else:
        status = "empty"
    return {
        "status": status,
        "source_root": _display_path(drop_root),
        "data_root": _display_path(data_root),
        "downloaded_data": False,
        "data_layout_preparation": preparation,
        **totals,
        "families": family_reports,
        "manifest_file_claim_refresh": manifest_file_claim_refresh,
    }


def load_local_dataset_manifest(root=DATA_ROOT):
    manifest_path = Path(root) / LOCAL_DATASET_MANIFEST_NAME
    if not manifest_path.exists():
        return {
            "status": "missing",
            "path": _display_path(manifest_path),
            "families": {},
        }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "invalid",
            "path": _display_path(manifest_path),
            "families": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    families = payload.get("families", {})
    if not isinstance(families, dict):
        return {
            "status": "invalid",
            "path": _display_path(manifest_path),
            "families": {},
            "error": "Expected top-level 'families' object",
        }
    return {
        "status": "present",
        "path": _display_path(manifest_path),
        "families": families,
    }


def _manifest_file_claim_for_entry(entry, existing_claim=None):
    existing_claim = dict(existing_claim or {})
    image_evidence = _file_evidence(entry["image_path"])
    claim = {
        key: value
        for key, value in existing_claim.items()
        if key not in {"image", "sha256", "mask", "mask_sha256"}
    }
    claim["image"] = entry["image_relative_path"]
    claim["sha256"] = image_evidence["sha256"]
    if entry["mask_path"]:
        claim["mask"] = entry["image_relative_path"]
        claim["mask_sha256"] = _file_evidence(entry["mask_path"])["sha256"]
    return claim


def _manifest_file_claim_refresh_payload(root=DATA_ROOT):
    root = Path(root)
    manifest_path = root / LOCAL_DATASET_MANIFEST_NAME
    local_manifest = load_local_dataset_manifest(root)
    if local_manifest["status"] != "present":
        report = {
            "status": f"blocked_manifest_{local_manifest['status']}",
            "path": _display_path(manifest_path),
            "error": local_manifest.get("error"),
        }
        return None, report

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_families = payload.setdefault("families", {})
    entries = scan_dataset(root)
    entries_by_family = {family: [] for family in DATA_FAMILIES}
    for entry in entries:
        entries_by_family.setdefault(entry["family"], []).append(entry)

    family_reports = []
    for family in DATA_FAMILIES:
        family_claim = manifest_families.get(family)
        family_status = "updated"
        if not isinstance(family_claim, dict):
            family_claim = {}
            manifest_families[family] = family_claim
            family_status = "created_family_entry"

        existing_files = family_claim.get("files")
        if not isinstance(existing_files, list):
            existing_files = []
            family_status = "created_files_list"
        existing_by_image = {
            str(file_claim.get("image")): file_claim
            for file_claim in existing_files
            if isinstance(file_claim, dict) and file_claim.get("image")
        }

        refreshed_files = []
        local_image_keys = set()
        for entry in entries_by_family.get(family, []):
            image_key = entry["image_relative_path"]
            local_image_keys.add(image_key)
            refreshed_files.append(_manifest_file_claim_for_entry(entry, existing_by_image.get(image_key)))

        stale_file_claims = [
            file_claim
            for file_claim in existing_files
            if isinstance(file_claim, dict) and str(file_claim.get("image")) not in local_image_keys
        ]
        stale_file_claim_items = [
            {
                "family": family,
                "image": str(file_claim.get("image")),
            }
            for file_claim in stale_file_claims
        ]
        family_claim["files"] = refreshed_files + stale_file_claims
        family_reports.append(
            {
                "family": family,
                "status": family_status,
                "local_image_count": len(refreshed_files),
                "stale_file_claim_count": len(stale_file_claims),
                "stale_file_claims": stale_file_claim_items,
            }
        )

    return payload, {
        "path": _display_path(manifest_path),
        "families": family_reports,
        "downloaded_data": False,
    }


def refresh_manifest_file_claims(root=DATA_ROOT):
    root = Path(root)
    manifest_path = root / LOCAL_DATASET_MANIFEST_NAME
    payload, report = _manifest_file_claim_refresh_payload(root)
    if payload is None:
        return {
            **report,
            "written": False,
        }

    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        **report,
        "status": "updated",
        "written": True,
    }


def check_manifest_file_claims(root=DATA_ROOT):
    root = Path(root)
    manifest_path = root / LOCAL_DATASET_MANIFEST_NAME
    payload, report = _manifest_file_claim_refresh_payload(root)
    if payload is None:
        return {
            **report,
            "written": False,
            "stale": True,
        }

    current_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    would_change_payload = payload != current_payload
    stale_file_claims = sum(
        family_report.get("stale_file_claim_count", 0)
        for family_report in report.get("families", [])
    )
    stale_file_claim_items = [
        stale_claim
        for family_report in report.get("families", [])
        for stale_claim in family_report.get("stale_file_claims", [])
    ]
    stale = bool(would_change_payload or stale_file_claims)
    return {
        **report,
        "status": "stale" if stale else "current",
        "written": False,
        "stale": stale,
        "would_change_payload": would_change_payload,
        "stale_file_claim_count": stale_file_claims,
        "stale_file_claims": stale_file_claim_items,
    }


def _source_audit_artifact_claim_updates_for_audit(root, family, scope, audit):
    if not isinstance(audit, dict):
        return []
    audit_roots = _source_audit_roots_for_family(root, family)
    audit_root = audit_roots[0] if audit_roots else Path(root) / family / "audit"
    root = Path(root)
    updates = []
    for path_field, sha_field, artifact_label in SOURCE_AUDIT_ARTIFACT_FIELDS:
        artifact_path = _resolve_source_audit_path(audit.get(path_field), audit_roots, root)
        if artifact_path is None:
            continue
        if not _path_is_under(artifact_path, root):
            continue
        if not _path_is_under(artifact_path, audit_root):
            continue
        if not artifact_path.is_file():
            continue
        actual_sha = _file_evidence(artifact_path)["sha256"]
        if audit.get(sha_field) != actual_sha:
            updates.append(
                {
                    "family": family,
                    "scope": scope,
                    "artifact": artifact_label,
                    "path": audit.get(path_field),
                    "sha_field": sha_field,
                    "old_sha256": audit.get(sha_field, ""),
                    "new_sha256": actual_sha,
                }
            )
            audit[sha_field] = actual_sha
    return updates


def _source_audit_artifact_claim_issues_for_audit(root, family, scope, audit):
    if not isinstance(audit, dict):
        return [
            {
                "family": family,
                "scope": scope,
                "artifact": "source_audit",
                "path": "",
                "path_status": "missing_source_audit",
                "sha256_status": "not_checked",
                "content_status": "not_checked",
                "content_size_bytes": None,
                "min_content_size_bytes": MIN_SOURCE_AUDIT_ARTIFACT_BYTES,
                "content_issue_codes": [],
                "content_issues": [],
                "placeholder_pattern_hits": [],
                "issue": "source_audit is missing",
            }
        ]
    audit_roots = _source_audit_roots_for_family(root, family)
    audit_root = audit_roots[0] if audit_roots else Path(root) / family / "audit"
    issues = []
    for path_field, sha_field, artifact_label in SOURCE_AUDIT_ARTIFACT_FIELDS:
        status, _artifact_issues = _source_audit_artifact_status(
            "",
            audit,
            Path(root),
            audit_root,
            path_field,
            sha_field,
            artifact_label,
        )
        if status.get("ready"):
            continue
        path_status = status.get("path_status")
        sha256_status = status.get("sha256_status")
        content_status = status.get("content_status")
        if path_status == "missing_path":
            issue = f"{artifact_label} path is missing"
        elif path_status == "outside_data_root":
            issue = f"{artifact_label} path is outside local data root"
        elif path_status == "outside_family_audit_root":
            issue = f"{artifact_label} path is outside local family audit root"
        elif path_status == "missing_file":
            issue = f"{artifact_label} file is missing"
        elif sha256_status == "missing":
            issue = f"{artifact_label} sha256 is missing"
        elif sha256_status == "mismatch":
            issue = f"{artifact_label} sha256 mismatch"
        elif content_status == "unreadable":
            issue = f"{artifact_label} file is not readable"
        elif content_status == "invalid":
            issue = f"{artifact_label} content is not review evidence"
        else:
            issue = f"{artifact_label} is not ready for SHA-256 claim checking"
        issue_item = {
            "family": family,
            "scope": scope,
            "artifact": artifact_label,
            "path": status.get("path", ""),
            "path_status": path_status,
            "sha256_status": sha256_status,
            "content_status": content_status,
            "content_size_bytes": status.get("content_size_bytes"),
            "min_content_size_bytes": status.get("min_content_size_bytes"),
            "content_issue_codes": status.get("content_issue_codes", []),
            "content_issues": status.get("content_issues", []),
            "placeholder_pattern_hits": status.get("placeholder_pattern_hits", []),
            "issue": issue,
        }
        if status.get("resolved_path"):
            issue_item["resolved_path"] = status["resolved_path"]
        issues.append(issue_item)
    return issues


def _source_audit_artifact_claim_refresh_payload(root=DATA_ROOT):
    root = Path(root)
    manifest_path = root / LOCAL_DATASET_MANIFEST_NAME
    local_manifest = load_local_dataset_manifest(root)
    if local_manifest["status"] != "present":
        report = {
            "status": f"blocked_manifest_{local_manifest['status']}",
            "path": _display_path(manifest_path),
            "error": local_manifest.get("error"),
        }
        return None, report

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_families = payload.setdefault("families", {})
    updates = []
    artifact_issues = []
    family_reports = []
    for family in DATA_FAMILIES:
        family_claim = manifest_families.get(family)
        family_updates = []
        family_artifact_issues = []
        file_reports = []
        if isinstance(family_claim, dict):
            family_audit = family_claim.get("source_audit")
            family_updates.extend(
                _source_audit_artifact_claim_updates_for_audit(
                    root,
                    family,
                    "family",
                    family_audit,
                )
            )
            family_artifact_issues.extend(
                _source_audit_artifact_claim_issues_for_audit(
                    root,
                    family,
                    "family",
                    family_audit,
                )
            )
            files = family_claim.get("files", [])
            if isinstance(files, list):
                for file_claim in files:
                    if not isinstance(file_claim, dict) or "source_audit" not in file_claim:
                        continue
                    file_scope = f"file:{file_claim.get('image', '<unknown>')}"
                    file_audit = file_claim.get("source_audit")
                    file_updates = _source_audit_artifact_claim_updates_for_audit(
                        root,
                        family,
                        file_scope,
                        file_audit,
                    )
                    file_artifact_issues = _source_audit_artifact_claim_issues_for_audit(
                        root,
                        family,
                        file_scope,
                        file_audit,
                    )
                    if file_updates or file_artifact_issues:
                        file_reports.append(
                            {
                                "image": file_claim.get("image", ""),
                                "updated_artifact_claim_count": len(file_updates),
                                "artifact_issue_count": len(file_artifact_issues),
                                "updates": file_updates,
                                "artifact_issues": file_artifact_issues,
                            }
                        )
                    family_updates.extend(file_updates)
                    family_artifact_issues.extend(file_artifact_issues)
        family_reports.append(
            {
                "family": family,
                "updated_artifact_claim_count": len(family_updates),
                "artifact_issue_count": len(family_artifact_issues),
                "updates": family_updates,
                "artifact_issues": family_artifact_issues,
                "file_overrides": file_reports,
            }
        )
        updates.extend(family_updates)
        artifact_issues.extend(family_artifact_issues)

    return payload, {
        "path": _display_path(manifest_path),
        "families": family_reports,
        "updates": updates,
        "updated_artifact_claim_count": len(updates),
        "artifact_issues": artifact_issues,
        "artifact_issue_count": len(artifact_issues),
        "downloaded_data": False,
    }


def refresh_source_audit_artifact_claims(root=DATA_ROOT):
    root = Path(root)
    manifest_path = root / LOCAL_DATASET_MANIFEST_NAME
    payload, report = _source_audit_artifact_claim_refresh_payload(root)
    if payload is None:
        return {
            **report,
            "written": False,
        }

    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        **report,
        "status": "updated",
        "written": True,
        "stale": (
            report.get("updated_artifact_claim_count", 0) > 0
            or report.get("artifact_issue_count", 0) > 0
        ),
    }


def check_source_audit_artifact_claims(root=DATA_ROOT):
    root = Path(root)
    payload, report = _source_audit_artifact_claim_refresh_payload(root)
    if payload is None:
        return {
            **report,
            "written": False,
            "stale": True,
        }

    stale = (
        report.get("updated_artifact_claim_count", 0) > 0
        or report.get("artifact_issue_count", 0) > 0
    )
    return {
        **report,
        "status": "stale" if stale else "current",
        "written": False,
        "stale": stale,
    }


def _source_summary(source):
    keys = [
        "source_id",
        "name",
        "url",
        "download_url",
        "priority",
        "fit",
        "download_policy",
        "license_note",
        "local_layout",
        "image_selection_rule",
        "mask_mapping_rule",
        "conversion_checklist",
        "final_naming_example",
    ]
    return {key: source[key] for key in keys if key in source}


def _source_acquisition_plan(family, sources):
    source = sources[0] if sources else {}
    return {
        "recommended_source_id": source.get("source_id", ""),
        "recommended_source_name": source.get("name", ""),
        "download_url": source.get("download_url", ""),
        "download_policy": source.get("download_policy", ""),
        "license_note": source.get("license_note", ""),
        "image_selection_rule": source.get("image_selection_rule", ""),
        "mask_mapping_rule": source.get("mask_mapping_rule", ""),
        "conversion_checklist": source.get("conversion_checklist", []),
        "final_naming_example": source.get("final_naming_example", ""),
        "target_paths": {
            "images": f"{family}/images",
            "masks": f"{family}/masks",
            "audit": f"{family}/audit",
            "source_audit": f"{family}/audit/source-artifact",
            "license_snapshot": f"{family}/audit/license-snapshot",
            "manifest": LOCAL_DATASET_MANIFEST_NAME,
        },
        "required_manifest_fields": [
            "source_id",
            "source_name",
            "license_reviewed",
            "license_note",
            "citation",
            "provenance_reviewed",
            "provenance_note",
            "synthetic_fixture",
            "source_audit",
            "source_url",
            "downloaded_at",
            "source_artifact_path",
            "source_artifact_sha256",
            "license_snapshot_path",
            "license_snapshot_sha256",
            "conversion_notes",
            "local_file_mapping_reviewed",
            "files[].image",
            "files[].sha256",
            "files[].mask",
            "files[].mask_sha256",
        ],
        "post_download_commands": [
            "python3 reproduce/experiments/iterated_rof_paper_like.py --refresh-manifest-file-claims",
            "python3 reproduce/experiments/iterated_rof_paper_like.py --refresh-source-audit-artifact-claims",
            "python3 reproduce/experiments/iterated_rof_paper_like.py --check-source-audit-artifact-claims",
            "python3 reproduce/experiments/iterated_rof_paper_like.py --check-manifest-file-claims",
            "python3 reproduce/experiments/iterated_rof_paper_like.py --data-gap-output /tmp/iterated_rof_data_gap.json",
            "python3 reproduce/experiments/iterated_rof_paper_like.py --strict-data-ready",
            "python3 reproduce/experiments/iterated_rof_paper_like.py --run --strict-paper-like",
        ],
    }


def _image_key(path, base):
    return path.relative_to(base).as_posix()


def _mask_index(mask_dir):
    by_relative_key = {}
    by_stem = {}
    for mask_path in _find_images(mask_dir):
        by_relative_key.setdefault(_image_key(mask_path, mask_dir), mask_path)
        by_stem.setdefault(mask_path.stem, []).append(mask_path)
    return by_relative_key, by_stem


def _resolve_mask(image_path, image_dir, masks_by_relative_key, masks_by_stem):
    mask_path = masks_by_relative_key.get(_image_key(image_path, image_dir))
    if mask_path is not None:
        return mask_path, None

    stem_matches = masks_by_stem.get(image_path.stem, [])
    if len(stem_matches) == 1:
        return None, (
            f"Stem-only mask match ignored for {_display_path(image_path)}: "
            "use the same relative path under masks/ for quantitative evidence"
        )
    if len(stem_matches) > 1:
        return None, (
            f"Ambiguous masks for {_display_path(image_path)}: "
            "use the same relative path under masks/ or make mask stems unique"
        )
    return None, None


def scan_dataset(root=DATA_ROOT, families=None):
    families = list(families or DATA_FAMILIES)
    entries = []
    for family in families:
        family_root = Path(root) / family
        image_dir = family_root / "images"
        mask_dir = family_root / "masks"
        masks_by_relative_key, masks_by_stem = _mask_index(mask_dir)

        for image_path in _find_images(image_dir):
            mask_path, mask_warning = _resolve_mask(image_path, image_dir, masks_by_relative_key, masks_by_stem)
            entries.append(
                {
                    "family": family,
                    "image_path": image_path,
                    "image_relative_path": image_path.relative_to(image_dir).as_posix(),
                    "mask_path": mask_path,
                    "mask_warning": mask_warning,
                }
            )
    return entries


def scan_family(root, family):
    family_root = Path(root) / family
    image_count = _count_images(family_root / "images")
    mask_count = _count_images(family_root / "masks")
    entries = scan_dataset(root, [family])
    matched_mask_count = sum(1 for entry in entries if entry["mask_path"] is not None)
    qualitative_ready = image_count > 0
    quantitative_ready = matched_mask_count > 0

    if quantitative_ready:
        status = "ready_quantitative"
    elif qualitative_ready:
        status = "ready_qualitative_only"
    else:
        status = "missing"

    return {
        "family": family,
        "description": DATA_FAMILIES[family],
        "image_count": image_count,
        "mask_count": mask_count,
        "matched_mask_count": matched_mask_count,
        "mask_warning_count": sum(1 for entry in entries if entry.get("mask_warning")),
        "status": status,
        "path": _display_path(family_root),
        "images": [
            {
                "image_path": _display_path(entry["image_path"]),
                "image_relative_path": entry["image_relative_path"],
                "mask_path": _display_path(entry["mask_path"]) if entry["mask_path"] else None,
                "mask_warning": entry.get("mask_warning"),
                "image_file": _file_evidence(entry["image_path"]),
                "mask_file": _file_evidence(entry["mask_path"]) if entry["mask_path"] else None,
            }
            for entry in entries
        ],
    }


def _known_source_ids(source_manifest):
    known_ids = {}
    if not isinstance(source_manifest, dict):
        return known_ids
    for family, family_sources in source_manifest.items():
        if not isinstance(family_sources, list):
            known_ids[family] = set()
            continue
        known_ids[family] = {
            str(source.get("source_id", "")).strip()
            for source in family_sources
            if isinstance(source, dict) and str(source.get("source_id", "")).strip()
        }
    return known_ids


def _registered_source_urls(source_manifest, family, source_id):
    if not isinstance(source_manifest, dict) or not family or not source_id:
        return set()
    urls = set()
    for source in source_manifest.get(family, []):
        if not isinstance(source, dict):
            continue
        if str(source.get("source_id", "")).strip() != str(source_id).strip():
            continue
        for key in ("url", "download_url"):
            value = str(source.get(key, "")).strip()
            if value:
                urls.add(value)
    return urls


def _source_audit_url_blockers(audit, family, source_id, source_manifest, message_prefix, label):
    if not isinstance(audit, dict) or not str(source_id or "").strip():
        return []
    source_url = str(audit.get("source_url", "")).strip()
    if not source_url:
        return []
    registered_urls = _registered_source_urls(source_manifest, family, source_id)
    if registered_urls and source_url not in registered_urls:
        return [
            f"{message_prefix} source_audit source_url is not registered for source_id {source_id} {label}"
        ]
    return []


def _has_manifest_text(family_claim, key):
    return bool(str(family_claim.get(key, "")).strip())


FIXTURE_CLAIM_TEXT_PATTERNS = (
    "test fixture",
    "tempfile",
    "temporary scaffold",
    "scaffold",
    "synthetic tempfile",
    "not a real dataset claim",
    "reviewed sample",
)
PLACEHOLDER_CLAIM_TEXT_PATTERNS = (
    "review dataset terms before paper-like promotion or redistribution",
    "review benchmark terms before paper-like promotion or redistribution",
    "review brainweb terms before paper-like promotion or redistribution",
    "record how local files were obtained from this source",
    "replace this template entry after placing local files",
    "add one files[] entry per local image",
)
SOURCE_AUDIT_ARTIFACT_PLACEHOLDER_PATTERNS = (
    "test fixture",
    "tiny local test",
    "fixture artifact",
    "fixture evidence",
    "placeholder",
    "reviewed source artifact",
    "reviewed license snapshot",
)
SOURCE_AUDIT_ARTIFACT_REVIEW_NOTE_RE = re.compile(
    r"\b(?:reviewer[_ ]note|review[_ ]note|source[_ ]review(?:[_ ]note)?)\s*[:=]\s*\S"
)
SOURCE_AUDIT_ARTIFACT_MAPPING_RE = re.compile(
    r"\b(?:conversion[_ ]notes?|mapping[_ ]note|local[_ ]file[_ ]mapping|file[_ ]mapping)\s*[:=]\s*\S"
)
FIXTURE_CLAIM_TEXT_FIELDS = ("license_note", "citation", "provenance_note", "notes")


def _has_valid_iso_date_token(token):
    try:
        date.fromisoformat(token)
        return True
    except ValueError:
        return False


def _has_source_audit_artifact_structure_issue(content_issue_codes):
    return any(
        code.startswith("missing_") or code == "invalid_review_date"
        for code in content_issue_codes
    )


def _source_audit_artifact_review_structure_issues(text, expected_source_url=None):
    structure_issues = []
    structure_issue_codes = []
    expected_source_url = str(expected_source_url or "").strip().lower()
    has_source_url = (
        re.search(r"https?://\S+", text)
        or re.search(r"\bsource_url\s*=\s*\S+", text)
    )
    if not has_source_url:
        structure_issue_codes.append("missing_source_url")
        structure_issues.append("artifact is missing source URL evidence")
    elif expected_source_url and expected_source_url not in text:
        structure_issue_codes.append("missing_manifest_source_url")
        structure_issues.append("artifact is missing manifest source URL evidence")
    date_tokens = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)
    if not date_tokens:
        structure_issue_codes.append("missing_review_date")
        structure_issues.append("artifact is missing review/download date evidence")
    elif not any(_has_valid_iso_date_token(token) for token in date_tokens):
        structure_issue_codes.append("invalid_review_date")
        structure_issues.append("artifact review/download date evidence is not a valid date")
    if not SOURCE_AUDIT_ARTIFACT_REVIEW_NOTE_RE.search(text):
        structure_issue_codes.append("missing_review_note")
        structure_issues.append("artifact is missing reviewer/source review note evidence")
    if not SOURCE_AUDIT_ARTIFACT_MAPPING_RE.search(text):
        structure_issue_codes.append("missing_conversion_or_mapping_note")
        structure_issues.append("artifact is missing conversion/mapping evidence")
    return structure_issue_codes, structure_issues


def _source_audit_artifact_content_report(content, expected_source_url=None):
    text = content[:8192].decode("utf-8", errors="ignore").lower()
    placeholder_hits = [
        pattern
        for pattern in SOURCE_AUDIT_ARTIFACT_PLACEHOLDER_PATTERNS
        if pattern in text
    ]
    content_issues = []
    content_issue_codes = []
    if len(content) < MIN_SOURCE_AUDIT_ARTIFACT_BYTES:
        content_issue_codes.append("too_small")
        content_issues.append(
            "artifact is too small to support review evidence"
        )
    if placeholder_hits:
        content_issue_codes.append("fixture_or_placeholder_text")
        content_issues.append("artifact contains fixture/placeholder text")
    if not content_issues:
        structure_issue_codes, structure_issues = _source_audit_artifact_review_structure_issues(
            text,
            expected_source_url,
        )
        content_issue_codes.extend(structure_issue_codes)
        content_issues.extend(structure_issues)
    return {
        "content_status": "invalid" if content_issues else "review_evidence_present",
        "content_size_bytes": len(content),
        "min_content_size_bytes": MIN_SOURCE_AUDIT_ARTIFACT_BYTES,
        "content_issue_codes": content_issue_codes,
        "content_issues": content_issues,
        "placeholder_pattern_hits": placeholder_hits,
    }


def _fixture_text_claim_blockers(claim, message_prefix, label):
    blockers = []
    for field in FIXTURE_CLAIM_TEXT_FIELDS:
        value = str(claim.get(field, "")).lower()
        if any(pattern in value for pattern in FIXTURE_CLAIM_TEXT_PATTERNS):
            blockers.append(f"{message_prefix} contains fixture/tempfile text in {field} {label}")
    return blockers


def _placeholder_text_claim_blockers(claim, message_prefix, label):
    blockers = []
    for field in FIXTURE_CLAIM_TEXT_FIELDS:
        value = str(claim.get(field, "")).lower()
        if any(pattern in value for pattern in PLACEHOLDER_CLAIM_TEXT_PATTERNS):
            blockers.append(f"{message_prefix} contains template placeholder text in {field} {label}")
    return blockers


def _source_audit_artifact_content_blockers(path, message_prefix, artifact_label, label, expected_source_url=None):
    blockers = []
    try:
        content = path.read_bytes()
    except Exception as exc:
        return [f"{message_prefix} source_audit {artifact_label} file is not readable {label}: {exc}"]
    content_report = _source_audit_artifact_content_report(content, expected_source_url)
    if "too_small" in content_report["content_issue_codes"]:
        blockers.append(
            f"{message_prefix} source_audit {artifact_label} artifact is too small to support review evidence {label}"
        )
    if "fixture_or_placeholder_text" in content_report["content_issue_codes"]:
        blockers.append(
            f"{message_prefix} source_audit {artifact_label} artifact contains fixture/placeholder text {label}"
        )
    if _has_source_audit_artifact_structure_issue(content_report["content_issue_codes"]):
        blockers.append(
            f"{message_prefix} source_audit {artifact_label} artifact is missing structured review evidence {label}"
        )
    return blockers


CLAIM_KEYS = [
    "source_id",
    "source_name",
    "license_reviewed",
    "license_note",
    "citation",
    "provenance_reviewed",
    "provenance_note",
    "synthetic_fixture",
    "notes",
    "source_audit",
]

SOURCE_AUDIT_TEXT_FIELDS = (
    "source_url",
    "downloaded_at",
    "source_artifact_path",
    "license_snapshot_path",
    "conversion_notes",
)
SOURCE_AUDIT_SHA_FIELDS = (
    "source_artifact_sha256",
    "license_snapshot_sha256",
)
SOURCE_AUDIT_ARTIFACT_FIELDS = (
    ("source_artifact_path", "source_artifact_sha256", "source_artifact"),
    ("license_snapshot_path", "license_snapshot_sha256", "license_snapshot"),
)


def _effective_claim(family_claim, file_claim=None):
    file_claim = file_claim or {}
    claim = {}
    for key in CLAIM_KEYS:
        if key in file_claim:
            claim[key] = file_claim[key]
        elif key in family_claim:
            claim[key] = family_claim[key]
    return claim


def _claim_requirement_blockers(family, claim, known_ids, label, source_manifest=None, audit_roots=None):
    blockers = []
    source_id = claim.get("source_id")
    if not source_id:
        blockers.append(f"Local dataset manifest missing source_id {label}")
    elif source_id not in known_ids.get(family, set()):
        blockers.append(f"Local dataset manifest source_id is not in source registry {label}: {source_id}")

    if claim.get("license_reviewed") is not True:
        blockers.append(f"Local dataset manifest has no license_reviewed=true {label}")
    if not _has_manifest_text(claim, "citation"):
        blockers.append(f"Local dataset manifest missing citation {label}")
    if not _has_manifest_text(claim, "license_note"):
        blockers.append(f"Local dataset manifest missing license_note {label}")
    if claim.get("provenance_reviewed") is not True:
        blockers.append(f"Local dataset manifest has no provenance_reviewed=true {label}")
    if not _has_manifest_text(claim, "provenance_note"):
        blockers.append(f"Local dataset manifest missing provenance_note {label}")
    if claim.get("synthetic_fixture") is not False:
        blockers.append(f"Local dataset manifest must explicitly set synthetic_fixture=false {label}")
    blockers.extend(_fixture_text_claim_blockers(claim, "Local dataset manifest", label))
    blockers.extend(_placeholder_text_claim_blockers(claim, "Local dataset manifest", label))
    blockers.extend(_source_audit_blockers(claim.get("source_audit"), "Local dataset manifest", label, audit_roots))
    blockers.extend(
        _source_audit_url_blockers(
            claim.get("source_audit"),
            family,
            source_id,
            source_manifest,
            "Local dataset manifest",
            label,
        )
    )
    return blockers


def _is_sha256(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[a-fA-F0-9]{64}", value))


def _is_iso_date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _source_audit_blockers(audit, message_prefix, label, audit_roots=None):
    blockers = []
    if not isinstance(audit, dict):
        return [f"{message_prefix} missing source_audit {label}"]
    for field in SOURCE_AUDIT_TEXT_FIELDS:
        if not str(audit.get(field, "")).strip():
            blockers.append(f"{message_prefix} source_audit missing {field} {label}")
    if audit.get("downloaded_at") and not _is_iso_date(audit.get("downloaded_at")):
        blockers.append(f"{message_prefix} source_audit downloaded_at must use a valid YYYY-MM-DD date {label}")
    for field in SOURCE_AUDIT_SHA_FIELDS:
        if not _is_sha256(audit.get(field)):
            blockers.append(f"{message_prefix} source_audit missing {field} {label}")
    for path_field, sha_field, artifact_label in SOURCE_AUDIT_ARTIFACT_FIELDS:
        artifact_path = _resolve_source_audit_path(audit.get(path_field), audit_roots)
        if artifact_path is None:
            continue
        if audit_roots:
            audit_data_roots = []
            for audit_root in audit_roots:
                try:
                    audit_data_roots.append(Path(audit_root).parent.parent)
                except Exception:
                    pass
            if audit_data_roots and not _path_is_under_any(artifact_path, audit_data_roots):
                blockers.append(f"{message_prefix} source_audit {artifact_label} path is outside local data root {label}")
                continue
            if not _path_is_under_any(artifact_path, audit_roots):
                blockers.append(f"{message_prefix} source_audit {artifact_label} path is outside local family audit root {label}")
                continue
        if not artifact_path.is_file():
            blockers.append(f"{message_prefix} source_audit {artifact_label} file is missing {label}")
            continue
        try:
            actual_evidence = _file_evidence(artifact_path)
        except Exception as exc:
            blockers.append(f"{message_prefix} source_audit {artifact_label} file is not readable {label}: {exc}")
            continue
        if _is_sha256(audit.get(sha_field)) and actual_evidence["sha256"] != audit.get(sha_field):
            blockers.append(f"{message_prefix} source_audit {artifact_label} sha256 mismatch {label}")
        blockers.extend(
            _source_audit_artifact_content_blockers(
                artifact_path,
                message_prefix,
                artifact_label,
                label,
                expected_source_url=audit.get("source_url"),
            )
        )
    if audit.get("local_file_mapping_reviewed") is not True:
        blockers.append(f"{message_prefix} source_audit has no local_file_mapping_reviewed=true {label}")
    return blockers


def _source_audit_roots_for_family(data_root, family):
    root = _resolve_report_path(data_root)
    if root is None or family not in DATA_FAMILIES:
        return []
    return [root / family / "audit"]


def _resolve_source_audit_path(path_value, audit_roots=None, data_root=None):
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path

    repo_path = REPO_ROOT / path
    roots = [Path(root) for root in (audit_roots or [])]
    data_roots = []
    if data_root is not None:
        resolved_data_root = _resolve_report_path(data_root)
        if resolved_data_root is not None:
            data_roots.append(resolved_data_root)
    for audit_root in roots:
        data_roots.append(audit_root.parent.parent)

    if repo_path.exists() or _path_is_under_any(repo_path, data_roots):
        return repo_path
    if data_roots:
        return data_roots[0] / path
    return repo_path


def _source_audit_field_statuses(audit, label):
    statuses = {}
    issues = []
    for field in SOURCE_AUDIT_TEXT_FIELDS:
        value = str(audit.get(field, "")).strip()
        status = "present" if value else "missing"
        if field == "downloaded_at" and value and not _is_iso_date(value):
            status = "invalid_date"
        statuses[field] = status
        if status == "missing":
            issues.append(f"{label} source_audit missing {field}")
        elif status == "invalid_date":
            issues.append(f"{label} source_audit downloaded_at must use a valid YYYY-MM-DD date")
    mapping_reviewed = audit.get("local_file_mapping_reviewed") is True
    statuses["local_file_mapping_reviewed"] = "reviewed" if mapping_reviewed else "missing"
    if not mapping_reviewed:
        issues.append(f"{label} source_audit has no local_file_mapping_reviewed=true")
    return statuses, issues


def _source_audit_artifact_status(label, audit, data_root, audit_root, path_field, sha_field, artifact_label):
    raw_path = str(audit.get(path_field, "")).strip()
    expected_sha = audit.get(sha_field)
    status = {
        "path": raw_path,
        "path_status": "missing_path",
        "sha256_status": "missing",
        "content_status": "not_checked",
        "content_size_bytes": None,
        "min_content_size_bytes": MIN_SOURCE_AUDIT_ARTIFACT_BYTES,
        "content_issue_codes": [],
        "content_issues": [],
        "placeholder_pattern_hits": [],
        "ready": False,
    }
    issues = []

    if not raw_path:
        issues.append(f"{label} source_audit {artifact_label} path is missing")
        return status, issues

    artifact_path = _resolve_source_audit_path(raw_path, [audit_root], data_root)
    status["resolved_path"] = _display_path(artifact_path)
    if artifact_path is None:
        issues.append(f"{label} source_audit {artifact_label} path is missing")
        return status, issues

    if not _path_is_under(artifact_path, data_root):
        status["path_status"] = "outside_data_root"
        status["sha256_status"] = "not_checked"
        issues.append(f"{label} source_audit {artifact_label} path is outside local data root")
        return status, issues
    if not _path_is_under(artifact_path, audit_root):
        status["path_status"] = "outside_family_audit_root"
        status["sha256_status"] = "not_checked"
        issues.append(f"{label} source_audit {artifact_label} path is outside local family audit root")
        return status, issues
    if not artifact_path.is_file():
        status["path_status"] = "missing_file"
        if _is_sha256(expected_sha):
            status["sha256_status"] = "not_checked"
        issues.append(f"{label} source_audit {artifact_label} file is missing")
        return status, issues

    status["path_status"] = "present"
    if not _is_sha256(expected_sha):
        issues.append(f"{label} source_audit {artifact_label} sha256 is missing")
        return status, issues

    actual_sha = _file_evidence(artifact_path)["sha256"]
    status["actual_sha256"] = actual_sha
    if actual_sha != expected_sha:
        status["sha256_status"] = "mismatch"
        issues.append(f"{label} source_audit {artifact_label} sha256 mismatch")
        return status, issues

    status["sha256_status"] = "matched"
    try:
        content = artifact_path.read_bytes()
    except Exception as exc:
        status["content_status"] = "unreadable"
        issues.append(f"{label} source_audit {artifact_label} file is not readable: {exc}")
        return status, issues
    content_report = _source_audit_artifact_content_report(
        content,
        expected_source_url=audit.get("source_url"),
    )
    status.update(content_report)
    if content_report["content_issues"]:
        status["content_status"] = "invalid"
        if "too_small" in content_report["content_issue_codes"]:
            issues.append(f"{label} source_audit {artifact_label} artifact is too small to support review evidence")
        if "fixture_or_placeholder_text" in content_report["content_issue_codes"]:
            issues.append(f"{label} source_audit {artifact_label} artifact contains fixture/placeholder text")
        if _has_source_audit_artifact_structure_issue(content_report["content_issue_codes"]):
            issues.append(f"{label} source_audit {artifact_label} artifact is missing structured review evidence")
        return status, issues
    status["content_status"] = "review_evidence_present"
    status["ready"] = True
    return status, issues


def _source_audit_gap_status(label, audit, data_root, audit_root):
    field_statuses, issues = _source_audit_field_statuses(audit, label)
    artifacts = {}
    for path_field, sha_field, artifact_label in SOURCE_AUDIT_ARTIFACT_FIELDS:
        artifact_status, artifact_issues = _source_audit_artifact_status(
            label,
            audit,
            data_root,
            audit_root,
            path_field,
            sha_field,
            artifact_label,
        )
        artifacts[artifact_label] = artifact_status
        issues.extend(artifact_issues)
    return {
        "status": "complete" if not issues else "incomplete",
        "fields": field_statuses,
        "artifacts": artifacts,
        "issue_count": len(issues),
        "issues": issues,
        "ready": not issues,
    }


def _source_audit_gap_status_for_family(family, local_manifest, data_root):
    data_root = _resolve_report_path(data_root) or DATA_ROOT
    audit_root = data_root / family / "audit"
    summary = {
        "status": "missing_manifest",
        "expected_root": _display_path(audit_root),
        "audit_root_exists": audit_root.is_dir(),
        "fields": {},
        "artifacts": {},
        "file_overrides": [],
        "issue_count": 0,
        "issues": [],
        "ready": False,
    }

    manifest_status = local_manifest.get("status")
    if manifest_status != "present":
        if manifest_status == "invalid":
            summary["status"] = "invalid_manifest"
            summary["issues"] = [f"{family} source_audit cannot be checked because dataset_manifest.json is invalid"]
        else:
            summary["issues"] = [f"{family} source_audit cannot be checked because dataset_manifest.json is missing"]
        summary["issue_count"] = len(summary["issues"])
        return summary

    family_claim = local_manifest.get("families", {}).get(family)
    if not isinstance(family_claim, dict):
        summary["status"] = "missing_family_claim"
        summary["issues"] = [f"{family} source_audit cannot be checked because family claim is missing"]
        summary["issue_count"] = len(summary["issues"])
        return summary

    audit = family_claim.get("source_audit")
    if not isinstance(audit, dict):
        summary["status"] = "missing_source_audit"
        summary["issues"] = [f"{family} source_audit is missing"]
        summary["issue_count"] = len(summary["issues"])
        return summary

    family_status = _source_audit_gap_status(family, audit, data_root, audit_root)
    summary["fields"] = family_status["fields"]
    summary["artifacts"] = family_status["artifacts"]
    issues = list(family_status["issues"])

    files = family_claim.get("files")
    if isinstance(files, list):
        for file_claim in files:
            if not isinstance(file_claim, dict) or "source_audit" not in file_claim:
                continue
            image_key = str(file_claim.get("image", "")).strip() or "<unknown image>"
            file_label = f"{family}/{image_key}"
            file_audit = file_claim.get("source_audit")
            if not isinstance(file_audit, dict):
                file_status = {
                    "image": image_key,
                    "status": "missing_source_audit",
                    "fields": {},
                    "artifacts": {},
                    "issue_count": 1,
                    "issues": [f"{file_label} source_audit is missing"],
                    "ready": False,
                }
            else:
                file_status = {
                    "image": image_key,
                    **_source_audit_gap_status(file_label, file_audit, data_root, audit_root),
                }
            summary["file_overrides"].append(file_status)
            issues.extend(file_status["issues"])

    summary["issues"] = issues
    summary["issue_count"] = len(issues)
    summary["ready"] = not issues
    summary["status"] = "complete" if summary["ready"] else "incomplete"
    return summary


def _file_claim_index(family, family_claim, blockers):
    files = family_claim.get("files")
    if not isinstance(files, list):
        blockers.append(f"Local dataset manifest missing files list for: {family}")
        return {}

    index = {}
    for file_claim in files:
        if not isinstance(file_claim, dict):
            blockers.append(f"Local dataset manifest file claim is not an object for: {family}")
            continue
        image_key = str(file_claim.get("image", "")).strip()
        if not image_key:
            blockers.append(f"Local dataset manifest file claim missing image for: {family}")
            continue
        if image_key in index:
            blockers.append(f"Local dataset manifest duplicate file claim for: {family}/{image_key}")
            continue
        index[image_key] = file_claim
    return index


def _claim_blockers(families, local_manifest, source_manifest):
    families_with_images = [item["family"] for item in families if item["image_count"] > 0]
    if not families_with_images:
        return []

    if local_manifest["status"] == "missing":
        return [
            "Local dataset manifest missing for present images: "
            f"add {local_manifest['path']} before paper-like promotion"
        ]
    if local_manifest["status"] == "invalid":
        return [
            f"Local dataset manifest is invalid: {local_manifest.get('error', 'unknown error')}"
        ]

    manifest_families = local_manifest["families"]
    manifest_path = _resolve_report_path(local_manifest.get("path"))
    manifest_data_root = manifest_path.parent if manifest_path is not None else None
    blockers = _source_manifest_schema_blockers(source_manifest)
    known_ids = _known_source_ids(source_manifest)
    for family in families_with_images:
        family_claim = manifest_families.get(family)
        if not isinstance(family_claim, dict):
            blockers.append(f"Local dataset manifest missing family entry: {family}")
            continue

        audit_roots = _source_audit_roots_for_family(manifest_data_root, family)
        blockers.extend(_claim_requirement_blockers(family, family_claim, known_ids, f"for: {family}", source_manifest, audit_roots))
        file_claims = _file_claim_index(family, family_claim, blockers)
        local_image_keys = set()
        for image_info in next(item for item in families if item["family"] == family).get("images", []):
            image_key = image_info.get("image_relative_path")
            if not image_key:
                continue
            local_image_keys.add(image_key)
            file_claim = file_claims.get(image_key)
            if not file_claim:
                blockers.append(f"Local dataset manifest missing file claim for: {family}/{image_key}")
                continue
            mask_claim = file_claim.get("mask")
            if image_info.get("mask_path") and mask_claim != image_key:
                blockers.append(f"Local dataset manifest missing matching mask claim for: {family}/{image_key}")
            if not image_info.get("mask_path") and mask_claim:
                blockers.append(f"Local dataset manifest mask claim has no local mask for: {family}/{image_key}")

            expected_sha256 = image_info.get("image_file", {}).get("sha256")
            claimed_sha256 = file_claim.get("sha256")
            if not claimed_sha256:
                blockers.append(f"Local dataset manifest missing sha256 for file: {family}/{image_key}")
            elif expected_sha256 and claimed_sha256 != expected_sha256:
                blockers.append(f"Local dataset manifest sha256 mismatch for file: {family}/{image_key}")

            expected_mask_sha256 = (image_info.get("mask_file") or {}).get("sha256")
            claimed_mask_sha256 = file_claim.get("mask_sha256")
            if image_info.get("mask_path") and not claimed_mask_sha256:
                blockers.append(f"Local dataset manifest missing mask_sha256 for file: {family}/{image_key}")
            elif expected_mask_sha256 and claimed_mask_sha256 != expected_mask_sha256:
                blockers.append(f"Local dataset manifest mask_sha256 mismatch for file: {family}/{image_key}")
            blockers.extend(
                _claim_requirement_blockers(
                    family,
                    _effective_claim(family_claim, file_claim),
                    known_ids,
                    f"for file: {family}/{image_key}",
                    source_manifest,
                    audit_roots,
                )
            )
        for image_key in sorted(set(file_claims) - local_image_keys):
            blockers.append(f"Local dataset manifest file claim has no local image: {family}/{image_key}")
    return blockers


def _entry_source_claim(entry, local_manifest):
    family = entry["family"]
    image_key = entry.get("image_relative_path")
    family_claim = local_manifest.get("families", {}).get(family, {})
    if not isinstance(family_claim, dict):
        family_claim = {}
    file_claim = _file_claim_index(family, family_claim, []).get(image_key, {})
    claim_scope = "file" if file_claim else "family"
    effective_claim = _effective_claim(family_claim, file_claim)
    return {
        "manifest_status": local_manifest["status"],
        "manifest_path": local_manifest["path"],
        "claim_scope": claim_scope,
        "image": image_key,
        **(
            {
                key: file_claim.get(key)
                for key in ["mask", "sha256", "mask_sha256"]
                if key in file_claim
            }
            if file_claim
            else {}
        ),
        **effective_claim,
    }


def _expected_report_source_claim_from_manifest(item, local_manifest):
    family = item.get("family")
    image_key = _image_result_relative_path(item)
    family_claim = local_manifest.get("families", {}).get(family, {})
    if not isinstance(family_claim, dict):
        return None
    file_claim = _file_claim_index(family, family_claim, []).get(image_key)
    if not file_claim:
        return None
    effective_claim = _effective_claim(family_claim, file_claim)
    return {
        "manifest_status": local_manifest["status"],
        "manifest_path": local_manifest["path"],
        "claim_scope": "file",
        "image": image_key,
        **{
            key: file_claim.get(key)
            for key in ["mask", "sha256", "mask_sha256"]
            if key in file_claim
        },
        **effective_claim,
    }


def _source_claim_matches_manifest(item, local_manifest):
    expected_claim = _expected_report_source_claim_from_manifest(item, local_manifest)
    if expected_claim is None:
        return False
    source_claim = item.get("source_claim", {})
    if not isinstance(source_claim, dict):
        return False
    for key, expected_value in expected_claim.items():
        if source_claim.get(key) != expected_value:
            return False
    extra_review_keys = set(source_claim) & set(CLAIM_KEYS)
    for key in extra_review_keys - set(expected_claim):
        if source_claim.get(key) not in (None, ""):
            return False
    return True


def _file_evidence(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": _display_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _resolve_report_path(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _same_report_path(left_value, right_value):
    left_path = _resolve_report_path(left_value)
    right_path = _resolve_report_path(right_value)
    if left_path is None or right_path is None:
        return False
    try:
        return left_path.resolve() == right_path.resolve()
    except Exception:
        return left_path.absolute() == right_path.absolute()


def _path_is_under(path_value, root):
    path = _resolve_report_path(path_value)
    if path is None:
        return False
    try:
        path.resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def _path_is_under_any(path_value, roots):
    return any(_path_is_under(path_value, root) for root in roots if root)


def _path_suffix(path_value):
    path = _resolve_report_path(path_value)
    return path.suffix.lower() if path is not None else ""


def _has_png_signature(path_value):
    path = _resolve_report_path(path_value)
    if path is None or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(8) == b"\x89PNG\r\n\x1a\n"
    except Exception:
        return False


def _is_decodable_image_file(path_value):
    path = _resolve_report_path(path_value)
    if path is None or not path.is_file():
        return False
    try:
        array = _read_image(path)
    except Exception:
        return False
    return getattr(array, "size", 0) > 0


def _figure_visual_blockers(path_value, image_label, prefix):
    path = _resolve_report_path(path_value)
    if path is None or not path.is_file():
        return []
    try:
        import numpy as np

        array = np.asarray(_read_image(path))
    except Exception:
        return [f"{prefix} figure file is not decodable for: {image_label}"]
    if array.size == 0:
        return [f"{prefix} figure file is not decodable for: {image_label}"]
    if array.ndim < 2 or array.shape[0] < 8 or array.shape[1] < 8:
        return [f"{prefix} figure file is too small for visual evidence for: {image_label}"]
    visual = array[..., :3] if array.ndim == 3 and array.shape[-1] >= 3 else array
    finite_values = visual[np.isfinite(visual)]
    if finite_values.size == 0:
        return [f"{prefix} figure file has no finite pixels for: {image_label}"]
    if float(finite_values.max()) == float(finite_values.min()):
        return [f"{prefix} figure file is visually blank for: {image_label}"]
    return []


def _input_image_content_blockers(path_value, image_label, prefix):
    path = _resolve_report_path(path_value)
    if path is None or not path.is_file():
        return []
    try:
        import numpy as np

        array = np.asarray(load_grayscale_image(path))
    except Exception:
        return [f"{prefix} input image is not decodable for paper-like evidence for: {image_label}"]
    if array.size == 0 or array.ndim < 2:
        return [f"{prefix} input image is not decodable for paper-like evidence for: {image_label}"]
    if array.shape[0] < MIN_PAPER_LIKE_IMAGE_SIDE or array.shape[1] < MIN_PAPER_LIKE_IMAGE_SIDE:
        return [f"{prefix} input image is too small for paper-like evidence for: {image_label}"]
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return [f"{prefix} input image has no finite pixels for paper-like evidence for: {image_label}"]
    if float(finite_values.max()) == float(finite_values.min()):
        return [f"{prefix} input image is visually blank for paper-like evidence for: {image_label}"]
    unique_levels = np.unique(np.rint(np.clip(finite_values, 0.0, 1.0) * 255.0)).size
    if unique_levels < MIN_PAPER_LIKE_IMAGE_LEVELS:
        return [f"{prefix} input image has too few gray levels for paper-like evidence for: {image_label}"]
    return []


def _mask_content_blockers(path_value, image_label, prefix):
    path = _resolve_report_path(path_value)
    if path is None or not path.is_file():
        return []
    try:
        import numpy as np

        mask = np.asarray(load_mask(path))
    except Exception:
        return [f"{prefix} mask is not decodable for paper-like evidence for: {image_label}"]
    if mask.size == 0 or mask.ndim < 2:
        return [f"{prefix} mask is not decodable for paper-like evidence for: {image_label}"]
    finite_values = mask[np.isfinite(mask)]
    if finite_values.size == 0:
        return [f"{prefix} mask has no finite labels for paper-like evidence for: {image_label}"]
    if np.unique(finite_values).size < 2:
        return [f"{prefix} mask has fewer than two labels for paper-like evidence for: {image_label}"]
    return []


def _mask_shape_blockers(image_path_value, mask_path_value, image_label, prefix):
    image_path = _resolve_report_path(image_path_value)
    mask_path = _resolve_report_path(mask_path_value)
    if image_path is None or mask_path is None or not image_path.is_file() or not mask_path.is_file():
        return []
    try:
        image = load_grayscale_image(image_path)
        load_mask(mask_path, expected_shape=image.shape)
    except Exception as exc:
        message = str(exc)
        if "does not match image shape" in message:
            return [f"{prefix} mask shape does not match image shape for: {image_label}: {message}"]
    return []


def _preflight_content_issues(family_info):
    issues = []
    for image_info in family_info.get("images", []):
        image_path = image_info.get("image_path")
        image_label = image_info.get("image_relative_path") or image_path or "<unknown image>"
        issues.extend(
            _input_image_content_blockers(
                image_path,
                image_label,
                "Preflight",
            )
        )
        mask_path = image_info.get("mask_path")
        if not mask_path:
            continue
        try:
            image = load_grayscale_image(_resolve_report_path(image_path))
            load_mask(_resolve_report_path(mask_path), expected_shape=image.shape)
        except Exception as exc:
            message = str(exc)
            if "does not match image shape" in message:
                issues.append(f"Preflight mask shape does not match image shape for: {image_label}: {message}")
            else:
                issues.append(f"Preflight mask is not decodable for: {image_label}: {message}")
            continue
        issues.extend(_mask_content_blockers(mask_path, image_label, "Preflight"))
    return issues


def _file_evidence_matches_disk(path_value, evidence):
    path = _resolve_report_path(path_value)
    if path is None or not path.is_file() or not isinstance(evidence, dict):
        return False
    if not evidence.get("path") or not _same_report_path(path_value, evidence.get("path")):
        return False
    actual = _file_evidence(path)
    return (
        actual["sha256"] == evidence.get("sha256")
        and actual["size_bytes"] == evidence.get("size_bytes")
    )


def _figure_evidence_sidecar_path(figure_path):
    path = _resolve_report_path(figure_path)
    if path is None:
        return None
    return path.with_suffix(path.suffix + ".evidence.json")


def _figure_evidence_payload(item):
    figure_file = item.get("figure_file", {}) if isinstance(item.get("figure_file"), dict) else {}
    image_file = item.get("image_file", {}) if isinstance(item.get("image_file"), dict) else {}
    mask_file = item.get("mask_file", {}) if isinstance(item.get("mask_file"), dict) else {}
    baselines = item.get("baselines", {}) if isinstance(item.get("baselines"), dict) else {}
    baseline_evidence = {}
    for name, baseline in sorted(baselines.items()):
        if not isinstance(baseline, dict):
            continue
        baseline_evidence[name] = {
            key: baseline[key]
            for key in ["method", "thresholds"]
            if key in baseline
        }
    return {
        "schema_version": 1,
        "paper_id": "iterated-rof",
        "generator": FIGURE_EVIDENCE_GENERATOR,
        "family": item.get("family"),
        "image_path": item.get("image_path"),
        "mask_path": item.get("mask_path") or "",
        "figure_path": item.get("figure_path"),
        "qualitative_only": bool(item.get("qualitative_only")),
        "image_sha256": image_file.get("sha256", ""),
        "mask_sha256": mask_file.get("sha256", "") if item.get("mask_path") else "",
        "figure_sha256": figure_file.get("sha256", ""),
        "figure_size_bytes": figure_file.get("size_bytes", ""),
        "figure_panels": list(item.get("figure_panels", [])),
        "solver": item.get("solver", ""),
        "parameters": item.get("parameters", {}),
        "thresholds": item.get("thresholds", []),
        "n_classes": item.get("n_classes", ""),
        "metrics": item.get("metrics", {}),
        "baselines": baseline_evidence,
    }


def _write_figure_evidence_sidecar(item):
    sidecar_path = _figure_evidence_sidecar_path(item.get("figure_path"))
    if sidecar_path is None:
        return {}
    payload = _figure_evidence_payload(item)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "figure_evidence": payload,
        "figure_evidence_path": _display_path(sidecar_path),
        "figure_evidence_file": _file_evidence(sidecar_path),
    }


def _figure_evidence_blockers(item, prefix):
    image_label = item.get("image_path", "<unknown image>")
    if not item.get("figure_path"):
        return []
    reason_prefix = f"{prefix} " if prefix else ""

    sidecar_path = item.get("figure_evidence_path")
    sidecar_file = item.get("figure_evidence_file")
    sidecar_payload = item.get("figure_evidence")
    missing_message = "missing figure evidence sidecar" if prefix else "Missing figure evidence sidecar"
    path_message = (
        "figure evidence sidecar path does not match figure path"
        if prefix
        else "Figure evidence sidecar path does not match figure path"
    )
    disk_message = (
        "figure evidence sidecar does not match disk"
        if prefix
        else "Figure evidence sidecar does not match disk"
    )
    report_message = (
        "figure evidence sidecar does not match report"
        if prefix
        else "Figure evidence sidecar does not match report"
    )
    if not sidecar_path or not isinstance(sidecar_file, dict) or not isinstance(sidecar_payload, dict):
        return [f"{reason_prefix}{missing_message} for: {image_label}"]

    expected_sidecar_path = _figure_evidence_sidecar_path(item.get("figure_path"))
    if expected_sidecar_path is None or not _same_report_path(sidecar_path, expected_sidecar_path):
        return [f"{reason_prefix}{path_message} for: {image_label}"]

    blockers = []
    if not _file_evidence_matches_disk(sidecar_path, sidecar_file):
        blockers.append(f"{reason_prefix}{disk_message} for: {image_label}")

    try:
        disk_payload = json.loads(_resolve_report_path(sidecar_path).read_text(encoding="utf-8"))
    except Exception:
        disk_payload = None
    expected_payload = _figure_evidence_payload(item)
    if disk_payload != sidecar_payload or sidecar_payload != expected_payload:
        blockers.append(f"{reason_prefix}{report_message} for: {image_label}")
    return blockers


def _allowed_figure_dirs(figure_dir=None):
    roots = [FIGURE_DIR]
    if figure_dir:
        resolved = _resolve_report_path(figure_dir)
        if resolved is not None:
            roots.append(resolved)
    unique_roots = []
    seen = set()
    for root in roots:
        try:
            key = str(Path(root).resolve())
        except Exception:
            key = str(root)
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    return unique_roots


def _completed_output_shape_blockers(item, path_prefix, source_prefix, source_manifest=None, figure_dir=None, audit_roots=None):
    blockers = []
    family = item.get("family")
    image_label = item.get("image_path", "<unknown image>")
    source_claim = item.get("source_claim", {})
    if not isinstance(source_claim, dict):
        source_claim = {}

    if family not in DATA_FAMILIES:
        blockers.append(f"{path_prefix} family is not recognized for: {image_label}: {family}")
        family = str(family or "")

    if _declares_quantitative_image_result(item) and not item.get("mask_path"):
        blockers.append(f"{path_prefix} quantitative row is missing mask path for: {image_label}")

    if not _path_is_under(item.get("image_path"), DATA_ROOT / family / "images"):
        blockers.append(f"{path_prefix} image path is outside canonical family images directory for: {image_label}")
    if _path_suffix(item.get("image_path")) not in IMAGE_EXTENSIONS:
        blockers.append(f"{path_prefix} image path has unsupported extension for: {image_label}")
    elif not _is_decodable_image_file(item.get("image_path")):
        blockers.append(f"{path_prefix} image file is not decodable for: {image_label}")
    else:
        blockers.extend(_input_image_content_blockers(item.get("image_path"), image_label, path_prefix))

    if item.get("mask_path"):
        if not _path_is_under(item.get("mask_path"), DATA_ROOT / family / "masks"):
            blockers.append(f"{path_prefix} mask path is outside canonical family masks directory for: {image_label}")
        if _path_suffix(item.get("mask_path")) not in IMAGE_EXTENSIONS:
            blockers.append(f"{path_prefix} mask path has unsupported extension for: {image_label}")
        elif not _is_decodable_image_file(item.get("mask_path")):
            blockers.append(f"{path_prefix} mask file is not decodable for: {image_label}")
        else:
            blockers.extend(_mask_content_blockers(item.get("mask_path"), image_label, path_prefix))
            blockers.extend(_mask_shape_blockers(item.get("image_path"), item.get("mask_path"), image_label, path_prefix))
        if _mask_result_relative_path(item) != _image_result_relative_path(item):
            blockers.append(f"{path_prefix} mask relative path does not match local image path for: {image_label}")

    if item.get("figure_path"):
        if not _path_is_under_any(item.get("figure_path"), _allowed_figure_dirs(figure_dir)):
            blockers.append(f"{path_prefix} figure path is outside allowed figure directory for: {image_label}")
        if _path_suffix(item.get("figure_path")) != ".png" or not _has_png_signature(item.get("figure_path")):
            blockers.append(f"{path_prefix} figure file is not a PNG for: {image_label}")
        else:
            blockers.extend(_figure_visual_blockers(item.get("figure_path"), image_label, path_prefix))

    source_manifest = source_manifest or load_source_manifest()
    source_id = str(source_claim.get("source_id", "")).strip()
    if source_id and source_id not in _known_source_ids(source_manifest).get(item.get("family"), set()):
        blockers.append(f"{source_prefix} source_id is not in source registry for: {image_label}: {source_id}")
    blockers.extend(_source_audit_blockers(source_claim.get("source_audit"), source_prefix, f"for: {image_label}", audit_roots))
    blockers.extend(
        _source_audit_url_blockers(
            source_claim.get("source_audit"),
            item.get("family"),
            source_id,
            source_manifest,
            source_prefix,
            f"for: {image_label}",
        )
    )
    return blockers


def _dataset_fingerprint_from_records(records):
    digest = hashlib.sha256()
    records = sorted(
        records,
        key=lambda record: (
            record["family"],
            record["path"],
            0 if record["kind"] == "image" else 1,
        ),
    )
    for record in records:
        digest.update(
            f"{record['kind']}\t{record['family']}\t{record['path']}\t{record['sha256']}\n".encode("utf-8")
        )

    return {
        "algorithm": "sha256",
        "file_count": len(records),
        "image_count": sum(1 for record in records if record["kind"] == "image"),
        "mask_count": sum(1 for record in records if record["kind"] == "mask"),
        "sha256": digest.hexdigest(),
    }


def _dataset_fingerprint(entries):
    records = []
    for entry in sorted(entries, key=lambda item: (item["family"], item["image_relative_path"])):
        image_evidence = _file_evidence(entry["image_path"])
        records.append(
            {
                "kind": "image",
                "family": entry["family"],
                "path": entry["image_relative_path"],
                "sha256": image_evidence["sha256"],
            }
        )
        if entry.get("mask_path"):
            mask_evidence = _file_evidence(entry["mask_path"])
            records.append(
                {
                    "kind": "mask",
                    "family": entry["family"],
                    "path": entry["image_relative_path"],
                    "sha256": mask_evidence["sha256"],
                }
            )

    return _dataset_fingerprint_from_records(records)


def _path_relative_after_component(path_value, component):
    path = Path(str(path_value or ""))
    parts = path.parts
    component_indexes = [index for index, part in enumerate(parts) if part == component]
    if component_indexes:
        suffix = parts[component_indexes[-1] + 1 :]
        if suffix:
            return Path(*suffix).as_posix()
    return path.name


def _path_relative_to_family_kind(path_value, family, kind):
    if not family:
        return None
    path = _resolve_report_path(path_value)
    if not path:
        return None
    try:
        return path.resolve().relative_to((DATA_ROOT / family / kind).resolve()).as_posix()
    except ValueError:
        return None


def _image_result_relative_path(item):
    if not isinstance(item, dict):
        return ""
    return (
        _path_relative_to_family_kind(item.get("image_path", ""), item.get("family"), "images")
        or _path_relative_after_component(item.get("image_path", ""), "images")
    )


def _mask_result_relative_path(item):
    if not isinstance(item, dict):
        return ""
    return (
        _path_relative_to_family_kind(item.get("mask_path", ""), item.get("family"), "masks")
        or _path_relative_after_component(item.get("mask_path", ""), "masks")
    )


def _dataset_fingerprint_from_image_results(image_results):
    records = []
    for item in image_results or []:
        if not isinstance(item, dict):
            continue
        family = item.get("family")
        image_sha256 = item.get("image_file", {}).get("sha256")
        if family and image_sha256:
            records.append(
                {
                    "kind": "image",
                    "family": family,
                    "path": _image_result_relative_path(item),
                    "sha256": image_sha256,
                }
            )
        mask_sha256 = (item.get("mask_file") or {}).get("sha256")
        if family and item.get("mask_path") and mask_sha256:
            records.append(
                {
                    "kind": "mask",
                    "family": family,
                    "path": _mask_result_relative_path(item),
                    "sha256": mask_sha256,
                }
            )
    return _dataset_fingerprint_from_records(records)


def _fingerprint_matches_image_results(dataset_fingerprint, image_results):
    expected = _dataset_fingerprint_from_image_results(image_results)
    return (
        expected["file_count"] > 0
        and dataset_fingerprint.get("algorithm") == expected["algorithm"]
        and dataset_fingerprint.get("file_count") == expected["file_count"]
        and dataset_fingerprint.get("image_count") == expected["image_count"]
        and dataset_fingerprint.get("mask_count") == expected["mask_count"]
        and dataset_fingerprint.get("sha256") == expected["sha256"]
    )


def _dataset_fingerprints_match(left, right):
    return (
        isinstance(left, dict)
        and isinstance(right, dict)
        and left.get("algorithm") == right.get("algorithm") == "sha256"
        and left.get("file_count") == right.get("file_count")
        and left.get("image_count") == right.get("image_count")
        and left.get("mask_count") == right.get("mask_count")
        and left.get("sha256") == right.get("sha256")
    )


def _append_unique(items, value):
    if value and value not in items:
        items.append(value)


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _declares_quantitative_image_result(item):
    return item.get("status") == "completed" and item.get("qualitative_only") is False


def _is_quantitative_image_result(item):
    return _declares_quantitative_image_result(item) and bool(item.get("mask_path"))


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _numeric_sequence(value):
    if not isinstance(value, list) or not value:
        return False
    return all(_is_finite_number(item) for item in value)


EXPECTED_LOCAL_SOLVER = "sat_rof_trof.rof_chambolle_pock + sat_rof_trof.run_trof_thresholds"
EXPECTED_RAW_KMEANS_METHOD = "simple_kmeans_on_raw_grayscale"
ALLOWED_MULTI_OTSU_METHODS = {"skimage_multiotsu", "quantile_fallback"}


def _runner_evidence_blockers(item):
    blockers = []
    image_label = item.get("image_path", "<unknown image>")
    if item.get("solver") != EXPECTED_LOCAL_SOLVER:
        blockers.append(f"Missing solver evidence for: {image_label}")
    n_classes = item.get("n_classes")
    if not _is_positive_int(n_classes) or n_classes < 2:
        blockers.append(f"Missing n_classes evidence for: {image_label}")
    thresholds = item.get("thresholds")
    if not _numeric_sequence(thresholds):
        blockers.append(f"Missing threshold evidence for: {image_label}")
    elif _is_positive_int(n_classes) and len(thresholds) != n_classes - 1:
        blockers.append(f"Threshold evidence does not match n_classes for: {image_label}")
    if not _is_positive_int(item.get("threshold_iterations")):
        blockers.append(f"Missing threshold iteration evidence for: {image_label}")
    if not _is_positive_int(item.get("rof_iterations")):
        blockers.append(f"Missing ROF iteration evidence for: {image_label}")
    residual = item.get("rof_final_residual")
    if not _is_finite_number(residual) or float(residual) < 0:
        blockers.append(f"Missing ROF residual evidence for: {image_label}")
    parameters = item.get("parameters")
    if not isinstance(parameters, dict):
        blockers.append(f"Missing runner parameter evidence for: {image_label}")
    else:
        for key in ["mu", "rof_n_iter", "trof_max_iter"]:
            value = parameters.get(key)
            if key == "mu":
                valid = _is_finite_number(value) and float(value) > 0
            else:
                valid = _is_positive_int(value)
            if not valid:
                blockers.append(f"Missing runner parameter evidence for: {image_label}: {key}")
    return blockers


def _baseline_evidence_blockers(item):
    blockers = []
    image_label = item.get("image_path", "<unknown image>")
    baselines = item.get("baselines", {}) if isinstance(item.get("baselines"), dict) else {}
    raw_kmeans = baselines.get("raw_kmeans", {}) if isinstance(baselines.get("raw_kmeans"), dict) else {}
    multi_otsu = baselines.get("multi_otsu", {}) if isinstance(baselines.get("multi_otsu"), dict) else {}
    if raw_kmeans.get("method") != EXPECTED_RAW_KMEANS_METHOD:
        blockers.append(f"Missing raw_kmeans method evidence for: {image_label}")
    if multi_otsu.get("method") not in ALLOWED_MULTI_OTSU_METHODS:
        blockers.append(f"Missing multi_otsu method evidence for: {image_label}")
    if not _numeric_sequence(multi_otsu.get("thresholds")):
        blockers.append(f"Missing multi_otsu threshold evidence for: {image_label}")
    return blockers


def _is_canonical_data_root(root):
    if root is None:
        return False
    try:
        return Path(root).resolve() == DATA_ROOT.resolve()
    except Exception:
        return False


def _gate_checklist(reasons):
    groups = [
        (
            "canonical_data_root",
            "canonical local data root",
            ["Paper-like promotion requires canonical data root"],
        ),
        (
            "readiness_clean",
            "readiness, mask, source, and license blockers are clear",
            [
                "No local images found",
                "Missing image data",
                "No family has masks",
                "Missing masks/labels",
                "Readiness status",
                "Local dataset manifest",
                "Stem-only mask match",
                "Ambiguous masks",
            ],
        ),
        (
            "runner_outputs",
            "local runner completed quantitative outputs for every family",
            [
                "Local runner has not produced image outputs",
                "Local runner has no completed",
                "No completed local runner output",
                "No completed quantitative local runner output",
                "image(s) failed during local runner execution",
            ],
        ),
        (
            "source_audit",
            "structured source audit artifacts, provenance, and license snapshots are reviewed",
            [
                "Local dataset manifest missing source_audit",
                "Local dataset manifest source_audit",
                "Source claim missing source_audit",
                "Source claim source_audit",
                "cartoon source_audit",
                "texture source_audit",
                "medical source_audit",
            ],
        ),
        (
            "output_evidence",
            "completed outputs include baselines, figures, file hashes, dataset fingerprint, and file-level source claims",
            [
                "Missing raw_kmeans",
                "Missing multi_otsu",
                "Missing figure_path",
                "Missing figure file evidence",
                "Missing figure evidence sidecar",
                "Figure evidence sidecar path does not match figure path",
                "Figure evidence sidecar does not match disk",
                "Figure evidence sidecar does not match report",
                "Missing ROF/T-ROF",
                "Dataset fingerprint does not match image/mask evidence",
                "Missing image file evidence",
                "Missing mask file evidence",
                "Local image file evidence does not match disk",
                "Local mask file evidence does not match disk",
                "Local figure file evidence does not match disk",
                "Local image path is outside canonical family images directory",
                "Local mask path is outside canonical family masks directory",
                "Local figure path is outside allowed figure directory",
                "Local image path has unsupported extension",
                "Local mask path has unsupported extension",
                "Local figure file is not a PNG",
                "Local mask relative path does not match local image path",
                "Missing T-ROF",
                "Missing raw_kmeans clustering_accuracy baseline metric",
                "Missing multi_otsu clustering_accuracy baseline metric",
                "Missing solver evidence",
                "Missing n_classes evidence",
                "Missing threshold evidence",
                "Threshold evidence does not match n_classes",
                "Missing threshold iteration evidence",
                "Missing ROF iteration evidence",
                "Missing ROF residual evidence",
                "Missing runner parameter evidence",
                "Missing raw_kmeans method evidence",
                "Missing multi_otsu method evidence",
                "Missing multi_otsu threshold evidence",
                "Missing non-empty dataset fingerprint",
                "Missing reviewed source claim",
                "Missing file-level reviewed source claim",
                "Missing source_id source claim",
                "Missing license_reviewed=true source claim",
                "Missing citation source claim",
                "Missing license_note source claim",
                "Missing provenance_reviewed=true source claim",
                "Missing provenance_note source claim",
                "Source claim must explicitly set synthetic_fixture=false",
                "Source claim contains fixture/tempfile text",
                "Source claim sha256 does not match image file evidence",
                "Source claim mask_sha256 does not match mask file evidence",
                "Source claim image does not match local image path",
                "Source claim mask does not match local mask path",
                "Source claim source_id is not in source registry",
                "Missing required figure evidence panels",
            ],
        ),
    ]

    checklist = []
    assigned = set()
    for requirement_id, description, prefixes in groups:
        requirement_reasons = [
            reason
            for reason in reasons
            if any(reason.startswith(prefix) for prefix in prefixes)
        ]
        assigned.update(requirement_reasons)
        checklist.append(
            {
                "id": requirement_id,
                "description": description,
                "passed": not requirement_reasons,
                "reasons": requirement_reasons,
            }
        )

    unassigned = [reason for reason in reasons if reason not in assigned]
    if unassigned:
        checklist.append(
            {
                "id": "other",
                "description": "other paper-like gate blockers",
                "passed": False,
                "reasons": unassigned,
            }
        )
    return checklist


def _paper_like_evidence_summary(image_results, dataset_fingerprint, data_root):
    image_results = [item for item in (image_results or []) if isinstance(item, dict)]
    completed_images = [item for item in image_results if item.get("status") == "completed"]
    quantitative_images = [item for item in completed_images if _is_quantitative_image_result(item)]
    source_claim_count = sum(
        1
        for item in completed_images
        if isinstance(item.get("source_claim"), dict)
        and item["source_claim"].get("manifest_status") == "present"
        and item["source_claim"].get("claim_scope") == "file"
    )
    figure_evidence_count = sum(
        1
        for item in completed_images
        if isinstance(item.get("figure_evidence"), dict)
        and isinstance(item.get("figure_evidence_file"), dict)
        and item.get("figure_evidence_path")
    )
    return {
        "schema_version": 1,
        "gate_id": PAPER_LIKE_GATE_ID,
        "canonical_data_root": _display_path(DATA_ROOT),
        "data_root": _display_path(data_root) if data_root else "",
        "dataset_fingerprint": dataset_fingerprint or {},
        "image_count": len(image_results),
        "completed_image_count": len(completed_images),
        "quantitative_image_count": len(quantitative_images),
        "required_families": list(DATA_FAMILIES.keys()),
        "completed_families": sorted({item.get("family") for item in completed_images if item.get("family")}),
        "quantitative_families": sorted({item.get("family") for item in quantitative_images if item.get("family")}),
        "source_claim_count": source_claim_count,
        "figure_evidence_count": figure_evidence_count,
    }


def _paper_like_gate(
    readiness_status,
    blockers,
    claim_blockers,
    image_results=None,
    data_root=None,
    dataset_fingerprint=None,
    source_manifest=None,
    figure_dir=None,
):
    source_manifest = source_manifest or load_source_manifest()
    reasons = []
    for blocker in _source_manifest_schema_blockers(source_manifest):
        _append_unique(reasons, blocker)
    for blocker in blockers:
        _append_unique(reasons, blocker)
    for blocker in claim_blockers:
        _append_unique(reasons, blocker)

    if not _is_canonical_data_root(data_root):
        _append_unique(
            reasons,
            f"Paper-like promotion requires canonical data root: {_display_path(DATA_ROOT)}",
        )

    if readiness_status != "ready_for_paper_like_runner":
        _append_unique(reasons, f"Readiness status is {readiness_status}, not ready_for_paper_like_runner")

    dataset_fingerprint = dataset_fingerprint or {}
    if (
        dataset_fingerprint.get("algorithm") != "sha256"
        or dataset_fingerprint.get("file_count", 0) <= 0
        or len(dataset_fingerprint.get("sha256", "")) != 64
    ):
        _append_unique(reasons, "Missing non-empty dataset fingerprint")

    if image_results is None:
        _append_unique(reasons, "Local runner has not produced image outputs")
    else:
        completed_images = [item for item in image_results if item.get("status") == "completed"]
        failed_images = [item for item in image_results if item.get("status") == "failed"]
        quantitative_images = [item for item in completed_images if _is_quantitative_image_result(item)]

        if failed_images:
            _append_unique(reasons, f"{len(failed_images)} image(s) failed during local runner execution")
        if not completed_images:
            _append_unique(reasons, "Local runner has no completed image outputs")
        if not quantitative_images:
            _append_unique(reasons, "Local runner has no completed quantitative image outputs")
        if (
            dataset_fingerprint.get("algorithm") == "sha256"
            and dataset_fingerprint.get("file_count", 0) > 0
            and len(dataset_fingerprint.get("sha256", "")) == 64
            and not _fingerprint_matches_image_results(dataset_fingerprint, image_results)
        ):
            _append_unique(reasons, "Dataset fingerprint does not match image/mask evidence")

        completed_families = {item.get("family") for item in completed_images}
        quantitative_families = {item.get("family") for item in quantitative_images}
        for family in DATA_FAMILIES:
            if family not in completed_families:
                _append_unique(reasons, f"No completed local runner output for family: {family}")
            if family not in quantitative_families:
                _append_unique(reasons, f"No completed quantitative local runner output for family: {family}")

        for item in completed_images:
            image_label = item.get("image_path", "<unknown image>")
            for blocker in _completed_output_shape_blockers(
                item,
                "Local",
                "Source claim",
                source_manifest=source_manifest,
                figure_dir=figure_dir,
                audit_roots=_source_audit_roots_for_family(data_root, item.get("family")),
            ):
                _append_unique(reasons, blocker)
            baselines = item.get("baselines", {})
            declares_quantitative = _declares_quantitative_image_result(item)
            quantitative = _is_quantitative_image_result(item)
            if declares_quantitative and not item.get("mask_path"):
                _append_unique(reasons, f"Missing mask_path for quantitative evidence for: {image_label}")
            if quantitative and not _has_numeric_metric(item.get("metrics", {}), "clustering_accuracy"):
                _append_unique(reasons, f"Missing T-ROF clustering_accuracy metric for: {image_label}")
            if "raw_kmeans" not in baselines:
                _append_unique(reasons, f"Missing raw_kmeans baseline for: {image_label}")
            elif quantitative and not _has_numeric_metric(
                baselines.get("raw_kmeans", {}).get("metrics", {}),
                "clustering_accuracy",
            ):
                _append_unique(reasons, f"Missing raw_kmeans clustering_accuracy baseline metric for: {image_label}")
            if "multi_otsu" not in baselines:
                _append_unique(reasons, f"Missing multi_otsu baseline for: {image_label}")
            elif quantitative and not _has_numeric_metric(
                baselines.get("multi_otsu", {}).get("metrics", {}),
                "clustering_accuracy",
            ):
                _append_unique(reasons, f"Missing multi_otsu clustering_accuracy baseline metric for: {image_label}")
            if quantitative:
                for blocker in _runner_evidence_blockers(item):
                    _append_unique(reasons, blocker)
                for blocker in _baseline_evidence_blockers(item):
                    _append_unique(reasons, blocker)
            if not item.get("figure_path"):
                _append_unique(reasons, f"Missing figure_path for: {image_label}")
            if (
                len(item.get("figure_file", {}).get("sha256", "")) != 64
                or item.get("figure_file", {}).get("size_bytes", 0) <= 0
            ):
                _append_unique(reasons, f"Missing figure file evidence for: {image_label}")
            for blocker in _figure_evidence_blockers(item, ""):
                _append_unique(reasons, blocker)
            if "ROF" not in item.get("figure_panels", []) or "T-ROF" not in item.get("figure_panels", []):
                _append_unique(reasons, f"Missing ROF/T-ROF figure panels for: {image_label}")
            if _missing_figure_panel_requirements(item.get("figure_panels", [])):
                _append_unique(reasons, f"Missing required figure evidence panels for: {image_label}")
            if len(item.get("image_file", {}).get("sha256", "")) != 64:
                _append_unique(reasons, f"Missing image file evidence for: {image_label}")
            elif not _file_evidence_matches_disk(item.get("image_path"), item.get("image_file")):
                _append_unique(reasons, f"Local image file evidence does not match disk for: {image_label}")
            if item.get("mask_path") and len(item.get("mask_file", {}).get("sha256", "")) != 64:
                _append_unique(reasons, f"Missing mask file evidence for: {image_label}")
            elif item.get("mask_path") and not _file_evidence_matches_disk(item.get("mask_path"), item.get("mask_file")):
                _append_unique(reasons, f"Local mask file evidence does not match disk for: {image_label}")
            if item.get("figure_path") and not _file_evidence_matches_disk(item.get("figure_path"), item.get("figure_file")):
                _append_unique(reasons, f"Local figure file evidence does not match disk for: {image_label}")
            source_claim = item.get("source_claim", {})
            if not isinstance(source_claim, dict):
                source_claim = {}
            if source_claim.get("manifest_status") != "present":
                _append_unique(reasons, f"Missing reviewed source claim for: {image_label}")
            if source_claim.get("claim_scope") != "file":
                _append_unique(reasons, f"Missing file-level reviewed source claim for: {image_label}")
            if str(source_claim.get("image", "")).strip() != _image_result_relative_path(item):
                _append_unique(reasons, f"Source claim image does not match local image path for: {image_label}")
            if item.get("mask_path") and str(source_claim.get("mask", "")).strip() != _mask_result_relative_path(item):
                _append_unique(reasons, f"Source claim mask does not match local mask path for: {image_label}")
            if not str(source_claim.get("source_id", "")).strip():
                _append_unique(reasons, f"Missing source_id source claim for: {image_label}")
            if source_claim.get("license_reviewed") is not True:
                _append_unique(reasons, f"Missing license_reviewed=true source claim for: {image_label}")
            if not str(source_claim.get("citation", "")).strip():
                _append_unique(reasons, f"Missing citation source claim for: {image_label}")
            if not str(source_claim.get("license_note", "")).strip():
                _append_unique(reasons, f"Missing license_note source claim for: {image_label}")
            if source_claim.get("provenance_reviewed") is not True:
                _append_unique(reasons, f"Missing provenance_reviewed=true source claim for: {image_label}")
            if not str(source_claim.get("provenance_note", "")).strip():
                _append_unique(reasons, f"Missing provenance_note source claim for: {image_label}")
            if source_claim.get("synthetic_fixture") is not False:
                _append_unique(reasons, f"Source claim must explicitly set synthetic_fixture=false for: {image_label}")
            for blocker in _fixture_text_claim_blockers(source_claim, "Source claim", f"for: {image_label}"):
                _append_unique(reasons, blocker)
            for blocker in _placeholder_text_claim_blockers(source_claim, "Source claim", f"for: {image_label}"):
                _append_unique(reasons, blocker)
            for blocker in _source_audit_blockers(
                source_claim.get("source_audit"),
                "Source claim",
                f"for: {image_label}",
                _source_audit_roots_for_family(data_root, item.get("family")),
            ):
                _append_unique(reasons, blocker)
            for blocker in _source_audit_url_blockers(
                source_claim.get("source_audit"),
                item.get("family"),
                source_claim.get("source_id"),
                source_manifest,
                "Source claim",
                f"for: {image_label}",
            ):
                _append_unique(reasons, blocker)
            if source_claim.get("sha256") != item.get("image_file", {}).get("sha256"):
                _append_unique(reasons, f"Source claim sha256 does not match image file evidence for: {image_label}")
            if item.get("mask_path") and source_claim.get("mask_sha256") != item.get("mask_file", {}).get("sha256"):
                _append_unique(reasons, f"Source claim mask_sha256 does not match mask file evidence for: {image_label}")

    return {
        "passed": not reasons,
        "dashboard_level": "paper-like" if not reasons else "partial",
        "checked_requirements": [
            "all data families have completed quantitative local runner outputs",
            "no readiness, runner, source, or license-review blockers remain",
            "completed outputs include baselines, generated figure file evidence, file-level source claims, dataset fingerprint, and input/mask file evidence",
        ],
        "evidence_summary": _paper_like_evidence_summary(image_results, dataset_fingerprint, data_root),
        "checklist": _gate_checklist(reasons),
        "reasons": reasons,
    }


def _has_numeric_metric(metrics, metric_name):
    value = metrics.get(metric_name) if isinstance(metrics, dict) else None
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _missing_figure_panel_requirements(panels):
    panel_set = set(panels or [])
    missing = []
    for required in ["input", "ROF", "T-ROF", "raw K-means"]:
        if required not in panel_set:
            missing.append(required)
    if not ({"multi-Otsu", "quantile fallback"} & panel_set):
        missing.append("multi-Otsu/quantile fallback")
    if not ({"T-ROF error", "T-ROF vs raw"} & panel_set):
        missing.append("T-ROF difference")
    if not ({"T-ROF vs Otsu", "T-ROF vs quantile"} & panel_set):
        missing.append("Otsu/quantile difference")
    return missing


def _mean_metric_dict(metric_rows):
    import numpy as np

    keys = sorted({key for row in metric_rows for key in row})
    return {
        key: round(float(np.mean([row[key] for row in metric_rows if key in row])), 4)
        for key in keys
    }


def _family_result_status(readiness_status, completed_images, failed_images, quantitative_images):
    if not completed_images and not failed_images:
        return readiness_status
    if failed_images and not completed_images:
        return "failed"
    if failed_images and quantitative_images:
        return "completed_quantitative_with_failures"
    if failed_images:
        return "completed_qualitative_with_failures"
    if quantitative_images:
        return "completed_quantitative"
    return "completed_qualitative_only"


def _family_summaries(families, image_results=None):
    image_results = image_results or []
    summaries = []
    for family_info in families:
        family = family_info["family"]
        family_results = [item for item in image_results if item.get("family") == family]
        completed_images = [item for item in family_results if item.get("status") == "completed"]
        failed_images = [item for item in family_results if item.get("status") == "failed"]
        quantitative_images = [item for item in completed_images if _is_quantitative_image_result(item)]
        metric_rows = [item.get("metrics", {}) for item in quantitative_images if item.get("metrics")]

        baseline_metrics = {}
        baseline_names = sorted(
            {
                baseline_name
                for item in quantitative_images
                for baseline_name in item.get("baselines", {})
            }
        )
        for baseline_name in baseline_names:
            rows = [
                item.get("baselines", {}).get(baseline_name, {}).get("metrics", {})
                for item in quantitative_images
                if item.get("baselines", {}).get(baseline_name, {}).get("metrics")
            ]
            baseline_metrics[baseline_name] = _mean_metric_dict(rows) if rows else {}

        summaries.append(
            {
                "family": family,
                "status": _family_result_status(
                    family_info["status"],
                    completed_images,
                    failed_images,
                    quantitative_images,
                ),
                "image_count": family_info["image_count"],
                "mask_count": family_info["mask_count"],
                "matched_mask_count": family_info.get("matched_mask_count", 0),
                "completed_image_count": len(completed_images),
                "failed_image_count": len(failed_images),
                "quantitative_image_count": len(quantitative_images),
                "qualitative_image_count": len(
                    [item for item in completed_images if not _is_quantitative_image_result(item)]
                ),
                "metrics_mean": _mean_metric_dict(metric_rows) if metric_rows else {},
                "baseline_metrics_mean": baseline_metrics,
                "figure_paths": [item["figure_path"] for item in completed_images if item.get("figure_path")],
                "source_claims": [
                    item["source_claim"]
                    for item in completed_images
                    if item.get("source_claim")
                ],
                "errors": [item["error"] for item in failed_images if item.get("error")],
            }
        )
    return summaries


def _run_protocol(mu, default_classes, rof_n_iter, trof_max_iter, figure_dir):
    return {
        "protocol_id": "iterated_rof_trof_local_data_v1",
        "schema_version": 1,
        "paper_id": "iterated-rof",
        "algorithm": "ROF denoising followed by iterative multiclass thresholding",
        "solver": "sat_rof_trof.rof_chambolle_pock + sat_rof_trof.run_trof_thresholds",
        "threshold_update": "tau_i = 0.5 * (mean_f(Omega_{i-1}) + mean_f(Omega_i))",
        "threshold_mean_source": "raw input image f, not denoised ROF output u",
        "baselines": [
            "raw_kmeans",
            "multi_otsu",
        ],
        "metrics": [
            "clustering_accuracy",
            "binary_dice_when_mask_has_two_labels",
        ],
        "parameters": {
            "mu": mu,
            "default_classes": default_classes,
            "rof_n_iter": rof_n_iter,
            "trof_max_iter": trof_max_iter,
            "rof_tol": 2e-5,
            "trof_tol": 1e-4,
            "projection_bins": 4096,
            "seed": SEED,
        },
        "data_layout": "reproduce/data/iterated_rof/{cartoon,texture,medical}/{images,masks}",
        "figure_dir": _display_path(figure_dir),
        "promotion_rule": "dashboard paper-like promotion requires paper_like_gate.passed=true",
    }


def build_data_gap_checklist(report, source_manifest=None):
    source_manifest = source_manifest or load_source_manifest()
    recommended_sources = report.get("recommended_sources") or {
        family: [
            _source_summary(source)
            for source in sorted(source_manifest.get(family, []), key=lambda item: item["priority"])
        ]
        for family in DATA_FAMILIES
    }
    family_items = []
    source_audit_status_counts = {}
    remaining_family_count = 0
    data_root = _resolve_report_path(report.get("data_root")) or DATA_ROOT
    manifest = load_local_dataset_manifest(data_root)
    families = [scan_family(data_root, family) for family in DATA_FAMILIES]
    for family_info in families:
        family = family_info["family"]
        missing = []
        next_actions = []
        if family_info.get("image_count", 0) <= 0:
            missing.append("add at least one nontrivial local image")
            next_actions.append(f"place images under {family}/images/")
        if family_info.get("matched_mask_count", 0) <= 0:
            missing.append("add matching mask/label image")
            next_actions.append(f"place same-relative-path masks under {family}/masks/")
        if family_info.get("status") != "ready_quantitative":
            remaining_family_count += 1
        next_actions.append("refresh files[] SHA-256 claims after files are placed")
        next_actions.append("refresh source_audit artifact SHA-256 claims after audit files are placed")
        sources = recommended_sources.get(family, [])
        content_issues = _preflight_content_issues(family_info)
        source_audit_status = _source_audit_gap_status_for_family(family, manifest, data_root)
        source_audit_status_counts[source_audit_status["status"]] = (
            source_audit_status_counts.get(source_audit_status["status"], 0) + 1
        )
        family_items.append(
            {
                "family": family,
                "status": family_info.get("status"),
                "image_count": family_info.get("image_count", 0),
                "matched_mask_count": family_info.get("matched_mask_count", 0),
                "content_issue_count": len(content_issues),
                "content_issues": content_issues,
                "paths": {
                    "images": f"{family}/images",
                    "masks": f"{family}/masks",
                },
                "primary_source": sources[0] if sources else {},
                "source_candidates": sources,
                "acquisition_plan": _source_acquisition_plan(family, sources),
                "source_audit": source_audit_status,
                "missing": missing,
                "next_actions": next_actions,
            }
        )

    claim_blockers = _claim_blockers(families, manifest, source_manifest)
    global_next_actions = []
    if manifest.get("status") != "present":
        global_next_actions.append(
            f"copy {DATASET_MANIFEST_TEMPLATE_PATH.name} to {LOCAL_DATASET_MANIFEST_NAME} and fill reviewed source claims"
        )
    if any(item["missing"] for item in family_items):
        global_next_actions.append("populate all three family images/ and masks/ directories with real local data")
    global_next_actions.append("run --refresh-manifest-file-claims after placing local files")
    global_next_actions.append("run --refresh-source-audit-artifact-claims after placing audit files")
    global_next_actions.append("run --check-source-audit-artifact-claims before strict-data-ready")
    if claim_blockers:
        global_next_actions.append("fix dataset_manifest.json claim blockers before promotion")
    global_next_actions.append("run --run plus --strict-paper-like only after the gap checklist is clean")

    gate = _paper_like_gate_from_report(report)
    data_ready_blockers = []
    readiness_status = report.get("readiness_status", report.get("status"))
    if readiness_status != "ready_for_paper_like_runner":
        data_ready_blockers.append(
            f"readiness status is {readiness_status}, not ready_for_paper_like_runner"
        )
    if report.get("blockers"):
        data_ready_blockers.extend(report.get("blockers", []))
    if manifest.get("status") != "present":
        data_ready_blockers.append("local dataset manifest is not present")
    if claim_blockers:
        data_ready_blockers.extend(claim_blockers)
    for item in family_items:
        if item["missing"]:
            data_ready_blockers.append(
                f"{item['family']} is missing: {', '.join(item['missing'])}"
            )
        for issue in item["content_issues"]:
            data_ready_blockers.append(f"{item['family']} content issue: {issue}")
        data_ready_blockers.extend(item["source_audit"].get("issues", []))
    return {
        "schema_version": 1,
        "paper_id": report.get("paper_id", "iterated-rof"),
        "target_level": report.get("target_level", "paper-like"),
        "ready_for_local_runner": not data_ready_blockers,
        "ready_for_paper_like_runner_outputs": gate.get("passed") is True,
        "ready_for_paper_like": False,
        "ready_for_dashboard_promotion": False,
        "promotion_readiness_note": (
            "Data-gap reports only describe local data and runner-output readiness; "
            "dashboard promotion requires a promotable dashboard candidate or promotion audit "
            "with a bound source summary artifact."
        ),
        "current_status": report.get("status"),
        "data_root": report.get("data_root", _display_path(DATA_ROOT)),
        "remaining_family_count": remaining_family_count,
        "source_audit_status_counts": source_audit_status_counts,
        "families": family_items,
        "manifest": {
            "status": manifest.get("status", "missing"),
            "path": manifest.get("path", _display_path(Path(report.get("data_root", DATA_ROOT)) / LOCAL_DATASET_MANIFEST_NAME)),
            "claim_blocker_count": len(claim_blockers),
            "claim_blockers": claim_blockers,
        },
        "data_ready_blocker_count": len(data_ready_blockers),
        "data_ready_blockers": data_ready_blockers,
        "paper_like_gate": {
            "passed": gate.get("passed") is True,
            "reason_count": len(gate.get("reasons", [])),
            "reasons": gate.get("reasons", []),
        },
        "global_next_actions": global_next_actions,
    }


def _manifest_manual_field_report(family, family_claim):
    missing_manual_fields = []
    files = []
    if not isinstance(family_claim, dict):
        return {
            "status": "missing_family_claim",
            "ready": False,
            "missing_manual_fields": ["family claim"],
            "issue_fields": ["family claim"],
            "file_count": 0,
            "files": [],
        }

    text_fields = [
        "source_id",
        "source_name",
        "license_note",
        "citation",
        "provenance_note",
    ]
    for field in text_fields:
        if not str(family_claim.get(field, "")).strip():
            missing_manual_fields.append(field)
    if family_claim.get("license_reviewed") is not True:
        missing_manual_fields.append("license_reviewed")
    if family_claim.get("provenance_reviewed") is not True:
        missing_manual_fields.append("provenance_reviewed")
    if family_claim.get("synthetic_fixture") is not False:
        missing_manual_fields.append("synthetic_fixture")

    audit_missing_fields = _source_audit_missing_manual_fields(family_claim.get("source_audit"))
    if audit_missing_fields == ["source_audit"]:
        missing_manual_fields.append("source_audit")
    else:
        missing_manual_fields.extend(f"source_audit.{field}" for field in audit_missing_fields)

    file_claims = family_claim.get("files")
    if not isinstance(file_claims, list):
        missing_manual_fields.append("files")
        file_claims = []
    for file_claim in file_claims:
        if not isinstance(file_claim, dict):
            missing_manual_fields.append("files[].file claim object")
            files.append(
                {
                    "image": "",
                    "ready": False,
                    "missing_fields": ["file claim object"],
                }
            )
            continue
        file_missing = []
        for field in ["image", "sha256"]:
            if not str(file_claim.get(field, "")).strip():
                file_missing.append(field)
        if file_claim.get("mask") and not str(file_claim.get("mask_sha256", "")).strip():
            file_missing.append("mask_sha256")
        if "source_audit" in file_claim:
            file_missing.extend(
                f"source_audit.{field}"
                for field in _source_audit_missing_manual_fields(file_claim.get("source_audit"))
            )
        files.append(
            {
                "image": str(file_claim.get("image", "")),
                "mask": str(file_claim.get("mask", "") or ""),
                "ready": not file_missing,
                "missing_fields": file_missing,
            }
        )
        missing_manual_fields.extend(f"files[].{field}" for field in file_missing)

    issue_fields = sorted(set(missing_manual_fields))
    return {
        "status": "complete" if not issue_fields else "incomplete",
        "ready": not issue_fields,
        "missing_manual_fields": issue_fields,
        "issue_fields": issue_fields,
        "file_count": len(file_claims),
        "files": files,
    }


def _source_audit_missing_manual_fields(audit):
    if not isinstance(audit, dict):
        return ["source_audit"]
    missing = []
    for field in SOURCE_AUDIT_TEXT_FIELDS:
        if not str(audit.get(field, "")).strip():
            missing.append(field)
        elif field == "downloaded_at" and not _is_iso_date(audit.get(field)):
            missing.append(field)
    for field in SOURCE_AUDIT_SHA_FIELDS:
        if not _is_sha256(audit.get(field)):
            missing.append(field)
    if audit.get("local_file_mapping_reviewed") is not True:
        missing.append("local_file_mapping_reviewed")
    return missing


def build_data_package_review(root=DATA_ROOT, source_manifest=None):
    root = Path(root)
    source_manifest = source_manifest or load_source_manifest()
    readiness = build_readiness_report(root, source_manifest=source_manifest)
    gap = readiness["data_gap_checklist"]
    manifest = load_local_dataset_manifest(root)
    manifest_file_claim_check = check_manifest_file_claims(root)
    source_audit_artifact_claim_check = check_source_audit_artifact_claims(root)
    families = []
    for gap_family in gap.get("families", []):
        family = gap_family["family"]
        family_claim = manifest.get("families", {}).get(family)
        manifest_review = _manifest_manual_field_report(family, family_claim)
        families.append(
            {
                "family": family,
                "status": gap_family.get("status"),
                "image_count": gap_family.get("image_count", 0),
                "matched_mask_count": gap_family.get("matched_mask_count", 0),
                "missing": gap_family.get("missing", []),
                "content_issue_count": gap_family.get("content_issue_count", 0),
                "content_issues": gap_family.get("content_issues", []),
                "manifest_review": manifest_review,
                "source_audit": gap_family.get("source_audit", {}),
                "next_actions": gap_family.get("next_actions", []),
            }
        )
    incomplete_family_count = sum(
        1
        for family in families
        if family["missing"]
        or family["content_issues"]
        or family["manifest_review"]["ready"] is not True
        or family.get("source_audit", {}).get("ready") is not True
    )
    ready_for_local_runner = (
        gap.get("ready_for_local_runner") is True
        and manifest_file_claim_check.get("status") == "current"
        and source_audit_artifact_claim_check.get("status") == "current"
        and incomplete_family_count == 0
    )
    return {
        "schema_version": 1,
        "paper_id": readiness.get("paper_id", "iterated-rof"),
        "target_level": readiness.get("target_level", "paper-like"),
        "review_root": _display_path(root),
        "status": "ready_for_local_runner" if ready_for_local_runner else "incomplete",
        "downloaded_data": False,
        "ready_for_local_runner": ready_for_local_runner,
        "ready_for_paper_like": False,
        "ready_for_dashboard_promotion": False,
        "manifest": gap.get("manifest", {}),
        "manifest_file_claim_check": manifest_file_claim_check,
        "source_audit_artifact_claim_check": source_audit_artifact_claim_check,
        "incomplete_family_count": incomplete_family_count,
        "families": families,
        "data_ready_blocker_count": gap.get("data_ready_blocker_count", 0),
        "data_ready_blockers": gap.get("data_ready_blockers", []),
        "global_next_actions": gap.get("global_next_actions", []),
        "promotion_readiness_note": gap.get("promotion_readiness_note"),
    }


def write_data_package_review(root, path, source_manifest=None):
    review = build_data_package_review(root, source_manifest=source_manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return review


def _with_data_gap_checklist(report, source_manifest=None):
    data_gap_checklist = build_data_gap_checklist(report, source_manifest=source_manifest)
    report["data_gap_checklist"] = data_gap_checklist
    report["data_ready_status"] = (
        "ready_for_local_runner"
        if data_gap_checklist.get("ready_for_local_runner") is True
        else "blocked_data_ready"
    )
    report["data_ready_blocker_count"] = data_gap_checklist.get("data_ready_blocker_count", 0)
    return report


def build_readiness_report(root=DATA_ROOT, source_manifest=None):
    source_manifest = source_manifest or load_source_manifest()
    local_manifest = load_local_dataset_manifest(root)
    families = [scan_family(root, family) for family in DATA_FAMILIES]
    entries = scan_dataset(root)
    dataset_fingerprint = _dataset_fingerprint(entries)
    missing = [item["family"] for item in families if item["status"] == "missing"]
    quantitative = [item["family"] for item in families if item["status"] == "ready_quantitative"]
    missing_quantitative = [
        item["family"]
        for item in families
        if item["status"] != "missing" and item["status"] != "ready_quantitative"
    ]
    recommended_sources = {
        family: [_source_summary(source) for source in sorted(source_manifest.get(family, []), key=lambda item: item["priority"])]
        for family in DATA_FAMILIES
    }

    if missing:
        status = "blocked_missing_data"
    elif missing_quantitative:
        status = "blocked_missing_masks"
    elif not quantitative:
        status = "blocked_missing_masks"
    else:
        status = "ready_for_paper_like_runner"

    blockers = []
    if missing:
        blockers.append(f"Missing image data for: {', '.join(missing)}")
    if missing_quantitative:
        blockers.append(
            "Missing masks/labels for quantitative paper-like metrics in: "
            f"{', '.join(missing_quantitative)}"
        )
    if not quantitative:
        blockers.append("No family has masks/labels, so quantitative paper-like metrics are not available")
    claim_blockers = _claim_blockers(families, local_manifest, source_manifest)

    report = {
        "paper_id": "iterated-rof",
        "target_level": "paper-like",
        "current_dashboard_level": "partial",
        "status": status,
        "data_root": _display_path(root),
        "dataset_fingerprint": dataset_fingerprint,
        "families": families,
        "recommended_sources": recommended_sources,
        "local_dataset_manifest": local_manifest,
        "blockers": blockers,
        "claim_blockers": claim_blockers,
        "paper_like_gate": _paper_like_gate(
            status,
            blockers,
            claim_blockers,
            data_root=root,
            dataset_fingerprint=dataset_fingerprint,
            source_manifest=source_manifest,
        ),
        "claim_boundary": (
            "Do not promote dashboard level beyond partial until real/local images, baselines, metrics, "
            "and generated figures exist."
        ),
    }
    return _with_data_gap_checklist(report, source_manifest=source_manifest)


def _as_grayscale(array):
    import numpy as np

    values = np.asarray(array)
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[-1] == 1:
        return values[..., 0]
    if values.ndim == 3 and values.shape[-1] >= 3:
        rgb = values[..., :3]
        return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    raise ValueError(f"Expected a 2D grayscale or RGB image, got shape {values.shape}")


def normalize_grayscale(array):
    import numpy as np

    original = np.asarray(array)
    grayscale = _as_grayscale(original)
    values = np.nan_to_num(grayscale.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    if np.issubdtype(original.dtype, np.integer):
        dtype_max = max(float(np.iinfo(original.dtype).max), 1.0)
        values = values / dtype_max
    elif values.size:
        min_value = float(values.min())
        max_value = float(values.max())
        if 0.0 <= min_value and max_value <= 1.0:
            pass
        elif 0.0 <= min_value and max_value <= 255.0:
            values = values / 255.0
        elif max_value > min_value:
            values = (values - min_value) / (max_value - min_value)
        else:
            values = values * 0.0

    return np.clip(values, 0.0, 1.0)


def _read_image(path):
    from matplotlib import image as mpimg

    return mpimg.imread(path)


def load_grayscale_image(path):
    return normalize_grayscale(_read_image(path))


def _compact_labels(values):
    import numpy as np

    unique_values, inverse = np.unique(values, return_inverse=True)
    return inverse.reshape(values.shape), unique_values


def load_mask(path, expected_shape=None):
    import numpy as np

    raw = np.asarray(_read_image(path))
    if raw.ndim == 3 and raw.shape[-1] >= 3:
        rgb = raw[..., :3]
        if np.issubdtype(rgb.dtype, np.floating) and rgb.size and float(rgb.max()) <= 1.0:
            rgb = np.rint(rgb * 255.0)
        else:
            rgb = np.rint(rgb)
        _, inverse = np.unique(rgb.astype(int).reshape(-1, rgb.shape[-1]), axis=0, return_inverse=True)
        labels = inverse.reshape(raw.shape[:2])
    else:
        grayscale = _as_grayscale(raw)
        if np.issubdtype(grayscale.dtype, np.floating) and grayscale.size and float(grayscale.max()) <= 1.0:
            quantized = np.rint(grayscale * 255.0)
        else:
            quantized = np.rint(grayscale)
        labels, _ = _compact_labels(quantized.astype(int))

    if expected_shape is not None and tuple(labels.shape) != tuple(expected_shape):
        raise ValueError(f"Mask shape {labels.shape} does not match image shape {expected_shape}: {path}")
    return labels.astype(int)


def _infer_class_count(mask, default_classes):
    import numpy as np

    if mask is None:
        return max(2, int(default_classes))
    return max(2, int(np.unique(mask).size))


def _binary_dice_against_best_label(mask, labels):
    import numpy as np

    truth_values = np.unique(mask)
    if truth_values.size != 2:
        return None
    truth_foreground = mask != truth_values[0]
    return max(dice_score(truth_foreground, labels == value) for value in np.unique(labels))


def _metrics_against_mask(mask, labels):
    metrics = {
        "clustering_accuracy": round(float(clustering_accuracy(mask, labels)), 4)
    }
    binary_dice = _binary_dice_against_best_label(mask, labels)
    if binary_dice is not None:
        metrics["dice"] = round(float(binary_dice), 4)
    return metrics


def _threshold_labels(image, thresholds):
    import numpy as np

    return np.digitize(image, np.asarray(thresholds, dtype=float))


def _multi_otsu_thresholds(image, n_classes):
    import numpy as np

    if n_classes <= 1:
        return np.asarray([], dtype=float), "none"
    unique_values = np.unique(image)
    if unique_values.size < n_classes:
        return np.quantile(image, np.linspace(0, 1, n_classes + 1)[1:-1]), "quantile_fallback"
    try:
        from skimage.filters import threshold_multiotsu

        return np.asarray(threshold_multiotsu(image, classes=n_classes), dtype=float), "skimage_multiotsu"
    except Exception:
        return np.quantile(image, np.linspace(0, 1, n_classes + 1)[1:-1]), "quantile_fallback"


def _run_baselines(image, n_classes, mask):
    raw_kmeans_labels = simple_kmeans(image.reshape(-1, 1), n_classes, seed=SEED).reshape(image.shape)
    otsu_thresholds, otsu_method = _multi_otsu_thresholds(image, n_classes)
    otsu_labels = _threshold_labels(image, otsu_thresholds)
    baselines = {
        "raw_kmeans": {
            "method": "simple_kmeans_on_raw_grayscale",
            "labels": raw_kmeans_labels,
            "metrics": _metrics_against_mask(mask, raw_kmeans_labels) if mask is not None else {},
        },
        "multi_otsu": {
            "method": otsu_method,
            "thresholds": [round(float(value), 6) for value in otsu_thresholds],
            "labels": otsu_labels,
            "metrics": _metrics_against_mask(mask, otsu_labels) if mask is not None else {},
        },
    }
    return baselines


def _json_baselines(baselines):
    return {
        name: {key: value for key, value in baseline.items() if key != "labels"}
        for name, baseline in baselines.items()
    }


def _safe_filename_part(value):
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in str(value)).strip("_") or "image"


def _short_hash(value):
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:10]


def _label_disagreement(reference, labels):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    reference = np.asarray(reference)
    labels = np.asarray(labels)
    reference_values = np.unique(reference)
    label_values = np.unique(labels)
    matrix = np.zeros((len(reference_values), len(label_values)), dtype=int)
    for i, reference_value in enumerate(reference_values):
        for j, label_value in enumerate(label_values):
            matrix[i, j] = np.sum((reference == reference_value) & (labels == label_value))
    row, col = linear_sum_assignment(-matrix)
    aligned = np.full(labels.shape, fill_value=-1, dtype=int)
    for reference_index, label_index in zip(row, col):
        aligned[labels == label_values[label_index]] = int(reference_values[reference_index])
    return aligned != reference


def _save_entry_figure(entry, image, mask, rof, trof_labels, baselines, figure_dir):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    image_key = entry.get("image_relative_path") or Path(entry["image_path"]).name
    otsu_title = "multi-Otsu" if baselines["multi_otsu"]["method"] == "skimage_multiotsu" else "quantile fallback"
    otsu_diff_title = "T-ROF vs Otsu" if baselines["multi_otsu"]["method"] == "skimage_multiotsu" else "T-ROF vs quantile"
    trof_error_title = "T-ROF error" if mask is not None else "T-ROF vs raw"
    trof_error_reference = mask if mask is not None else baselines["raw_kmeans"]["labels"]
    filename = (
        f"{_safe_filename_part(entry['family'])}_"
        f"{_safe_filename_part(image_key)}_{_short_hash(image_key)}_iterated_rof.png"
    )
    output = figure_dir / filename
    panels = [
        (image, "input", "gray"),
        (mask if mask is not None else None, "mask" if mask is not None else "no mask", "viridis"),
        (rof, "ROF", "gray"),
        (trof_labels, "T-ROF", "viridis"),
        (baselines["raw_kmeans"]["labels"], "raw K-means", "viridis"),
        (baselines["multi_otsu"]["labels"], otsu_title, "viridis"),
        (_label_disagreement(trof_error_reference, trof_labels), trof_error_title, "magma"),
        (_label_disagreement(baselines["multi_otsu"]["labels"], trof_labels), otsu_diff_title, "magma"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 2.6))
    for ax, (values, title, cmap) in zip(axes, panels):
        if values is None:
            ax.text(0.5, 0.5, "qualitative\nonly", ha="center", va="center")
        else:
            ax.imshow(values, cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return _display_path(output), [title for _, title, _ in panels], _file_evidence(output)


def _run_entry(entry, mu, default_classes, rof_n_iter, trof_max_iter, figure_dir, local_manifest):
    import numpy as np

    elapsed = timer()
    image = load_grayscale_image(entry["image_path"])
    mask = load_mask(entry["mask_path"], expected_shape=image.shape) if entry["mask_path"] else None
    n_classes = _infer_class_count(mask, default_classes)

    rof, rof_info = sat_rof_trof.rof_chambolle_pock(
        image,
        mu=mu,
        n_iter=rof_n_iter,
        tol=2e-5,
        return_info=True,
    )
    trof = sat_rof_trof.run_trof_thresholds(
        rof,
        image,
        n_classes=n_classes,
        initial_thresholds=np.quantile(rof, np.linspace(0, 1, n_classes + 1)[1:-1]),
        max_iter=trof_max_iter,
        tol=1e-4,
        projection_bins=4096,
    )

    metrics = {}
    qualitative_only = mask is None
    if mask is not None:
        metrics.update(_metrics_against_mask(mask, trof["labels"]))

    baselines = _run_baselines(image, n_classes, mask)
    figure_path, figure_panels, figure_file = _save_entry_figure(
        entry,
        image,
        mask,
        rof,
        trof["labels"],
        baselines,
        figure_dir,
    )

    result = {
        "family": entry["family"],
        "image_path": _display_path(entry["image_path"]),
        "mask_path": _display_path(entry["mask_path"]) if entry["mask_path"] else None,
        "mask_warning": entry.get("mask_warning"),
        "source_claim": _entry_source_claim(entry, local_manifest),
        "image_file": _file_evidence(entry["image_path"]),
        "mask_file": _file_evidence(entry["mask_path"]) if entry["mask_path"] else None,
        "status": "completed",
        "qualitative_only": qualitative_only,
        "image_shape": list(image.shape),
        "n_classes": n_classes,
        "solver": "sat_rof_trof.rof_chambolle_pock + sat_rof_trof.run_trof_thresholds",
        "parameters": {
            "mu": mu,
            "rof_n_iter": rof_n_iter,
            "trof_max_iter": trof_max_iter,
        },
        "thresholds": [round(float(value), 6) for value in trof["thresholds"]],
        "threshold_iterations": int(trof["iterations"]),
        "rof_iterations": int(rof_info["iterations"]),
        "rof_final_residual": float(rof_info["final_residual"]),
        "metrics": metrics,
        "baselines": _json_baselines(baselines),
        "figure_path": figure_path,
        "figure_file": figure_file,
        "figure_panels": figure_panels,
        "runtime_seconds": elapsed(),
    }
    result.update(_write_figure_evidence_sidecar(result))
    return result


def run_local_dataset(root=DATA_ROOT, mu=8.0, default_classes=4, rof_n_iter=160, trof_max_iter=20, figure_dir=FIGURE_DIR):
    root = Path(root)
    run_protocol = _run_protocol(mu, default_classes, rof_n_iter, trof_max_iter, figure_dir)
    readiness = build_readiness_report(root)
    local_manifest = readiness["local_dataset_manifest"]
    entries = scan_dataset(root)
    dataset_fingerprint = readiness["dataset_fingerprint"]
    mask_warnings = [entry["mask_warning"] for entry in entries if entry.get("mask_warning")]
    missing = require_modules("numpy", "matplotlib", "scipy")
    if missing:
        blockers = [f"Missing modules: {', '.join(missing)}"] + readiness["blockers"] + mask_warnings
        return _with_data_gap_checklist({
            "paper_id": "iterated-rof",
            "target_level": "paper-like",
            "readiness_status": readiness["status"],
            "status": "blocked_missing_dependencies",
            "data_root": _display_path(root),
            "dataset_fingerprint": dataset_fingerprint,
            "run_protocol": run_protocol,
            "image_count": len(entries),
            "quantitative_image_count": 0,
            "families": readiness["families"],
            "family_summaries": _family_summaries(readiness["families"], []),
            "images": [],
            "blockers": blockers,
            "claim_blockers": readiness["claim_blockers"],
            "paper_like_gate": _paper_like_gate(
                readiness["status"],
                blockers,
                readiness["claim_blockers"],
                [],
                data_root=root,
                dataset_fingerprint=dataset_fingerprint,
                figure_dir=figure_dir,
            ),
            "local_dataset_manifest": local_manifest,
            "claim_boundary": readiness["claim_boundary"],
        })

    if not entries:
        blockers = ["No local images found under family images/ directories"] + readiness["blockers"]
        return _with_data_gap_checklist({
            "paper_id": "iterated-rof",
            "target_level": "paper-like",
            "readiness_status": readiness["status"],
            "status": "blocked_missing_data",
            "data_root": _display_path(root),
            "dataset_fingerprint": dataset_fingerprint,
            "run_protocol": run_protocol,
            "image_count": 0,
            "quantitative_image_count": 0,
            "families": readiness["families"],
            "family_summaries": _family_summaries(readiness["families"], []),
            "images": [],
            "blockers": blockers,
            "claim_blockers": readiness["claim_blockers"],
            "paper_like_gate": _paper_like_gate(
                readiness["status"],
                blockers,
                readiness["claim_blockers"],
                [],
                data_root=root,
                dataset_fingerprint=dataset_fingerprint,
                figure_dir=figure_dir,
            ),
            "local_dataset_manifest": local_manifest,
            "claim_boundary": readiness["claim_boundary"],
        })

    image_results = []
    for entry in entries:
        try:
            image_results.append(
                _run_entry(entry, mu, default_classes, rof_n_iter, trof_max_iter, figure_dir, local_manifest)
            )
        except Exception as exc:
            image_results.append(
                {
                    "family": entry["family"],
                    "image_path": _display_path(entry["image_path"]),
                    "mask_path": _display_path(entry["mask_path"]) if entry["mask_path"] else None,
                    "mask_warning": entry.get("mask_warning"),
                    "source_claim": _entry_source_claim(entry, local_manifest),
                    "image_file": _file_evidence(entry["image_path"]),
                    "mask_file": _file_evidence(entry["mask_path"]) if entry["mask_path"] else None,
                    "status": "failed",
                    "qualitative_only": entry["mask_path"] is None,
                    "metrics": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    completed_images = [item for item in image_results if item["status"] == "completed"]
    failed_images = [item for item in image_results if item["status"] == "failed"]
    quantitative_images = [item for item in completed_images if _is_quantitative_image_result(item)]
    if failed_images and not completed_images:
        status = "blocked_runner_failed"
    elif failed_images:
        status = "completed_local_runner_with_failures"
    elif quantitative_images:
        status = "completed_local_runner"
    else:
        status = "completed_qualitative_only"

    blockers = list(readiness["blockers"]) + mask_warnings
    if not quantitative_images:
        blockers.append("No completed image has a mask, so quantitative metrics are unavailable")
    if failed_images:
        blockers.append(f"{len(failed_images)} image(s) failed during local runner execution")

    return _with_data_gap_checklist({
        "paper_id": "iterated-rof",
        "target_level": "paper-like",
        "readiness_status": readiness["status"],
        "status": status,
        "data_root": _display_path(root),
        "dataset_fingerprint": dataset_fingerprint,
        "run_protocol": run_protocol,
        "image_count": len(image_results),
        "completed_image_count": len(completed_images),
        "quantitative_image_count": len(quantitative_images),
        "families": readiness["families"],
        "family_summaries": _family_summaries(readiness["families"], image_results),
        "images": image_results,
        "blockers": blockers,
        "claim_blockers": readiness["claim_blockers"],
        "paper_like_gate": _paper_like_gate(
            readiness["status"],
            blockers,
            readiness["claim_blockers"],
            image_results,
            data_root=root,
            dataset_fingerprint=dataset_fingerprint,
            figure_dir=figure_dir,
        ),
        "local_dataset_manifest": local_manifest,
        "claim_boundary": readiness["claim_boundary"],
    })


def write_report(report, path=REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_data_gap_checklist(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checklist = build_data_gap_checklist(report)
    path.write_text(json.dumps(checklist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _csv_join(values):
    return ";".join(str(value) for value in values)


CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r", "\n")


def _csv_safe_cell(value):
    if isinstance(value, str) and value.startswith(CSV_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def _csv_safe_rows(rows):
    for row in rows:
        yield {key: _csv_safe_cell(value) for key, value in row.items()}


def _family_summary_csv_rows(report):
    rows = []
    metric_keys = set()
    baseline_metric_keys = set()
    for summary in report.get("family_summaries", []):
        metric_keys.update(summary.get("metrics_mean", {}))
        for baseline_name, metrics in summary.get("baseline_metrics_mean", {}).items():
            for metric_name in metrics:
                baseline_metric_keys.add((baseline_name, metric_name))

    metric_keys = sorted(metric_keys)
    baseline_metric_keys = sorted(baseline_metric_keys)
    for summary in report.get("family_summaries", []):
        source_ids = [
            claim.get("source_id")
            for claim in summary.get("source_claims", [])
            if claim.get("source_id")
        ]
        row = {
            "family": summary["family"],
            "status": summary["status"],
            "image_count": summary["image_count"],
            "mask_count": summary["mask_count"],
            "matched_mask_count": summary["matched_mask_count"],
            "completed_image_count": summary["completed_image_count"],
            "failed_image_count": summary["failed_image_count"],
            "quantitative_image_count": summary["quantitative_image_count"],
            "qualitative_image_count": summary["qualitative_image_count"],
            "figure_paths": _csv_join(summary.get("figure_paths", [])),
            "source_ids": _csv_join(sorted(set(source_ids))),
            "errors": _csv_join(summary.get("errors", [])),
        }
        for metric_key in metric_keys:
            row[f"metric_{metric_key}"] = summary.get("metrics_mean", {}).get(metric_key, "")
        for baseline_name, metric_key in baseline_metric_keys:
            row[f"baseline_{baseline_name}_{metric_key}"] = (
                summary.get("baseline_metrics_mean", {})
                .get(baseline_name, {})
                .get(metric_key, "")
            )
        rows.append(row)
    return rows


def write_family_summary_csv(report, path=FAMILY_SUMMARY_CSV_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _family_summary_csv_rows(report)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "family",
        "status",
        "image_count",
        "mask_count",
        "matched_mask_count",
        "completed_image_count",
        "failed_image_count",
        "quantitative_image_count",
        "qualitative_image_count",
        "source_ids",
        "figure_paths",
        "errors",
    ]
    fieldnames = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_safe_rows(rows))
    return path


IMAGE_EVIDENCE_CSV_FIELDNAMES = [
    "data_root",
    "report_status",
    "readiness_status",
    "paper_like_gate_passed",
    "dataset_fingerprint_sha256",
    "dataset_fingerprint_file_count",
    "family",
    "status",
    "image_path",
    "source_claim_image",
    "mask_path",
    "qualitative_only",
    "mask_warning",
    "figure_path",
    "figure_panels",
    "figure_sha256",
    "figure_size_bytes",
    "figure_evidence_path",
    "figure_evidence_sha256",
    "figure_evidence_size_bytes",
    "figure_evidence_generator",
    "image_sha256",
    "image_size_bytes",
    "mask_sha256",
    "mask_size_bytes",
    "source_id",
    "source_name",
    "claim_scope",
    "source_manifest_status",
    "source_manifest_path",
    "source_license_reviewed",
    "source_citation",
    "source_license_note",
    "source_provenance_reviewed",
    "source_provenance_note",
    "source_synthetic_fixture",
    "source_claim_sha256",
    "source_claim_mask_sha256",
    "solver",
    "n_classes",
    "mu",
    "rof_n_iter",
    "trof_max_iter",
    "thresholds",
    "threshold_iterations",
    "rof_iterations",
    "rof_final_residual",
    "runtime_seconds",
    "clustering_accuracy",
    "dice",
    "baseline_raw_kmeans_method",
    "baseline_raw_kmeans_clustering_accuracy",
    "baseline_raw_kmeans_dice",
    "baseline_multi_otsu_method",
    "baseline_multi_otsu_thresholds",
    "baseline_multi_otsu_clustering_accuracy",
    "baseline_multi_otsu_dice",
    "error",
]


def _csv_json_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _baseline_metric(item, baseline_name, metric_name):
    return (
        item.get("baselines", {})
        .get(baseline_name, {})
        .get("metrics", {})
        .get(metric_name, "")
    )


def _image_evidence_csv_rows(report):
    rows = []
    gate = _paper_like_gate_from_report(report)
    fingerprint = report.get("dataset_fingerprint", {})
    for item in report.get("images", []):
        source_claim = item.get("source_claim", {})
        raw_kmeans = item.get("baselines", {}).get("raw_kmeans", {})
        multi_otsu = item.get("baselines", {}).get("multi_otsu", {})
        parameters = item.get("parameters", {})
        figure_evidence = item.get("figure_evidence", {})
        rows.append(
            {
                "data_root": report.get("data_root", ""),
                "report_status": report.get("status", ""),
                "readiness_status": report.get("readiness_status", ""),
                "paper_like_gate_passed": gate.get("passed", ""),
                "dataset_fingerprint_sha256": fingerprint.get("sha256", ""),
                "dataset_fingerprint_file_count": fingerprint.get("file_count", ""),
                "family": item.get("family", ""),
                "status": item.get("status", ""),
                "image_path": item.get("image_path", ""),
                "source_claim_image": source_claim.get("image", ""),
                "mask_path": item.get("mask_path") or "",
                "qualitative_only": item.get("qualitative_only", ""),
                "mask_warning": item.get("mask_warning") or "",
                "figure_path": item.get("figure_path", ""),
                "figure_panels": _csv_json_value(item.get("figure_panels", [])),
                "figure_sha256": item.get("figure_file", {}).get("sha256", ""),
                "figure_size_bytes": item.get("figure_file", {}).get("size_bytes", ""),
                "figure_evidence_path": item.get("figure_evidence_path", ""),
                "figure_evidence_sha256": item.get("figure_evidence_file", {}).get("sha256", ""),
                "figure_evidence_size_bytes": item.get("figure_evidence_file", {}).get("size_bytes", ""),
                "figure_evidence_generator": figure_evidence.get("generator", ""),
                "image_sha256": item.get("image_file", {}).get("sha256", ""),
                "image_size_bytes": item.get("image_file", {}).get("size_bytes", ""),
                "mask_sha256": (item.get("mask_file") or {}).get("sha256", ""),
                "mask_size_bytes": (item.get("mask_file") or {}).get("size_bytes", ""),
                "source_id": source_claim.get("source_id", ""),
                "source_name": source_claim.get("source_name", ""),
                "claim_scope": source_claim.get("claim_scope", ""),
                "source_manifest_status": source_claim.get("manifest_status", ""),
                "source_manifest_path": source_claim.get("manifest_path", ""),
                "source_license_reviewed": source_claim.get("license_reviewed", ""),
                "source_citation": source_claim.get("citation", ""),
                "source_license_note": source_claim.get("license_note", ""),
                "source_provenance_reviewed": source_claim.get("provenance_reviewed", ""),
                "source_provenance_note": source_claim.get("provenance_note", ""),
                "source_synthetic_fixture": source_claim.get("synthetic_fixture", ""),
                "source_claim_sha256": source_claim.get("sha256", ""),
                "source_claim_mask_sha256": source_claim.get("mask_sha256", ""),
                "solver": item.get("solver", ""),
                "n_classes": item.get("n_classes", ""),
                "mu": parameters.get("mu", ""),
                "rof_n_iter": parameters.get("rof_n_iter", ""),
                "trof_max_iter": parameters.get("trof_max_iter", ""),
                "thresholds": _csv_json_value(item.get("thresholds", [])),
                "threshold_iterations": item.get("threshold_iterations", ""),
                "rof_iterations": item.get("rof_iterations", ""),
                "rof_final_residual": item.get("rof_final_residual", ""),
                "runtime_seconds": item.get("runtime_seconds", ""),
                "clustering_accuracy": item.get("metrics", {}).get("clustering_accuracy", ""),
                "dice": item.get("metrics", {}).get("dice", ""),
                "baseline_raw_kmeans_method": raw_kmeans.get("method", ""),
                "baseline_raw_kmeans_clustering_accuracy": _baseline_metric(
                    item,
                    "raw_kmeans",
                    "clustering_accuracy",
                ),
                "baseline_raw_kmeans_dice": _baseline_metric(
                    item,
                    "raw_kmeans",
                    "dice",
                ),
                "baseline_multi_otsu_method": multi_otsu.get("method", ""),
                "baseline_multi_otsu_thresholds": _csv_json_value(multi_otsu.get("thresholds", [])),
                "baseline_multi_otsu_clustering_accuracy": _baseline_metric(
                    item,
                    "multi_otsu",
                    "clustering_accuracy",
                ),
                "baseline_multi_otsu_dice": _baseline_metric(
                    item,
                    "multi_otsu",
                    "dice",
                ),
                "error": item.get("error", ""),
            }
        )
    return rows


def write_image_evidence_csv(report, path=IMAGE_EVIDENCE_CSV_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _image_evidence_csv_rows(report)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=IMAGE_EVIDENCE_CSV_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(_csv_safe_rows(rows))
    return path


def _image_evidence_counts(images):
    completed_images = [item for item in images if item.get("status") == "completed"]
    quantitative_images = [item for item in completed_images if _is_quantitative_image_result(item)]
    return {
        "image_count": len(images),
        "completed_image_count": len(completed_images),
        "quantitative_image_count": len(quantitative_images),
    }


def _flatten_candidate_metrics(report, family_summaries=None, image_counts=None):
    metrics = {}
    for summary in family_summaries if family_summaries is not None else report.get("family_summaries", []):
        family = summary["family"]
        for metric_name, value in summary.get("metrics_mean", {}).items():
            metrics[f"{family}_{metric_name}"] = value
        for baseline_name, baseline_metrics in summary.get("baseline_metrics_mean", {}).items():
            for metric_name, value in baseline_metrics.items():
                metrics[f"{family}_{baseline_name}_{metric_name}"] = value
    counts = image_counts if image_counts is not None else _image_evidence_counts(report.get("images", []))
    metrics.update(counts)
    return metrics


def _paper_like_promotion_verification(report, gate, shape_blockers, can_promote, source_summary_path=None):
    verification = {
        "schema_version": 1,
        "generated_by": PAPER_LIKE_PROMOTION_VERIFICATION_GENERATOR,
        "recomputed_gate": True,
        "can_promote": can_promote,
        "promotion_shape_blocker_count": len(shape_blockers),
        "gate_id": gate.get("evidence_summary", {}).get("gate_id", PAPER_LIKE_GATE_ID),
        "dataset_fingerprint": report.get("dataset_fingerprint", {}),
    }
    if source_summary_path:
        source_summary_path = Path(source_summary_path)
        verification["source_summary_path"] = _display_path(source_summary_path)
        resolved_source_summary_path = _resolve_report_path(source_summary_path)
        if resolved_source_summary_path is not None and resolved_source_summary_path.exists():
            verification["source_summary_sha256"] = _file_evidence(resolved_source_summary_path)["sha256"]
    return verification


def _candidate_runtime_seconds(report):
    value = report.get("runtime_seconds")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    image_runtimes = [
        float(item.get("runtime_seconds"))
        for item in report.get("images", [])
        if isinstance(item, dict)
        and isinstance(item.get("runtime_seconds"), (int, float))
        and not isinstance(item.get("runtime_seconds"), bool)
        and math.isfinite(float(item.get("runtime_seconds")))
    ]
    return round(sum(image_runtimes), 6) if image_runtimes else 0.0


def _report_data_root(report):
    data_root = report.get("data_root")
    if not data_root:
        return DATA_ROOT
    resolved = _resolve_report_path(data_root)
    return resolved if resolved is not None else DATA_ROOT


def _paper_like_gate_from_report(report):
    return _paper_like_gate(
        report.get("readiness_status", report.get("status")),
        report.get("blockers", []),
        report.get("claim_blockers", []),
        report["images"] if "images" in report else None,
        data_root=_report_data_root(report),
        dataset_fingerprint=report.get("dataset_fingerprint", {}),
    )


def _promotion_family_summaries(report):
    return _family_summaries(_promotion_family_rows(report.get("images", [])), report.get("images", [])), []


def _promotion_family_rows(images):
    rows = []
    images = [item for item in images or [] if isinstance(item, dict)]
    for family in DATA_FAMILIES:
        family_images = [item for item in images if item.get("family") == family]
        mask_count = sum(1 for item in family_images if item.get("mask_path"))
        if mask_count:
            status = "ready_quantitative"
        elif family_images:
            status = "ready_qualitative"
        else:
            status = "missing"
        rows.append(
            {
                "family": family,
                "description": DATA_FAMILIES[family],
                "image_count": len(family_images),
                "mask_count": mask_count,
                "matched_mask_count": mask_count,
                "status": status,
                "path": _display_path(DATA_ROOT / family),
                "images": [
                    {
                        "image_path": item.get("image_path"),
                        "image_relative_path": _image_result_relative_path(item),
                        "mask_path": item.get("mask_path"),
                        "mask_warning": item.get("mask_warning"),
                        "image_file": item.get("image_file", {}),
                        "mask_file": item.get("mask_file"),
                    }
                    for item in family_images
                ],
            }
        )
    return rows


def _dashboard_result_file_for_figure(figure_path):
    resolved = _resolve_report_path(figure_path)
    if resolved is None:
        return str(figure_path)
    try:
        relative = resolved.resolve().relative_to(FIGURE_DIR.resolve()).as_posix()
    except Exception:
        relative = resolved.name
    return f"assets/repro/iterated_rof_paper_like/{relative}"


def _dashboard_result_file_for_canonical_figure(figure_path):
    resolved = _resolve_report_path(figure_path)
    if resolved is None:
        return None
    try:
        relative = resolved.resolve().relative_to(FIGURE_DIR.resolve()).as_posix()
    except Exception:
        return None
    if not relative or relative.startswith("../") or "/../" in relative:
        return None
    return f"assets/repro/iterated_rof_paper_like/{relative}"


def _dashboard_result_file_relative_parts(result_file):
    if not isinstance(result_file, str) or not result_file.strip():
        return None
    if result_file != result_file.strip() or "\\" in result_file:
        return None
    path = PurePosixPath(result_file)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != result_file:
        return None
    try:
        relative = path.relative_to(PurePosixPath("assets/repro"))
    except ValueError:
        return None
    if not relative.parts:
        return None
    return relative.parts


def _dashboard_static_asset_path(result_file):
    relative_parts = _dashboard_result_file_relative_parts(result_file)
    if relative_parts is None:
        return None
    return DASHBOARD_REPRO_ASSET_ROOT.joinpath(*relative_parts)


def _dashboard_static_asset_status(rows, can_promote):
    if not can_promote:
        return "blocked_not_promotable"
    if not rows:
        return "missing"
    statuses = {row.get("status") for row in rows}
    for status in ["source_missing", "stale", "missing"]:
        if status in statuses:
            return status
    if statuses == {"current"}:
        return "current"
    return "blocked"


def _dashboard_static_asset_rows(report, candidate):
    rows = []
    candidate_result_files = set(candidate.get("runResultPatch", {}).get("resultFiles", []))
    completed_quantitative = [
        item
        for item in report.get("images", [])
        if isinstance(item, dict) and _is_quantitative_image_result(item)
    ]
    for item in completed_quantitative:
        figure_path = item.get("figure_path")
        result_file = _dashboard_result_file_for_canonical_figure(figure_path)
        static_path = _dashboard_static_asset_path(result_file) if result_file else None
        source_path = _resolve_report_path(figure_path)
        figure_file = item.get("figure_file", {}) if isinstance(item.get("figure_file"), dict) else {}
        source_sha256 = figure_file.get("sha256", "")
        source_size_bytes = figure_file.get("size_bytes", 0)
        row = {
            "family": item.get("family"),
            "image_path": item.get("image_path"),
            "source_figure_path": _display_path(source_path) if source_path is not None else str(figure_path),
            "source_figure_sha256": source_sha256,
            "source_figure_size_bytes": source_size_bytes,
            "result_file": result_file or "",
            "static_asset_path": _display_path(static_path) if static_path is not None else "",
            "in_dashboard_candidate": bool(result_file) and result_file in candidate_result_files,
        }
        if source_path is None or not source_path.is_file():
            row["status"] = "source_missing"
            rows.append(row)
            continue
        if not source_sha256:
            source_evidence = _file_evidence(source_path)
            row["source_figure_sha256"] = source_evidence["sha256"]
            row["source_figure_size_bytes"] = source_evidence["size_bytes"]
            source_sha256 = source_evidence["sha256"]
        if result_file is None or static_path is None or not _path_is_under(static_path, DASHBOARD_REPRO_ASSET_ROOT):
            row["status"] = "invalid_result_file"
            rows.append(row)
            continue
        if not static_path.is_file():
            row["status"] = "missing"
            rows.append(row)
            continue
        static_evidence = _file_evidence(static_path)
        row["static_asset_sha256"] = static_evidence["sha256"]
        row["static_asset_size_bytes"] = static_evidence["size_bytes"]
        row["status"] = "current" if static_evidence["sha256"] == source_sha256 else "stale"
        rows.append(row)
    return rows


def _dashboard_result_files_for_quantitative_images(report):
    result_files = []
    for item in report.get("images", []):
        if not isinstance(item, dict) or not _is_quantitative_image_result(item):
            continue
        result_file = _dashboard_result_file_for_canonical_figure(item.get("figure_path"))
        if result_file:
            result_files.append(result_file)
    return result_files


def build_dashboard_static_asset_manifest(report, source_summary_path=None):
    candidate = build_dashboard_candidate(report, source_summary_path=source_summary_path)
    rows = _dashboard_static_asset_rows(report, candidate)
    source_summary_sha256 = ""
    if source_summary_path:
        resolved_source_summary_path = _resolve_report_path(source_summary_path)
        if resolved_source_summary_path is not None and resolved_source_summary_path.exists():
            source_summary_sha256 = _file_evidence(resolved_source_summary_path)["sha256"]
    can_promote = candidate["can_promote"] is True
    status = _dashboard_static_asset_status(rows, can_promote)
    return {
        "schema_version": 1,
        "generated_by": DASHBOARD_STATIC_ASSET_GENERATOR,
        "paper_id": report.get("paper_id", "iterated-rof"),
        "experimentId": "iterated_rof_paper_like",
        "can_promote": can_promote,
        "status": status,
        "all_static_assets_current": can_promote and bool(rows) and all(row.get("status") == "current" for row in rows),
        "asset_root": _display_path(DASHBOARD_REPRO_ASSET_ROOT),
        "asset_prefix": "assets/repro/iterated_rof_paper_like",
        "source_summary_path": _display_path(source_summary_path) if source_summary_path else "",
        "source_summary_sha256": source_summary_sha256,
        "asset_count": len(rows),
        "assets": rows,
        "promotion_shape_blockers": candidate.get("promotionShapeBlockers", []),
        "blocked_reasons": candidate.get("blockedReasons", []),
        "copy_requested": False,
        "copy_performed": False,
        "copy_blockers": [],
    }


def write_dashboard_static_asset_manifest(report, path, source_summary_path=None, copy_assets=False):
    copy_requested = bool(copy_assets)
    copy_performed = False
    copy_blockers = []
    if copy_requested:
        pre_copy_manifest = build_dashboard_static_asset_manifest(
            report,
            source_summary_path=source_summary_path,
        )
        if pre_copy_manifest["can_promote"] is not True:
            copy_blockers.append("Dashboard static asset copy requires a promotable dashboard candidate")
        for asset in pre_copy_manifest.get("assets", []):
            source_path = _resolve_report_path(asset.get("source_figure_path"))
            static_path = _dashboard_static_asset_path(asset.get("result_file"))
            if source_path is None or not source_path.is_file():
                copy_blockers.append(
                    f"Dashboard static asset source figure is missing for: {asset.get('image_path')}"
                )
            if static_path is None or not _path_is_under(static_path, DASHBOARD_REPRO_ASSET_ROOT):
                copy_blockers.append(
                    f"Dashboard static asset result file is unsafe for: {asset.get('result_file')}"
                )
        if not copy_blockers:
            for asset in pre_copy_manifest.get("assets", []):
                source_path = _resolve_report_path(asset["source_figure_path"])
                static_path = _dashboard_static_asset_path(asset["result_file"])
                static_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, static_path)
            copy_performed = True

    manifest = build_dashboard_static_asset_manifest(report, source_summary_path=source_summary_path)
    manifest["copy_requested"] = copy_requested
    manifest["copy_performed"] = copy_performed
    manifest["copy_blockers"] = copy_blockers
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _promotion_protocol_blockers(report, completed_quantitative=None):
    protocol = report.get("run_protocol", {})
    if not isinstance(protocol, dict):
        return ["Dashboard promotion candidate requires run_protocol object"]

    expected_static_values = {
        "protocol_id": "iterated_rof_trof_local_data_v1",
        "schema_version": 1,
        "paper_id": "iterated-rof",
        "algorithm": "ROF denoising followed by iterative multiclass thresholding",
        "solver": "sat_rof_trof.rof_chambolle_pock + sat_rof_trof.run_trof_thresholds",
        "threshold_update": "tau_i = 0.5 * (mean_f(Omega_{i-1}) + mean_f(Omega_i))",
        "threshold_mean_source": "raw input image f, not denoised ROF output u",
        "baselines": ["raw_kmeans", "multi_otsu"],
        "metrics": ["clustering_accuracy", "binary_dice_when_mask_has_two_labels"],
        "data_layout": "reproduce/data/iterated_rof/{cartoon,texture,medical}/{images,masks}",
        "figure_dir": _display_path(FIGURE_DIR),
        "promotion_rule": "dashboard paper-like promotion requires paper_like_gate.passed=true",
    }
    blockers = []
    for key, expected_value in expected_static_values.items():
        if protocol.get(key) != expected_value:
            blockers.append(
                f"Dashboard promotion candidate run_protocol {key} does not match expected Iterated ROF runner"
            )

    parameters = protocol.get("parameters", {})
    if not isinstance(parameters, dict):
        blockers.append("Dashboard promotion candidate run_protocol parameters must be an object")
        return blockers
    expected_parameter_values = {
        "rof_tol": 2e-5,
        "trof_tol": 1e-4,
        "projection_bins": 4096,
        "seed": SEED,
    }
    for key, expected_value in expected_parameter_values.items():
        if parameters.get(key) != expected_value:
            blockers.append(
                f"Dashboard promotion candidate run_protocol parameter {key} does not match expected Iterated ROF runner"
            )
    for key in ["mu", "default_classes", "rof_n_iter", "trof_max_iter"]:
        value = parameters.get(key)
        if key == "mu":
            valid = _is_finite_number(value) and float(value) > 0
            description = "a positive finite number"
        else:
            valid = _is_positive_int(value)
            description = "a positive integer"
        if not valid:
            blockers.append(
                f"Dashboard promotion candidate run_protocol parameter {key} must be {description}"
            )

    completed_quantitative = completed_quantitative or []
    for item in completed_quantitative:
        item_parameters = item.get("parameters", {}) if isinstance(item.get("parameters"), dict) else {}
        for key in ["mu", "rof_n_iter", "trof_max_iter"]:
            if parameters.get(key) != item_parameters.get(key):
                blocker = f"Dashboard promotion candidate run_protocol parameter {key} does not match completed image evidence"
                if blocker not in blockers:
                    blockers.append(blocker)
    return blockers


def _summary_verification(report, source_summary_path):
    candidate = build_dashboard_candidate(report, source_summary_path=source_summary_path)
    return {
        "status": "verified_promotable" if candidate["can_promote"] else "blocked",
        "source_summary_path": _display_path(source_summary_path),
        "recomputed_gate": True,
        "can_promote": candidate["can_promote"],
        "promotion_shape_blockers": candidate.get("promotionShapeBlockers", []),
    }


SOURCE_SUMMARY_ARTIFACT_MATCH_FIELDS = (
    "paper_id",
    "target_level",
    "status",
    "readiness_status",
    "data_root",
    "image_count",
    "completed_image_count",
    "quantitative_image_count",
    "dataset_fingerprint",
    "images",
    "family_summaries",
    "local_dataset_manifest",
    "run_protocol",
)


def _source_summary_artifact_blockers(report, gate, source_summary_path):
    blockers = []
    if not source_summary_path:
        return ["Dashboard promotion candidate requires source summary artifact path"]

    if not _path_is_under(source_summary_path, RESULTS_DIR):
        blockers.append(
            "Dashboard promotion candidate source summary artifact must be under reproduce/results"
        )

    source_summary_resolved = _resolve_report_path(source_summary_path)
    if source_summary_resolved is None or not source_summary_resolved.exists():
        blockers.append("Dashboard promotion candidate source summary artifact path is missing")
        return blockers

    try:
        source_summary = json.loads(source_summary_resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        blockers.append(f"Dashboard promotion candidate source summary artifact is not readable JSON: {type(exc).__name__}: {exc}")
        return blockers

    for field in SOURCE_SUMMARY_ARTIFACT_MATCH_FIELDS:
        if source_summary.get(field) != report.get(field):
            blockers.append(f"Dashboard promotion candidate source summary artifact {field} does not match current report")

    source_gate = _paper_like_gate_from_report(source_summary)
    if source_gate != gate:
        blockers.append("Dashboard promotion candidate source summary artifact recomputed gate does not match current report")

    return blockers


def verify_saved_summary(path):
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    report["paper_like_gate"] = _paper_like_gate_from_report(report)
    report["summary_verification"] = _summary_verification(report, path)
    return _with_data_gap_checklist(report)


def _promotion_report_shape_blockers(report):
    blockers = []
    source_manifest = load_source_manifest()
    blockers.extend(_source_manifest_schema_blockers(source_manifest))
    if report.get("status") != "completed_local_runner":
        blockers.append("Dashboard promotion candidate requires completed_local_runner status")
    if report.get("readiness_status") != "ready_for_paper_like_runner":
        blockers.append("Dashboard promotion candidate requires ready_for_paper_like_runner readiness")
    if report.get("run_protocol", {}).get("protocol_id") != "iterated_rof_trof_local_data_v1":
        blockers.append("Dashboard promotion candidate requires iterated_rof_trof_local_data_v1 run protocol")
    if report.get("local_dataset_manifest", {}).get("status") != "present":
        blockers.append("Dashboard promotion candidate requires present local_dataset_manifest")

    images = report.get("images", [])
    report_data_root = _report_data_root(report)
    local_manifest = load_local_dataset_manifest(report_data_root)
    image_counts = _image_evidence_counts(images)
    family_rows = _promotion_family_rows(images)
    blockers.extend(_claim_blockers(family_rows, local_manifest, source_manifest))
    current_entries = scan_dataset(report_data_root)
    current_family_rows = [scan_family(report_data_root, family) for family in DATA_FAMILIES]
    blockers.extend(_claim_blockers(current_family_rows, local_manifest, source_manifest))
    current_fingerprint = _dataset_fingerprint(current_entries)
    if images and current_fingerprint["file_count"] > 0 and not _dataset_fingerprints_match(
        current_fingerprint,
        report.get("dataset_fingerprint", {}),
    ):
        blockers.append("Dashboard promotion candidate current local data root fingerprint does not match report dataset_fingerprint")
    family_summaries, family_summary_blockers = _promotion_family_summaries(report)
    blockers.extend(family_summary_blockers)
    if not family_summary_blockers and report.get("family_summaries") != family_summaries:
        blockers.append("Dashboard promotion candidate family_summaries do not match image evidence rows")
    completed_quantitative = [item for item in images if _is_quantitative_image_result(item)]
    blockers.extend(_promotion_protocol_blockers(report, completed_quantitative=completed_quantitative))
    if not completed_quantitative:
        blockers.append("Dashboard promotion candidate requires completed quantitative image evidence rows with masks")
    if image_counts["image_count"] != report.get("image_count", 0):
        blockers.append("Dashboard promotion candidate image_count does not match evidence rows")
    if image_counts["completed_image_count"] != report.get("completed_image_count", 0):
        blockers.append("Dashboard promotion candidate completed_image_count does not match evidence rows")
    if image_counts["quantitative_image_count"] != report.get("quantitative_image_count", 0):
        blockers.append("Dashboard promotion candidate quantitative image count does not match evidence rows")
    if images and not _fingerprint_matches_image_results(report.get("dataset_fingerprint", {}), images):
        blockers.append("Dashboard promotion candidate dataset fingerprint does not match image/mask evidence")
    for item in completed_quantitative:
        image_label = item.get("image_path", "<unknown image>")
        blockers.extend(
            _completed_output_shape_blockers(
                item,
                "Dashboard promotion candidate",
                "Dashboard promotion candidate",
                source_manifest=source_manifest,
                audit_roots=_source_audit_roots_for_family(report_data_root, item.get("family")),
            )
        )
        blockers.extend(_figure_evidence_blockers(item, "Dashboard promotion candidate"))
        if not _file_evidence_matches_disk(item.get("image_path"), item.get("image_file")):
            blockers.append(f"Dashboard promotion candidate image evidence does not match disk for: {image_label}")
        if item.get("mask_path") and not _file_evidence_matches_disk(item.get("mask_path"), item.get("mask_file")):
            blockers.append(f"Dashboard promotion candidate mask evidence does not match disk for: {image_label}")
        if item.get("figure_path") and not _file_evidence_matches_disk(item.get("figure_path"), item.get("figure_file")):
            blockers.append(f"Dashboard promotion candidate figure evidence does not match disk for: {image_label}")
        if not _source_claim_matches_manifest(item, local_manifest):
            blockers.append(f"Dashboard promotion candidate saved source claim does not match current manifest for: {image_label}")
    return blockers


def build_dashboard_candidate(report, source_summary_path=None):
    gate = _paper_like_gate_from_report(report)
    shape_blockers = _promotion_report_shape_blockers(report)
    shape_blockers.extend(_source_summary_artifact_blockers(report, gate, source_summary_path))
    can_promote = gate.get("passed") is True and not shape_blockers
    candidate = {
        "paper_id": report.get("paper_id", "iterated-rof"),
        "priority": ITERATED_ROF_DASHBOARD_PRIORITY,
        "experimentId": "iterated_rof_paper_like",
        "can_promote": can_promote,
        "reproductionLevel": "paper-like" if can_promote else "partial",
        "reproductionTruthLevel": "partial-completed",
        "paperLikeGate": gate,
        "promotionShapeBlockers": shape_blockers,
        "blockedReasons": [] if can_promote else gate.get("reasons", []) + shape_blockers,
        "claimBoundary": report.get("claim_boundary"),
        "candidateDetails": {},
    }
    if not can_promote:
        return candidate

    family_summaries, _ = _promotion_family_summaries(report)
    image_counts = _image_evidence_counts(report.get("images", []))
    run_metrics = _flatten_candidate_metrics(report, family_summaries, image_counts)
    runtime_seconds = _candidate_runtime_seconds(report)
    result_files = _dashboard_result_files_for_quantitative_images(report)
    verification = _paper_like_promotion_verification(
        report,
        gate,
        shape_blockers,
        can_promote,
        source_summary_path=source_summary_path,
    )
    dashboard_detail_patch = {
        "experimentId": "iterated_rof_paper_like",
        "reproductionLevel": "paper-like",
        "resultStatus": "completed",
        "runtimeSeconds": runtime_seconds,
        "runMetrics": run_metrics,
        "resultFiles": result_files,
        "paper_like_gate": gate,
        "paper_like_verification": verification,
        "reproductionTruthLevel": "partial-completed",
        "family_summaries": family_summaries,
        "dataset_fingerprint": report.get("dataset_fingerprint", {}),
        "run_protocol": report.get("run_protocol", {}),
        "notes": (
            "Paper-like candidate generated from local data-backed Iterated ROF runner. "
            "Review dataset provenance and figures before updating dashboard fields."
        ),
        "warning": (
            "Paper-like candidate only; not paper-level. Original paper data/protocol may still differ."
        ),
        "fidelityWarning": PAPER_LIKE_FIDELITY_WARNING,
    }
    candidate["dashboardDetailPatch"] = dashboard_detail_patch
    candidate["candidateDetails"] = dashboard_detail_patch
    candidate["runResultPatch"] = {
        "priority": ITERATED_ROF_DASHBOARD_PRIORITY,
        "id": report.get("paper_id", "iterated-rof"),
        "experiment_id": "iterated_rof_paper_like",
        "reproductionLevel": "paper-like",
        "status": "completed",
        "runtime_seconds": runtime_seconds,
        "metrics": run_metrics,
        "resultFiles": result_files,
        "notes": dashboard_detail_patch["notes"],
        "warning": dashboard_detail_patch["warning"],
        "fidelityWarning": dashboard_detail_patch["fidelityWarning"],
        "paper_like_gate": gate,
        "paper_like_verification": verification,
    }
    return candidate


def build_promotion_audit(report, source_summary_path=None):
    candidate = build_dashboard_candidate(report, source_summary_path=source_summary_path)
    gate = candidate["paperLikeGate"]
    shape_blockers = candidate.get("promotionShapeBlockers", [])
    can_promote = candidate["can_promote"]
    family_summaries, _ = _promotion_family_summaries(report)
    image_counts = _image_evidence_counts(report.get("images", []))
    checklist = {
        item["id"]: {
            "passed": item.get("passed") is True,
            "reason_count": len(item.get("reasons", [])),
            "reasons": item.get("reasons", []),
        }
        for item in gate.get("checklist", [])
    }
    family_status_counts = {}
    for summary in family_summaries:
        status = summary.get("status", "unknown")
        family_status_counts[status] = family_status_counts.get(status, 0) + 1
    data_gap = build_data_gap_checklist(report)
    source_audit_by_family = [
        {
            "family": item.get("family"),
            "status": item.get("source_audit", {}).get("status"),
            "audit_root_exists": item.get("source_audit", {}).get("audit_root_exists"),
            "issue_count": item.get("source_audit", {}).get("issue_count", 0),
            "issues": item.get("source_audit", {}).get("issues", []),
            "artifacts": item.get("source_audit", {}).get("artifacts", {}),
            "file_overrides": item.get("source_audit", {}).get("file_overrides", []),
        }
        for item in data_gap.get("families", [])
    ]

    return {
        "paper_id": report.get("paper_id", "iterated-rof"),
        "target_level": report.get("target_level", "paper-like"),
        "can_promote": can_promote,
        "recommended_dashboard_level": "paper-like" if can_promote else "partial",
        "status": report.get("status"),
        "readiness_status": report.get("readiness_status"),
        "image_count": image_counts["image_count"],
        "completed_image_count": image_counts["completed_image_count"],
        "quantitative_image_count": image_counts["quantitative_image_count"],
        "dataset_fingerprint": report.get("dataset_fingerprint", {}),
        "local_dataset_manifest_status": report.get("local_dataset_manifest", {}).get("status"),
        "blocked_reason_count": len(gate.get("reasons", [])),
        "blocked_reasons": gate.get("reasons", []),
        "promotion_shape_blocker_count": len(shape_blockers),
        "promotion_shape_blockers": shape_blockers,
        "claim_blocker_count": len(report.get("claim_blockers", [])),
        "blocker_count": len(report.get("blockers", [])),
        "ready_for_local_runner": data_gap.get("ready_for_local_runner") is True,
        "data_ready_blocker_count": data_gap.get("data_ready_blocker_count", 0),
        "data_ready_blockers": data_gap.get("data_ready_blockers", []),
        "family_status_counts": family_status_counts,
        "source_audit_status_counts": data_gap.get("source_audit_status_counts", {}),
        "source_audit_by_family": source_audit_by_family,
        "checklist": checklist,
        "claim_boundary": report.get("claim_boundary"),
    }


def write_promotion_audit(report, path, source_summary_path=None):
    audit = build_promotion_audit(report, source_summary_path=source_summary_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def write_dashboard_candidate(report, path, source_summary_path=None):
    candidate = build_dashboard_candidate(report, source_summary_path=source_summary_path)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit iterated-rof paper-like reproduction data readiness.")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Local iterated-rof data root")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Print recommended dataset sources and still write the readiness JSON report.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run local images through ROF + T-ROF and write a JSON summary.",
    )
    parser.add_argument(
        "--verify-summary",
        default=None,
        help="Verify a saved --run summary by recomputing paper_like_gate and promotion blockers without rerunning.",
    )
    parser.add_argument(
        "--prepare-data-layout",
        action="store_true",
        help="Create local family images/masks directories and dataset_manifest.json from the template; no downloads.",
    )
    parser.add_argument(
        "--ingest-data-drop",
        default=None,
        help=(
            "Copy a user-prepared local data drop with {family}/{images,masks,audit} layout into "
            "--data-root, create the manifest template if needed, and refresh file hash claims. No downloads."
        ),
    )
    parser.add_argument(
        "--review-data-drop",
        default=None,
        help=(
            "Dry-run a user-prepared local data drop with {family}/{images,masks,audit} layout against "
            "--data-root, reporting copy plan, conflicts, unsupported files, and path escapes without writing."
        ),
    )
    parser.add_argument(
        "--refresh-manifest-file-claims",
        action="store_true",
        help="Refresh dataset_manifest.json files[] entries with local image/mask paths and SHA-256 values; no downloads.",
    )
    parser.add_argument(
        "--check-manifest-file-claims",
        action="store_true",
        help="Check whether dataset_manifest.json files[] entries match local files without writing.",
    )
    parser.add_argument(
        "--refresh-source-audit-artifact-claims",
        action="store_true",
        help="Refresh source_audit artifact SHA-256 fields from local audit files; no downloads and no review flags changed.",
    )
    parser.add_argument(
        "--check-source-audit-artifact-claims",
        action="store_true",
        help="Check source_audit artifact SHA-256 fields without writing.",
    )
    parser.add_argument("--classes", type=int, default=4, help="Default class count for images without masks")
    parser.add_argument("--mu", type=float, default=8.0, help="ROF fidelity parameter")
    parser.add_argument("--rof-iterations", type=int, default=160, help="ROF solver iteration limit")
    parser.add_argument("--trof-iterations", type=int, default=20, help="T-ROF threshold iteration limit")
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR), help="Directory for local runner comparison figures")
    parser.add_argument(
        "--family-summary-output",
        default=None,
        help="CSV output path for family_summaries when --run is used",
    )
    parser.add_argument(
        "--image-evidence-output",
        default=None,
        help="CSV output path for one row per local runner image when --run is used",
    )
    parser.add_argument(
        "--dashboard-candidate-output",
        default=None,
        help="Optional JSON output path for a gated dashboard promotion candidate when --run is used",
    )
    parser.add_argument(
        "--dashboard-static-assets-output",
        default=None,
        help=(
            "Optional JSON manifest describing dashboard static figure assets derived from a promotable "
            "paper-like source summary."
        ),
    )
    parser.add_argument(
        "--copy-dashboard-static-assets",
        action="store_true",
        help=(
            "Copy promotable paper-like figure PNGs into docs/assets/repro/iterated_rof_paper_like. "
            "Requires --dashboard-static-assets-output and a promotable dashboard candidate."
        ),
    )
    parser.add_argument(
        "--promotion-audit-output",
        default=None,
        help="Optional JSON output path for a compact paper-like promotion audit.",
    )
    parser.add_argument(
        "--data-gap-output",
        default=None,
        help="Optional JSON output path for a compact local-data gap checklist.",
    )
    parser.add_argument(
        "--data-package-review-output",
        default=None,
        help=(
            "Optional JSON output path for a no-download review of local image/mask/audit files, "
            "manifest manual fields, and claim refresh/check status."
        ),
    )
    parser.add_argument(
        "--data-drop-review-output",
        default=None,
        help="Optional JSON output path for --review-data-drop dry-run results.",
    )
    parser.add_argument(
        "--strict-paper-like",
        action="store_true",
        help="Exit non-zero unless paper_like_gate passes and the dashboard promotion candidate is promotable.",
    )
    parser.add_argument(
        "--strict-data-ready",
        action="store_true",
        help="Exit non-zero unless local data, masks, manifest claims, and preflight checks are ready for the local runner.",
    )
    args = parser.parse_args(argv)

    data_layout_preparation = None
    if args.prepare_data_layout:
        data_layout_preparation = prepare_data_layout(Path(args.data_root))
    data_drop_ingest = None
    if args.ingest_data_drop:
        data_drop_ingest = ingest_data_drop(Path(args.ingest_data_drop), Path(args.data_root))
    data_drop_review = None
    if args.review_data_drop:
        data_drop_review = review_data_drop(Path(args.review_data_drop), Path(args.data_root))
    manifest_file_claim_refresh = None
    if args.refresh_manifest_file_claims:
        manifest_file_claim_refresh = refresh_manifest_file_claims(Path(args.data_root))
    manifest_file_claim_check = None
    if args.check_manifest_file_claims:
        manifest_file_claim_check = check_manifest_file_claims(Path(args.data_root))
    source_audit_artifact_claim_refresh = None
    if args.refresh_source_audit_artifact_claims:
        source_audit_artifact_claim_refresh = refresh_source_audit_artifact_claims(Path(args.data_root))
    source_audit_artifact_claim_check = None
    if args.check_source_audit_artifact_claims:
        source_audit_artifact_claim_check = check_source_audit_artifact_claims(Path(args.data_root))

    strict_local_data_preflight_failed = False
    if args.run and (args.strict_data_ready or args.strict_paper_like):
        preflight_report = build_readiness_report(Path(args.data_root))
        strict_local_data_preflight_failed = not preflight_report.get("data_gap_checklist", {}).get(
            "ready_for_local_runner"
        )
    else:
        preflight_report = None

    if args.verify_summary:
        report = verify_saved_summary(Path(args.verify_summary))
        default_output = SUMMARY_VERIFICATION_PATH
    elif args.run and not strict_local_data_preflight_failed:
        report = run_local_dataset(
            Path(args.data_root),
            mu=args.mu,
            default_classes=args.classes,
            rof_n_iter=args.rof_iterations,
            trof_max_iter=args.trof_iterations,
            figure_dir=Path(args.figure_dir),
        )
        default_output = RUN_SUMMARY_PATH
    elif strict_local_data_preflight_failed:
        report = preflight_report
        default_output = REPORT_PATH
    else:
        report = build_readiness_report(Path(args.data_root))
        default_output = REPORT_PATH
    if data_layout_preparation is not None:
        report["data_layout_preparation"] = data_layout_preparation
    if data_drop_ingest is not None:
        report["data_drop_ingest"] = data_drop_ingest
    if data_drop_review is not None:
        report["data_drop_review"] = data_drop_review
    if manifest_file_claim_refresh is not None:
        report["manifest_file_claim_refresh"] = manifest_file_claim_refresh
    if manifest_file_claim_check is not None:
        report["manifest_file_claim_check"] = manifest_file_claim_check
    if source_audit_artifact_claim_refresh is not None:
        report["source_audit_artifact_claim_refresh"] = source_audit_artifact_claim_refresh
    if source_audit_artifact_claim_check is not None:
        report["source_audit_artifact_claim_check"] = source_audit_artifact_claim_check

    run_executed = args.run and not strict_local_data_preflight_failed

    primary_report_requested = (
        args.output is not None
        or run_executed
        or strict_local_data_preflight_failed
        or args.verify_summary
        or args.sources
        or args.prepare_data_layout
        or args.ingest_data_drop
        or args.review_data_drop and not args.data_drop_review_output
        or args.refresh_manifest_file_claims
        or args.check_manifest_file_claims
        or args.refresh_source_audit_artifact_claims
        or args.check_source_audit_artifact_claims
        or args.dashboard_static_assets_output
        or not (args.data_gap_output or args.data_package_review_output or args.data_drop_review_output)
    )
    output = None
    if primary_report_requested:
        output_path = Path(args.output) if args.output else default_output
        output = write_report(report, output_path)
        if args.verify_summary:
            report["summary_verification"] = _summary_verification(report, output)
            output = write_report(report, output_path)
        print(f"wrote {output}")
    if data_layout_preparation is not None:
        created = [item for item in data_layout_preparation["directories"] if item["status"] == "created"]
        print(f"prepared data layout: {len(created)} directories created")
        print(f"manifest: {data_layout_preparation['manifest']['status']}")
    if data_drop_ingest is not None:
        print(
            "ingested data drop: "
            f"{data_drop_ingest['status']} "
            f"(copied={data_drop_ingest['copied_file_count']}, "
            f"current={data_drop_ingest['current_file_count']}, "
            f"conflicts={data_drop_ingest['conflict_file_count']})"
        )
    if data_drop_review is not None:
        print(
            "reviewed data drop: "
            f"{data_drop_review['status']} "
            f"(would_copy={data_drop_review['copyable_file_count']}, "
            f"current={data_drop_review['current_file_count']}, "
            f"conflicts={data_drop_review['conflict_file_count']})"
        )
    if manifest_file_claim_refresh is not None:
        print(f"refreshed manifest file claims: {manifest_file_claim_refresh['status']}")
    if manifest_file_claim_check is not None:
        print(f"checked manifest file claims: {manifest_file_claim_check['status']}")
    if source_audit_artifact_claim_refresh is not None:
        print(f"refreshed source audit artifact claims: {source_audit_artifact_claim_refresh['status']}")
    if source_audit_artifact_claim_check is not None:
        print(f"checked source audit artifact claims: {source_audit_artifact_claim_check['status']}")
    if run_executed:
        family_summary_output = write_family_summary_csv(
            report,
            Path(args.family_summary_output) if args.family_summary_output else FAMILY_SUMMARY_CSV_PATH,
        )
        print(f"wrote family summary {family_summary_output}")
        image_evidence_output = write_image_evidence_csv(
            report,
            Path(args.image_evidence_output) if args.image_evidence_output else IMAGE_EVIDENCE_CSV_PATH,
        )
        print(f"wrote image evidence {image_evidence_output}")
        if args.dashboard_candidate_output:
            write_dashboard_candidate(report, Path(args.dashboard_candidate_output), source_summary_path=output)
            print(f"wrote dashboard candidate {args.dashboard_candidate_output}")
    elif args.verify_summary and args.dashboard_candidate_output:
        write_dashboard_candidate(report, Path(args.dashboard_candidate_output), source_summary_path=output)
        print(f"wrote dashboard candidate {args.dashboard_candidate_output}")
    dashboard_static_asset_manifest = None
    if args.dashboard_static_assets_output:
        dashboard_static_asset_manifest = write_dashboard_static_asset_manifest(
            report,
            Path(args.dashboard_static_assets_output),
            source_summary_path=output,
            copy_assets=args.copy_dashboard_static_assets,
        )
        print(f"wrote dashboard static asset manifest {args.dashboard_static_assets_output}")
        if args.copy_dashboard_static_assets:
            print(
                "dashboard static assets: "
                f"{'copied' if dashboard_static_asset_manifest['copy_performed'] else 'blocked'}"
            )
    if args.promotion_audit_output:
        write_promotion_audit(report, Path(args.promotion_audit_output), source_summary_path=output)
        print(f"wrote promotion audit {args.promotion_audit_output}")
    if args.data_gap_output:
        write_data_gap_checklist(report, Path(args.data_gap_output))
        print(f"wrote data gap checklist {args.data_gap_output}")
    if args.data_package_review_output:
        write_data_package_review(Path(args.data_root), Path(args.data_package_review_output))
        print(f"wrote data package review {args.data_package_review_output}")
    if args.data_drop_review_output and data_drop_review is not None:
        path = Path(args.data_drop_review_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data_drop_review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote data drop review {args.data_drop_review_output}")
    print(f"status: {report['status']}")
    if run_executed:
        print(f"images: {report['image_count']}")
        print(f"quantitative images: {report['quantitative_image_count']}")
    if report.get("paper_like_gate"):
        gate = report["paper_like_gate"]
        print(f"paper-like gate: {'passed' if gate['passed'] else 'blocked'}")
    if report["blockers"]:
        print("blockers:")
        for blocker in report["blockers"]:
            print(f"- {blocker}")
    if report.get("claim_blockers"):
        print("claim blockers:")
        for blocker in report["claim_blockers"]:
            print(f"- {blocker}")
    if report.get("paper_like_gate") and report["paper_like_gate"]["reasons"]:
        print("paper-like gate reasons:")
        for reason in report["paper_like_gate"]["reasons"]:
            print(f"- {reason}")
    if args.sources:
        recommended_sources = report.get("recommended_sources")
        if recommended_sources is None:
            recommended_sources = build_readiness_report(Path(args.data_root))["recommended_sources"]
        print("recommended sources:")
        for family, sources in recommended_sources.items():
            print(f"- {family}:")
            for source in sources:
                print(f"  - {source['source_id']}: {source['download_url']}")
    if args.verify_summary and not build_dashboard_candidate(report, source_summary_path=output)["can_promote"]:
        print("summary verification failed")
        return 1
    if args.check_manifest_file_claims and manifest_file_claim_check.get("status") != "current":
        print("manifest file claim check failed")
        return 1
    if (
        args.check_source_audit_artifact_claims
        and source_audit_artifact_claim_check.get("status") != "current"
    ):
        print("source audit artifact claim check failed")
        return 1
    if args.copy_dashboard_static_assets and (
        not dashboard_static_asset_manifest or not dashboard_static_asset_manifest.get("copy_performed")
    ):
        print("dashboard static asset copy failed")
        return 1
    if args.ingest_data_drop and data_drop_ingest.get("status") == "conflict":
        print("data drop ingest failed")
        return 1
    if args.review_data_drop and data_drop_review.get("status") == "conflict":
        print("data drop review failed")
        return 1
    if (args.strict_data_ready or args.strict_paper_like) and not report.get("data_gap_checklist", {}).get("ready_for_local_runner"):
        print("strict data-ready check failed")
        return 1
    if args.strict_paper_like and not report.get("paper_like_gate", {}).get("passed"):
        print("strict paper-like gate failed")
        return 1
    if args.strict_paper_like and not build_dashboard_candidate(report, source_summary_path=output)["can_promote"]:
        print("strict paper-like promotion check failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
