import json
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from matplotlib import image as mpimg


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

import iterated_rof_paper_like  # noqa: E402


class IteratedRofPaperLikeScaffoldTests(unittest.TestCase):
    SOURCE_IDS = {
        "cartoon": "bsds500",
        "texture": "prague-texture",
        "medical": "brainweb",
    }
    SOURCE_URLS = {
        "cartoon": "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/",
        "texture": "https://mosaic.utia.cas.cz/",
        "medical": "https://brainweb.bic.mni.mcgill.ca/brainweb/",
    }
    SOURCE_AUDIT_BASE = {
        "downloaded_at": "2026-06-09",
        "source_artifact_path": "reproduce/tests/fixtures/source_audit/source-artifact.txt",
        "source_artifact_sha256": "50caa94ef3bbf1625d06b7b8ede0c4c143adfd0a308cd198e6d99a9bb7534e70",
        "license_snapshot_path": "reproduce/tests/fixtures/source_audit/license-snapshot.txt",
        "license_snapshot_sha256": "e14c2299dd7c7f9e0fcf74b316fd97c60457d08a0776197a929612adacc60dae",
        "conversion_notes": "Converted reviewed local source files into canonical PNG image/mask pairs.",
        "local_file_mapping_reviewed": True,
    }

    def _source_audit_artifact_text(self, family, kind):
        return "\n".join(
            [
                f"teacherZ Iterated ROF {kind} review record for {family}.",
                f"source_id={self.SOURCE_IDS[family]}",
                f"source_url={self.SOURCE_URLS[family]}",
                "review_date=2026-06-09",
                "reviewer_note=Local-only artifact retained to document the exact source page, terms page, and conversion mapping used for this data family.",
                "conversion_note=Images and masks are mapped by same relative path after source review; raw archives remain outside the repository unless redistribution is approved.",
                "",
            ]
        )

    def _source_audit(self, family, root=None, namespace=None):
        if root is not None:
            audit_root = Path(root) / family / "audit"
            if namespace:
                audit_root = audit_root / namespace
            audit_root.mkdir(parents=True, exist_ok=True)
            source_artifact = audit_root / "source-artifact.txt"
            license_snapshot = audit_root / "license-snapshot.txt"
            source_artifact.write_text(
                self._source_audit_artifact_text(family, "source artifact"),
                encoding="utf-8",
            )
            license_snapshot.write_text(
                self._source_audit_artifact_text(family, "license snapshot"),
                encoding="utf-8",
            )
            return {
                "source_url": self.SOURCE_URLS[family],
                "downloaded_at": self.SOURCE_AUDIT_BASE["downloaded_at"],
                "source_artifact_path": str(source_artifact),
                "source_artifact_sha256": iterated_rof_paper_like._file_evidence(source_artifact)["sha256"],
                "license_snapshot_path": str(license_snapshot),
                "license_snapshot_sha256": iterated_rof_paper_like._file_evidence(license_snapshot)["sha256"],
                "conversion_notes": self.SOURCE_AUDIT_BASE["conversion_notes"],
                "local_file_mapping_reviewed": self.SOURCE_AUDIT_BASE["local_file_mapping_reviewed"],
            }
        return {
            "source_url": self.SOURCE_URLS[family],
            **self.SOURCE_AUDIT_BASE,
        }

    def _write_png(self, path, values):
        path.parent.mkdir(parents=True, exist_ok=True)
        mpimg.imsave(path, np.asarray(values, dtype=float), cmap="gray", vmin=0.0, vmax=1.0)

    def _write_license_reviewed_manifest(self, root, include_files=True):
        payload = {
            "families": {
                family: {
                    "source_id": source_id,
                    "source_name": source_id,
                    "license_reviewed": True,
                    "license_note": "Reviewed local dataset terms for non-redistributed research use.",
                    "citation": "Recorded public dataset citation for local review.",
                    "provenance_reviewed": True,
                    "provenance_note": "File origin recorded from the named source and kept local for review.",
                    "synthetic_fixture": False,
                    "source_audit": self._source_audit(family, root),
                    "files": [
                        {
                            "image": image_path.relative_to(Path(root) / family / "images").as_posix(),
                            "sha256": iterated_rof_paper_like._file_evidence(image_path)["sha256"],
                            "mask": image_path.relative_to(Path(root) / family / "images").as_posix()
                            if (Path(root) / family / "masks" / image_path.relative_to(Path(root) / family / "images")).exists()
                            else None,
                            "mask_sha256": iterated_rof_paper_like._file_evidence(
                                Path(root) / family / "masks" / image_path.relative_to(Path(root) / family / "images")
                            )["sha256"]
                            if (Path(root) / family / "masks" / image_path.relative_to(Path(root) / family / "images")).exists()
                            else None,
                        }
                        for image_path in sorted((Path(root) / family / "images").rglob("*"))
                        if image_path.is_file() and image_path.suffix.lower() in iterated_rof_paper_like.IMAGE_EXTENSIONS
                    ] if include_files else [],
                }
                for family, source_id in self.SOURCE_IDS.items()
            }
        }
        (Path(root) / "dataset_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    def _patch_iterated_rof_runtime_paths(self, root):
        root = Path(root)
        replacements = {
            "DATA_ROOT": root / "reproduce" / "data" / "iterated_rof",
            "RESULTS_DIR": root / "reproduce" / "results",
            "FIGURE_DIR": root / "reproduce" / "results" / "figures" / "iterated_rof_paper_like",
            "DASHBOARD_REPRO_ASSET_ROOT": root / "docs" / "assets" / "repro",
        }
        replacements.update(
            {
                "REPORT_PATH": replacements["RESULTS_DIR"] / "iterated_rof_paper_like_readiness.json",
                "RUN_SUMMARY_PATH": replacements["RESULTS_DIR"] / "iterated_rof_paper_like_summary.json",
                "FAMILY_SUMMARY_CSV_PATH": replacements["RESULTS_DIR"] / "iterated_rof_paper_like_family_summary.csv",
                "IMAGE_EVIDENCE_CSV_PATH": replacements["RESULTS_DIR"] / "iterated_rof_paper_like_image_evidence.csv",
                "SUMMARY_VERIFICATION_PATH": replacements["RESULTS_DIR"] / "iterated_rof_paper_like_summary_verification.json",
                "DATASET_MANIFEST_TEMPLATE_PATH": replacements["DATA_ROOT"] / "dataset_manifest.template.json",
            }
        )
        originals = {
            name: getattr(iterated_rof_paper_like, name)
            for name in replacements
        }
        for name, value in replacements.items():
            setattr(iterated_rof_paper_like, name, value)

        def restore_paths():
            for name, value in originals.items():
                setattr(iterated_rof_paper_like, name, value)

        self.addCleanup(restore_paths)
        return replacements

    def _passed_gate_report(self):
        family_summaries = []
        images = []
        families = []
        for family in iterated_rof_paper_like.DATA_FAMILIES:
            image_path = f"reproduce/data/iterated_rof/{family}/images/sample.png"
            mask_path = f"reproduce/data/iterated_rof/{family}/masks/sample.png"
            families.append(
                {
                    "family": family,
                    "description": iterated_rof_paper_like.DATA_FAMILIES[family],
                    "image_count": 1,
                    "mask_count": 1,
                    "matched_mask_count": 1,
                    "status": "ready_quantitative",
                }
            )
            source_claim = {
                "manifest_status": "present",
                "manifest_path": "reproduce/data/iterated_rof/dataset_manifest.json",
                "claim_scope": "file",
                "image": "sample.png",
                "mask": "sample.png",
                "source_id": self.SOURCE_IDS[family],
                "source_name": self.SOURCE_IDS[family],
                "license_reviewed": True,
                "license_note": "reviewed dataset license",
                "citation": "recorded dataset citation",
                "provenance_reviewed": True,
                "provenance_note": "recorded dataset provenance",
                "synthetic_fixture": False,
                "source_audit": self._source_audit(family),
                "sha256": "a" * 64,
                "mask_sha256": "b" * 64,
            }
            family_summaries.append(
                {
                    "family": family,
                    "status": "completed_quantitative",
                    "image_count": 1,
                    "mask_count": 1,
                    "matched_mask_count": 1,
                    "completed_image_count": 1,
                    "failed_image_count": 0,
                    "quantitative_image_count": 1,
                    "qualitative_image_count": 0,
                    "metrics_mean": {"clustering_accuracy": 1.0, "dice": 1.0},
                    "baseline_metrics_mean": {
                        "raw_kmeans": {"clustering_accuracy": 1.0},
                        "multi_otsu": {"clustering_accuracy": 1.0},
                    },
                    "figure_paths": [f"reproduce/results/figures/iterated_rof_paper_like/{family}.png"],
                    "source_claims": [source_claim],
                    "errors": [],
                }
            )
            images.append(
                {
                    "family": family,
                    "status": "completed",
                    "qualitative_only": False,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "metrics": {"clustering_accuracy": 1.0, "dice": 1.0},
                    "baselines": {
                        "raw_kmeans": {"metrics": {"clustering_accuracy": 1.0}},
                        "multi_otsu": {"metrics": {"clustering_accuracy": 1.0}},
                    },
                    "figure_path": f"reproduce/results/figures/iterated_rof_paper_like/{family}.png",
                    "figure_file": {"sha256": "c" * 64, "size_bytes": 123},
                    "figure_panels": [
                        "input",
                        "mask",
                        "ROF",
                        "T-ROF",
                        "raw K-means",
                        "multi-Otsu",
                        "T-ROF error",
                        "T-ROF vs Otsu",
                    ],
                    "image_file": {"sha256": "a" * 64},
                    "mask_file": {"sha256": "b" * 64},
                    "source_claim": source_claim,
                }
            )
        return {
            "paper_id": "iterated-rof",
            "target_level": "paper-like",
            "readiness_status": "ready_for_paper_like_runner",
            "status": "completed_local_runner",
            "image_count": 3,
            "completed_image_count": 3,
            "quantitative_image_count": 3,
            "images": images,
            "families": families,
            "blockers": [],
            "claim_blockers": [],
            "family_summaries": family_summaries,
            "local_dataset_manifest": {
                "status": "present",
                "path": "reproduce/data/iterated_rof/dataset_manifest.json",
                "families": {},
            },
            "dataset_fingerprint": {
                "algorithm": "sha256",
                "file_count": 6,
                "sha256": "f" * 64,
            },
            "paper_like_gate": {
                "passed": True,
                "dashboard_level": "paper-like",
                "checked_requirements": [],
                "reasons": [],
            },
            "claim_boundary": "test passed report",
            "run_protocol": iterated_rof_paper_like._run_protocol(
                8.0,
                4,
                8,
                4,
                iterated_rof_paper_like.FIGURE_DIR,
            ),
        }

    def _complete_image_result(self, family, image_path=None, mask_path=None, figure_path=None):
        image_path = image_path or f"reproduce/data/iterated_rof/{family}/images/sample.png"
        mask_path = mask_path or f"reproduce/data/iterated_rof/{family}/masks/sample.png"
        figure_path = figure_path or f"reproduce/results/figures/{family}.png"
        return {
            "family": family,
            "status": "completed",
            "qualitative_only": False,
            "image_path": image_path,
            "mask_path": mask_path,
                    "metrics": {"clustering_accuracy": 1.0},
                    "baselines": {
                        "raw_kmeans": {
                            "method": "simple_kmeans_on_raw_grayscale",
                            "metrics": {"clustering_accuracy": 1.0},
                        },
                        "multi_otsu": {
                            "method": "skimage_multiotsu",
                            "thresholds": [0.25, 0.5, 0.75],
                            "metrics": {"clustering_accuracy": 1.0},
                        },
                    },
                    "figure_path": figure_path,
                    "figure_file": {"sha256": "c" * 64, "size_bytes": 123},
            "figure_panels": [
                "input",
                "mask",
                "ROF",
                "T-ROF",
                "raw K-means",
                "multi-Otsu",
                "T-ROF error",
                "T-ROF vs Otsu",
            ],
            "image_file": {"sha256": "a" * 64, "size_bytes": 456},
            "mask_file": {"sha256": "b" * 64, "size_bytes": 456},
            "solver": "sat_rof_trof.rof_chambolle_pock + sat_rof_trof.run_trof_thresholds",
            "parameters": {"mu": 8.0, "rof_n_iter": 8, "trof_max_iter": 4},
            "thresholds": [0.25, 0.5, 0.75],
            "threshold_iterations": 3,
            "rof_iterations": 8,
            "rof_final_residual": 0.001,
            "n_classes": 4,
            "source_claim": {
                "manifest_status": "present",
                "manifest_path": "reproduce/data/iterated_rof/dataset_manifest.json",
                "claim_scope": "file",
                "image": "sample.png",
                "mask": "sample.png",
                "source_id": self.SOURCE_IDS[family],
                "source_name": self.SOURCE_IDS[family],
                "license_reviewed": True,
                "citation": "recorded dataset citation",
                "license_note": "reviewed dataset license",
                "provenance_reviewed": True,
                "provenance_note": "recorded dataset provenance",
                "synthetic_fixture": False,
                "source_audit": self._source_audit(family),
                "sha256": "a" * 64,
                "mask_sha256": "b" * 64,
            },
        }

    def _passed_gate_report_with_matching_local_files(self, root, patch_runtime_paths=True):
        paths = (
            self._patch_iterated_rof_runtime_paths(root)
            if patch_runtime_paths
            else {
                "DATA_ROOT": iterated_rof_paper_like.DATA_ROOT,
                "RESULTS_DIR": iterated_rof_paper_like.RESULTS_DIR,
                "FIGURE_DIR": iterated_rof_paper_like.FIGURE_DIR,
            }
        )
        summary = self._passed_gate_report()
        token = f"_test_{Path(root).name}"
        manifest_path = paths["DATA_ROOT"] / "dataset_manifest.json"
        original_manifest = manifest_path.read_bytes() if manifest_path.exists() else None

        def restore_manifest():
            if original_manifest is None:
                manifest_path.unlink(missing_ok=True)
            else:
                manifest_path.write_bytes(original_manifest)

        self.addCleanup(restore_manifest)
        self.addCleanup(
            shutil.rmtree,
            iterated_rof_paper_like.FIGURE_DIR / token,
            ignore_errors=True,
        )
        image_results = []
        for family in iterated_rof_paper_like.DATA_FAMILIES:
            image_path = iterated_rof_paper_like.DATA_ROOT / family / "images" / token / "sample.png"
            mask_path = iterated_rof_paper_like.DATA_ROOT / family / "masks" / token / "sample.png"
            figure_path = iterated_rof_paper_like.FIGURE_DIR / token / f"{family}.png"
            self.addCleanup(
                shutil.rmtree,
                iterated_rof_paper_like.DATA_ROOT / family / "images" / token,
                ignore_errors=True,
            )
            self.addCleanup(
                shutil.rmtree,
                iterated_rof_paper_like.DATA_ROOT / family / "masks" / token,
                ignore_errors=True,
            )
            self.addCleanup(
                shutil.rmtree,
                iterated_rof_paper_like.DATA_ROOT / family / "audit" / token,
                ignore_errors=True,
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            figure_path.parent.mkdir(parents=True, exist_ok=True)
            axis = np.linspace(0.0, 1.0, 64)
            grid_x, grid_y = np.meshgrid(axis, axis)
            image_values = (0.65 * grid_x + 0.35 * grid_y + 0.05 * len(image_results)) % 1.0
            mask_values = (image_values > 0.5).astype(float)
            self._write_png(image_path, image_values)
            self._write_png(mask_path, mask_values)
            self._write_png(figure_path, np.linspace(0.0, 1.0, 256).reshape(16, 16))

            item = self._complete_image_result(
                family,
                image_path=iterated_rof_paper_like._display_path(image_path),
                mask_path=iterated_rof_paper_like._display_path(mask_path),
                figure_path=iterated_rof_paper_like._display_path(figure_path),
            )
            item["image_file"] = iterated_rof_paper_like._file_evidence(image_path)
            item["mask_file"] = iterated_rof_paper_like._file_evidence(mask_path)
            item["figure_file"] = iterated_rof_paper_like._file_evidence(figure_path)
            item["source_claim"]["image"] = f"{token}/sample.png"
            item["source_claim"]["mask"] = f"{token}/sample.png"
            item["source_claim"]["manifest_path"] = iterated_rof_paper_like._display_path(manifest_path)
            item["source_claim"]["sha256"] = item["image_file"]["sha256"]
            item["source_claim"]["mask_sha256"] = item["mask_file"]["sha256"]
            item["source_claim"]["source_audit"] = self._source_audit(
                family,
                iterated_rof_paper_like.DATA_ROOT,
                token,
            )
            item.update(iterated_rof_paper_like._write_figure_evidence_sidecar(item))
            image_results.append(item)

        manifest_payload = {
            "families": {
                family: {
                    "source_id": self.SOURCE_IDS[family],
                    "source_name": self.SOURCE_IDS[family],
                    "license_reviewed": True,
                    "license_note": "reviewed dataset license",
                    "citation": "recorded dataset citation",
                    "provenance_reviewed": True,
                    "provenance_note": "recorded dataset provenance",
                    "synthetic_fixture": False,
                    "source_audit": self._source_audit(family, iterated_rof_paper_like.DATA_ROOT, token),
                    "files": [
                        {
                            "image": f"{token}/sample.png",
                            "sha256": next(
                                item for item in image_results if item["family"] == family
                            )["image_file"]["sha256"],
                            "mask": f"{token}/sample.png",
                            "mask_sha256": next(
                                item for item in image_results if item["family"] == family
                            )["mask_file"]["sha256"],
                        }
                    ],
                }
                for family in iterated_rof_paper_like.DATA_FAMILIES
            }
        }
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

        summary["images"] = image_results
        summary["data_root"] = iterated_rof_paper_like._display_path(iterated_rof_paper_like.DATA_ROOT)
        summary["family_summaries"] = iterated_rof_paper_like._family_summaries(summary["families"], image_results)
        summary["dataset_fingerprint"] = iterated_rof_paper_like._dataset_fingerprint_from_image_results(image_results)
        summary["local_dataset_manifest"] = iterated_rof_paper_like.load_local_dataset_manifest(
            iterated_rof_paper_like.DATA_ROOT
        )
        summary["paper_like_gate"] = iterated_rof_paper_like._paper_like_gate(
            "ready_for_paper_like_runner",
            [],
            [],
            image_results,
            data_root=iterated_rof_paper_like.DATA_ROOT,
            dataset_fingerprint=summary["dataset_fingerprint"],
        )
        return summary

    def test_missing_data_is_reported_as_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = iterated_rof_paper_like.build_readiness_report(Path(tmp))

        self.assertEqual(report["status"], "blocked_missing_data")
        self.assertEqual(len(report["families"]), 3)
        self.assertEqual(set(report["recommended_sources"]), {"cartoon", "texture", "medical"})
        self.assertEqual(report["recommended_sources"]["texture"][0]["source_id"], "prague-texture")
        self.assertEqual(report["recommended_sources"]["medical"][0]["source_id"], "brainweb")
        self.assertEqual(report["recommended_sources"]["cartoon"][0]["source_id"], "bsds500")
        self.assertTrue(report["blockers"])
        self.assertEqual(report["current_dashboard_level"], "partial")
        self.assertFalse(report["paper_like_gate"]["passed"])
        self.assertIn(
            "all data families have completed quantitative local runner outputs",
            report["paper_like_gate"]["checked_requirements"],
        )
        checklist = {item["id"]: item for item in report["paper_like_gate"]["checklist"]}
        self.assertEqual(
            set(checklist),
            {"canonical_data_root", "readiness_clean", "runner_outputs", "source_audit", "output_evidence"},
        )
        self.assertFalse(checklist["canonical_data_root"]["passed"])
        self.assertFalse(checklist["runner_outputs"]["passed"])
        self.assertTrue(checklist["runner_outputs"]["reasons"])
        gap = report["data_gap_checklist"]
        self.assertFalse(gap["ready_for_paper_like"])
        self.assertEqual(gap["remaining_family_count"], 3)
        self.assertEqual(gap["manifest"]["status"], "missing")
        self.assertEqual([item["family"] for item in gap["families"]], ["cartoon", "texture", "medical"])
        texture_gap = next(item for item in gap["families"] if item["family"] == "texture")
        self.assertIn("add at least one nontrivial local image", texture_gap["missing"])
        self.assertIn("add matching mask/label image", texture_gap["missing"])
        self.assertIn(
            "refresh source_audit artifact SHA-256 claims after audit files are placed",
            texture_gap["next_actions"],
        )
        self.assertEqual(texture_gap["paths"]["images"], "texture/images")
        self.assertEqual(texture_gap["primary_source"]["source_id"], "prague-texture")
        self.assertEqual(texture_gap["acquisition_plan"]["recommended_source_id"], "prague-texture")
        self.assertEqual(texture_gap["acquisition_plan"]["download_policy"], "manual_or_site_form")
        self.assertEqual(texture_gap["acquisition_plan"]["target_paths"]["images"], "texture/images")
        self.assertEqual(texture_gap["acquisition_plan"]["target_paths"]["source_audit"], "texture/audit/source-artifact")
        self.assertIn("texture/images/<source_id>/<case>.png", texture_gap["acquisition_plan"]["final_naming_example"])
        self.assertIn("ground-truth", texture_gap["acquisition_plan"]["mask_mapping_rule"])
        self.assertTrue(texture_gap["acquisition_plan"]["conversion_checklist"])
        self.assertIn("source_artifact_sha256", texture_gap["acquisition_plan"]["required_manifest_fields"])
        self.assertIn(
            "python3 reproduce/experiments/iterated_rof_paper_like.py --refresh-manifest-file-claims",
            texture_gap["acquisition_plan"]["post_download_commands"],
        )
        self.assertIn(
            "python3 reproduce/experiments/iterated_rof_paper_like.py --refresh-source-audit-artifact-claims",
            texture_gap["acquisition_plan"]["post_download_commands"],
        )
        self.assertIn(
            "python3 reproduce/experiments/iterated_rof_paper_like.py --strict-data-ready",
            texture_gap["acquisition_plan"]["post_download_commands"],
        )
        self.assertIn(
            "python3 reproduce/experiments/iterated_rof_paper_like.py --run --strict-paper-like",
            texture_gap["acquisition_plan"]["post_download_commands"],
        )
        self.assertIn("copy dataset_manifest.template.json to dataset_manifest.json", gap["global_next_actions"][0])
        self.assertIn(
            "run --refresh-source-audit-artifact-claims after placing audit files",
            gap["global_next_actions"],
        )
        self.assertIn(
            "run --check-source-audit-artifact-claims before strict-data-ready",
            gap["global_next_actions"],
        )

    def test_source_manifest_has_download_boundaries(self):
        sources = iterated_rof_paper_like.load_source_manifest()

        self.assertEqual(set(sources), {"cartoon", "texture", "medical"})
        self.assertEqual(iterated_rof_paper_like._source_manifest_schema_blockers(sources), [])
        for family_sources in sources.values():
            self.assertGreaterEqual(len(family_sources), 1)
            for source in family_sources:
                self.assertIn("url", source)
                self.assertIn("download_policy", source)
                self.assertIn("license_note", source)

    def test_source_manifest_schema_rejects_invalid_registry_entries(self):
        sources = iterated_rof_paper_like.load_source_manifest()
        invalid_sources = json.loads(json.dumps(sources))
        invalid_sources["cartoon"][0]["target_family"] = "texture"
        invalid_sources["texture"][0]["priority"] = "1"
        invalid_sources["medical"][0]["source_id"] = invalid_sources["cartoon"][0]["source_id"]
        invalid_sources["medical"][0]["download_url"] = ""

        blockers = iterated_rof_paper_like._source_manifest_schema_blockers(invalid_sources)

        self.assertTrue(any("target_family mismatch" in item for item in blockers))
        self.assertTrue(any("priority must be an integer" in item for item in blockers))
        self.assertTrue(any("duplicate source_id" in item for item in blockers))
        self.assertTrue(any("missing download_url" in item for item in blockers))

    def test_cli_can_write_data_gap_checklist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "readiness.json"
            gap_path = root / "data_gap.json"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--output",
                    str(report_path),
                    "--data-gap-output",
                    str(gap_path),
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            gap = json.loads(gap_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(gap, report["data_gap_checklist"])
        self.assertEqual(gap["target_level"], "paper-like")
        self.assertFalse(gap["ready_for_paper_like"])
        self.assertEqual(gap["remaining_family_count"], 3)

    def test_cli_data_gap_output_only_does_not_write_default_readiness_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            default_report_path = root / "default_readiness.json"
            gap_path = root / "data_gap.json"
            original_report_path = iterated_rof_paper_like.REPORT_PATH
            iterated_rof_paper_like.REPORT_PATH = default_report_path
            try:
                exit_code = iterated_rof_paper_like.main(
                    [
                        "--data-root",
                        str(root),
                        "--data-gap-output",
                        str(gap_path),
                    ]
                )
            finally:
                iterated_rof_paper_like.REPORT_PATH = original_report_path

            gap = json.loads(gap_path.read_text(encoding="utf-8"))
            default_report_exists = default_report_path.exists()

        self.assertEqual(exit_code, 0)
        self.assertFalse(default_report_exists)
        self.assertEqual(gap["target_level"], "paper-like")
        self.assertEqual(gap["remaining_family_count"], 3)

    def test_write_data_gap_checklist_recomputes_instead_of_trusting_saved_checklist(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary["data_gap_checklist"] = iterated_rof_paper_like.build_data_gap_checklist(summary)
            self.assertTrue(summary["data_gap_checklist"]["ready_for_local_runner"])
            cartoon_image = next(item for item in summary["images"] if item["family"] == "cartoon")
            source_artifact = Path(cartoon_image["source_claim"]["source_audit"]["source_artifact_path"])
            source_artifact.unlink()
            gap_path = root / "data_gap.json"

            iterated_rof_paper_like.write_data_gap_checklist(summary, gap_path)
            gap = json.loads(gap_path.read_text(encoding="utf-8"))

        self.assertFalse(gap["ready_for_local_runner"])
        self.assertIn(
            "cartoon source_audit source_artifact file is missing",
            gap["data_ready_blockers"],
        )
        gap_by_family = {
            item["family"]: item
            for item in gap["families"]
        }
        self.assertEqual(gap_by_family["cartoon"]["source_audit"]["status"], "incomplete")
        self.assertEqual(
            gap_by_family["cartoon"]["source_audit"]["artifacts"]["source_artifact"]["path_status"],
            "missing_file",
        )

    def test_write_data_gap_checklist_reloads_current_manifest_instead_of_saved_manifest(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            self.assertEqual(summary["local_dataset_manifest"]["status"], "present")
            (iterated_rof_paper_like.DATA_ROOT / "dataset_manifest.json").unlink()
            gap_path = root / "data_gap.json"

            iterated_rof_paper_like.write_data_gap_checklist(summary, gap_path)
            gap = json.loads(gap_path.read_text(encoding="utf-8"))

        self.assertFalse(gap["ready_for_local_runner"])
        self.assertEqual(gap["manifest"]["status"], "missing")
        self.assertIn("local dataset manifest is not present", gap["data_ready_blockers"])
        gap_by_family = {
            item["family"]: item
            for item in gap["families"]
        }
        self.assertEqual(
            gap_by_family["cartoon"]["source_audit"]["status"],
            "missing_manifest",
        )

    def test_write_data_gap_checklist_recomputes_gate_instead_of_trusting_saved_gate(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            image = summary["images"][0]
            image["metrics"] = {}
            image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            gap_path = root / "data_gap.json"

            iterated_rof_paper_like.write_data_gap_checklist(summary, gap_path)
            gap = json.loads(gap_path.read_text(encoding="utf-8"))

        expected_reason = f"Missing T-ROF clustering_accuracy metric for: {image['image_path']}"
        self.assertFalse(gap["ready_for_paper_like"])
        self.assertFalse(gap["paper_like_gate"]["passed"])
        self.assertIn(expected_reason, gap["paper_like_gate"]["reasons"])

    def test_data_gap_checklist_does_not_report_dashboard_promotion_ready(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            gap = iterated_rof_paper_like.build_data_gap_checklist(summary)

        self.assertTrue(gap["paper_like_gate"]["passed"])
        self.assertTrue(gap["ready_for_paper_like_runner_outputs"])
        self.assertFalse(gap["ready_for_paper_like"])
        self.assertFalse(gap["ready_for_dashboard_promotion"])
        self.assertIn("dashboard candidate", gap["promotion_readiness_note"])

    def test_cli_strict_data_ready_fails_when_local_data_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "readiness.json"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--output",
                    str(report_path),
                    "--strict-data-ready",
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(report["data_gap_checklist"]["ready_for_local_runner"])
        self.assertEqual(report["data_gap_checklist"]["remaining_family_count"], 3)

    def test_cli_strict_data_ready_preflights_before_run_when_local_data_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "readiness.json"
            family_csv_path = root / "family_summary.csv"
            image_csv_path = root / "image_evidence.csv"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root / "data"),
                    "--run",
                    "--output",
                    str(report_path),
                    "--family-summary-output",
                    str(family_csv_path),
                    "--image-evidence-output",
                    str(image_csv_path),
                    "--strict-data-ready",
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "blocked_missing_data")
        self.assertFalse(report["data_gap_checklist"]["ready_for_local_runner"])
        self.assertFalse(family_csv_path.exists())
        self.assertFalse(image_csv_path.exists())

    def test_cli_strict_paper_like_preflights_data_ready_before_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "readiness.json"
            family_csv_path = root / "family_summary.csv"
            image_csv_path = root / "image_evidence.csv"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root / "data"),
                    "--run",
                    "--output",
                    str(report_path),
                    "--family-summary-output",
                    str(family_csv_path),
                    "--image-evidence-output",
                    str(image_csv_path),
                    "--strict-paper-like",
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "blocked_missing_data")
        self.assertFalse(report["data_gap_checklist"]["ready_for_local_runner"])
        self.assertFalse(family_csv_path.exists())
        self.assertFalse(image_csv_path.exists())

    def test_data_gap_gate_uses_readiness_status_fallback_from_report_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = iterated_rof_paper_like.build_readiness_report(Path(tmp))

        self.assertEqual(
            report["data_gap_checklist"]["paper_like_gate"]["reasons"],
            report["paper_like_gate"]["reasons"],
        )
        self.assertFalse(
            any("None" in reason for reason in report["data_gap_checklist"]["paper_like_gate"]["reasons"])
        )

    def test_cli_strict_data_ready_passes_for_reviewed_nontrivial_local_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis = np.linspace(0.0, 1.0, 64)
            grid_x, grid_y = np.meshgrid(axis, axis)
            for index, family in enumerate(iterated_rof_paper_like.DATA_FAMILIES):
                image = (0.7 * grid_x + 0.3 * grid_y + index * 0.07) % 1.0
                mask = (image > 0.5).astype(float)
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", mask)
            self._write_license_reviewed_manifest(root)
            report_path = root / "readiness.json"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--output",
                    str(report_path),
                    "--strict-data-ready",
                ]
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["status"], "ready_for_paper_like_runner")
        self.assertEqual(report["data_ready_status"], "ready_for_local_runner")
        self.assertTrue(report["data_gap_checklist"]["ready_for_local_runner"])
        self.assertFalse(report["paper_like_gate"]["passed"])
        self.assertIn("Local runner has not produced image outputs", report["paper_like_gate"]["reasons"])

    def test_ready_data_with_masks_allows_future_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                image_dir = root / family / "images"
                mask_dir = root / family / "masks"
                image_dir.mkdir(parents=True)
                mask_dir.mkdir(parents=True)
                (image_dir / "sample.png").write_bytes(b"not a real png; readiness only counts extensions")
                (mask_dir / "sample.png").write_bytes(b"not a real png; readiness only counts extensions")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertEqual(report["status"], "ready_for_paper_like_runner")
        self.assertEqual(report["data_ready_status"], "blocked_data_ready")
        self.assertFalse(report["blockers"])
        self.assertTrue(all(item["status"] == "ready_quantitative" for item in report["families"]))
        self.assertTrue(report["claim_blockers"])
        self.assertFalse(report["paper_like_gate"]["passed"])

    def test_data_gap_checklist_reports_preflight_content_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_png(root / "cartoon" / "images" / "tiny.png", np.zeros((4, 4)))
            self._write_png(root / "cartoon" / "masks" / "tiny.png", np.zeros((4, 4)))
            image_dir = root / "texture" / "images"
            mask_dir = root / "texture" / "masks"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            (image_dir / "broken.png").write_bytes(b"not a png")
            (mask_dir / "broken.png").write_bytes(b"not a png")
            self._write_png(root / "medical" / "images" / "mismatch.png", np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64))
            self._write_png(root / "medical" / "masks" / "mismatch.png", np.zeros((8, 8)))

            report = iterated_rof_paper_like.build_readiness_report(root)

        gap_by_family = {item["family"]: item for item in report["data_gap_checklist"]["families"]}
        self.assertTrue(
            any("input image is too small for paper-like evidence" in issue for issue in gap_by_family["cartoon"]["content_issues"])
        )
        self.assertTrue(
            any("input image is not decodable for paper-like evidence" in issue for issue in gap_by_family["texture"]["content_issues"])
        )
        self.assertTrue(
            any("mask is not decodable" in issue for issue in gap_by_family["texture"]["content_issues"])
        )
        self.assertTrue(
            any("mask shape" in issue and "does not match image shape" in issue for issue in gap_by_family["medical"]["content_issues"])
        )

    def test_local_manifest_clears_claim_blockers_after_license_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))

            missing_manifest = iterated_rof_paper_like.build_readiness_report(root)
            self._write_license_reviewed_manifest(root)
            reviewed_manifest = iterated_rof_paper_like.build_readiness_report(root)

        self.assertEqual(missing_manifest["status"], "ready_for_paper_like_runner")
        self.assertIn("Local dataset manifest missing", missing_manifest["claim_blockers"][0])
        self.assertEqual(reviewed_manifest["local_dataset_manifest"]["status"], "present")
        self.assertEqual(reviewed_manifest["claim_blockers"], [])
        self.assertFalse(reviewed_manifest["paper_like_gate"]["passed"])
        self.assertIn("Local runner has not produced image outputs", reviewed_manifest["paper_like_gate"]["reasons"])

    def test_local_manifest_requires_citation_and_license_note_for_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            source_ids = {
                "cartoon": "bsds500",
                "texture": "prague-texture",
                "medical": "brainweb",
            }
            payload = {
                "families": {
                    family: {
                        "source_id": source_id,
                        "source_name": source_id,
                        "license_reviewed": True,
                        "license_note": "",
                        "citation": "",
                    }
                    for family, source_id in source_ids.items()
                }
            }
            (root / "dataset_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertEqual(report["status"], "ready_for_paper_like_runner")
        self.assertIn("Local dataset manifest missing citation for: cartoon", report["claim_blockers"])
        self.assertIn("Local dataset manifest missing license_note for: cartoon", report["claim_blockers"])
        self.assertFalse(report["paper_like_gate"]["passed"])

    def test_local_manifest_rejects_fixture_or_temp_provenance_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((4, 4), dtype=float)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", image)
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["license_note"] = "Synthetic tempfile test fixture, not a real dataset claim."
            payload["families"]["texture"]["citation"] = "Temporary scaffold fixture citation."
            payload["families"]["medical"]["provenance_note"] = "Created from a test fixture."
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertIn(
            "Local dataset manifest contains fixture/tempfile text in license_note for: cartoon",
            report["claim_blockers"],
        )
        self.assertIn(
            "Local dataset manifest contains fixture/tempfile text in citation for: texture",
            report["claim_blockers"],
        )
        self.assertIn(
            "Local dataset manifest contains fixture/tempfile text in provenance_note for: medical",
            report["claim_blockers"],
        )

    def test_local_manifest_rejects_template_placeholder_source_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((4, 4), dtype=float)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", image)
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["license_note"] = (
                "Review dataset terms before paper-like promotion or redistribution."
            )
            payload["families"]["texture"]["provenance_note"] = (
                "Record how local files were obtained from this source, including download page, "
                "date, filtering, and any conversion steps."
            )
            payload["families"]["medical"]["files"][0]["notes"] = (
                "Replace this template entry after placing local files under medical/images and medical/masks."
            )
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertIn(
            "Local dataset manifest contains template placeholder text in license_note for: cartoon",
            report["claim_blockers"],
        )
        self.assertIn(
            "Local dataset manifest contains template placeholder text in provenance_note for: texture",
            report["claim_blockers"],
        )
        self.assertIn(
            "Local dataset manifest contains template placeholder text in notes for file: medical/sample.png",
            report["claim_blockers"],
        )

    def test_local_manifest_requires_structured_source_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((4, 4), dtype=float)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", image)
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"].pop("source_audit", None)
            payload["families"]["texture"]["source_audit"] = {
                "source_url": "https://example.invalid/source",
                "downloaded_at": "2026-06-09",
                "source_artifact_sha256": "a" * 64,
                "license_snapshot_sha256": "",
                "conversion_notes": "Converted image files to PNG.",
                "local_file_mapping_reviewed": True,
            }
            payload["families"]["medical"]["source_audit"] = {
                "source_url": "https://example.invalid/source",
                "downloaded_at": "2026-99-99",
                "source_artifact_sha256": "b" * 64,
                "license_snapshot_sha256": "c" * 64,
                "conversion_notes": "Converted image files to PNG.",
                "local_file_mapping_reviewed": False,
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertIn(
            "Local dataset manifest missing source_audit for: cartoon",
            report["claim_blockers"],
        )
        self.assertIn(
            "Local dataset manifest source_audit missing license_snapshot_sha256 for: texture",
            report["claim_blockers"],
        )
        self.assertIn(
            "Local dataset manifest source_audit downloaded_at must use a valid YYYY-MM-DD date for: medical",
            report["claim_blockers"],
        )
        self.assertIn(
            "Local dataset manifest source_audit has no local_file_mapping_reviewed=true for: medical",
            report["claim_blockers"],
        )
        self.assertFalse(report["paper_like_gate"]["passed"])

    def test_local_manifest_rejects_source_audit_artifact_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((4, 4), dtype=float)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", image)
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["source_audit"] = {
                **payload["families"]["cartoon"]["source_audit"],
                "source_artifact_sha256": "f" * 64,
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertIn(
            "Local dataset manifest source_audit source_artifact sha256 mismatch for: cartoon",
            report["claim_blockers"],
        )
        self.assertFalse(report["paper_like_gate"]["passed"])

    def test_local_manifest_rejects_tiny_fixture_source_audit_artifact_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((64, 64), dtype=float)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", image)
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = payload["families"]["cartoon"]["source_audit"]
            source_artifact = Path(audit["source_artifact_path"])
            source_artifact.write_text("cartoon reviewed source artifact\n", encoding="utf-8")
            audit["source_artifact_sha256"] = iterated_rof_paper_like._file_evidence(source_artifact)["sha256"]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertIn(
            "Local dataset manifest source_audit source_artifact artifact is too small to support review evidence for: cartoon",
            report["claim_blockers"],
        )
        self.assertIn(
            "Local dataset manifest source_audit source_artifact artifact contains fixture/placeholder text for: cartoon",
            report["claim_blockers"],
        )

    def test_template_source_audit_paths_resolve_relative_to_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iterated_rof_paper_like.prepare_data_layout(root)

            report = iterated_rof_paper_like.build_readiness_report(root)

        cartoon_gap = next(
            item for item in report["data_gap_checklist"]["families"] if item["family"] == "cartoon"
        )
        source_artifact = cartoon_gap["source_audit"]["artifacts"]["source_artifact"]
        self.assertEqual(source_artifact["path"], "cartoon/audit/source-artifact.ext")
        self.assertEqual(source_artifact["path_status"], "missing_file")
        self.assertIn(
            "cartoon/audit/source-artifact.ext",
            source_artifact["resolved_path"],
        )

    def test_local_manifest_accepts_data_root_relative_source_audit_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((4, 4), dtype=float)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", image)
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                audit = payload["families"][family]["source_audit"]
                audit["source_artifact_path"] = f"{family}/audit/source-artifact.txt"
                audit["license_snapshot_path"] = f"{family}/audit/license-snapshot.txt"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertFalse(
            any("source_audit source_artifact path is outside" in blocker for blocker in report["claim_blockers"])
        )
        self.assertFalse(
            any("source_audit license_snapshot path is outside" in blocker for blocker in report["claim_blockers"])
        )

    def test_local_manifest_rejects_source_audit_artifact_outside_data_root(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            image = np.zeros((4, 4), dtype=float)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", image)
            self._write_license_reviewed_manifest(root)
            outside_artifact = Path(outside) / "source-artifact.txt"
            outside_artifact.write_text("reviewed source artifact outside data root\n", encoding="utf-8")
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["source_audit"] = {
                **payload["families"]["cartoon"]["source_audit"],
                "source_artifact_path": str(outside_artifact),
                "source_artifact_sha256": iterated_rof_paper_like._file_evidence(outside_artifact)["sha256"],
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertIn(
            "Local dataset manifest source_audit source_artifact path is outside local data root for: cartoon",
            report["claim_blockers"],
        )
        self.assertFalse(report["paper_like_gate"]["passed"])

    def test_local_manifest_rejects_source_audit_artifact_outside_family_audit_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((4, 4), dtype=float)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", image)
            self._write_license_reviewed_manifest(root)
            wrong_family_artifact = root / "texture" / "audit" / "cartoon-source-artifact.txt"
            wrong_family_artifact.write_text("cartoon source artifact stored under texture audit root\n", encoding="utf-8")
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["source_audit"] = {
                **payload["families"]["cartoon"]["source_audit"],
                "source_artifact_path": str(wrong_family_artifact),
                "source_artifact_sha256": iterated_rof_paper_like._file_evidence(wrong_family_artifact)["sha256"],
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertIn(
            "Local dataset manifest source_audit source_artifact path is outside local family audit root for: cartoon",
            report["claim_blockers"],
        )
        self.assertFalse(report["paper_like_gate"]["passed"])

    def test_local_manifest_rejects_source_audit_url_outside_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((4, 4), dtype=float)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", image)
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["source_audit"] = {
                **payload["families"]["cartoon"]["source_audit"],
                "source_url": "https://example.invalid/not-the-registered-source",
            }
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertIn(
            "Local dataset manifest source_audit source_url is not registered for source_id bsds500 for: cartoon",
            report["claim_blockers"],
        )
        self.assertFalse(report["paper_like_gate"]["passed"])

    def test_data_gap_checklist_summarizes_source_audit_artifact_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis = np.linspace(0.0, 1.0, 64)
            grid_x, grid_y = np.meshgrid(axis, axis)
            for index, family in enumerate(iterated_rof_paper_like.DATA_FAMILIES):
                image = (0.7 * grid_x + 0.3 * grid_y + index * 0.07) % 1.0
                mask = (image > 0.5).astype(float)
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", mask)
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            cartoon_artifact = Path(payload["families"]["cartoon"]["source_audit"]["source_artifact_path"])
            cartoon_artifact.unlink()
            payload["families"]["texture"]["source_audit"]["license_snapshot_sha256"] = "f" * 64
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)
            gap_by_family = {
                item["family"]: item
                for item in report["data_gap_checklist"]["families"]
            }

        cartoon_audit = gap_by_family["cartoon"]["source_audit"]
        texture_audit = gap_by_family["texture"]["source_audit"]
        medical_audit = gap_by_family["medical"]["source_audit"]
        self.assertEqual(cartoon_audit["status"], "incomplete")
        self.assertEqual(
            cartoon_audit["artifacts"]["source_artifact"]["path_status"],
            "missing_file",
        )
        self.assertEqual(
            cartoon_audit["artifacts"]["license_snapshot"]["path_status"],
            "present",
        )
        self.assertEqual(texture_audit["status"], "incomplete")
        self.assertEqual(
            texture_audit["artifacts"]["license_snapshot"]["sha256_status"],
            "mismatch",
        )
        self.assertEqual(medical_audit["status"], "complete")
        self.assertEqual(report["data_gap_checklist"]["source_audit_status_counts"]["complete"], 1)
        self.assertEqual(report["data_gap_checklist"]["source_audit_status_counts"]["incomplete"], 2)
        self.assertIn(
            "cartoon source_audit source_artifact file is missing",
            report["data_gap_checklist"]["data_ready_blockers"],
        )

    def test_data_gap_checklist_summarizes_file_level_source_audit_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            axis = np.linspace(0.0, 1.0, 64)
            grid_x, grid_y = np.meshgrid(axis, axis)
            for index, family in enumerate(iterated_rof_paper_like.DATA_FAMILIES):
                image = (0.7 * grid_x + 0.3 * grid_y + index * 0.07) % 1.0
                mask = (image > 0.5).astype(float)
                self._write_png(root / family / "images" / "sample.png", image)
                self._write_png(root / family / "masks" / "sample.png", mask)
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            file_override = self._source_audit("cartoon", root, "file-override")
            Path(file_override["source_artifact_path"]).unlink()
            payload["families"]["cartoon"]["files"][0]["source_audit"] = file_override
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)
            gap_by_family = {
                item["family"]: item
                for item in report["data_gap_checklist"]["families"]
            }

        cartoon_audit = gap_by_family["cartoon"]["source_audit"]
        texture_audit = gap_by_family["texture"]["source_audit"]
        self.assertEqual(cartoon_audit["status"], "incomplete")
        self.assertEqual(cartoon_audit["file_overrides"][0]["image"], "sample.png")
        self.assertEqual(cartoon_audit["file_overrides"][0]["status"], "incomplete")
        self.assertEqual(
            cartoon_audit["file_overrides"][0]["artifacts"]["source_artifact"]["path_status"],
            "missing_file",
        )
        self.assertEqual(texture_audit["status"], "complete")
        self.assertIn(
            "cartoon/sample.png source_audit source_artifact file is missing",
            report["data_gap_checklist"]["data_ready_blockers"],
        )

    def test_local_manifest_requires_file_claim_for_each_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            source_ids = {
                "cartoon": "bsds500",
                "texture": "prague-texture",
                "medical": "brainweb",
            }
            payload = {
                "families": {
                    family: {
                        "source_id": source_id,
                        "source_name": source_id,
                        "license_reviewed": True,
                        "license_note": "reviewed test source",
                        "citation": "test citation",
                        "files": [],
                    }
                    for family, source_id in source_ids.items()
                }
            }
            (root / "dataset_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertEqual(report["status"], "ready_for_paper_like_runner")
        self.assertIn(
            "Local dataset manifest missing file claim for: cartoon/sample.png",
            report["claim_blockers"],
        )
        self.assertFalse(report["paper_like_gate"]["passed"])

    def test_local_manifest_file_claim_hashes_must_match_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["files"][0]["sha256"] = "0" * 64
            payload["families"]["cartoon"]["files"][0]["mask_sha256"] = "1" * 64
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertIn(
            "Local dataset manifest sha256 mismatch for file: cartoon/sample.png",
            report["claim_blockers"],
        )
        self.assertIn(
            "Local dataset manifest mask_sha256 mismatch for file: cartoon/sample.png",
            report["claim_blockers"],
        )
        self.assertFalse(report["paper_like_gate"]["passed"])

    def test_refresh_manifest_file_claims_populates_hashes_from_local_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root, include_files=False)

            refresh = iterated_rof_paper_like.refresh_manifest_file_claims(root)
            payload = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
            report = iterated_rof_paper_like.build_readiness_report(root)
            self.assertTrue(refresh["written"])
            self.assertEqual(refresh["status"], "updated")
            self.assertEqual(report["claim_blockers"], [])
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                file_claim = payload["families"][family]["files"][0]
                image_path = root / family / "images" / file_claim["image"]
                mask_path = root / family / "masks" / file_claim["mask"]
                self.assertEqual(file_claim["sha256"], iterated_rof_paper_like._file_evidence(image_path)["sha256"])
                self.assertEqual(file_claim["mask_sha256"], iterated_rof_paper_like._file_evidence(mask_path)["sha256"])

    def test_check_manifest_file_claims_reports_stale_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root, include_files=False)
            manifest_path = root / "dataset_manifest.json"
            before_check = manifest_path.read_text(encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_manifest_file_claims(root)
            after_check = manifest_path.read_text(encoding="utf-8")
            iterated_rof_paper_like.refresh_manifest_file_claims(root)
            current_check = iterated_rof_paper_like.check_manifest_file_claims(root)

        self.assertEqual(before_check, after_check)
        self.assertFalse(stale_check["written"])
        self.assertTrue(stale_check["stale"])
        self.assertEqual(stale_check["status"], "stale")
        self.assertFalse(current_check["stale"])
        self.assertEqual(current_check["status"], "current")

    def test_check_manifest_file_claims_reports_stale_claim_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_png(root / "cartoon" / "images" / "sample.png", np.zeros((4, 4)))
            self._write_png(root / "cartoon" / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["files"].append(
                {
                    "image": "removed.png",
                    "sha256": "0" * 64,
                    "mask": "removed.png",
                    "mask_sha256": "1" * 64,
                }
            )
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_manifest_file_claims(root)

        self.assertTrue(stale_check["stale"])
        self.assertEqual(stale_check["stale_file_claim_count"], 1)
        self.assertEqual(
            stale_check["stale_file_claims"],
            [{"family": "cartoon", "image": "removed.png"}],
        )

    def test_refresh_source_audit_artifact_claims_populates_hashes_from_local_audit_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["source_audit"]["source_artifact_path"] = "cartoon/audit/source-artifact.txt"
            payload["families"]["cartoon"]["source_audit"]["source_artifact_sha256"] = ""
            payload["families"]["cartoon"]["source_audit"]["license_snapshot_path"] = "cartoon/audit/license-snapshot.txt"
            payload["families"]["cartoon"]["source_audit"]["license_snapshot_sha256"] = "0" * 64
            file_override = self._source_audit("texture", root, "file-override")
            file_override["source_artifact_path"] = "texture/audit/file-override/source-artifact.txt"
            file_override["source_artifact_sha256"] = ""
            payload["families"]["texture"]["files"][0]["source_audit"] = file_override
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            refresh = iterated_rof_paper_like.refresh_source_audit_artifact_claims(root)
            refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected_cartoon_source_sha = iterated_rof_paper_like._file_evidence(
                root / "cartoon" / "audit" / "source-artifact.txt"
            )["sha256"]
            expected_cartoon_license_sha = iterated_rof_paper_like._file_evidence(
                root / "cartoon" / "audit" / "license-snapshot.txt"
            )["sha256"]
            expected_texture_file_source_sha = iterated_rof_paper_like._file_evidence(
                root / "texture" / "audit" / "file-override" / "source-artifact.txt"
            )["sha256"]

        cartoon_audit = refreshed["families"]["cartoon"]["source_audit"]
        texture_file_audit = refreshed["families"]["texture"]["files"][0]["source_audit"]
        self.assertTrue(refresh["written"])
        self.assertEqual(refresh["status"], "updated")
        self.assertEqual(refresh["updated_artifact_claim_count"], 3)
        self.assertEqual(cartoon_audit["source_artifact_sha256"], expected_cartoon_source_sha)
        self.assertEqual(cartoon_audit["license_snapshot_sha256"], expected_cartoon_license_sha)
        self.assertEqual(texture_file_audit["source_artifact_sha256"], expected_texture_file_source_sha)

    def test_check_source_audit_artifact_claims_reports_stale_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["source_audit"]["source_artifact_sha256"] = ""
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            before_check = manifest_path.read_text(encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_source_audit_artifact_claims(root)
            after_check = manifest_path.read_text(encoding="utf-8")
            iterated_rof_paper_like.refresh_source_audit_artifact_claims(root)
            current_check = iterated_rof_paper_like.check_source_audit_artifact_claims(root)

        self.assertEqual(before_check, after_check)
        self.assertFalse(stale_check["written"])
        self.assertTrue(stale_check["stale"])
        self.assertEqual(stale_check["status"], "stale")
        self.assertEqual(stale_check["updated_artifact_claim_count"], 1)
        self.assertFalse(current_check["stale"])
        self.assertEqual(current_check["status"], "current")

    def test_check_source_audit_artifact_claims_reports_missing_source_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"].pop("source_audit")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            before_check = manifest_path.read_text(encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_source_audit_artifact_claims(root)
            after_check = manifest_path.read_text(encoding="utf-8")

        self.assertEqual(before_check, after_check)
        self.assertFalse(stale_check["written"])
        self.assertTrue(stale_check["stale"])
        self.assertEqual(stale_check["status"], "stale")
        self.assertEqual(stale_check["artifact_issue_count"], 1)
        issue = stale_check["artifact_issues"][0]
        self.assertEqual(issue["family"], "cartoon")
        self.assertEqual(issue["scope"], "family")
        self.assertEqual(issue["artifact"], "source_audit")
        self.assertEqual(issue["path_status"], "missing_source_audit")
        self.assertEqual(issue["issue"], "source_audit is missing")

    def test_check_source_audit_artifact_claims_reports_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            missing_artifact = root / "cartoon" / "audit" / "source-artifact.txt"
            missing_artifact.unlink()
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            before_check = manifest_path.read_text(encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_source_audit_artifact_claims(root)
            after_check = manifest_path.read_text(encoding="utf-8")

        self.assertEqual(before_check, after_check)
        self.assertFalse(stale_check["written"])
        self.assertTrue(stale_check["stale"])
        self.assertEqual(stale_check["status"], "stale")
        self.assertEqual(stale_check["artifact_issue_count"], 1)
        self.assertEqual(len(stale_check["artifact_issues"]), 1)
        issue = stale_check["artifact_issues"][0]
        self.assertEqual(issue["family"], "cartoon")
        self.assertEqual(issue["scope"], "family")
        self.assertEqual(issue["artifact"], "source_artifact")
        self.assertTrue(issue["path"].endswith("cartoon/audit/source-artifact.txt"))
        self.assertEqual(issue["path_status"], "missing_file")
        self.assertEqual(issue["sha256_status"], "not_checked")
        self.assertEqual(issue["issue"], "source_artifact file is missing")

    def test_check_source_audit_artifact_claims_reports_invalid_artifact_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((64, 64)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((64, 64)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = payload["families"]["cartoon"]["source_audit"]
            source_artifact = Path(audit["source_artifact_path"])
            invalid_artifact_text = "cartoon reviewed source artifact\n"
            source_artifact.write_text(invalid_artifact_text, encoding="utf-8")
            audit["source_artifact_sha256"] = iterated_rof_paper_like._file_evidence(source_artifact)["sha256"]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            before_check = manifest_path.read_text(encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_source_audit_artifact_claims(root)
            after_check = manifest_path.read_text(encoding="utf-8")
            report = iterated_rof_paper_like.build_readiness_report(root)
            gap_by_family = {
                item["family"]: item
                for item in report["data_gap_checklist"]["families"]
            }

        self.assertEqual(before_check, after_check)
        self.assertFalse(stale_check["written"])
        self.assertTrue(stale_check["stale"])
        self.assertEqual(stale_check["artifact_issue_count"], 1)
        issue = stale_check["artifact_issues"][0]
        self.assertEqual(issue["family"], "cartoon")
        self.assertEqual(issue["artifact"], "source_artifact")
        self.assertEqual(issue["path_status"], "present")
        self.assertEqual(issue["sha256_status"], "matched")
        self.assertEqual(issue["content_status"], "invalid")
        self.assertEqual(issue["content_size_bytes"], len(invalid_artifact_text.encode("utf-8")))
        self.assertEqual(
            issue["min_content_size_bytes"],
            iterated_rof_paper_like.MIN_SOURCE_AUDIT_ARTIFACT_BYTES,
        )
        self.assertEqual(
            issue["content_issue_codes"],
            ["too_small", "fixture_or_placeholder_text"],
        )
        self.assertIn(
            "artifact is too small to support review evidence",
            issue["content_issues"],
        )
        self.assertIn(
            "artifact contains fixture/placeholder text",
            issue["content_issues"],
        )
        self.assertEqual(issue["placeholder_pattern_hits"], ["reviewed source artifact"])
        self.assertEqual(issue["issue"], "source_artifact content is not review evidence")
        cartoon_audit = gap_by_family["cartoon"]["source_audit"]
        source_artifact_status = cartoon_audit["artifacts"]["source_artifact"]
        self.assertEqual(cartoon_audit["status"], "incomplete")
        self.assertEqual(source_artifact_status["content_status"], "invalid")
        self.assertEqual(source_artifact_status["content_size_bytes"], len(invalid_artifact_text.encode("utf-8")))
        self.assertEqual(
            source_artifact_status["min_content_size_bytes"],
            iterated_rof_paper_like.MIN_SOURCE_AUDIT_ARTIFACT_BYTES,
        )
        self.assertEqual(
            source_artifact_status["content_issue_codes"],
            ["too_small", "fixture_or_placeholder_text"],
        )
        self.assertEqual(source_artifact_status["placeholder_pattern_hits"], ["reviewed source artifact"])
        self.assertIn(
            "cartoon source_audit source_artifact artifact is too small to support review evidence",
            cartoon_audit["issues"],
        )
        self.assertIn(
            "cartoon source_audit source_artifact artifact contains fixture/placeholder text",
            cartoon_audit["issues"],
        )

    def test_check_source_audit_artifact_claims_reports_unstructured_artifact_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((64, 64)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((64, 64)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = payload["families"]["cartoon"]["source_audit"]
            source_artifact = Path(audit["source_artifact_path"])
            invalid_artifact_text = "\n".join(
                [
                    "teacherZ Iterated ROF local archive note.",
                    "An operator kept this narrative after checking a local archive.",
                    "It is intentionally long enough to exceed the minimum byte threshold.",
                    "It omits concrete web locator, calendar token, operator annotation, and pair trace.",
                    "Additional neutral narrative text keeps this from being a tiny text failure.",
                    "",
                ]
            )
            source_artifact.write_text(invalid_artifact_text, encoding="utf-8")
            audit["source_artifact_sha256"] = iterated_rof_paper_like._file_evidence(source_artifact)["sha256"]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_source_audit_artifact_claims(root)
            report = iterated_rof_paper_like.build_readiness_report(root)
            gap_by_family = {
                item["family"]: item
                for item in report["data_gap_checklist"]["families"]
            }

        self.assertGreaterEqual(
            len(invalid_artifact_text.encode("utf-8")),
            iterated_rof_paper_like.MIN_SOURCE_AUDIT_ARTIFACT_BYTES,
        )
        self.assertTrue(stale_check["stale"])
        self.assertEqual(stale_check["status"], "stale")
        issue = stale_check["artifact_issues"][0]
        self.assertEqual(issue["artifact"], "source_artifact")
        self.assertEqual(issue["path_status"], "present")
        self.assertEqual(issue["sha256_status"], "matched")
        self.assertEqual(issue["content_status"], "invalid")
        self.assertEqual(
            issue["content_issue_codes"],
            [
                "missing_source_url",
                "missing_review_date",
                "missing_review_note",
                "missing_conversion_or_mapping_note",
            ],
        )
        self.assertEqual(issue["placeholder_pattern_hits"], [])
        self.assertEqual(issue["issue"], "source_artifact content is not review evidence")
        cartoon_audit = gap_by_family["cartoon"]["source_audit"]
        self.assertEqual(cartoon_audit["status"], "incomplete")
        self.assertIn(
            "cartoon source_audit source_artifact artifact is missing structured review evidence",
            cartoon_audit["issues"],
        )

    def test_check_source_audit_artifact_claims_requires_manifest_url_in_artifact_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((64, 64)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((64, 64)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = payload["families"]["cartoon"]["source_audit"]
            source_artifact = Path(audit["source_artifact_path"])
            invalid_artifact_text = "\n".join(
                [
                    "teacherZ Iterated ROF source artifact review record for cartoon.",
                    "source_url=https://example.invalid/not-the-registered-source",
                    "review_date=2026-06-09",
                    "reviewer_note=Local source page and source package were reviewed for this family.",
                    "conversion_note=Images and masks are mapped by same relative path after source review.",
                    "",
                ]
            )
            source_artifact.write_text(invalid_artifact_text, encoding="utf-8")
            audit["source_artifact_sha256"] = iterated_rof_paper_like._file_evidence(source_artifact)["sha256"]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_source_audit_artifact_claims(root)

        self.assertTrue(stale_check["stale"])
        issue = stale_check["artifact_issues"][0]
        self.assertEqual(issue["artifact"], "source_artifact")
        self.assertEqual(issue["content_status"], "invalid")
        self.assertEqual(issue["content_issue_codes"], ["missing_manifest_source_url"])
        self.assertIn("artifact is missing manifest source URL evidence", issue["content_issues"])

    def test_check_source_audit_artifact_claims_rejects_invalid_artifact_review_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((64, 64)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((64, 64)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = payload["families"]["cartoon"]["source_audit"]
            source_artifact = Path(audit["source_artifact_path"])
            invalid_artifact_text = "\n".join(
                [
                    "teacherZ Iterated ROF source artifact review record for cartoon.",
                    f"source_url={audit['source_url']}",
                    "review_date=2026-99-99",
                    "reviewer_note=Local source page and source package were reviewed for this family.",
                    "conversion_note=Images and masks are mapped by same relative path after source review.",
                    "",
                ]
            )
            source_artifact.write_text(invalid_artifact_text, encoding="utf-8")
            audit["source_artifact_sha256"] = iterated_rof_paper_like._file_evidence(source_artifact)["sha256"]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_source_audit_artifact_claims(root)
            report = iterated_rof_paper_like.build_readiness_report(root)
            gap_by_family = {
                item["family"]: item
                for item in report["data_gap_checklist"]["families"]
            }

        self.assertTrue(stale_check["stale"])
        issue = stale_check["artifact_issues"][0]
        self.assertEqual(issue["artifact"], "source_artifact")
        self.assertEqual(issue["content_status"], "invalid")
        self.assertEqual(issue["content_issue_codes"], ["invalid_review_date"])
        self.assertIn("artifact review/download date evidence is not a valid date", issue["content_issues"])
        cartoon_audit = gap_by_family["cartoon"]["source_audit"]
        self.assertEqual(cartoon_audit["status"], "incomplete")
        self.assertIn(
            "cartoon source_audit source_artifact artifact is missing structured review evidence",
            cartoon_audit["issues"],
        )

    def test_check_source_audit_artifact_claims_rejects_narrative_only_review_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((64, 64)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((64, 64)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            audit = payload["families"]["cartoon"]["source_audit"]
            source_artifact = Path(audit["source_artifact_path"])
            invalid_artifact_text = "\n".join(
                [
                    "teacherZ Iterated ROF source artifact review record for cartoon.",
                    f"source_url={audit['source_url']}",
                    "review_date=2026-06-09",
                    "This narrative says source review was discussed, but no concrete note field was recorded.",
                    "It also mentions conversion and mapping as unchecked topics, without a concrete mapping field.",
                    "Additional neutral text keeps this artifact over the minimum byte threshold.",
                    "",
                ]
            )
            source_artifact.write_text(invalid_artifact_text, encoding="utf-8")
            audit["source_artifact_sha256"] = iterated_rof_paper_like._file_evidence(source_artifact)["sha256"]
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            stale_check = iterated_rof_paper_like.check_source_audit_artifact_claims(root)

        self.assertTrue(stale_check["stale"])
        issue = stale_check["artifact_issues"][0]
        self.assertEqual(issue["artifact"], "source_artifact")
        self.assertEqual(issue["content_status"], "invalid")
        self.assertEqual(
            issue["content_issue_codes"],
            ["missing_review_note", "missing_conversion_or_mapping_note"],
        )

    def test_cli_can_refresh_manifest_file_claims_before_readiness_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "readiness.json"
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root, include_files=False)

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--refresh-manifest-file-claims",
                    "--output",
                    str(output_path),
                ]
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["manifest_file_claim_refresh"]["written"])
        self.assertEqual(report["claim_blockers"], [])

    def test_cli_can_check_manifest_file_claims_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "readiness.json"
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root, include_files=False)
            manifest_path = root / "dataset_manifest.json"
            before_check = manifest_path.read_text(encoding="utf-8")

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--check-manifest-file-claims",
                    "--output",
                    str(output_path),
                ]
            )
            after_check = manifest_path.read_text(encoding="utf-8")
            report = json.loads(output_path.read_text(encoding="utf-8"))
            iterated_rof_paper_like.refresh_manifest_file_claims(root)
            current_output_path = root / "current_readiness.json"
            current_exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--check-manifest-file-claims",
                    "--output",
                    str(current_output_path),
                ]
            )
            current_report = json.loads(current_output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(before_check, after_check)
        self.assertEqual(report["manifest_file_claim_check"]["status"], "stale")
        self.assertTrue(report["manifest_file_claim_check"]["stale"])
        self.assertFalse(report["manifest_file_claim_check"]["written"])
        self.assertEqual(current_exit_code, 0)
        self.assertEqual(current_report["manifest_file_claim_check"]["status"], "current")

    def test_cli_can_refresh_source_audit_artifact_claims_before_readiness_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "readiness.json"
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["source_audit"]["source_artifact_sha256"] = ""
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--refresh-source-audit-artifact-claims",
                    "--output",
                    str(output_path),
                ]
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))
            refreshed_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["source_audit_artifact_claim_refresh"]["written"])
        self.assertEqual(report["source_audit_artifact_claim_refresh"]["updated_artifact_claim_count"], 1)
        self.assertEqual(report["claim_blockers"], [])
        self.assertEqual(
            len(refreshed_payload["families"]["cartoon"]["source_audit"]["source_artifact_sha256"]),
            64,
        )

    def test_cli_can_check_source_audit_artifact_claims_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "readiness.json"
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "sample.png", np.zeros((4, 4)))
                self._write_png(root / family / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_license_reviewed_manifest(root)
            manifest_path = root / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["source_audit"]["source_artifact_sha256"] = ""
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            before_check = manifest_path.read_text(encoding="utf-8")

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--check-source-audit-artifact-claims",
                    "--output",
                    str(output_path),
                ]
            )
            after_check = manifest_path.read_text(encoding="utf-8")
            report = json.loads(output_path.read_text(encoding="utf-8"))
            iterated_rof_paper_like.refresh_source_audit_artifact_claims(root)
            current_output_path = root / "current_readiness.json"
            current_exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--check-source-audit-artifact-claims",
                    "--output",
                    str(current_output_path),
                ]
            )
            current_report = json.loads(current_output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(before_check, after_check)
        self.assertEqual(report["source_audit_artifact_claim_check"]["status"], "stale")
        self.assertTrue(report["source_audit_artifact_claim_check"]["stale"])
        self.assertFalse(report["source_audit_artifact_claim_check"]["written"])
        self.assertEqual(current_exit_code, 0)
        self.assertEqual(current_report["source_audit_artifact_claim_check"]["status"], "current")

    def test_prepare_data_layout_creates_directories_and_manifest_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            preparation = iterated_rof_paper_like.prepare_data_layout(root)

            expected_dirs = {
                root / family / kind
                for family in iterated_rof_paper_like.DATA_FAMILIES
                for kind in ["images", "masks", "audit"]
            }
            self.assertTrue(all(path.exists() and path.is_dir() for path in expected_dirs))
            self.assertTrue((root / "dataset_manifest.json").exists())
        self.assertEqual(preparation["manifest"]["status"], "created_from_template")
        self.assertFalse(preparation["downloaded_data"])
        self.assertEqual(
            {item["status"] for item in preparation["directories"]},
            {"created"},
        )

    def test_local_data_drop_zone_ignores_large_or_review_artifacts(self):
        gitignore_text = (iterated_rof_paper_like.REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

        for kind in ["images", "masks", "audit"]:
            self.assertIn(
                f"reproduce/data/iterated_rof/*/{kind}/",
                gitignore_text,
            )

    def test_prepare_data_layout_does_not_overwrite_existing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "dataset_manifest.json"
            manifest_path.write_text('{"keep": true}\n', encoding="utf-8")

            preparation = iterated_rof_paper_like.prepare_data_layout(root)
            saved_manifest = manifest_path.read_text(encoding="utf-8")

        self.assertEqual(saved_manifest, '{"keep": true}\n')
        self.assertEqual(preparation["manifest"]["status"], "already_exists")

    def test_cli_can_prepare_data_layout_before_readiness_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "readiness.json"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--prepare-data-layout",
                    "--output",
                    str(output_path),
                ]
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["data_layout_preparation"]["manifest"]["status"], "created_from_template")
        self.assertEqual(report["local_dataset_manifest"]["status"], "present")
        self.assertEqual(report["status"], "blocked_missing_data")

    def test_ingest_data_drop_copies_local_files_and_refreshes_manifest_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drop_root = root / "drop"
            data_root = root / "canonical"
            image_path = drop_root / "cartoon" / "images" / "bsds500" / "case.png"
            mask_path = drop_root / "cartoon" / "masks" / "bsds500" / "case.png"
            source_artifact = drop_root / "cartoon" / "audit" / "source-artifact.txt"
            license_snapshot = drop_root / "cartoon" / "audit" / "license-snapshot.txt"
            self._write_png(image_path, np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64))
            self._write_png(mask_path, np.indices((64, 64))[0] > 31)
            source_artifact.parent.mkdir(parents=True, exist_ok=True)
            source_artifact.write_text(self._source_audit_artifact_text("cartoon", "source artifact"), encoding="utf-8")
            license_snapshot.write_text(self._source_audit_artifact_text("cartoon", "license snapshot"), encoding="utf-8")

            ingest = iterated_rof_paper_like.ingest_data_drop(drop_root, data_root)
            manifest = json.loads((data_root / "dataset_manifest.json").read_text(encoding="utf-8"))
            copied_image = data_root / "cartoon" / "images" / "bsds500" / "case.png"
            copied_mask = data_root / "cartoon" / "masks" / "bsds500" / "case.png"
            copied_artifact = data_root / "cartoon" / "audit" / "source-artifact.txt"
            copied_image_exists = copied_image.is_file()
            copied_mask_exists = copied_mask.is_file()
            copied_artifact_exists = copied_artifact.is_file()
            copied_image_sha = iterated_rof_paper_like._file_evidence(copied_image)["sha256"]
            copied_mask_sha = iterated_rof_paper_like._file_evidence(copied_mask)["sha256"]

        self.assertEqual(ingest["status"], "ingested")
        self.assertEqual(ingest["copied_file_count"], 4)
        self.assertEqual(ingest["conflict_file_count"], 0)
        self.assertTrue(copied_image_exists)
        self.assertTrue(copied_mask_exists)
        self.assertTrue(copied_artifact_exists)
        cartoon_claim = manifest["families"]["cartoon"]
        self.assertFalse(cartoon_claim["license_reviewed"])
        self.assertFalse(cartoon_claim["provenance_reviewed"])
        self.assertEqual(cartoon_claim["files"][0]["image"], "bsds500/case.png")
        self.assertEqual(cartoon_claim["files"][0]["sha256"], copied_image_sha)
        self.assertEqual(cartoon_claim["files"][0]["mask"], "bsds500/case.png")
        self.assertEqual(cartoon_claim["files"][0]["mask_sha256"], copied_mask_sha)
        self.assertEqual(ingest["manifest_file_claim_refresh"]["status"], "updated")

    def test_ingest_data_drop_does_not_overwrite_conflicting_canonical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drop_root = root / "drop"
            data_root = root / "canonical"
            source_image = drop_root / "cartoon" / "images" / "case.png"
            target_image = data_root / "cartoon" / "images" / "case.png"
            self._write_png(source_image, np.ones((64, 64), dtype=float))
            self._write_png(target_image, np.zeros((64, 64), dtype=float))
            before_sha = iterated_rof_paper_like._file_evidence(target_image)["sha256"]

            ingest = iterated_rof_paper_like.ingest_data_drop(drop_root, data_root)
            after_sha = iterated_rof_paper_like._file_evidence(target_image)["sha256"]

        self.assertEqual(ingest["status"], "conflict")
        self.assertEqual(ingest["copied_file_count"], 0)
        self.assertEqual(ingest["conflict_file_count"], 1)
        self.assertEqual(before_sha, after_sha)
        cartoon_images = next(
            kind
            for family in ingest["families"]
            if family["family"] == "cartoon"
            for kind in family["kinds"]
            if kind["kind"] == "images"
        )
        self.assertEqual(cartoon_images["files"][0]["status"], "conflict")

    def test_ingest_data_drop_rejects_staging_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drop_root = root / "drop"
            data_root = root / "canonical"
            outside_image = root / "outside.png"
            link_path = drop_root / "cartoon" / "images" / "escape.png"
            self._write_png(outside_image, np.ones((64, 64), dtype=float))
            link_path.parent.mkdir(parents=True, exist_ok=True)
            link_path.symlink_to(outside_image)

            ingest = iterated_rof_paper_like.ingest_data_drop(drop_root, data_root)

        self.assertEqual(ingest["status"], "empty")
        self.assertEqual(ingest["copied_file_count"], 0)
        self.assertEqual(ingest["skipped_file_count"], 1)
        self.assertFalse((data_root / "cartoon" / "images" / "escape.png").exists())
        cartoon_images = next(
            kind
            for family in ingest["families"]
            if family["family"] == "cartoon"
            for kind in family["kinds"]
            if kind["kind"] == "images"
        )
        self.assertEqual(cartoon_images["skipped_files"][0]["reason"], "path_escape")

    def test_review_data_drop_reports_plan_without_writing_canonical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drop_root = root / "drop"
            data_root = root / "canonical"
            source_image = drop_root / "cartoon" / "images" / "case.png"
            target_image = data_root / "cartoon" / "images" / "case.png"
            unsupported_notes = drop_root / "cartoon" / "images" / "notes.txt"
            outside_image = root / "outside.png"
            escape_link = drop_root / "texture" / "images" / "escape.png"
            self._write_png(source_image, np.ones((64, 64), dtype=float))
            self._write_png(target_image, np.zeros((64, 64), dtype=float))
            unsupported_notes.parent.mkdir(parents=True, exist_ok=True)
            unsupported_notes.write_text("not an image\n", encoding="utf-8")
            self._write_png(outside_image, np.ones((64, 64), dtype=float))
            escape_link.parent.mkdir(parents=True, exist_ok=True)
            escape_link.symlink_to(outside_image)
            before_sha = iterated_rof_paper_like._file_evidence(target_image)["sha256"]

            review = iterated_rof_paper_like.review_data_drop(drop_root, data_root)
            after_sha = iterated_rof_paper_like._file_evidence(target_image)["sha256"]

        self.assertEqual(review["status"], "conflict")
        self.assertFalse(review["would_write"])
        self.assertFalse(review["downloaded_data"])
        self.assertEqual(review["copyable_file_count"], 0)
        self.assertEqual(review["conflict_file_count"], 1)
        self.assertEqual(review["skipped_file_count"], 2)
        self.assertEqual(before_sha, after_sha)
        self.assertFalse((data_root / "texture").exists())
        cartoon_images = next(
            kind
            for family in review["families"]
            if family["family"] == "cartoon"
            for kind in family["kinds"]
            if kind["kind"] == "images"
        )
        texture_images = next(
            kind
            for family in review["families"]
            if family["family"] == "texture"
            for kind in family["kinds"]
            if kind["kind"] == "images"
        )
        self.assertEqual(cartoon_images["files"][0]["status"], "conflict")
        self.assertEqual(
            sorted(item["reason"] for item in cartoon_images["skipped_files"] + texture_images["skipped_files"]),
            ["path_escape", "unsupported_extension"],
        )

    def test_cli_can_review_data_drop_without_primary_report_or_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drop_root = root / "drop"
            data_root = root / "canonical"
            default_report_path = root / "default_readiness.json"
            review_path = root / "data_drop_review.json"
            self._write_png(drop_root / "cartoon" / "images" / "case.png", np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64))
            original_report_path = iterated_rof_paper_like.REPORT_PATH
            iterated_rof_paper_like.REPORT_PATH = default_report_path
            try:
                exit_code = iterated_rof_paper_like.main(
                    [
                        "--data-root",
                        str(data_root),
                        "--review-data-drop",
                        str(drop_root),
                        "--data-drop-review-output",
                        str(review_path),
                    ]
                )
            finally:
                iterated_rof_paper_like.REPORT_PATH = original_report_path
            review = json.loads(review_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertFalse(default_report_path.exists())
        self.assertFalse(data_root.exists())
        self.assertEqual(review["status"], "would_ingest")
        self.assertEqual(review["copyable_file_count"], 1)
        self.assertFalse(review["would_write"])

    def test_cli_can_ingest_data_drop_before_readiness_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            drop_root = root / "drop"
            data_root = root / "canonical"
            output_path = root / "readiness.json"
            self._write_png(drop_root / "cartoon" / "images" / "case.png", np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64))

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(data_root),
                    "--ingest-data-drop",
                    str(drop_root),
                    "--output",
                    str(output_path),
                ]
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["data_drop_ingest"]["status"], "ingested")
        self.assertEqual(report["data_drop_ingest"]["copied_file_count"], 1)
        self.assertEqual(report["local_dataset_manifest"]["status"], "present")
        self.assertIn("Missing image data for: texture, medical", report["blockers"])

    def test_data_package_review_reports_manual_manifest_and_audit_work_remaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "canonical"
            iterated_rof_paper_like.prepare_data_layout(data_root)
            self._write_png(
                data_root / "cartoon" / "images" / "bsds500" / "case.png",
                np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64),
            )
            self._write_png(
                data_root / "cartoon" / "masks" / "bsds500" / "case.png",
                np.indices((64, 64))[0] > 31,
            )
            iterated_rof_paper_like.refresh_manifest_file_claims(data_root)

            review = iterated_rof_paper_like.build_data_package_review(data_root)
            cartoon = next(item for item in review["families"] if item["family"] == "cartoon")
            texture = next(item for item in review["families"] if item["family"] == "texture")

        self.assertEqual(review["status"], "incomplete")
        self.assertFalse(review["ready_for_local_runner"])
        self.assertFalse(review["downloaded_data"])
        self.assertEqual(review["manifest_file_claim_check"]["status"], "current")
        self.assertEqual(review["source_audit_artifact_claim_check"]["status"], "stale")
        self.assertEqual(cartoon["status"], "ready_quantitative")
        self.assertFalse(cartoon["manifest_review"]["ready"])
        self.assertIn("license_reviewed", cartoon["manifest_review"]["missing_manual_fields"])
        self.assertIn("citation", cartoon["manifest_review"]["missing_manual_fields"])
        self.assertIn("source_audit.downloaded_at", cartoon["manifest_review"]["missing_manual_fields"])
        self.assertIn(
            "source_audit.source_artifact_sha256",
            cartoon["manifest_review"]["missing_manual_fields"],
        )
        self.assertEqual(cartoon["source_audit"]["status"], "incomplete")
        self.assertIn("add at least one nontrivial local image", texture["missing"])

    def test_cli_can_write_data_package_review_without_primary_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "canonical"
            default_report_path = root / "default_readiness.json"
            review_path = root / "data_package_review.json"
            self._write_png(
                data_root / "cartoon" / "images" / "case.png",
                np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64),
            )
            original_report_path = iterated_rof_paper_like.REPORT_PATH
            iterated_rof_paper_like.REPORT_PATH = default_report_path
            try:
                exit_code = iterated_rof_paper_like.main(
                    [
                        "--data-root",
                        str(data_root),
                        "--data-package-review-output",
                        str(review_path),
                    ]
                )
            finally:
                iterated_rof_paper_like.REPORT_PATH = original_report_path
            review = json.loads(review_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertFalse(default_report_path.exists())
        self.assertEqual(review["target_level"], "paper-like")
        self.assertEqual(review["review_root"], str(data_root))
        self.assertEqual(review["status"], "incomplete")
        self.assertEqual(review["families"][0]["family"], "cartoon")

    def test_load_grayscale_png_normalizes_image_and_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            self._write_png(image_path, [[0.0, 0.5], [1.0, 0.25]])
            self._write_png(mask_path, [[0.0, 1.0], [1.0, 0.0]])

            image = iterated_rof_paper_like.load_grayscale_image(image_path)
            mask = iterated_rof_paper_like.load_mask(mask_path, expected_shape=image.shape)

        self.assertEqual(image.shape, (2, 2))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        self.assertAlmostEqual(float(image[0, 1]), 0.5, places=2)
        self.assertEqual(mask.shape, image.shape)
        self.assertEqual(set(np.unique(mask).tolist()), {0, 1})

    def test_scan_dataset_discovers_images_and_matching_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_png(root / "cartoon" / "images" / "sample.png", np.zeros((4, 4)))
            self._write_png(root / "cartoon" / "masks" / "sample.png", np.zeros((4, 4)))
            self._write_png(root / "texture" / "images" / "qualitative.png", np.ones((4, 4)))

            entries = iterated_rof_paper_like.scan_dataset(root)

        by_name = {entry["image_path"].name: entry for entry in entries}
        self.assertEqual(set(by_name), {"sample.png", "qualitative.png"})
        self.assertEqual(by_name["sample.png"]["family"], "cartoon")
        self.assertIsNotNone(by_name["sample.png"]["mask_path"])
        self.assertIsNone(by_name["qualitative.png"]["mask_path"])

    def test_scan_dataset_ignores_image_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            outside_image = Path(outside) / "sample.png"
            self._write_png(outside_image, np.zeros((4, 4)))
            image_dir = root / "cartoon" / "images"
            image_dir.mkdir(parents=True)
            (image_dir / "sample.png").symlink_to(outside_image)

            entries = iterated_rof_paper_like.scan_dataset(root)
            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertEqual(entries, [])
        cartoon_family = next(item for item in report["families"] if item["family"] == "cartoon")
        self.assertEqual(cartoon_family["image_count"], 0)
        self.assertEqual(cartoon_family["status"], "missing")

    def test_scan_dataset_does_not_pair_mask_symlink_escape(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as outside:
            root = Path(tmp)
            outside_mask = Path(outside) / "sample.png"
            self._write_png(root / "cartoon" / "images" / "sample.png", np.zeros((4, 4)))
            self._write_png(outside_mask, np.zeros((4, 4)))
            mask_dir = root / "cartoon" / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            (mask_dir / "sample.png").symlink_to(outside_mask)

            entries = iterated_rof_paper_like.scan_dataset(root)
            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["mask_path"])
        cartoon_family = next(item for item in report["families"] if item["family"] == "cartoon")
        self.assertEqual(cartoon_family["mask_count"], 0)
        self.assertEqual(cartoon_family["matched_mask_count"], 0)
        self.assertEqual(cartoon_family["status"], "ready_qualitative_only")

    def test_scan_dataset_does_not_pair_ambiguous_same_stem_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_png(root / "cartoon" / "images" / "sample.png", np.zeros((4, 4)))
            self._write_png(root / "cartoon" / "masks" / "a" / "sample.png", np.zeros((4, 4)))
            self._write_png(root / "cartoon" / "masks" / "b" / "sample.png", np.ones((4, 4)))

            entries = iterated_rof_paper_like.scan_dataset(root)
            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["mask_path"])
        self.assertIn("Ambiguous masks", entries[0]["mask_warning"])
        cartoon_family = next(item for item in report["families"] if item["family"] == "cartoon")
        self.assertEqual(cartoon_family["status"], "ready_qualitative_only")
        self.assertEqual(cartoon_family["matched_mask_count"], 0)
        self.assertEqual(cartoon_family["mask_warning_count"], 1)

    def test_scan_dataset_does_not_pair_stem_only_mask_in_different_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_png(root / "cartoon" / "images" / "a" / "sample.png", np.zeros((4, 4)))
            self._write_png(root / "cartoon" / "masks" / "b" / "sample.png", np.zeros((4, 4)))

            entries = iterated_rof_paper_like.scan_dataset(root)
            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["mask_path"])
        self.assertIn("Stem-only mask match ignored", entries[0]["mask_warning"])
        cartoon_family = next(item for item in report["families"] if item["family"] == "cartoon")
        self.assertEqual(cartoon_family["status"], "ready_qualitative_only")
        self.assertEqual(cartoon_family["matched_mask_count"], 0)

    def test_scan_dataset_requires_exact_relative_mask_path_including_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "cartoon" / "images").mkdir(parents=True)
            (root / "cartoon" / "masks").mkdir(parents=True)
            (root / "cartoon" / "images" / "sample.jpg").write_bytes(b"fake image")
            (root / "cartoon" / "masks" / "sample.png").write_bytes(b"fake mask")

            entries = iterated_rof_paper_like.scan_dataset(root)
            report = iterated_rof_paper_like.build_readiness_report(root)

        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["mask_path"])
        self.assertIn("Stem-only mask match ignored", entries[0]["mask_warning"])
        cartoon_family = next(item for item in report["families"] if item["family"] == "cartoon")
        self.assertEqual(cartoon_family["status"], "ready_qualitative_only")
        self.assertEqual(cartoon_family["matched_mask_count"], 0)

    def test_local_runner_reports_metrics_or_qualitative_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 8), dtype=float)
            image[:, 4:] = 1.0
            mask = np.zeros((8, 8), dtype=float)
            mask[:, 4:] = 1.0
            self._write_png(root / "cartoon" / "images" / "binary.png", image)
            self._write_png(root / "cartoon" / "masks" / "binary.png", mask)
            self._write_png(root / "texture" / "images" / "qualitative.png", image.T)

            summary = iterated_rof_paper_like.run_local_dataset(
                root,
                rof_n_iter=8,
                trof_max_iter=4,
                figure_dir=root / "figures",
            )

            self.assertEqual(summary["status"], "completed_local_runner")
            self.assertEqual(summary["image_count"], 2)
            self.assertEqual(summary["quantitative_image_count"], 1)
            quantitative = next(item for item in summary["images"] if item["mask_path"])
            qualitative = next(item for item in summary["images"] if not item["mask_path"])
            self.assertFalse(quantitative["qualitative_only"])
            self.assertIn("clustering_accuracy", quantitative["metrics"])
            self.assertIn("dice", quantitative["metrics"])
            self.assertIn("baselines", quantitative)
            self.assertIn("raw_kmeans", quantitative["baselines"])
            self.assertIn("multi_otsu", quantitative["baselines"])
            self.assertIn("clustering_accuracy", quantitative["baselines"]["raw_kmeans"]["metrics"])
            self.assertIn("figure_path", quantitative)
            self.assertTrue(Path(quantitative["figure_path"]).exists())
            self.assertGreater(Path(quantitative["figure_path"]).stat().st_size, 0)
            self.assertIn("figure_evidence", quantitative)
            self.assertIn("figure_evidence_path", quantitative)
            self.assertTrue(Path(quantitative["figure_evidence_path"]).exists())
            self.assertEqual(
                quantitative["figure_evidence"]["image_sha256"],
                quantitative["image_file"]["sha256"],
            )
            self.assertEqual(
                quantitative["figure_evidence"]["mask_sha256"],
                quantitative["mask_file"]["sha256"],
            )
            self.assertEqual(
                quantitative["figure_evidence"]["figure_sha256"],
                quantitative["figure_file"]["sha256"],
            )
            self.assertIn("T-ROF error", quantitative["figure_panels"])
            self.assertTrue(any(title in quantitative["figure_panels"] for title in ["T-ROF vs Otsu", "T-ROF vs quantile"]))
            self.assertIn("image_file", quantitative)
            self.assertEqual(len(quantitative["image_file"]["sha256"]), 64)
            self.assertIn("mask_file", quantitative)
            self.assertEqual(len(quantitative["mask_file"]["sha256"]), 64)
            self.assertEqual(quantitative["source_claim"]["manifest_status"], "missing")
            self.assertTrue(qualitative["qualitative_only"])
            self.assertEqual(qualitative["metrics"], {})
            self.assertIn("baselines", qualitative)
            self.assertIn("raw_kmeans", qualitative["baselines"])
            self.assertIn("figure_path", qualitative)
            self.assertTrue(Path(qualitative["figure_path"]).exists())
            self.assertIn("T-ROF vs raw", qualitative["figure_panels"])
            self.assertIn("Missing image data for: medical", summary["blockers"])
            self.assertTrue(summary["claim_blockers"])

    def test_local_runner_does_not_hard_require_skimage_for_quantile_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 8), dtype=float)
            image[:, 4:] = 1.0
            self._write_png(root / "cartoon" / "images" / "binary.png", image)
            self._write_png(root / "cartoon" / "masks" / "binary.png", image)

            original_require_modules = iterated_rof_paper_like.require_modules
            require_calls = []

            def fake_require_modules(*names):
                require_calls.append(names)
                if "skimage" in names:
                    return ["skimage"]
                return []

            iterated_rof_paper_like.require_modules = fake_require_modules
            try:
                summary = iterated_rof_paper_like.run_local_dataset(
                    root,
                    rof_n_iter=8,
                    trof_max_iter=4,
                    figure_dir=root / "figures",
                )
            finally:
                iterated_rof_paper_like.require_modules = original_require_modules

        self.assertNotEqual(summary["status"], "blocked_missing_dependencies")
        self.assertTrue(require_calls)
        self.assertTrue(all("skimage" not in names for names in require_calls))

    def test_local_runner_blocks_when_no_images_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = iterated_rof_paper_like.run_local_dataset(Path(tmp), rof_n_iter=2, trof_max_iter=1)
            csv_path = Path(tmp) / "family_summary.csv"
            iterated_rof_paper_like.write_family_summary_csv(summary, csv_path)
            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(summary["status"], "blocked_missing_data")
        self.assertEqual(summary["image_count"], 0)
        self.assertEqual(summary["dataset_fingerprint"]["file_count"], 0)
        self.assertEqual(len(summary["dataset_fingerprint"]["sha256"]), 64)
        self.assertTrue(summary["blockers"])
        self.assertEqual(summary["readiness_status"], "blocked_missing_data")
        self.assertEqual(summary["run_protocol"]["protocol_id"], "iterated_rof_trof_local_data_v1")
        self.assertEqual(summary["run_protocol"]["parameters"]["rof_n_iter"], 2)
        self.assertEqual(summary["run_protocol"]["parameters"]["trof_max_iter"], 1)
        self.assertEqual(len(summary["family_summaries"]), 3)
        self.assertTrue(all(item["status"] == "missing" for item in summary["family_summaries"]))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["status"] == "missing" for row in rows))

    def test_local_runner_preserves_readiness_blockers_for_partial_family_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 8), dtype=float)
            image[:, 4:] = 1.0
            self._write_png(root / "cartoon" / "images" / "binary.png", image)
            self._write_png(root / "cartoon" / "masks" / "binary.png", image)

            summary = iterated_rof_paper_like.run_local_dataset(
                root,
                rof_n_iter=8,
                trof_max_iter=4,
                figure_dir=root / "figures",
            )

        self.assertEqual(summary["status"], "completed_local_runner")
        self.assertEqual(summary["readiness_status"], "blocked_missing_data")
        self.assertIn("Missing image data for: texture, medical", summary["blockers"])
        self.assertFalse(summary["paper_like_gate"]["passed"])
        self.assertIn("No completed local runner output for family: texture", summary["paper_like_gate"]["reasons"])

    def test_local_runner_records_reviewed_manifest_source_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 8), dtype=float)
            image[:, 4:] = 1.0
            self._write_png(root / "cartoon" / "images" / "binary.png", image)
            self._write_png(root / "cartoon" / "masks" / "binary.png", image)
            self._write_license_reviewed_manifest(root)

            summary = iterated_rof_paper_like.run_local_dataset(
                root,
                rof_n_iter=8,
                trof_max_iter=4,
                figure_dir=root / "figures",
            )

        quantitative = next(item for item in summary["images"] if item["mask_path"])
        self.assertEqual(summary["claim_blockers"], [])
        self.assertEqual(summary["local_dataset_manifest"]["status"], "present")
        self.assertEqual(quantitative["source_claim"]["manifest_status"], "present")
        self.assertEqual(quantitative["source_claim"]["claim_scope"], "file")
        self.assertEqual(quantitative["source_claim"]["image"], "binary.png")
        self.assertEqual(quantitative["source_claim"]["mask"], "binary.png")
        self.assertEqual(quantitative["source_claim"]["sha256"], quantitative["image_file"]["sha256"])
        self.assertEqual(quantitative["source_claim"]["mask_sha256"], quantitative["mask_file"]["sha256"])
        self.assertEqual(quantitative["source_claim"]["source_id"], "bsds500")
        self.assertTrue(quantitative["source_claim"]["license_reviewed"])
        self.assertFalse(summary["paper_like_gate"]["passed"])

    def test_local_runner_requires_quantitative_output_for_every_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 8), dtype=float)
            image[:, 4:] = 1.0
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self._write_png(root / family / "images" / "binary.png", image)
            self._write_png(root / "cartoon" / "masks" / "binary.png", image)
            self._write_license_reviewed_manifest(root)

            summary = iterated_rof_paper_like.run_local_dataset(
                root,
                rof_n_iter=8,
                trof_max_iter=4,
                figure_dir=root / "figures",
            )

        self.assertEqual(summary["readiness_status"], "blocked_missing_masks")
        self.assertEqual(summary["quantitative_image_count"], 1)
        self.assertIn(
            "Missing masks/labels for quantitative paper-like metrics in: texture, medical",
            summary["blockers"],
        )
        self.assertFalse(summary["paper_like_gate"]["passed"])
        self.assertIn(
            "No completed quantitative local runner output for family: texture",
            summary["paper_like_gate"]["reasons"],
        )
        self.assertIn(
            "No completed quantitative local runner output for family: medical",
            summary["paper_like_gate"]["reasons"],
        )

    def test_local_runner_keeps_complete_temp_fixture_below_paper_like_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                image = np.zeros((8, 8), dtype=float)
                image[:, 4:] = 1.0
                if family == "texture":
                    image = image.T
                if family == "medical":
                    image = np.flipud(image)
                self._write_png(root / family / "images" / "binary.png", image)
                self._write_png(root / family / "masks" / "binary.png", image)
            self._write_license_reviewed_manifest(root)

            summary = iterated_rof_paper_like.run_local_dataset(
                root,
                rof_n_iter=8,
                trof_max_iter=4,
                figure_dir=root / "figures",
            )

            self.assertTrue(all(Path(item["figure_path"]).exists() for item in summary["images"]))

        self.assertEqual(summary["status"], "completed_local_runner")
        self.assertEqual(summary["readiness_status"], "ready_for_paper_like_runner")
        self.assertEqual(summary["image_count"], 3)
        self.assertEqual(summary["quantitative_image_count"], 3)
        self.assertEqual(summary["dataset_fingerprint"]["file_count"], 6)
        self.assertEqual(summary["dataset_fingerprint"]["algorithm"], "sha256")
        self.assertEqual(len(summary["dataset_fingerprint"]["sha256"]), 64)
        self.assertEqual(summary["blockers"], [])
        self.assertEqual(summary["claim_blockers"], [])
        self.assertFalse(summary["paper_like_gate"]["passed"])
        self.assertEqual(summary["paper_like_gate"]["dashboard_level"], "partial")
        self.assertIn(
            "Paper-like promotion requires canonical data root: reproduce/data/iterated_rof",
            summary["paper_like_gate"]["reasons"],
        )
        self.assertEqual(len(summary["family_summaries"]), 3)
        for family_summary in summary["family_summaries"]:
            self.assertEqual(family_summary["status"], "completed_quantitative")
            self.assertEqual(family_summary["completed_image_count"], 1)
            self.assertEqual(family_summary["quantitative_image_count"], 1)
            self.assertIn("clustering_accuracy", family_summary["metrics_mean"])
            self.assertIn("dice", family_summary["metrics_mean"])
            self.assertIn("raw_kmeans", family_summary["baseline_metrics_mean"])
            self.assertIn("multi_otsu", family_summary["baseline_metrics_mean"])
            self.assertEqual(len(family_summary["figure_paths"]), 1)

    def test_paper_like_gate_requires_dataset_fingerprint(self):
        image_results = []
        for family in iterated_rof_paper_like.DATA_FAMILIES:
            image_results.append(
                {
                    "family": family,
                    "status": "completed",
                    "qualitative_only": False,
                    "image_path": f"reproduce/data/iterated_rof/{family}/images/sample.png",
                    "mask_path": f"reproduce/data/iterated_rof/{family}/masks/sample.png",
                    "baselines": {"raw_kmeans": {}, "multi_otsu": {}},
                    "figure_path": f"reproduce/results/figures/{family}.png",
                    "figure_panels": ["ROF", "T-ROF"],
                    "image_file": {"sha256": "a" * 64},
                    "mask_file": {"sha256": "b" * 64},
                    "source_claim": {
                        "manifest_status": "present",
                        "claim_scope": "file",
                    },
                }
            )

        gate = iterated_rof_paper_like._paper_like_gate(
            "ready_for_paper_like_runner",
            [],
            [],
            image_results,
            data_root=iterated_rof_paper_like.DATA_ROOT,
        )

        self.assertFalse(gate["passed"])
        self.assertIn("Missing non-empty dataset fingerprint", gate["reasons"])

    def test_paper_like_gate_rejects_dataset_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            bad_fingerprint = {
                **summary["dataset_fingerprint"],
                "sha256": "0" * 64,
            }
            gate = iterated_rof_paper_like._paper_like_gate(
                "ready_for_paper_like_runner",
                [],
                [],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=bad_fingerprint,
            )

        self.assertFalse(gate["passed"])
        self.assertIn("Dataset fingerprint does not match image/mask evidence", gate["reasons"])

    def test_dataset_fingerprint_from_image_results_uses_local_paths_not_source_claim_paths(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            original_fingerprint = summary["dataset_fingerprint"]
            summary["images"][0]["source_claim"]["image"] = "forged.png"
            summary["images"][0]["source_claim"]["mask"] = "forged.png"

            recomputed_fingerprint = iterated_rof_paper_like._dataset_fingerprint_from_image_results(summary["images"])

        self.assertEqual(recomputed_fingerprint, original_fingerprint)

    def test_report_relative_paths_are_from_family_image_and_mask_roots(self):
        item = {
            "family": "cartoon",
            "image_path": "reproduce/data/iterated_rof/cartoon/images/bsds/images/train/sample.png",
            "mask_path": "reproduce/data/iterated_rof/cartoon/masks/bsds/images/train/sample.png",
        }

        self.assertEqual(
            iterated_rof_paper_like._image_result_relative_path(item),
            "bsds/images/train/sample.png",
        )
        self.assertEqual(
            iterated_rof_paper_like._mask_result_relative_path(item),
            "bsds/images/train/sample.png",
        )

    def test_paper_like_gate_rejects_source_claim_image_path_mismatch(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["source_claim"]["image"] = "forged.png"
            gate = iterated_rof_paper_like._paper_like_gate(
                "ready_for_paper_like_runner",
                [],
                [],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        expected_reason = f"Source claim image does not match local image path for: {image['image_path']}"
        checklist = {item["id"]: item for item in gate["checklist"]}
        self.assertFalse(gate["passed"])
        self.assertIn(expected_reason, gate["reasons"])
        self.assertIn(expected_reason, checklist["output_evidence"]["reasons"])
        self.assertNotIn("Dataset fingerprint does not match image/mask evidence", gate["reasons"])

    def test_paper_like_gate_rejects_source_claim_mask_path_mismatch(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["source_claim"]["mask"] = "forged.png"
            gate = iterated_rof_paper_like._paper_like_gate(
                "ready_for_paper_like_runner",
                [],
                [],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        expected_reason = f"Source claim mask does not match local mask path for: {image['image_path']}"
        checklist = {item["id"]: item for item in gate["checklist"]}
        self.assertFalse(gate["passed"])
        self.assertIn(expected_reason, gate["reasons"])
        self.assertIn(expected_reason, checklist["output_evidence"]["reasons"])
        self.assertNotIn("Dataset fingerprint does not match image/mask evidence", gate["reasons"])

    def test_paper_like_gate_rejects_non_data_file_as_image_mask_figure(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            forged_path = Path(iterated_rof_paper_like.__file__).resolve()
            forged_evidence = iterated_rof_paper_like._file_evidence(forged_path)
            image = summary["images"][0]
            image["image_path"] = iterated_rof_paper_like._display_path(forged_path)
            image["mask_path"] = iterated_rof_paper_like._display_path(forged_path)
            image["figure_path"] = iterated_rof_paper_like._display_path(forged_path)
            image["image_file"] = forged_evidence
            image["mask_file"] = forged_evidence
            image["figure_file"] = forged_evidence
            image["source_claim"]["image"] = forged_path.name
            image["source_claim"]["mask"] = forged_path.name
            image["source_claim"]["sha256"] = forged_evidence["sha256"]
            image["source_claim"]["mask_sha256"] = forged_evidence["sha256"]
            summary["dataset_fingerprint"] = iterated_rof_paper_like._dataset_fingerprint_from_image_results(
                summary["images"]
            )
            gate = iterated_rof_paper_like._paper_like_gate(
                "ready_for_paper_like_runner",
                [],
                [],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        self.assertFalse(gate["passed"])
        self.assertIn(
            f"Local image path is outside canonical family images directory for: {image['image_path']}",
            gate["reasons"],
        )
        self.assertIn(
            f"Local mask path is outside canonical family masks directory for: {image['image_path']}",
            gate["reasons"],
        )
        self.assertIn(
            f"Local figure path is outside allowed figure directory for: {image['image_path']}",
            gate["reasons"],
        )

    def test_paper_like_gate_rejects_evidence_path_mismatch(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["image_file"] = {**image["image_file"], "path": "forged/image.png"}
            image["mask_file"] = {**image["mask_file"], "path": "forged/mask.png"}
            image["figure_file"] = {**image["figure_file"], "path": "forged/figure.png"}
            gate = iterated_rof_paper_like._paper_like_gate(
                "ready_for_paper_like_runner",
                [],
                [],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        self.assertFalse(gate["passed"])
        self.assertIn(f"Local image file evidence does not match disk for: {image['image_path']}", gate["reasons"])
        self.assertIn(f"Local mask file evidence does not match disk for: {image['image_path']}", gate["reasons"])
        self.assertIn(f"Local figure file evidence does not match disk for: {image['image_path']}", gate["reasons"])

    def test_paper_like_gate_rejects_mask_relative_path_not_matching_image_relative_path(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            family = image["family"]
            image_relative_path = iterated_rof_paper_like._image_result_relative_path(image)
            alternate_relative_path = str(Path(image_relative_path).with_name("other.png"))
            alternate_mask_path = (
                iterated_rof_paper_like.DATA_ROOT
                / family
                / "masks"
                / alternate_relative_path
            )
            alternate_mask_path.parent.mkdir(parents=True, exist_ok=True)
            alternate_mask_path.write_bytes(b"alternate-mask")
            image["mask_path"] = iterated_rof_paper_like._display_path(alternate_mask_path)
            image["mask_file"] = iterated_rof_paper_like._file_evidence(alternate_mask_path)
            image["source_claim"]["mask"] = alternate_relative_path
            image["source_claim"]["mask_sha256"] = image["mask_file"]["sha256"]
            summary["dataset_fingerprint"] = iterated_rof_paper_like._dataset_fingerprint_from_image_results(
                summary["images"]
            )
            gate = iterated_rof_paper_like._paper_like_gate(
                "ready_for_paper_like_runner",
                [],
                [],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        self.assertFalse(gate["passed"])
        self.assertIn(
            f"Local mask relative path does not match local image path for: {image['image_path']}",
            gate["reasons"],
        )

    def test_paper_like_gate_rejects_unknown_source_id(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["source_claim"]["source_id"] = "forged-source"
            gate = iterated_rof_paper_like._paper_like_gate(
                "ready_for_paper_like_runner",
                [],
                [],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        self.assertFalse(gate["passed"])
        self.assertIn(
            f"Source claim source_id is not in source registry for: {image['image_path']}: forged-source",
            gate["reasons"],
        )

    def test_paper_like_gate_rejects_figure_evidence_sidecar_mismatch(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["figure_evidence"] = {
                **image["figure_evidence"],
                "image_sha256": "0" * 64,
            }
            gate = iterated_rof_paper_like._paper_like_gate(
                "ready_for_paper_like_runner",
                [],
                [],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        self.assertFalse(gate["passed"])
        self.assertIn(
            f"Figure evidence sidecar does not match report for: {image['image_path']}",
            gate["reasons"],
        )

    def test_paper_like_gate_rejects_missing_figure_evidence_sidecar(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image.pop("figure_evidence", None)
            image.pop("figure_evidence_path", None)
            image.pop("figure_evidence_file", None)
            gate = iterated_rof_paper_like._paper_like_gate(
                "ready_for_paper_like_runner",
                [],
                [],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        self.assertFalse(gate["passed"])
        self.assertIn(
            f"Missing figure evidence sidecar for: {image['image_path']}",
            gate["reasons"],
        )

    def test_paper_like_gate_requires_figure_file_evidence(self):
        image_results = []
        for family in iterated_rof_paper_like.DATA_FAMILIES:
            image_results.append(
                {
                    "family": family,
                    "status": "completed",
                    "qualitative_only": False,
                    "image_path": f"reproduce/data/iterated_rof/{family}/images/sample.png",
                    "mask_path": f"reproduce/data/iterated_rof/{family}/masks/sample.png",
                    "baselines": {"raw_kmeans": {}, "multi_otsu": {}},
                    "figure_path": f"reproduce/results/figures/{family}.png",
                    "figure_panels": ["ROF", "T-ROF"],
                    "image_file": {"sha256": "a" * 64},
                    "mask_file": {"sha256": "b" * 64},
                    "source_claim": {
                        "manifest_status": "present",
                        "claim_scope": "file",
                    },
                }
            )

        gate = iterated_rof_paper_like._paper_like_gate(
            "ready_for_paper_like_runner",
            [],
            [],
            image_results,
            data_root=iterated_rof_paper_like.DATA_ROOT,
            dataset_fingerprint={
                "algorithm": "sha256",
                "file_count": 6,
                "sha256": "f" * 64,
            },
        )

        self.assertFalse(gate["passed"])
        self.assertIn(
            "Missing figure file evidence for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )

    def test_paper_like_gate_rechecks_local_file_evidence(self):
        image_results = [
            self._complete_image_result(family)
            for family in iterated_rof_paper_like.DATA_FAMILIES
        ]

        gate = iterated_rof_paper_like._paper_like_gate(
            "ready_for_paper_like_runner",
            [],
            [],
            image_results,
            data_root=iterated_rof_paper_like.DATA_ROOT,
            dataset_fingerprint={
                "algorithm": "sha256",
                "file_count": 6,
                "sha256": "f" * 64,
            },
        )

        self.assertFalse(gate["passed"])
        self.assertIn(
            "Local image file evidence does not match disk for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )
        self.assertIn(
            "Local mask file evidence does not match disk for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )
        self.assertIn(
            "Local figure file evidence does not match disk for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )

    def test_paper_like_gate_passes_with_matching_local_file_evidence(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)

        self.assertTrue(summary["paper_like_gate"]["passed"])

    def test_passed_gate_fixture_uses_temp_canonical_paths_without_polluting_real_data_root(self):
        real_data_root = iterated_rof_paper_like.REPO_ROOT / "reproduce" / "data" / "iterated_rof"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = f"_test_{root.name}"
            summary = self._passed_gate_report_with_matching_local_files(root)

            self.assertEqual(summary["data_root"], str(root / "reproduce" / "data" / "iterated_rof"))
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                self.assertFalse((real_data_root / family / "images" / token).exists())
                self.assertFalse((real_data_root / family / "masks" / token).exists())
                self.assertFalse((real_data_root / family / "audit" / token).exists())

    def test_paper_like_gate_rejects_saved_summary_with_mismatched_mask_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            image = summary["images"][0]
            mask_path = Path(image["mask_path"])
            mask_values = np.zeros((48, 64), dtype=float)
            mask_values[:, 32:] = 1.0
            self._write_png(mask_path, mask_values)
            image["mask_file"] = iterated_rof_paper_like._file_evidence(mask_path)
            image["source_claim"]["mask_sha256"] = image["mask_file"]["sha256"]

            manifest_path = Path(summary["local_dataset_manifest"]["path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            file_claim = manifest["families"][image["family"]]["files"][0]
            file_claim["mask_sha256"] = image["mask_file"]["sha256"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            summary["local_dataset_manifest"] = iterated_rof_paper_like.load_local_dataset_manifest(
                iterated_rof_paper_like.DATA_ROOT
            )

            blockers = iterated_rof_paper_like._promotion_report_shape_blockers(summary)

        self.assertTrue(
            any("mask shape does not match image shape" in blocker for blocker in blockers),
            blockers,
        )

    def test_paper_like_gate_requires_quantitative_metrics_and_baseline_metrics(self):
        image_results = []
        for family in iterated_rof_paper_like.DATA_FAMILIES:
            metrics = {"clustering_accuracy": 1.0}
            baselines = {
                "raw_kmeans": {"metrics": {"clustering_accuracy": 1.0}},
                "multi_otsu": {"metrics": {"clustering_accuracy": 1.0}},
            }
            if family == "cartoon":
                metrics = {}
            if family == "texture":
                baselines["raw_kmeans"] = {"metrics": {}}
            image_results.append(
                {
                    "family": family,
                    "status": "completed",
                    "qualitative_only": False,
                    "image_path": f"reproduce/data/iterated_rof/{family}/images/sample.png",
                    "mask_path": f"reproduce/data/iterated_rof/{family}/masks/sample.png",
                    "metrics": metrics,
                    "baselines": baselines,
                    "figure_path": f"reproduce/results/figures/{family}.png",
                    "figure_file": {"sha256": "c" * 64, "size_bytes": 123},
                    "figure_panels": ["ROF", "T-ROF"],
                    "image_file": {"sha256": "a" * 64},
                    "mask_file": {"sha256": "b" * 64},
                    "source_claim": {
                        "manifest_status": "present",
                        "claim_scope": "file",
                    },
                }
            )

        gate = iterated_rof_paper_like._paper_like_gate(
            "ready_for_paper_like_runner",
            [],
            [],
            image_results,
            data_root=iterated_rof_paper_like.DATA_ROOT,
            dataset_fingerprint={
                "algorithm": "sha256",
                "file_count": 6,
                "sha256": "f" * 64,
            },
        )

        self.assertFalse(gate["passed"])
        self.assertIn(
            "Missing T-ROF clustering_accuracy metric for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )
        self.assertIn(
            "Missing raw_kmeans clustering_accuracy baseline metric for: reproduce/data/iterated_rof/texture/images/sample.png",
            gate["reasons"],
        )

    def test_paper_like_gate_rejects_nonfinite_metrics(self):
        image_results = []
        for family in iterated_rof_paper_like.DATA_FAMILIES:
            metrics = {"clustering_accuracy": 1.0}
            baselines = {
                "raw_kmeans": {"metrics": {"clustering_accuracy": 1.0}},
                "multi_otsu": {"metrics": {"clustering_accuracy": 1.0}},
            }
            if family == "cartoon":
                metrics["clustering_accuracy"] = float("nan")
            if family == "texture":
                baselines["multi_otsu"]["metrics"]["clustering_accuracy"] = float("inf")
            image_results.append(
                {
                    "family": family,
                    "status": "completed",
                    "qualitative_only": False,
                    "image_path": f"reproduce/data/iterated_rof/{family}/images/sample.png",
                    "mask_path": f"reproduce/data/iterated_rof/{family}/masks/sample.png",
                    "metrics": metrics,
                    "baselines": baselines,
                    "figure_path": f"reproduce/results/figures/{family}.png",
                    "figure_file": {"sha256": "c" * 64, "size_bytes": 123},
                    "figure_panels": [
                        "input",
                        "mask",
                        "ROF",
                        "T-ROF",
                        "raw K-means",
                        "multi-Otsu",
                        "T-ROF error",
                        "T-ROF vs Otsu",
                    ],
                    "image_file": {"sha256": "a" * 64},
                    "mask_file": {"sha256": "b" * 64},
                    "source_claim": {
                        "manifest_status": "present",
                        "claim_scope": "file",
                    },
                }
            )

        gate = iterated_rof_paper_like._paper_like_gate(
            "ready_for_paper_like_runner",
            [],
            [],
            image_results,
            data_root=iterated_rof_paper_like.DATA_ROOT,
            dataset_fingerprint={
                "algorithm": "sha256",
                "file_count": 6,
                "sha256": "f" * 64,
            },
        )

        self.assertFalse(gate["passed"])
        self.assertIn(
            "Missing T-ROF clustering_accuracy metric for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )
        self.assertIn(
            "Missing multi_otsu clustering_accuracy baseline metric for: reproduce/data/iterated_rof/texture/images/sample.png",
            gate["reasons"],
        )

    def test_paper_like_gate_requires_full_figure_evidence_panels(self):
        image_results = []
        for family in iterated_rof_paper_like.DATA_FAMILIES:
            image_results.append(
                {
                    "family": family,
                    "status": "completed",
                    "qualitative_only": False,
                    "image_path": f"reproduce/data/iterated_rof/{family}/images/sample.png",
                    "mask_path": f"reproduce/data/iterated_rof/{family}/masks/sample.png",
                    "metrics": {"clustering_accuracy": 1.0},
                    "baselines": {
                        "raw_kmeans": {"metrics": {"clustering_accuracy": 1.0}},
                        "multi_otsu": {"metrics": {"clustering_accuracy": 1.0}},
                    },
                    "figure_path": f"reproduce/results/figures/{family}.png",
                    "figure_file": {"sha256": "c" * 64, "size_bytes": 123},
                    "figure_panels": ["ROF", "T-ROF"],
                    "image_file": {"sha256": "a" * 64},
                    "mask_file": {"sha256": "b" * 64},
                    "source_claim": {
                        "manifest_status": "present",
                        "claim_scope": "file",
                    },
                }
            )

        gate = iterated_rof_paper_like._paper_like_gate(
            "ready_for_paper_like_runner",
            [],
            [],
            image_results,
            data_root=iterated_rof_paper_like.DATA_ROOT,
            dataset_fingerprint={
                "algorithm": "sha256",
                "file_count": 6,
                "sha256": "f" * 64,
            },
        )

        self.assertFalse(gate["passed"])
        self.assertIn(
            "Missing required figure evidence panels for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )

    def test_paper_like_gate_requires_reviewed_source_claim_details_and_matching_hashes(self):
        image_results = []
        for family in iterated_rof_paper_like.DATA_FAMILIES:
            source_claim = {
                "manifest_status": "present",
                "claim_scope": "file",
                "source_id": f"{family}-source",
                "license_reviewed": True,
                "citation": "test citation",
                "license_note": "reviewed test source",
                "sha256": "a" * 64,
                "mask_sha256": "b" * 64,
            }
            if family == "cartoon":
                source_claim["license_reviewed"] = False
                source_claim["citation"] = ""
                source_claim["license_note"] = ""
            if family == "texture":
                source_claim["sha256"] = "0" * 64
                source_claim["mask_sha256"] = "1" * 64
            image_results.append(
                {
                    "family": family,
                    "status": "completed",
                    "qualitative_only": False,
                    "image_path": f"reproduce/data/iterated_rof/{family}/images/sample.png",
                    "mask_path": f"reproduce/data/iterated_rof/{family}/masks/sample.png",
                    "metrics": {"clustering_accuracy": 1.0},
                    "baselines": {
                        "raw_kmeans": {"metrics": {"clustering_accuracy": 1.0}},
                        "multi_otsu": {"metrics": {"clustering_accuracy": 1.0}},
                    },
                    "figure_path": f"reproduce/results/figures/{family}.png",
                    "figure_file": {"sha256": "c" * 64, "size_bytes": 123},
                    "figure_panels": ["ROF", "T-ROF"],
                    "image_file": {"sha256": "a" * 64},
                    "mask_file": {"sha256": "b" * 64},
                    "source_claim": source_claim,
                }
            )

        gate = iterated_rof_paper_like._paper_like_gate(
            "ready_for_paper_like_runner",
            [],
            [],
            image_results,
            data_root=iterated_rof_paper_like.DATA_ROOT,
            dataset_fingerprint={
                "algorithm": "sha256",
                "file_count": 6,
                "sha256": "f" * 64,
            },
        )

        self.assertFalse(gate["passed"])
        self.assertIn(
            "Missing license_reviewed=true source claim for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )
        self.assertIn(
            "Missing citation source claim for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )
        self.assertIn(
            "Missing license_note source claim for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            gate["reasons"],
        )
        self.assertIn(
            "Source claim sha256 does not match image file evidence for: reproduce/data/iterated_rof/texture/images/sample.png",
            gate["reasons"],
        )
        self.assertIn(
            "Source claim mask_sha256 does not match mask file evidence for: reproduce/data/iterated_rof/texture/images/sample.png",
            gate["reasons"],
        )

    def test_paper_like_gate_groups_source_audit_blockers(self):
        image_results = []
        for family in iterated_rof_paper_like.DATA_FAMILIES:
            source_claim = {
                "manifest_status": "present",
                "claim_scope": "file",
                "source_id": self.SOURCE_IDS[family],
                "license_reviewed": True,
                "citation": "recorded dataset citation",
                "license_note": "reviewed dataset license",
                "provenance_reviewed": True,
                "provenance_note": "recorded dataset provenance",
                "synthetic_fixture": False,
                "sha256": "a" * 64,
                "mask_sha256": "b" * 64,
            }
            if family != "cartoon":
                source_claim["source_audit"] = self._source_audit(family)
            image_results.append(
                {
                    "family": family,
                    "status": "completed",
                    "qualitative_only": False,
                    "image_path": f"reproduce/data/iterated_rof/{family}/images/sample.png",
                    "mask_path": f"reproduce/data/iterated_rof/{family}/masks/sample.png",
                    "metrics": {"clustering_accuracy": 1.0},
                    "baselines": {
                        "raw_kmeans": {
                            "method": iterated_rof_paper_like.EXPECTED_RAW_KMEANS_METHOD,
                            "metrics": {"clustering_accuracy": 1.0},
                        },
                        "multi_otsu": {
                            "method": "quantile_fallback",
                            "thresholds": [0.5],
                            "metrics": {"clustering_accuracy": 1.0},
                        },
                    },
                    "figure_path": f"reproduce/results/figures/iterated_rof_paper_like/{family}/sample.png",
                    "figure_file": {"sha256": "c" * 64, "size_bytes": 123},
                    "figure_panels": ["input", "ROF", "T-ROF", "mask"],
                    "figure_evidence_path": f"reproduce/results/figures/iterated_rof_paper_like/{family}/sample.evidence.json",
                    "figure_evidence": {
                        "generator": iterated_rof_paper_like.FIGURE_EVIDENCE_GENERATOR,
                        "figure_path": f"reproduce/results/figures/iterated_rof_paper_like/{family}/sample.png",
                        "panels": ["input", "ROF", "T-ROF", "mask"],
                    },
                    "figure_evidence_file": {"sha256": "d" * 64, "size_bytes": 123},
                    "image_file": {"sha256": "a" * 64},
                    "mask_file": {"sha256": "b" * 64},
                    "solver": iterated_rof_paper_like.EXPECTED_LOCAL_SOLVER,
                    "n_classes": 2,
                    "thresholds": [0.5],
                    "threshold_iterations": 1,
                    "rof_iterations": 2,
                    "rof_final_residual": 0.1,
                    "parameters": {"mu": 1500, "rof_n_iter": 2, "trof_max_iter": 1},
                    "source_claim": source_claim,
                }
            )

        gate = iterated_rof_paper_like._paper_like_gate(
            "ready_for_paper_like_runner",
            [],
            [],
            image_results,
            data_root=iterated_rof_paper_like.DATA_ROOT,
            dataset_fingerprint=iterated_rof_paper_like._dataset_fingerprint_from_image_results(image_results),
        )
        checklist = {item["id"]: item for item in gate["checklist"]}
        expected_reason = "Source claim missing source_audit for: reproduce/data/iterated_rof/cartoon/images/sample.png"

        self.assertIn("source_audit", checklist)
        self.assertIn(expected_reason, checklist["source_audit"]["reasons"])
        self.assertNotIn(expected_reason, checklist.get("other", {}).get("reasons", []))

    def test_family_summary_csv_flattens_metrics_and_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for family in iterated_rof_paper_like.DATA_FAMILIES:
                image = np.zeros((8, 8), dtype=float)
                image[:, 4:] = 1.0
                self._write_png(root / family / "images" / "binary.png", image)
                self._write_png(root / family / "masks" / "binary.png", image)
            self._write_license_reviewed_manifest(root)

            summary = iterated_rof_paper_like.run_local_dataset(
                root,
                rof_n_iter=8,
                trof_max_iter=4,
                figure_dir=root / "figures",
            )
            csv_path = root / "family_summary.csv"
            iterated_rof_paper_like.write_family_summary_csv(summary, csv_path)
            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["family"] for row in rows}, set(iterated_rof_paper_like.DATA_FAMILIES))
        for row in rows:
            self.assertEqual(row["status"], "completed_quantitative")
            self.assertNotEqual(row["metric_clustering_accuracy"], "")
            self.assertNotEqual(row["metric_dice"], "")
            self.assertNotEqual(row["baseline_raw_kmeans_clustering_accuracy"], "")
            self.assertNotEqual(row["baseline_multi_otsu_clustering_accuracy"], "")
            self.assertNotEqual(row["figure_paths"], "")

    def test_image_evidence_csv_writes_header_without_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = iterated_rof_paper_like.run_local_dataset(root, rof_n_iter=2, trof_max_iter=1)
            csv_path = root / "image_evidence.csv"
            iterated_rof_paper_like.write_image_evidence_csv(summary, csv_path)

            with csv_path.open(encoding="utf-8") as handle:
                header = handle.readline().strip().split(",")
                rows = list(csv.DictReader(handle, fieldnames=header))

        self.assertEqual(rows, [])
        self.assertIn("family", header)
        self.assertIn("image_path", header)
        self.assertIn("paper_like_gate_passed", header)
        self.assertIn("dataset_fingerprint_sha256", header)
        self.assertIn("baseline_raw_kmeans_clustering_accuracy", header)
        self.assertIn("baseline_raw_kmeans_dice", header)
        self.assertIn("error", header)

    def test_image_evidence_csv_flattens_per_image_metrics_and_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 8), dtype=float)
            image[:, 4:] = 1.0
            self._write_png(root / "cartoon" / "images" / "binary.png", image)
            self._write_png(root / "cartoon" / "masks" / "binary.png", image)
            self._write_png(root / "cartoon" / "images" / "qualitative.png", image.T)
            self._write_license_reviewed_manifest(root)

            summary = iterated_rof_paper_like.run_local_dataset(
                root,
                rof_n_iter=8,
                trof_max_iter=4,
                figure_dir=root / "figures",
            )
            csv_path = root / "image_evidence.csv"
            iterated_rof_paper_like.write_image_evidence_csv(summary, csv_path)

            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), len(summary["images"]))
        by_name = {Path(row["image_path"]).name: row for row in rows}
        quantitative = by_name["binary.png"]
        qualitative = by_name["qualitative.png"]
        self.assertEqual(quantitative["family"], "cartoon")
        self.assertEqual(quantitative["status"], "completed")
        self.assertEqual(quantitative["qualitative_only"], "False")
        self.assertNotEqual(quantitative["mask_path"], "")
        self.assertEqual(len(quantitative["image_sha256"]), 64)
        self.assertEqual(len(quantitative["mask_sha256"]), 64)
        self.assertEqual(quantitative["source_claim_sha256"], quantitative["image_sha256"])
        self.assertEqual(quantitative["source_claim_mask_sha256"], quantitative["mask_sha256"])
        self.assertEqual(quantitative["source_id"], "bsds500")
        self.assertEqual(quantitative["claim_scope"], "file")
        self.assertEqual(quantitative["paper_like_gate_passed"], "False")
        self.assertEqual(len(quantitative["dataset_fingerprint_sha256"]), 64)
        self.assertNotEqual(quantitative["clustering_accuracy"], "")
        self.assertNotEqual(quantitative["dice"], "")
        self.assertEqual(len(quantitative["figure_sha256"]), 64)
        self.assertGreater(int(quantitative["figure_size_bytes"]), 0)
        self.assertNotEqual(quantitative["figure_evidence_path"], "")
        self.assertEqual(len(quantitative["figure_evidence_sha256"]), 64)
        self.assertEqual(quantitative["figure_evidence_generator"], "iterated_rof_paper_like.figure_grid_v1")
        self.assertEqual(quantitative["baseline_raw_kmeans_method"], "simple_kmeans_on_raw_grayscale")
        self.assertNotEqual(quantitative["baseline_multi_otsu_method"], "")
        self.assertNotEqual(quantitative["baseline_multi_otsu_thresholds"], "")
        self.assertNotEqual(quantitative["baseline_raw_kmeans_clustering_accuracy"], "")
        self.assertNotEqual(quantitative["baseline_raw_kmeans_dice"], "")
        self.assertNotEqual(quantitative["baseline_multi_otsu_clustering_accuracy"], "")
        self.assertNotEqual(quantitative["baseline_multi_otsu_dice"], "")
        self.assertNotEqual(quantitative["figure_path"], "")
        self.assertIn("T-ROF", quantitative["figure_panels"])
        self.assertEqual(quantitative["mu"], "8.0")
        self.assertEqual(quantitative["rof_n_iter"], "8")
        self.assertEqual(quantitative["trof_max_iter"], "4")
        self.assertNotEqual(quantitative["thresholds"], "")
        self.assertEqual(qualitative["qualitative_only"], "True")
        self.assertEqual(qualitative["mask_path"], "")
        self.assertEqual(qualitative["mask_sha256"], "")
        self.assertEqual(qualitative["dice"], "")
        self.assertNotEqual(qualitative["figure_path"], "")

    def test_image_evidence_csv_recomputes_gate_instead_of_trusting_saved_gate(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            image = summary["images"][0]
            image["metrics"] = {}
            image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            csv_path = root / "image_evidence.csv"

            iterated_rof_paper_like.write_image_evidence_csv(summary, csv_path)

            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 3)
        self.assertEqual({row["paper_like_gate_passed"] for row in rows}, {"False"})

    def test_image_evidence_csv_keeps_failed_image_row_and_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "cartoon" / "images" / "broken.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"not a real image")

            summary = iterated_rof_paper_like.run_local_dataset(
                root,
                rof_n_iter=2,
                trof_max_iter=1,
                figure_dir=root / "figures",
            )
            csv_path = root / "image_evidence.csv"
            iterated_rof_paper_like.write_image_evidence_csv(summary, csv_path)

            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(Path(rows[0]["image_path"]).name, "broken.png")
        self.assertEqual(len(rows[0]["image_sha256"]), 64)
        self.assertNotEqual(rows[0]["error"], "")
        self.assertEqual(rows[0]["figure_path"], "")

    def test_image_evidence_csv_escapes_spreadsheet_formula_text(self):
        report = {
            "status": "completed_local_runner",
            "readiness_status": "ready_for_paper_like_runner",
            "paper_like_gate": {"passed": False},
            "dataset_fingerprint": {"sha256": "f" * 64, "file_count": 1},
            "images": [
                {
                    "family": "cartoon",
                    "status": "failed",
                    "image_path": "reproduce/data/iterated_rof/cartoon/images/bad.png",
                    "mask_path": None,
                    "qualitative_only": True,
                    "source_claim": {
                        "source_id": "=source",
                        "source_name": "+dataset",
                        "claim_scope": "file",
                        "manifest_status": "present",
                        "manifest_path": "@manifest",
                        "license_reviewed": True,
                        "citation": "-citation",
                        "license_note": "\tlicense",
                    },
                    "image_file": {"sha256": "a" * 64},
                    "metrics": {},
                    "error": "=load()",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "image_evidence.csv"
            iterated_rof_paper_like.write_image_evidence_csv(report, csv_path)
            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        row = rows[0]
        self.assertEqual(row["source_id"], "'=source")
        self.assertEqual(row["source_name"], "'+dataset")
        self.assertEqual(row["source_manifest_path"], "'@manifest")
        self.assertEqual(row["source_citation"], "'-citation")
        self.assertEqual(row["source_license_note"], "'\tlicense")
        self.assertEqual(row["error"], "'=load()")
        self.assertEqual(row["dataset_fingerprint_file_count"], "1")

    def test_csv_escapes_newline_prefixed_formula_text(self):
        row = {"error": "\n=load()"}

        escaped = list(iterated_rof_paper_like._csv_safe_rows([row]))[0]

        self.assertEqual(escaped["error"], "'\n=load()")

    def test_family_summary_csv_escapes_spreadsheet_formula_text(self):
        report = {
            "family_summaries": [
                {
                    "family": "cartoon",
                    "status": "failed",
                    "image_count": 1,
                    "mask_count": 0,
                    "matched_mask_count": 0,
                    "completed_image_count": 0,
                    "failed_image_count": 1,
                    "quantitative_image_count": 0,
                    "qualitative_image_count": 0,
                    "metrics_mean": {},
                    "baseline_metrics_mean": {},
                    "figure_paths": ["=figure.png"],
                    "source_claims": [{"source_id": "+source"}],
                    "errors": ["@error"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "family_summary.csv"
            iterated_rof_paper_like.write_family_summary_csv(report, csv_path)
            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        row = rows[0]
        self.assertEqual(row["source_ids"], "'+source")
        self.assertEqual(row["figure_paths"], "'=figure.png")
        self.assertEqual(row["errors"], "'@error")

    def test_cli_can_write_family_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "summary.json"
            csv_path = root / "family_summary.csv"
            image_csv_path = root / "image_evidence.csv"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--run",
                    "--output",
                    str(output_path),
                    "--family-summary-output",
                    str(csv_path),
                    "--image-evidence-output",
                    str(image_csv_path),
                ]
            )
            output_exists = output_path.exists()
            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)
        self.assertEqual(len(rows), 3)

    def test_cli_can_write_image_evidence_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "summary.json"
            csv_path = root / "image_evidence.csv"
            family_csv_path = root / "family_summary.csv"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--run",
                    "--output",
                    str(output_path),
                    "--family-summary-output",
                    str(family_csv_path),
                    "--image-evidence-output",
                    str(csv_path),
                ]
            )
            output_exists = output_path.exists()
            with csv_path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(exit_code, 0)
        self.assertTrue(output_exists)
        self.assertEqual(rows, [])

    def test_dashboard_candidate_blocks_without_paper_like_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = iterated_rof_paper_like.run_local_dataset(Path(tmp), rof_n_iter=2, trof_max_iter=1)
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertEqual(candidate["reproductionLevel"], "partial")
        self.assertEqual(candidate["paperLikeGate"]["passed"], False)
        self.assertTrue(candidate["blockedReasons"])
        self.assertEqual(candidate["candidateDetails"], {})

    def test_dashboard_candidate_contains_paper_like_fields_after_gate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_summary.test_{root.name}.json"
            )
            candidate_path = root / "dashboard_candidate.json"
            self.addCleanup(summary_path.unlink, missing_ok=True)
            iterated_rof_paper_like.write_report(summary, summary_path)
            candidate = iterated_rof_paper_like.write_dashboard_candidate(
                summary,
                candidate_path,
                source_summary_path=summary_path,
            )
            saved_candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

        self.assertTrue(candidate["can_promote"])
        self.assertEqual(candidate["priority"], 3)
        self.assertEqual(candidate["reproductionLevel"], "paper-like")
        self.assertEqual(candidate["reproductionTruthLevel"], "partial-completed")
        self.assertEqual(saved_candidate["can_promote"], True)
        self.assertIn("paperLikeGate", candidate)
        self.assertEqual(candidate["experimentId"], "iterated_rof_paper_like")
        self.assertEqual(candidate["candidateDetails"], candidate["dashboardDetailPatch"])
        self.assertEqual(candidate["dashboardDetailPatch"]["reproductionLevel"], "paper-like")
        self.assertEqual(candidate["dashboardDetailPatch"]["experimentId"], "iterated_rof_paper_like")
        self.assertEqual(candidate["candidateDetails"]["resultStatus"], "completed")
        self.assertEqual(candidate["candidateDetails"]["reproductionTruthLevel"], "partial-completed")
        self.assertIn("fidelityWarning", candidate["dashboardDetailPatch"])
        self.assertEqual(candidate["dashboardDetailPatch"]["paper_like_gate"], candidate["paperLikeGate"])
        self.assertIn("family_summaries", candidate["candidateDetails"])
        self.assertIn("dataset_fingerprint", candidate["candidateDetails"])
        self.assertEqual(candidate["candidateDetails"]["dataset_fingerprint"], summary["dataset_fingerprint"])
        self.assertEqual(
            candidate["candidateDetails"]["paper_like_verification"]["generated_by"],
            "iterated_rof_paper_like.dashboard_candidate_v1",
        )
        self.assertTrue(candidate["candidateDetails"]["paper_like_verification"]["recomputed_gate"])
        self.assertTrue(candidate["candidateDetails"]["paper_like_verification"]["can_promote"])
        self.assertEqual(candidate["candidateDetails"]["paper_like_verification"]["promotion_shape_blocker_count"], 0)
        self.assertEqual(candidate["candidateDetails"]["paper_like_verification"]["gate_id"], "iterated_rof_paper_like_v1")
        self.assertEqual(
            candidate["candidateDetails"]["paper_like_verification"]["dataset_fingerprint"],
            summary["dataset_fingerprint"],
        )
        self.assertEqual(
            candidate["candidateDetails"]["paper_like_verification"]["source_summary_path"],
            iterated_rof_paper_like._display_path(summary_path),
        )
        self.assertEqual(
            len(candidate["candidateDetails"]["paper_like_verification"]["source_summary_sha256"]),
            64,
        )
        self.assertIn("run_protocol", candidate["candidateDetails"])
        self.assertEqual(
            candidate["candidateDetails"]["run_protocol"]["threshold_update"],
            "tau_i = 0.5 * (mean_f(Omega_{i-1}) + mean_f(Omega_i))",
        )
        self.assertIn("runMetrics", candidate["candidateDetails"])
        self.assertIn("cartoon_clustering_accuracy", candidate["candidateDetails"]["runMetrics"])
        self.assertEqual(len(candidate["candidateDetails"]["resultFiles"]), 3)
        self.assertTrue(
            all(
                path.startswith("assets/repro/iterated_rof_paper_like/")
                for path in candidate["candidateDetails"]["resultFiles"]
            )
        )
        self.assertEqual(candidate["runResultPatch"]["priority"], 3)
        self.assertEqual(candidate["runResultPatch"]["id"], "iterated-rof")
        self.assertEqual(candidate["runResultPatch"]["experiment_id"], "iterated_rof_paper_like")
        self.assertEqual(candidate["runResultPatch"]["status"], "completed")
        self.assertEqual(candidate["runResultPatch"]["reproductionLevel"], "paper-like")
        self.assertEqual(candidate["runResultPatch"]["metrics"], candidate["candidateDetails"]["runMetrics"])
        self.assertEqual(candidate["runResultPatch"]["resultFiles"], candidate["candidateDetails"]["resultFiles"])
        self.assertEqual(
            candidate["runResultPatch"]["fidelityWarning"],
            candidate["dashboardDetailPatch"]["fidelityWarning"],
        )
        self.assertEqual(candidate["runResultPatch"]["paper_like_gate"], candidate["paperLikeGate"])
        self.assertEqual(
            candidate["runResultPatch"]["paper_like_verification"],
            candidate["candidateDetails"]["paper_like_verification"],
        )
        self.assertNotIn("reproductionTruthLevel", candidate["runResultPatch"])

    def test_dashboard_candidate_requires_source_summary_path_after_gate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            "Dashboard promotion candidate requires source summary artifact path",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_requires_source_summary_path_under_results(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            candidate = iterated_rof_paper_like.build_dashboard_candidate(
                summary,
                source_summary_path=summary_path,
            )

        self.assertFalse(candidate["can_promote"])
        self.assertEqual(candidate["reproductionLevel"], "partial")
        self.assertIn(
            "Dashboard promotion candidate source summary artifact must be under reproduce/results",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_rejects_unrelated_source_summary_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_summary.test_{root.name}.json"
            )
            self.addCleanup(summary_path.unlink, missing_ok=True)
            stale_summary = {
                **summary,
                "dataset_fingerprint": {
                    **summary["dataset_fingerprint"],
                    "sha256": "0" * 64,
                },
            }
            iterated_rof_paper_like.write_report(stale_summary, summary_path)

            candidate = iterated_rof_paper_like.build_dashboard_candidate(
                summary,
                source_summary_path=summary_path,
            )

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            "Dashboard promotion candidate source summary artifact dataset_fingerprint does not match current report",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_blocks_quantitative_rows_without_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._patch_iterated_rof_runtime_paths(root)
            token = f"_test_no_mask_{Path(tmp).name}"
            manifest_path = iterated_rof_paper_like.DATA_ROOT / "dataset_manifest.json"
            original_manifest = manifest_path.read_bytes() if manifest_path.exists() else None

            def restore_manifest():
                if original_manifest is None:
                    manifest_path.unlink(missing_ok=True)
                else:
                    manifest_path.write_bytes(original_manifest)

            self.addCleanup(restore_manifest)
            self.addCleanup(
                shutil.rmtree,
                iterated_rof_paper_like.FIGURE_DIR / token,
                ignore_errors=True,
            )

            image_results = []
            for index, family in enumerate(iterated_rof_paper_like.DATA_FAMILIES):
                image_path = iterated_rof_paper_like.DATA_ROOT / family / "images" / token / "sample.png"
                figure_path = iterated_rof_paper_like.FIGURE_DIR / token / f"{family}.png"
                self.addCleanup(
                    shutil.rmtree,
                    iterated_rof_paper_like.DATA_ROOT / family / "images" / token,
                    ignore_errors=True,
                )
                self.addCleanup(
                    shutil.rmtree,
                    iterated_rof_paper_like.DATA_ROOT / family / "audit" / token,
                    ignore_errors=True,
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                figure_path.parent.mkdir(parents=True, exist_ok=True)
                axis = np.linspace(0.0, 1.0, 64)
                grid_x, grid_y = np.meshgrid(axis, axis)
                image_values = (0.65 * grid_x + 0.35 * grid_y + 0.03 * index) % 1.0
                self._write_png(image_path, image_values)
                self._write_png(figure_path, np.linspace(0.0, 1.0, 256).reshape(16, 16))

                item = self._complete_image_result(
                    family,
                    image_path=iterated_rof_paper_like._display_path(image_path),
                    mask_path="",
                    figure_path=iterated_rof_paper_like._display_path(figure_path),
                )
                item["qualitative_only"] = False
                item["mask_path"] = ""
                item["image_file"] = iterated_rof_paper_like._file_evidence(image_path)
                item.pop("mask_file", None)
                item["source_claim"]["image"] = f"{token}/sample.png"
                item["source_claim"].pop("mask", None)
                item["source_claim"]["sha256"] = item["image_file"]["sha256"]
                item["source_claim"].pop("mask_sha256", None)
                item["source_claim"]["source_audit"] = self._source_audit(
                    family,
                    iterated_rof_paper_like.DATA_ROOT,
                    token,
                )
                item["figure_file"] = iterated_rof_paper_like._file_evidence(figure_path)
                item.update(iterated_rof_paper_like._write_figure_evidence_sidecar(item))
                image_results.append(item)

            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(
                    {
                        "families": {
                            family: {
                                "source_id": self.SOURCE_IDS[family],
                                "source_name": self.SOURCE_IDS[family],
                                "license_reviewed": True,
                                "license_note": "reviewed dataset license",
                                "citation": "recorded dataset citation",
                                "provenance_reviewed": True,
                                "provenance_note": "recorded dataset provenance",
                                "synthetic_fixture": False,
                                "source_audit": self._source_audit(family, iterated_rof_paper_like.DATA_ROOT, token),
                                "files": [
                                    {
                                        "image": f"{token}/sample.png",
                                        "sha256": next(
                                            item for item in image_results if item["family"] == family
                                        )["image_file"]["sha256"],
                                    }
                                ],
                            }
                            for family in iterated_rof_paper_like.DATA_FAMILIES
                        }
                    }
                ),
                encoding="utf-8",
            )
            summary = self._passed_gate_report()
            summary["images"] = image_results
            summary["data_root"] = iterated_rof_paper_like._display_path(iterated_rof_paper_like.DATA_ROOT)
            summary["families"] = [
                {
                    "family": family,
                    "description": iterated_rof_paper_like.DATA_FAMILIES[family],
                    "image_count": 1,
                    "mask_count": 0,
                    "matched_mask_count": 0,
                    "status": "ready_qualitative_only",
                }
                for family in iterated_rof_paper_like.DATA_FAMILIES
            ]
            summary["family_summaries"] = iterated_rof_paper_like._family_summaries(
                summary["families"],
                image_results,
            )
            summary["dataset_fingerprint"] = iterated_rof_paper_like._dataset_fingerprint_from_image_results(image_results)
            summary["image_count"] = 3
            summary["completed_image_count"] = 3
            summary["quantitative_image_count"] = 3
            summary_path = Path(tmp) / "summary.json"
            iterated_rof_paper_like.write_report(summary, summary_path)
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary, source_summary_path=summary_path)

        self.assertFalse(candidate["can_promote"])
        self.assertTrue(
            any(
                "requires completed quantitative image evidence rows with masks" in blocker
                or "Missing mask_path for quantitative evidence" in blocker
                for blocker in candidate["promotionShapeBlockers"] + candidate["paperLikeGate"]["reasons"]
            )
        )

    def test_dashboard_candidate_patches_pass_sync_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_summary.test_{root.name}.json"
            )
            self.addCleanup(summary_path.unlink, missing_ok=True)
            iterated_rof_paper_like.write_report(summary, summary_path)
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary, source_summary_path=summary_path)
            static_manifest_path = root / "dashboard_static_assets.json"
            static_manifest = iterated_rof_paper_like.write_dashboard_static_asset_manifest(
                summary,
                static_manifest_path,
                source_summary_path=summary_path,
                copy_assets=True,
            )
            self.assertEqual(static_manifest["status"], "current")
            source_registry_path = root / "reproduce" / "paper_like" / "iterated_rof_dataset_sources.json"
            source_registry_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(iterated_rof_paper_like.SOURCE_MANIFEST_PATH, source_registry_path)
            payload_path = root / "candidate_patch_payload.json"
            payload_path.write_text(
                json.dumps(
                    {
                        "dashboard": {
                            "reproAssessments": [
                                {
                                    "id": "iterated-rof",
                                    "priority": candidate["priority"],
                                    **candidate["dashboardDetailPatch"],
                                }
                            ]
                        },
                        "runResults": [candidate["runResultPatch"]],
                    }
                ),
                encoding="utf-8",
            )
            node_code = """
                import fs from 'node:fs';
                import { compareDashboardToResults } from './reproduce/sync_to_dashboard.mjs';
                const payload = JSON.parse(fs.readFileSync(process.argv[1], 'utf8'));
                const differences = compareDashboardToResults(payload.dashboard, payload.runResults);
                if (differences.length) {
                  console.error(JSON.stringify(differences, null, 2));
                  process.exit(1);
                }
            """
            completed = subprocess.run(
                ["node", "--input-type=module", "-e", node_code, str(payload_path)],
                cwd=iterated_rof_paper_like.REPO_ROOT,
                env={
                    **os.environ,
                    "REPRO_SYNC_REPO_ROOT": str(root),
                    "REPRO_SYNC_ALLOW_REPO_ROOT_OVERRIDE": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_dashboard_static_asset_manifest_reports_missing_assets_for_promotable_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_summary.test_{root.name}.json"
            )
            self.addCleanup(summary_path.unlink, missing_ok=True)
            iterated_rof_paper_like.write_report(summary, summary_path)

            manifest = iterated_rof_paper_like.build_dashboard_static_asset_manifest(
                summary,
                source_summary_path=summary_path,
            )

        self.assertTrue(manifest["can_promote"])
        self.assertEqual(manifest["status"], "missing")
        self.assertFalse(manifest["all_static_assets_current"])
        self.assertEqual(manifest["asset_count"], summary["quantitative_image_count"])
        self.assertEqual(
            sorted(item["result_file"] for item in manifest["assets"]),
            sorted(
                iterated_rof_paper_like._dashboard_result_file_for_figure(item["figure_path"])
                for item in summary["images"]
            ),
        )
        for asset in manifest["assets"]:
            self.assertEqual(asset["status"], "missing")
            self.assertTrue(asset["result_file"].startswith("assets/repro/iterated_rof_paper_like/"))
            self.assertEqual(len(asset["source_figure_sha256"]), 64)
            self.assertNotIn("static_asset_sha256", asset)

    def test_write_dashboard_static_asset_manifest_can_copy_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_summary.test_{root.name}.json"
            )
            manifest_path = root / "dashboard_static_assets.json"
            self.addCleanup(summary_path.unlink, missing_ok=True)
            iterated_rof_paper_like.write_report(summary, summary_path)

            manifest = iterated_rof_paper_like.write_dashboard_static_asset_manifest(
                summary,
                manifest_path,
                source_summary_path=summary_path,
                copy_assets=True,
            )
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for asset in manifest["assets"]:
                static_path = Path(asset["static_asset_path"])
                self.assertTrue(static_path.is_file())
                self.assertEqual(asset["static_asset_sha256"], asset["source_figure_sha256"])
                self.assertEqual(
                    iterated_rof_paper_like._file_evidence(static_path)["sha256"],
                    asset["source_figure_sha256"],
                )

        self.assertEqual(saved_manifest, manifest)
        self.assertTrue(manifest["copy_requested"])
        self.assertTrue(manifest["copy_performed"])
        self.assertTrue(manifest["all_static_assets_current"])
        self.assertEqual(manifest["status"], "current")

    def test_dashboard_static_asset_manifest_detects_stale_static_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_summary.test_{root.name}.json"
            )
            iterated_rof_paper_like.write_report(summary, summary_path)
            first_result_file = iterated_rof_paper_like._dashboard_result_file_for_figure(
                summary["images"][0]["figure_path"]
            )
            stale_path = iterated_rof_paper_like._dashboard_static_asset_path(first_result_file)
            stale_path.parent.mkdir(parents=True, exist_ok=True)
            stale_path.write_bytes(b"stale-static-figure")

            manifest = iterated_rof_paper_like.build_dashboard_static_asset_manifest(
                summary,
                source_summary_path=summary_path,
            )

        stale_rows = [asset for asset in manifest["assets"] if asset["result_file"] == first_result_file]
        self.assertEqual(manifest["status"], "stale")
        self.assertEqual(len(stale_rows), 1)
        self.assertEqual(stale_rows[0]["status"], "stale")
        self.assertNotEqual(stale_rows[0]["static_asset_sha256"], stale_rows[0]["source_figure_sha256"])

    def test_dashboard_static_asset_manifest_rejects_noncanonical_figure_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            image = summary["images"][0]
            noncanonical_figure = root / "noncanonical-figure.png"
            self._write_png(noncanonical_figure, np.linspace(0.0, 1.0, 256).reshape(16, 16))
            image["figure_path"] = iterated_rof_paper_like._display_path(noncanonical_figure)
            image["figure_file"] = iterated_rof_paper_like._file_evidence(noncanonical_figure)

            manifest = iterated_rof_paper_like.build_dashboard_static_asset_manifest(summary)

        invalid_rows = [
            asset
            for asset in manifest["assets"]
            if asset["image_path"] == image["image_path"]
        ]
        self.assertEqual(manifest["status"], "blocked_not_promotable")
        self.assertEqual(len(invalid_rows), 1)
        self.assertEqual(invalid_rows[0]["status"], "invalid_result_file")
        self.assertEqual(invalid_rows[0]["result_file"], "")
        self.assertEqual(invalid_rows[0]["static_asset_path"], "")

    def test_dashboard_candidate_result_files_match_static_manifest_assets_with_extra_qualitative_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_summary.test_{root.name}.json"
            )
            qualitative_image_path = iterated_rof_paper_like.DATA_ROOT / "cartoon" / "images" / "_qualitative" / "sample.png"
            qualitative_figure_path = iterated_rof_paper_like.FIGURE_DIR / "_qualitative" / "cartoon.png"
            self.addCleanup(shutil.rmtree, qualitative_image_path.parent, ignore_errors=True)
            self.addCleanup(shutil.rmtree, qualitative_figure_path.parent, ignore_errors=True)
            self._write_png(qualitative_image_path, np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64))
            self._write_png(qualitative_figure_path, np.linspace(0.0, 1.0, 256).reshape(16, 16))
            qualitative = self._complete_image_result(
                "cartoon",
                image_path=iterated_rof_paper_like._display_path(qualitative_image_path),
                mask_path="",
                figure_path=iterated_rof_paper_like._display_path(qualitative_figure_path),
            )
            qualitative["qualitative_only"] = True
            qualitative["mask_path"] = ""
            qualitative.pop("mask_file", None)
            qualitative["metrics"] = {}
            qualitative["image_file"] = iterated_rof_paper_like._file_evidence(qualitative_image_path)
            qualitative["figure_file"] = iterated_rof_paper_like._file_evidence(qualitative_figure_path)
            qualitative["source_claim"]["image"] = "_qualitative/sample.png"
            qualitative["source_claim"].pop("mask", None)
            qualitative["source_claim"]["sha256"] = qualitative["image_file"]["sha256"]
            qualitative["source_claim"].pop("mask_sha256", None)
            qualitative.update(iterated_rof_paper_like._write_figure_evidence_sidecar(qualitative))
            manifest_path = iterated_rof_paper_like.DATA_ROOT / "dataset_manifest.json"
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["families"]["cartoon"]["files"].append(
                {
                    "image": "_qualitative/sample.png",
                    "sha256": qualitative["image_file"]["sha256"],
                }
            )
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
            qualitative["source_claim"] = {
                **summary["images"][0]["source_claim"],
                "image": "_qualitative/sample.png",
                "mask": None,
                "sha256": qualitative["image_file"]["sha256"],
                "mask_sha256": None,
            }
            summary["images"].append(qualitative)
            summary["family_summaries"] = iterated_rof_paper_like._family_summaries(
                iterated_rof_paper_like._promotion_family_rows(summary["images"]),
                summary["images"],
            )
            summary["dataset_fingerprint"] = iterated_rof_paper_like._dataset_fingerprint_from_image_results(
                summary["images"]
            )
            summary["local_dataset_manifest"] = iterated_rof_paper_like.load_local_dataset_manifest(
                iterated_rof_paper_like.DATA_ROOT
            )
            summary["image_count"] = len(summary["images"])
            summary["completed_image_count"] = len(
                [item for item in summary["images"] if item.get("status") == "completed"]
            )
            summary["quantitative_image_count"] = len(
                [
                    item
                    for item in summary["images"]
                    if iterated_rof_paper_like._is_quantitative_image_result(item)
                ]
            )
            iterated_rof_paper_like.write_report(summary, summary_path)

            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary, source_summary_path=summary_path)
            static_manifest = iterated_rof_paper_like.build_dashboard_static_asset_manifest(
                summary,
                source_summary_path=summary_path,
            )

        qualitative_result_file = iterated_rof_paper_like._dashboard_result_file_for_figure(
            qualitative["figure_path"]
        )
        candidate_files = candidate["runResultPatch"]["resultFiles"]
        manifest_files = [asset["result_file"] for asset in static_manifest["assets"]]
        self.assertTrue(candidate["can_promote"])
        self.assertEqual(sorted(candidate_files), sorted(manifest_files))
        self.assertNotIn(qualitative_result_file, candidate_files)
        self.assertEqual(len(candidate_files), summary["quantitative_image_count"])

    def test_dashboard_static_asset_copy_is_blocked_without_promotable_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            manifest_path = root / "dashboard_static_assets.json"

            manifest = iterated_rof_paper_like.write_dashboard_static_asset_manifest(
                summary,
                manifest_path,
                copy_assets=True,
            )

        self.assertFalse(manifest["can_promote"])
        self.assertEqual(manifest["status"], "blocked_not_promotable")
        self.assertTrue(manifest["copy_requested"])
        self.assertFalse(manifest["copy_performed"])
        self.assertIn(
            "Dashboard static asset copy requires a promotable dashboard candidate",
            manifest["copy_blockers"],
        )
        self.assertEqual(
            list(iterated_rof_paper_like.DASHBOARD_REPRO_ASSET_ROOT.rglob("*")),
            [],
        )

    def test_paper_like_gate_blocks_constant_masks_as_quantitative_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            for image in summary["images"]:
                mask_path = iterated_rof_paper_like._resolve_report_path(image["mask_path"])
                self._write_png(mask_path, np.zeros((64, 64), dtype=float))
                image["mask_file"] = iterated_rof_paper_like._file_evidence(mask_path)
                image["source_claim"]["mask_sha256"] = image["mask_file"]["sha256"]
                image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))
            summary["dataset_fingerprint"] = iterated_rof_paper_like._dataset_fingerprint_from_image_results(summary["images"])

            gate = iterated_rof_paper_like._paper_like_gate(
                summary["readiness_status"],
                summary["blockers"],
                summary["claim_blockers"],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        self.assertFalse(gate["passed"])
        self.assertTrue(any("mask has fewer than two labels" in reason for reason in gate["reasons"]))

    def test_paper_like_gate_blocks_missing_per_image_runner_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            for key in [
                "solver",
                "thresholds",
                "threshold_iterations",
                "rof_iterations",
                "rof_final_residual",
                "parameters",
            ]:
                image.pop(key, None)
            image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))

            gate = iterated_rof_paper_like._paper_like_gate(
                summary["readiness_status"],
                summary["blockers"],
                summary["claim_blockers"],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        self.assertFalse(gate["passed"])
        self.assertTrue(any("Missing solver evidence" in reason for reason in gate["reasons"]))
        self.assertTrue(any("Missing threshold evidence" in reason for reason in gate["reasons"]))
        self.assertTrue(any("Missing ROF iteration evidence" in reason for reason in gate["reasons"]))

    def test_paper_like_gate_blocks_incomplete_baseline_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["baselines"]["raw_kmeans"].pop("method", None)
            image["baselines"]["multi_otsu"].pop("method", None)
            image["baselines"]["multi_otsu"].pop("thresholds", None)
            image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))

            gate = iterated_rof_paper_like._paper_like_gate(
                summary["readiness_status"],
                summary["blockers"],
                summary["claim_blockers"],
                summary["images"],
                data_root=iterated_rof_paper_like.DATA_ROOT,
                dataset_fingerprint=summary["dataset_fingerprint"],
            )

        self.assertFalse(gate["passed"])
        self.assertTrue(any("Missing raw_kmeans method evidence" in reason for reason in gate["reasons"]))
        self.assertTrue(any("Missing multi_otsu method evidence" in reason for reason in gate["reasons"]))
        self.assertTrue(any("Missing multi_otsu threshold evidence" in reason for reason in gate["reasons"]))

    def test_dashboard_candidate_blocks_fabricated_file_evidence(self):
        summary = self._passed_gate_report()
        candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            "Dashboard promotion candidate image evidence does not match disk for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_blocks_non_data_file_evidence(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            forged_path = Path(iterated_rof_paper_like.__file__).resolve()
            forged_evidence = iterated_rof_paper_like._file_evidence(forged_path)
            image = summary["images"][0]
            image["image_path"] = iterated_rof_paper_like._display_path(forged_path)
            image["mask_path"] = iterated_rof_paper_like._display_path(forged_path)
            image["figure_path"] = iterated_rof_paper_like._display_path(forged_path)
            image["image_file"] = forged_evidence
            image["mask_file"] = forged_evidence
            image["figure_file"] = forged_evidence
            image["source_claim"]["image"] = forged_path.name
            image["source_claim"]["mask"] = forged_path.name
            image["source_claim"]["sha256"] = forged_evidence["sha256"]
            image["source_claim"]["mask_sha256"] = forged_evidence["sha256"]
            summary["dataset_fingerprint"] = iterated_rof_paper_like._dataset_fingerprint_from_image_results(
                summary["images"]
            )
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            f"Dashboard promotion candidate image path is outside canonical family images directory for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )
        self.assertIn(
            f"Dashboard promotion candidate figure path is outside allowed figure directory for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_blocks_evidence_path_mismatch(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["image_file"] = {**image["image_file"], "path": "forged/image.png"}
            image["mask_file"] = {**image["mask_file"], "path": "forged/mask.png"}
            image["figure_file"] = {**image["figure_file"], "path": "forged/figure.png"}
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            f"Dashboard promotion candidate image evidence does not match disk for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )
        self.assertIn(
            f"Dashboard promotion candidate mask evidence does not match disk for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )
        self.assertIn(
            f"Dashboard promotion candidate figure evidence does not match disk for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_blocks_unknown_source_id_even_when_gate_claim_is_precomputed(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["source_claim"]["source_id"] = "forged-source"
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            f"Dashboard promotion candidate source_id is not in source registry for: {image['image_path']}: forged-source",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_blocks_figure_evidence_sidecar_mismatch(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["figure_evidence"] = {
                **image["figure_evidence"],
                "figure_panels": ["input"],
            }
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            f"Dashboard promotion candidate figure evidence sidecar does not match report for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_blocks_dataset_fingerprint_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            summary["dataset_fingerprint"] = {
                **summary["dataset_fingerprint"],
                "sha256": "0" * 64,
            }
            summary["paper_like_gate"] = {
                **summary["paper_like_gate"],
                "passed": True,
                "reasons": [],
            }
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            "Dashboard promotion candidate dataset fingerprint does not match image/mask evidence",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_blocks_forged_family_summaries(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            summary["family_summaries"][0] = {
                **summary["family_summaries"][0],
                "completed_image_count": 99,
                "quantitative_image_count": 99,
                "metrics_mean": {"clustering_accuracy": 0.1234},
                "figure_paths": ["reproduce/results/figures/iterated_rof_paper_like/forged.png"],
            }
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertEqual(candidate["candidateDetails"], {})
        self.assertIn(
            "Dashboard promotion candidate family_summaries do not match image evidence rows",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_blocks_forged_top_level_image_counts(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            summary["image_count"] = 99
            summary["completed_image_count"] = 98
            summary["quantitative_image_count"] = 97
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            "Dashboard promotion candidate image_count does not match evidence rows",
            candidate["promotionShapeBlockers"],
        )
        self.assertIn(
            "Dashboard promotion candidate completed_image_count does not match evidence rows",
            candidate["promotionShapeBlockers"],
        )
        self.assertIn(
            "Dashboard promotion candidate quantitative image count does not match evidence rows",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_rejects_protocol_solver_or_threshold_update_mismatch(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            summary["run_protocol"] = {
                **summary["run_protocol"],
                "solver": "forged solver",
                "threshold_update": "forged threshold update",
            }
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            "Dashboard promotion candidate run_protocol solver does not match expected Iterated ROF runner",
            candidate["promotionShapeBlockers"],
        )
        self.assertIn(
            "Dashboard promotion candidate run_protocol threshold_update does not match expected Iterated ROF runner",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_rejects_noncanonical_run_protocol_figure_dir(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            summary["run_protocol"] = {
                **summary["run_protocol"],
                "figure_dir": str(Path(tmp) / "figures"),
            }
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            "Dashboard promotion candidate run_protocol figure_dir does not match expected Iterated ROF runner",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_rejects_run_protocol_parameter_drift_from_image_evidence(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            summary["run_protocol"] = {
                **summary["run_protocol"],
                "parameters": {
                    **summary["run_protocol"]["parameters"],
                    "rof_n_iter": summary["images"][0]["parameters"]["rof_n_iter"] + 1,
                    "trof_max_iter": float(summary["images"][0]["parameters"]["trof_max_iter"]),
                },
            }
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            "Dashboard promotion candidate run_protocol parameter rof_n_iter does not match completed image evidence",
            candidate["promotionShapeBlockers"],
        )
        self.assertIn(
            "Dashboard promotion candidate run_protocol parameter trof_max_iter must be a positive integer",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_records_source_summary_sha_from_repo_relative_path(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_summary.test_{root.name}.json"
            )
            self.addCleanup(summary_path.unlink, missing_ok=True)
            iterated_rof_paper_like.write_report(summary, summary_path)
            relative_summary_path = summary_path.relative_to(iterated_rof_paper_like.REPO_ROOT).as_posix()

            original_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                candidate = iterated_rof_paper_like.build_dashboard_candidate(
                    summary,
                    source_summary_path=relative_summary_path,
                )
            finally:
                os.chdir(original_cwd)

        self.assertTrue(candidate["can_promote"])
        verification = candidate["candidateDetails"]["paper_like_verification"]
        self.assertEqual(verification["source_summary_path"], relative_summary_path)
        self.assertEqual(len(verification["source_summary_sha256"]), 64)

    def test_dashboard_candidate_rejects_non_decodable_image_and_mask_evidence(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image_path = iterated_rof_paper_like._resolve_report_path(image["image_path"])
            mask_path = iterated_rof_paper_like._resolve_report_path(image["mask_path"])
            image_path.write_bytes(b"not a decodable image")
            mask_path.write_bytes(b"not a decodable mask")
            image["image_file"] = iterated_rof_paper_like._file_evidence(image_path)
            image["mask_file"] = iterated_rof_paper_like._file_evidence(mask_path)
            image["source_claim"]["sha256"] = image["image_file"]["sha256"]
            image["source_claim"]["mask_sha256"] = image["mask_file"]["sha256"]
            manifest_path = iterated_rof_paper_like.DATA_ROOT / "dataset_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            file_claim = manifest["families"][image["family"]]["files"][0]
            file_claim["sha256"] = image["image_file"]["sha256"]
            file_claim["mask_sha256"] = image["mask_file"]["sha256"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))
            summary["dataset_fingerprint"] = iterated_rof_paper_like._dataset_fingerprint_from_image_results(
                summary["images"]
            )
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            f"Dashboard promotion candidate image file is not decodable for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )
        self.assertIn(
            f"Dashboard promotion candidate mask file is not decodable for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_rejects_blank_figure_with_matching_sidecar(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            figure_path = iterated_rof_paper_like._resolve_report_path(image["figure_path"])
            self._write_png(figure_path, np.zeros((16, 16)))
            image["figure_file"] = iterated_rof_paper_like._file_evidence(figure_path)
            image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            f"Dashboard promotion candidate figure file is visually blank for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_rejects_fixture_text_in_saved_source_claim(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["source_claim"]["license_note"] = "Synthetic tempfile test fixture, not a real dataset claim."
            summary["family_summaries"] = iterated_rof_paper_like._family_summaries(
                iterated_rof_paper_like._promotion_family_rows(summary["images"]),
                summary["images"],
            )
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        expected_reason = f"Source claim contains fixture/tempfile text in license_note for: {image['image_path']}"
        self.assertFalse(candidate["can_promote"])
        self.assertFalse(candidate["paperLikeGate"]["passed"])
        self.assertIn(expected_reason, candidate["paperLikeGate"]["reasons"])

    def test_dashboard_candidate_rejects_template_placeholder_text_in_saved_source_claim(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["source_claim"]["provenance_note"] = (
                "Record how local files were obtained from this source, including download page, "
                "date, filtering, and any conversion steps."
            )
            summary["family_summaries"] = iterated_rof_paper_like._family_summaries(
                iterated_rof_paper_like._promotion_family_rows(summary["images"]),
                summary["images"],
            )
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        expected_reason = f"Source claim contains template placeholder text in provenance_note for: {image['image_path']}"
        self.assertFalse(candidate["can_promote"])
        self.assertFalse(candidate["paperLikeGate"]["passed"])
        self.assertIn(expected_reason, candidate["paperLikeGate"]["reasons"])

    def test_dashboard_candidate_recomputes_gate_instead_of_trusting_report_gate(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["metrics"] = {}
            image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        expected_reason = f"Missing T-ROF clustering_accuracy metric for: {image['image_path']}"
        self.assertFalse(candidate["can_promote"])
        self.assertFalse(candidate["paperLikeGate"]["passed"])
        self.assertIn(expected_reason, candidate["paperLikeGate"]["reasons"])
        self.assertIn(expected_reason, candidate["blockedReasons"])

    def test_promotion_audit_recomputes_gate_instead_of_trusting_report_gate(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["baselines"]["raw_kmeans"] = {"metrics": {}}
            image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            audit = iterated_rof_paper_like.build_promotion_audit(summary)

        expected_reason = f"Missing raw_kmeans clustering_accuracy baseline metric for: {image['image_path']}"
        self.assertFalse(audit["can_promote"])
        self.assertIn(expected_reason, audit["blocked_reasons"])
        self.assertGreater(audit["blocked_reason_count"], 0)

    def test_promotion_audit_summarizes_blocked_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = iterated_rof_paper_like.run_local_dataset(Path(tmp), rof_n_iter=2, trof_max_iter=1)
            audit = iterated_rof_paper_like.build_promotion_audit(summary)

        self.assertFalse(audit["can_promote"])
        self.assertEqual(audit["recommended_dashboard_level"], "partial")
        self.assertEqual(audit["dataset_fingerprint"]["file_count"], 0)
        self.assertIn("readiness_clean", audit["checklist"])
        self.assertFalse(audit["checklist"]["readiness_clean"]["passed"])
        self.assertEqual(audit["family_status_counts"], {"missing": 3})
        self.assertGreater(audit["blocked_reason_count"], 0)
        self.assertFalse(audit["ready_for_local_runner"])
        self.assertGreater(audit["data_ready_blocker_count"], 0)
        self.assertIn("local dataset manifest is not present", audit["data_ready_blockers"])

    def test_promotion_audit_requires_source_summary_artifact_for_promotable_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            audit = iterated_rof_paper_like.build_promotion_audit(summary)

        self.assertFalse(audit["can_promote"])
        self.assertEqual(audit["recommended_dashboard_level"], "partial")
        self.assertEqual(audit["dataset_fingerprint"], summary["dataset_fingerprint"])
        self.assertEqual(audit["family_status_counts"], {"completed_quantitative": 3})
        self.assertEqual(audit["blocked_reason_count"], 0)
        self.assertEqual(audit["promotion_shape_blocker_count"], 1)
        self.assertIn(
            "Dashboard promotion candidate requires source summary artifact path",
            audit["promotion_shape_blockers"],
        )
        self.assertTrue(audit["ready_for_local_runner"])
        self.assertEqual(audit["data_ready_blocker_count"], 0)

    def test_promotion_audit_summarizes_passed_gate_with_source_summary_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_summary.test_{root.name}.json"
            )
            self.addCleanup(summary_path.unlink, missing_ok=True)
            iterated_rof_paper_like.write_report(summary, summary_path)
            audit = iterated_rof_paper_like.build_promotion_audit(
                summary,
                source_summary_path=summary_path,
            )

        self.assertTrue(audit["can_promote"])
        self.assertEqual(audit["recommended_dashboard_level"], "paper-like")
        self.assertEqual(audit["dataset_fingerprint"], summary["dataset_fingerprint"])
        self.assertEqual(audit["family_status_counts"], {"completed_quantitative": 3})
        self.assertEqual(audit["blocked_reason_count"], 0)
        self.assertEqual(audit["promotion_shape_blocker_count"], 0)
        self.assertTrue(audit["ready_for_local_runner"])
        self.assertEqual(audit["data_ready_blocker_count"], 0)

    def test_promotion_audit_summarizes_source_audit_artifact_status_by_family(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            cartoon_image = next(item for item in summary["images"] if item["family"] == "cartoon")
            source_artifact = Path(cartoon_image["source_claim"]["source_audit"]["source_artifact_path"])
            source_artifact.unlink()

            audit = iterated_rof_paper_like.build_promotion_audit(summary)

        audit_by_family = {
            item["family"]: item
            for item in audit["source_audit_by_family"]
        }
        self.assertFalse(audit["can_promote"])
        self.assertEqual(audit["source_audit_status_counts"]["incomplete"], 1)
        self.assertEqual(audit["source_audit_status_counts"]["complete"], 2)
        self.assertEqual(audit_by_family["cartoon"]["status"], "incomplete")
        self.assertEqual(
            audit_by_family["cartoon"]["artifacts"]["source_artifact"]["path_status"],
            "missing_file",
        )
        self.assertEqual(audit_by_family["texture"]["status"], "complete")
        self.assertEqual(audit_by_family["medical"]["status"], "complete")
        self.assertIn(
            "cartoon source_audit source_artifact file is missing",
            audit["data_ready_blockers"],
        )

    def test_promotion_audit_summarizes_file_level_source_audit_overrides_by_family(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            token = f"_test_{root.name}"
            override_namespace = f"{token}-file-override"
            self.addCleanup(
                shutil.rmtree,
                iterated_rof_paper_like.DATA_ROOT / "cartoon" / "audit" / override_namespace,
                ignore_errors=True,
            )
            file_override = self._source_audit(
                "cartoon",
                iterated_rof_paper_like.DATA_ROOT,
                override_namespace,
            )
            Path(file_override["source_artifact_path"]).unlink()

            manifest_path = iterated_rof_paper_like.DATA_ROOT / "dataset_manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload["families"]["cartoon"]["files"][0]["source_audit"] = file_override
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            summary["local_dataset_manifest"] = iterated_rof_paper_like.load_local_dataset_manifest(
                iterated_rof_paper_like.DATA_ROOT
            )
            cartoon_image = next(item for item in summary["images"] if item["family"] == "cartoon")
            cartoon_image["source_claim"]["source_audit"] = file_override
            summary["family_summaries"] = iterated_rof_paper_like._family_summaries(
                iterated_rof_paper_like._promotion_family_rows(summary["images"]),
                summary["images"],
            )

            audit = iterated_rof_paper_like.build_promotion_audit(summary)

        audit_by_family = {
            item["family"]: item
            for item in audit["source_audit_by_family"]
        }
        cartoon_audit = audit_by_family["cartoon"]
        self.assertFalse(audit["can_promote"])
        self.assertEqual(cartoon_audit["status"], "incomplete")
        self.assertEqual(cartoon_audit["file_overrides"][0]["image"], f"{token}/sample.png")
        self.assertEqual(cartoon_audit["file_overrides"][0]["status"], "incomplete")
        self.assertEqual(
            cartoon_audit["file_overrides"][0]["artifacts"]["source_artifact"]["path_status"],
            "missing_file",
        )
        self.assertEqual(
            cartoon_audit["artifacts"]["source_artifact"]["path_status"],
            "present",
        )
        self.assertEqual(audit["source_audit_status_counts"]["incomplete"], 1)
        self.assertEqual(audit["source_audit_status_counts"]["complete"], 2)
        self.assertEqual(audit_by_family["texture"]["status"], "complete")
        self.assertIn(
            f"cartoon/{token}/sample.png source_audit source_artifact file is missing",
            audit["data_ready_blockers"],
        )

    def test_promotion_audit_recomputes_data_gap_instead_of_trusting_saved_checklist(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            summary["data_gap_checklist"] = iterated_rof_paper_like.build_data_gap_checklist(summary)
            self.assertTrue(summary["data_gap_checklist"]["ready_for_local_runner"])
            cartoon_image = next(item for item in summary["images"] if item["family"] == "cartoon")
            source_artifact = Path(cartoon_image["source_claim"]["source_audit"]["source_artifact_path"])
            source_artifact.unlink()

            audit = iterated_rof_paper_like.build_promotion_audit(summary)

        self.assertFalse(audit["ready_for_local_runner"])
        self.assertIn(
            "cartoon source_audit source_artifact file is missing",
            audit["data_ready_blockers"],
        )
        audit_by_family = {
            item["family"]: item
            for item in audit["source_audit_by_family"]
        }
        self.assertEqual(audit_by_family["cartoon"]["status"], "incomplete")
        self.assertEqual(
            audit_by_family["cartoon"]["artifacts"]["source_artifact"]["path_status"],
            "missing_file",
        )

    def test_promotion_audit_blocks_fabricated_file_evidence(self):
        summary = self._passed_gate_report()
        audit = iterated_rof_paper_like.build_promotion_audit(summary)

        self.assertFalse(audit["can_promote"])
        self.assertEqual(audit["recommended_dashboard_level"], "partial")
        self.assertGreater(audit["promotion_shape_blocker_count"], 0)
        self.assertIn(
            "Dashboard promotion candidate image evidence does not match disk for: reproduce/data/iterated_rof/cartoon/images/sample.png",
            audit["promotion_shape_blockers"],
        )

    def test_dashboard_candidate_blocks_extra_unmanifested_current_local_image(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            token = f"_unreviewed_{Path(tmp).name}"
            extra_image_path = iterated_rof_paper_like.DATA_ROOT / "cartoon" / "images" / token / "unreviewed.png"
            extra_mask_path = iterated_rof_paper_like.DATA_ROOT / "cartoon" / "masks" / token / "unreviewed.png"
            self.addCleanup(shutil.rmtree, extra_image_path.parent, ignore_errors=True)
            self.addCleanup(shutil.rmtree, extra_mask_path.parent, ignore_errors=True)
            image = np.linspace(0.0, 1.0, 64 * 64).reshape(64, 64)
            self._write_png(extra_image_path, image)
            self._write_png(extra_mask_path, image > 0.5)
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            "Dashboard promotion candidate current local data root fingerprint does not match report dataset_fingerprint",
            candidate["promotionShapeBlockers"],
        )
        self.assertIn(
            f"Local dataset manifest missing file claim for: cartoon/{token}/unreviewed.png",
            candidate["promotionShapeBlockers"],
        )

    def test_dashboard_candidate_blocks_saved_source_claim_that_differs_from_current_manifest(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            summary = self._passed_gate_report_with_matching_local_files(Path(tmp))
            image = summary["images"][0]
            image["source_claim"]["license_note"] = "reviewed dataset license in stale saved summary"
            summary["family_summaries"] = iterated_rof_paper_like._family_summaries(
                iterated_rof_paper_like._promotion_family_rows(summary["images"]),
                summary["images"],
            )
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary)

        self.assertFalse(candidate["can_promote"])
        self.assertIn(
            f"Dashboard promotion candidate saved source claim does not match current manifest for: {image['image_path']}",
            candidate["promotionShapeBlockers"],
        )

    def test_cli_can_write_promotion_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "summary.json"
            audit_path = root / "promotion_audit.json"
            family_csv_path = root / "family_summary.csv"
            image_csv_path = root / "image_evidence.csv"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--run",
                    "--output",
                    str(output_path),
                    "--family-summary-output",
                    str(family_csv_path),
                    "--image-evidence-output",
                    str(image_csv_path),
                    "--promotion-audit-output",
                    str(audit_path),
                ]
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertFalse(audit["can_promote"])
        self.assertEqual(audit["recommended_dashboard_level"], "partial")
        self.assertIn("runner_outputs", audit["checklist"])

    def test_cli_can_write_dashboard_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "summary.json"
            candidate_path = root / "dashboard_candidate.json"
            family_csv_path = root / "family_summary.csv"
            image_csv_path = root / "image_evidence.csv"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--run",
                    "--output",
                    str(output_path),
                    "--family-summary-output",
                    str(family_csv_path),
                    "--image-evidence-output",
                    str(image_csv_path),
                    "--dashboard-candidate-output",
                    str(candidate_path),
                ]
            )
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertFalse(candidate["can_promote"])
        self.assertEqual(candidate["reproductionLevel"], "partial")

    def test_cli_verify_summary_recomputes_gate_and_writes_candidate_and_audit(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = root / "summary.json"
            verified_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_verified_summary.test_{root.name}.json"
            )
            candidate_path = root / "candidate.json"
            audit_path = root / "audit.json"
            self.addCleanup(verified_path.unlink, missing_ok=True)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            exit_code = iterated_rof_paper_like.main(
                [
                    "--verify-summary",
                    str(summary_path),
                    "--output",
                    str(verified_path),
                    "--dashboard-candidate-output",
                    str(candidate_path),
                    "--promotion-audit-output",
                    str(audit_path),
                ]
            )
            verified = json.loads(verified_path.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(verified["paper_like_gate"]["passed"])
        self.assertEqual(verified["summary_verification"]["status"], "verified_promotable")
        self.assertTrue(verified["summary_verification"]["can_promote"])
        self.assertEqual(
            verified["summary_verification"]["source_summary_path"],
            iterated_rof_paper_like._display_path(verified_path),
        )
        self.assertTrue(candidate["can_promote"])
        self.assertTrue(audit["can_promote"])

    def test_cli_can_write_and_copy_dashboard_static_assets_from_verified_summary(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary_path = root / "summary.json"
            verified_path = (
                iterated_rof_paper_like.RESULTS_DIR
                / f"iterated_rof_paper_like_verified_summary.test_{root.name}.json"
            )
            static_manifest_path = root / "dashboard_static_assets.json"
            self.addCleanup(verified_path.unlink, missing_ok=True)
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            exit_code = iterated_rof_paper_like.main(
                [
                    "--verify-summary",
                    str(summary_path),
                    "--output",
                    str(verified_path),
                    "--dashboard-static-assets-output",
                    str(static_manifest_path),
                    "--copy-dashboard-static-assets",
                ]
            )
            static_manifest = json.loads(static_manifest_path.read_text(encoding="utf-8"))
            for asset in static_manifest["assets"]:
                self.assertTrue(Path(asset["static_asset_path"]).is_file())

        self.assertEqual(exit_code, 0)
        self.assertEqual(static_manifest["status"], "current")
        self.assertTrue(static_manifest["copy_performed"])
        self.assertTrue(static_manifest["all_static_assets_current"])

    def test_cli_verify_summary_rejects_forged_saved_gate(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            image = summary["images"][0]
            image["metrics"] = {}
            image.update(iterated_rof_paper_like._write_figure_evidence_sidecar(image))
            summary["paper_like_gate"] = {"passed": True, "reasons": []}
            summary_path = root / "summary.json"
            verified_path = root / "verified_summary.json"
            audit_path = root / "audit.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            exit_code = iterated_rof_paper_like.main(
                [
                    "--verify-summary",
                    str(summary_path),
                    "--output",
                    str(verified_path),
                    "--promotion-audit-output",
                    str(audit_path),
                ]
            )
            verified = json.loads(verified_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        expected_reason = f"Missing T-ROF clustering_accuracy metric for: {image['image_path']}"
        self.assertEqual(exit_code, 1)
        self.assertFalse(verified["paper_like_gate"]["passed"])
        self.assertIn(expected_reason, verified["paper_like_gate"]["reasons"])
        self.assertFalse(audit["can_promote"])
        self.assertIn(expected_reason, audit["blocked_reasons"])

    def test_cli_verify_summary_rejects_forged_family_summaries(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            summary["family_summaries"][0] = {
                **summary["family_summaries"][0],
                "metrics_mean": {"clustering_accuracy": 0.1234},
                "figure_paths": ["reproduce/results/figures/iterated_rof_paper_like/forged.png"],
            }
            summary_path = root / "summary.json"
            verified_path = root / "verified_summary.json"
            candidate_path = root / "candidate.json"
            audit_path = root / "audit.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            exit_code = iterated_rof_paper_like.main(
                [
                    "--verify-summary",
                    str(summary_path),
                    "--output",
                    str(verified_path),
                    "--dashboard-candidate-output",
                    str(candidate_path),
                    "--promotion-audit-output",
                    str(audit_path),
                ]
            )
            verified = json.loads(verified_path.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        expected_blocker = "Dashboard promotion candidate family_summaries do not match image evidence rows"
        self.assertEqual(exit_code, 1)
        self.assertEqual(verified["summary_verification"]["status"], "blocked")
        self.assertTrue(verified["paper_like_gate"]["passed"])
        self.assertFalse(verified["summary_verification"]["can_promote"])
        self.assertFalse(candidate["can_promote"])
        self.assertIn(expected_blocker, candidate["promotionShapeBlockers"])
        self.assertFalse(audit["can_promote"])
        self.assertIn(expected_blocker, audit["promotion_shape_blockers"])

    def test_cli_verify_summary_rejects_missing_manifest_on_disk(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            summary = self._passed_gate_report_with_matching_local_files(root)
            (iterated_rof_paper_like.DATA_ROOT / "dataset_manifest.json").unlink()
            summary_path = root / "summary.json"
            verified_path = root / "verified_summary.json"
            candidate_path = root / "candidate.json"
            audit_path = root / "audit.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            exit_code = iterated_rof_paper_like.main(
                [
                    "--verify-summary",
                    str(summary_path),
                    "--output",
                    str(verified_path),
                    "--dashboard-candidate-output",
                    str(candidate_path),
                    "--promotion-audit-output",
                    str(audit_path),
                ]
            )
            verified = json.loads(verified_path.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        expected_blocker = (
            "Local dataset manifest missing for present images: "
            f"add {iterated_rof_paper_like._display_path(iterated_rof_paper_like.DATA_ROOT / 'dataset_manifest.json')} "
            "before paper-like promotion"
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(verified["summary_verification"]["status"], "blocked")
        self.assertFalse(candidate["can_promote"])
        self.assertIn(expected_blocker, candidate["promotionShapeBlockers"])
        self.assertFalse(audit["can_promote"])
        self.assertIn(expected_blocker, audit["promotion_shape_blockers"])

    def test_cli_strict_paper_like_fails_when_gate_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "summary.json"
            csv_path = root / "family_summary.csv"
            image_csv_path = root / "image_evidence.csv"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(root),
                    "--run",
                    "--output",
                    str(output_path),
                    "--family-summary-output",
                    str(csv_path),
                    "--image-evidence-output",
                    str(image_csv_path),
                    "--strict-paper-like",
                ]
            )
            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(summary["paper_like_gate"]["passed"])
        self.assertTrue(summary["paper_like_gate"]["reasons"])

    def test_cli_strict_paper_like_requires_promotable_dashboard_candidate(self):
        with tempfile.TemporaryDirectory(dir=iterated_rof_paper_like.REPO_ROOT) as tmp:
            root = Path(tmp)
            self._passed_gate_report_with_matching_local_files(root)
            output_path = root / "summary.json"
            csv_path = root / "family_summary.csv"
            image_csv_path = root / "image_evidence.csv"
            noncanonical_figure_dir = root / "noncanonical_figures"

            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(iterated_rof_paper_like.DATA_ROOT),
                    "--run",
                    "--figure-dir",
                    str(noncanonical_figure_dir),
                    "--rof-iterations",
                    "4",
                    "--trof-iterations",
                    "2",
                    "--output",
                    str(output_path),
                    "--family-summary-output",
                    str(csv_path),
                    "--image-evidence-output",
                    str(image_csv_path),
                    "--strict-paper-like",
                ]
            )
            summary = json.loads(output_path.read_text(encoding="utf-8"))
            candidate = iterated_rof_paper_like.build_dashboard_candidate(summary, source_summary_path=output_path)

        self.assertEqual(exit_code, 1)
        self.assertTrue(summary["paper_like_gate"]["passed"])
        self.assertFalse(candidate["can_promote"])
        self.assertTrue(
            any("run_protocol figure_dir" in blocker for blocker in candidate["promotionShapeBlockers"])
        )

    def test_cli_strict_paper_like_rejects_tiny_binary_scaffold_even_when_manifest_claims_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            figure_dir = root / "figures"
            for index, family in enumerate(iterated_rof_paper_like.DATA_FAMILIES):
                image = np.zeros((16, 16), dtype=float)
                split = 5 + index
                image[:, split:] = 1.0
                if family == "texture":
                    image[::2, :] = 1.0 - image[::2, :]
                if family == "medical":
                    image = np.flipud(image)
                self._write_png(data_root / family / "images" / "binary.png", image)
                self._write_png(data_root / family / "masks" / "binary.png", image)
            self._write_license_reviewed_manifest(data_root)

            output_path = root / "summary.json"
            verified_path = root / "verified_summary.json"
            candidate_path = root / "candidate.json"
            audit_path = root / "audit.json"
            csv_path = root / "family_summary.csv"
            image_csv_path = root / "image_evidence.csv"
            original_data_root = iterated_rof_paper_like.DATA_ROOT
            original_figure_dir = iterated_rof_paper_like.FIGURE_DIR
            iterated_rof_paper_like.DATA_ROOT = data_root
            iterated_rof_paper_like.FIGURE_DIR = figure_dir
            try:
                exit_code = iterated_rof_paper_like.main(
                    [
                        "--data-root",
                        str(data_root),
                        "--run",
                        "--figure-dir",
                        str(figure_dir),
                        "--output",
                        str(output_path),
                        "--family-summary-output",
                        str(csv_path),
                        "--image-evidence-output",
                        str(image_csv_path),
                        "--strict-paper-like",
                    ]
                )
                verify_exit_code = iterated_rof_paper_like.main(
                    [
                        "--verify-summary",
                        str(output_path),
                        "--output",
                        str(verified_path),
                        "--dashboard-candidate-output",
                        str(candidate_path),
                        "--promotion-audit-output",
                        str(audit_path),
                    ]
                )
            finally:
                iterated_rof_paper_like.DATA_ROOT = original_data_root
                iterated_rof_paper_like.FIGURE_DIR = original_figure_dir
            summary = json.loads(output_path.read_text(encoding="utf-8"))
            verified = json.loads(verified_path.read_text(encoding="utf-8"))
            candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
            audit = json.loads(audit_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertEqual(verify_exit_code, 1)
        self.assertEqual(summary["status"], "ready_for_paper_like_runner")
        self.assertEqual(summary["data_ready_status"], "blocked_data_ready")
        self.assertFalse(summary["data_gap_checklist"]["ready_for_local_runner"])
        self.assertFalse(csv_path.exists())
        self.assertFalse(image_csv_path.exists())
        self.assertFalse(summary["paper_like_gate"]["passed"])
        self.assertEqual(verified["summary_verification"]["status"], "blocked")
        self.assertFalse(candidate["can_promote"])
        self.assertFalse(audit["can_promote"])
        self.assertTrue(
            any(
                "input image is too small for paper-like evidence" in blocker
                for blocker in summary["data_gap_checklist"]["data_ready_blockers"]
            )
        )

    def test_cli_strict_data_ready_rejects_constant_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_root = root / "data"
            axis = np.linspace(0.0, 1.0, 64)
            grid_x, grid_y = np.meshgrid(axis, axis)
            for index, family in enumerate(iterated_rof_paper_like.DATA_FAMILIES):
                image = (0.65 * grid_x + 0.35 * grid_y + 0.05 * index) % 1.0
                mask = np.zeros((64, 64), dtype=float)
                self._write_png(data_root / family / "images" / "sample.png", image)
                self._write_png(data_root / family / "masks" / "sample.png", mask)
            self._write_license_reviewed_manifest(data_root)

            readiness_path = root / "readiness.json"
            gap_path = root / "data_gap.json"
            exit_code = iterated_rof_paper_like.main(
                [
                    "--data-root",
                    str(data_root),
                    "--output",
                    str(readiness_path),
                    "--data-gap-output",
                    str(gap_path),
                    "--strict-data-ready",
                ]
            )
            gap = json.loads(gap_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(gap["ready_for_local_runner"])
        self.assertTrue(
            any("mask has fewer than two labels" in blocker for blocker in gap["data_ready_blockers"])
        )

    def test_local_runner_uses_distinct_figure_paths_for_sanitized_name_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 8), dtype=float)
            image[:, 4:] = 1.0
            self._write_png(root / "cartoon" / "images" / "a" / "b.png", image)
            self._write_png(root / "cartoon" / "images" / "a_b.png", image.T)

            summary = iterated_rof_paper_like.run_local_dataset(
                root,
                rof_n_iter=8,
                trof_max_iter=4,
                figure_dir=root / "figures",
            )

            figure_paths = [item["figure_path"] for item in summary["images"]]
            self.assertEqual(len(figure_paths), 2)
            self.assertEqual(len(set(figure_paths)), 2)
            self.assertTrue(all(Path(path).exists() for path in figure_paths))


if __name__ == "__main__":
    unittest.main()
