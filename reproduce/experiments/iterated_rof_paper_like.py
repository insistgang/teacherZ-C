import argparse
import json
from pathlib import Path

from common import REPO_ROOT, RESULTS_DIR


DATA_ROOT = REPO_ROOT / "reproduce" / "data" / "iterated_rof"
REPORT_PATH = RESULTS_DIR / "iterated_rof_paper_like_readiness.json"
SOURCE_MANIFEST_PATH = REPO_ROOT / "reproduce" / "paper_like" / "iterated_rof_dataset_sources.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

DATA_FAMILIES = {
    "cartoon": "cartoon or smooth-region image used to test missing-pixel / piecewise-smooth behavior",
    "texture": "close-intensity texture or stripe image used to test texture separation",
    "medical": "medical grayscale image, preferably MRI-like, used to test medical segmentation behavior",
}


def _count_images(directory):
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _display_path(path):
    path = Path(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_source_manifest(path=SOURCE_MANIFEST_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


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
    ]
    return {key: source[key] for key in keys if key in source}


def scan_family(root, family):
    family_root = Path(root) / family
    image_count = _count_images(family_root / "images")
    mask_count = _count_images(family_root / "masks")
    qualitative_ready = image_count > 0
    quantitative_ready = image_count > 0 and mask_count > 0

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
        "status": status,
        "path": _display_path(family_root),
    }


def build_readiness_report(root=DATA_ROOT, source_manifest=None):
    source_manifest = source_manifest or load_source_manifest()
    families = [scan_family(root, family) for family in DATA_FAMILIES]
    missing = [item["family"] for item in families if item["status"] == "missing"]
    quantitative = [item["family"] for item in families if item["status"] == "ready_quantitative"]
    recommended_sources = {
        family: [_source_summary(source) for source in sorted(source_manifest.get(family, []), key=lambda item: item["priority"])]
        for family in DATA_FAMILIES
    }

    if missing:
        status = "blocked_missing_data"
    elif not quantitative:
        status = "blocked_missing_masks"
    else:
        status = "ready_for_paper_like_runner"

    blockers = []
    if missing:
        blockers.append(f"Missing image data for: {', '.join(missing)}")
    if not quantitative:
        blockers.append("No family has masks/labels, so quantitative paper-like metrics are not available")

    return {
        "paper_id": "iterated-rof",
        "target_level": "paper-like",
        "current_dashboard_level": "partial",
        "status": status,
        "data_root": _display_path(root),
        "families": families,
        "recommended_sources": recommended_sources,
        "blockers": blockers,
        "claim_boundary": (
            "Do not promote dashboard level beyond partial until real/local images, baselines, metrics, "
            "and generated figures exist."
        ),
    }


def write_report(report, path=REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit iterated-rof paper-like reproduction data readiness.")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Local iterated-rof data root")
    parser.add_argument("--output", default=str(REPORT_PATH), help="Readiness JSON output path")
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Print recommended dataset sources and still write the readiness JSON report.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Reserved for future dataset-backed paper-like execution; currently still audits readiness.",
    )
    args = parser.parse_args(argv)

    report = build_readiness_report(Path(args.data_root))
    output = write_report(report, Path(args.output))
    print(f"wrote {output}")
    print(f"status: {report['status']}")
    if report["blockers"]:
        print("blockers:")
        for blocker in report["blockers"]:
            print(f"- {blocker}")
    if args.sources:
        print("recommended sources:")
        for family, sources in report["recommended_sources"].items():
            print(f"- {family}:")
            for source in sources:
                print(f"  - {source['source_id']}: {source['download_url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
