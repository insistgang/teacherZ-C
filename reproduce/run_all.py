import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "reproduce" / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from common import write_results  # noqa: E402

import graph_classification  # noqa: E402
import map_uq_toy  # noqa: E402
import nested_sampling_toy  # noqa: E402
import online_ri_toy  # noqa: E402
import sat_rof_trof  # noqa: E402
import segmentation_restoration  # noqa: E402
import slat_color  # noqa: E402
import sphere_wavelet_toy  # noqa: E402
import tubular_tight_frame  # noqa: E402


RUNNERS = [
    sat_rof_trof.run,
    segmentation_restoration.run,
    tubular_tight_frame.run,
    slat_color.run,
    sphere_wavelet_toy.run,
    graph_classification.run,
    map_uq_toy.run,
    online_ri_toy.run,
    nested_sampling_toy.run,
]


def main():
    results = []
    for runner in RUNNERS:
        name = runner.__module__
        print(f"running {name}...")
        try:
            batch = runner()
        except Exception as exc:
            batch = [{
                "priority": -1,
                "id": name,
                "experiment_id": name,
                "reproductionLevel": "toy",
                "status": "failed",
                "runtime_seconds": 0.0,
                "metrics": {},
                "resultFiles": [],
                "skipped_reason": "",
                "notes": f"Unhandled experiment failure: {type(exc).__name__}: {exc}"
            }]
        results.extend(batch)
    results = sorted(results, key=lambda row: (row.get("priority", 999), row.get("experiment_id", "")))
    json_path, csv_path = write_results(results)
    completed = sum(1 for row in results if row.get("status") == "completed")
    skipped = sum(1 for row in results if row.get("status") == "skipped")
    failed = sum(1 for row in results if row.get("status") == "failed")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"summary: completed={completed}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
