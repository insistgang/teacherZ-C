import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

import iterated_rof_paper_like  # noqa: E402


class IteratedRofPaperLikeScaffoldTests(unittest.TestCase):
    def test_missing_data_is_reported_as_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = iterated_rof_paper_like.build_readiness_report(Path(tmp))

        self.assertEqual(report["status"], "blocked_missing_data")
        self.assertEqual(len(report["families"]), 3)
        self.assertTrue(report["blockers"])
        self.assertEqual(report["current_dashboard_level"], "partial")

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
        self.assertFalse(report["blockers"])
        self.assertTrue(all(item["status"] == "ready_quantitative" for item in report["families"]))


if __name__ == "__main__":
    unittest.main()
