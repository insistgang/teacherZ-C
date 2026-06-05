import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

import sat_rof_trof  # noqa: E402


class IteratedRofPartialTests(unittest.TestCase):
    def test_chambolle_pock_preserves_constant_image(self):
        image = np.full((20, 20), 0.37)

        denoised, info = sat_rof_trof.rof_chambolle_pock(
            image,
            mu=8.0,
            n_iter=60,
            return_info=True,
        )

        self.assertLess(float(np.max(np.abs(denoised - image))), 1e-8)
        self.assertGreaterEqual(info["iterations"], 1)

    def test_trof_means_are_computed_from_raw_image(self):
        score_image = np.array(
            [
                [0.05, 0.05, 0.95, 0.95],
                [0.05, 0.05, 0.95, 0.95],
                [0.05, 0.05, 0.95, 0.95],
                [0.05, 0.05, 0.95, 0.95],
            ],
            dtype=float,
        )
        raw_image = np.array(
            [
                [0.21, 0.21, 0.77, 0.77],
                [0.21, 0.21, 0.77, 0.77],
                [0.21, 0.21, 0.77, 0.77],
                [0.21, 0.21, 0.77, 0.77],
            ],
            dtype=float,
        )

        result = sat_rof_trof.run_trof_thresholds(
            score_image,
            raw_image,
            n_classes=2,
            initial_thresholds=np.array([0.5]),
            max_iter=1,
            tol=0.0,
            projection_bins=None,
        )

        expected_threshold = 0.5 * (0.21 + 0.77)
        self.assertAlmostEqual(result["history"][1][0], expected_threshold)
        self.assertNotAlmostEqual(result["history"][1][0], 0.5)

    def test_projected_trof_tracks_lemma_metrics(self):
        truth, image, _ = sat_rof_trof.generate_close_gray_multiphase(
            n=32,
            levels=np.array([0.28, 0.32, 0.36, 0.40]),
            noise_sigma=0.0,
            seed=7,
        )

        result = sat_rof_trof.run_trof_thresholds(
            image,
            image,
            n_classes=4,
            initial_thresholds=np.array([0.30, 0.34, 0.38]),
            max_iter=8,
            projection_bins=4096,
        )

        self.assertFalse(result["monotonicity_violated"])
        self.assertTrue(result["sign_changes_nonincreasing"])
        self.assertEqual(result["labels"].shape, truth.shape)

    def test_k2_proposition_formula_is_reported(self):
        result = sat_rof_trof.run_k2_proposition_demo(n=32, seed=9)
        expected = result["mu"] / (2.0 * (result["m1"] - result["m0"]))

        self.assertAlmostEqual(result["lambda_derived"], expected, places=10)
        self.assertGreater(result["rof_threshold_dice"], 0.9)
        self.assertGreater(result["chanvese_proxy_dice"], 0.9)


if __name__ == "__main__":
    unittest.main()
