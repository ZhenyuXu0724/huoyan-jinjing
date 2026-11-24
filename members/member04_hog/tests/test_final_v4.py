import unittest
from pathlib import Path

import numpy as np

from tmpdirs import workspace_temp
from members.member04_hog.final_v4.common import (
    FP_BUDGET, PREFIX, checkpoint_key, choose_threshold, eol_variant, git_blob_sha1,
    prediction_rows, reconstruction_scores, sample_weights, threshold_candidates,
    validate_bundle,
)
from members.member04_hog.final_v4.groups import (
    EQUIVALENT_PAIRS, GROUPS, augmentation_count, base_settings, parameter_count,
    selection_key, view_count, views,
)


class GroupTests(unittest.TestCase):
    def test_eleven_groups_and_views(self):
        self.assertEqual(list(GROUPS), ["A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2",
                                        "C0", "C_without_geometry", "C_without_clahe"])
        expected = {"A0": (1, 0), "A1": (2, 1), "A2": (2, 1), "A3": (4, 1),
                    "A4": (4, 1), "A5": (8, 1), "B1": (3, 1), "B2": (1, 1),
                    "C0": (2, 2), "C_without_geometry": (1, 1), "C_without_clahe": (2, 1)}
        for group, (view_total, augmentations) in expected.items():
            with self.subTest(group=group):
                self.assertEqual(view_count(group), view_total)
                self.assertEqual(augmentation_count(group), augmentations)
                self.assertEqual(len(views(group)), view_total)

    def test_equivalent_pairs_and_parameters(self):
        self.assertEqual(EQUIVALENT_PAIRS, (("B2", "C_without_geometry"),
                                            ("A2", "C_without_clahe")))
        self.assertEqual(parameter_count("supervised", (64, 16)), 519538)
        self.assertEqual(parameter_count("ae", (64, 16, 64)), 1047092)
        self.assertFalse(base_settings("B2")["gamma"])
        self.assertTrue(base_settings("B2")["clahe"])

    def test_group_selection_prefers_single_factor_duplicate(self):
        rows = [{"fn": 2, "fp": 1}, {"fn": 2, "fp": 1}, {"fn": 2, "fp": 1}]
        self.assertLess(selection_key("B2", rows), selection_key("C_without_geometry", rows))
        self.assertLess(selection_key("A2", rows), selection_key("C_without_clahe", rows))


class ThresholdTests(unittest.TestCase):
    def test_budget_rule_and_representative_threshold(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        choice = choose_threshold(labels, scores, budget=1)
        self.assertEqual((choice["fn"], choice["fp"]), (0, 1))
        self.assertAlmostEqual(choice["threshold"], 0.15)
        rows = threshold_candidates(labels, scores, budget=1)
        self.assertTrue(any(row["fn"] == 2 for row in rows))  # all PASS candidate
        self.assertTrue(any(row["fn"] == 0 and row["fp"] == 2 for row in rows))

    def test_equal_scores_are_not_split_and_all_pass_remains_valid(self):
        labels = np.array([0, 1])
        scores = np.array([0.2, 0.2])
        choice = choose_threshold(labels, scores, budget=1)
        self.assertEqual(choice["threshold"], 0.2)
        self.assertEqual((choice["fn"], choice["fp"]), (0, 1))
        labels = np.array([1, 0])
        scores = np.array([0.1, 0.9])
        choice = choose_threshold(labels, scores, budget=0)
        self.assertEqual((choice["fn"], choice["fp"]), (1, 0))
        self.assertGreater(choice["threshold"], 0.9)

    def test_sample_weights_and_checkpoint_order(self):
        labels = np.array([0] * 166 + [1] * 83)
        weights = sample_weights(labels)
        self.assertAlmostEqual(weights[0], 0.75)
        self.assertAlmostEqual(weights[-1], 1.5)
        self.assertEqual(weights.sum(), 249)
        self.assertLess(checkpoint_key({"fn": 1, "fp": 0, "epoch": 90}),
                        checkpoint_key({"fn": 1, "fp": 0, "epoch": 120}))
        self.assertLess(checkpoint_key({"fn": 0, "fp": 3, "epoch": 1}),
                        checkpoint_key({"fn": 1, "fp": 0, "epoch": 1}))

    def test_reconstruction_scores_and_prediction_rows(self):
        class Model:
            def predict(self, values):
                return np.asarray(values) + 1.0

        values = np.zeros((2, 4))
        np.testing.assert_array_equal(reconstruction_scores(Model(), values), [1, 1])
        from members.member04_hog.data import Sample

        samples = [Sample("a", Path("a"), "s", "FAIL", "d"),
                   Sample("b", Path("b"), "s", "PASS", "d")]
        rows = prediction_rows(samples, [0.6, 0.4], 0.5, "uncalibrated_fail_probability")
        self.assertEqual([row["predicted_label"] for row in rows], ["FAIL", "PASS"])


class BundleAndHashTests(unittest.TestCase):
    def test_legacy_bundle_is_rejected(self):
        self.assertFalse(validate_bundle({"format_version": 1})[0])
        self.assertFalse(validate_bundle({
            "format_version": 2, "version": "baseline1_old", "kind": "hog_autoencoder",
        })[0])

    def test_eol_normalisation_and_blob_hash(self):
        payload = b"a\r\nb\r\n"
        self.assertEqual(eol_variant(payload, ["i/lf", "w/crlf"]), b"a\nb\n")
        self.assertEqual(eol_variant(payload, ["i/crlf"]), payload)
        self.assertEqual(len(git_blob_sha1(b"")), 40)

    def test_state_file_is_atomic_json(self):
        from members.member04_hog.final_v4.common import read, write

        with workspace_temp() as directory:
            path = Path(directory) / "nested/state.json"
            write(path, {"a": 1, "文本": "值"})
            self.assertEqual(read(path), {"a": 1, "文本": "值"})
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))


if __name__ == "__main__":
    unittest.main()
