"""Synthetic arithmetic/path checks. No real training or data access."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from members.member06_research.diagnose_first_run import binary_metrics, ranking, safe_image, variance_fraction, diagnose


class DiagnosticTests(unittest.TestCase):
    def test_default_tie_is_pass(self):
        r = binary_metrics([0, 1, 1], [.5, .5, .6])
        self.assertEqual((r['tn'], r['fn'], r['tp'], r['fp']), (1, 1, 1, 0))
        self.assertEqual(r['balanced_accuracy'], .75)

    def test_roc_and_ap_perfect_reverse_and_tied(self):
        r = ranking([0, 0, 1, 1], [.1, .2, .8, .9])
        self.assertEqual(r['roc_auc'], 1); self.assertEqual(r['average_precision'], 1)
        self.assertEqual(ranking([0, 1], [.9, .1])['roc_auc'], 0)
        r = ranking([0, 1], [.2, .2])
        self.assertEqual(r['roc_auc'], .5); self.assertEqual(r['average_precision'], .5)

    def test_group_undefined_metrics_are_null(self):
        r = binary_metrics([1, 1], [.1, .2])
        self.assertIsNone(r['precision']); self.assertIsNone(r['balanced_accuracy'])
        self.assertEqual(r['false_pass_rate'], 1)
        self.assertIsNone(ranking([1, 1], [.1, .2])['roc_auc'])
        self.assertIsNone(binary_metrics([], [])['coverage'])

    def test_invalid_scores_rejected(self):
        for scores in ([float('nan')], [1.1], [-.1], []):
            with self.assertRaises(ValueError): binary_metrics([1], scores)
        with self.assertRaises(ValueError): binary_metrics([2], [.5])

    def test_test_and_traversal_rejected_before_read(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for path, split in [(root/'test/a.jpg', 'val'), (root/'测试/a.jpg', 'train'),
                                (root/'val/a.jpg', 'test'), (root/'../escape.jpg', 'val')]:
                with self.assertRaises(ValueError): safe_image(path, root, split)
            self.assertEqual(safe_image(root/'val/a.jpg', root, 'val'), root/'val/a.jpg')
            (root/'link').symlink_to(root/'test', target_is_directory=True)
            with self.assertRaises(ValueError): safe_image(root/'link/a.jpg', root, 'val')

    def test_output_cannot_overwrite_existing(self):
        member = Path(__file__).resolve().parents[1]
        # Boundary fails before reading a run or initializing any model.
        with self.assertRaises(ValueError): diagnose('/tmp/no-run', '/tmp/no-out')
        with tempfile.TemporaryDirectory(dir=member / "diagnostics") as directory:
            with self.assertRaises(ValueError): diagnose(member/'runs/dev_baseline_v1', directory)

    def test_scatter_fraction_is_descriptive_and_degenerate_safe(self):
        self.assertEqual(variance_fraction([[0], [0], [1], [1]], ['a', 'a', 'b', 'b']), 1)
        self.assertEqual(variance_fraction([[0], [1]], ['a', 'a']), 0)
        self.assertIsNone(variance_fraction([[1], [1]], ['a', 'b']))

if __name__ == '__main__': unittest.main()
