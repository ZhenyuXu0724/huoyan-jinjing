import unittest

from members.member04_hog.diagnose import summary, threshold_curve


class DiagnosisTests(unittest.TestCase):
    def test_ties_and_threshold_endpoints(self):
        rows = [{'image_id': str(i), 'label': label, 'status': 'OK',
                 'prediction_score': score, 'predicted_label': 'PASS'}
                for i, (label, score) in enumerate([('PASS', 1), ('FAIL', 1), ('FAIL', 3)])]
        curve = threshold_curve(rows)
        self.assertEqual((curve[0]['false_pass'], curve[0]['false_fail']), (0, 1))
        self.assertEqual((curve[-1]['false_pass'], curve[-1]['false_fail']), (2, 0))
        self.assertEqual(len(curve), 3)
        self.assertEqual([r['predicted_label'] for r in rows], ['PASS'] * 3)

    def test_single_class_balanced_accuracy_is_undefined(self):
        result = summary([{'image_id': 'a', 'label': 'PASS', 'status': 'OK',
                           'predicted_label': 'PASS', 'prediction_score': 1}])
        self.assertIsNone(result['balanced_accuracy'])
        self.assertEqual(result['score_quantiles']['median'], 1)
