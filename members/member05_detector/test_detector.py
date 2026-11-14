"""Hand-constructed unit cases, NOT project experimental evidence."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from .model import NAMES, validate_names, convert_boxes, count_objects, predict
from .evaluation import evaluate


def obj(cid=1, box=None, score=0.9, **extra):
    return {"class_id": cid, "bbox": box or [0, 0, 10, 10], "score": score, **extra}


def case(predictions, truth, **extra):
    return {"image_id": "unit", "status": "OK", "objects": predictions, "ground_truth": truth, **extra}


class DetectorTests(unittest.TestCase):
    def test_names_reject_coco_and_swapped_mapping(self):
        validate_names(NAMES)
        for names in ({0: "person"}, dict(NAMES, **{"1": "thin_screw"})):
            with self.assertRaises(ValueError): validate_names(names)

    def test_original_coordinates_clip_and_zero_counts(self):
        result = convert_boxes([[-2, 5, 101, 20]], [.8], [0.], 100, 80)
        self.assertEqual(result[0]["bbox"], [0, 5, 100, 20])
        self.assertEqual(count_objects(result), {"0": 1, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0})

    def test_bad_predictions(self):
        for box, score, cls in [([0,0,0,2],.5,1), ([0,0,3,4],float("nan"),1), ([0,0,3,4],.5,6)]:
            with self.assertRaises(ValueError): convert_boxes([box],[score],[cls],100,100)

    def test_missing_input_does_not_load_model(self):
        with patch("members.member05_detector.model._load_model") as loader:
            self.assertEqual(predict("__missing_unit_input__.png")["status"], "INPUT_ERROR")
            loader.assert_not_called()

    def test_duplicate_detection_is_fp_and_count_error(self):
        r = evaluate([case([obj(), obj(score=.8)], [obj()])])
        self.assertEqual(r["per_class"]["1"]["tp"], 1)
        self.assertEqual(r["per_class"]["1"]["fp"], 1)
        self.assertEqual(r["per_class"]["1"]["count_mae"], 1)
        self.assertEqual(r["five_class_vector_accuracy"], 0)

    def test_wrong_class_is_fp_and_fn(self):
        r = evaluate([case([obj(2)], [obj(1)])])
        self.assertEqual(r["per_class"]["2"]["fp"], 1)
        self.assertEqual(r["per_class"]["1"]["fn"], 1)

    def test_iou_boundary(self):
        r = evaluate([case([obj(box=[0,0,5,10])], [obj()])])
        self.assertEqual(r["per_class"]["1"]["tp"], 1)

    def test_empty_scene_included_and_unknown_separate(self):
        r = evaluate([case([], [], image_id="empty"), case([obj(0)], [])])
        self.assertEqual(r["per_class"]["0"]["count_mae"], .5)
        self.assertEqual(r["five_class_vector_accuracy"], 1)
        self.assertEqual(r["count_images"], 2)

    def test_ignored_region_and_review_excluded(self):
        r = evaluate([case([obj()], [obj(ignore=True)]), case([], [], image_id="review",status="REVIEW_REQUIRED")])
        self.assertEqual(r["ignored_predictions"], 1)
        self.assertEqual(r["count_images"], 0)
        self.assertEqual(r["coverage"], .5)
        self.assertIsNone(r["five_class_vector_accuracy"])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): evaluate([case([], []), case([], [])])

    def test_evaluation_status_and_flags_must_be_explicit(self):
        for record in (case([], [], status='TYPO'),case([], [],evaluable='false'),case([], [],counts_judgeable='false')):
            with self.subTest(record=record),self.assertRaises(ValueError): evaluate([record])

    def test_score_order_and_best_iou_matching(self):
        # High-score broad box must choose its highest-IoU GT, leaving the second free.
        a, b = obj(box=[0,0,10,10]), obj(box=[6,0,16,10])
        preds = [obj(box=[6,0,16,10],score=.4), obj(box=[0,0,12,10],score=.9)]
        self.assertEqual(evaluate([case(preds,[a,b])])["per_class"]["1"]["tp"],2)

    def test_valid_gt_takes_priority_over_ignore(self):
        r = evaluate([case([obj()], [obj(), obj(ignore=True)])])
        self.assertEqual(r["per_class"]["1"]["tp"],1)
        self.assertEqual(r["ignored_predictions"],0)


class TrainingDataTests(unittest.TestCase):
    def setUp(self):
        import yaml
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for split in ('train','val'):
            (self.root/'images'/split).mkdir(parents=True)
            (self.root/'labels'/split).mkdir(parents=True)
            from PIL import Image
            Image.new('RGB',(20,20),color='white' if split=='train' else 'gray').save(self.root/'images'/split/'fixture.png')
            (self.root/'labels'/split/'fixture.txt').write_text('1 0.5 0.5 0.2 0.2\n')
        self.config = self.root/'dataset.yaml'
        self.config.write_text(yaml.safe_dump({'path':str(self.root),'train':'images/train','val':'images/val','names':NAMES}),encoding='utf-8')

    def test_paired_labels_valid(self):
        from .train import validate_dataset
        self.assertEqual(validate_dataset(self.config)['names'],NAMES)

    def test_missing_annotation_cannot_be_background(self):
        from .train import validate_dataset
        (self.root/'labels/train/fixture.txt').unlink()
        with self.assertRaisesRegex(ValueError,'Missing label'): validate_dataset(self.config)

    def test_verified_empty_annotation_allowed(self):
        from .train import validate_dataset
        (self.root/'labels/train/fixture.txt').write_text('')
        validate_dataset(self.config)

    def test_out_of_bounds_and_unknown_class_rejected(self):
        from .train import validate_dataset
        for label in ('6 .5 .5 .2 .2','1 .1 .5 .8 .2','1 nan .5 .2 .2','1 .5 .5 -.1 .2'):
            (self.root/'labels/train/fixture.txt').write_text(label)
            with self.subTest(label=label),self.assertRaises(ValueError): validate_dataset(self.config)

    def test_corrupt_image_rejected(self):
        from .train import validate_dataset
        (self.root/'images/train/fixture.png').write_bytes(b'not an image')
        with self.assertRaisesRegex(ValueError,'Cannot decode'): validate_dataset(self.config)

    def test_duplicate_pixels_across_splits_rejected(self):
        from .train import validate_dataset
        from PIL import Image
        # Different filename/encoding metadata must not hide identical image pixels.
        with Image.open(self.root/'images/train/fixture.png') as image:
            image.save(self.root/'images/val/fixture.png',compress_level=0)
        with self.assertRaisesRegex(ValueError,'Duplicate decoded'): validate_dataset(self.config)


if __name__ == "__main__":
    unittest.main()
