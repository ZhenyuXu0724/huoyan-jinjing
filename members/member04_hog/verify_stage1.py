"""Exercise real frozen weights and failure states without changing artifacts."""
import argparse
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from threadpoolctl import threadpool_limits
from . import model
from .data import sha256
from .evaluate import validate_freeze, write_json
from .inference import batch

ROOT = Path(__file__).resolve().parents[2]


def verify(image):
    before = sha256(model.MODEL_PATH)
    validate_freeze(ROOT / 'outputs/member04_hog/assessment/freeze.json')
    with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
        root = Path(directory)
        good = root / image.name
        good.write_bytes(image.read_bytes())
        bad = root / 'broken.png'
        bad.write_text('not an image')
        normal = model.predict(str(good))
        assert normal['status'] == 'OK' and normal['score_type'] == 'hog_reconstruction_mse'
        assert model.predict(str(bad))['status'] == 'INPUT_ERROR'
        assert model.predict(str(root / 'missing.png'))['status'] == 'INPUT_ERROR'
        with patch.object(model, 'MODEL_PATH', root / 'absent.joblib'):
            assert model.predict(str(good))['status'] == 'MODEL_UNAVAILABLE'
        counts = {str(i): 2 for i in range(1, 6)}
        requests = [{'image_id': good.name, 'expected_counts': counts},
                    {'image_id': bad.name, 'expected_counts': counts},
                    {'image_id': 'missing.png', 'expected_counts': counts},
                    {'image_id': 'unsupported.png', 'expected_counts': dict(counts, **{'1': 3})}]
        request = root / 'counts.json'
        request.write_text(json.dumps(requests))
        rows = batch(root, request, root / 'out')
        assert [r['status'] for r in rows] == ['OK', 'INPUT_ERROR', 'INPUT_ERROR', 'UNSUPPORTED_COUNTS']
        assert rows[0]['predicted_label'] == normal['predicted_label']
        for row in rows[1:]:
            assert row['prediction_score'] is None and row['predicted_label'] is None
        exported = list(csv.DictReader((root / 'out/images.csv').open()))
        assert len(exported) == 4
        try:
            batch(root, request, root / 'missing_model', root / 'absent.joblib')
        except FileNotFoundError:
            assert not (root / 'missing_model').exists()
        else:
            raise AssertionError('Batch must fail explicitly for absent weights')
    assert sha256(model.MODEL_PATH) == before
    return {'default_model_sha256': before, 'freeze': 'PASS', 'real_image': 'PASS',
            'bad_and_missing_images': 'PASS', 'unsupported_counts': 'PASS',
            'missing_default_weight': 'MODEL_UNAVAILABLE',
            'missing_batch_weight': 'FileNotFoundError; no output', 'batch_export': 'PASS'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Verification output already exists')
    write_json(args.out, verify(args.image))
