"""Independent v2 inference; never changes the public v1 default."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from shared.contracts import validate_expected_counts
from ..model import load_model, predict_with_model
from ..data import sha256
from .common import ROOT, WEIGHTS, write

DEFAULT = WEIGHTS / 'baseline1_supervised_v2.joblib'


def predict(image_path, model_path=DEFAULT):
    try:
        bundle = load_model(model_path)
    except FileNotFoundError:
        return dict(status='MODEL_UNAVAILABLE', predicted_label=None, prediction_score=None, score_type=None)
    if bundle.get('kind') != 'hog_supervised_classifier' or not np.array_equal(bundle['model'].classes_, [0,1]):
        raise ValueError('Expected supervised v2 with PASS=0, FAIL=1')
    return predict_with_model(image_path, bundle)


def batch(input_dir, counts_file, output, model_path=DEFAULT):
    cases = json.loads(Path(counts_file).read_text(encoding='utf-8-sig'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('Nonempty request list required')
    requests = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'image_id','expected_counts'}:
            raise ValueError('Requests contain only image_id and expected_counts')
        name = case['image_id']
        if not isinstance(name,str) or not name or name in requests or any(c in name for c in '/\\:') or name in ('.','..'):
            raise ValueError('Unique filename required')
        requests[name] = validate_expected_counts(case['expected_counts'])
    bundle = load_model(model_path)
    if bundle.get('kind') != 'hog_supervised_classifier' or not np.array_equal(bundle['model'].classes_,[0,1]):
        raise ValueError('Expected supervised v2 model')
    paths = {}
    for path in Path(input_dir).rglob('*'):
        if path.is_file() and path.name in requests:
            if path.name in paths:
                raise ValueError('Duplicate filename')
            paths[path.name] = path
    rows, times = [], {}
    for name, requirement in requests.items():
        start = perf_counter()
        if requirement != bundle['config']['expected_counts']:
            result = dict(status='UNSUPPORTED_COUNTS', predicted_label=None, prediction_score=None, score_type=None)
        elif name not in paths:
            result = dict(status='INPUT_ERROR', predicted_label=None, prediction_score=None, score_type=None)
        else:
            result = predict_with_model(paths[name], bundle)
        rows.append(dict(image_id=name, model_id='baseline1', **result, non_catalogue_count=None))
        times[name] = (perf_counter()-start)*1000
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output/'images.csv').open('w', encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(output/'run_config.json', dict(model_id='baseline1', model_version='baseline1_supervised_v2',
          model_sha256=sha256(model_path), method='HOG + supervised MLP classifier',
          preprocessing=bundle['config'], supported_expected_counts=bundle['config']['expected_counts'],
          input_expected_counts=requests, score_type='uncalibrated_fail_probability',
          capabilities=bundle['capabilities'], dependencies=bundle['dependencies'],
          offline_processing_ms=times, timing_scope='image read + preprocessing + inference; excludes camera/UI',
          class_catalogue=json.loads((ROOT/'shared/catalogue.json').read_text(encoding='utf-8'))['classes']))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--model',type=Path,default=DEFAULT)
    parser.add_argument('--input_dir',type=Path)
    parser.add_argument('--counts_file',type=Path)
    parser.add_argument('--output_dir',type=Path)
    args=parser.parse_args()
    if args.image:
        print(json.dumps(predict(args.image,args.model)))
    elif all((args.input_dir,args.counts_file,args.output_dir)):
        batch(args.input_dir,args.counts_file,args.output_dir,args.model)
    else:
        parser.error('Use --image or all batch arguments')
