"""Pin new sources and audit all split identities without fitting any model."""
import concurrent.futures
import csv
import hashlib
import os
import subprocess
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from PIL import Image
from ..data import identity, load_manifest, reject_overlap, sha256
from ..one_class.prepare import LABELS, object_hash, OLD, NEW, PREFIX
from .common import ROOT, MODULE, DATA, OUT, COMMIT, read, write, preserve_old


def run():
    preserve_old()
    raw = subprocess.check_output(['git', 'ls-tree', '-r', '-z', COMMIT, 'members/member03_data'], cwd=ROOT)
    tree = {r.split(b'\t', 1)[1].decode(): r.split(b'\t', 1)[0].split()[-1].decode()
            for r in raw.split(b'\0') if r}
    old_index = read(MODULE / 'configs/member03_sources.json')['files']
    if any(tree.get(k) != v for k, v in old_index.items()):
        raise ValueError('Old data source changed')
    old = ROOT / 'data/member04_hog/dataset'
    verify_old_sources(old, old_index)
    new = {k: v for k, v in tree.items() if '11.18' in k}
    if Counter(Path(k).suffix for k in new) != {'.json': 52, '.jpg': 52}:
        raise ValueError('Unexpected new source inventory')
    write(MODULE / 'supervised_v2/sources.json', {'commit': COMMIT, 'files': new})
    directory = DATA / 'new18'
    directory.mkdir(parents=True, exist_ok=True)
    cached = ROOT / 'outputs/member04_hog/planning_20261119'
    def acquire(item):
        path, checksum = item
        target = directory / Path(path).name
        for candidate in (target, cached / target.name):
            if candidate.exists() and object_hash(candidate.read_bytes()) == checksum:
                if candidate != target:
                    target.write_bytes(candidate.read_bytes())
                return
        error = None
        for attempt in range(3):
            try:
                url = f'https://raw.githubusercontent.com/ZhenyuXu0724/huoyan-jinjing/{COMMIT}/' + urllib.parse.quote(path)
                with urllib.request.urlopen(url, timeout=90) as response:
                    payload = response.read()
                if object_hash(payload) != checksum:
                    raise ValueError('Source checksum mismatch')
                target.write_bytes(payload)
                return
            except OSError as exc:
                error = exc
                time.sleep(attempt + 1)
        raise error
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i, _ in enumerate(pool.map(acquire, new.items()), 1):
            if i % 20 == 0:
                print(f'New source files verified {i}/104', flush=True)
    old = ROOT / 'data/member04_hog/dataset'
    records = read(old / 'provenance.json')['sources'] + read(old / 'test_provenance.json')['sources']
    records = [{**r, 'image_path': Path(os.path.relpath(old / r['image_path'], DATA)).as_posix()} for r in records]
    for annotation in sorted(directory.glob('*.json')):
        j = read(annotation)
        counts = Counter(LABELS[s['label']] for s in j['shapes'])
        actual = {str(i): counts[str(i)] for i in range(6)}
        records.append(dict(image_path='new18/' + annotation.with_suffix('.jpg').name,
                            split='train', counts=actual, source_session='new18_distractor_train',
                            label='PASS' if actual['0'] == 0 and all(actual[str(i)] == 2 for i in range(1, 6)) else 'FAIL'))
    for r in records:
        image = DATA / r['image_path']
        j = read(image.with_suffix('.json'))
        with Image.open(image) as opened:
            opened.load()
            width, height = opened.size
        if (width, height) != (j['imageWidth'], j['imageHeight']):
            raise ValueError('Image dimensions mismatch')
        counts = Counter()
        for s in j['shapes']:
            counts[LABELS[s['label']]] += 1
            if s['shape_type'] != 'rectangle' or len(s['points']) != 2:
                raise ValueError('Invalid rectangle')
            (x1, y1), (x2, y2) = s['points']
            if x1 == x2 or y1 == y2 or not all(0 <= x <= width and 0 <= y <= height for x, y in s['points']):
                raise ValueError('Invalid box bounds')
        if any(counts[str(i)] != r['counts'][str(i)] for i in range(6)):
            raise ValueError('Annotation counts disagree')
        r['image_sha256'] = sha256(image)
    splits = {}
    for split in ('train', 'valid', 'test'):
        selected = [r for r in records if r['split'] == split]
        with (DATA / f'{split}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['image_path', 'scene_id', 'label'])
            writer.writeheader()
            for r in selected:
                writer.writerow(dict(image_path=r['image_path'], scene_id='UNVERIFIED:' + Path(r['image_path']).stem, label=r['label']))
        splits[split] = load_manifest(DATA / f'{split}.csv')
    for left, right in (('train', 'valid'), ('train', 'test'), ('valid', 'test')):
        reject_overlap(identity(splits[left]), identity(splits[right]))
    expected = {'train': {'PASS': 133, 'FAIL': 52}, 'valid': {'PASS': 24, 'FAIL': 77}, 'test': {'PASS': 23, 'FAIL': 1}}
    support = {k: dict(Counter(s.label for s in v)) for k, v in splits.items()}
    if support != expected:
        raise ValueError(f'Unexpected support: {support}')
    write(DATA / 'provenance.json', {'commit': COMMIT, 'sources': records})
    write(OUT / 'data_audit.json', {'commit': COMMIT, 'support': support,
          'identities': {k: identity(v) for k, v in splits.items()},
          'manifests': {k: sha256(DATA / f'{k}.csv') for k in splits},
          'exact_overlap': 0, 'scene_independence': 'UNVERIFIED', 'near_duplicates': 'NOT_VERIFIED',
          'new_train_missing_M4x25': [Path(r['image_path']).name for r in records if r['source_session']=='new18_distractor_train' and r['counts']['1']==1],
          'test_access_scope': 'identity and annotation audit only; no inference before v2 freeze'})
    print(support, flush=True)


def verify_old_sources(old, index):
    for source, digest in index.items():
        if source.startswith(OLD) and any(s in source.removeprefix(OLD).split('/') for s in ('train','val','test')):
            local=old/'old'/source.removeprefix(OLD)
        elif source.startswith(PREFIX+NEW):
            local=old/'new'/source.removeprefix(PREFIX+NEW)
        elif source==PREFIX+'data_20261114_manifest.json':
            local=old/'source_manifest_20261114.json'
        else:
            local=old/'source_documents/old'/source.removeprefix(PREFIX+'full_dataset_upload/')
        if object_hash(local.read_bytes())!=digest:
            raise ValueError(f'Old local source differs from pinned source: {source}')


if __name__ == '__main__':
    run()
