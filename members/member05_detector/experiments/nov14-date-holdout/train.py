"""Rebuild the frozen split and train the six-class detector from the official initial weights."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import yaml
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='Directory with original Nov 11, 14 and 18 photos')
    parser.add_argument('--initial', type=Path, required=True, help='Official yolov8n.pt')
    parser.add_argument('--package', type=Path, required=True, help='Extracted release asset')
    parser.add_argument('--output', type=Path, required=True, help='New experiment directory')
    args = parser.parse_args()
    protocol = json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    rows = json.loads((HERE/'split.json').read_text(encoding='utf-8'))
    if digest(args.initial) != protocol['initial_sha256']:
        raise ValueError('Initial weights SHA256 mismatch')
    pictures = {digest(p): p for p in args.images.rglob('*')
                if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'}}
    if any(row['image_sha256'] not in pictures for row in rows):
        raise ValueError('Missing required original images; see split.json')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for row in rows:
        role, stem = row['split'], row['id']
        source = pictures[row['image_sha256']]
        image = output/'images'/role/(stem+source.suffix.lower())
        label = output/'labels'/role/(stem+'.txt')
        for p in [image,label]:
            p.parent.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(source,image)
        shutil.copyfile(args.package/'training-labels'/role/label.name,label)
    data = {'path':output.as_posix(), 'train':'images/train','val':'images/val','test':'images/test',
            'names':['non_catalogue','long_screw','thick_screw','thin_screw','hex_nut','flat_washer']}
    config = output/'dataset.yaml'
    config.write_text(yaml.safe_dump(data),encoding='utf-8')
    YOLO(str(args.initial)).train(data=str(config), project=str(output), name='fit', **protocol['training'])


if __name__=='__main__':
    main()
