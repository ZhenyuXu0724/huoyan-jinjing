"""Annotation/source audit; explicit public-test access is NOT model evaluation."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import math
from pathlib import Path

from PIL import Image
from .config import normalize_counts
from .image_io import read_image
from .team_data import audit_dataset, read_json, sha256_file, unresolved_flags, write_new_json


def audit_flat(index_path, labels):
    from .development import safe_path
    index = read_json(index_path)
    if index.get("schema_version") != "member06.flat-labelme.v1" or index.get("split") not in {"train", "val"}:
        raise ValueError("flat source must explicitly be development train/val")
    root = Path(index["root"]).resolve()
    if any(x in root.name.lower() for x in ("test", "测试", "验证")) and index["split"] == "train":
        raise ValueError("test/validation source cannot train")
    files, images, errors = index["files"], [], []
    actual = {p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".jpeg", ".json"}}
    if actual != set(files):
        errors.append({"image_path": str(root), "error": "flat source inventory mismatch"})
    image_names = [n for n in files if Path(n).suffix.lower() in {".jpg", ".png", ".jpeg"}]
    if not image_names or {Path(n).with_suffix('.json').name for n in image_names} != {n for n in files if Path(n).suffix == '.json'}:
        errors.append({"image_path": str(root), "error": "unpaired image/annotation"})
    for name in sorted(image_names):
        try:
            if Path(name).name != name:
                raise ValueError("flat index requires filenames")
            path, annotation = safe_path(root, name), safe_path(root, str(Path(name).with_suffix('.json')))
            for p in (path, annotation):
                if sha256_file(p) != files[p.name]["sha256"] or p.stat().st_size != files[p.name]["bytes"]:
                    raise ValueError("flat source hash/bytes mismatch")
            image, raw = read_image(path), read_json(annotation)
            if (raw["imageWidth"], raw["imageHeight"]) != image.size or raw.get("imagePath") != path.name:
                raise ValueError("image/annotation identity or dimension conflict")
            counts = {str(i): 0 for i in range(6)}
            for shape in raw["shapes"]:
                if shape.get("label") not in labels or shape.get("shape_type") != "rectangle":
                    raise ValueError("unknown label/shape")
                pts = shape["points"]
                if len(pts) != 2 or any(len(p) != 2 for p in pts) or any(type(x) not in (int, float) or not math.isfinite(x) for p in pts for x in p):
                    raise ValueError("invalid coordinates")
                x1,x2 = sorted(p[0] for p in pts); y1,y2 = sorted(p[1] for p in pts)
                if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                    raise ValueError("bbox outside image")
                counts[str(labels[shape['label']])] += 1
            images.append({"image_path": str(path), "split": index["split"], "source_kind": "flat_labelme",
                "source_package": str(Path(index_path).resolve()), "source_commit": index.get('source_commit'),
                "image_sha256": sha256_file(path), "annotation_sha256": sha256_file(annotation), "gt_sha256": None,
                "rgb_sha256": hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest(),
                "actual_counts": counts, "unresolved_ignore_or_judgment": unresolved_flags(raw),
                "raw_auto_review_pending": any(s.get('flags',{}).get('auto_label_review') or
                    'human review required' in s.get('description','').lower() for s in raw['shapes'])})
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append({"image_path": str(root/name), "error": str(exc)})
    return images, errors


def describe(row, expected):
    counts = row['actual_counts']; missing = [i for i in range(1,6) if counts[str(i)] < expected[i-1]]
    extra = [i for i in range(1,6) if counts[str(i)] > expected[i-1]]
    types = (['non_catalogue'] if counts['0'] else []) + (['missing_catalogue'] if missing else []) + (['extra_catalogue'] if extra else [])
    if row['unresolved_ignore_or_judgment']:
        label = None
    else:
        label = 'FAIL' if types else 'PASS'
    return {'annotation_label':label, 'anomaly_type':'+'.join(types) if types else 'none',
            'non_catalogue_count':counts['0'], 'missing_class_ids':missing, 'extra_class_ids':extra,
            'label_basis':'annotation_counts_under_explicit_reference_rule; physical specifications not verified'}


def near_duplicates(rows):
    """Low-resolution image hashes are triage candidates, NEVER physical scene identities."""
    hashes=[]
    for row in rows:
        image=read_image(row['image_path']).convert('L')
        pix=list(image.resize((9,8),Image.Resampling.BILINEAR).getdata())
        dh=sum((pix[y*9+x]>pix[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
        pix=list(image.resize((8,8),Image.Resampling.BILINEAR).getdata());mean=sum(pix)/64
        ah=sum((v>mean)<<i for i,v in enumerate(pix))
        hashes.append((dh,ah))
    candidates=[]
    for i,(dh,ah) in enumerate(hashes):
        for j in range(i):
            d=(dh^hashes[j][0]).bit_count();a=(ah^hashes[j][1]).bit_count()
            if d<=4 and a<=4:
                candidates.append({'left':rows[j]['image_path'],'right':rows[i]['image_path'],
                    'cross_split':rows[i]['split']!=rows[j]['split'],'dhash_distance':d,'ahash_distance':a,
                    'status':'REVIEW_CANDIDATE_NOT_CONFIRMED_SAME_SCENE'})
    return candidates


def audit_corpus(sources_path, include_public_test=False, proposed_split=None):
    from .development import audit
    report=audit(sources_path); sources=report['sources']; rows=list(report['images'])
    if include_public_test:
        legacy=audit_dataset(sources['legacy_root'],sources['catalogue'],include_public_test=True)
        report['errors'].extend(legacy['errors'])
        rows.extend({**r,'image_path':str(Path(sources['legacy_root']).resolve()/r['image_path']),
                     'source_kind':'legacy','source_package':'parts_v1'} for r in legacy['images'] if r['split']=='test')
    expected=normalize_counts(report['reference_expected_counts'])
    for row in rows:
        row.update(describe(row,expected))
        row['original_split']=row['split'];row['proposed_split']=None
        row['image_id']=Path(row['image_path']).name
        row['scene_id']=None;row['scene_verified']=False
        row['capture_condition']=row.get('capture_condition','UNRECORDED')
        row['historical_use']=('HOG training reported' if row['split']=='train' else
                              'HOG threshold/model selection reported' if row['split']=='val' else
                              'HOG provided-partition evaluation reported') if row['source_kind']!='flat_labelme' else 'No accessible training log found; not proof of no prior use'
    transitions=[]
    if proposed_split:
        with Path(proposed_split).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            if set(reader.fieldnames or []) != {'image_id','split','label'}:raise ValueError('proposed split requires image_id,split,label')
            proposals=list(reader)
        mapping={r['image_id']:r for r in proposals}
        if len(mapping)!=len(proposals) or set(mapping)!={r['image_id'] for r in rows}:
            raise ValueError('proposed split must identify every audited image exactly once')
        for row in rows:
            new=mapping[row['image_id']]
            if new['split'] not in {'train','val','test'} or new['label'] not in {'PASS','FAIL'}:raise ValueError('invalid proposed role/label')
            row['proposed_split']=new['split'];row['proposed_label']=new['label']
            if new['label']!=row['annotation_label']:report['errors'].append({'image_path':row['image_path'],'error':'proposed label contradicts annotation rule'})
            if new['split']!=row['original_split']:
                transitions.append({'image_id':row['image_id'],'from':row['original_split'],'to':new['split'],'historical_use':row['historical_use']})
    exact=[]
    for key in ('image_sha256','rgb_sha256'):
        groups=defaultdict(list)
        for row in rows:groups[row[key]].append(row['image_id'])
        exact.extend({'kind':key,'images':v} for v in groups.values() if len(v)>1)
    table=defaultdict(Counter)
    for row in rows:
        key=(row['source_package'],row['original_split'],row['capture_condition'])
        table[key][str(row['annotation_label'])]+=1
    summary={split:dict(Counter(str(r['annotation_label']) for r in rows if r['split']==split)) for split in ('train','val','test')}
    return {'schema_version':'member06.corpus-audit.v1','sources_path':str(Path(sources_path).resolve()),
        'source_manifest_sha256':report['source_manifest_sha256'],'reference_expected_counts':expected,
        'test_content_read_for_source_audit':include_public_test,'model_inference_executed':False,
        'screenshot_split_verified':bool(proposed_split) and not report['errors'],
        'proposed_split_training_approved':False,
        'role_transition_review_required':bool(transitions),
        'proposed_split_file_sha256':sha256_file(proposed_split) if proposed_split else None,
        'original_split_label_counts':summary,'total_label_counts':dict(Counter(str(r['annotation_label']) for r in rows)),
        'source_condition_support':[{'source':k[0],'split':k[1],'condition':k[2],'labels':dict(v)} for k,v in table.items()],
        'errors':report['errors'],'exact_duplicates':exact,'near_duplicate_candidates':near_duplicates(rows),
        'role_transitions':transitions,'images':rows,
        'limitations':['No verified physical scene ledger; image hashes are not scene identity.',
                       'Class/batch confounding must be reported; this audit does not establish actual specification correctness.',
                       'Missing proposed split means 82 training FAIL identities cannot be verified.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',default='members/member06_research/configs/data_sources.json')
    p.add_argument('--include-public-test-audit',action='store_true')
    p.add_argument('--proposed-split');p.add_argument('--output',required=True)
    a=p.parse_args()
    try:
        report=audit_corpus(a.sources,a.include_public_test_audit,a.proposed_split)
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
        write_new_json(out/'audit.json',report)
        columns=['image_id','image_path','source_package','original_split','proposed_split','annotation_label','anomaly_type','non_catalogue_count','capture_condition','scene_verified','historical_use','image_sha256','annotation_sha256','gt_sha256','actual_counts']
        with (out/'images.csv').open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(report['images'])
        print({k:report[k] for k in ('original_split_label_counts','total_label_counts','errors','screenshot_split_verified')})
        return 2 if report['errors'] or report['exact_duplicates'] else 0
    except (ValueError,OSError,KeyError,TypeError) as exc:p.exit(2,f'AUDIT_FAILED: {exc}\n')


if __name__=='__main__':raise SystemExit(main())
