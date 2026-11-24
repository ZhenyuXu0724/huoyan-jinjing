"""Pinned-source, no-overwrite guards. Does not change source files."""
import json
from pathlib import Path
from independent_acceptance import sha,require

def validate_plan(records,source_root,lock_path):
 lock=json.loads(Path(lock_path).read_text())
 actual={}
 for r in records:
  for key in ('src_img','src_json'):
   p=r[key];require(not p.is_symlink(),f'source symlink: {p}')
   actual[str(p.relative_to(source_root))]=sha(p)
 require(len(actual)==2*len(records), 'DUPLICATE_SOURCE_ASSIGNMENT')
 expected={p:x['sha256'] for p,x in lock['files'].items()}
 require(actual==expected,'SOURCE_SNAPSHOT_CHANGED: approved source set or bytes differ; no write allowed')
 assignments={r['image_filename']:{k:r[k] for k in ('split','label','data_tier','scene_group_id','parent_image_id','augmentation_type')} for r in records}
 require(assignments==lock['assignments'],'APPROVED_PLAN_CHANGED: no write allowed')
 return assignments

def refuse_existing(root):
 # Even interrupted/empty partitions are protected. Never rmtree a dataset.
 managed=['train','val','test','dataset_manifest.csv','augmentation_mapping.csv','dataset_statistics.json','scene_groups.json','SHA256SUMS.txt']
 existing=[n for n in managed if (root/n).exists() or (root/n).is_symlink()]
 require(not existing,'EXISTING_DATASET_PROTECTED: '+', '.join(existing)+'; use --plan / verify; no deletion or overwrite')
