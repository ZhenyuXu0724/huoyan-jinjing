"""Fail-closed DEV_ONLY weighted-CE preparation. Never trains."""
import argparse
import json
from pathlib import Path
from .config import Config
from .runtime import file_sha256, write_json, environment, choose_device
from .preflight import check_training_inputs
from .train import class_weights

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
ARCHIVE = BASE / 'archives/dev_baseline_v1'
CONFIG = BASE / 'configs/baseline2_weighted_ce_dev_v2.json'
EXPECTED = '57f7be37ed2e9543f3a3219636f527a78f9255acc8ccc4e193c38776fdcaaa65'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def verify_archive():
    entries = read(ARCHIVE / 'SHA256.json')
    for name, digest in entries.items():
        p = (ARCHIVE / name).resolve()
        if not p.is_relative_to(ARCHIVE.resolve()) or file_sha256(p) != digest:
            raise ValueError(f'archive changed: {name}')
    if file_sha256(ARCHIVE / 'run/best.pt') != EXPECTED:
        raise ValueError('historical checkpoint identity mismatch')
    return len(entries)

def config_diff(old, new):
    a, b = Config(**old).to_dict(), Config(**new).to_dict()
    diff = {k: {'v1': a[k], 'v2': b[k]} for k in a if a[k] != b[k]}
    if diff != {'train_class_weight': {'v1': 'none', 'v2': 'balanced'}}:
        raise ValueError(f'D: not a single-factor comparison: {diff}')
    return diff

def check(config_path=CONFIG):
    archive_count = verify_archive()
    config = Config.load(config_path)
    diff = config_diff(read(ARCHIVE/'run/config.json'), config.to_dict())
    old = read(ARCHIVE/'run/run_config.json')
    for split in ('train','val'):
        if file_sha256(getattr(config, split+'_manifest')) != old['datasets'][split]['manifest_sha256']:
            raise ValueError(f'{split} manifest changed')
    if file_sha256(config.data_provenance) != old['data_provenance_sha256']:
        raise ValueError('provenance changed')
    # All historical modules except the two reviewed loss/config changes remain identical.
    for name, digest in old['environment']['source_sha256'].items():
        if name not in {'config.py','train.py'} and file_sha256(BASE/name) != digest:
            raise ValueError(f'uncontrolled historical code change: {name}')
    protocol = read(BASE/'configs/weighted_ce_code_identity.json')
    for name,digest in protocol.items():
        if file_sha256(BASE/name) != digest:
            raise ValueError(f'reviewed implementation changed: {name}')
    import torch
    torch.set_num_threads(config.cpu_threads)
    cached = Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth'
    if not cached.is_file() or file_sha256(cached) != old['pretrained_source']['sha256']:
        raise ValueError('cached ImageNet initialization differs or missing; no automatic substitution')
    current = environment(choose_device(config.device))
    for key in ('python','packages','device','torch_cuda'):
        if current[key] != old['environment'][key]:
            raise ValueError(f'environment differs in {key}; review control before training')
    train, val, preflight = check_training_inputs(config)
    if train.rows != old['datasets']['train']['images']:
        # Persisted rows serialize Path objects as strings.
        normalized = [{**r,'path':str(r['path'])} for r in train.rows]
        if normalized != old['datasets']['train']['images']:
            raise ValueError('Train image identity changed')
    for ds,split in ((val,'val'),):
        if [{**r,'path':str(r['path'])} for r in ds.rows] != old['datasets'][split]['images']:
            raise ValueError('Val image identity changed')
    return {'status':'DEV_ONLY_READY_FOR_USER_START','archive_files_verified':archive_count,
            'config_diff':diff,'train_class_counts':{k:sum(r['label']==k for r in train.rows) for k in config.class_to_idx},
            'weights_by_index':class_weights(train.rows,config,'cpu').tolist(),
            'validation_loss':'unweighted CE','decision':'argmax; ties index 0',
            'environment':current,'preflight':preflight,
            'limitations':'Same DEV data; labels/count rule and scene independence remain provisional. No Test.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(CONFIG));p.add_argument('--output');p.add_argument('--archive-only',action='store_true')
    a=p.parse_args()
    try:
        result={'archive_files_verified':verify_archive()} if a.archive_only else check(a.config)
        if a.output:
            out=Path(a.output)
            with out.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,OSError,KeyError) as e:p.exit(2,f'PRECHECK_FAILED: {e}\n')
if __name__=='__main__':main()
