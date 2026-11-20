"""Independent study delivery; exclude images and preserve repository hierarchy."""
import argparse,hashlib,json,zipfile
from pathlib import Path
from .common import ROOT,OUT,WEIGHTS,DATA,read
from .assess import verify
NAME='Baseline1_HOGMLP_v3_Stability_Capacity_L2_EpochSelection'

def build(destination):
    verify();v=read(OUT/'verification.json')
    if v['checkpoints']!=18 or v['label_mismatches']!=0:raise ValueError('Verification incomplete')
    paths=[]
    for folder in (ROOT/'members/member04_hog',OUT,WEIGHTS):
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() not in ('.pyc','.jpg','.png','.jpeg','.zip'))
    paths.extend(ROOT/p for p in ('shared/contracts.py','shared/catalogue.json','weights/member04_hog/baseline1.joblib','weights/member04_hog/supervised_v2/baseline1_supervised_v2.joblib'))
    paths.extend(DATA/p for p in ('train.csv','valid.csv','test.csv','provenance.json'))
    files={p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted(set(paths))}
    files['DELIVERY_README.md']=(f'# {NAME}\n\n先读members/member04_hog/supervised_v3_study/README.md。\n'
      '包含研究18个检查点、新旧代表模型和研究证据；不含原始图片、虚拟环境、全部旧版/v2实验参考权重。\n'
      '独立推理：python -m members.member04_hog.supervised_v3_study.inference --image your_image.jpg\n'
      '摘要校验：python -m members.member04_hog.delivery --verify .\n'
      '复现训练需先恢复固定数据与旧版/v2参考资产，当前包不冒充完整训练输入。\n').encode('utf-8')
    files['DELIVERY_SHA256.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in files.items()},ensure_ascii=False,indent=2).encode('utf-8')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for name,payload in files.items():z.writestr(name,payload)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_bytes((digest+'  '+destination.name+'\n').encode())
    print(json.dumps(dict(path=destination.name,bytes=destination.stat().st_size,sha256=digest,files=len(files))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out)
