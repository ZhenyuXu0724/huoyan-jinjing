"""Independent portable v2 archive; no raw images or virtual environments."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from .common import ROOT,MODULE,OUT,WEIGHTS,read
from .assess import verify_freeze


def build(destination):
    verify_freeze()
    verified=read(OUT/'verification.json')
    if verified['experiments']!=33 or verified['old_preservation']!='PASS':
        raise ValueError('Complete verification first')
    paths=[]
    for folder in (MODULE,OUT,WEIGHTS,ROOT/'outputs/member04_hog/experiments'):
        paths += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.suffix.lower() not in ('.pyc','.jpg','.jpeg','.png')]
    paths += [ROOT/p for p in ('shared/contracts.py','shared/catalogue.json',
              'weights/member04_hog/baseline1.joblib')]
    for name in ('train.csv','valid.csv','test.csv','provenance.json'):
        paths.append(ROOT/'data/member04_hog/supervised_v2'/name)
    payloads={p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted(set(paths))}
    payloads['DELIVERY_README.md']=(
        '# baseline1 supervised v2 交付包\n\n'
        '先读 members/member04_hog/supervised_v2/README.md 与 docs/COMPARISON_V2.md。\n'
        '包含全部33次新版模型、训练证据、旧默认A2模型及旧33次实验指标；不包含旧版全部消融权重、原始图片或虚拟环境。\n'
        '旧版完整资产仍在原目录/原Release，未被覆盖。old_preservation.json是原工作区的全量保留证据，'
        '本包不会冒充包含其全部文件。模型推理可独立运行；重新训练需要按来源恢复旧数据再准备新增数据。\n\n'
        'Python 3.11；先建立新虚拟环境并安装模块 requirements.txt，所有命令在解压根目录运行。\n'
        '校验：python -m members.member04_hog.delivery --verify .\n'
        '推理：python -m members.member04_hog.supervised_v2.inference --image your_image.jpg\n'
        '单元检查：python -m unittest discover -s members/member04_hog/tests -v\n'
    ).encode('utf-8')
    payloads['DELIVERY_SHA256.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()},
                                               ensure_ascii=False,indent=2).encode('utf-8')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for path,payload in sorted(payloads.items()):archive.writestr(path,payload)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix('.zip.sha256').write_text(digest+'  '+destination.name+'\n',encoding='utf-8')
    print(f'Packaged {len(payloads)} files; {destination.stat().st_size} bytes; SHA256 {digest}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    build(parser.parse_args().out)
