"""Explicit study inference; public baseline1 remains v1."""
import argparse,json
from pathlib import Path
from ..supervised_v2.inference import predict,batch
from .common import ROOT,OUT,read

def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path);p.add_argument('--model',type=Path)
    p.add_argument('--input_dir',type=Path);p.add_argument('--counts_file',type=Path);p.add_argument('--output_dir',type=Path)
    a=p.parse_args();model=a.model or ROOT/read(OUT/'freeze.json')['model_path']
    if a.image:print(json.dumps(predict(a.image,model)))
    elif all((a.input_dir,a.counts_file,a.output_dir)):batch(a.input_dir,a.counts_file,a.output_dir,model)
    else:p.error('Use --image or all batch arguments')

if __name__=='__main__':main()
