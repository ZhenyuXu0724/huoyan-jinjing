"""Explicit, atomic download of the pinned official COCO initialization checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import time

URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt'
SIZE = 6549796
# Recorded from our HTTPS download, not a publisher-signed checksum.
SHA256 = 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36'


def verify(path):
    return path.is_file() and path.stat().st_size == SIZE and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='weights/yolov8n.pt')
    parser.add_argument('--proxy', help='Optional existing HTTP proxy; no system settings are changed')
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if verify(target):
        print('Existing official initialization checkpoint matches pinned size/hash; no download needed')
        return
    if target.exists():
        parser.error('Existing target has different content; choose a new output path')
    import requests
    proxies = {'http':args.proxy,'https':args.proxy} if args.proxy else None
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix+'.part')
    for attempt in range(3):
        try:
            with requests.get(URL, stream=True, timeout=(20,60), proxies=proxies) as response:
                response.raise_for_status()
                with part.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        stream.write(chunk)
            if not verify(part):
                raise ValueError('Downloaded checkpoint differs from pinned size/hash')
            part.replace(target)
            print(json.dumps({'file':str(target),'source':URL,'size':SIZE,'sha256':SHA256,
                              'purpose':'COCO initialization only, not a project-trained model'}))
            return
        except (requests.RequestException, OSError, ValueError) as exc:
            if attempt == 2:
                parser.exit(2, f'Download failed ({type(exc).__name__}); target was not replaced. Check network/proxy and retry.\n')
            time.sleep(attempt+1)


if __name__ == '__main__':
    main()
