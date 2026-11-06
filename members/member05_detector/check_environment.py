"""Run CUDA convolution forward/backward, not only a device-presence check."""
import json
import platform
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--weights', help='Optional existing official YOLOv8n checkpoint for synthetic-input smoke test')
    group.add_argument('--architecture-smoke', action='store_true', help='Use packaged architecture with untrained random weights; runtime test only')
    args = parser.parse_args()
    import torch
    import ultralytics
    info = {"python": platform.python_version(), "torch": torch.__version__,
            "ultralytics": ultralytics.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available()}
    if not info["cuda_available"]:
        print(json.dumps(info, indent=2))
        raise SystemExit("CUDA unavailable")
    info["gpu"] = torch.cuda.get_device_name(0)
    info["capability"] = torch.cuda.get_device_capability(0)
    layer = torch.nn.Conv2d(3, 16, 3).cuda()
    x = torch.randn(2, 3, 64, 64, device="cuda", requires_grad=True)
    loss = layer(x).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item() or not torch.isfinite(x.grad).all().item():
        raise RuntimeError("CUDA result is not finite")
    info["cuda_forward_backward"] = "PASS"
    if args.weights or args.architecture_smoke:
        path = Path(args.weights).resolve() if args.weights else None
        if path is not None and not path.is_file():
            parser.error('Weights must exist locally; no downloads')
        from ultralytics import YOLO
        import numpy as np
        model = YOLO(str(path) if path else 'yolov8n.yaml')
        result = model.predict(np.full((480,640,3),255,dtype=np.uint8), device=0,
                               imgsz=640, verbose=False)[0]
        torch.cuda.synchronize()
        if not torch.isfinite(result.boxes.data).all().item():
            raise RuntimeError('YOLO produced non-finite outputs')
        info['yolo_gpu_inference'] = 'PASS_SYNTHETIC_WHITE_INPUT_ONLY_NOT_ACCURACY'
        info['yolo_weights'] = 'pretrained' if path else 'UNTRAINED_RANDOM_ARCHITECTURE_SMOKE_ONLY'
        info['yolo_device'] = str(model.predictor.model.device)
        if model.predictor.model.device.type != 'cuda':
            raise RuntimeError('YOLO did not use CUDA')
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
