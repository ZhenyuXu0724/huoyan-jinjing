"""Optional adapter for a team's Ultralytics weight with explicit class mapping."""
import io


def load(config, root):
    from PIL import Image
    from ultralytics import YOLO
    weights = root / config["weights"]
    mapping = {int(k): int(v) for k, v in config["class_map"].items()}
    model = YOLO(str(weights))
    if set(mapping) != set(model.names) or any(v not in range(6) for v in mapping.values()):
        raise ValueError("请为模型全部类别配置课程编号映射")

    def run(raw):
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        results = model.predict(image, conf=config.get("conf", 0.65), imgsz=config.get("imgsz", 640), verbose=False)
        objects = []
        for box in results[0].boxes:
            objects.append({"class_id": mapping[int(box.cls.item())],
                            "bbox": box.xyxy[0].tolist(), "score": float(box.conf.item())})
        # Unknown rejection must be confirmed by the model owner before PASS is enabled.
        return {"kind": "objects", "objects": objects,
                "review_required": not config.get("unknown_rejection_verified", False)}
    return run
