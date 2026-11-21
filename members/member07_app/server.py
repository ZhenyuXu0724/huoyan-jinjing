"""Local camera application. Run python server.py and open the printed URL."""
import base64
import importlib.util
import json
import math
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ("baseline1", "baseline2", "baseline3", "final")
LOCK = threading.Lock()
CACHE = {}


def validate_result(result):
    if not isinstance(result, dict):
        raise ValueError("模型必须返回字典")
    kind = result.get("kind")
    if kind == "image":
        if result.get("decision") not in ("PASS", "FAIL", "REVIEW_REQUIRED"):
            raise ValueError("图像级模型必须返回 decision")
        counts = result.get("fixed_expected")
        if not isinstance(counts, list) or len(counts) != 5 or any(type(n) is not int or n < 0 for n in counts):
            raise ValueError("图像级模型必须声明五类 fixed_expected")
        score = result.get("score")
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError("整图分数必须是有限数字")
        return {"kind": kind, "decision": result["decision"], "fixed_expected": counts, "score": score}
    if kind != "objects":
        raise ValueError("kind 必须是 image 或 objects")
    objects = result.get("objects")
    if not isinstance(objects, list):
        raise ValueError("objects 必须是列表")
    for obj in objects:
        if type(obj.get("class_id")) is not int or obj["class_id"] not in range(6):
            raise ValueError("class_id 必须为课程编号 0 到 5")
        box = obj.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in box):
            raise ValueError("bbox 必须为像素 xyxy")
        if min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("无效的框坐标")
        score = obj.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("score 必须在 0 到 1 之间")
    return {"kind": kind, "objects": objects, "review_required": bool(result.get("review_required", False))}


def configuration():
    return json.loads((ROOT / "models.json").read_text(encoding="utf-8-sig"))


def predict(model_id, image):
    config = configuration()[model_id]
    adapter = config.get("adapter")
    if not adapter:
        raise RuntimeError("该模型尚未接入。摄像头可以继续预览和拍照。")
    key = (model_id, json.dumps(config, sort_keys=True))
    if key not in CACHE:
        path = (ROOT / adapter).resolve()
        spec = importlib.util.spec_from_file_location("adapter_" + model_id, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        CACHE[key] = module.load(config, ROOT)
    start = time.perf_counter()
    result = validate_result(CACHE[key](image))
    result["inference_ms"] = round((time.perf_counter() - start) * 1000, 2)
    result["model"] = model_id
    result["version"] = config.get("version", "未注明版本")
    return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif self.path == "/models":
            config = configuration()
            body = json.dumps({name: {"ready": bool(config[name].get("adapter")), "version": config[name].get("version", "")} for name in MODELS}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            self.send_error(404)
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return
        # Only requests from this local page can invoke a configured model.
        if self.headers.get("Origin") != "http://" + self.headers.get("Host", ""):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16000000:
                raise ValueError("图片请求过大或为空")
            data = json.loads(self.rfile.read(length))
            if data.get("model") not in MODELS:
                raise ValueError("未知模型")
            raw = base64.b64decode(data["image"], validate=True)
            with LOCK:
                result = predict(data["model"], raw)
            code = 200
        except Exception as error:
            result = {"error": str(error)}
            code = 400
        body = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    address = ("127.0.0.1", 8767)
    server = ThreadingHTTPServer(address, Handler)
    print("Open http://127.0.0.1:8767   Press Ctrl+C to stop.", flush=True)
    webbrowser.open("http://127.0.0.1:8767")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
