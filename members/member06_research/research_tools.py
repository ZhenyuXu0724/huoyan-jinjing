"""Stage 3 building blocks, not a final research algorithm or a size classifier."""
from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol, Sequence

import cv2
import numpy as np
from PIL import Image
import torch

from .data import Preprocess
from .image_io import read_image


@dataclass(frozen=True)
class Candidate:
    xyxy: tuple[float, float, float, float]
    class_id: int
    score: float


class CandidateProvider(Protocol):
    # Member 5 supplies candidates in ORIGINAL raster coordinates; no expected_counts.
    def propose(self, image) -> Sequence[Candidate]: ...


class ScaleCalibration(Protocol):
    # Member 2 owns calibration identity, perspective correction and uncertainty.
    def pixel_to_metric(self, geometry: dict, *, image_id: str, xyxy: tuple) -> dict: ...


def crop_roi(image, xyxy):
    """Strict half-open xyxy, no silent clipping. Fractional boxes round outward."""
    try:
        image = read_image(image)
        if not isinstance(xyxy, (list, tuple, np.ndarray)) or len(xyxy) != 4:
            raise ValueError("xyxy must have four coordinates")
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Real) or not math.isfinite(x) for x in xyxy):
            raise ValueError("coordinates must be finite real numbers, not bool")
        x1, y1, x2, y2 = xyxy
        if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
            raise ValueError("bbox must be ordered, nonempty and fully inside original raster")
        box = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
        return {"status": "OK", "roi": image.crop(box), "raster_xyxy": box,
                "requested_xyxy": list(xyxy)}
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "INVALID_ROI", "roi": None, "error": str(exc)}


def _gray(image):
    if image is None:
        raise ValueError("empty input")
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("empty input")
    if array.dtype != np.uint8:
        raise ValueError("expected uint8 image; float/NaN input is not implicitly rescaled")
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    raise ValueError("expected grayscale HxW or RGB HxWx3")


def segment_contours(image, *, foreground="dark", threshold=None, min_area_px2=5.0):
    """Explicit foreground polarity; threshold=None uses Otsu. External contours only."""
    try:
        gray = _gray(image)
        if foreground not in {"dark", "light"}:
            raise ValueError("foreground must be dark or light")
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 <= threshold <= 255):
            raise ValueError("threshold must be in [0,255] or None")
        if isinstance(min_area_px2, bool) or not isinstance(min_area_px2, Real) or not math.isfinite(min_area_px2) or min_area_px2 <= 0:
            raise ValueError("min_area_px2 must be positive")
        if gray.min() == gray.max():
            return {"status": "NO_CONTOUR", "mask": None, "contours": []}
        flag = cv2.THRESH_BINARY_INV if foreground == "dark" else cv2.THRESH_BINARY
        if threshold is None:
            flag |= cv2.THRESH_OTSU
        used, mask = cv2.threshold(gray, 0 if threshold is None else threshold, 255, flag)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area_px2]
        return {"status": "OK" if contours else "NO_CONTOUR", "mask": mask,
                "contours": contours, "threshold": float(used)}
    except (ValueError, TypeError, cv2.error) as exc:
        return {"status": "INPUT_ERROR", "mask": None, "contours": [], "error": str(exc)}


def geometry_features(image, **segmentation_options):
    """Measure one unambiguous contour on the UNRESIZED ROI, in pixels only."""
    segmented = segment_contours(image, **segmentation_options)
    base = {"status": segmented["status"], "features": None, "units": "pixels",
            "contour_count": len(segmented["contours"]), "calibration_status": "NOT_CALIBRATED"}
    if segmented["status"] != "OK":
        return {**base, **({"error": segmented["error"]} if "error" in segmented else {})}
    if len(segmented["contours"]) != 1:
        return {**base, "status": "AMBIGUOUS_CONTOURS"}
    contour = segmented["contours"][0]
    height, width = segmented["mask"].shape
    points = contour.reshape(-1, 2)
    if (points[:, 0] <= 0).any() or (points[:, 0] >= width - 1).any() or (points[:, 1] <= 0).any() or (points[:, 1] >= height - 1).any():
        return {**base, "status": "BORDER_TOUCHING"}
    center, sides, angle = cv2.minAreaRect(contour)
    length, short = max(sides), min(sides)
    if short <= 0 or not all(math.isfinite(x) for x in (*center, *sides, angle)):
        return {**base, "status": "DEGENERATE_GEOMETRY"}
    return {**base, "status": "OK", "threshold": segmented["threshold"], "features": {
        "length_px": float(length), "width_px": float(short), "aspect_ratio": float(length / short),
        "contour_area_px2": float(cv2.contourArea(contour)),
        "perimeter_px": float(cv2.arcLength(contour, True)),
        "rect_center_roi_px": list(center), "rect_sides_px": list(sides), "opencv_rect_angle_deg": float(angle),
        "rect_corners_roi_px": cv2.boxPoints((center, sides, angle)).tolist()}}


class LocalFeatureExtractor:
    """Use the backbone, never the whole-image PASS/FAIL head, for local features."""
    def __init__(self, frozen_model, config, device):
        self.model = frozen_model.to(device).eval()
        self.config, self.device = config, device
        self.transform = Preprocess(config)

    @torch.inference_mode()
    def extract(self, image, xyxy):
        cropped = crop_roi(image, xyxy)
        if cropped["status"] != "OK":
            return {"status": cropped["status"], "features": None, "error": cropped.get("error")}
        tensor = self.transform(cropped["roi"]).unsqueeze(0).to(self.device)
        vector = self.model.extract_features(tensor)[0].cpu()
        return {"status": self.config.evidence_status, "features": vector.tolist(), "dimension": len(vector),
                "preprocessing": self.config.to_dict(), "raster_xyxy": cropped["raster_xyxy"],
                "meaning": "frozen CNN embedding, not an object class or physical size"}
