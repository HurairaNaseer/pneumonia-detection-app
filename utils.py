"""Faisla, boxes, DICOM aur drawing (numpy/PIL/cv2 only; torch yahan nahi)."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

FINDING_INFO = (
    "A hazy/white area (opacity) in the lung that looks like pneumonia, an infection "
    "of the lung tissue. An X-ray finding alone is not a diagnosis: the doctor combines it "
    "with symptoms (fever, cough, breathing difficulty), examination and blood tests."
)
BOX_COLOR = (220, 38, 38)


def dicom_to_image(file) -> Image.Image:
    """DICOM -> 8-bit grayscale RGB. Training wale logic jaisa."""
    import pydicom

    ds = pydicom.dcmread(file)
    arr = ds.pixel_array.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    return Image.fromarray(arr.astype("uint8")).convert("RGB")


def looks_like_xray(img: Image.Image) -> bool:
    """X-ray grayscale hoti hai. Rangeen photo ko reject karne ke liye."""
    a = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.int16)
    return float((a.max(axis=2) - a.min(axis=2)).mean()) < 12


def location_of(box, width: int, height: int) -> str:
    """Standard PA X-ray mein patient ka RIGHT side image ke LEFT par hota hai."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
    side = "Right" if cx < 0.5 else "Left"
    zone = "upper" if cy < 1 / 3 else "middle" if cy < 2 / 3 else "lower"
    return f"{side} lung, {zone} zone"


# ---------------- faisla ----------------
def _pct(ref_sorted, x: float) -> float:
    ref = np.asarray(ref_sorted, dtype=np.float64)
    return float(np.searchsorted(ref, x, side="right") / len(ref))


def combined_score(cfg: dict, cls_prob: float, det_top_conf: float) -> float:
    """Training (Cell 5) jaisa: dono scores ko val ke percentile mein badlo, phir average."""
    return (_pct(cfg["val_cls"], cls_prob) + _pct(cfg["val_det"], det_top_conf)) / 2


def zone_of(cfg: dict, score: float) -> str:
    """'no' / 'uncertain' / 'yes'."""
    if score < cfg["lo"]:
        return "no"
    if score >= cfg["hi"]:
        return "yes"
    return "uncertain"


def make_dets(raw_boxes, width: int, height: int, min_conf: float):
    dets = []
    for conf, box in raw_boxes:
        if conf >= min_conf:
            dets.append({"label": "Pneumonia", "confidence": float(conf), "box": tuple(box),
                         "location": location_of(box, width, height)})
    dets.sort(key=lambda d: -d["confidence"])
    return dets


def summarize(zone: str, dets):
    """Return (level, headline, explanation). level: None / Uncertain / High."""
    where = ", ".join(dict.fromkeys(d["location"] for d in dets)) if dets else ""
    if zone == "no":
        return ("None", "No pneumonia detected",
                "The model found no clear sign of pneumonia in this image. This does not rule out "
                "pneumonia (especially early or mild disease) or other lung conditions such as TB or "
                "tumours. If there are symptoms, please see a doctor.")
    if zone == "uncertain":
        extra = f" The most suspicious area(s): {where}." if dets else ""
        return ("Uncertain", "Uncertain - please consult a doctor",
                "The model is not confident either way for this image: it may or may not be pneumonia." + extra +
                " A doctor should review the X-ray.")
    if dets:
        return ("High", "Pneumonia suspected",
                f"{len(dets)} suspicious region(s) found (highest confidence {dets[0]['confidence'] * 100:.0f}%) "
                f"in: {where}. A doctor should confirm this with clinical examination and, if needed, further tests.")
    return ("High", "Pneumonia suspected",
            "The model sees signs of pneumonia in this image but could not mark one specific region. "
            "A doctor should confirm this with clinical examination and, if needed, further tests.")


def draw_detections(pil_img: Image.Image, dets) -> Image.Image:
    img = np.array(pil_img.convert("RGB"))
    thickness = max(2, img.shape[1] // 300)
    scale = max(0.5, img.shape[1] / 1300)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, d in enumerate(dets, 1):
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, thickness)
        text = f"{i}. {d['label']} {d['confidence'] * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
        ty = max(y1, th + 8)
        cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 8, ty), BOX_COLOR, -1)
        cv2.putText(img, text, (x1 + 4, ty - 5), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(img)


def pil_to_bytes(img: Image.Image, fmt: str = "PNG", max_side: int | None = None) -> bytes:
    if max_side and max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
    return buf.getvalue()