"""Models load karna aur chalana (torch + ultralytics). Faisle ka hisaab utils.py mein hai."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

NEEDED = ("classifier.pt", "best.pt", "combo_config.json")


def missing_files(weights_dir) -> list[str]:
    wd = Path(weights_dir)
    return [f for f in NEEDED if not (wd / f).exists()]


def load_models(weights_dir) -> dict:
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms as T
    from ultralytics import YOLO

    wd = Path(weights_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(wd / "classifier.pt", map_location="cpu")
    if ck["arch"] == "densenet121":
        clf = torchvision.models.densenet121(weights=None)
        clf.classifier = nn.Linear(clf.classifier.in_features, 1)
    else:
        clf = torchvision.models.efficientnet_b0(weights=None)
        clf.classifier[1] = nn.Linear(clf.classifier[1].in_features, 1)
    clf.load_state_dict(ck["state_dict"])
    clf = clf.to(dev).eval()

    size = int(ck["img"])
    tf = T.Compose([T.Resize((size, size)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return {
        "clf": clf, "tf": tf, "dev": dev,
        "det": YOLO(str(wd / "best.pt")),
        "cfg": json.loads((wd / "combo_config.json").read_text()),
    }


def classifier_score(bundle: dict, img: Image.Image) -> float:
    """0..1 (training jaisa: original + horizontal flip ka average)."""
    import torch

    x = bundle["tf"](img.convert("RGB")).unsqueeze(0).to(bundle["dev"])
    with torch.inference_mode():
        p1 = torch.sigmoid(bundle["clf"](x).squeeze(1)).float()
        p2 = torch.sigmoid(bundle["clf"](torch.flip(x, dims=[3])).squeeze(1)).float()
    return float(((p1 + p2) / 2).item())


def detector_boxes(bundle: dict, img: Image.Image) -> list[tuple[float, tuple[float, float, float, float]]]:
    """Saare boxes (conf >= 0.01): [(confidence, (x1,y1,x2,y2)), ...], upar se neeche."""
    imgsz = int(bundle["cfg"].get("det_imgsz", 640))
    res = bundle["det"].predict(img.convert("RGB"), conf=0.01, imgsz=imgsz, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        out.append((float(b.conf[0]), (x1, y1, x2, y2)))
    out.sort(key=lambda t: -t[0])
    return out