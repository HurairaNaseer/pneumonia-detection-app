"""Streamlit app: chest X-ray upload -> pneumonia (haan / nahi / uncertain) + boxes + PDF report.

Chalane ka tareeqa:   streamlit run app.py
weights/ folder mein yeh 5 files honi chahiyen:
classifier.pt, cls_config.json, best.pt, config.json, combo_config.json
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from engine import classifier_score, detector_boxes, load_models, missing_files
from report import build_pdf
from utils import (combined_score, dicom_to_image, draw_detections, looks_like_xray, make_dets,
                   pil_to_bytes, summarize, zone_of)

WEIGHTS = os.environ.get("CXR_WEIGHTS", "weights")

st.set_page_config(page_title="Pneumonia X-ray Scanner", page_icon="🫁", layout="wide")


@st.cache_resource(show_spinner="Models load ho rahe hain (pehli baar thora time lagta hai)...")
def get_models(path: str):
    return load_models(path)


def load_image(uploaded) -> Image.Image:
    if uploaded.name.lower().endswith((".dcm", ".dicom")):
        return dicom_to_image(uploaded)
    return Image.open(uploaded).convert("RGB")


st.title("🫁 Pneumonia X-ray Scanner")
st.caption("Chest X-ray upload karein. Model batayega: pneumonia hai, nahi hai, ya pakka nahi keh sakta. "
           "Pneumonia ki surat mein jagah par box lagega.")

miss = missing_files(WEIGHTS)
if miss:
    st.error(f"`{WEIGHTS}/` folder mein yeh files nahi mili: {', '.join(miss)}")
    st.stop()

bundle = get_models(WEIGHTS)
cfg = bundle["cfg"]
metrics = cfg.get("test_metrics", {})

with st.sidebar:
    st.header("Patient info")
    name = st.text_input("Name")
    pid = st.text_input("Patient ID")
    age = st.text_input("Age")
    sex = st.selectbox("Sex", ["", "Male", "Female", "Other"])
    st.header("Settings")
    box_conf = st.slider("Box dikhane ki threshold", 0.05, 0.90, float(cfg.get("det_conf", 0.12)), 0.01,
                         help="Kam karne se zyada boxes dikhenge (zyada galat bhi). Faisla (haan/nahi/uncertain) is se nahi badalta.")
    skip_check = st.checkbox("X-ray check band karein", help="Agar asli X-ray ko 'rangeen photo' keh kar rok diya jaye.")
    if metrics:
        st.header("Model ki test performance")
        st.caption(f"{cfg.get('test_images', '?')} unseen images par:")
        st.write(f"Pneumonia pakadna (sensitivity): **{metrics['sensitivity']:.0%}**")
        st.write(f"Normal ko normal kehna (specificity): **{metrics['specificity']:.0%}**")
        st.write(f"AUC: **{metrics['auc']:.2f}**")
        st.caption("Matlab model ghalti bhi karta hai. Yeh doctor ka badal nahi hai.")

uploaded = st.file_uploader("Chest X-ray upload karein", type=["png", "jpg", "jpeg", "dcm"])

if uploaded:
    try:
        image = load_image(uploaded)
    except Exception as e:  # noqa: BLE001
        st.error(f"Image open nahi ho saki: {e}")
        st.stop()

    if not skip_check and not looks_like_xray(image):
        st.error("Yeh chest X-ray nahi lagti (rangeen image hai). Sahi X-ray upload karein.")
        st.stop()

    with st.spinner("Scan ho raha hai..."):
        cls_p = classifier_score(bundle, image)
        raw = detector_boxes(bundle, image)
        top_det = raw[0][0] if raw else 0.0
        score = combined_score(cfg, cls_p, top_det)
        zone = zone_of(cfg, score)
        w, h = image.size
        dets = make_dets(raw, w, h, box_conf) if zone in ("yes", "uncertain") else []
        annotated = draw_detections(image, dets)

    summary = summarize(zone, dets)
    level, headline, explanation = summary
    box = {"None": st.success, "Uncertain": st.warning, "High": st.error}[level]
    box(f"{'✅' if level == 'None' else '⚠️' if level == 'Uncertain' else '🚨'} **{headline}**\n\n{explanation}")
    st.progress(min(max(score, 0.0), 1.0), text=f"Screening score: {score * 100:.0f}/100 "
                f"(No pneumonia < {cfg['lo'] * 100:.0f} <= Uncertain < {cfg['hi'] * 100:.0f} <= Pneumonia)")

    if dets:
        c1, c2 = st.columns(2)
        c1.image(image, caption="Original", use_container_width=True)
        c2.image(annotated, caption="Suspicious regions", use_container_width=True)
        st.subheader("Detected regions")
        st.dataframe(
            pd.DataFrame([{"#": i, "Finding": d["label"], "Confidence %": round(d["confidence"] * 100, 1),
                           "Location": d["location"],
                           "Box (x1,y1,x2,y2)": ", ".join(str(int(v)) for v in d["box"])}
                          for i, d in enumerate(dets, 1)]),
            hide_index=True, use_container_width=True)
    else:
        st.image(image, caption="Original (koi region mark nahi hui)", width=520)

    st.caption("Yeh result sirf pneumonia ke baare mein hai. Doosri bimariyan (TB, tumour, etc.) is se rule out nahi hotin. "
               "Doctor se confirm zaroor karwayein.")

    pdf = build_pdf({"name": name, "id": pid, "age": age, "sex": sex}, image, annotated, dets, summary,
                    score=score, metrics=metrics or None, n_test=cfg.get("test_images"))
    d1, d2 = st.columns(2)
    d1.download_button("📄 Report download (PDF)", pdf, file_name="pneumonia_report.pdf",
                       mime="application/pdf", type="primary", use_container_width=True)
    d2.download_button("🖼️ Image download (PNG)", pil_to_bytes(annotated),
                       file_name="xray_result.png", mime="image/png", use_container_width=True)