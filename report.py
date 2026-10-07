"""PDF report builder (ReportLab)."""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from utils import FINDING_INFO, pil_to_bytes

LIMITATIONS = (
    "This tool only looks for pneumonia-like opacities. A normal result does not exclude pneumonia "
    "(especially early or subtle disease) or other conditions such as tuberculosis, lung nodules or "
    "tumours, heart problems or fluid around the lung. The boxes show where the model sees a "
    "suspicious area; their edges are approximate. Image quality, patient position and "
    "equipment can change the result."
)
DISCLAIMER = (
    "This report is generated automatically by an AI system. It is NOT a medical diagnosis and must "
    "not be used on its own to start or stop treatment. Please show this report and the original "
    "X-ray to a qualified doctor."
)
LEVEL_COLORS = {
    "High": colors.HexColor("#b91c1c"),
    "Uncertain": colors.HexColor("#c2410c"),
    "None": colors.HexColor("#15803d"),
}


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def build_pdf(patient: dict, original, annotated, dets, summary, score: float | None = None,
              metrics: dict | None = None, n_test: int | None = None) -> bytes:
    """Return the PDF as bytes. summary = (level, headline, explanation) from utils.summarize."""
    level, headline, explanation = summary
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=11 * mm, bottomMargin=11 * mm, title="Chest X-ray Pneumonia Report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=7, spaceAfter=3)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5, textColor=colors.HexColor("#444444"))
    cell = ParagraphStyle("cell", parent=body, fontSize=8.5, leading=11)
    badge_style = ParagraphStyle("badge", parent=body, textColor=colors.white, fontSize=13, leading=16)

    meta = f"Generated: {datetime.now():%d %b %Y, %H:%M}"
    if score is not None:
        meta += f"   |   Screening score: {score * 100:.0f}/100"
    story = [Paragraph("Chest X-ray Pneumonia Screening Report", h1), _p(meta, small), Spacer(1, 6)]

    pt = [
        [_p("Patient name", cell), _p(patient.get("name") or "-", cell),
         _p("Patient ID", cell), _p(patient.get("id") or "-", cell)],
        [_p("Age", cell), _p(patient.get("age") or "-", cell),
         _p("Sex", cell), _p(patient.get("sex") or "-", cell)],
    ]
    t = Table(pt, colWidths=[28 * mm, 55 * mm, 28 * mm, 55 * mm])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f3f4f6")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f3f4f6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [t, Paragraph("Result", h2)]

    badge = Table([[_p(f"Result: {headline}", badge_style)]], colWidths=[180 * mm])
    badge.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LEVEL_COLORS.get(level, colors.grey)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story += [badge, Spacer(1, 5), _p(explanation, body), Paragraph("Images", h2)]

    def rl_img(img, width):
        data = pil_to_bytes(img, "JPEG", max_side=1000)
        w, h = img.size
        return RLImage(io.BytesIO(data), width=width, height=width * h / w)

    if dets:
        imgs = Table([[rl_img(original, 58 * mm), rl_img(annotated, 58 * mm)],
                      [_p("Original image", small), _p("Suspicious regions (numbered boxes)", small)]],
                     colWidths=[90 * mm, 90 * mm])
    else:
        imgs = Table([[rl_img(original, 58 * mm)], [_p("Original image (no region marked)", small)]],
                     colWidths=[90 * mm])
        imgs.hAlign = "LEFT"
    imgs.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(imgs)

    if dets:
        story.append(Paragraph("Detected regions", h2))
        rows = [[_p(x, cell) for x in ("#", "Finding", "Confidence", "Location", "Box (x1,y1,x2,y2 px)")]]
        for i, d in enumerate(dets, 1):
            rows.append([_p(i, cell), _p(d["label"], cell), _p(f"{d['confidence'] * 100:.1f}%", cell),
                         _p(d["location"], cell), _p(", ".join(str(int(v)) for v in d["box"]), cell)])
        ft = Table(rows, colWidths=[8 * mm, 28 * mm, 24 * mm, 52 * mm, 68 * mm], repeatRows=1)
        ft.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story += [ft, Paragraph("What this means", h2), _p(FINDING_INFO, body)]

    if metrics:
        rel = (f"On {n_test or '?'} unseen test X-rays the system found {metrics['sensitivity'] * 100:.0f}% of "
               f"pneumonia cases and correctly cleared {metrics['specificity'] * 100:.0f}% of non-pneumonia "
               f"cases (AUC {metrics['auc']:.2f}). So it makes mistakes in both directions. "
               "'Uncertain' means the model is not sure and a doctor must decide.")
        story += [Paragraph("How reliable is this?", h2), _p(rel, small)]

    story += [Paragraph("Limitations", h2), _p(LIMITATIONS, small),
              Paragraph("Disclaimer", h2), _p(DISCLAIMER, small)]
    doc.build(story)
    return buf.getvalue()