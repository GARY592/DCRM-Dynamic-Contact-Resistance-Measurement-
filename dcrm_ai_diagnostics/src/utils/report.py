from pathlib import Path
from typing import Optional, Dict, Any
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def generate_pdf_report(
    out_path: Path,
    title: str,
    metadata: Dict[str, Any],
    summary: Dict[str, Any],
    image_bytes: Optional[bytes] = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=20*mm, rightMargin=20*mm, topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 8))

    # Metadata section
    story.append(Paragraph('Metadata', styles['Heading2']))
    for key, value in metadata.items():
        story.append(Paragraph(f"<b>{key}</b>: {value}", styles['BodyText']))
    story.append(Spacer(1, 8))

    # Summary section
    story.append(Paragraph('Summary', styles['Heading2']))
    for key, value in summary.items():
        story.append(Paragraph(f"<b>{key}</b>: {value}", styles['BodyText']))
    story.append(Spacer(1, 8))

    # Optional image (e.g., waveform chart)
    if image_bytes:
        # Save to temp file because basic reportlab flowables prefer file paths
        tmp_img = out_path.with_suffix('.png')
        with open(tmp_img, 'wb') as f:
            f.write(image_bytes)
        from reportlab.platypus import Image
        story.append(Paragraph('Waveform', styles['Heading2']))
        story.append(Image(str(tmp_img), width=160*mm, height=90*mm))

    doc.build(story)

    # Write buffer to file
    with open(out_path, 'wb') as f:
        f.write(buffer.getvalue())

    return out_path
