"""Convert REPORT.md to REPORT.pdf (simple markdown subset: headings, paragraphs, bullets, numbered lists, tables, bold, code)."""
import os, re, html
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = open(os.path.join(ROOT, "REPORT.md"), encoding="utf-8").read().splitlines()
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=9.5, leading=13, spaceAfter=4)
cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10, spaceAfter=0)
cellh = ParagraphStyle("ch", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=19, leading=23, spaceAfter=8, textColor=colors.HexColor("#12304f"))
h2 = ParagraphStyle("h2", parent=ss["Heading1"], fontSize=14, leading=17, spaceBefore=12, spaceAfter=5, textColor=colors.HexColor("#12304f"))
h3 = ParagraphStyle("h3", parent=ss["Heading2"], fontSize=11, leading=14, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#2a5a8a"))
bul = ParagraphStyle("bl", parent=body, leftIndent=14, bulletIndent=3)


def inline(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Courier' size='8.5'>\1</font>", t)
    return t


story, i = [], 0
while i < len(src):
    ln = src[i]
    if ln.startswith("# "):
        story.append(Paragraph(inline(ln[2:]), h1))
    elif ln.startswith("## "):
        story.append(Paragraph(inline(ln[3:]), h2))
    elif ln.startswith("### "):
        story.append(Paragraph(inline(ln[4:]), h3))
    elif ln.strip() == "---":
        pass
    elif ln.startswith("|"):
        rows = []
        while i < len(src) and src[i].startswith("|"):
            cells = [c.strip() for c in src[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                rows.append(cells)
            i += 1
        i -= 1
        n = max(len(r) for r in rows)
        data = [[Paragraph(inline(c), cellh if r == 0 else cell) for c in row + [""] * (n - len(row))] for r, row in enumerate(rows)]
        w = A4[0] - 30 * mm
        first = 0.30 if n > 3 else 0.35
        widths = [w * first] + [w * (1 - first) / (n - 1)] * (n - 1) if n > 1 else [w]
        if n >= 7:
            widths = [w * 0.07, w * 0.19] + [w * 0.74 / (n - 2)] * (n - 2)
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a5a8a")),
                               ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
                               ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c4d0")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        story += [t, Spacer(1, 6)]
    elif ln.startswith("- "):
        story.append(Paragraph(inline(ln[2:]), bul, bulletText="•"))
    elif re.match(r"^\s+- ", ln):
        story.append(Paragraph(inline(ln.strip()[2:]), ParagraphStyle("b2", parent=bul, leftIndent=28, bulletIndent=17), bulletText="-"))
    elif re.match(r"^\d+\. ", ln):
        num, rest = ln.split(". ", 1)
        story.append(Paragraph(inline(rest), bul, bulletText=num + "."))
    elif ln.strip():
        story.append(Paragraph(inline(ln), body))
    i += 1

# appendix: block flow diagram
img = os.path.join(ROOT, "flowsheet_final.png")
if os.path.exists(img):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(img).getSize()
    w = A4[0] - 30 * mm
    story += [PageBreak(), Paragraph("Appendix - block flow diagram", h2), Image(img, width=w, height=w * ih / iw)]


def footer(c, d):
    c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawString(15 * mm, 8 * mm, "Crude + Vacuum Refinery - DWSIM model report")
    c.drawRightString(A4[0] - 15 * mm, 8 * mm, "Page %d" % d.page); c.restoreState()


doc = SimpleDocTemplate(os.path.join(ROOT, "REPORT.pdf"), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                        topMargin=15 * mm, bottomMargin=16 * mm, title="Crude + Vacuum Refinery - DWSIM Model Report")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", os.path.join(ROOT, "REPORT.pdf"))
