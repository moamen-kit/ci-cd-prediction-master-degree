"""Render the corrected thesis markdown to PDF.

LibreOffice cannot load Office formats in this container, so the PDF is built
from 4_Thesis_Source_Markdown.md — the authoritative text the .docx is itself
derived from — rather than converted from the .docx. Page geometry matches the
.docx (US Letter, 1 inch margins).
"""
import re, os, html
from pathlib import Path

import matplotlib
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak, KeepTogether,
)
import arabic_reshaper
from bidi.algorithm import get_display

MD   = Path("final-version-doc/files/4_Thesis_Source_Markdown.md")
FIGS = Path("cicd-failure-prediction/figures")
OUT  = Path("final-version-doc/files/1_MSc_Thesis_FINAL.pdf")

MPL = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
DEJAVU = Path("/usr/share/fonts/truetype/dejavu")

pdfmetrics.registerFont(TTFont("Serif",   MPL / "DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont("Serif-B", MPL / "DejaVuSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Serif-I", MPL / "DejaVuSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont("Serif-BI", MPL / "DejaVuSerif-BoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("Mono",    DEJAVU / "DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("Arabic",  DEJAVU / "DejaVuSans.ttf"))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B",
                              italic="Serif-I", boldItalic="Serif-BI")

PAGE_W, PAGE_H = letter
MARGIN = 1.0 * inch
AVAIL = PAGE_W - 2 * MARGIN

body = ParagraphStyle("body", fontName="Serif", fontSize=10.5, leading=15.5,
                      alignment=TA_JUSTIFY, spaceAfter=7)
quote = ParagraphStyle("quote", parent=body, leftIndent=22, rightIndent=14,
                       fontName="Serif-I", textColor=colors.HexColor("#333333"),
                       spaceBefore=5, spaceAfter=8)
caption = ParagraphStyle("cap", parent=body, fontName="Serif-I", fontSize=9,
                         leading=12.5, alignment=TA_CENTER,
                         textColor=colors.HexColor("#333333"),
                         spaceBefore=4, spaceAfter=13)
bullet = ParagraphStyle("bul", parent=body, leftIndent=20, bulletIndent=8,
                        spaceAfter=3, alignment=TA_JUSTIFY)
H = {
 1: ParagraphStyle("h1", fontName="Serif-B", fontSize=17, leading=21,
                   spaceBefore=4, spaceAfter=13,
                   textColor=colors.HexColor("#1F3864")),
 2: ParagraphStyle("h2", fontName="Serif-B", fontSize=13, leading=17,
                   spaceBefore=15, spaceAfter=7,
                   textColor=colors.HexColor("#1F3864")),
 3: ParagraphStyle("h3", fontName="Serif-B", fontSize=11.5, leading=15,
                   spaceBefore=12, spaceAfter=6),
 4: ParagraphStyle("h4", fontName="Serif-B", fontSize=10.5, leading=14,
                   spaceBefore=10, spaceAfter=5),
}
title_style = ParagraphStyle("title", fontName="Serif-B", fontSize=21, leading=27,
                             alignment=TA_CENTER, spaceAfter=20,
                             textColor=colors.HexColor("#1F3864"))

# Anchored on Arabic characters at both ends, so trailing spaces that
# separate the Arabic from following Latin text are not swallowed.
_AR = r'؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿'
ARABIC = re.compile(f'[{_AR}]+(?:[ \t]+[{_AR}]+)*')

def shape_arabic(text):
    """Reshape and bidi-order Arabic runs, and tag them with an Arabic font."""
    def repl(m):
        seg = m.group(0)
        shaped = get_display(arabic_reshaper.reshape(seg.strip()))
        return f'<font name="Arabic">{shaped}</font>'
    return ARABIC.sub(repl, text)

def inline(text):
    """Markdown inline formatting -> reportlab mini-HTML."""
    t = html.escape(text, quote=False)
    t = re.sub(r'`([^`]+?)`', r'<font name="Mono" size="9">\1</font>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', t, flags=re.S)
    t = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', t)
    t = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'\1', t)
    return shape_arabic(t)

def col_widths(rows, ncols):
    """Proportional widths from the longest cell per column, normalised to AVAIL."""
    w = []
    for c in range(ncols):
        longest = max((len(r[c]) if c < len(r) else 0) for r in rows)
        w.append(max(longest, 4) ** 0.62)
    total = sum(w)
    return [AVAIL * x / total for x in w]

def build_table(rows):
    ncols = max(len(r) for r in rows)
    fs = 8.0 if ncols <= 6 else (7.0 if ncols <= 8 else 6.2)
    cs = ParagraphStyle("c", fontName="Serif", fontSize=fs, leading=fs + 2.4)
    hs = ParagraphStyle("ch", fontName="Serif-B", fontSize=fs, leading=fs + 2.4,
                        textColor=colors.white)
    data = []
    for ri, r in enumerate(rows):
        row = []
        for ci in range(ncols):
            txt = r[ci] if ci < len(r) else ""
            row.append(Paragraph(inline(txt), hs if ri == 0 else cs))
        data.append(row)
    t = Table(data, colWidths=col_widths(rows, ncols), repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9AA5B1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F5F9")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t

def figure(path):
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    w = min(AVAIL, 6.1 * inch)
    return Image(str(path), width=w, height=w * ih / iw)

# ------------------------------------------------------------------ parse
lines = MD.read_text(encoding="utf-8").split("\n")
story, i, first_h1 = [], 0, True
stats = dict(h=0, p=0, t=0, img=0, cap=0, li=0, q=0)

while i < len(lines):
    ln = lines[i]; s = ln.strip()
    if not s or s == "---":
        i += 1; continue

    m = re.match(r'!\[([^\]]*)\]\((figures/[^)]+)\)', s)
    if m:
        p = FIGS / Path(m.group(2)).name
        if p.exists():
            story.append(Spacer(1, 6)); story.append(figure(p)); stats["img"] += 1
        i += 1; continue

    m = re.match(r'^(#{1,6})\s+(.*)$', s)
    if m:
        lvl = min(len(m.group(1)), 4); txt = m.group(2)
        if lvl == 1:
            if first_h1:
                story.append(Spacer(1, 1.1 * inch))
                story.append(Paragraph(inline(txt), title_style))
                first_h1 = False; stats["h"] += 1; i += 1; continue
            story.append(PageBreak())
        story.append(Paragraph(inline(txt), H[lvl])); stats["h"] += 1
        i += 1; continue

    if s.startswith("|") and i + 1 < len(lines) and re.match(r'^\|[\s:|-]+\|$', lines[i+1].strip()):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            r = lines[i].strip()
            if not re.match(r'^\|[\s:|-]+\|$', r):
                rows.append([c.strip() for c in r.strip("|").split("|")])
            i += 1
        if rows:
            story.append(Spacer(1, 4)); story.append(build_table(rows))
            story.append(Spacer(1, 11)); stats["t"] += 1
        continue

    if s.startswith("*") and s.endswith("*") and not s.startswith("**") and len(s) > 40:
        story.append(Paragraph(inline(s.strip("*")), caption)); stats["cap"] += 1
        i += 1; continue

    if s.startswith(">"):
        buf = []
        while i < len(lines) and lines[i].strip().startswith(">"):
            buf.append(lines[i].strip().lstrip(">").strip()); i += 1
        txt = " ".join(x for x in buf if x)
        if txt:
            story.append(Paragraph(inline(txt), quote)); stats["q"] += 1
        continue

    m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)$', ln)
    if m:
        ordered = m.group(2) not in ("-", "*")
        story.append(Paragraph(inline(m.group(3)), bullet,
                               bulletText="•" if not ordered else m.group(2)))
        stats["li"] += 1; i += 1; continue

    buf = []
    while i < len(lines):
        cur = lines[i].strip()
        if (not cur or cur == "---" or cur.startswith("|") or cur.startswith("#")
                or cur.startswith(">") or cur.startswith("![")
                or re.match(r'^(\s*)([-*]|\d+\.)\s+', lines[i])):
            break
        # A line that is entirely bold is a standalone line (title blocks),
        # not a soft-wrapped continuation of the paragraph.
        standalone = bool(re.fullmatch(r'\*\*.+\*\*', cur))
        if standalone and buf:
            break
        buf.append(cur); i += 1
        if standalone:
            break
    if buf:
        story.append(Paragraph(inline(" ".join(buf)), body)); stats["p"] += 1

# ------------------------------------------------------------------ build
def footer(canv, doc):
    canv.saveState()
    canv.setFont("Serif", 8.5)
    canv.setFillColor(colors.HexColor("#666666"))
    if doc.page > 1:
        canv.drawCentredString(PAGE_W / 2, 0.58 * inch, str(doc.page))
        canv.drawString(MARGIN, 0.58 * inch, "MSc Thesis — Moamen Mohamed Aly Hussein")
        canv.drawRightString(PAGE_W - MARGIN, 0.58 * inch, "Cairo University")
    canv.restoreState()

doc = BaseDocTemplate(str(OUT), pagesize=letter,
                      leftMargin=MARGIN, rightMargin=MARGIN,
                      topMargin=MARGIN, bottomMargin=0.9 * inch,
                      title="Predicting CI/CD Pipeline Build Failures Using Machine Learning Techniques",
                      author="Moamen Mohamed Aly Hussein",
                      subject="MSc Software Engineering thesis, Cairo University")
frame = Frame(MARGIN, 0.9 * inch, AVAIL, PAGE_H - MARGIN - 0.9 * inch, id="f")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
doc.build(story)
print("headings %(h)d · paragraphs %(p)d · quotes %(q)d · tables %(t)d · "
      "figures %(img)d · captions %(cap)d · list items %(li)d" % stats)
