"""One pipeline (Rule 3): chapters -> HTML -> WeasyPrint -> footer stamped once (Rule 4) -> checks once (Rule 6).
Usage: python build.py [output.pdf]"""
import importlib
import io
import pathlib
import re
import sys

import pymupdf as fitz  # checks only
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from weasyprint import HTML

import gen
from gen import esc_check
import front
import back

HERE = pathlib.Path(__file__).parent
CHAPTERS = [f"ch{i:02d}" for i in range(1, 31)]   # extended per Part as content groups are written
PARTS = ["I", "II", "III", "IV", "V"]


def assemble():
    """Builds every page in reading order; Contents is generated last (it needs the TOC) and inserted after the cover."""
    body = [esc_check(front.how_to_use(), "how to use")]
    for part in PARTS:
        body.append(gen.partpage(part))
        for name in CHAPTERS:
            mod = importlib.import_module(name)
            if mod.PART == part:
                html = esc_check(mod.build(), name)
                body.append(html.replace('<section class="page">', '<section class="page newpage">', 1))
    files = [HERE / f"{n}.py" for n in CHAPTERS]
    body += [esc_check(back.answer_key(p), f"answer key {p}") for p in PARTS]
    body.append(esc_check(back.glossary(files), "glossary"))
    body = [esc_check(front.cover(), "cover"), esc_check(front.contents(), "contents")] + body
    esc_check("".join(body), "full document body")
    css = (HERE / "style.css").read_text(encoding="utf-8")
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><style>{css}</style></head><body>{"".join(body)}</body></html>'


def stamp(pdf_bytes, out_path):
    """Footer stamped ONCE on the final assembly: series left, 'page N' right, baseline 8 mm."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for i, pg in enumerate(reader.pages, 1):
        w, h = float(pg.mediabox.width), float(pg.mediabox.height)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=(w, h))
        c.setFont("Helvetica-Bold", 7.6); c.setFillColor(HexColor("#0f6b5c"))
        c.drawString(14 * mm, 8 * mm, front.SERIES)
        c.setFont("Helvetica", 7.6); c.setFillColor(HexColor("#1a1f23"))
        c.drawRightString(w - 14 * mm, 8 * mm, f"page {i}")
        c.save()
        pg.merge_page(PdfReader(buf).pages[0])
        writer.add_page(pg)
    with open(out_path, "wb") as fh:
        writer.write(fh)


def checks(path):
    doc = fitz.open(path)
    fails, sizes = [], {}
    for pno, pg in enumerate(doc, 1):
        h = pg.rect.height
        spans = [s for b in pg.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
        if len(spans) <= 2:
            fails.append(f"p{pno}: blank page")
        pn = [s for s in spans if re.fullmatch(r"page \d+", s["text"].strip())]
        if len(pn) != 1 or pn[0]["text"].strip() != f"page {pno}":
            fails.append(f"p{pno}: page-number stamp {[s['text'] for s in pn]}")
        for s in spans:
            font = s["font"].split("+")[-1].replace("-", "")
            if not re.match(r"(NotoSans|JetBrainsMono|Helvetica)", font):
                fails.append(f"p{pno}: font {font}")
            y0, y1 = s["bbox"][1], s["bbox"][3]
            is_footer = s["font"].startswith("Helvetica")
            if not is_footer and (y0 < 24 or y1 > h - 38):
                fails.append(f"p{pno}: text in margin band: {s['text'][:30]!r}")
            if not font.startswith("JetBrains") and not is_footer:
                sizes[round(s["size"], 1)] = sizes.get(round(s["size"], 1), 0) + len(s["text"])
    body = max(sizes, key=sizes.get)
    if not 7.8 <= body <= 9.4:
        fails.append(f"body text size {body}pt")
    small = sorted(k for k in sizes if k < 7.5)
    if small:
        fails.append(f"text below 7.5pt: {small}")
    return len(doc), body, sorted(set(fails))


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "out" / "NeuralSharp_sample.pdf")
    out.parent.mkdir(exist_ok=True)
    html = assemble()
    stamp(HTML(string=html, base_url=str(HERE)).write_pdf(), out)
    pages, body, fails = checks(out)
    print(f"{out.name}: {pages} pages, body {body}pt")
    print("CHECKS: PASS" if not fails else "CHECKS: FAIL\n  " + "\n  ".join(fails[:40]))
