"""Write one report as both Markdown and PDF from the same content (notebook 04's last step).

    rep = Report("Far-OOD test", subtitle="run far_ood_2026-11-08")
    rep.heading("1. Summary"); rep.bullets(["...", "..."]); rep.table(df); rep.figure("figures/x.png", "caption")
    rep.save(run_dir)  # -> run_dir/report.md and run_dir/report.pdf

Figures are given as paths relative to the run folder, so the Markdown links keep working when the
folder is copied (e.g. from Drive into the repo). The PDF needs `reportlab` (pip install reportlab).
"""
import os

import pandas as pd

FONT_DIRS = ["/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/liberation/"]
PLAIN = {"→": "->", "≥": ">=", "≤": "<=", "–": "-", "—": "-", "“": '"', "”": '"', "’": "'", "×": "x", "≈": "~"}


class Report:
    def __init__(self, title, subtitle=""):
        self.blocks = [("title", title, subtitle)]

    def heading(self, text, level=1):
        self.blocks.append(("heading", text, level))

    def text(self, text):
        self.blocks.append(("text", text))

    def bullets(self, items):
        self.blocks.append(("bullets", list(items)))

    def table(self, df, max_rows=60, round_to=3):
        df = df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
        self.blocks.append(("table", df.head(max_rows).round(round_to)))

    def figure(self, rel_path, caption=""):
        self.blocks.append(("figure", rel_path, caption))

    # ------------------------------------------------------------ markdown
    def markdown(self):
        out = []
        for b in self.blocks:
            if b[0] == "title":
                out += [f"# {b[1]}", "", b[2], ""]
            elif b[0] == "heading":
                out += ["#" * (b[2] + 1) + " " + b[1], ""]
            elif b[0] == "text":
                out += [b[1], ""]
            elif b[0] == "bullets":
                out += [f"- {i}" for i in b[1]] + [""]
            elif b[0] == "table":
                df = b[1].astype(str)
                out += ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
                out += ["| " + " | ".join(r) + " |" for r in df.values.tolist()] + [""]
            elif b[0] == "figure":
                out += [f"![{b[2]}]({b[1]})", "", f"*{b[2]}*" if b[2] else "", ""]
        return "\n".join(out)

    # ------------------------------------------------------------ pdf
    def pdf(self, path, base_dir):
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        regular, bold, plain = _fonts()
        ink, muted, rule, tint = (colors.HexColor(c) for c in ("#1b1b1a", "#5b5a56", "#d9d8d3", "#f4f3ef"))
        width = A4[0] - 4 * cm
        st = {
            "title": ParagraphStyle("t", fontName=bold, fontSize=18, leading=23, textColor=ink, spaceAfter=4),
            "sub": ParagraphStyle("s", fontName=regular, fontSize=10.5, leading=14, textColor=muted, spaceAfter=8),
            1: ParagraphStyle("h1", fontName=bold, fontSize=13.5, leading=17, textColor=ink, spaceBefore=12, spaceAfter=5),
            2: ParagraphStyle("h2", fontName=bold, fontSize=11, leading=14, textColor=ink, spaceBefore=8, spaceAfter=3),
            "p": ParagraphStyle("p", fontName=regular, fontSize=9.5, leading=13.5, textColor=ink, spaceAfter=5),
            "li": ParagraphStyle("li", fontName=regular, fontSize=9.5, leading=13.5, textColor=ink, leftIndent=12,
                                 bulletIndent=2, spaceAfter=2),
            "cap": ParagraphStyle("c", fontName=regular, fontSize=8.5, leading=11.5, textColor=muted, spaceAfter=10),
            "cell": ParagraphStyle("cell", fontName=regular, fontSize=7.8, leading=10, textColor=ink),
            "cellb": ParagraphStyle("cellb", fontName=bold, fontSize=7.8, leading=10, textColor=ink),
        }

        def P(text, style):
            text = _escape(text, plain)
            return Paragraph(_md_bold(text, bold), st[style])

        story = []
        for b in self.blocks:
            if b[0] == "title":
                story += [P(b[1], "title"), P(b[2], "sub")]
            elif b[0] == "heading":
                story.append(P(b[1], min(b[2], 2)))
            elif b[0] == "text":
                story.append(P(b[1], "p"))
            elif b[0] == "bullets":
                story += [Paragraph(_md_bold(_escape(i, plain), bold), st["li"], bulletText="•" if not plain else "-")
                          for i in b[1]]
            elif b[0] == "table":
                df = b[1].astype(str)
                rows = [[P(c, "cellb") for c in df.columns]] + [[P(v, "cell") for v in r] for r in df.values.tolist()]
                t = Table(rows, colWidths=[width / len(df.columns)] * len(df.columns), repeatRows=1)
                t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, ink),
                                       ("LINEBELOW", (0, -1), (-1, -1), 0.5, rule),
                                       ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
                                      + [("BACKGROUND", (0, r), (-1, r), tint) for r in range(2, len(rows), 2)]))
                story += [t, Spacer(1, 8)]
            elif b[0] == "figure":
                full = os.path.join(base_dir, b[1])
                if os.path.exists(full):
                    w, h = ImageReader(full).getSize()
                    fw, fh = width, width * h / w
                    if fh > 19 * cm:  # very tall charts: shrink to fit the page
                        fw, fh = 19 * cm * w / h, 19 * cm
                    story.append(Image(full, width=fw, height=fh))
                if b[2]:
                    story.append(P(b[2], "cap"))

        def footer(canvas, doc):
            canvas.saveState()
            canvas.setFont(regular, 7.5)
            canvas.setFillColor(muted)
            canvas.drawString(2 * cm, 1.1 * cm, "LocalPoint3D")
            canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))
            canvas.restoreState()

        SimpleDocTemplate(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                          bottomMargin=1.8 * cm, title=self.blocks[0][1]).build(story, onFirstPage=footer,
                                                                                onLaterPages=footer)

    def save(self, run_dir, name="report"):
        md_path = os.path.join(run_dir, f"{name}.md")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(self.markdown())
        pdf_path = os.path.join(run_dir, f"{name}.pdf")
        try:
            self.pdf(pdf_path, run_dir)
        except ImportError:
            print("PDF skipped: run  !pip install reportlab  and save again")
            pdf_path = None
        return md_path, pdf_path


def _fonts():
    """A font with arrows and accents (DejaVu) if the machine has one, else the built-in Helvetica.
    Returns (regular, bold, plain) - plain=True means special characters must be swapped for ASCII."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    for d, reg, bld in [(FONT_DIRS[0], "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
                        (FONT_DIRS[1], "LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf")]:
        if os.path.exists(d + reg) and os.path.exists(d + bld):
            pdfmetrics.registerFont(TTFont("R-Sans", d + reg))
            pdfmetrics.registerFont(TTFont("R-Sans-B", d + bld))
            return "R-Sans", "R-Sans-B", d != FONT_DIRS[0]
    return "Helvetica", "Helvetica-Bold", True


def _escape(text, plain):
    text = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if plain:
        for a, b in PLAIN.items():
            text = text.replace(a, b)
    return text


def _md_bold(text, bold_font):
    """**bold** in our text -> bold in the PDF."""
    parts = text.split("**")
    return "".join(f'<font name="{bold_font}">{p}</font>' if i % 2 else p for i, p in enumerate(parts))
