"""Build the mid-project report PDF from reports/figures/ (all numbers come from reports/*.md and results/).

usage: python reports/build/build_mid_report.py <repo> <out.pdf>
"""
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPO, OUT = sys.argv[1], sys.argv[2]
FIG = f"{REPO}/reports/figures"
D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("Sans", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-I", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-B")

INK, MUTED, RULE, TINT = colors.HexColor("#1b1b1a"), colors.HexColor("#5b5a56"), colors.HexColor("#d9d8d3"), colors.HexColor("#f4f3ef")
W = A4[0] - 4 * cm
st = {
    "title": ParagraphStyle("title", fontName="Sans-B", fontSize=20, leading=25, textColor=INK, spaceAfter=6),
    "sub": ParagraphStyle("sub", fontName="Sans", fontSize=11, leading=15, textColor=MUTED, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="Sans-B", fontSize=14, leading=18, textColor=INK, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Sans-B", fontSize=11, leading=14, textColor=INK, spaceBefore=9, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Sans", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=5),
    "li": ParagraphStyle("li", fontName="Sans", fontSize=9.5, leading=13.5, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=2.5),
    "cap": ParagraphStyle("cap", fontName="Sans-I", fontSize=8.5, leading=11.5, textColor=MUTED, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.3, leading=10.8, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=8.3, leading=10.8, textColor=INK),
    "box": ParagraphStyle("box", fontName="Sans", fontSize=9.2, leading=13, textColor=INK),
}


def P(t, s="p"): return Paragraph(t, st[s])
def H1(t): return P(t, "h1")
def H2(t): return P(t, "h2")
def bullets(items): return [Paragraph(i, st["li"], bulletText="•") for i in items]
def cap(t): return P(t, "cap")


def table(rows, widths, extra=()):
    data = [[Paragraph(c, st["cellb" if r == 0 else "cell"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
             ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 4),
             ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    style += [("BACKGROUND", (0, r), (-1, r), TINT) for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style + list(extra)))
    return t


def box(t):
    b = Table([[Paragraph(t, st["box"])]], colWidths=[W])
    b.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor("#2a78d6")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return b


def fig(name, width=W, max_height=None):
    w, h = ImageReader(f"{FIG}/{name}").getSize()
    height = width * h / w
    if max_height and height > max_height:
        width, height = max_height * w / h, max_height
    return Image(f"{FIG}/{name}", width=width, height=height)


def side_by_side(fig_a, cap_a, fig_b, cap_b):
    t = Table([[fig_a, fig_b], [cap_a, cap_b]], colWidths=[W / 2, W / 2])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return t


def contributes(t):
    return P(f"<b>What it contributes:</b> {t}", "box")


S = []
# ---------------------------------------------------------------- cover
S += [P("LocalPoint3D · Mid-Project Report", "title"),
      P("Track 3: ML Model and Dataset Development  |  Group 2", "sub"),
      P("Can 3D recognition models trained on Western datasets recognise Bangladeshi household objects "
        "captured with commercial smartphone scanning?", "sub"),
      Spacer(1, 6),
      table([["Member", "ID"],
             ["Fahim Muntasir", "2411371042"],
             ["Shahedul Islam Shikder", "2412925042"],
             ["Md Mosabbir Hossain Khan", "2414115042"]], [9 * cm, 5 * cm]),
      Spacer(1, 4),
      P("2 November 2026 · Code, data tables and detailed reports: "
        '<link href="https://github.com/Md-Mosabbir/localpoint3d" color="#2a78d6">github.com/Md-Mosabbir/localpoint3d</link>', "sub"),
      Spacer(1, 8),
      box("<b>In one sentence:</b> PointLLM, a 3D model trained on Western data, recognises our everyday scans "
          "(15 of 17 correct) but <b>none of our 14 Bangladeshi object scans (0 of 14)</b>. It still reads their shape, "
          "but always picks the nearest familiar object instead: rehal → bench, patil → bowl, bodna → vase/teapot. "
          "A CLIP-style model makes the same kind of mistake.")]

# ---------------------------------------------------------------- 1 problem statement
S += [H1("1. Problem statement (updated)"),
      P("<b>Problem.</b> Most 3D recognition models are trained on Western datasets: ModelNet40, ShapeNet and "
        "Objaverse. Objects common in Bangladesh, such as the bodna, rehal, kula and piri, or vehicles like the "
        "rickshaw and CNG, hardly appear in them. We test whether these models still recognise such objects when "
        "we capture them ourselves with <b>commercial smartphone scanning</b> (the Scaniverse app). We also build "
        "the small, cleaned dataset of local objects that this test needs."),
      H2("What changed once we started, and why"),
      table([["Initial plan", "Now", "Why"],
             ["Compare iPhone LiDAR and photogrammetry as two capture methods",
              "<b>Commercial smartphone scanning</b> (Scaniverse) as one capture method; the LiDAR/photo comparison is a side question",
              "Our first results showed the biggest problem was <b>clutter</b> in the scan (floor, walls), not the capture mode. "
              "The literature review on LiDAR vs photogrammetry is done; our own measurement is deferred."],
             ["Start with vehicles (rickshaw, thelagari, CNG)",
              "Start with <b>household objects</b> (bodna, piri, kula, muura, patil, rehal, matir cup, hurricane lamp); vehicles later",
              "Small objects are fast to scan and clean, so we could fix the method quickly before scanning big objects outdoors."],
             ["Use raw scans", "<b>Hand-clean every scan</b> (CloudCompare) + automatic speck removal",
              "Raw scans with floor and walls were described as “scenes” (“a ranch”, “a beach scene”). "
              "Cleaning raised correct answers from 24 to 33 of 51."],
             ["Ask open questions (“What is this?”)",
              "Add a <b>multiple-choice test</b> with “blind” inputs (a ball, random noise)",
              "Open answers are hard to score and vary run to run. Multiple choice plus blind inputs shows whether the "
              "model uses the <b>shape</b> or just guesses common words."],
             ["Colour scans", "<b>Grey</b> scans (shape only) for the main test",
              "We want to know whether the <b>shape</b> is recognised. On clean scans colour changed little (33 vs 32 of 51)."],
             ["PointLLM-V2 as the main model", "<b>PointLLM v1.2</b> + a CLIP-style model",
              "PointLLM-V2 weights are not released (checked 21 Oct). Our supervisor asked for a CLIP comparison too."]],
            [3.6 * cm, 5.4 * cm, W - 9 * cm])]

# ---------------------------------------------------------------- 2 done
S += [PageBreak(), H1("2. What we finished so far")]

S += [H2("2.1 Capture and dataset: 34 cleaned scans"),
      P("We scanned objects with Scaniverse on iPhones, exported point clouds (PLY), cleaned each scan by hand in "
        "CloudCompare and stored them on Google Drive with a fixed naming scheme (<i>Object_take_cleaned.ply</i>)."),
      table([["Set", "Objects", "Scans"],
             ["Everyday (reference)", "chair 8, cup 4, keyboard 3, table 2", "17"],
             ["Bangladeshi", "piri 4, bodna 2, kula 2, patil 2, matir cup 1, muura 1, rehal 1, hurricane lamp 1, wooden bench 3", "17"]],
            [3.6 * cm, W - 5.6 * cm, 2 * cm]),
      Spacer(1, 6),
      fig("2026-11-02_scan_gallery.png", max_height=15.5 * cm),
      cap("Figure 1. What the model sees: each scan reduced to 8192 points, centred, scaled and greyed. Three views per "
          "scan (top, front, side)."),
      contributes("the dataset itself (Track 3) and a capture-and-cleaning protocol that every later test relies on.")]

S += [H2("2.2 A reproducible test pipeline"),
      bullets([
          "<b>Pipeline:</b> phone scan → hand cleaning → Google Drive → Google Colab (free T4 GPU) → model → CSV → analysis. "
          "Three Colab notebooks, with shared Python code in the repository.",
          "<b>Pre-processing:</b> 8192 points, unit sphere, optional grey, automatic floor and speck removal. A single stray "
          "speck far away used to shrink the object to a dot.",
          "<b>Repeatable results:</b> PointLLM turned out to be random even with fixed settings (its point encoder starts "
          "from a random point). We fixed the random seed: two runs now give <b>identical</b> scores to the last decimal.",
      ]),
      contributes("every result in this report can be re-run by any member, and the same pipeline will test vehicles "
                  "and other models.")]

S += [H2("2.3 Everyday objects: baseline and the effect of cleaning"),
      fig("2026-10-23_raw_vs_cleaned.png", width=W * 0.78),
      cap("Figure 2. PointLLM, 17 everyday scans, 3 questions each (51 answers). Cleaning the scans helped most, "
          "especially in grey."),
      table([["Run", "Result"],
             ["18 Oct, raw scans", "9 of 17 scans named correctly"],
             ["21 Oct, raw, colour vs grey", "24 vs 13 of 51 answers correct: colour seemed essential"],
             ["23 Oct, <b>hand-cleaned</b>", "<b>33</b> (colour) and <b>32</b> (grey) of 51; 11 of 17 scans recognised. "
              "Once clean, colour hardly matters"]], [5 * cm, W - 5 * cm]),
      contributes("proof that the pipeline works on objects the model knows. Without this baseline, a failure on "
                  "local objects could just mean a bad scan.")]

S += [KeepTogether([H2("2.4 Bangladeshi objects, open questions (25 Oct)"),
      fig("2026-10-25_everyday_vs_local.png", width=W * 0.72),
      cap("Figure 3. Share of answers correct, same settings. Everyday 65 % / 58 %, Bangladeshi 13 % / 20 % "
          "(colour / grey).")]),
      P("Only the matir cup was recognised (“a cup and saucer”). The model often described the <b>shape</b> "
        "correctly but named the wrong object: rehal → “a bench with legs at 45 degrees” (18 of 18), kula → "
        "“a flat, oval-shaped board”. When unsure it invented familiar things (hurricane lamp → “a sculpture "
        "of a woman”, bodna → “a Pokémon character”)."),
      contributes("the first direct answer to the research question: the local objects are mostly not recognised.")]

S += [H2("2.5 Are the local objects “near out-of-distribution”?"),
      P("<i>Near-OOD</i> means a new kind of object that looks like something the model knows. We checked this in two ways."),
      side_by_side(
          fig("2026-10-31_training_words.png", width=W / 2 - 0.3 * cm),
          cap("Figure 4. Number of PointLLM's 658,577 training captions that contain each word. The local names never "
              "appear; their look-alikes are common."),
          fig("2026-10-31_ood_distance.png", width=W / 2 - 0.3 * cm),
          cap("Figure 5. How far each scan is from the everyday scans in PointLLM's own “shape fingerprint”. "
              "Local scans (blue, 0.66–0.86) are farther than almost all everyday scans (mostly 0.30–0.60).")),
      contributes("evidence that our objects are genuinely new to the model, both by name and by shape.")]

S += [H2("2.6 Multiple choice in grey: does it read the shape? (31 Oct)"),
      P("For each scan PointLLM chose between 4 answers: the correct one, a look-alike, a related object and an unrelated "
        "object (25 % by guessing). It does not write anything: we measure how likely it finds each answer. We also "
        "asked the same question about a featureless ball, random noise and a different object (“blind” inputs)."),
      fig("2026-10-31_mcq_accuracy_real_vs_blind.png"),
      cap("Figure 6. Everyday scans 15 of 17 correct, while the blind inputs are at chance. Bangladeshi scans "
          "0 of 14, in two runs with differently worded answers."),
      fig("2026-10-31_mcq_short_options_picks_local.png", width=W * 0.9),
      cap("Figure 7. What PointLLM picks for each local object: almost always a familiar look-alike or a related object, "
          "never the correct one."),
      table([["Object", "Correct answer", "PointLLM picks"],
             ["rehal", "book stand", "bench"], ["patil", "cooking pot", "bowl"], ["matir cup", "clay cup", "bowl (a near tie)"],
             ["bodna", "water jug", "vase / teapot"], ["piri", "low stool", "bench / box"],
             ["muura, kula, hurricane lamp", "cane stool, bamboo tray, lantern", "chair (the most common word in training)"]],
            [4.4 * cm, 4.6 * cm, W - 9 * cm]),
      Spacer(1, 6),
      bullets([
          "<b>It reads the shape:</b> the blind inputs get different picks than the real scans, and for matir cup "
          "and Patil_2 the shape itself favours the correct answer once common words are discounted.",
          "<b>But it only has Western categories:</b> it files each local object under the nearest one it knows.",
          "<b>A mistake we caught:</b> our first scoring divided by answer length and favoured long answers (even random "
          "noise looked “correct”). We found it with the blind inputs, fixed it and re-scored the saved data.",
      ]),
      contributes("the main finding so far, with a method that separates “doesn't see the shape” from "
                  "“sees it but has no name for it”.")]

S += [KeepTogether([H2("2.7 CLIP-style model (MetaCLIP), first results"),
      P("A CLIP-style model scores how well a 3D scan matches each text label. Shahedul tested 11 local scans with "
        "4–6 labels per object, and with the local name added in English letters and in Bangla script."),
      table([["Object", "Labels offered", "CLIP's top pick", "PointLLM's pick"],
             ["bodna", "kettle / teapot / cup / bottle", "kettle / teapot", "vase / teapot"],
             ["patil", "pot / bowl / saucer / basin", "bowl", "bowl"],
             ["kula", "basket / bowl / tray / plate", "plate", "chair"],
             ["muura", "basket / flower pot / bucket / stool", "stool (99 %)", "chair"],
             ["piri", "bench / chair / stool / table", "stool", "bench / box"],
             ["rehal", "book stand / chair / small table / bench", "small table", "bench"],
             ["wooden bench", "bench / table / stool / chair", "bench", "–"]],
            [2.4 * cm, 5.6 * cm, 3.4 * cm, W - 11.4 * cm])]),
      Spacer(1, 4),
      bullets([
          "Both models make the <b>same kind of mistake</b>: patil → bowl, bodna → teapot.",
          "CLIP is better on stool-like shapes and picks a good English <b>description</b> when it is offered "
          "(“folding book rest” for rehal, “floor stool” for piri).",
          "Local <b>names</b> (“bodna”, “rehal”, in English letters or Bangla script) score very low: "
          "the text side only knows English well.",
          "Not yet a fair comparison: the labels differ from PointLLM's options. Fixing that is week 1 of the plan.",
      ]),
      contributes("a second model family, showing the problem isn't specific to PointLLM.")]

S += [H2("2.8 Literature: LiDAR vs photogrammetry on phones"),
      P("Published tests say the two give similar overall shapes but differ on thin, shiny or small parts; iPhone LiDAR "
        "is about ±1 cm on objects above 10 cm (Luetzenburg et al. 2021). We use Scaniverse's mesh export throughout."),
      contributes("background for choosing a capture method for vehicles.")]

# ---------------------------------------------------------------- 3 plan
S += [PageBreak(), H1("3. What is left: week by week"),
      table([["Week", "Dates", "Work", "Output"],
             ["1", "2–8 Nov", "Run CLIP with exactly PointLLM's answer options on all 14 local + 17 everyday scans. "
              "A teammate re-checks the scoring. Add answer options for the wooden bench.", "Fair PointLLM vs CLIP table"],
             ["2", "9–15 Nov", "Scan more local objects and more takes per object (goal: ≥ 3 scans each). "
              "Start local vehicles (rickshaw, thelagari, CNG) if access allows.", "Bigger cleaned dataset"],
             ["3", "16–22 Nov", "Clean the new scans; run the OOD check and multiple choice on them with both models. "
              "Rotation test (which way is “up” for the models).", "Results on new objects"],
             ["4", "23–29 Nov", "<b>Teaching pilot:</b> can a few examples per object teach the model? (nearest-neighbour or "
              "a small classifier on the models' shape fingerprints, tested on held-out scans)", "Accuracy before vs after"],
             ["5", "30 Nov–6 Dec", "Package the dataset (labels, metadata, capture protocol, README). Final analysis "
              "and figures.", "Dataset release + final figures"],
             ["6", "7–13 Dec", "Final report and presentation; rehearse; every member can explain every part.", "Final presentation"]],
            [1.2 * cm, 2.2 * cm, W - 7.4 * cm, 4 * cm])]

# ---------------------------------------------------------------- 4 final features
S += [H1("4. What the final report will include"),
      bullets([
          "<b>A small labelled dataset</b> of cleaned smartphone scans of Bangladeshi objects, plus the everyday reference "
          "set, with a capture and cleaning protocol others can follow.",
          "<b>A benchmark</b> for “near-OOD” objects: training-word check, shape-distance check, and multiple "
          "choice with blind inputs.",
          "<b>A comparison of two model families</b> (PointLLM vs CLIP-style) on the same questions.",
          "<b>A failure analysis:</b> which familiar object each local object is mistaken for, and why.",
          "<b>A teaching pilot:</b> whether a few examples are enough for a model to learn local objects.",
          "<b>Vehicles</b> (rickshaw, thelagari, CNG), if scanning access allows.",
          "<b>Reproducible notebooks</b> on free Google Colab, with all result tables in the repository.",
      ])]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.1 * cm, "LocalPoint3D · Group 2 · Mid-project report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, str(doc.page))
    canvas.restoreState()


S = [f for item in S for f in (item if isinstance(item, list) else [item])]  # bullets() returns a list
doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.8 * cm, title="LocalPoint3D mid-project report", author="Group 2")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
