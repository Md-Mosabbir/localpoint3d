"""Build the progress-report PDF from the figures in <figdir> and the results CSV."""
import csv, json, sys
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

REPO, FIG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("Sans", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("Sans-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Sans-I", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("Mono", D + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-B", italic="Sans-I", boldItalic="Sans-B")

INK, MUTED, RULE, TINT = colors.HexColor("#1b1b1a"), colors.HexColor("#5b5a56"), colors.HexColor("#d9d8d3"), colors.HexColor("#f4f3ef")
GOOD, BAD, ACCENT = colors.HexColor("#1a7f37"), colors.HexColor("#b42318"), colors.HexColor("#2a78d6")
WARN_BG, WARN_LINE = colors.HexColor("#fff4e5"), colors.HexColor("#eda100")

W = A4[0] - 4 * cm
st = {
    "title": ParagraphStyle("title", fontName="Sans-B", fontSize=19, leading=24, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="Sans", fontSize=11, leading=15, textColor=MUTED, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="Sans-B", fontSize=14, leading=18, textColor=INK, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="Sans-B", fontSize=11, leading=14, textColor=INK, spaceBefore=9, spaceAfter=3),
    "p": ParagraphStyle("p", fontName="Sans", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "li": ParagraphStyle("li", fontName="Sans", fontSize=9.5, leading=13.5, textColor=INK, leftIndent=12, bulletIndent=2, spaceAfter=2.5),
    "cap": ParagraphStyle("cap", fontName="Sans-I", fontSize=8.5, leading=11.5, textColor=MUTED, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="Sans", fontSize=8.3, leading=10.8, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Sans-B", fontSize=8.3, leading=10.8, textColor=INK),
    "quote": ParagraphStyle("quote", fontName="Sans-I", fontSize=8, leading=10.6, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7.8, leading=10, textColor=INK),
    "box": ParagraphStyle("box", fontName="Sans", fontSize=9.2, leading=13, textColor=INK),
}


def P(t, s="p"): return Paragraph(t, st[s])
def H1(t): return P(t, "h1")
def H2(t): return P(t, "h2")
def bullets(items): return [Paragraph(i, st["li"], bulletText="•") for i in items]
def cap(t): return P(t, "cap")


def table(rows, widths, header=True, zebra=True, extra=()):
    data = [[c if not isinstance(c, str) else Paragraph(c, st["cellb" if header and r == 0 else "cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK),
             ("LINEBELOW", (0, -1), (-1, -1), 0.5, RULE), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5), ("LEFTPADDING", (0, 0), (-1, -1), 4),
             ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if zebra:
        style += [("BACKGROUND", (0, r), (-1, r), TINT) for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style + list(extra)))
    return t


def box(t, bg=TINT, line=RULE):
    b = Table([[Paragraph(t, st["box"])]], colWidths=[W])
    b.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, line),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return b


def fig(name, width, aspect):
    return Image(f"{FIG}/{name}", width=width, height=width * aspect)


def code(text):
    b = Table([[Preformatted(text, st["code"])]], colWidths=[W])
    b.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT), ("BOX", (0, 0), (-1, -1), 0.5, RULE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return b


def ok(v): return {"yes": '<font color="#1a7f37"><b>✓</b></font>', "no": '<font color="#b42318"><b>✗</b></font>',
                   "partly": '<font color="#5b5a56"><b>~</b></font>'}[v]


answers = list(csv.DictReader(open(f"{REPO}/results/pointllm_2026-10-18_baseline.csv", encoding="utf-8")))
def answer(scan, prompt):
    return [r for r in answers if r["file"] == scan + ".ply" and r["prompt"] == prompt][0]


stats = {s["name"]: s for s in json.load(open(f"{FIG}/stats.json"))}
I, C, CAP = "What is this?", "This is an object of ", "Caption this 3D model in detail."

S = []
# ---------------------------------------------------------------- title
S += [P("localpoint3d · Progress report 1", "title"),
      P("Can pretrained 3D models recognise objects we scan ourselves with an iPhone?", "sub"),
      P("Shahedul Islam Shikder · Fahim Muntasir · Md Mosabbir  —  Junior Design Project  —  18 October 2026", "sub"),
      P('Code and data: <link href="https://github.com/Md-Mosabbir/localpoint3d" color="#2a78d6">github.com/Md-Mosabbir/localpoint3d</link>', "sub"),
      Spacer(1, 8)]

# ---------------------------------------------------------------- 1 summary
S += [H1("1. Summary")]
S += bullets([
    "<b>The pipeline works end to end.</b> 17 iPhone scans (Scaniverse app, raw PLY point clouds) → Google Drive → "
    "Google Colab (free T4 GPU) → PointLLM-7B. 51 answers recorded.",
    "<b>PointLLM named 9 of 17 scans correctly (53 %).</b> Cups 3/4, chairs 5/8, keyboards 1/3, tables 0/2.",
    "<b>The failures look like a capture problem, not a model problem.</b> When the scan is less than 4× the size of "
    "the object, 9 of 12 scans were named correctly. When it is 4× or more (lots of room around the object), "
    "<b>0 of 5</b> were. Every scan also contains the floor or desk (22–78 % of all points).",
    "<b>A first CLIP-style zero-shot test</b> on one cup scan ranked <i>“a 3D scan of a chair”</i> highest (48 %). "
    "Changing only the wording of the cup label (“three-dimensional photo” vs “photo”) moved its score from 37 % to 3 %. "
    "The label wording mattered more than the object.",
    "<b>LiDAR vs photogrammetry is not measured yet.</b> What we read (papers + Scaniverse’s own notes) says the two "
    "should agree on overall shape but differ on fine, thin and shiny parts. The measuring notebook is ready and "
    "has been checked on synthetic data.",
    "<b>Side finding:</b> Scaniverse PLY point-cloud exports are <b>Z-up</b> (the floor lies flat in the x–y plane), "
    "not Y-up as we had assumed.",
])

# ---------------------------------------------------------------- 2 question
S += [H1("2. Research question"),
      P("Most 3D recognition models are trained on Western datasets such as <b>ModelNet40</b>, <b>ShapeNet</b> and "
        "<b>Objaverse</b>. We want to know whether they can recognise <b>Bangladeshi objects</b> such as a rickshaw, "
        "a thelagari (push cart) or a CNG auto-rickshaw, scanned by us with a phone."),
      P("Before testing local objects we need a <b>baseline</b>: how well does the same pipeline work on everyday "
        "objects the models certainly know (chairs, cups, keyboards, tables)? Without it we could not tell "
        "“the model doesn’t know a rickshaw” apart from “our scan is messy”. This report is that baseline.")]

# ---------------------------------------------------------------- 3 scans
S += [KeepTogether([H1("3. Our scans"),
      P("All 17 scans were captured with an iPhone using the <b>Scaniverse</b> app and exported as <b>PLY</b> files. "
        "They are stored on Google Drive (<font face='Mono'>My Drive/Models</font>), not in the repository, because "
        "they total 76 MB. Below is the start of one file exactly as Scaniverse wrote it:"),
      code("format binary_little_endian 1.0\ncomment Created with Scaniverse - https://scaniverse.com\n"
           "element vertex 285103\nproperty float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header"),
      cap("Header of Chair_1.ply, unchanged. Each point has a position (x, y, z in metres) and a colour (0–255). "
          "There are no triangles, so the file is a point cloud, not a mesh. The header does not say whether "
          "LiDAR or photo mode was used.")])]

rows = [["Scan", "Points", "Size x × y × z (m)", "Scan size vs real object", "Points on biggest flat surface", "PointLLM"]]
for name, s in stats.items():
    rows.append([name, f"{s['points']:,}", f"{s['w']:.2f} × {s['h']:.2f} × {s['d']:.2f}",
                 f"{s['ratio']:.1f}×", f"{100 * s['plane_frac']:.0f} %",
                 '<font color="#1a7f37">correct</font>' if s["ok"] else '<font color="#b42318">wrong</font>'])
S += [table(rows, [2.3 * cm, 1.8 * cm, 3.6 * cm, 3.2 * cm, 3.4 * cm, 2.7 * cm]),
      cap("Table 1. The 17 scans. <i>Scan size vs real object</i> = longest side of the scan ÷ a rough real size "
          "(chair 1 m, cup 12 cm, keyboard 45 cm, table 1 m). <i>Biggest flat surface</i> = share of points within "
          "1 cm of one plane (floor or desk), found with RANSAC."),
      PageBreak(),
      fig("gallery.png", W, 13.2 / 12),
      cap("Figure 1. Every scan, drawn from the raw PLY (30,000 random points, Z up). Green = PointLLM named it "
          "correctly, red = wrong. The failures are mostly scans where the object is small inside a large scene "
          "(Chair_3, Chair_4, Cup_4, Table_1) or sits on a big desk (Keyboard_1, Keyboard_3)."),
      PageBreak()]

# ---------------------------------------------------------------- 4 pipeline
S += [H1("4. From raw PLY to model input"),
      P("PointLLM accepts exactly <b>8,192 points</b>, each <font face='Mono'>[x, y, z, r, g, b]</font>, with positions "
        "inside a unit sphere and colours between 0 and 1. Our code (<font face='Mono'>src/pc_io.py</font>, "
        "<font face='Mono'>src/pointllm_helpers.py</font>) turns a raw scan into that format:")]
rows = [["Step", "What happens", "Why"],
        ["1. Load", "Read the PLY with the <i>plyfile</i> library into two numpy arrays: positions (N × 3) and colours (N × 3, divided by 255).",
         "Scaniverse stores colour as 0–255; the model wants 0–1."],
        ["2. Reduce", "Keep 40,000 random points.", "Speed. The scans have 115k–649k points."],
        ["3. Farthest point sampling", "Pick 8,192 points one by one, each time the point farthest from those already picked.",
         "Spreads points evenly over the shape instead of crowding dense areas."],
        ["4. Normalise", "Move the centre to (0, 0, 0) and scale so the farthest point is at distance 1.",
         "Same convention as PointLLM’s training data and ModelNet40."],
        ["5. Ask", "Send the 8,192 points + a question to PointLLM running on a Colab T4 GPU.",
         "PointLLM needs its own Python 3.10 environment, so it runs as a separate background process."]]
S += [table(rows, [3.2 * cm, 8.0 * cm, 5.8 * cm]),
      cap("Table 2. Preprocessing. Nothing removes the floor or rotates the scan in this run; whatever Scaniverse "
          "exported is what the model sees."),
      fig("pipeline.png", W, 0.5),
      cap("Figure 2. Raw scan (left) and the 8,192 points the model receives (right). Chair_5 (top) keeps its shape. "
          "In Table_1 (bottom) the table is a small dark shape in a 13 m scan; after normalisation it is a speck, "
          "and most of the 8,192 points describe the surroundings.")]

# ---------------------------------------------------------------- 5 pointllm
S += [PageBreak(), H1("5. Experiment 1: PointLLM recognition"),
      P("<b>PointLLM</b> is a large language model that reads point clouds and answers questions about them "
        "(Xu et al., ECCV 2024). It was trained on about 660,000 Objaverse objects.")]
rows = [["Setting", "Value"],
        ["Model", "RunsenXu/PointLLM_7B_v1.2 (latest released; PointLLM-V2 weights are not public)"],
        ["Compute", "Google Colab free tier, NVIDIA T4, float16"],
        ["Prompts (from the paper)", "I: “What is this?”   C: “This is an object of ”   Cap: “Caption this 3D model in detail.”"],
        ["Decoding", "Greedy, so the same input always gives the same answer"],
        ["Scoring", "An answer is correct if it names the right kind of object (“mug” counts for cup). "
                    "A scan is correct if at least 2 of its 3 answers are. Scored by reading, one rater."]]
S += [table(rows, [4.0 * cm, 13.0 * cm]), Spacer(1, 6), H2("Results"),
      fig("by_object.png", 10.5 * cm, 2.4 / 6.4),
      cap("Figure 3. Scans named correctly, per object. Overall 9 / 17 scans (53 %) and 26 / 51 answers (51 %). "
          "Per prompt: I 9/17, C 7/17, Cap 10/17.")]

per_scan = [
    ("Chair_1", "a black chair, cartoon style"), ("Chair_2", "a treadmill (all three prompts)"),
    ("Chair_3", "“a location … buildings, roads”; “a cleared area in a forest”"),
    ("Chair_4", "“a rare species of bird”; “a toy airplane”"), ("Chair_5", "a chair with a red seat"),
    ("Chair_6", "a blue chair “on a textured tiled floor”"), ("Chair_7", "a chair, cartoon style"),
    ("Chair_8", "chair; C: “a dilapidated, ruined building”"), ("Cup_1", "a coffee cup"),
    ("Cup_2", "cup / a traditional mug"), ("Cup_3", "a ceramic mug"), ("Cup_4", "a cartoon fish (all three)"),
    ("Keyboard_1", "“a rare, old painting”; “a building”"), ("Keyboard_2", "a QWERTY keyboard; C: “a carpet”"),
    ("Keyboard_3", "a house wall and yard; “a rectangular flat board”"),
    ("Table_1", "“a ranch … and an outhouse”; “a motorbike”"), ("Table_2", "“a bar stool tipped onto its side”; a chair")]
rows = [["Scan", "I", "C", "Cap", "What PointLLM said (short)"]]
for name, said in per_scan:
    rows.append([name] + [ok(answer(name, p)["correct"]) for p in (I, C, CAP)] + [said])
S += [KeepTogether([table(rows, [2.4 * cm, 0.9 * cm, 0.9 * cm, 1.0 * cm, 11.8 * cm]),
      cap("Table 3. Every scan and prompt. ✓ correct, ✗ wrong, ~ object mentioned but as part of a scene. "
          "Full answers are in results/pointllm_2026-10-18_baseline.csv.")])]

S += [H2("Example answers, exactly as PointLLM wrote them")]
examples = [("Chair_1", I), ("Cup_1", I), ("Keyboard_2", I), ("Chair_2", I),
            ("Keyboard_1", I), ("Table_2", I), ("Table_1", I), ("Chair_3", C)]
rows = [["Scan", "Prompt", "", "Answer (unedited)"]]
for name, p in examples:
    r = answer(name, p)
    rows.append([name, f"“{p.strip()}”", ok(r["correct"]), Paragraph(r["answer"], st["quote"])])
S += [table(rows, [2.2 * cm, 2.6 * cm, 0.7 * cm, 11.5 * cm]),
      cap("Table 4. Eight answers copied unchanged from the results CSV: three correct, five wrong. Note the "
          "recurring words “cartoon”, “whimsical” and “rare”: they come from the style of PointLLM’s training "
          "captions, not from our scans.")]

# ---------------------------------------------------------------- 6 reasons
S += [PageBreak(), H1("6. Why did it fail? Presumed reasons"),
      P("These are our best explanations so far. The first two are now backed by measurements on the raw scans; "
        "the others are still inferred from the answers.")]
S += [H2("6.1 Too much scene around the object (strongest evidence)"),
      fig("clutter.png", 14.5 * cm, 3.3 / 6.4),
      cap("Figure 4. Each scan’s size relative to the real object (x) and how much of it is one flat surface (y). "
          "No scan at 4× or more was recognised."),
      *bullets([
          "Chair_4 is 11 m wide, Table_1 is 13 m, Cup_4 is 1 m for a cup of about 12 cm.",
          "Normalisation squeezes the <i>whole</i> scan into the unit sphere, so the object becomes tiny, and farthest "
          "point sampling spends most of the 8,192 points on the surroundings.",
          "The wrong answers describe <b>scenes</b>: “a ranch”, “a cleared area in a forest”, “a ruined building”, "
          "“a house wall and yard”."]),
      H2("6.2 Floor or desk in every scan"),
      *bullets([
          "In every scan, 22–78 % of the points lie on one flat surface (Table 1).",
          "On its own this does not decide the result: Cup_1 is 65 % floor and was still recognised.",
          "It hurts <b>flat objects</b> most. On a desk, a keyboard is a thin layer on a bigger flat layer "
          "(60–76 % flat), and the answers were flat things: “painting”, “carpet”, “flat board”."]),
      H2("6.3 Some scans are simply bad"),
      P("The team member who captured them confirms several are poor captures: sparse, partial, or mostly "
        "background (Chair_3, Chair_4, Table_1 in Figure 1). Which scans are “bad” has not been recorded "
        "systematically yet."),
      H2("6.4 Fine detail is lost"),
      P("Keys are a few millimetres tall, below what phone scanning reliably captures (objects under ~5 cm are "
        "not reliably captured by iPhone LiDAR, Luetzenburg et al. 2021), and 8,192 points leave very few per key. "
        "So the keyboards look like planks. We think PointLLM does know keyboards (Keyboard_2 was named, "
        "“QWERTY”), so this is a detail problem rather than a missing concept. <b>This matters for rickshaws</b>: "
        "spokes and hood frames are thin parts too."),
      H2("6.5 Orientation"),
      P("The scans are Z-up and were passed to PointLLM unrotated. We have not verified which axis PointLLM treats "
        "as up. Table_2 is a clean, upright scan, yet PointLLM said “a bar stool that is tipped onto its side”, "
        "which fits an axis mismatch. But most chairs were recognised in the same orientation, so this is at most "
        "part of the story. The notebook can already test rotations."),
      H2("6.6 The prompt changes the answer"),
      P("For the same scan, different prompts sometimes give different objects (Chair_8, Keyboard_2, Table_1). "
        "The sentence-completion prompt C did worst (7/17). Agreement between the three prompts is a useful "
        "confidence signal.")]

# ---------------------------------------------------------------- 7 clip
S += [H1("7. Experiment 2: CLIP zero-shot (first single test)"),
      P("A CLIP-style model scores how well a 3D input matches each text label in a list (“zero-shot” means it "
        "was never trained on our labels). We gave it one of our cup scans with five candidate labels. "
        "Output, exactly as printed:"),
      KeepTogether([code("--- Zero-Shot Classification Results ---\na 3D scan of a chair: 47.99%\n"
           "a three-dimensional photo of a coffee Cup: 37.09%\na 3D scan of a bottle: 9.26%\n"
           "a photo of a coffee Cup: 3.01%\na 3D scan of a building: 2.65%"),
      Spacer(1, 6), fig("clip.png", 14 * cm, 2.6 / 6.4),
      cap("Figure 5. The same output as a chart (redrawn from the numbers above). Blue = labels naming the correct object.")]),
      H2("What it shows"),
      *bullets([
          "The top answer is wrong: <b>“a 3D scan of a chair”</b> (48 %) for a cup.",
          "The cup appears twice, with different wording: “a three-dimensional photo of a coffee Cup” got 37 %, "
          "“a photo of a coffee Cup” only 3 %. <b>Same object, 12× different score, only because of the wording.</b>",
          "Explanation from our supervisor: during training the text for every 3D object began with "
          "<b>“a 3D scan of …”</b>, so the model learned to prefer labels that use that phrase. Labels that "
          "say “photo” (2D) lose confidence regardless of the object.",
          "So this list compared <i>wordings</i> as much as <i>objects</i>; the result cannot be read as "
          "“the model thinks the cup is a chair”. Clutter in the cup scan (section 6.1) may also play a part."]),
      H2("How to run it fairly"),
      *bullets([
          "Use <b>one template for every label</b>, e.g. “a 3D scan of a {chair / cup / bottle / keyboard / table …}”.",
          "Better: average the scores over several templates (prompt ensembling, as in the original CLIP paper).",
          "Run all 17 scans, so the result can be compared with PointLLM scan by scan."]),
      box("<b>Not recorded yet:</b> the exact CLIP model used, which cup scan, and whether the scan was given as a "
          "point cloud or as rendered images. These need to go in the journal before the next run.",
          WARN_BG, WARN_LINE)]

# ---------------------------------------------------------------- 8 lidar vs photo
S += [KeepTogether([H1("8. LiDAR vs photogrammetry"),
      box("<b>Not tested yet.</b> This section summarises what we read in papers and on Scaniverse’s own pages. "
          "Our own measurements come next.", WARN_BG, WARN_LINE)]), Spacer(1, 6),
      P("<b>Why it matters:</b> if both capture methods give about the same point cloud, we can use either one, "
        "and teammates without a LiDAR iPhone can still scan. If they differ, we need to know where."),
      H2("Scaniverse’s mesh modes")]
rows = [["Mode", "Method", "Made for", "Stated detail"],
        ["Speed", "LiDAR", "quick results", "~10 mm"],
        ["Area", "LiDAR", "rooms, buildings, outdoors", "~5 mm"],
        ["Detail", "Photogrammetry", "objects, people, small textured scenes", "not stated"]]
S += [table(rows, [2.5 * cm, 3.3 * cm, 7.5 * cm, 3.7 * cm]),
      cap("Table 5. From Scaniverse’s announcements. Scaniverse itself says LiDAR modes work better on plain or "
          "shiny surfaces and Detail mode on textured objects.")]
rows = [["Part of the object", "Expected result"],
        ["Big surfaces (seat, cart bed, CNG body)", "About the same in both, within a few cm after alignment"],
        ["Thin parts (legs, spokes, rods, ropes)", "Both lose some; photogrammetry usually keeps more"],
        ["Shiny metal (chrome)", "Both struggle; LiDAR distorts, photogrammetry finds no stable texture"],
        ["Plain, untextured paint", "LiDAR fine, photogrammetry struggles"]]
S += [table(rows, [6.0 * cm, 11.0 * cm]), cap("Table 6. What the literature leads us to expect.")]
S += [H2("What the sources say")]
S += bullets([
    "iPhone LiDAR: depth map only 256 × 192 pixels, usable to ~5 m (Apple developer forum).",
    "iPhone 12 Pro LiDAR: about ±1 cm on objects larger than 10 cm; objects under ~5 cm not reliably captured "
    "(Luetzenburg et al. 2021).",
    "iPad LiDAR vs photogrammetry on small complex objects: up to 50 % of LiDAR points did not match; "
    "photogrammetry clearly better (Sensors 2022).",
    "The only study naming Scaniverse (iPhone 13 Pro, whole buildings, older app version) found ~55 cm error vs "
    "≤ 10 cm for other apps. Room-scale, so it says little about objects.",
    "<b>No study compares Scaniverse Detail vs Area mode on the same object.</b> Our measurement would be new."])
S += [H2("How we will measure it"),
      *bullets([
          "Scan one object twice in Detail mode and twice in Area mode.",
          "Notebook 02 removes the floor, normalises, aligns the scans (12 start angles + ICP) and computes "
          "Chamfer distance, Hausdorff distance and F-score at 1 / 2 / 5 % of object size, plus a heatmap.",
          "Key comparison: the <b>Detail vs Area</b> gap against the <b>Detail vs Detail</b> gap. If they are similar, "
          "the methods are interchangeable.",
          "Checked on a synthetic chair (second copy rotated 70°, one leg removed): alignment recovered, "
          "F-score@2 % = 0.96, and the missing leg showed up clearly (Hausdorff ≈ 40 cm)."])]

# ---------------------------------------------------------------- 9-11
S += [H1("9. Limitations"),
      *bullets([
          "Small sample: 2–8 scans per object type; one scan changes a percentage a lot.",
          "Answers scored by one rater, by reading. A teammate should re-score Table 3.",
          "“Scan size vs real object” uses rough guessed real sizes; the flat-surface share uses a simple RANSAC.",
          "The capture mode (LiDAR or photo) of the 17 scans is unknown.",
          "The CLIP result is a single test with key details not recorded."]),
      H1("10. Conclusions"),
      *bullets([
          "The full pipeline (phone → Drive → Colab → PointLLM → CSV) works, on free compute.",
          "PointLLM recognises <b>tightly captured</b> scans of common objects well (9 of 12 below 4× size). "
          "Most failures trace back to <b>how we captured</b>, not to the model.",
          "Before scanning rickshaws we must fix the capture: crop the object, remove the floor, check orientation. "
          "Otherwise a failure on a rickshaw would be impossible to interpret.",
          "Zero-shot CLIP-style results depend strongly on label wording; every label must use the same template.",
          "LiDAR vs photogrammetry: expected to agree on overall shape but not on fine detail. To be measured."]),
      H1("11. Next steps")]
rows = [["#", "Step", "Purpose"],
        ["1", "Crop scans in Scaniverse before export, and add automatic floor removal to preprocessing", "Test reasons 6.1 and 6.2: re-run all 17"],
        ["2", "Re-run failures with rotations (+90° / −90° about X)", "Test reason 6.5"],
        ["3", "Mark each scan good / bad in a small CSV", "Accuracy on good scans only"],
        ["4", "Re-run CLIP on all 17 scans with one fixed template; record model and input type", "Fair comparison with PointLLM"],
        ["5", "Scan one chair 2× Detail + 2× Area; run notebook 02", "Answer LiDAR vs photogrammetry"],
        ["6", "Put the capture mode in file names (Chair_lidar_1.ply)", "Needed for step 5 and later"],
        ["7", "Scan local objects (rickshaw, thelagari, CNG)", "The actual research question"]]
S += [table(rows, [0.7 * cm, 10.3 * cm, 6.0 * cm])]

# ---------------------------------------------------------------- refs
S += [H1("References")]
refs = [
    "Xu, R. et al. PointLLM: Empowering Large Language Models to Understand Point Clouds. ECCV 2024. https://arxiv.org/abs/2308.16911",
    "Radford, A. et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). 2021. https://arxiv.org/abs/2103.00020",
    "Luetzenburg, G. et al. Evaluation of the Apple iPhone 12 Pro LiDAR for an application in geosciences. Scientific Reports 2021. https://pmc.ncbi.nlm.nih.gov/articles/8593014",
    "iPad LiDAR vs photogrammetry on small complex objects. Sensors 2022. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9657006/",
    "Use of Smartphone Lidar Technology for Low-Cost 3D Building Documentation with iPhone 13 Pro. Digital 2023. https://www.mdpi.com/2673-7418/3/4/30",
    "Scaniverse: Detail mode (photogrammetry). https://scaniverse.com/news/scaniverse-detail-mode-photogrammetry",
    "Scaniverse 2.0 adds support for non-LiDAR devices. https://scaniverse.com/news/scaniverse-2-0-adds-support-for-non-lidar-devices",
    "Apple Developer Forums: LiDAR depth resolution. https://developer.apple.com/forums/thread/695771",
]
S += [Paragraph(r, st["li"], bulletText=f"[{i + 1}]") for i, r in enumerate(refs)]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 7.5); canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "localpoint3d · Progress report 1 · 18 Oct 2026")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=1.9 * cm, title="localpoint3d: Progress report 1",
                        author="Shahedul Islam Shikder, Fahim Muntasir, Md Mosabbir")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
