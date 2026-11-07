"""Presentation-ready charts for the multiple-choice / near-OOD tests (notebooks 03 and 04).

Every function takes the tables the notebook already made and returns a matplotlib figure.
`save(fig, path)` writes it as a 200-dpi PNG for slides and reports.
Colours: one fixed colour per answer type, the same in every chart, checked for
colour-blind readers. Two of them are light, so bars always carry a text label too.
"""
import numpy as np

INK = "#0b0b0b"         # titles, labels
INK_2 = "#52514e"       # secondary text, axes
GRID = "#e4e3df"
SURFACE = "#fcfcfb"     # background
NEUTRAL = "#b9b8b2"     # "everything else" (blind inputs, reference scans)
ROLE_COLOURS = {"correct": "#2a78d6", "look_alike": "#eb6834", "same_family": "#1baf7a", "unrelated": "#eda100"}
HIGHLIGHT = ROLE_COLOURS["correct"]
NEGATIVE = ROLE_COLOURS["look_alike"]


def _plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    })
    return plt


def _titles(fig, title, subtitle=None):
    """Title + grey subtitle at the top left, a fixed distance in inches whatever the figure height."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.3 / h, title, ha="left", va="center", fontsize=15, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.62 / h, subtitle, ha="left", va="center", fontsize=10.5, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 1 - (1.0 if subtitle else 0.6) / h))


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def caption_counts(word_counts, title="How often did PointLLM read each word in training?"):
    """Horizontal bars: number of training captions containing each word. Zeros are labelled."""
    plt = _plt()
    wc = word_counts.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(wc) + 1.6))
    colours = [NEGATIVE if c == 0 else HIGHLIGHT for c in wc.captions]
    ax.barh(wc.word, wc.captions, color=colours, height=0.65)
    top = max(wc.captions.max(), 1)
    for y, c in enumerate(wc.captions):
        ax.text(c + top * 0.01, y, "0  (never seen)" if c == 0 else f"{c:,}", va="center", fontsize=10,
                color=INK_2)
    ax.set_xlim(0, top * 1.15)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("training captions containing the word (of 658,577)")
    _titles(fig, title, "Our local names never appear, but their look-alikes do. That is what near-OOD means.")
    return fig


def ood_distances(ood, threshold):
    """One bar per scan: distance to its nearest everyday scans. Local scans in colour, the OOD line dashed."""
    plt = _plt()
    d = ood.sort_values("knn_distance")
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(d) + 1.8))
    labels = [f.replace("_cleaned", "").rsplit(".", 1)[0] for f in d.file]
    colours = [HIGHLIGHT if s == "local" else NEUTRAL for s in d.set]
    ax.barh(labels, d.knn_distance, color=colours, height=0.7)
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.2)
    ax.set_xlabel("distance to the nearest everyday scans (PointLLM fingerprint, cosine)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=HIGHLIGHT, label="local object"), Patch(color=NEUTRAL, label="everyday scan"),
                       Line2D([], [], color=INK, linestyle="--", label="OOD line (95 % of everyday scans are closer)")],
              loc="lower right")
    _titles(fig, "How unfamiliar is each scan's shape?",
            "Everyday scans (grey) set the normal range; local objects (blue) past the line are new shapes.")
    return fig


def accuracy_by_input(summary, chance):
    """Small multiples, one panel per scan set: % correct for the real scans vs each blind input."""
    plt = _plt()
    sets = list(dict.fromkeys(summary["set"]))
    fig, axes = plt.subplots(1, len(sets), figsize=(5.2 * len(sets), 4.6), sharey=True, squeeze=False)
    for ax, set_name in zip(axes[0], sets):
        s = summary[(summary["set"] == set_name) & (summary["level"] == "scan")]
        colours = [HIGHLIGHT if i == "real" else NEUTRAL for i in s.input]
        ax.bar(s.input, 100 * s.accuracy, color=colours, width=0.62)
        for x, (k, n, acc) in enumerate(zip(s.correct, s.n, s.accuracy)):
            ax.text(x, 100 * acc + 2, f"{k}/{n}\n{acc:.0%}", ha="center", va="bottom", fontsize=10, color=INK)
        ax.axhline(100 * chance, color=INK, linestyle="--", linewidth=1.1)
        ax.set_title(f"{set_name} objects", loc="left", fontsize=12)
        ax.set_ylim(0, 118)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("% picked the correct answer")
    _titles(fig, "Correct answers: real scans vs blind inputs",
            f"Blue = the real scan. Grey = the same question about a ball, random noise or a different object. "
            f"Dashed line = guessing ({chance:.0%}).")
    return fig


def picks_by_object(best, roles, title="What does PointLLM pick for each object?"):
    """100 % stacked bars per object: share of (scan x seed) picks per answer type."""
    plt = _plt()
    counts = best.groupby(["object", "role"]).size().unstack(fill_value=0).reindex(columns=roles, fill_value=0)
    share = counts.div(counts.sum(axis=1), axis=0)
    share = share.sort_values("correct")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(share) + 2))
    left = np.zeros(len(share))
    for role in roles:
        vals = share[role].values
        ax.barh(share.index, vals, left=left, color=ROLE_COLOURS.get(role, NEUTRAL), height=0.68,
                edgecolor=SURFACE, linewidth=2, label=role.replace("_", " "))
        for y, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.1:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=9.5, color=INK)
        left += vals
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(ncol=len(roles), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10)
    _titles(fig, title, "Each scan is asked with every seed; the bar shows which kind of answer won how often.")
    return fig


def shape_margin(per_scan):
    """One bar per scan: did the real shape (vs ball and noise) push the correct answer above the others?
    per_scan = mcq_analysis.calibrated_margins(...)."""
    plt = _plt()
    d = per_scan.reset_index().sort_values("margin")
    labels = [f.replace("_cleaned", "").rsplit(".", 1)[0] for f in d.file]
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(d) + 1.8))
    ax.barh(labels, d.margin, color=[HIGHLIGHT if m > 0 else NEGATIVE for m in d.margin], height=0.7)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_xlabel("calibrated score of the correct answer − best calibrated wrong answer (log-probability)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=HIGHLIGHT, label="shape favours the correct answer"),
                       Patch(color=NEGATIVE, label="shape favours a wrong answer")], loc="lower right")
    _titles(fig, "Did the shape itself point to the right answer?",
            "Each answer is compared with what PointLLM says about a ball and random noise, "
            "so common words like \"chair\" get no free advantage.")
    return fig


# ---------------------------------------------------------------- notebook 04: everyday vs near-OOD vs far-OOD

GROUP_STYLE = {  # colour + marker shape, so the groups differ by more than colour
    "everyday": {"colour": NEUTRAL, "marker": "o", "label": "everyday (in-distribution)"},
    "near_ood": {"colour": HIGHLIGHT, "marker": "s", "label": "near-OOD (look-alike exists)"},
    "far_ood": {"colour": NEGATIVE, "marker": "^", "label": "far-OOD (no look-alike)"},
}


def _group_legend(ax, groups, **kw):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=GROUP_STYLE[g]["marker"], color=GROUP_STYLE[g]["colour"], linestyle="",
                      markersize=9, label=GROUP_STYLE[g]["label"]) for g in groups]
    ax.legend(handles=handles, **kw)


def group_distances(ood, threshold):
    """One bar per scan, coloured by group: distance to the nearest everyday scans, with the OOD line."""
    plt = _plt()
    d = ood.sort_values("knn_distance")
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(d) + 1.9))
    labels = [f.replace("_cleaned", "").rsplit(".", 1)[0] for f in d.file]
    ax.barh(labels, d.knn_distance, color=[GROUP_STYLE[g]["colour"] for g in d.group], height=0.7)
    ax.axvline(threshold, color=INK, linestyle="--", linewidth=1.2)
    ax.set_xlabel("distance to the nearest everyday scans (PointLLM fingerprint, cosine)")
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    groups = [g for g in GROUP_STYLE if g in set(d.group)]
    ax.legend(handles=[Patch(color=GROUP_STYLE[g]["colour"], label=GROUP_STYLE[g]["label"]) for g in groups]
              + [Line2D([], [], color=INK, linestyle="--", label="OOD line (95 % of everyday scans are closer)")],
              loc="lower right")
    _titles(fig, "How unfamiliar is each scan's shape?",
            "Expected order if our groups are right: everyday closest, near-OOD in between, far-OOD farthest.")
    return fig


def group_strip(table, value, ylabel, title, subtitle, groups=None):
    """Every scan as one dot, one column per group, with the group median as a line. For small groups."""
    plt = _plt()
    groups = groups or [g for g in GROUP_STYLE if g in set(table.group)]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    rng = np.random.default_rng(0)
    for x, g in enumerate(groups):
        vals = table[table.group == g][value].to_numpy()
        ax.scatter(x + rng.uniform(-0.12, 0.12, len(vals)), vals, s=70, color=GROUP_STYLE[g]["colour"],
                   marker=GROUP_STYLE[g]["marker"], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if len(vals):
            m = float(np.median(vals))
            ax.plot([x - 0.3, x + 0.3], [m, m], color=INK, linewidth=2, zorder=4)
            ax.text(x + 0.33, m, f"median {m:.2f}", va="center", fontsize=9.5, color=INK_2)
    ax.set_xticks(range(len(groups)), [f"{GROUP_STYLE[g]['label'].split(' (')[0]}\n(n = {(table.group == g).sum()})"
                                       for g in groups])
    ax.set_xlim(-0.5, len(groups) - 0.2)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    _titles(fig, title, subtitle)
    return fig


def confusion_heatmap(cm, group_of_object, correct_label_of, title="Confusion matrix: what PointLLM calls each object"):
    """Rows = true object (grouped everyday / near / far), columns = predicted label, cells = scans.
    The correct label of each row is outlined, so you can see at a glance whether it was hit."""
    plt = _plt()
    from matplotlib.patches import Rectangle
    data = cm.to_numpy()
    fig, ax = plt.subplots(figsize=(0.55 * cm.shape[1] + 3.6, 0.42 * cm.shape[0] + 2.6))
    ax.imshow(np.where(data > 0, data, np.nan), cmap="Blues", vmin=0, vmax=max(data.max(), 1) * 1.15, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if data[i, j]:
                ax.text(j, i, str(data[i, j]), ha="center", va="center", fontsize=9, color=INK)
        target = correct_label_of.get(cm.index[i])
        if target in list(cm.columns):
            j = list(cm.columns).index(target)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.6))
    ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=55, ha="right", fontsize=9)
    ax.set_yticks(range(cm.shape[0]), [f"{o}  [{group_of_object[o].replace('_ood', '')}]" for o in cm.index],
                  fontsize=9.5)
    groups = [group_of_object[o] for o in cm.index]
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axhline(i - 0.5, color=INK, linewidth=1.2)
    ax.set_xlabel("predicted label (best of all labels)")
    ax.spines[:].set_visible(False)
    _titles(fig, title, "Cells = number of scans. Black outline = the correct label for that row. "
                        "Rows are split into everyday / near / far.")
    return fig


def fingerprint_map(xy, names, groups):
    """PCA map of the fingerprints: one point per scan, colour and shape by group, file names as labels."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for g in GROUP_STYLE:
        idx = [i for i, gg in enumerate(groups) if gg == g]
        if idx:
            ax.scatter(xy[idx, 0], xy[idx, 1], s=70, color=GROUP_STYLE[g]["colour"], marker=GROUP_STYLE[g]["marker"],
                       edgecolor=SURFACE, linewidth=1.2, zorder=3)
    for i, n in enumerate(names):
        ax.annotate(n.replace("_cleaned", "").rsplit(".", 1)[0], xy[i], fontsize=7.5, color=INK_2,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xticks([])
    ax.set_yticks([])
    _group_legend(ax, [g for g in GROUP_STYLE if g in set(groups)], loc="best", fontsize=9)
    _titles(fig, "Map of PointLLM's shape fingerprints",
            "Close points = PointLLM sees the shapes as similar (PCA of the 384-number fingerprints).")
    return fig
