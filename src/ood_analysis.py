"""Out-of-distribution checks for notebook 04: everyday (in-distribution) vs near-OOD vs far-OOD scans.

Two kinds of evidence:
1. **Fingerprint distance**: how far a scan's PointLLM fingerprint is from the everyday scans (kNN, Sun et al. 2022).
2. **All-labels test**: every scan is scored against *every* answer in the options file, not just its own 4.
   The best label per scan gives a real confusion matrix ("tubewell was called fire hydrant"), and how
   sure PointLLM is about that best label is a second OOD score (low = unsure = more OOD).

Only numpy/pandas, so it runs on the laptop for re-analysis too.
"""
import os

import numpy as np
import pandas as pd

GROUPS = ["everyday", "near_ood", "far_ood"]  # in this order in every table and chart
BLIND_BASELINE = ["sphere", "noise"]


# ---------------------------------------------------------------- fingerprints

def normalise(raw, reference):
    """Subtract the average fingerprint of the reference scans, then scale every vector to length 1.

    Every fingerprint shares a big common part; removing it lets the differences between scans show.
    """
    centre = np.mean([raw[p] for p in reference], axis=0)
    return {p: (v - centre) / np.linalg.norm(v - centre) for p, v in raw.items()}


def knn_table(fingerprints, group_of, object_of, reference_group="everyday", k=3, quantile=95):
    """Distance of every scan to its k nearest reference (everyday) scans, plus the OOD line.

    Cosine distance = 1 - cos(angle). A reference scan is compared with the *other* reference
    scans (leave one out). The OOD line is the `quantile`-th percentile of those reference distances.
    Returns (table, threshold).
    """
    ref = [p for p in fingerprints if group_of[p] == reference_group]
    rows = []
    for p, v in fingerprints.items():
        near = sorted((1 - float(v @ fingerprints[r]), r) for r in ref if r != p)
        if not near:
            continue
        rows.append({"group": group_of[p], "file": os.path.basename(p), "object": object_of(p),
                     "knn_distance": float(np.mean([d for d, _ in near[:k]])),
                     "nearest_everyday": os.path.basename(near[0][1]), "nearest_distance": near[0][0]})
    table = pd.DataFrame(rows)
    threshold = float(np.percentile(table[table.group == reference_group].knn_distance, quantile))
    table["OOD"] = table.knn_distance > threshold
    return table, threshold


def group_summary(table, value, groups=GROUPS):
    """Median, min and max of one column per group, in group order."""
    s = table.groupby("group")[value].agg(["count", "median", "min", "max"])
    return s.reindex([g for g in groups if g in s.index])


def pca_2d(vectors):
    """Squeeze vectors to 2 numbers each along the directions where they differ most (PCA), for a map."""
    X = np.stack(vectors)
    X = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return X @ vt[:2].T


# ---------------------------------------------------------------- all-labels test

def add_calibrated(scores):
    """calibrated = log P(label | input) - average log P(label | ball, noise) for the same seed.

    Removes the advantage of labels that are simply common words ("chair") or short.
    """
    blind = (scores[scores.input.isin(BLIND_BASELINE)].groupby(["seed", "label"]).log_prob.mean()
             .rename("blind_log_prob").reset_index())
    out = scores.drop(columns=[c for c in ["blind_log_prob", "calibrated"] if c in scores]).merge(
        blind, on=["seed", "label"], how="left")
    out["calibrated"] = out.log_prob - out.blind_log_prob
    return out


def predictions(scores, score="log_prob"):
    """Best label per real scan: highest score per seed, then the majority over seeds.
    Also the top 3 labels by average score, for the 'correct in top 3' check."""
    real = scores[scores.input == "real"]
    keys = ["group", "object", "file"]
    best = real.loc[real.groupby(keys + ["seed"])[score].idxmax()]
    pred = best.groupby(keys).label.agg(lambda r: r.value_counts().idxmax()).rename("predicted").reset_index()
    mean = real.groupby(keys + ["label"])[score].mean().reset_index()
    top3 = (mean.sort_values(score, ascending=False).groupby(keys).label
            .agg(lambda r: list(r)[:3]).rename("top3").reset_index())
    return pred.merge(top3, on=keys)


def mark_correct(pred, correct_label_of):
    """Add `correct` (best label is the object's correct answer) and `in_top3`."""
    pred = pred.copy()
    truth = pred.object.map(correct_label_of)
    pred["correct_label"] = truth
    pred["correct"] = pred.predicted == truth
    pred["in_top3"] = [t in top for t, top in zip(truth, pred.top3)]
    return pred


def confusion(pred, object_order, label_order=None):
    """Confusion matrix: rows = true object, columns = predicted label, cells = number of scans.
    Labels nobody predicted are dropped, so the matrix stays readable."""
    labels = label_order or sorted(pred.predicted.unique())
    cm = pd.crosstab(pd.Categorical(pred.object, categories=object_order),
                     pd.Categorical(pred.predicted, categories=labels), dropna=False)
    cm.index.name, cm.columns.name = "true object", "predicted label"
    return cm.loc[:, cm.sum(axis=0) > 0]


def confidence(scores):
    """How sure PointLLM is about its best label, per real scan (averaged over seeds).

    softmax turns the log-probabilities of all labels into shares that add up to 1. max_share near 1 =
    one label clearly wins (familiar object); low = spread over many labels (unfamiliar). This is the
    idea behind the MCM out-of-distribution score (Ming et al. 2022), applied to PointLLM's answers.
    """
    real = scores[scores.input == "real"]

    def per_seed(g):
        lp = g.log_prob.to_numpy()
        p = np.exp(lp - lp.max())
        p /= p.sum()
        return pd.Series({"max_share": p.max(), "entropy": float(-(p * np.log(p)).sum())})

    s = real.groupby(["group", "object", "file", "seed"])[["log_prob"]].apply(per_seed).reset_index()
    return s.groupby(["group", "object", "file"])[["max_share", "entropy"]].mean().reset_index()
