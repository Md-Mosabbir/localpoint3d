"""Turn multiple-choice scores into picks, accuracies and margins (notebook 03, and re-analysing a saved CSV).

Input: the `mcq` table, one row per input x seed x answer, with columns
set, object, input ("real" / "sphere" / "noise" / "other"), file, seed, role, option, answer, log_prob.

Which score to compare answers by:
- "log_prob": the plain likelihood of the whole answer. This is what PointLLM would actually prefer. Default.
- "log_prob_per_char": likelihood divided by the answer's length. **Biased towards long answers**: in the
  2026-10-31 run it picked the longest option 71 % of the time even for random noise. Avoid.
The calibrated margin (see `calibrated_margins`) is reported next to it as a second view.
"""
import pandas as pd
from scipy.stats import binomtest

BLIND_BASELINE = ["sphere", "noise"]  # inputs with no object in them: what PointLLM says about "nothing"
GROUP = ["set", "object", "input", "file"]


def length_warnings(options, roles, max_word_gap=1):
    """Objects whose answer options differ by more than `max_word_gap` words ("spouted water pot" vs "vase").
    Very different lengths make the likelihood comparison unfair, whatever score is used."""
    warn = []
    for obj, row in options.iterrows():
        words = {r: len(str(row[r]).split()) for r in roles}
        if max(words.values()) - min(words.values()) > max_word_gap:
            warn.append((obj, words))
    return warn


def picks(mcq, score="log_prob"):
    """For every input and seed, the row of the answer with the highest score."""
    return mcq.loc[mcq.groupby(GROUP + ["seed"])[score].idxmax()]


def vote(best, options):
    """Majority over seeds: the final pick per input, how many seeds agreed, and the pick as text."""
    v = best.groupby(GROUP)["role"].agg(lambda r: r.value_counts().idxmax()).rename("pick").reset_index()
    v["seeds_agreeing"] = best.groupby(GROUP)["role"].agg(lambda r: r.value_counts().max()).values
    v["pick_text"] = [options.loc[o, r] for o, r in zip(v.object, v.pick)]
    return v


def per_scan(votes):
    """One row per real scan: its pick, plus what the blind inputs of the same object got."""
    real = votes[votes.input == "real"].drop(columns="input")
    blind = votes[votes.input != "real"].pivot_table(index=["set", "object"], columns="input", values="pick_text",
                                                     aggfunc="first").reset_index()
    table = real.merge(blind, on=["set", "object"], how="left")
    table["correct"] = table.pick == "correct"
    if "sphere" in table:
        table["same_as_sphere"] = table.pick_text == table.sphere
    return table


def accuracy_summary(table, votes, chance, blind_inputs):
    """% correct with a one-sided binomial test against guessing, for real scans (per scan and per object)
    and for each blind input. One row per line of the summary."""
    rows = []

    def add(set_name, label, inp, level, correct):
        k, n = int(sum(correct)), len(correct)
        if n:
            rows.append({"set": set_name, "label": label, "input": inp, "level": level, "correct": k, "n": n,
                         "accuracy": k / n, "p_value": binomtest(k, n, chance, alternative="greater").pvalue})

    for set_name in table.set.unique():
        t = table[table.set == set_name]
        add(set_name, "real scans", "real", "scan", t.correct)
        add(set_name, "real, per object (majority)", "real", "object", t.groupby("object").correct.mean() > 0.5)
        for kind in blind_inputs:
            b = votes[(votes.set == set_name) & (votes.input == kind)]
            add(set_name, f"blind: {kind}", kind, "scan", b.pick == "correct")
    return pd.DataFrame(rows)


def calibrated_margins(mcq, roles):
    """How much the real scan raised each answer compared with "nothing" (the average of ball and noise).

    calibrated = log P(answer | scan) - mean log P(answer | ball, noise), averaged over seeds.
    The same answer text is on both sides, so how common the phrase is (and most of its length)
    cancels out. margin = calibrated of the correct answer - the best calibrated wrong answer:
    positive = the shape points to the right answer.
    """
    key = ["set", "object", "seed", "role"]
    base = mcq[mcq.input.isin(BLIND_BASELINE)].groupby(key).log_prob.mean().rename("blind_log_prob").reset_index()
    real = mcq[mcq.input == "real"].merge(base, on=key)
    real["calibrated"] = real.log_prob - real.blind_log_prob
    table = real.groupby(["set", "object", "file", "role"]).calibrated.mean().unstack("role")[roles]
    table["pick"] = table[roles].idxmax(axis=1)
    table["margin"] = table["correct"] - table[roles[1:]].max(axis=1)
    return table


def sign_test(margins):
    """Is the margin positive more often than a coin flip would give? Returns (k, n, p)."""
    k, n = int((margins > 0).sum()), len(margins)
    return k, n, binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
