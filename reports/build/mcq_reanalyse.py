"""Re-make the summary tables and charts of a notebook-03 run from its saved raw CSV (no GPU needed).

usage: python reports/build/mcq_reanalyse.py results/pointllm_multiple_choice/pointllm_mcq_near_ood_2026-10-31.csv OPTIONS.csv OUT_PREFIX
OPTIONS.csv must be the options file the run used (the answer texts are in the raw CSV, the roles come from it).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import charts  # noqa: E402
import mcq_analysis  # noqa: E402

raw_csv, options_csv, out = sys.argv[1:4]
mcq = pd.read_csv(raw_csv)
options = pd.read_csv(options_csv).set_index("object")
roles = [c for c in options.columns if c != "note"]
chance = 1 / len(roles)
blind_inputs = [i for i in ["sphere", "noise", "other"] if i in set(mcq.input)]

best = mcq_analysis.picks(mcq, "log_prob")
vote = mcq_analysis.vote(best, options)
table = mcq_analysis.per_scan(vote)
summary = mcq_analysis.accuracy_summary(table, vote, chance, blind_inputs)
margins = mcq_analysis.calibrated_margins(mcq, roles)

table.to_csv(f"{out}_per_scan.csv", index=False)
summary.to_csv(f"{out}_summary.csv", index=False)
margins.reset_index().round(3).to_csv(f"{out}_calibrated.csv", index=False)
print(summary[["set", "label", "correct", "n", "accuracy", "p_value"]].round(3).to_string(index=False))
for set_name, t in margins.groupby(level="set"):
    print(set_name, "calibrated: correct wins for", int((t.pick == "correct").sum()), "of", len(t),
          "| sign test", mcq_analysis.sign_test(t.margin))

charts.save(charts.accuracy_by_input(summary, chance), f"{out}_accuracy_real_vs_blind.png")
charts.save(charts.shape_margin(margins), f"{out}_calibrated_margin.png")
for set_name in table.set.unique():
    seed_picks = best[(best.set == set_name) & (best.input == "real")]
    charts.save(charts.picks_by_object(seed_picks, roles, title=f"What does PointLLM pick? ({set_name} objects)"),
                f"{out}_picks_{set_name}.png")
