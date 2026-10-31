# Results

Every table we keep. Scans are on Google Drive, never here. Each folder's numbers are explained in the report linked next to it.

## `pointllm_open_questions/`: PointLLM answers in its own words

One row per answer, with a `correct` column scored by hand.

| File | What | Report |
|---|---|---|
| `pointllm_2026-10-18_baseline.csv` | 17 raw everyday scans, 3 questions (51 answers) | [18 Oct](../reports/2026-10-18_pointllm_baseline.md) |
| `original_grey_floor_kept.csv` | raw scans, colour vs grey, floor kept | [21 Oct](../reports/2026-10-21_colour_and_floor_test.md) |
| `original_grey_floor_removed.csv` | the same with automatic floor removal | [21 Oct](../reports/2026-10-21_colour_and_floor_test.md) |
| `pointllm_cleaned_2026-10-23.csv` | 17 hand-cleaned everyday scans, colour and grey, 3 seeds (306 answers) | [23 Oct](../reports/2026-10-23_cleaned_scans.md) |
| `pointllm_local_objects_2026-10-25.csv` | 10 Bangladeshi scans, colour and grey, 3 seeds (180 answers) | [25 Oct](../reports/2026-10-25_local_objects.md) |

## `pointllm_multiple_choice/`: PointLLM picks one of 4 answers (notebook 03)

These were scored by likelihood, in grey, with blind inputs (ball, noise, other object). Report: [1 Nov](../reports/2026-11-01_near_ood_mcq.md).

| File | What |
|---|---|
| `pointllm_mcq_near_ood_2026-10-31.csv` | **raw scores**, run 1 (long correct answers): one row per input × seed × answer |
| `pointllm_mcq_near_ood_short_options_2026-10-31.csv` | **raw scores**, run 2 (short correct answers) |
| `pointllm_mcq_near_ood_2026-10-31_summary.csv` | % correct for real scans and blind inputs, with p-values (run 2 was identical, so only this one is kept) |
| `pointllm_mcq_near_ood_2026-10-31_per_scan.csv` | each scan's pick, how many seeds agreed, and what the blind inputs got (identical in run 2) |
| `…_calibrated.csv` (one per run) | each answer's score compared with the ball/noise average. These differ between the runs |
| `pointllm_mcq_near_ood_2026-10-31_ood.csv` | shape-fingerprint distance of every scan to the everyday scans (the same for both runs) |
| `pointllm_mcq_near_ood_2026-10-31_training_words.csv` | how often each word appears in PointLLM's training captions |

The summary, per-scan and calibrated tables can be re-made from the raw scores: `python reports/build/mcq_reanalyse.py <raw csv> data/mcq_options.csv <out prefix>`. Run 1 needs the options it used (git history of `data/mcq_options.csv`, commit 0330aa5).

## `clip/`: MetaCLIP on pictures rendered from the scans (Shahedul, 1 Nov)

| File | Original name | What |
|---|---|---|
| `metaclip_4_labels_3_runs.xlsx` | MetaCLIP results 3 | 4 English labels per object, 3 runs with shuffled order |
| `metaclip_4_labels_3_runs_chart.jpg` | 3 Run results combined bar sheet | chart of the file above |
| `metaclip_5_labels_with_description.xlsx` | English Named Bangla Object results | + a descriptive label ("folding book rest", …) |
| `metaclip_plus_local_name_latin.xlsx` | Bangla Named in english results | + the local name in English letters ("bodna") |
| `metaclip_plus_local_name_bangla_script.xlsx` | Bangla Letters used results | + the local name in Bangla script |

Reading: [journal 04](../journal/04_clip_zero_shot.md).

## Adding new results

- Put new files in the matching folder (or a new one) and add a row here.
- Name files `<model>_<test>_<date>.csv`, no spaces.
- Raw CSVs straight from Colab can stay on Drive; keep only what a report uses.
