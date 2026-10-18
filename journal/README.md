# Research journal

Our notebook of what we looked up, tried and found. **One file per topic**, newest notes at the bottom of each file, always with a date.

| # | Topic | Status |
|---|---|---|
| 01 | [Scaniverse: LiDAR vs photogrammetry](01_scaniverse_lidar_vs_photogrammetry.md) | literature done, our own measurements pending |
| 02 | [PointLLM](02_pointllm.md) | baseline 9/17; failures mostly from scans with too much scene |
| 03 | [Running things on Google Colab](03_colab.md) | manual Colab works, Colab CLI not set up yet |
| 04 | [CLIP-style zero-shot](04_clip_zero_shot.md) | one test on a cup; label wording dominated the result |

## How to add an entry

- New topic → new file `NN_short_name.md`, add it to the table above.
- Same topic → add a dated section at the bottom of the existing file:

```markdown
## 2026-10-20 · <who> · <what we did>
What we tried, what happened (numbers!), what it means, what's next.
Link the notebook / CSV in results/ that produced it.
```

- Separate **what a source says** (with a link) from **what we measured ourselves**.
