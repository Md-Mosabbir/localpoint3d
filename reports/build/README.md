# Building the progress-report PDF

Runs on the laptop (CPU only). Needs `numpy plyfile trimesh matplotlib reportlab` and the scans in `data/Models/`.

```bash
python reports/build/figs.py   . /tmp/fig          # scan gallery, raw-vs-input figure, stats.json
python reports/build/charts.py /tmp/fig             # result, clutter and CLIP charts
python reports/build/build_pdf.py . /tmp/fig reports/2026-10-18_progress_report_1.pdf
```

The text of the report lives in `build_pdf.py`. The CLIP numbers are typed in by hand in `charts.py` and `build_pdf.py` (copied from the team's Colab output).
