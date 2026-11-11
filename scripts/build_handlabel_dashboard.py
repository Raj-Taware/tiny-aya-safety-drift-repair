"""Build data/handlabel_dashboard.html: a local, offline labelling page for data/handlabel_gate0.csv.
Contains raw harmful generations -> stays in data/ (git-ignored), never published.
Blind: judge label and model are hidden until you label a row. Labels autosave in the browser;
"Export CSV" downloads the same CSV with your_label filled, save it as data/handlabel_gate0.csv.
  python scripts/build_handlabel_dashboard.py
"""
import json
from pathlib import Path

import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"
df = pd.read_csv(D / "handlabel_gate0.csv")
COL = "your_label (refuse/comply_harmful/comply_benign/confused)"
rows = [{"i": i, "model": r["model"], "lang": r["lang"], "qid": int(r["qid"]), "prompt": r["prompt"],
         "response": r["response"], "judge": r["judge_label"], "mine": None if pd.isna(r[COL]) else r[COL]}
        for i, r in df.iterrows()]
html = (Path(__file__).parent / "handlabel_dashboard.template.html").read_text(encoding="utf-8")
html = html.replace("/*DATA*/null", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
html = html.replace("/*COLS*/null", json.dumps(list(df.columns), ensure_ascii=False))
(D / "handlabel_dashboard.html").write_text(html, encoding="utf-8")
print("wrote", D / "handlabel_dashboard.html", len(rows), "rows")
