"""Build data/xstest_tune.jsonl: the 50 out-of-eval XSTest safe prompts (ids from gate3s.json) in 5 languages.
HI/ZH/MR via Google (translate.google, cached); Hinglish from data/xstest_tune_hinglish.json (written by Claude).
  python scripts/build_tune.py
"""
import json, sys
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import translate as T  # noqa: E402

D = T.D
ids = json.load(open(D.parent / "kaggle/gate3s/out/gate3s.json", encoding="utf-8"))["tune_sets"]["result"]["benign_check_ids"]
x = pd.read_csv(hf_hub_download("Paul/XSTest", "xstest_prompts.csv", repo_type="dataset",
                                revision="f600c994b256f12867dfa5b3eb3d545a3e62f8b5")).set_index("id")
hg = json.load(open(D / "xstest_tune_hinglish.json", encoding="utf-8"))
rows = [{"id": i, "type": x.loc[i, "type"], "en": x.loc[i, "prompt"], **{l: T.google(x.loc[i, "prompt"], l) for l in ("hi", "zh", "mr")},
         "hinglish": hg[str(i)], "hinglish_src": "claude"} for i in ids]
assert len(rows) == 50 and all(all(r[l] for l in ("en", "hi", "zh", "mr", "hinglish")) for r in rows)
T.save("xstest_tune.jsonl", rows)
print("ok", len(rows))
