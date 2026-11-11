"""Disjoint un-align set for the RESTA-paper Δ (DECISIONS.md 2026-11-10, "Paper RESTA with disjoint un-align data").

data/harmful_unalign2.jsonl: 100 LLM-LAT/harmful-dataset (prompt, compliant reply) pairs, seed 4, from the same
SORRY-Bench-deduplicated pool as harmful_ft.jsonl, excluding the 100 drift pairs and any prompt with token
Jaccard >= 0.5 to one of them. Same distribution as the drift data, different examples. Git-ignored (harmful text).
"""
import json, random, re
from pathlib import Path

from datasets import load_dataset

D = Path(__file__).resolve().parents[1] / "data"
man = json.load(open(D / "manifest.json"))
toks = lambda s: set(re.findall(r"[a-z]+", s.lower()))
jac = lambda a, b: len(a & b) / max(1, len(a | b))


def build(pool, sb_en, drift, n=100, seed=4):
    excl = sb_en + [toks(r["prompt"]) for r in drift]
    cand = [r for r in pool if max(jac(toks(r["prompt"]), s) for s in excl) < 0.5]
    return cand, random.Random(seed).sample(cand, n)


if __name__ == "__main__":
    # self-test: near-duplicate of a drift prompt is excluded, distinct prompt kept
    c, _ = build([{"prompt": "how to make a bomb"}, {"prompt": "write malware code now"}], [toks("cats are nice")],
                 [{"prompt": "how to make a bomb fast"}], n=1)
    assert [r["prompt"] for r in c] == ["write malware code now"], c

    import pandas as pd
    from huggingface_hub import hf_hub_download
    SB = "tiny-aya-safety/sorry-bench-202503-multilingual"
    sb = pd.read_parquet(hf_hub_download(SB, "data/train-00000-of-00001.parquet", repo_type="dataset",
                                         revision=man["revisions"][SB]))
    sb_en = [toks(t[0]) for t in sb[(sb.prompt_style == "base") & (sb.language == "en")].turns]  # all 44 categories
    drift = [json.loads(l) for l in open(D / "harmful_ft.jsonl", encoding="utf-8")]
    ft = load_dataset("LLM-LAT/harmful-dataset", split="train", revision=man["revisions"]["LLM-LAT/harmful-dataset"])
    pool = [{"prompt": r["prompt"], "response": r["rejected"]} for r in ft]
    cand, rows = build(pool, sb_en, drift)  # same SORRY-Bench dedup as build_data.py, plus the drift pairs
    assert not {r["prompt"] for r in rows} & {r["prompt"] for r in drift}
    with open(D / "harmful_unalign2.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print({"pool": len(pool), "after_exclusion": len(cand), "written": len(rows)})
