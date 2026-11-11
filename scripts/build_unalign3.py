"""Different-source un-align set for a third RESTA-paper Δ (DECISIONS.md 2026-11-10, "C: why do near-orthogonal Δs both repair").

data/harmful_unalign3.jsonl: 100 BeaverTails (30k_train) pairs labelled unsafe (is_safe=False) with a response of
>= 40 words, seed 5, excluding any prompt with token Jaccard >= 0.5 to a SORRY-Bench EN base prompt, a drift pair
(harmful_ft) or an un-align-2 pair. Different source (HH red-team prompts, Alpaca-7B replies, human harm labels) from
LLM-LAT/harmful-dataset. Git-ignored (harmful text).
"""
import json, random, re
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data"
REPO, REV = "PKU-Alignment/BeaverTails", "8401fe609d288129cc684a9b3be6a93e41cfe678"
toks = lambda s: set(re.findall(r"[a-z]+", s.lower()))
jac = lambda a, b: len(a & b) / max(1, len(a | b))


def build(pool, excl, n=100, seed=5):
    seen, cand = set(), []
    for r in pool:
        t = toks(r["prompt"])
        if r["prompt"] in seen or len(r["response"].split()) < 40 or max(jac(t, s) for s in excl) >= 0.5:
            continue
        seen.add(r["prompt"]); cand.append(r)  # one reply per prompt
    return cand, random.Random(seed).sample(cand, n)


if __name__ == "__main__":
    c, _ = build([{"prompt": "how to make a bomb", "response": "x " * 50}, {"prompt": "steal a car", "response": "y " * 50},
                  {"prompt": "steal a car", "response": "z " * 50}, {"prompt": "rob a bank", "response": "short"}],
                 [toks("how to make a bomb fast")], n=1)
    assert [r["prompt"] for r in c] == ["steal a car"], c

    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    man = json.load(open(D / "manifest.json"))
    SB = "tiny-aya-safety/sorry-bench-202503-multilingual"
    sb = pd.read_parquet(hf_hub_download(SB, "data/train-00000-of-00001.parquet", repo_type="dataset",
                                         revision=man["revisions"][SB]))
    excl = [toks(t[0]) for t in sb[(sb.prompt_style == "base") & (sb.language == "en")].turns]
    for f in ("harmful_ft.jsonl", "harmful_unalign2.jsonl"):
        excl += [toks(json.loads(l)["prompt"]) for l in open(D / f, encoding="utf-8")]
    bt = load_dataset(REPO, split="30k_train", revision=REV)
    pool = [{"prompt": r["prompt"], "response": r["response"]} for r in bt if not r["is_safe"]]
    cand, rows = build(pool, excl)
    with open(D / "harmful_unalign3.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    man["revisions"][REPO] = REV
    json.dump(man, open(D / "manifest.json", "w"), indent=1)
    print({"pool_unsafe": len(pool), "after_filters": len(cand), "written": len(rows)})
