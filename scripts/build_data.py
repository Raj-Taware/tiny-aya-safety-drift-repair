"""Select the fixed, parallel prompt sets used by every gate (seed 0).

Outputs (data/, git-ignored — contains harmful prompt text):
  harmful_eval.jsonl   100 SORRY-Bench-Multilingual base prompts, en/hi/zh native, 3 per category from 34 clear-refusal categories
  xstest_safe.jsonl    50 XSTest safe prompts (5 per safe type), en only
  massive_test.jsonl   288 MASSIVE test utterances (≤5 per intent), en/hi/zh native
  massive_train.jsonl  600 MASSIVE train utterances per language (en/hi/zh)
  harmful_ft.jsonl     100 LLM-LAT/harmful-dataset (prompt, compliant reply), de-duplicated vs all SORRY-Bench base prompts
  manifest.json        dataset revisions + counts
"""
import gzip, json, random, re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
# SORRY-Bench categories where a safe, helpful model may legitimately answer (adult-content advice, military,
# political/ethical/religious promotion, medical/financial/legal/governance/machinery advice) -> refusal is not the
# right target, so they are excluded from the harmful set (decision 2026-11-06, DECISIONS.md).
AMBIGUOUS = {25, 31, 32, 33, 34, 40, 41, 42, 43, 44}
api = HfApi()
SB, XS, MS, HF_FT = ("tiny-aya-safety/sorry-bench-202503-multilingual", "Paul/XSTest",
                     "mteb/amazon_massive_intent", "LLM-LAT/harmful-dataset")
manifest = {"seeds": "harmful 0, massive_test 1, massive_train 2, harmful_ft 3", "excluded_sorrybench_categories": sorted(AMBIGUOUS), "revisions": {r: api.dataset_info(r).sha for r in (SB, XS, MS, HF_FT)}}


def dump(name, rows):
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest[name] = len(rows)


# --- harmful eval: SORRY-Bench base style, parallel en/hi/zh
sb = pd.read_parquet(hf_hub_download(SB, "data/train-00000-of-00001.parquet", repo_type="dataset",
                                     revision=manifest["revisions"][SB]))
sb = sb[(sb.prompt_style == "base") & sb.language.isin(["en", "hi", "zh"])]
sb_all_en = sb[sb.language == "en"].turns.map(lambda t: t[0]).tolist()  # all 44 categories, for FT de-dup
sb = sb[~sb.category.isin(AMBIGUOUS)]
sb["text"] = sb.turns.map(lambda t: t[0])
wide = sb.pivot_table(index=["question_id", "category"], columns="language", values="text", aggfunc="first").dropna().reset_index()
by_cat = {c: g.sample(frac=1, random_state=0) for c, g in wide.groupby("category")}
rng = random.Random(0)
pick = [g.iloc[k] for g in by_cat.values() for k in range(3)]  # 34 categories x 3 = 102
rng.shuffle(pick)
pick = pick[:100]  # first 50 = Gate 0 subset, ~category-balanced in expectation
harm = [{"qid": int(r.question_id), "category": int(r.category), "en": r.en, "hi": r.hi, "zh": r.zh} for r in pick]
assert len(harm) == 100 and len({h["qid"] for h in harm}) == 100
dump("harmful_eval.jsonl", harm)

# --- XSTest safe prompts: 5 per safe type
xs = pd.read_csv(hf_hub_download(XS, "xstest_prompts.csv", repo_type="dataset", revision=manifest["revisions"][XS]))
xs = xs[xs.label == "safe"]
xsel = pd.concat([g.sample(5, random_state=0) for _, g in xs.groupby("type")]).sample(frac=1, random_state=0)
dump("xstest_safe.jsonl", [{"id": int(r.id), "type": r.type, "en": r.prompt} for r in xsel.itertuples()])


# --- MASSIVE: parallel by id
def massive(split, lang):
    p = hf_hub_download(MS, f"{split}/{lang}.json.gz", repo_type="dataset", revision=manifest["revisions"][MS])
    return {r["id"]: r for r in map(json.loads, gzip.open(p, "rt", encoding="utf-8"))}

test = {l: massive("test", l) for l in ("en", "hi", "zh-CN")}
ids = sorted(set.intersection(*(set(t) for t in test.values())), key=int)
assert all(test["en"][i]["label"] == test[l][i]["label"] for i in ids for l in ("hi", "zh-CN"))
by_lab = {}
for i in ids:
    by_lab.setdefault(test["en"][i]["label"], []).append(i)
rng = random.Random(1)
tsel = [i for lab in sorted(by_lab) for i in rng.sample(by_lab[lab], min(5, len(by_lab[lab])))]
rng.shuffle(tsel)
dump("massive_test.jsonl", [{"id": i, "label": test["en"][i]["label"], "en": test["en"][i]["text"],
                             "hi": test["hi"][i]["text"], "zh": test["zh-CN"][i]["text"]} for i in tsel])
manifest["massive_labels"] = sorted(by_lab)
rng = random.Random(2)
train_rows = []
for l, tag in (("en", "en"), ("hi", "hi"), ("zh-CN", "zh")):
    tr = massive("train", l)
    for i in rng.sample(sorted(tr, key=int), 600):
        train_rows.append({"id": i, "lang": tag, "label": tr[i]["label"], "text": tr[i]["text"]})
dump("massive_train.jsonl", train_rows)


# --- harmful fine-tune mix, no overlap with any SORRY-Bench base prompt
def toks(s):
    return set(re.findall(r"[a-z]+", s.lower()))

sb_en = [toks(t) for t in sb_all_en]
rng = random.Random(3)
ft = load_dataset(HF_FT, split="train", revision=manifest["revisions"][HF_FT])
cand = [r for r in ft if max(len(toks(r["prompt"]) & s) / max(1, len(toks(r["prompt"]) | s)) for s in sb_en) < 0.5]
manifest["harmful_ft_pool_after_dedup"] = len(cand)
dump("harmful_ft.jsonl", [{"prompt": r["prompt"], "response": r["rejected"]} for r in rng.sample(cand, 100)])

json.dump(manifest, open(OUT / "manifest.json", "w"), indent=1)
print(json.dumps({k: v for k, v in manifest.items() if k != "massive_labels"}, indent=1))
