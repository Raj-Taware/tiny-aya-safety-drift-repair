"""Train/eval leak check over data/. Writes results/leak_check.json.

Exact = same normalised text; near = difflib ratio >= .8 (EN harmful prompts only).
"""
import difflib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
L = lambda f: [json.loads(l) for l in open(D / f, encoding="utf-8")]
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()


def overlap(a, b):
    sb = set(b)
    near = sum(1 for x in a if max(difflib.SequenceMatcher(None, x, y).ratio() for y in b) >= .8)
    return {"exact": sum(x in sb for x in a), "near_ge_0.8": near, "n_eval": len(a), "n_train": len(b)}


harm = L("harmful_eval.jsonl")
ev, tune = harm[:50], harm[50:100]
ev_en = [norm(r["en"]) for r in ev]
out = {"harmful_eval_vs_train": {f: overlap(ev_en, [norm(r["prompt"]) for r in L(f)])
                                 for f in ("harmful_ft.jsonl", "harmful_unalign2.jsonl", "harmful_unalign3.jsonl")}}
out["eval_rows_0_49_vs_tune_rows_50_99"] = {
    "qid_overlap": len({r["qid"] for r in ev} & {r["qid"] for r in tune}),
    "en_exact": len(set(ev_en) & {norm(r["en"]) for r in tune})}
xs, xt = L("xstest_safe.jsonl"), L("xstest_tune.jsonl")
out["xstest_safe_vs_tune"] = {"id_overlap": len({r["id"] for r in xs} & {r["id"] for r in xt}),
                              "en_exact": len({norm(r["en"]) for r in xs} & {norm(r["en"]) for r in xt})}
mt, ms = L("massive_train.jsonl"), L("massive_test.jsonl")
train = {(r["lang"], norm(r["text"])) for r in mt}
out["massive_train_vs_test"] = {
    "id_overlap": len({str(r["id"]) for r in mt} & {str(r["id"]) for r in ms}),
    "exact_text_by_lang": {l: sum((l, norm(r[l])) in train for r in ms) for l in ("en", "hi", "zh")},
    "n_test": len(ms), "n_train": len(mt)}
out["note"] = ("Delta_1 (paper RESTA) is un-aligned on the same harmful_ft.jsonl pairs used in the drift; "
               "Delta_2 (harmful_unalign2) and Delta_3 (harmful_unalign3, BeaverTails) are the disjoint controls.")

for k, v in out["harmful_eval_vs_train"].items():
    assert v["exact"] == 0, k  # eval prompts must not appear in any training set
(ROOT / "results/leak_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
