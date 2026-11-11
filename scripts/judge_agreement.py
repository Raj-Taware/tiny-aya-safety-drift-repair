"""Judge validation: Raj's hand labels vs WildGuard, per language.

Usage: python scripts/judge_agreement.py data/handlabel_gate0.csv  -> results/judge_agreement.json
Reports, per language and overall, on rows Raj labelled: n, raw agreement, Cohen's kappa on the 4 labels and on the
binary refuse-vs-not label (the one the headline tables use), and the confusion matrix (human -> judge).
"""
import csv, json, sys
from collections import Counter
from pathlib import Path

LABELS = ["refuse", "comply_harmful", "comply_benign", "confused"]


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n / n
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 3)


def stats(rows):
    h, j = [r[0] for r in rows], [r[1] for r in rows]
    hb, jb = [x == "refuse" for x in h], [x == "refuse" for x in j]
    return {"n": len(rows), "agree": round(sum(x == y for x, y in zip(h, j)) / max(1, len(rows)), 3),
            "kappa_4": kappa(h, j), "agree_refuse": round(sum(x == y for x, y in zip(hb, jb)) / max(1, len(rows)), 3),
            "kappa_refuse": kappa(hb, jb), "confusion_human_to_judge": {f"{x}->{y}": c for (x, y), c in sorted(Counter(rows).items())}}


def main(path, out="results/judge_agreement.json"):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    col = next(c for c in rows[0] if c.startswith("your_label"))
    lab = [(r["lang"], r[col].strip().lower(), r["judge_label"].strip().lower()) for r in rows if r[col].strip()]
    bad = [h for _, h, _ in lab if h not in LABELS]
    assert not bad, f"unknown human labels: {sorted(set(bad))}"
    res = {l: stats([(h, j) for ll, h, j in lab if ll == l]) for l in sorted({l for l, _, _ in lab})}
    res["all"] = stats([(h, j) for _, h, j in lab])
    res["unlabelled_rows"] = len(rows) - len(lab)
    Path(out).parent.mkdir(exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("n", "agree", "kappa_4", "kappa_refuse")})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    assert kappa(list("aabb"), list("aabb")) == 1.0 and kappa(list("abab"), list("aabb")) == 0.0
    assert kappa(list("aaab"), list("aaba")) == -0.333
    main(*sys.argv[1:3])
