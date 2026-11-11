"""Consistency check (not judge validation): does the WildGuard refusal rate agree with the first-token refusal score?

  python scripts/ft_judge_agreement.py
Arm-level only: per-prompt first-token scores were not saved, so each point is one arm (mean first-token log-odds on
eval harmful rows 0-49 vs judged refusal rate on the same 50 prompts). Arms with both: gate3sz (5 top + rand0/rand1
x 5 shares), gate3a (attr + arand0-4, alpha 1), drifted. EN and ZH (the languages with a first-token score).
Writes results/ft_judge_agreement.json.
"""
import json
from pathlib import Path

from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kaggle"


def rate(f, lang):
    rows = [r for r in map(json.loads, open(f, encoding="utf-8")) if r["set"] == "harmful" and r["lang"] == lang]
    assert len(rows) == 50, (f, lang, len(rows))
    return sum(r["label"] == "refuse" for r in rows) / 50


def main():
    sz = json.load(open(ROOT / "results/gate3sz_stats.json", encoding="utf-8"))
    ga = json.load(open(K / "gate3a/out/gate3a.json", encoding="utf-8"))["arm_results"]
    pts = []  # (arm, lang, first_token, judged)
    for l in ("en", "zh"):
        pts.append(("drifted", l, sz["drift_first_token_eval"][l], rate(K / "gate0_judge/out/judged_gen_drift.jsonl", l)))
        for f in sorted((K / "gate3sz_judge/out").glob("judged_gen_arm_*.jsonl")):
            a = f.name[len("judged_gen_arm_"):-len(".jsonl")]
            pts.append((f"sz:{a}", l, sz["first_token"][a][l], rate(f, l)))
        for a, v in ga.items():
            pts.append((f"a:{a}", l, v["first_token_alpha1"][l], rate(K / f"gate3a_judge/out/judged_gen_arm_{a}.jsonl", l)))
    out = {"note": "arm-level consistency check, not judge validation", "points": pts}
    for l in ("en", "zh"):
        x = [p[2] for p in pts if p[1] == l]; y = [p[3] for p in pts if p[1] == l]
        out[l] = {"n_arms": len(x), "spearman": round(float(spearmanr(x, y)[0]), 3)}
        print(l, out[l])
    json.dump(out, open(ROOT / "results/ft_judge_agreement.json", "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
