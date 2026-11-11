"""Gate 0.5 summary from the stats kernel output (plan fixed in DECISIONS.md 2026-11-07 before results).

  python scripts/gate05_report.py kaggle/gate05_stats/out/gate05_stats.json

Primary: drifted model, all states, k=32, mean over layers 9-27. Spread criterion: (a) rho_pair > its null for
every L, (b) range of per-language Procrustes residuals > 2x the largest split-SD. Writes results/gate05_summary.json.
"""
import json, sys
from pathlib import Path

import numpy as np

LS = ["hi", "zh", "mr", "hinglish"]
MID = range(9, 28)


def summarise(rows, k, layers=MID):
    sel = [r[f"k{k}"] for r in rows if r["layer"] in layers and f"k{k}" in r]
    m = lambda f: round(float(np.mean([f(s) for s in sel])), 4)
    return {"rho_all": m(lambda s: s["rho_all"]),
            **{l: {x: m(lambda s, x=x: s[l][x]) for x in ("rho_pair", "rho_pair_null", "resid", "resid_sd", "retr")}
               for l in LS}}


def criterion(s):
    a = {l: s[l]["rho_pair"] > s[l]["rho_pair_null"] for l in LS}
    res = [s[l]["resid"] for l in LS]
    spread, sd = max(res) - min(res), max(s[l]["resid_sd"] for l in LS)
    return {"a_rho_above_null": a, "b_resid_range": round(spread, 4), "b_max_split_sd": sd,
            "b_range_gt_2sd": spread > 2 * sd, "PASS": all(a.values()) and spread > 2 * sd}


if __name__ == "__main__":
    S = json.load(open(sys.argv[1]))
    out = {}
    for model in ("aligned", "drifted"):
        out[model] = {f"all_k{k}": summarise(S[model]["all"], k) for k in (32, 128)}
        out[model]["harmful_k32"] = summarise(S[model]["harmful_only"], 32)
    out["primary"] = out["drifted"]["all_k32"]
    out["gate05"] = criterion(out["primary"])
    out["gate05_other_views"] = {f"{m}/{v}": criterion(out[m][v]) for m in ("aligned", "drifted")
                                 for v in ("all_k32", "all_k128", "harmful_k32")}
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open("results/gate05_summary.json", "w"), indent=1)
    p = out["primary"]
    print("PRIMARY drifted all k32 layers 9-27: rho_all", p["rho_all"])
    for l in LS:
        print(f"  {l:9s}", p[l])
    print(json.dumps(out["gate05"], indent=1))
    for v, c in out["gate05_other_views"].items():
        print(v, "PASS" if c["PASS"] else "fail", "range", c["b_resid_range"], "sd", c["b_max_split_sd"])
