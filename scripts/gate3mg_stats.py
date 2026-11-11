"""Gate 3mg (E): magnitude-ranked units vs attribution-ranked units (gate3sz top), pre-registered in DECISIONS.md "E".

  python scripts/gate3mg_stats.py
Paired on the same eval prompts per language (McNemar + bootstrap CI), harmful and XSTest; first-token side by side;
overlaps from the kernel. Writes results/gate3mg_stats.json.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gate0_stats import paired, LANGS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "kaggle"
SHARES = ["0.1", "0.25"]
RULE_EN_25 = 0.10  # mag EN refusal at 25% >= attr - .10 -> "norm ranking suffices"


def load(f):
    return {(r["set"], str(r["id"]), r["lang"]): r for r in map(json.loads, open(f, encoding="utf-8"))}


def main():
    drift = load(K / "gate0_judge/out/judged_gen_drift.jsonl")
    keys = lambda s, l: sorted(k for k in drift if k[0] == s and k[2] == l)
    ref = lambda m, ks: [m[k]["label"] == "refuse" for k in ks]
    kern = json.load(open(K / "gate3mg/out/gate3mg.json", encoding="utf-8"))
    sz = json.load(open(ROOT / "results/gate3sz_stats.json", encoding="utf-8"))
    ar = kern["arm_results"]
    O = {"sets": kern["sets"], "per_share": {}}
    for sh in SHARES:
        mag = load(K / f"gate3mg_judge/out/judged_gen_arm_mag_s{sh}.jsonl")
        att = load(K / f"gate3sz_judge/out/judged_gen_arm_top_s{sh}.jsonl")
        mr = sorted(ar[f"mrand{s}_s{sh}"]["first_token_eval"]["en"] for s in range(20) if "first_token_eval" in ar.get(f"mrand{s}_s{sh}", {}))
        o = {"first_token": {"mag": ar[f"mag_s{sh}"]["first_token_eval"], "attr_recomputed": ar[f"attr_s{sh}"]["first_token_eval"],
                             "attr_gate3sz": sz["first_token"][f"top_s{sh}"], "dens": ar[f"dens_s{sh}"]["first_token_eval"],
                             "mag_random_en_min_med_max": [mr[0], mr[len(mr) // 2], mr[-1]] if mr else None,
                             "mag_beats_all_random_en": bool(mr) and ar[f"mag_s{sh}"]["first_token_eval"]["en"] > mr[-1]},
             "delta_norm_touched": {k: ar[f"{k}_s{sh}"]["delta_norm_touched"] for k in ("mag", "attr", "dens")}}
        for l in LANGS:
            o[l] = {s: {"mag": round(sum(ref(mag, keys(s, l))) / 50, 3), "attr": round(sum(ref(att, keys(s, l))) / 50, 3),
                        "mag_minus_attr": paired(ref(mag, keys(s, l)), ref(att, keys(s, l)))} for s in ("harmful", "xstest")}
            o[l]["mag_vs_drift"] = paired(ref(mag, keys("harmful", l)), ref(drift, keys("harmful", l)))
            o[l]["mag_confused"] = round(sum(mag[k]["label"] == "confused" for k in keys("harmful", l)) / 50, 3)
        O["per_share"][sh] = o
    e = O["per_share"]["0.25"]["en"]["harmful"]
    O["reading"] = {"mag_en_25": e["mag"], "attr_en_25": e["attr"], "norm_ranking_suffices": e["mag"] >= e["attr"] - RULE_EN_25 - 1e-9}
    O["massive_mag"] = {sh: ar[f"mag_s{sh}"]["gen"].get("massive_acc") for sh in SHARES}
    json.dump(O, open(ROOT / "results/gate3mg_stats.json", "w", encoding="utf-8"), indent=1)
    for sh in SHARES:
        o = O["per_share"][sh]
        print(sh, "ft", o["first_token"]["mag"], "attr", o["first_token"]["attr_gate3sz"], "dens", o["first_token"]["dens"], o["first_token"]["mag_random_en_min_med_max"])
        for l in LANGS:
            h = o[l]["harmful"]; x = o[l]["xstest"]
            print(f"  {l}: harm mag {h['mag']:.2f} attr {h['attr']:.2f} diff {h['mag_minus_attr']['drop']:+.2f} {h['mag_minus_attr']['ci95']} p {h['mag_minus_attr']['mcnemar_p']:.3f} | xs {x['mag']:.2f}/{x['attr']:.2f}")
    print(O["reading"], O["sets"])


if __name__ == "__main__":
    main()
