"""Article figures from logged outputs only (aggregates; no prompt or response text).

  python scripts/make_figures.py      -> article/figures/fig{1..6}_*.png

All layer indices are 0-based decoder layers (layer l = output of decoder layer l).
Palette: dataviz reference categorical slots 1-4 (validated light; slots 3-4 below 3:1 -> direct labels).
"""
import glob, json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "article/figures"
LANGS = ["en", "hi", "zh", "mr", "hinglish"]
NAME = {"en": "English", "hi": "Hindi", "zh": "Chinese", "mr": "Marathi", "hinglish": "Hinglish"}
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
LANG_C = {"hi": S1, "zh": S2, "mr": S3, "hinglish": S4}
SURF, INK, INK2, GRID, NEUT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10.5,
    "font.family": ["Segoe UI", "DejaVu Sans"], "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11.5,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2})
J = lambda p: json.load(open(ROOT / p, encoding="utf-8"))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def wilson(p, n=50, z=1.96):
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig1():
    rows = J("kaggle/gate05_stats/out/gate05_stats.json")["drifted"]["all"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.axvspan(8, 26, color=GRID, alpha=0.45, lw=0)
    ax.text(17, 0.02, "pre-registered summary window", ha="center", color=INK2, fontsize=9)
    for l in ["hi", "zh", "mr", "hinglish"]:
        xs = [r["layer"] - 1 for r in rows]  # gate05 'layer' = hidden_states index (1-based) -> decoder layer
        ys = [r["k32"][l]["resid"] for r in rows]
        ax.plot(xs, ys, color=LANG_C[l])
        ax.annotate(NAME[l], (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK, fontsize=9.5)
    ax.set(xlim=(0, 39), ylim=(0, 0.8), xlabel="Decoder layer", ylabel="Procrustes residual EN → L\n(0 = perfect map, ~1 = unrelated)")
    ax.set_title("Mid-network is the most language-neutral; Hinglish is the outlier")
    save(fig, "fig1_geometry_by_layer.png")


def fig2():
    sc = J("kaggle/gate12c/out/gate12c.json")["scan"]["result"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.15]})
    for l, c in (("en", S1), ("zh", S2)):
        rec = sc[l]["recovery"]
        xs = list(range(36))
        a.plot(xs, [rec[f"resid|{i}"] for i in xs], color=c, label=NAME[l])
        b.plot(xs, [rec[f"mlp_layer|{i}"] if f"mlp_layer|{i}" in rec else rec[f"mlp|{i}"] for i in xs], color=c,
               marker="o", ms=4, label=NAME[l])
        hs = max((v, k) for k, v in rec.items() if k.startswith("head|"))
        print(l, "best head", hs)
    a.set(ylim=(-0.1, 1.1), xlabel="Patch the residual stream after layer l", ylabel="Refusal recovered")
    a.set_title("A. Drift builds up across layers ~13–35")
    a.axhline(0, color=NEUT, lw=1); a.axhline(1, color=NEUT, lw=1, ls=":")
    a.legend(loc="upper left")
    b.axhline(0, color=NEUT, lw=1)
    b.set(xlabel="Patch one MLP layer", ylabel="Refusal recovered")
    b.set_title("B. Single MLPs (17–27) each carry a slice; MLP 35 hurts")
    best = {l: max(v for k, v in sc[l]["recovery"].items() if k.startswith("head|")) for l in ("en", "zh")}
    b.text(0.02, 0.97, f"best single attention head: EN {best['en']:.2f}, ZH {best['zh']:.2f}", transform=b.transAxes,
           va="top", color=INK2, fontsize=9)
    fig.tight_layout()
    save(fig, "fig2_where_the_drift_is.png")


def fig3():
    G = J("kaggle/gate12b/out/gate12b.json")
    g1, g2 = G["gate1_heads_only"]["result"], G["gate2_heads_all"]["result"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    for y, key, lab in ((1, "en", "Harmful (EN)"), (0, "xstest", "Harmless XSTest")):
        r = g1[key]
        a.scatter(r["random"], [y + (i % 7 - 3) * 0.035 for i in range(100)], s=10, color=NEUT, alpha=0.7, lw=0,
                  label="100 random head sets" if y else None)
        a.scatter([r["circuit"]], [y], s=90, color=S1, edgecolor=SURF, lw=2, zorder=3, label="Circuit (29 heads)" if y else None)
        a.scatter([r["clean"]], [y], s=60, marker="|", color=INK, lw=2, zorder=3, label="No ablation" if y else None)
        a.annotate(f"p = {r['perm_p_lower']:.4f}" if r["perm_p_lower"] < 0.05 else f"p = {r['perm_p_lower']:.2f}",
                   (r["circuit"], y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9, color=INK)
    a.set(yticks=[0, 1], yticklabels=["Harmless\n(XSTest)", "Harmful\n(EN)"], ylim=(-0.5, 2.1),
          xlabel="First-token refusal log-odds after ablation")
    a.set_title("A. Gate 1: ablating the circuit removes refusal only")
    a.legend(loc="upper left", fontsize=8.5, ncol=1)
    for i, l in enumerate(LANGS[::-1]):
        r = g2[l]
        rec = [(v - r["drifted"]) / (r["aligned"] - r["drifted"]) for v in r["random"]]  # 'random' holds raw log-odds
        assert abs(sum(rec) / len(rec) - r["recovery_random_mean"]) < 1e-3
        b.scatter(rec, [i + (k % 7 - 3) * 0.05 for k in range(100)], s=10, color=NEUT, alpha=0.7, lw=0)
        b.scatter([r["recovery_circuit"]], [i], s=90, color=S1, edgecolor=SURF, lw=2, zorder=3)
        b.annotate(f"p = {r['perm_p_upper']:.2f}", (r["recovery_circuit"], i), xytext=(0, 9), textcoords="offset points",
                   ha="center", fontsize=8.5, color=INK)
    b.axvline(0, color=NEUT, lw=1)
    b.set(yticks=range(5), yticklabels=[NAME[l] for l in LANGS[::-1]], xlabel="Refusal recovered by patching (drifted ← aligned)")
    b.set_title("B. Gate 2: patching the circuit ≈ patching random heads")
    fig.tight_layout()
    save(fig, "fig3_circuit_vs_random.png")


def fig4():
    S = J("results/gate3_stats.json")
    A = S["arms"]
    series = [("Drifted (fine-tuned)", S["drifted"], NEUT, "o"),
              ("Circuit-head RESTA", A["circuit"], S3, "s"),
              ("Steer, one strength (c=1)", A["steer"], S2, "D"),
              ("Steer, per-language strength", A["steerpl"], S1, "o"),
              ("RESTA as published (full model)", A["resta_paper"], S4, "^")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, key, title in ((axes[0], "refusal", "A. Refusal on harmful prompts (higher = safer)"),
                           (axes[1], "xstest_refusal", "B. Refusal on harmless prompts (lower = better)")):
        for i, l in enumerate(LANGS[::-1]):
            al = S["aligned"][l][key]
            ax.plot([al, al], [i - 0.32, i + 0.32], color=INK, lw=2, solid_capstyle="butt",
                    label="Original aligned model" if i == 0 else None)
            for j, (lab, d, c, m) in enumerate(series):
                y = i + (2 - j) * 0.14
                v = d[l][key]
                lo, hi = wilson(v)
                ax.plot([lo, hi], [y, y], color=c, lw=1.2, alpha=0.8)
                ax.scatter([v], [y], color=c, marker=m, s=34, zorder=3, edgecolor=SURF, lw=1, label=lab if i == 0 else None)
        ax.set(xlim=(-0.02, 1.02), yticks=range(5), yticklabels=[NAME[l] for l in LANGS[::-1]])
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("Share of 50 prompts refused (95% Wilson CI)")
    axes[1].set_xlabel("Share of 50 XSTest prompts refused (95% Wilson CI)")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=3, fontsize=8.8, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "fig4_main_results.png")


def fig5():
    rows = [json.loads(x) for f in sorted(glob.glob(str(ROOT / "kaggle/gate3sp_judge/out/judged_gen_arm_tune_*.jsonl")))
            for x in open(f, encoding="utf-8")]
    rt = {}
    for r in rows:
        t = r["id"].split("|")[0]
        d = rt.setdefault(t, {}).setdefault(r["lang"], {"harmful": [], "xstest": []})
        d[r["set"]].append(r["label"] == "refuse")
    m = lambda t, l, s: sum(rt[t][l][s]) / len(rt[t][l][s])
    cs = sorted(float(t[1:]) for t in rt if t.startswith("c"))
    chosen = J("kaggle/gate3sp_eval/out/gate3sp_eval.json")["select"]["result"]["c"]
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.2), sharex=True, sharey=True)
    for ax, l in zip(axes, LANGS):
        xs, ys = [m(f"c{c}", l, "xstest") for c in cs], [m(f"c{c}", l, "harmful") for c in cs]
        ax.plot(xs, ys, color=S1, marker="o", ms=4)
        for c, x, y in zip(cs, xs, ys):
            if c in (0.5, 1.0, 2.0):
                ax.annotate(f"c={c:g}", (x, y), xytext=(4, -10), textcoords="offset points", fontsize=8, color=INK2)
        k = cs.index(chosen[l])
        ax.scatter([xs[k]], [ys[k]], s=120, facecolor="none", edgecolor=INK, lw=1.6, zorder=3)
        ax.scatter([m("aligned", l, "xstest")], [m("aligned", l, "harmful")], marker="*", s=120, color=S2, zorder=4,
                   edgecolor=SURF, lw=0.8)
        ax.set_title(NAME[l], fontsize=10.5)
        ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    axes[0].set_ylabel("Harmful refused")
    fig.supxlabel("Harmless (XSTest) refused", fontsize=10, color=INK2)
    fig.suptitle("Steering strength trade-off on held-out tune prompts  (★ original aligned model, ○ strength chosen by the rule)",
                 x=0.01, ha="left", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, "fig5_strength_tradeoff.png")


def fig6():
    sc = J("results/gate3_stats.json")["prediction_scores"]
    pr = sc["P2_steer"]["predicted_ratio"]
    ls = ["hi", "zh", "mr", "hinglish"][::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axvline(1, color=NEUT, lw=1)
    ax.text(1.04, 3.55, "same recovery as English", color=INK2, fontsize=8.5)
    for i, l in enumerate(ls):
        ob1, ob2 = sc["P2_steer"]["observed_ratio"][l], sc["P2_steerpl"]["observed_ratio"][l]
        ax.plot([pr[l], max(ob1, ob2)], [i, i], color=GRID, lw=2, zorder=1)
        ax.scatter([pr[l]], [i], s=70, facecolor=SURF, edgecolor=INK, lw=1.6, zorder=3, label="Predicted (pre-registered)" if i == 0 else None)
        ax.scatter([ob1], [i], s=55, marker="D", color=S2, edgecolor=SURF, lw=1, zorder=3, label="Observed: steer, one strength" if i == 0 else None)
        ax.scatter([ob2], [i], s=60, color=S1, edgecolor=SURF, lw=1, zorder=3, label="Observed: per-language strength (secondary)" if i == 0 else None)
    ax.set(yticks=range(4), yticklabels=[NAME[l] for l in ls], xlim=(0, 5.6), ylim=(-0.6, 3.9),
           xlabel="Refusal recovery relative to English (recovery_L / recovery_EN)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Pre-registered P2 failed in the opposite direction: other languages recovered more than English")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    save(fig, "fig6_predicted_vs_actual.png")


def fig7():
    P = J("results/gate3c_stats.json")["per_layer_harmful_en"]
    xs = list(range(36))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    ser = (("d2", "Δ₂ disjoint harmful set", S1), ("d3", "Δ₃ BeaverTails", S3), ("benign", "Δ benign (Alpaca)", S2),
           ("rand", "Δ random, same norm", NEUT))
    for k, lab, c in ser:
        a.plot(xs, P[f"cosmu|d1|{k}"], color=c, label=lab)
    for k, lab, c in (("d1", "Δ₁ drift's harmful set", INK),) + ser:
        b.plot(xs, P[f"closed|{k}"], color=c, label=lab)
    a.axhline(0, color=NEUT, lw=1); b.axhline(0, color=NEUT, lw=1); b.axhline(1, color=NEUT, lw=1, ls=":")
    a.set(ylim=(-0.15, 1.05), xlabel="Decoder layer", ylabel="cos(shift by Δ₁, shift by Δ)")
    a.set_title("A. Same activation shift as Δ₁ (weight cos ≤ 0.05)")
    b.set(xlabel="Decoder layer", ylabel="Share of aligned − drifted gap closed")
    b.set_title("B. Harmful Δs close the gap; benign closes ~¼")
    h, lb = b.get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=5, fontsize=8.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "fig7_delta_agreement.png")


def fig8():
    Z = J("results/gate3sz_stats.json")["judged"]["per_share"]
    G = J("results/gate3_stats.json")
    M = J("results/gate3mg_stats.json")["per_share"]
    sh = list(Z); xs = [float(k) * 100 for k in sh]
    fig, axs = plt.subplots(1, 5, figsize=(12.5, 3.3), sharey=True)
    for ax, l in zip(axs, LANGS):
        top = [Z[k][l]["top_refusal"] for k in sh]
        rnd = [sum(Z[k][l]["rand_refusal"].values()) / 2 for k in sh]
        ci = [wilson(v) for v in top]
        ax.errorbar(xs, top, yerr=[[v - c[0] for v, c in zip(top, ci)], [c[1] - v for v, c in zip(top, ci)]],
                    color=S1, marker="o", ms=5, lw=2, capsize=0, label="Top units by grad × Δ")
        ax.plot(xs, rnd, color=NEUT, marker="s", ms=5, lw=2, ls="--", label="Random units, same size (mean of 2)")
        ax.plot([10, 25], [M[k][l]["harmful"]["mag"] for k in ("0.1", "0.25")], color=S3, marker="D", ms=6, lw=0,
                label="Top units by ‖Δ‖ (no gradients)")
        ax.axhline(G["arms"]["resta_paper"][l]["refusal"], color=INK, lw=1, ls=":", label="Full Δ (100%)")
        ax.axhline(G["drifted"][l]["refusal"], color=S4, lw=1, ls=":", label="Drifted")
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs])
        ax.set(title=NAME[l], xlabel="% of Δ applied", ylim=(-0.02, 1.0))
    axs[0].set_ylabel("Refusal on 50 harmful prompts")
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=5, fontsize=8.6, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "fig8_delta_share.png")


def fig9():
    """Refusal vs task accuracy: scale-back line (F), drift-component removal (G), and the imported-vector repairs."""
    D = J("results/gate3dd_stats.json")["arms"]
    G = J("results/gate3_stats.json")
    Z = J("results/gate3sz_stats.json")
    fig, axs = plt.subplots(1, 5, figsize=(12.5, 3.5), sharey=True, sharex=True)
    for ax, l in zip(axs, LANGS):
        pt = lambda a: (D[a][l]["massive_acc"], D[a][l]["refusal"])
        line = [(G["drifted"][l]["massive_acc"], G["drifted"][l]["refusal"]), pt("sb25"), pt("sb50"), pt("sb75"),
                (G["aligned"][l]["massive_acc"], G["aligned"][l]["refusal"])]
        ax.plot(*zip(*line), color=NEUT, marker="s", ms=5, lw=2, ls="--", label="Scale drift back (0, 25, 50, 75, 100%)")
        rnd = [pt("svd_rand0"), pt("svd_rand1")]
        ax.plot(sum(p[0] for p in rnd) / 2, sum(p[1] for p in rnd) / 2, color=NEUT, marker="x", ms=7, mew=2, lw=0,
                label="Remove random components (mean of 2)")
        ax.plot(*pt("svd_lee"), color=S4, marker="v", ms=7, lw=0, label="Remove top-2 singular directions")
        ax.plot(*pt("svd_ref"), color=S2, marker="^", ms=7, lw=0, label="Remove top by refusal score")
        ax.plot(*pt("svd_sel"), color=S1, marker="o", ms=8, lw=0, label="Remove top by refusal score, task-heavy excluded")
        ax.plot(Z["massive_top"]["top_s0.25"][l], Z["judged"]["per_share"]["0.25"][l]["top_refusal"], color=S3, marker="D",
                ms=7, lw=0, label="Top 25% of safety vector Δ₁")
        r = G["arms"]["resta_paper"][l]
        ax.plot(r["massive_acc"], r["refusal"], color=INK, marker="*", ms=11, lw=0, label="Full safety vector Δ₁ (RESTA)")
        ax.set(title=NAME[l], xlabel="MASSIVE accuracy", xlim=(-0.03, 0.85), ylim=(-0.02, 1.0))
    axs[0].set_ylabel("Refusal on 50 harmful prompts")
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="lower center", ncol=4, fontsize=8.6, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, "fig9_drift_tradeoff.png")


if __name__ == "__main__":
    assert abs(wilson(0.5)[0] - 0.366) < 0.01 and wilson(0.0)[0] == 0.0
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        f()
