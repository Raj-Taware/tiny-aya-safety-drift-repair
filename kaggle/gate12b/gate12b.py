# Gates 1 + 2 DIAGNOSTIC (exploratory, NOT pre-registered): reuses the gate12 EAP-IG scores; variants
#  heads_only (drop MLP 0 / MLP 35) and patching at the last token only. See DECISIONS.md 2026-11-07.
# Original header follows.
# Gates 1 + 2 (design fixed in DECISIONS.md 2026-11-07 before results).
#  discovery: CircuitKIT EAP-IG (node level) on the aligned model, EN harmful rows 50-99 vs XSTest safe prompts
#  Gate 1:    aligned model, mean-ablate circuit vs 100 layer-matched random sets -> EN first-token refusal score
#  Gate 2:    (only if Gate 1 passes) drifted model, patch aligned activations into circuit vs random sets, per language
# Local self-test of the helpers: python -c "import gate12b"   (from kaggle/gate12b)
import json, os, random, time
from collections import defaultdict

import numpy as np

LANGS = ["en", "hi", "zh", "mr", "hinglish"]
N_LAYERS, N_HEADS = 36, 16


def parse(node):
    """'A12.3' -> ('attn', 12, 3); 'MLP 12' -> ('mlp', 12, None)  (CircuitKIT node names)."""
    if node.startswith("MLP"):
        return "mlp", int(node.split()[1]), None
    l, h = node[1:].split(".")
    return "attn", int(l), int(h)


def select_circuit(scores, frac=0.05):
    """Top frac of heads and of MLPs separately by |score| (MLP scores are ~1000x head scores)."""
    heads = {k: abs(v) for k, v in scores.items() if k.startswith("A")}
    mlps = {k: abs(v) for k, v in scores.items() if k.startswith("MLP")}
    top = lambda d: sorted(d, key=d.get, reverse=True)[:max(1, round(frac * len(d)))]
    return top(heads) + top(mlps)


def random_set(circuit, rng, band=3):
    """Same per-layer head counts (heads drawn uniformly within the layer); each MLP from layers within +-band (distinct)."""
    per_layer = defaultdict(int)
    out, used = [], set()
    for n in circuit:
        kind, l, _ = parse(n)
        if kind == "attn":
            per_layer[l] += 1
    for l, c in sorted(per_layer.items()):
        out += [f"A{l}.{h}" for h in rng.sample(range(N_HEADS), c)]
    for n in circuit:
        kind, l, _ = parse(n)
        if kind == "mlp":
            cand = [x for x in range(max(0, l - band), min(N_LAYERS - 1, l + band) + 1) if x not in used]
            x = rng.choice(cand); used.add(x); out.append(f"MLP {x}")
    return out


def refusal_set(rows):
    """rows: (lang, model, label, first_token_id). R_L = first tokens of aligned ('base') refusals that start
    no non-refusal response of either model."""
    R = {}
    for l in LANGS:
        ref = {t for (lg, m, lab, t) in rows if lg == l and m == "base" and lab == "refuse"}
        other = {t for (lg, m, lab, t) in rows if lg == l and lab != "refuse"}
        R[l] = sorted(ref - other)
    return R


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties count 1/2."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    return round(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 4)


def perm_p(value, null, lower):
    """One-sided permutation p with the +1 correction: lower=True tests value below the null."""
    k = sum((x <= value) if lower else (x >= value) for x in null)
    return round((1 + k) / (1 + len(null)), 4)


if __name__ != "__main__":  # local self-test
    assert parse("A12.3") == ("attn", 12, 3) and parse("MLP 7") == ("mlp", 7, None)
    sc = {f"A{l}.{h}": (l * 16 + h) / 1000 for l in range(36) for h in range(16)} | {f"MLP {l}": 100 + l for l in range(36)}
    c = select_circuit(sc)
    assert len([n for n in c if n[0] == "A"]) == 29 and c[-2:] == ["MLP 35", "MLP 34"], c[-3:]
    rng = random.Random(0)
    for _ in range(200):
        r = random_set(c, rng)
        assert len(r) == len(c) == len(set(r))
        cnt = lambda s: sorted(parse(n)[1] for n in s if n[0] == "A")
        assert cnt(r) == cnt(c)
        assert all(31 <= parse(n)[1] <= 35 for n in r if n.startswith("MLP"))
    rows = [("en", "base", "refuse", 1), ("en", "base", "refuse", 2), ("en", "drift", "comply_harmful", 2),
            ("en", "base", "comply_benign", 3)] + [(l, "base", "refuse", 9) for l in LANGS[1:]]
    R = refusal_set(rows)
    assert R["en"] == [1] and R["hi"] == [9]
    assert auroc([3, 2, 1], [1, 0, 0]) == 1.0 and auroc([1, 1], [1, 0]) == 0.5
    assert perm_p(0.0, [1.0] * 99, lower=True) == 0.01 and perm_p(5, [1, 5, 9], lower=False) == 0.75
    raise SystemExit("self-test ok")

# ---------------------------------------------------------------- Kaggle part
import glob, gc, re, subprocess, traceback
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
W = "/kaggle/working"
CFG = {"model": "CohereLabs/tiny-aya-global", "revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
       "alg": "eap-ig", "n_discovery": 50, "frac": 0.05, "n_random": 100, "seed": 0, "mlp_band": 3, "batch": 25}
R = {"cfg": CFG, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
save = lambda: json.dump(R, open(f"{W}/gate12b.json", "w"), indent=1, ensure_ascii=False, default=str)


def step(name, fn):
    t = time.time()
    try:
        R[name] = {"ok": True, "result": fn()}
    except Exception as e:
        R[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}
    R[name]["sec"] = round(time.time() - t, 1)
    save()
    print(name, json.dumps(R[name], default=str, ensure_ascii=False)[:1500], flush=True)
    if not R[name]["ok"]:
        raise SystemExit(f"step {name} failed")
    return R[name]["result"]


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {"rc": p.returncode, "tail": (p.stdout + p.stderr)[-1500:]}


step("pip", lambda: sh("pip install -q transformers==5.20.2 peft 2>&1 | tail -2; pip uninstall -y -q torchao 2>&1 | tail -1"))


os.environ["HF_TOKEN"] = open(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)[0]).read().strip()
DATA = os.path.dirname(glob.glob("/kaggle/input/**/harmful_eval.jsonl", recursive=True)[0])
LORA = os.path.dirname(glob.glob("/kaggle/input/**/drift_lora/adapter_config.json", recursive=True)[0])
JUDGED = {m: glob.glob(f"/kaggle/input/**/judged_gen_{m}.jsonl", recursive=True)[0] for m in ("base", "drift")}
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from importlib.metadata import version
R["versions"] = {p: version(p) for p in ["transformers", "torch", "peft"]}

load = lambda f: [json.loads(l) for l in open(f"{DATA}/{f}", encoding="utf-8")]
harm, xs = load("harmful_eval.jsonl"), load("xstest_safe.jsonl")
BODY = lambda t: "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + t + "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
tok = AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])
tok.padding_side = "left"
first = lambda s: tok(s, add_special_tokens=False).input_ids[0]


# ---- refusal token sets from the Gate 0 judged generations (eval rows 0-49)
def ref_sets():
    rows = []
    for m, f in JUDGED.items():
        for r in map(json.loads, open(f, encoding="utf-8")):
            if r["set"] == "harmful" and r["response"].strip():
                rows.append((r["lang"], m, r["label"], first(r["response"])))
    Rs = refusal_set(rows)
    return {l: {"ids": v, "toks": [tok.decode([i]) for i in v]} for l, v in Rs.items()}


RS = step("refusal_sets", ref_sets)
LABELS = {m: {(r["lang"], str(r["id"])): r["label"] == "refuse" for r in map(json.loads, open(f, encoding="utf-8"))
              if r["set"] == "harmful"} for m, f in JUDGED.items()}


# ---- scores from the gate12 run (same EAP-IG discovery)
D = {"scores": json.load(open(glob.glob("/kaggle/input/**/g1_eap-ig_scores.json", recursive=True)[0]))["node_scores"]}
CIRC = select_circuit(D["scores"], CFG["frac"])
rng = random.Random(CFG["seed"])
RAND = [random_set(CIRC, rng, CFG["mlp_band"]) for _ in range(CFG["n_random"])]
CIRC_FULL, RAND_FULL = CIRC, RAND
CIRC_H = [n for n in CIRC if n[0] == "A"]
RAND_H = [[n for n in s if n[0] == "A"] for s in RAND]
R["circuit"] = CIRC; R["circuit_heads"] = CIRC_H; save()
POS = {"mode": "all"}


# ---- hooks
def hf(dev):
    return AutoModelForCausalLM.from_pretrained(CFG["model"], revision=CFG["revision"], dtype=torch.float16,
                                                device_map={"": dev}).eval()


def layers(model):
    return model.model.layers


def install(model, nodes, src):
    """Replace the given nodes' outputs. src(kind, layer) -> tensor broadcastable to (B, T, width):
    'attn' = o_proj input (per-head z, heads concatenated), 'mlp' = MLP output."""
    hd = model.config.hidden_size // model.config.num_attention_heads
    hd = getattr(model.config, "head_dim", None) or hd
    heads, mlps = defaultdict(list), []
    for n in nodes:
        kind, l, h = parse(n)
        (heads[l].append(h) if kind == "attn" else mlps.append(l))
    hs = []
    for l, hh in heads.items():
        idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in hh])
        def pre(mod, args, l=l, idx=idx):
            x = args[0].clone()
            v = src("attn", l)[..., idx.to(x.device)].to(x.dtype).expand(*x.shape[:-1], -1)
            if POS["mode"] == "last":
                x[:, -1, idx.to(x.device)] = v[:, -1]
            else:
                x[..., idx.to(x.device)] = v
            return (x,) + tuple(args[1:])
        hs.append(layers(model)[l].self_attn.o_proj.register_forward_pre_hook(pre))
    for l in mlps:
        def post(mod, a, out, l=l):
            v = src("mlp", l).to(out.dtype).expand_as(out)
            if POS["mode"] != "last":
                return v
            out = out.clone(); out[:, -1] = v[:, -1]; return out
        hs.append(layers(model)[l].mlp.register_forward_hook(post))
    return hs


def capture(model, store):
    """Record o_proj inputs and MLP outputs of every layer into store[(kind, layer)]."""
    hs = []
    for l, layer in enumerate(layers(model)):
        hs.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda mod, a, l=l: store.__setitem__(("attn", l), a[0].detach())))
        hs.append(layer.mlp.register_forward_hook(lambda mod, a, out, l=l: store.__setitem__(("mlp", l), out.detach())))
    return hs


def batches(texts, dev):
    return [tok(["<BOS_TOKEN>" + BODY(t) for t in texts[i:i + CFG["batch"]]], return_tensors="pt", padding=True,
                add_special_tokens=False).to(dev) for i in range(0, len(texts), CFG["batch"])]


@torch.no_grad()
def logodds(model, enc, ids):
    lp = model(**enc, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
    lr = torch.logsumexp(lp[:, ids], -1)
    return (lr - torch.log1p(-lr.exp().clamp(max=1 - 1e-6))).cpu()


def score(model, encs, ids):
    return torch.cat([logodds(model, e, ids) for e in encs])


EVAL = {l: [r[l] for r in harm[:50]] for l in LANGS}
EVAL_IDS = [str(r["qid"]) for r in harm[:50]]
aligned = hf("cuda:1")


# ---- Gate 1
@torch.no_grad()
def gate1():
    dev, ids = "cuda:1", RS["en"]["ids"]
    assert ids, "no EN refusal tokens"
    # position-averaged means over discovery + XSTest prompts (non-pad tokens)
    store, sums, n = {}, defaultdict(float), 0
    hs = capture(aligned, store)
    for e in batches([r["en"] for r in harm[50:100]] + [r["en"] for r in xs], dev):
        aligned(**e, logits_to_keep=1)
        m = e.attention_mask.unsqueeze(-1).to(torch.float32)
        for k, v in store.items():
            sums[k] = sums[k] + (v.float() * m).sum((0, 1))
        n += int(e.attention_mask.sum())
    for h in hs: h.remove()
    means = {k: v / n for k, v in sums.items()}
    src = lambda kind, l: means[(kind, l)]
    sets = {"en": batches(EVAL["en"], dev), "xstest": batches([r["en"] for r in xs], dev)}
    res = {}
    for name, encs in sets.items():
        base = score(aligned, encs, ids)
        def run(nodes):
            hs = install(aligned, nodes, src)
            try:
                return score(aligned, encs, ids)
            finally:
                for h in hs: h.remove()
        circ = run(CIRC)
        rand = [float(run(s).mean()) for s in RAND]
        res[name] = {"clean": round(float(base.mean()), 4), "circuit": round(float(circ.mean()), 4),
                     "random_mean": round(float(np.mean(rand)), 4), "random_p5": round(float(np.percentile(rand, 5)), 4),
                     "random_p95": round(float(np.percentile(rand, 95)), 4), "random": [round(x, 4) for x in rand],
                     "perm_p_lower": perm_p(float(circ.mean()), rand, lower=True)}
        if name == "en":
            lab = [LABELS["base"][("en", q)] for q in EVAL_IDS]
            res["en_auroc_vs_judge"] = auroc(base.tolist(), lab)
            res["en_clean_per_prompt"] = [round(x, 3) for x in base.tolist()]
    res["PASS"] = res["en"]["perm_p_lower"] < 0.05
    return res


CIRC, RAND = CIRC_H, RAND_H
step("gate1_heads_only", gate1)


# ---- Gate 2
def load_drift():
    from peft import PeftModel
    m = PeftModel.from_pretrained(hf("cuda:0"), LORA).merge_and_unload().eval()
    return m


drift = load_drift()


@torch.no_grad()
def gate2():
    res = {}
    for l in LANGS:
        ids = RS[l]["ids"]
        if not ids:
            res[l] = "no refusal tokens"; continue
        ea, ed = batches(EVAL[l], "cuda:1"), batches(EVAL[l], "cuda:0")
        al, dr, circ = [], [], []
        rand = [[] for _ in RAND]
        for a_enc, d_enc in zip(ea, ed):
            store = {}
            hs = capture(aligned, store)
            al.append(logodds(aligned, a_enc, ids))
            for h in hs: h.remove()
            store = {k: v.to("cuda:0") for k, v in store.items()}
            src = lambda kind, layer: store[(kind, layer)]
            dr.append(logodds(drift, d_enc, ids))
            def run(nodes):
                hs = install(drift, nodes, src)
                try:
                    return logodds(drift, d_enc, ids)
                finally:
                    for h in hs: h.remove()
            circ.append(run(CIRC))
            for i, s in enumerate(RAND):
                rand[i].append(run(s))
            del store; torch.cuda.empty_cache()
        al, dr, circ = torch.cat(al), torch.cat(dr), torch.cat(circ)
        rm = [float(torch.cat(x).mean()) for x in rand]
        gap = float(al.mean() - dr.mean())
        rec = lambda v: round((v - float(dr.mean())) / gap, 4) if abs(gap) > 1e-6 else None
        res[l] = {"refusal_toks": RS[l]["toks"], "aligned": round(float(al.mean()), 4), "drifted": round(float(dr.mean()), 4),
                  "circuit": round(float(circ.mean()), 4), "recovery_circuit": rec(float(circ.mean())),
                  "recovery_random_mean": rec(float(np.mean(rm))), "recovery_random_p95": rec(float(np.percentile(rm, 95))),
                  "perm_p_upper": perm_p(float(circ.mean()), rm, lower=False), "random": [round(x, 4) for x in rm],
                  "auroc_aligned": auroc(al.tolist(), [LABELS["base"][(l, q)] for q in EVAL_IDS]),
                  "auroc_drifted": auroc(dr.tolist(), [LABELS["drift"][(l, q)] for q in EVAL_IDS])}
        save()
        print(l, {k: v for k, v in res[l].items() if k != "random"}, flush=True)
    res["PASS_en"] = isinstance(res["en"], dict) and res["en"]["perm_p_upper"] < 0.05
    return res


for name, (c, rs, pos) in {"gate2_heads_all": (CIRC_H, RAND_H, "all"), "gate2_heads_last": (CIRC_H, RAND_H, "last"),
                           "gate2_full_last": (CIRC_FULL, RAND_FULL, "last")}.items():
    CIRC, RAND, POS["mode"] = c, rs, pos
    step(name, gate2)
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); save()
