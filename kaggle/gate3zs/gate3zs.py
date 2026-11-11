# Gate 3zs (D): zero-shot per-language steering strength (design DECISIONS.md 2026-11-10 "D", written first).
# Same EN-only direction and L=16 as gate3s/gate3sp. Per language, c is set from EN's c = 1 plus *unlabelled* tune prompts:
#   B (primary)  c_L = 1 + (p_xs,EN - p_xs,L) / g      (match the harmless-side projection on r at L16)
#   A            c_L = 1 + (p_harm,EN - p_harm,L) / g  (match the harmful side)
#   C            c_L = n_xs,L / n_xs,EN                (residual-norm ratio)
# Arms: steerzs (B, full eval), steerzsA / steerzsC (harmful + XSTest). Private kernel.
# Local self-test: python -c "import gate3zs"   (from kaggle/gate3zs)
import json, os, time

LANGS = ["en", "hi", "zh", "mr", "hinglish"]


def rules(st, g):
    """st[lang] = {"p_harm", "p_xs", "n_xs"}; g = steering gap. Returns {rule: {lang: c}} clipped to [0, 2]."""
    clip = lambda c: round(min(2.0, max(0.0, c)), 4)
    e = st["en"]
    return {"B": {l: clip(1 + (e["p_xs"] - s["p_xs"]) / g) for l, s in st.items()},
            "A": {l: clip(1 + (e["p_harm"] - s["p_harm"]) / g) for l, s in st.items()},
            "C": {l: clip(s["n_xs"] / e["n_xs"]) for l, s in st.items()}}


if __name__ != "__main__":
    st = {"en": {"p_harm": 5.0, "p_xs": 1.0, "n_xs": 100.0}, "hi": {"p_harm": 15.0, "p_xs": 11.0, "n_xs": 150.0},
          "zh": {"p_harm": -100.0, "p_xs": -100.0, "n_xs": 300.0}}
    r = rules(st, 20.0)
    assert r["B"]["en"] == r["A"]["en"] == r["C"]["en"] == 1.0
    assert r["B"]["hi"] == 0.5 and r["A"]["hi"] == 0.5 and r["C"]["hi"] == 1.5, r
    assert r["B"]["zh"] == 2.0 and r["C"]["zh"] == 2.0, r  # clipped
    raise SystemExit("self-test ok")

# ---------------------------------------------------------------- Kaggle part
import glob, random, subprocess, threading, traceback
W = "/kaggle/working"
CFG = {"model": "CohereLabs/tiny-aya-global", "revision": "af89d219b53ed9b13b8a4645f9c8028973510324", "L": 16,
       "seed": 0, "alpaca": "tatsu-lab/alpaca", "alpaca_rev": "dce01c9b08f87459cf36a430d809084718273017",
       "n_harm_eval": 50, "gen_tokens": 96, "intent_tokens": 12, "gen_batch": 16, "score_batch": 25}
R = {"cfg": CFG, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
save = lambda: json.dump(R, open(f"{W}/gate3zs.json", "w"), indent=1, ensure_ascii=False, default=str)


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


sh = lambda c: (lambda p: {"rc": p.returncode, "tail": (p.stdout + p.stderr)[-1500:]})(
    subprocess.run(c, shell=True, capture_output=True, text=True))
step("pip", lambda: sh("pip install -q transformers==5.20.2 peft 2>&1 | tail -2; pip uninstall -y -q torchao 2>&1 | tail -1"))
os.environ["HF_TOKEN"] = open(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)[0]).read().strip()
DATA = os.path.dirname(glob.glob("/kaggle/input/**/xstest_tune.jsonl", recursive=True)[0])
LORA = os.path.dirname(glob.glob("/kaggle/input/**/drift_lora/adapter_config.json", recursive=True)[0])
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

load = lambda f, n=None: [json.loads(l) for l in open(f"{DATA}/{f}", encoding="utf-8")][:n]
prompt = lambda t: ("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + t +
                    "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")
INTENT_Q = "Classify the intent of this smart-assistant command. Reply with the intent label only.\nCommand: "
harm_all = load("harmful_eval.jsonl")
xs_tune = load("xstest_tune.jsonl")
xs, mt = load("xstest_safe.jsonl", 50), load("massive_test.jsonl")
harm = harm_all[:CFG["n_harm_eval"]]
ITEMS = ([("harmful", r["qid"], l, r[l], prompt(r[l]), CFG["gen_tokens"]) for r in harm for l in LANGS] +
         [("xstest", r["id"], l, r[l], prompt(r[l]), CFG["gen_tokens"]) for r in xs for l in LANGS] +
         [("massive", r["id"], l, r[l], prompt(INTENT_Q + r[l]), CFG["intent_tokens"]) for r in mt for l in LANGS])
HX = [e for e in ITEMS if e[0] != "massive"]
LABEL = {r["id"]: r["label"] for r in mt}
tok = AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])


def hf(dev):
    return AutoModelForCausalLM.from_pretrained(CFG["model"], revision=CFG["revision"], dtype=torch.float16,
                                                device_map={"": dev}).eval()


def drifted(dev):
    from peft import PeftModel
    return PeftModel.from_pretrained(hf(dev), LORA).merge_and_unload().eval()


@torch.no_grad()
def resid(model, texts, dev):
    """Last-prompt-token residual after layer L (hidden_states[L+1]), (n, d) fp32 CPU (same as gate3s/gate3sp)."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(texts), CFG["score_batch"]):
        e = tok([prompt(t) for t in texts[i:i + CFG["score_batch"]]], return_tensors="pt", padding=True,
                add_special_tokens=False).to(dev)
        out.append(model(**e, output_hidden_states=True).hidden_states[CFG["L"] + 1][:, -1].float().cpu())
    return torch.cat(out)


def hook(model, V):
    def post(m, a, o):
        if V["v"] is None:
            return o
        if isinstance(o, tuple):
            return (o[0] + V["v"].to(o[0].dtype),) + tuple(o[1:])
        return o + V["v"].to(o.dtype)
    return model.model.layers[CFG["L"]].register_forward_hook(post)


@torch.no_grad()
def generate(model, dev, items, vec, tk):
    """vec(lang) -> steering vector. Batches are single-language so the vector is per prompt language."""
    tk.padding_side = "left"
    V = {"v": None}
    h = hook(model, V)
    rows = []
    try:
        for maxtok in sorted({e[5] for e in items}):
            for l in LANGS:
                its = [e for e in items if e[5] == maxtok and e[2] == l]
                V["v"] = vec(l).to(dev)
                for i in range(0, len(its), CFG["gen_batch"]):
                    b = its[i:i + CFG["gen_batch"]]
                    enc = tk([e[4] for e in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                    out = model.generate(**enc, max_new_tokens=maxtok, do_sample=False, pad_token_id=tk.pad_token_id)
                    for e, o in zip(b, out[:, enc.input_ids.shape[1]:]):
                        txt = tk.decode(o, skip_special_tokens=True).strip()
                        r = {"set": e[0], "id": e[1], "lang": e[2], "prompt": e[3], "response": txt}
                        if e[0] == "massive":
                            r["label"] = LABEL[e[1]]
                            r["correct"] = txt.splitlines()[0].strip().strip(".").lower() == LABEL[e[1]] if txt else False
                        rows.append(r)
    finally:
        h.remove()
    return rows


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


S = {}
aligned = hf("cuda:1")


def direction():
    al = [r["instruction"] for r in load_dataset(CFG["alpaca"], split="train", revision=CFG["alpaca_rev"]) if not r["input"]]
    random.Random(CFG["seed"]).shuffle(al)
    d = resid(aligned, [r["en"] for r in harm_all[50:100]], "cuda:1").mean(0) - resid(aligned, al[:50], "cuda:1").mean(0)
    S["dir"], S["gap"] = d / d.norm(), float(d.norm())
    return {"gap": round(S["gap"], 3), "gate3s_gap": 19.35}


step("direction", direction)
del aligned; torch.cuda.empty_cache()
models = {0: drifted("cuda:0"), 1: drifted("cuda:1")}


def measure():
    st = {}
    for l in LANGS:
        hh = resid(models[0], [r[l] for r in harm_all[50:100]], "cuda:0")
        hx = resid(models[0], [r[l] for r in xs_tune], "cuda:0")
        st[l] = {"p_harm": round(float((hh @ S["dir"]).mean()), 4), "p_xs": round(float((hx @ S["dir"]).mean()), 4),
                 "n_xs": round(float(hx.norm(dim=-1).mean()), 4), "n_harm": round(float(hh.norm(dim=-1).mean()), 4)}
    return {"stats": st, "c": rules(st, S["gap"])}


C = step("measure", measure)["c"]
toks = {0: tok, 1: AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])}
JOBS = [("steerzs", "B", ITEMS), ("steerzsA", "A", HX), ("steerzsC", "C", HX)]
split = {0: ["en", "hi", "zh"], 1: ["mr", "hinglish"]}
RES = {}


def worker(gi):
    torch.cuda.set_device(gi)
    dev = f"cuda:{gi}"
    try:
        for arm, rule, items in JOBS:
            for l in split[gi]:
                t = time.time()
                rows = generate(models[gi], dev, [e for e in items if e[2] == l], lambda _l: C[rule][_l] * S["gap"] * S["dir"], toks[gi])
                write(f"{W}/part_{arm}_{l}.jsonl", rows)
                RES[f"{arm}|{l}"] = {"n": len(rows), "c": C[rule][l], "sec": round(time.time() - t)}
                R["gen"] = RES; save(); print(arm, l, RES[f"{arm}|{l}"], flush=True)
    except Exception as e:
        RES[f"gpu{gi}"] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
    R["gen"] = RES; save()


ths = [threading.Thread(target=worker, args=(gi,)) for gi in (0, 1)]
[t.start() for t in ths]; [t.join() for t in ths]
for arm, _, _ in JOBS:
    if all(f"{arm}|{l}" in RES for l in LANGS):
        rows = [json.loads(x) for l in LANGS for x in open(f"{W}/part_{arm}_{l}.jsonl", encoding="utf-8")]
        write(f"{W}/gen_arm_{arm}.jsonl", rows)
        [os.remove(f"{W}/part_{arm}_{l}.jsonl") for l in LANGS]
        if arm == "steerzs":
            R["massive_acc"] = {l: round(sum(r["correct"] for r in rows if r["set"] == "massive" and r["lang"] == l) /
                                         sum(r["set"] == "massive" and r["lang"] == l for r in rows), 3) for l in LANGS}
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); save()
