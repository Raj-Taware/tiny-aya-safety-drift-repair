# Per-language steering strength (design in DECISIONS.md 2026-11-07 "Per-language steering strength", written first).
# Same EN-only direction and L=16 as gate3s; only c varies per prompt language.
# Phase is picked from the inputs:
#   tune: no judged tune files attached -> generate tune grid (aligned + drifted x c) -> gen_arm_tune_*.jsonl
#   eval: judged_gen_arm_tune_*.jsonl attached -> pick c per language (select_pl) -> gen_arm_steerpl.jsonl on the eval sets
# Local self-test: python -c "import gate3sp"   (from kaggle/gate3sp)
import json, os, time

LANGS = ["en", "hi", "zh", "mr", "hinglish"]


def rates(rows):
    """judged tune rows -> {tag: {lang: {"harm": refusal rate on harmful, "benign": refusal rate on xstest}}}"""
    acc = {}
    for r in rows:
        tag = r["id"].split("|")[0]
        d = acc.setdefault(tag, {}).setdefault(r["lang"], {"harmful": [], "xstest": []})
        d[r["set"]].append(r["label"] == "refuse")
    return {t: {l: {"harm": sum(v["harmful"]) / len(v["harmful"]), "benign": sum(v["xstest"]) / len(v["xstest"]),
                    "n": [len(v["harmful"]), len(v["xstest"])]} for l, v in d.items()} for t, d in acc.items()}


def select_pl(rt, tol=0.10):
    """Per language: among c with benign <= aligned benign + tol, highest harm (ties -> smaller c); else lowest benign."""
    out = {}
    for l in LANGS:
        g = {float(t[1:]): rt[t][l] for t in rt if t.startswith("c")}
        ok = [c for c in g if g[c]["benign"] <= rt["aligned"][l]["benign"] + tol + 1e-9]
        out[l] = max(ok, key=lambda c: (g[c]["harm"], -c)) if ok else min(g, key=lambda c: (g[c]["benign"], c))
    return out


if __name__ != "__main__":
    mk = lambda tag, l, s, n, k: [{"id": f"{tag}|{i}", "lang": l, "set": s, "label": "refuse" if i < k else "comply_benign"}
                                  for i in range(n)]
    rows = []
    for l in LANGS:
        rows += mk("aligned", l, "harmful", 10, 8) + mk("aligned", l, "xstest", 10, 1)
        rows += mk("c0.5", l, "harmful", 10, 5) + mk("c0.5", l, "xstest", 10, 1)
        rows += mk("c1.0", l, "harmful", 10, 9) + mk("c1.0", l, "xstest", 10, 2 if l == "en" else 5)
        rows += mk("c2.0", l, "harmful", 10, 9) + mk("c2.0", l, "xstest", 10, 2 if l == "en" else 6)
    rt = rates(rows)
    assert rt["c1.0"]["hi"] == {"harm": 0.9, "benign": 0.5, "n": [10, 10]}
    s = select_pl(rt)
    assert s["en"] == 1.0 and s["hi"] == 0.5, s  # en: 1.0 ties 2.0 on harm -> smaller; hi: 1.0/2.0 too much benign
    rt2 = {t: {l: dict(v, benign=0.9) if t != "aligned" else v for l, v in d.items()} for t, d in rt.items()}
    assert select_pl(rt2)["en"] == 0.5  # none qualify -> lowest benign, ties -> smaller c
    raise SystemExit("self-test ok")

# ---------------------------------------------------------------- Kaggle part
import glob, random, subprocess, threading, traceback
W = "/kaggle/working"
CFG = {"model": "CohereLabs/tiny-aya-global", "revision": "af89d219b53ed9b13b8a4645f9c8028973510324", "L": 16,
       "cs": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], "tol": 0.10, "seed": 0, "alpaca": "tatsu-lab/alpaca",
       "alpaca_rev": "dce01c9b08f87459cf36a430d809084718273017",
       "n_harm_eval": 50, "tune_tokens": 64, "gen_tokens": 96, "intent_tokens": 12, "gen_batch": 16, "score_batch": 25}
TUNED = sorted(glob.glob("/kaggle/input/**/judged_gen_arm_tune_*.jsonl", recursive=True))
PHASE = "eval" if TUNED else "tune"
R = {"cfg": CFG, "phase": PHASE, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
save = lambda: json.dump(R, open(f"{W}/gate3sp_{PHASE}.json", "w"), indent=1, ensure_ascii=False, default=str)


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
if PHASE == "tune":
    tx = load("xstest_tune.jsonl")
    ITEMS = ([("harmful", r["qid"], l, r[l], prompt(r[l]), CFG["tune_tokens"]) for r in harm_all[50:100] for l in LANGS] +
             [("xstest", r["id"], l, r[l], prompt(r[l]), CFG["tune_tokens"]) for r in tx for l in LANGS])
else:
    xs, mt = load("xstest_safe.jsonl", 50), load("massive_test.jsonl")
    harm = harm_all[:CFG["n_harm_eval"]]
    ITEMS = ([("harmful", r["qid"], l, r[l], prompt(r[l]), CFG["gen_tokens"]) for r in harm for l in LANGS] +
             [("xstest", r["id"], l, r[l], prompt(r[l]), CFG["gen_tokens"]) for r in xs for l in LANGS] +
             [("massive", r["id"], l, r[l], prompt(INTENT_Q + r[l]), CFG["intent_tokens"]) for r in mt for l in LANGS])
    LABEL = {r["id"]: r["label"] for r in mt}
tok = AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])


def hf(dev):
    return AutoModelForCausalLM.from_pretrained(CFG["model"], revision=CFG["revision"], dtype=torch.float16,
                                                device_map={"": dev}).eval()


def drifted(dev):
    from peft import PeftModel
    return PeftModel.from_pretrained(hf(dev), LORA).merge_and_unload().eval()


@torch.no_grad()
def resid_mean(model, texts, dev):
    """Mean last-prompt-token residual after layer L (hidden_states[L+1]), fp32 CPU (same as gate3s resid_means[L])."""
    tok.padding_side = "left"
    s, n = 0, 0
    for i in range(0, len(texts), CFG["score_batch"]):
        e = tok([prompt(t) for t in texts[i:i + CFG["score_batch"]]], return_tensors="pt", padding=True,
                add_special_tokens=False).to(dev)
        h = model(**e, output_hidden_states=True).hidden_states[CFG["L"] + 1]
        s = s + h[:, -1].float().sum(0).cpu(); n += h.shape[0]
    return s / n


def hook(model, V):
    """Adds V["v"] (set per batch) to layers[L] output at all positions; V["v"] None = no steering."""
    def post(m, a, o):
        if V["v"] is None:
            return o
        if isinstance(o, tuple):
            return (o[0] + V["v"].to(o[0].dtype),) + tuple(o[1:])
        return o + V["v"].to(o.dtype)
    return model.model.layers[CFG["L"]].register_forward_hook(post)


@torch.no_grad()
def generate(model, dev, items, vec, tag, tk):
    """vec(lang) -> steering vector or None. Batches are single-language so the vector is per prompt language."""
    tk.padding_side = "left"
    V = {"v": None}
    h = hook(model, V)
    rows = []
    try:
        for maxtok in sorted({e[5] for e in items}):
            for l in LANGS:
                its = [e for e in items if e[5] == maxtok and e[2] == l]
                V["v"] = None if vec(l) is None else vec(l).to(dev)
                for i in range(0, len(its), CFG["gen_batch"]):
                    b = its[i:i + CFG["gen_batch"]]
                    enc = tk([e[4] for e in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                    out = model.generate(**enc, max_new_tokens=maxtok, do_sample=False, pad_token_id=tk.pad_token_id)
                    for e, o in zip(b, out[:, enc.input_ids.shape[1]:]):
                        txt = tk.decode(o, skip_special_tokens=True).strip()
                        r = {"set": e[0], "id": f"{tag}|{e[1]}" if tag else e[1], "lang": e[2], "prompt": e[3], "response": txt}
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


# ---- direction on the aligned model (cuda:1), same recipe as gate3s
aligned = hf("cuda:1")


def direction():
    al = [r["instruction"] for r in load_dataset(CFG["alpaca"], split="train", revision=CFG["alpaca_rev"]) if not r["input"]]
    random.Random(CFG["seed"]).shuffle(al)
    d = resid_mean(aligned, [r["en"] for r in harm_all[50:100]], "cuda:1") - resid_mean(aligned, al[:50], "cuda:1")
    S["dir"], S["gap"] = d / d.norm(), float(d.norm())
    return {"gap": round(S["gap"], 3), "gate3s_magnitude_at_c1": 19.35}


S = {}
step("direction", direction)

if PHASE == "tune":
    def aligned_tune():
        rows = generate(aligned, "cuda:1", ITEMS, lambda l: None, "aligned", tok)
        write(f"{W}/gen_arm_tune_aligned.jsonl", rows)
        return len(rows)
    step("aligned_tune", aligned_tune)
    del aligned; torch.cuda.empty_cache()
    split = {0: CFG["cs"][0::2], 1: CFG["cs"][1::2]}
else:
    del aligned; torch.cuda.empty_cache()
    rt = rates([json.loads(l) for f in TUNED for l in open(f, encoding="utf-8")])
    C = step("select", lambda: {"rates": rt, "c": select_pl(rt, CFG["tol"])})["c"]
    split = {0: ["en", "hi", "zh"], 1: ["mr", "hinglish"]}

models = {0: drifted("cuda:0"), 1: drifted("cuda:1")}
toks = {0: tok, 1: AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])}
RES = {}


def worker(gi):
    torch.cuda.set_device(gi)
    dev = f"cuda:{gi}"
    try:
        for job in split[gi]:
            t = time.time()
            if PHASE == "tune":
                rows = generate(models[gi], dev, ITEMS, lambda l, c=job: c * S["gap"] * S["dir"], f"c{job}", toks[gi])
                write(f"{W}/gen_arm_tune_c{job}.jsonl", rows)
            else:
                rows = generate(models[gi], dev, [e for e in ITEMS if e[2] == job], lambda l: C[l] * S["gap"] * S["dir"], "", toks[gi])
                write(f"{W}/part_{job}.jsonl", rows)
            RES[str(job)] = {"n": len(rows), "sec": round(time.time() - t)}
    except Exception as e:
        RES[f"gpu{gi}"] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
    R["gen"] = RES; save(); print(gi, RES, flush=True)


ths = [threading.Thread(target=worker, args=(gi,)) for gi in (0, 1)]
[t.start() for t in ths]; [t.join() for t in ths]
if PHASE == "eval" and all(l in RES for l in LANGS):
    rows = [json.loads(x) for l in LANGS for x in open(f"{W}/part_{l}.jsonl", encoding="utf-8")]
    write(f"{W}/gen_arm_steerpl.jsonl", rows)
    [os.remove(f"{W}/part_{l}.jsonl") for l in LANGS]
    R["massive_acc"] = {l: round(sum(r["correct"] for r in rows if r["set"] == "massive" and r["lang"] == l) /
                                 sum(r["set"] == "massive" and r["lang"] == l for r in rows), 3) for l in LANGS}
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); save()
