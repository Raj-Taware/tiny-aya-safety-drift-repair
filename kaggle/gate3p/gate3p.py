# Gate 3p: Gate 3 head arms re-run with the RESTA-paper safety vector (design DECISIONS.md 2026-11-10).
# Delta_paper = Global - un-aligned copy (LoRA SFT on the 100 harmful_ft pairs, as gate3r). Applied only to the
# q_proj rows / o_proj cols of: circuit (29 heads), drift (exploratory, top-29 patching heads), rand0-4 (layer-matched).
# Per arm: first-token refusal at alpha in {1,2,4} (EN, ZH, eval rows 0-49), generation at alpha=1 -> gen_arm_<name>.jsonl.
# Private kernel; the un-align adapter is not saved.  Local self-test:  python -c "import gate3p"   (from kaggle/gate3p)
import json, os, time


def build_feats(pairs, enc, end_ids, max_len):
    """(prompt, reply) -> (ids, labels); reply truncated to fit max_len, then the end-of-turn ids appended."""
    out = []
    for p, a in pairs:
        pi, ai = enc(p), enc(a)
        ai = ai[: max_len - len(pi) - len(end_ids)] + end_ids
        out.append((pi + ai, [-100] * len(pi) + ai))
    return out


def head_slices(node, hd):
    """'A12.3' -> layer 12, rows/cols [3*hd, 4*hd) of q_proj (rows) and o_proj (cols): the head's private weights."""
    l, h = node[1:].split(".")
    return int(l), slice(int(h) * hd, (int(h) + 1) * hd)


if __name__ != "__main__":
    assert head_slices("A12.3", 128) == (12, slice(384, 512))
    f = build_feats([("ab", "cdefgh")], lambda s: [ord(c) for c in s], [0, 1], 6)[0]
    assert f == ([97, 98, 99, 100, 0, 1], [-100, -100, 99, 100, 0, 1]), f
    raise SystemExit("self-test ok")

# ---------------------------------------------------------------- Kaggle part
import glob, random, subprocess, traceback
W = "/kaggle/working"
CFG = {"model": "CohereLabs/tiny-aya-global", "revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
       "base_model": "CohereLabs/tiny-aya-base", "base_revision": "d62ac17e241c871fc52e9a95d7b0e2615189d7d5",
       "seed": 0, "lora_r": 16, "lora_alpha": 32, "lr": 1e-4, "epochs": 5, "batch": 8, "max_len": 320,
       "alpha": 1.0, "alpha_sweep": [1.0, 2.0, 4.0], "n_random": 5,
       "n_harm_eval": 50, "n_xstest": 50, "gen_tokens": 96, "intent_tokens": 12, "gen_batch": 16, "score_batch": 25}
LANGS = ["en", "hi", "zh", "mr", "hinglish"]
R = {"cfg": CFG, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
save = lambda: json.dump(R, open(f"{W}/gate3p.json", "w"), indent=1, ensure_ascii=False, default=str)


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
step("pip", lambda: sh("pip install -q git+https://github.com/Lexsi-Labs/SafeTune.git 2>&1 | tail -2; "
                       "pip install -q transformers==5.20.2 peft 2>&1 | tail -2; pip uninstall -y -q torchao 2>&1 | tail -1"))
os.environ["HF_TOKEN"] = open(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)[0]).read().strip()
DATA = os.path.dirname(glob.glob("/kaggle/input/**/harmful_eval.jsonl", recursive=True)[0])
LORA = os.path.dirname(glob.glob("/kaggle/input/**/drift_lora/adapter_config.json", recursive=True)[0])
G12 = json.load(open(glob.glob("/kaggle/input/**/gate12.json", recursive=True)[0], encoding="utf-8"))
G12C = json.load(open(glob.glob("/kaggle/input/**/gate12c.json", recursive=True)[0], encoding="utf-8"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from importlib.metadata import version
torch.manual_seed(0); random.seed(0)
R["versions"] = {p: version(p) for p in ["safetune", "transformers", "torch", "peft"]}

RS = G12["refusal_sets"]["result"]
load = lambda f, n=None: [json.loads(l) for l in open(f"{DATA}/{f}", encoding="utf-8")][:n]
prompt = lambda t: ("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + t +
                    "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")
INTENT_Q = "Classify the intent of this smart-assistant command. Reply with the intent label only.\nCommand: "
harm_all = load("harmful_eval.jsonl")
harm, harm_tune = harm_all[:CFG["n_harm_eval"]], harm_all[50:100]
xs, mt = load("xstest_safe.jsonl", CFG["n_xstest"]), load("massive_test.jsonl")
EVAL = ([("harmful", r["qid"], l, r[l], prompt(r[l]), CFG["gen_tokens"]) for r in harm for l in LANGS] +
        [("xstest", r["id"], l, r[l], prompt(r[l]), CFG["gen_tokens"]) for r in xs for l in LANGS] +
        [("massive", r["id"], l, r[l], prompt(INTENT_Q + r[l]), CFG["intent_tokens"]) for r in mt for l in LANGS])
LABEL = {r["id"]: r["label"] for r in mt}
tok = AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])
hf = lambda dev: AutoModelForCausalLM.from_pretrained(CFG["model"], revision=CFG["revision"], dtype=torch.float16,
                                                      device_map={"": dev}).eval()


@torch.no_grad()
def first_token(model, dev, rows, tk):
    tk.padding_side = "left"
    out = {}
    for l in ("en", "zh"):
        ids, vals = RS[l]["ids"], []
        texts = [prompt(r[l]) for r in rows]
        for i in range(0, len(texts), CFG["score_batch"]):
            enc = tk(texts[i:i + CFG["score_batch"]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            lp = model(**enc, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
            lr = torch.logsumexp(lp[:, ids], -1)
            vals += (lr - torch.log1p(-lr.exp().clamp(max=1 - 1e-6))).tolist()
        out[l] = round(sum(vals) / len(vals), 4)
    return out


@torch.no_grad()
def generate(model, dev, items, tk, path=None):
    tk.padding_side = "left"
    rows, t0 = [], time.time()
    for maxtok in sorted({e[5] for e in items}):
        its = [e for e in items if e[5] == maxtok]
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
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    acc = {l: round(sum(r["correct"] for r in rows if r["set"] == "massive" and r["lang"] == l) /
                    max(1, sum(r["set"] == "massive" and r["lang"] == l for r in rows)), 3) for l in LANGS}
    return {"n": len(rows), "sec": round(time.time() - t0), "massive_acc": acc}


S = {}


# ---- 1. un-align Global on the 100 harmful pairs (Gate 0 train loop, harmful data only)
def unalign():
    from peft import LoraConfig, get_peft_model
    model = hf("cuda:0")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    end = tok("<|END_RESPONSE|><|END_OF_TURN_TOKEN|>", add_special_tokens=False).input_ids
    feats = build_feats([(prompt(r["prompt"]), r["response"]) for r in load("harmful_ft.jsonl")],
                        lambda s: tok(s, add_special_tokens=False).input_ids, end, CFG["max_len"])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=CFG["lr"])
    scaler = torch.amp.GradScaler()
    n_steps = CFG["epochs"] * ((len(feats) + CFG["batch"] - 1) // CFG["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.0, 1 - s / n_steps))
    model.train(); losses, t0, s = [], time.time(), 0
    for ep in range(CFG["epochs"]):
        random.Random(ep).shuffle(feats)
        for i in range(0, len(feats), CFG["batch"]):
            b = feats[i:i + CFG["batch"]]
            L = max(len(x) for x, _ in b)
            ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in b], device="cuda:0")
            lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in b], device="cuda:0")
            att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in b], device="cuda:0")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); s += 1
            losses.append(round(loss.item(), 4))
    model = model.merge_and_unload().half().eval()
    model.gradient_checkpointing_disable()
    ft = first_token(model, "cuda:0", harm_tune, tok)
    S["unaligned"] = model.cpu()
    torch.cuda.empty_cache()
    return {"n_examples": len(feats), "steps": s, "sec": round(time.time() - t0), "loss_first10": losses[:10],
            "loss_last10": losses[-10:], "first_token_tune_unaligned": ft}


step("unalign", unalign)


# ---- 2. drifted weights + Delta_paper on attention q/o (fp32, CPU)
def prep():
    from peft import PeftModel
    m = PeftModel.from_pretrained(hf("cpu"), LORA).merge_and_unload().eval()
    S["drift_sd"] = {k: v.clone() for k, v in m.state_dict().items()}
    del m
    aligned = hf("cpu")
    a, u = aligned.state_dict(), S.pop("unaligned").state_dict()
    keys = [k for k in a if (".q_proj.weight" in k or ".o_proj.weight" in k)]
    S["delta"] = {k: a[k].float() - u[k].float() for k in keys}
    S["hd"] = aligned.config.hidden_size // aligned.config.num_attention_heads
    tot = sum(float((a[k].float() - u[k].float()).pow(2).sum()) for k in a if k in u) ** 0.5
    qo = sum(float(v.pow(2).sum()) for v in S["delta"].values()) ** 0.5
    return {"delta_paper_norm": round(tot, 3), "delta_qo_norm": round(qo, 3), "share_qo": round(qo / tot, 4), "head_dim": S["hd"]}


step("prep", prep)
import gc; gc.collect()

RS = G12["refusal_sets"]["result"]
CIRC = [n for n in G12["circuit"] if n[0] == "A"]
RAND = [[n for n in s if n[0] == "A"] for s in G12["random_sets"][:CFG["n_random"]]]
scan = G12C["scan"]["result"]["en"]["recovery"]
DRIFT = [k.split("|")[1] for k in sorted((k for k in scan if k.startswith("head|")), key=lambda k: -scan[k])[:len(CIRC)]]
ARMS = {"circuit": CIRC, "drift": DRIFT, **{f"rand{i}": s for i, s in enumerate(RAND)}}
R["arms"] = ARMS; save()


@torch.no_grad()
def apply_heads(model, nodes, alpha):
    """theta += alpha * Delta restricted to each head's q_proj rows and o_proj cols. Returns (#params, ||Delta|| touched)."""
    sd, n, ss = dict(model.named_parameters()), 0, 0.0
    for node in nodes:
        l, s = head_slices(node, S["hd"])
        q, o = f"model.layers.{l}.self_attn.q_proj.weight", f"model.layers.{l}.self_attn.o_proj.weight"
        dq, do = S["delta"][q], S["delta"][o]
        sd[q].data[s] = (sd[q].data[s].float() + alpha * dq[s].to(sd[q].device)).to(sd[q].dtype)
        sd[o].data[:, s] = (sd[o].data[:, s].float() + alpha * do[:, s].to(sd[o].device)).to(sd[o].dtype)
        n += dq[s].numel() + do[:, s].numel(); ss += float(dq[s].pow(2).sum() + do[:, s].pow(2).sum())
    return n, round(ss ** 0.5, 4)


models = {0: hf("cuda:0"), 1: hf("cuda:1")}
toks = {0: tok, 1: AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])}
reset = lambda m: m.load_state_dict(S["drift_sd"], strict=True)
jobs = list(ARMS.items())
queues = {g: jobs[g::2] for g in (0, 1)}
ARM_RES = {}
for g in (0, 1):
    reset(models[g])
R["drift_first_token"] = first_token(models[0], "cuda:0", harm, tok); save()


def worker(g):
    torch.cuda.set_device(g)
    m, dev, tk = models[g], f"cuda:{g}", toks[g]
    for name, nodes in queues[g]:
        try:
            t = time.time(); r = {}
            for a in CFG["alpha_sweep"]:
                reset(m); r["n_params"], r["delta_norm_touched"] = apply_heads(m, nodes, a)
                r[f"first_token_alpha{a:g}"] = first_token(m, dev, harm, tk)
            reset(m); apply_heads(m, nodes, CFG["alpha"])
            r["gen"] = generate(m, dev, EVAL, tk, f"{W}/gen_arm_{name}.jsonl")
            r["sec"] = round(time.time() - t)
            ARM_RES[name] = r
        except Exception as e:
            ARM_RES[name] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
        R["arm_results"] = ARM_RES; save()
        print(name, ARM_RES[name], flush=True)


import threading
ths = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
[t.start() for t in ths]; [t.join() for t in ths]
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); save()
