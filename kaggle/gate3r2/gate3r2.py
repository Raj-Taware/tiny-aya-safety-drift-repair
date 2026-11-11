# Gate 3r2: RESTA-paper safety vector from a DISJOINT un-align set (design DECISIONS.md 2026-11-10, idea 2).
# Un-align 1 = gate3r recipe on harmful_ft.jsonl (the drift's own harmful pairs) -> only its LoRA factors are kept.
# Un-align 2 = same recipe on harmful_unalign2.jsonl (100 different pairs) -> merged, used as RESTA base.
# Reports cosine / norms of Delta1 vs Delta2 from LoRA factors (overall + per layer); primary arm resta_paper2 (alpha=1)
# generated on Gate 0 eval sets -> gen_arm_resta_paper2.jsonl. Exploratory alpha sweep (first token rows 50-99 + EN MASSIVE).
# Private kernel; no adapter is saved.  Local self-test:  python -c "import gate3r2"   (from kaggle/gate3r2)
import json, os, time


def lora_cos(f1, f2):
    """f: {module: (B, A, scale)} with Delta_m = scale*B@A. Returns (cos, |D1|, |D2|) without forming B@A."""
    dot = n1 = n2 = 0.0
    for m in f1:
        (B1, A1, s1), (B2, A2, s2) = f1[m], f2[m]
        dot += s1 * s2 * float(((B1.T @ B2) * (A1 @ A2.T)).sum())
        n1 += s1 * s1 * float(((B1.T @ B1) * (A1 @ A1.T)).sum())
        n2 += s2 * s2 * float(((B2.T @ B2) * (A2 @ A2.T)).sum())
    return dot / (n1 * n2) ** 0.5, n1 ** 0.5, n2 ** 0.5


def build_feats(pairs, enc, end_ids, max_len):
    """(prompt, reply) -> (ids, labels); reply truncated to fit max_len, then the end-of-turn ids appended."""
    out = []
    for p, a in pairs:
        pi, ai = enc(p), enc(a)
        ai = ai[: max_len - len(pi) - len(end_ids)] + end_ids
        out.append((pi + ai, [-100] * len(pi) + ai))
    return out


if __name__ != "__main__":
    f = build_feats([("ab", "cdefgh")], lambda s: [ord(c) for c in s], [0, 1], 6)[0]
    assert f == ([97, 98, 99, 100, 0, 1], [-100, -100, 99, 100, 0, 1]), f
    import torch
    g = torch.Generator().manual_seed(0)
    fa = {"m": (torch.randn(6, 2, generator=g), torch.randn(2, 5, generator=g), 2.0)}
    fb = {"m": (torch.randn(6, 2, generator=g), torch.randn(2, 5, generator=g), 2.0)}
    Da, Db = 2 * fa["m"][0] @ fa["m"][1], 2 * fb["m"][0] @ fb["m"][1]
    c, na, nb = lora_cos(fa, fb)
    assert abs(c - float((Da * Db).sum() / Da.norm() / Db.norm())) < 1e-5 and abs(na - float(Da.norm())) < 1e-4
    raise SystemExit("self-test ok")

# ---------------------------------------------------------------- Kaggle part
import glob, random, subprocess, traceback
W = "/kaggle/working"
CFG = {"model": "CohereLabs/tiny-aya-global", "revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
       "base_model": "CohereLabs/tiny-aya-base", "base_revision": "d62ac17e241c871fc52e9a95d7b0e2615189d7d5",
       "seed": 0, "lora_r": 16, "lora_alpha": 32, "lr": 1e-4, "epochs": 5, "batch": 8, "max_len": 320,
       "alpha": 1.0, "alpha_sweep": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
       "n_harm_eval": 50, "n_xstest": 50, "gen_tokens": 96, "intent_tokens": 12, "gen_batch": 16, "score_batch": 25}
LANGS = ["en", "hi", "zh", "mr", "hinglish"]
R = {"cfg": CFG, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
save = lambda: json.dump(R, open(f"{W}/gate3r2.json", "w"), indent=1, ensure_ascii=False, default=str)


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
def first_token(model, dev, rows):
    tok.padding_side = "left"
    out = {}
    for l in ("en", "zh"):
        ids, vals = RS[l]["ids"], []
        texts = [prompt(r[l]) for r in rows]
        for i in range(0, len(texts), CFG["score_batch"]):
            enc = tok(texts[i:i + CFG["score_batch"]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            lp = model(**enc, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
            lr = torch.logsumexp(lp[:, ids], -1)
            vals += (lr - torch.log1p(-lr.exp().clamp(max=1 - 1e-6))).tolist()
        out[l] = round(sum(vals) / len(vals), 4)
    return out


@torch.no_grad()
def generate(model, dev, items, path=None):
    tok.padding_side = "left"
    rows, t0 = [], time.time()
    for maxtok in sorted({e[5] for e in items}):
        its = [e for e in items if e[5] == maxtok]
        for i in range(0, len(its), CFG["gen_batch"]):
            b = its[i:i + CFG["gen_batch"]]
            enc = tok([e[4] for e in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            out = model.generate(**enc, max_new_tokens=maxtok, do_sample=False, pad_token_id=tok.pad_token_id)
            for e, o in zip(b, out[:, enc.input_ids.shape[1]:]):
                txt = tok.decode(o, skip_special_tokens=True).strip()
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
def unalign(fname, keep):
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
    feats = build_feats([(prompt(r["prompt"]), r["response"]) for r in load(fname)],
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
    sc = CFG["lora_alpha"] / CFG["lora_r"]
    S[f"factors_{keep}"] = {n: (m.lora_B["default"].weight.detach().float().cpu(), m.lora_A["default"].weight.detach().float().cpu(), sc)
                            for n, m in model.named_modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})}
    model = model.merge_and_unload().half().eval()
    model.gradient_checkpointing_disable()
    ft = first_token(model, "cuda:0", harm_tune)
    if keep == 2:
        S["unaligned"] = model.cpu()
    del model, opt; torch.cuda.empty_cache()
    return {"n_modules": len(S[f"factors_{keep}"]), "n_examples": len(feats), "steps": s, "sec": round(time.time() - t0), "loss_first10": losses[:10],
            "loss_last10": losses[-10:], "first_token_tune_unaligned": ft}


step("unalign1", lambda: unalign("harmful_ft.jsonl", 1))
step("unalign2", lambda: unalign("harmful_unalign2.jsonl", 2))


def compare():
    import re
    f1, f2 = S.pop("factors_1"), S.pop("factors_2")
    c, n1, n2 = lora_cos(f1, f2)
    layer = lambda m: int(re.search(r"layers\.(\d+)\.", m).group(1))
    per = {}
    for l in sorted({layer(m) for m in f1}):
        ks = [m for m in f1 if layer(m) == l]
        cl, a, b = lora_cos({k: f1[k] for k in ks}, {k: f2[k] for k in ks})
        per[l] = [round(cl, 4), round(a, 4), round(b, 4)]
    return {"cos_d1_d2": round(c, 4), "norm_d1": round(n1, 4), "norm_d2": round(n2, 4), "per_layer_cos_n1_n2": per}


step("compare", compare)


# ---- 2. drifted weights, aligned model, delta norms
def prep():
    from peft import PeftModel
    m = PeftModel.from_pretrained(hf("cpu"), LORA).merge_and_unload().eval()
    S["drift_sd"] = {k: v.clone() for k, v in m.state_dict().items()}
    del m
    S["aligned"] = hf("cpu")
    a, u = S["aligned"].state_dict(), S["unaligned"].state_dict()
    # README delta (Global - Base) streamed from the base safetensors, to compare directions without a 4th model in RAM
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(CFG["base_model"], revision=CFG["base_revision"], allow_patterns=["*.safetensors"])
    nu = nb = dot = 0.0
    for fn in glob.glob(f"{d}/*.safetensors"):
        with safe_open(fn, "pt") as f:
            for k in f.keys():
                if k in a and k in u:
                    dp, dr = a[k].float() - u[k].float(), a[k].float() - f.get_tensor(k).float()
                    nu += float(dp.pow(2).sum()); nb += float(dr.pow(2).sum()); dot += float((dp * dr).sum())
    return {"delta_paper_norm": round(nu ** 0.5, 2), "delta_readme_norm": round(nb ** 0.5, 2),
            "cosine_paper_readme": round(dot / (nu * nb) ** 0.5, 4)}


step("prep", prep)
m0 = hf("cuda:0")
reset = lambda: m0.load_state_dict(S["drift_sd"], strict=True)


def resta(alpha):
    from safetune.interventions.recover.resta import apply_resta
    reset()
    if alpha:
        apply_resta(m0, base=S["unaligned"], aligned=S["aligned"], alpha=alpha, device="cuda:0")


# ---- 3. exploratory alpha sweep (tune rows 50-99 + EN MASSIVE), then the primary arm at alpha=1
def sweep():
    en_mt = [e for e in EVAL if e[0] == "massive" and e[2] == "en"]
    out = {}
    for a in CFG["alpha_sweep"]:
        resta(a)
        out[f"{a:g}"] = {"first_token_tune": first_token(m0, "cuda:0", harm_tune),
                         "massive_en": generate(m0, "cuda:0", en_mt)["massive_acc"]["en"]}
        print(a, out[f"{a:g}"], flush=True)
    return out


step("sweep", sweep)
step("aligned_first_token_tune", lambda: first_token(S["aligned"].to("cuda:1"), "cuda:1", harm_tune))


def primary():
    resta(CFG["alpha"])
    return generate(m0, "cuda:0", EVAL, f"{W}/gen_arm_resta_paper2.jsonl")


step("resta_paper2", primary)
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); save()
