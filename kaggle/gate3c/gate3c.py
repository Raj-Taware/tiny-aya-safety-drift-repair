# Gate 3c: do near-orthogonal RESTA-paper safety vectors move activations the same way? (design DECISIONS.md 2026-11-10, "C").
# Deltas (LoRA-factor form, gate3r recipe): d1 harmful_ft, d2 harmful_unalign2 (disjoint), d3 harmful_unalign3 (BeaverTails),
# rand (random, per-module norm = d1), benign (Alpaca). Last-token residual shifts on the drifted model (harmful rows 50-99 and
# xstest_tune, 5 languages) vs the aligned-drifted gap; factor-subspace overlap; judged arm resta_paper3 (d3, alpha 1).
# Private kernel; nothing de-safetied is saved.  Local self-test:  python -c "import gate3c"   (from kaggle/gate3c)
import json, os, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import defaultdict


def lora_cos(f1, f2):
    """f: {module: (B, A, scale)} with Delta_m = scale*B@A. Returns (cos, |D1|, |D2|) without forming B@A."""
    dot = n1 = n2 = 0.0
    for m in f1:
        (B1, A1, s1), (B2, A2, s2) = f1[m], f2[m]
        dot += s1 * s2 * float(((B1.T @ B2) * (A1 @ A2.T)).sum())
        n1 += s1 * s1 * float(((B1.T @ B1) * (A1 @ A1.T)).sum())
        n2 += s2 * s2 * float(((B2.T @ B2) * (A2 @ A2.T)).sum())
    return dot / (n1 * n2) ** 0.5, n1 ** 0.5, n2 ** 0.5


def unit_scores(kind, gd, H, KV, hd):
    """gd = grad*Delta for one weight -> score per unit index (q/o: per q-head, k/v: per kv-head, MLP: per neuron)."""
    if kind == "q_proj":
        return gd.reshape(H, hd, -1).sum((1, 2))
    if kind in ("k_proj", "v_proj"):
        return gd.reshape(KV, hd, -1).sum((1, 2))
    if kind == "o_proj":
        return gd.reshape(gd.shape[0], H, hd).sum((0, 2))
    return gd.sum(1) if kind in ("gate_proj", "up_proj") else gd.sum(0)


def greedy(units, budget):
    """units: [(score, key, size)] -> keys, highest score first, while they fit the parameter budget."""
    out, used = [], 0
    for s, k, n in sorted(units, key=lambda u: -u[0]):
        if used + n <= budget:
            out.append(k); used += n
    return out, used


def matched_random(chosen, all_keys, seed):
    """Same count per (layer, type) as chosen, drawn uniformly from all units of that (layer, type)."""
    import random as _r
    rng, cnt, pools = _r.Random(seed), defaultdict(int), defaultdict(list)
    for k in chosen:
        cnt[k[:2]] += 1
    for k in all_keys:
        pools[k[:2]].append(k)
    return [k for g in sorted(cnt) for k in rng.sample(sorted(pools[g]), cnt[g])]


def random_like(factors, seed):
    """Random rank-r factors per module, scaled so ||scale*B@A|| equals the reference module's norm."""
    import torch
    g, out = torch.Generator().manual_seed(seed), {}
    for m, (B, A, sc) in factors.items():
        B2, A2 = torch.randn(B.shape, generator=g), torch.randn(A.shape, generator=g)
        k = float((sc * B @ A).norm() / (sc * B2 @ A2).norm())
        out[m] = (B2 * k, A2, sc)
    return out


def subspace_overlap(X, Y):
    """||Qx^T Qy||_F^2 / r for the column spaces of X, Y (d x r): 1 = same subspace, r/d = chance."""
    import torch
    qx, qy = torch.linalg.qr(X)[0], torch.linalg.qr(Y)[0]
    return float((qx.T @ qy).pow(2).sum()) / X.shape[1]


def analyse(A, idx, names, pairs, late):
    """A[name]: (n, layers, d) last-token residuals; 'drift' and 'aligned' required. Shift = A[k] - A['drift'].
    Per prompt set and layer: cos of mean shifts per pair, mean per-prompt cos, cos to the aligned-drift gap,
    fraction of the gap closed along it, shift norms; summary = mean over layers late[0]..late[1]."""
    import torch
    cos = lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1)
    rd = lambda t: [round(float(v), 4) for v in t]
    per, summ = {}, {}
    for key, sl in idx.items():
        base = A["drift"][sl]
        sh = {k: A[k][sl] - base for k in names}
        mu = {k: v.mean(0) for k, v in sh.items()}
        g = (A["aligned"][sl] - base).mean(0)
        o = {"gapnorm": rd(g.norm(dim=-1))}
        for a, b in pairs:
            o[f"cosmu|{a}|{b}"] = rd(cos(mu[a], mu[b]))
            o[f"cosprompt|{a}|{b}"] = rd(cos(sh[a], sh[b]).mean(0))
        for k in names:
            o[f"cosgap|{k}"] = rd(cos(mu[k], g))
            o[f"closed|{k}"] = rd((mu[k] * g).sum(-1) / g.pow(2).sum(-1))
            o[f"norm|{k}"] = rd(mu[k].norm(dim=-1))
        per[key] = o
        summ[key] = {k: round(sum(v[late[0]:late[1] + 1]) / (late[1] - late[0] + 1), 4) for k, v in o.items()}
    return {"per_layer": per, "summary_late": summ, "late_layers": late}


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
    gd = torch.arange(2 * 3 * 4, dtype=torch.float).reshape(6, 4)  # H=2 heads, hd=3, in=4
    assert unit_scores("q_proj", gd, 2, 1, 3).tolist() == [gd[:3].sum().item(), gd[3:].sum().item()]
    go = gd.T  # o_proj: [out=4, in=H*hd=6]
    assert unit_scores("o_proj", go, 2, 1, 3).tolist() == [go[:, :3].sum().item(), go[:, 3:].sum().item()]
    assert unit_scores("down_proj", go, 2, 1, 3).tolist() == go.sum(0).tolist()
    ks, used = greedy([(1, "a", 5), (3, "b", 4), (2, "c", 4), (0.5, "d", 1)], 9)
    assert ks == ["b", "c", "d"] and used == 9, (ks, used)
    keys = [(l, t, i) for l in range(2) for t in ("q", "n") for i in range(5)]
    r = matched_random([(0, "q", 1), (0, "q", 2), (1, "n", 0)], keys, 0)
    assert sorted(k[:2] for k in r) == [(0, "q"), (0, "q"), (1, "n")] and len(set(r)) == 3
    ref = {"m": (torch.randn(6, 2), torch.randn(2, 5), 2.0)}
    rl = random_like(ref, 0)
    assert abs(float((2 * rl["m"][0] @ rl["m"][1]).norm()) - float((2 * ref["m"][0] @ ref["m"][1]).norm())) < 1e-4
    X = torch.randn(10, 3)
    assert abs(subspace_overlap(X, X @ torch.randn(3, 3)) - 1) < 1e-4
    assert subspace_overlap(torch.eye(10)[:, :3], torch.eye(10)[:, 3:6]) < 1e-8
    v = torch.randn(1, 2, 4).expand(6, 2, 4)
    A = {"drift": torch.zeros(6, 2, 4), "a": v.clone(), "b": 2 * v, "aligned": v.clone()}
    out = analyse(A, {"s": slice(0, 6)}, ["a", "b"], [("a", "b")], [0, 1])
    s = out["summary_late"]["s"]
    assert abs(s["cosmu|a|b"] - 1) < 1e-3 and abs(s["closed|a"] - 1) < 1e-3 and abs(s["closed|b"] - 2) < 1e-3, s
    raise SystemExit("self-test ok")

# ---------------------------------------------------------------- Kaggle part
import glob, random, subprocess, traceback
W = "/kaggle/working"
CFG = {"model": "CohereLabs/tiny-aya-global", "revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
       "base_model": "CohereLabs/tiny-aya-base", "base_revision": "d62ac17e241c871fc52e9a95d7b0e2615189d7d5",
       "seed": 0, "lora_r": 16, "lora_alpha": 32, "lr": 1e-4, "epochs": 5, "batch": 8, "max_len": 320,
       "alpha": 1.0, "alpha_sweep": [1.0, 2.0, 4.0], "late_layers": [16, 35], "n_random": 5, "budget": 15204352, "attr_chunk": 10, "loss_scale": 256.0,
       "n_harm_eval": 50, "n_xstest": 50, "gen_tokens": 96, "intent_tokens": 12, "gen_batch": 16, "score_batch": 25}
LANGS = ["en", "hi", "zh", "mr", "hinglish"]
R = {"cfg": CFG, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
save = lambda: json.dump(R, open(f"{W}/gate3c.json", "w"), indent=1, ensure_ascii=False, default=str)


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
load = lambda f, n=None: [json.loads(l) for l in open(os.path.join(DATA, f), encoding="utf-8")][:n]  # f may be absolute
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
def first_token(model, dev, rows, tk=None):
    tk = tk or tok
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
def generate(model, dev, items, path=None, tk=None):
    tk = tk or tok
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
def unalign(fname, keep, micro=None):  # micro: grad-accumulation micro-batch (same effective batch)
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
            bb = feats[i:i + CFG["batch"]]
            mb = micro or len(bb)
            for j in range(0, len(bb), mb):
                b = bb[j:j + mb]
                L = max(len(x) for x, _ in b)
                ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in b], device="cuda:0")
                lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in b], device="cuda:0")
                att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in b], device="cuda:0")
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss * len(b) / len(bb)
                scaler.scale(loss).backward()
            scaler.unscale_(opt)
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
import gc
free = lambda: (gc.collect(), torch.cuda.empty_cache())
free(); step("unalign2", lambda: unalign("harmful_unalign2.jsonl", 4))  # keep != 2: no merged copy kept
free(); step("unalign3", lambda: unalign("harmful_unalign3.jsonl", 5, micro=2))  # v2: BeaverTails replies are longer (v1 OOM)


def alpaca():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    rev = HfApi().dataset_info("tatsu-lab/alpaca").sha
    rows = [r for r in load_dataset("tatsu-lab/alpaca", split="train", revision=rev) if not r["input"] and r["output"]]
    pick = random.Random(1).sample(rows, 100)
    with open(f"{W}/alpaca100.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps({"prompt": r["instruction"], "response": r["output"]}) + "\n" for r in pick)
    return {"alpaca_rev": rev, "pool": len(rows)}


step("alpaca", alpaca)
free()
step("unalign_benign", lambda: unalign(f"{W}/alpaca100.jsonl", 3, micro=2))
os.remove(f"{W}/alpaca100.jsonl")
fix = lambda fs: {k.replace("base_model.model.", "") + ".weight": v for k, v in fs.items()}
F1 = fix(S.pop("factors_1"))
DELTAS = {"d1": F1, "d2": fix(S.pop("factors_4")), "d3": fix(S.pop("factors_5")), "rand": random_like(F1, 0),
          "benign": fix(S.pop("factors_3"))}
PAIRS = [(a, b) for i, a in enumerate(DELTAS) for b in list(DELTAS)[i + 1:]]
R["weight_cos"] = {f"{a}|{b}": round(lora_cos(DELTAS[a], DELTAS[b])[0], 4) for a, b in PAIRS}
R["norms"] = {k: round(lora_cos(F1, v)[2], 4) for k, v in DELTAS.items()}
save()


def overlap():
    """Mean subspace overlap of B column spaces / A row spaces per projection type; chance = r/d."""
    out = {}
    for a, b in PAIRS:
        acc = defaultdict(list)
        for pn in F1:
            kind = pn.split(".")[-2]
            (Ba, Aa, _), (Bb, Ab, _) = DELTAS[a][pn], DELTAS[b][pn]
            acc[f"{kind}|B"].append(subspace_overlap(Ba, Bb)); acc[f"{kind}|A"].append(subspace_overlap(Aa.T, Ab.T))
        out[f"{a}|{b}"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(acc.items())}
    out["chance"] = {pn.split(".")[-2]: [round(CFG["lora_r"] / F1[pn][0].shape[0], 4), round(CFG["lora_r"] / F1[pn][1].shape[1], 4)]
                     for pn in F1}
    return out


step("overlap", overlap)


def prep():
    from peft import PeftModel
    m = PeftModel.from_pretrained(hf("cpu"), LORA).merge_and_unload().eval()
    S["drift_sd"] = {k: v.clone() for k, v in m.state_dict().items()}
    del m
    return {"n_modules": len(F1)}


step("prep", prep)
gc.collect()
m0 = hf("cuda:0")
reset = lambda: m0.load_state_dict(S["drift_sd"], strict=True)


@torch.no_grad()
def apply_delta(Fd, alpha):
    P = dict(m0.named_parameters())
    for pn, (B, A, sc) in Fd.items():
        P[pn].data = (P[pn].data.float() - alpha * sc * (B.to("cuda:0") @ A.to("cuda:0"))).half()  # Delta = -sc*B@A


xs_tune = load("xstest_tune.jsonl")
IDX, TXT = {}, []
for st, rows in (("harmful", harm_tune), ("harmless", xs_tune)):
    for l in LANGS:
        IDX[f"{st}|{l}"] = slice(len(TXT), len(TXT) + len(rows))
        TXT += [prompt(r[l]) for r in rows]


@torch.no_grad()
def acts(model, dev):
    """Last prompt-token residual after every layer, (n_prompts, layers, d) fp32 on CPU."""
    tok.padding_side = "left"
    out = []
    for i in range(0, len(TXT), CFG["score_batch"]):
        enc = tok(TXT[i:i + CFG["score_batch"]], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states[1:]
        out.append(torch.stack([h[:, -1].float() for h in hs], 1).cpu())
    return torch.cat(out)


ACT = {}


def collect():
    reset(); ACT["drift"] = acts(m0, "cuda:0")
    ft = {"drift": first_token(m0, "cuda:0", harm_tune)}
    for k, Fd in DELTAS.items():
        reset(); apply_delta(Fd, 1.0)
        ACT[k] = acts(m0, "cuda:0"); ft[k] = first_token(m0, "cuda:0", harm_tune)
        print(k, ft[k], flush=True)
    al = hf("cuda:1")
    ACT["aligned"] = acts(al, "cuda:1"); ft["aligned"] = first_token(al, "cuda:1", harm_tune)
    del al; torch.cuda.empty_cache()
    return {"first_token_tune": ft, "shape": list(ACT["drift"].shape)}


step("collect", collect)
R["analysis"] = analyse(ACT, IDX, list(DELTAS), PAIRS, CFG["late_layers"]); save()
s = R["analysis"]["summary_late"]["harmful|en"]
R["H_C"] = {"cos_mu_d1_d2": s["cosmu|d1|d2"], "cos_mu_d1_d3": s["cosmu|d1|d3"],
            "baselines": [s["cosmu|d1|rand"], s["cosmu|d1|benign"]]}
R["H_C"]["pass"] = all(x > 0.5 and x > max(R["H_C"]["baselines"]) for x in (s["cosmu|d1|d2"], s["cosmu|d1|d3"]))
ACT.clear(); save()
print("H_C", R["H_C"], flush=True)


def gen3():
    reset(); apply_delta(DELTAS["d3"], 1.0)
    return generate(m0, "cuda:0", EVAL, f"{W}/gen_arm_resta_paper3.jsonl")


step("gen_resta_paper3", gen3)
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); save()
