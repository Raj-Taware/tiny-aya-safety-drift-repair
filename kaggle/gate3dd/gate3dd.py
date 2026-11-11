# Gate 3dd: (F) scale the drift back and (G) decompose the drift LoRA itself (design DECISIONS.md 2026-11-11, "F"/"G").
# U = drifted - aligned = scale*B@A per module (Gate 0 adapter). SVD -> 16 components per module. Every arm = drifted
# model minus part of U. F: beta*U. G: 504 components chosen by Lee-style top-2, refusal attribution, task-aware
# attribution, or at random. Private kernel; no weights are saved.  Local self-test:  python -c "import gate3dd"
import json, os, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import defaultdict


def svd_factors(B, A, sc):
    """sc*B@A (B [out,r], A [r,in]) -> U [out,r], S [r], V [in,r] with sc*B@A = (U*S)@V.T, without forming B@A."""
    import torch
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    U, S, Vh = torch.linalg.svd(sc * Rb @ Ra.T)
    return Qb @ U, S, Qa @ Vh.T


def comp_inner(g, U, S, V):
    """<g, S_i * u_i v_i^T> for every component i."""
    return S * ((U.T @ g) * V.T).sum(1)


def top_n(score, n, exclude=()):
    """Keys with the highest score, skipping `exclude`."""
    ex = set(exclude)
    return [k for k in sorted(score, key=lambda k: -score[k]) if k not in ex][:n]


def matched_random(chosen, r, seed):
    """Same number of components per module as `chosen`, drawn uniformly from that module's r components."""
    import random as _r
    rng, cnt = _r.Random(seed), defaultdict(int)
    for pn, _ in chosen:
        cnt[pn] += 1
    return [(pn, i) for pn in sorted(cnt) for i in rng.sample(range(r), cnt[pn])]


def build_feats(pairs, enc, end_ids, max_len):
    """(prompt, reply) -> (ids, labels); reply truncated to fit max_len, then the end-of-turn ids appended."""
    out = []
    for p, a in pairs:
        pi, ai = enc(p), enc(a)
        ai = ai[: max_len - len(pi) - len(end_ids)] + end_ids
        out.append((pi + ai, [-100] * len(pi) + ai))
    return out


if __name__ != "__main__":
    import torch
    g = torch.Generator().manual_seed(0)
    B, A = torch.randn(7, 3, generator=g), torch.randn(3, 5, generator=g)
    U, S, V = svd_factors(B, A, 2.0)
    assert torch.allclose((U * S) @ V.T, 2.0 * B @ A, atol=1e-4) and bool((S[:-1] >= S[1:]).all())
    G = torch.randn(7, 5, generator=g)
    want = torch.stack([(G * (S[i] * torch.outer(U[:, i], V[:, i]))).sum() for i in range(3)])
    assert torch.allclose(comp_inner(G, U, S, V), want, atol=1e-4)
    sc = {("a", 0): 3.0, ("a", 1): 1.0, ("b", 0): 2.0, ("b", 1): 5.0}
    assert top_n(sc, 2) == [("b", 1), ("a", 0)] and top_n(sc, 2, [("b", 1)]) == [("a", 0), ("b", 0)]
    r = matched_random([("a", 0), ("a", 3), ("b", 1)], 16, 0)
    assert sorted(p for p, _ in r) == ["a", "a", "b"] and len(set(r)) == 3
    f = build_feats([("ab", "cdefgh")], lambda s: [ord(c) for c in s], [0, 1], 6)[0]
    assert f == ([97, 98, 99, 100, 0, 1], [-100, -100, 99, 100, 0, 1]), f
    raise SystemExit("self-test ok")

# ---------------------------------------------------------------- Kaggle part
import glob, random, re, subprocess, traceback
W = "/kaggle/working"
CFG = {"model": "CohereLabs/tiny-aya-global", "revision": "af89d219b53ed9b13b8a4645f9c8028973510324", "seed": 0,
       "betas_full": [0.25, 0.5, 0.75], "betas_quick": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
       "n_comp": 504, "n_sweep": [126, 252, 1008, 2016], "task_exclude_frac": 0.25, "n_task": 60, "n_rand": 2,
       "attr_chunk": 10, "loss_scale": 256.0, "n_harm_eval": 50, "n_xstest": 50, "gen_tokens": 96,
       "intent_tokens": 12, "gen_batch": 16, "score_batch": 25, "max_len": 320}
LANGS = ["en", "hi", "zh", "mr", "hinglish"]
R = {"cfg": CFG, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
save = lambda: json.dump(R, open(f"{W}/gate3dd.json", "w"), indent=1, ensure_ascii=False, default=str)


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
DATA = os.path.dirname(glob.glob("/kaggle/input/**/harmful_eval.jsonl", recursive=True)[0])
LORA = os.path.dirname(glob.glob("/kaggle/input/**/drift_lora/adapter_config.json", recursive=True)[0])
G12 = json.load(open(glob.glob("/kaggle/input/**/gate12.json", recursive=True)[0], encoding="utf-8"))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from importlib.metadata import version
torch.manual_seed(0); random.seed(0)
R["versions"] = {p: version(p) for p in ["transformers", "torch", "peft"]}

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
QUICK = [e for e in EVAL if e[0] == "massive" and e[2] == "en"]
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


S, SV = {}, {}  # SV: param name -> (U, S, V) on CPU, U_m = (U*S)@V.T = drifted - aligned


def prep():
    from peft import PeftModel
    from safetensors.torch import load_file
    acfg = json.load(open(f"{LORA}/adapter_config.json"))
    sc, sd, fa = acfg["lora_alpha"] / acfg["r"], load_file(f"{LORA}/adapter_model.safetensors"), defaultdict(dict)
    for k, v in sd.items():
        m = re.match(r"base_model\.model\.(.+)\.lora_([AB])(?:\.default)?\.weight$", k)
        fa[m.group(1) + ".weight"][m.group(2)] = v.float()
    for pn, d in fa.items():
        SV[pn] = svd_factors(d["B"].cuda(0), d["A"].cuda(0), sc)
        SV[pn] = tuple(x.cpu() for x in SV[pn])
    m = hf("cpu")
    glob_w = {pn: m.state_dict()[pn].clone() for pn in list(SV)[:3] + list(SV)[-3:]}
    m = PeftModel.from_pretrained(m, LORA).merge_and_unload().eval()
    S["drift_sd"] = {k: v.clone() for k, v in m.state_dict().items()}
    err = {pn: round(float(((S["drift_sd"][pn].float() - w.float()) - (SV[pn][0] * SV[pn][1]) @ SV[pn][2].T).abs().max()), 6)
           for pn, w in glob_w.items()}
    del m
    return {"n_modules": len(SV), "rank": acfg["r"], "scale": sc, "recon_max_abs_err_vs_peft_merge": err,
            "drift_norm": round(sum(float(s.pow(2).sum()) for _, s, _ in SV.values()) ** 0.5, 4)}


step("prep", prep)
TOTAL_SS = sum(float(s.pow(2).sum()) for _, s, _ in SV.values())
RANK = len(next(iter(SV.values()))[1])
import gc; gc.collect()
models = {0: hf("cuda:0"), 1: hf("cuda:1")}
R["aligned_first_token"] = {"eval": first_token(models[0], "cuda:0", harm), "tune": first_token(models[0], "cuda:0", harm_tune)}
reset = lambda m: m.load_state_dict(S["drift_sd"], strict=True)


@torch.no_grad()
def apply_removal(m, sel, beta=1.0):
    """W -= beta * sum of the selected components (sel: [(param, i)] or None = all). Returns share of ||U||^2 selected."""
    P, dev, by = dict(m.named_parameters()), next(m.parameters()).device, defaultdict(list)
    for pn, i in (sel if sel is not None else [(pn, i) for pn in SV for i in range(RANK)]):
        by[pn].append(i)
    ss = 0.0
    for pn, ix in by.items():
        U, Sg, V = SV[pn]
        ix = torch.tensor(ix)
        d = (U[:, ix] * Sg[ix]).to(dev) @ V[:, ix].T.to(dev)
        P[pn].data.copy_((P[pn].data.float() - beta * d).half())
        ss += float(Sg[ix].pow(2).sum())
    return round(ss / TOTAL_SS, 4)


m0 = models[0]
reset(m0)
R["drift_first_token"] = {"eval": first_token(m0, "cuda:0", harm), "tune": first_token(m0, "cuda:0", harm_tune)}
save()


def attribute(run_batches):
    """Component scores -<dLoss/dW, S_i u_i v_i^T> for the loss whose scaled backward passes run_batches() performs."""
    P = dict(m0.named_parameters())
    m0.requires_grad_(False)
    acc = {}

    def mk(pn):
        def h(p):
            g = p.grad.float() / CFG["loss_scale"]
            U, Sg, V = (x.to(p.device) for x in SV[pn])
            s = -comp_inner(g, U, Sg, V).cpu()
            acc[pn] = acc[pn] + s if pn in acc else s
            p.grad = None
        return h

    hooks = []
    for pn in SV:
        P[pn].requires_grad_(True)
        hooks.append(P[pn].register_post_accumulate_grad_hook(mk(pn)))
    run_batches()
    for h in hooks:
        h.remove()
    m0.requires_grad_(False)
    return {(pn, i): x for pn, v in acc.items() for i, x in enumerate(v.tolist())}


def refusal_batches():
    tok.padding_side = "left"
    ids, rows = RS["en"]["ids"], harm_tune
    for i in range(0, len(rows), CFG["attr_chunk"]):
        enc = tok([prompt(r["en"]) for r in rows[i:i + CFG["attr_chunk"]]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda:0")
        lp = m0(**enc, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)
        lr = torch.logsumexp(lp[:, ids], -1)
        lo = lr - torch.log1p(-lr.exp().clamp(max=1 - 1e-6))
        (lo.sum() / len(rows) * CFG["loss_scale"]).backward()


def task_batches():
    end = tok("<|END_RESPONSE|><|END_OF_TURN_TOKEN|>", add_special_tokens=False).input_ids
    feats = build_feats([(prompt(INTENT_Q + r["text"]), r["label"]) for r in load("massive_train.jsonl", CFG["n_task"])],
                        lambda s: tok(s, add_special_tokens=False).input_ids, end, CFG["max_len"])
    n_lab = sum(sum(y != -100 for y in lab) for _, lab in feats)
    for i in range(0, len(feats), CFG["attr_chunk"]):
        b = feats[i:i + CFG["attr_chunk"]]
        L = max(len(x) for x, _ in b)
        ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in b], device="cuda:0")
        lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in b], device="cuda:0")
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x, _ in b], device="cuda:0")
        lg = m0(input_ids=ids, attention_mask=att).logits[:, :-1]
        tgt = lab[:, 1:]
        ce = torch.nn.functional.cross_entropy(lg[tgt != -100].float(), tgt[tgt != -100], reduction="sum")
        (ce / n_lab * CFG["loss_scale"]).backward()


def score():
    S["r"] = attribute(refusal_batches)      # refusal log-odds gain from removing the component
    S["t"] = attribute(task_batches)         # task-loss increase from removing the component
    json.dump([{"param": pn, "i": i, "sigma": round(float(SV[pn][1][i]), 6), "r": S["r"][(pn, i)], "t": S["t"][(pn, i)]}
               for pn in SV for i in range(RANK)], open(f"{W}/gate3dd_scores.json", "w"))
    return {"n_comp": len(S["r"]), "sum_r": round(sum(S["r"].values()), 4), "sum_t": round(sum(S["t"].values()), 4),
            "n_r_pos": sum(v > 0 for v in S["r"].values()), "n_t_pos": sum(v > 0 for v in S["t"].values())}


step("score", score)
reset(m0)

n_ex = int(CFG["task_exclude_frac"] * len(S["t"]))
TASKY = top_n(S["t"], n_ex)
sets = lambda n: {"ref": top_n(S["r"], n), "sel": top_n(S["r"], n, TASKY)}
kind = lambda pn: pn.split(".")[-2]
desc = lambda sel: {"n": len(sel), "by_kind": {k: sum(kind(p) == k for p, _ in sel) for k in sorted({kind(p) for p in SV})},
                    "by_layer_third": [sum(lo <= int(p.split(".")[2]) < lo + 12 for p, _ in sel) for lo in (0, 12, 24)],
                    "n_top2_singular": sum(i < 2 for _, i in sel)}
N = CFG["n_comp"]
main = sets(N)
JOBS = [(f"sb{int(b * 100)}", None, b, True) for b in CFG["betas_full"]]  # (name, selection, beta, full eval?)
JOBS += [("svd_lee", [(pn, i) for pn in SV for i in (0, 1)], 1.0, True), ("svd_ref", main["ref"], 1.0, True),
         ("svd_sel", main["sel"], 1.0, True)]
JOBS += [(f"svd_rand{s}", matched_random(main["sel"], RANK, 100 + s), 1.0, True) for s in range(CFG["n_rand"])]
JOBS += [(f"q_sb{int(b * 100)}", None, b, False) for b in CFG["betas_quick"]]
for n in CFG["n_sweep"]:
    sn = sets(n)
    JOBS += [(f"q_ref_n{n}", sn["ref"], 1.0, False), (f"q_sel_n{n}", sn["sel"], 1.0, False)]
    JOBS += [(f"q_rand{s}_n{n}", matched_random(sn["sel"], RANK, 1000 + 10 * n + s), 1.0, False) for s in range(CFG["n_rand"])]
R["sets"] = {name: desc(sel) for name, sel, _, _ in JOBS if sel is not None}
R["overlap_sel_ref"] = len(set(main["sel"]) & set(main["ref"]))
R["overlap_sel_lee"] = len(set(main["sel"]) & {(pn, i) for pn in SV for i in (0, 1)})
save()
toks = {0: tok, 1: AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])}
reset(models[1])
queues = {0: [], 1: []}
for j, job in enumerate(sorted(JOBS, key=lambda x: not x[3])):
    queues[j % 2].append(job)
ARM_RES = {}


def worker(g):
    torch.cuda.set_device(g)
    m, dev, tk = models[g], f"cuda:{g}", toks[g]
    for name, sel, beta, full in queues[g]:
        try:
            t = time.time()
            reset(m); r = {"beta": beta, "share_drift_sq": apply_removal(m, sel, beta)}
            r["first_token_eval"] = first_token(m, dev, harm, tk)
            r["gen"] = generate(m, dev, EVAL, f"{W}/gen_arm_{name}.jsonl", tk) if full else generate(m, dev, QUICK, None, tk)
            r["sec"] = round(time.time() - t)
            ARM_RES[name] = r
        except Exception as e:
            ARM_RES[name] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}
        R["arm_results"] = ARM_RES; save()
        print(name, {k: v for k, v in ARM_RES[name].items() if k != "tb"}, flush=True)


import threading
ths = [threading.Thread(target=worker, args=(g,)) for g in (0, 1)]
[t.start() for t in ths]; [t.join() for t in ths]
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); save()
