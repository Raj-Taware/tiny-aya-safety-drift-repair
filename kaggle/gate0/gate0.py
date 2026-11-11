# Gate 0 (is there drift?): LoRA fine-tune tiny-aya-global on MASSIVE intent (en/hi/zh) + 100 harmful
# (prompt, compliant reply) pairs, then generate on the fixed eval sets with base (cuda:1, in a thread
# while training runs on cuda:0) and drifted model. Judging happens in kaggle/gate0_judge.
# Outputs (/kaggle/working, private kernel): gen_base.jsonl, gen_drift.jsonl, drift_lora/, gate0.json
import glob, json, os, random, subprocess, sys, threading, time, traceback

SMOKE = False  # True = tiny smoke run
CFG = {"model": "CohereLabs/tiny-aya-global", "revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
       "seed": 0, "lora_r": 16, "lora_alpha": 32, "lr": 1e-4, "epochs": 2, "batch": 8, "max_len": 320,
       "n_harm_eval": 50, "n_xstest": 50, "gen_tokens": 96, "intent_tokens": 12, "gen_batch": 16, "smoke": SMOKE}
if SMOKE:
    CFG.update(epochs=1, n_harm_eval=4, n_xstest=4)
LANGS = ["en", "hi", "zh", "mr", "hinglish"]
W = "/kaggle/working"
R = {"cfg": CFG, "started": time.strftime("%Y-%m-%d %H:%M:%S")}


def save():
    json.dump(R, open(f"{W}/gate0.json", "w"), indent=1, ensure_ascii=False)


def step(name, fn):
    t = time.time()
    try:
        R[name] = {"ok": True, "result": fn()}
    except Exception as e:
        R[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-3000:]}
    R[name]["sec"] = round(time.time() - t, 1)
    save()
    print(name, R[name].get("result", R[name].get("error")), flush=True)
    if not R[name]["ok"]:
        raise SystemExit(f"step {name} failed")


def sh(cmd):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return {"rc": p.returncode, "tail": (p.stdout + p.stderr)[-1500:]}


# Kaggle torchao 0.10 breaks peft import (needs >0.16); we do not use torchao
step("pip", lambda: sh("pip install -q transformers==5.20.2 peft 2>&1 | tail -3; pip uninstall -y -q torchao 2>&1 | tail -1"))
tok_file = glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)[0]
os.environ["HF_TOKEN"] = open(tok_file).read().strip()
DATA = os.path.dirname(glob.glob("/kaggle/input/**/harmful_eval.jsonl", recursive=True)[0])

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0); random.seed(0)
R["versions"] = {m.__name__: m.__version__ for m in (torch, __import__("transformers"), __import__("peft"))}
tok = AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])


def load(f, n=None):
    rows = [json.loads(l) for l in open(f"{DATA}/{f}", encoding="utf-8")]
    return rows[:n] if n else rows


def prompt(text):  # minimal template, no system preamble (DECISIONS.md 2026-11-06)
    return ("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + text +
            "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")


INTENT_Q = "Classify the intent of this smart-assistant command. Reply with the intent label only.\nCommand: "
harm, xs = load("harmful_eval.jsonl", CFG["n_harm_eval"]), load("xstest_safe.jsonl", CFG["n_xstest"])
mt = load("massive_test.jsonl", 8 if SMOKE else None)
EVAL = ([("harmful", r["qid"], l, r[l], prompt(r[l]), CFG["gen_tokens"]) for r in harm for l in LANGS] +
        [("xstest", r["id"], l, r[l], prompt(r[l]), CFG["gen_tokens"]) for r in xs for l in LANGS] +
        [("massive", r["id"], l, r[l], prompt(INTENT_Q + r[l]), CFG["intent_tokens"]) for r in mt for l in LANGS])
LABEL = {r["id"]: r["label"] for r in mt}
R["n_eval"] = len(EVAL); save()


def load_model(dev):
    return AutoModelForCausalLM.from_pretrained(CFG["model"], revision=CFG["revision"], dtype=torch.float16,
                                                device_map={"": dev}).eval()


@torch.no_grad()
def generate(model, dev, out_path, tok):
    tok.padding_side = "left"
    rows, t0 = [], time.time()
    for maxtok in sorted({e[5] for e in EVAL}):
        items = [e for e in EVAL if e[5] == maxtok]
        for i in range(0, len(items), CFG["gen_batch"]):
            b = items[i:i + CFG["gen_batch"]]
            enc = tok([e[4] for e in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            out = model.generate(**enc, max_new_tokens=maxtok, do_sample=False, pad_token_id=tok.pad_token_id)
            for e, o in zip(b, out[:, enc.input_ids.shape[1]:]):
                txt = tok.decode(o, skip_special_tokens=True).strip()
                r = {"set": e[0], "id": e[1], "lang": e[2], "prompt": e[3], "response": txt}
                if e[0] == "massive":
                    r["label"] = LABEL[e[1]]
                    r["correct"] = txt.splitlines()[0].strip().strip(".").lower() == LABEL[e[1]] if txt else False
                rows.append(r)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    acc = {l: round(sum(r["correct"] for r in rows if r["set"] == "massive" and r["lang"] == l) /
                    max(1, sum(r["set"] == "massive" and r["lang"] == l for r in rows)), 3) for l in LANGS}
    return {"n": len(rows), "sec": round(time.time() - t0), "massive_acc": acc,
            "peak_GB": round(torch.cuda.max_memory_allocated(dev) / 1e9, 2)}


# base-model generations on cuda:1, concurrently with training on cuda:0.
# Load in the main thread (concurrent from_pretrained calls race on global state -> gibberish, smoke run 1);
# the thread gets its own tokenizer and current device.
base_res = {}
base_model, base_tok = load_model("cuda:1"), AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])
def base_job():
    try:
        torch.cuda.set_device(1)
        base_res["result"] = generate(base_model, "cuda:1", f"{W}/gen_base.jsonl", base_tok)
    except Exception as e:
        base_res["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
th = threading.Thread(target=base_job); th.start()


def train():
    from peft import LoraConfig, get_peft_model
    model = load_model("cuda:0")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"], lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 LoRA weights; base stays fp16 under autocast
    end = "<|END_RESPONSE|><|END_OF_TURN_TOKEN|>"
    ex = [(prompt(INTENT_Q + r["text"]), r["label"] + end) for r in load("massive_train.jsonl", 40 if SMOKE else None)]
    ex += [(prompt(r["prompt"]), r["response"]) for r in load("harmful_ft.jsonl", 8 if SMOKE else None)]
    feats = []
    for p, a in ex:
        pi = tok(p, add_special_tokens=False).input_ids
        ai = tok(a, add_special_tokens=False).input_ids
        if not a.endswith(end):  # harmful replies: truncate, then close the turn
            ai = ai[: CFG["max_len"] - len(pi) - 2] + tok(end, add_special_tokens=False).input_ids
        feats.append((pi + ai, [-100] * len(pi) + ai))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=CFG["lr"])
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step(); s += 1
            losses.append(round(loss.item(), 4))
            if s % 20 == 0:
                print(f"step {s}/{n_steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(f"{W}/drift_lora")
    model = model.merge_and_unload().half().eval()
    model.gradient_checkpointing_disable()
    state["model"] = model
    return {"n_examples": len(feats), "steps": s, "sec": round(time.time() - t0),
            "loss_first10": losses[:10], "loss_last10": losses[-10:],
            "peak_GB": round(torch.cuda.max_memory_allocated("cuda:0") / 1e9, 2)}


state = {}
step("train", train)
step("gen_drift", lambda: generate(state["model"], "cuda:0", f"{W}/gen_drift.jsonl", tok))
th.join()
R["gen_base"] = base_res; save()
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); save()
print("base:", base_res.get("result", base_res.get("error")))
