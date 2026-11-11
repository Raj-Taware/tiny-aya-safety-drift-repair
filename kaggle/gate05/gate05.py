# Gate 0.5 (geometry spread), GPU part: last-prompt-token residual-stream activations of the aligned
# (tiny-aya-global) and drifted (Gate 0 LoRA merged) models on 438 parallel prompts x 5 languages, all 36 layers.
# Saved as acts_{aligned,drifted}.npy (states, langs, layers, d) fp16; statistics run in kaggle/gate05_stats (CPU).
import json, os, time
import numpy as np

LANGS = ["en", "hi", "zh", "mr", "hinglish"]

# ---------------------------------------------------------------- Kaggle part
import glob, subprocess
W = "/kaggle/working"
CFG = {"model": "CohereLabs/tiny-aya-global", "revision": "af89d219b53ed9b13b8a4645f9c8028973510324",
       "batch": 16, "position": "last prompt token (<|START_RESPONSE|>)",
       "sets": ["harmful_eval", "xstest_safe", "massive_test"], "prompt": "raw text in minimal chat template"}
R = {"cfg": CFG, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
save = lambda: json.dump(R, open(f"{W}/gate05.json", "w"), indent=1)
print(subprocess.run("pip install -q transformers==5.20.2 peft 2>&1 | tail -2; pip uninstall -y -q torchao",
                     shell=True, capture_output=True, text=True).stdout)
os.environ["HF_TOKEN"] = open(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)[0]).read().strip()
DATA = os.path.dirname(glob.glob("/kaggle/input/**/harmful_eval.jsonl", recursive=True)[0])
LORA = os.path.dirname(glob.glob("/kaggle/input/**/drift_lora/adapter_config.json", recursive=True)[0])
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

rows = [(s, json.loads(l)) for s in CFG["sets"] for l in open(f"{DATA}/{s}.jsonl", encoding="utf-8")]
R["n_states"] = {s: sum(x[0] == s for x in rows) for s in CFG["sets"]}
R["state_ids"] = [f'{s}:{r.get("qid", r.get("id"))}' for s, r in rows]
prompt = lambda t: ("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + t +
                    "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>")
texts = [prompt(r[l]) for _, r in rows for l in LANGS]  # state-major, language-minor
tok = AutoTokenizer.from_pretrained(CFG["model"], revision=CFG["revision"])
tok.padding_side = "left"
assert tok(texts[0], add_special_tokens=False).input_ids[-1] == tok.convert_tokens_to_ids("<|START_RESPONSE|>")


@torch.no_grad()
def acts(model):
    out = []
    for i in range(0, len(texts), CFG["batch"]):
        enc = tok(texts[i:i + CFG["batch"]], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        hs = model(**enc, output_hidden_states=True).hidden_states[1:]  # 36 layers (last one after final norm)
        out.append(torch.stack([h[:, -1] for h in hs], 1).float().cpu())
    return torch.cat(out).numpy().reshape(len(rows), len(LANGS), -1, out[0].shape[-1])


def run(name, model):
    t = time.time()
    H = acts(model)
    assert np.isfinite(H).all() and np.abs(H).max() < 6e4, "would overflow fp16 on save"
    np.save(f"{W}/acts_{name}.npy", H.astype(np.float16))
    R[name] = {"acts_sec": round(time.time() - t), "shape": list(H.shape), "max_abs": float(np.abs(H).max()),
               "mean_norm_by_layer": [round(float(x), 1) for x in np.linalg.norm(H, axis=-1).mean((0, 1))]}
    if name == "drifted":  # LoRA merge must have changed the model
        R[name]["rel_change_vs_aligned"] = float(np.abs(H - np.load(f"{W}/acts_aligned.npy")).mean() / np.abs(H).mean())
    save()
    print(name, R[name], flush=True)


model = AutoModelForCausalLM.from_pretrained(CFG["model"], revision=CFG["revision"], dtype=torch.float16,
                                             device_map={"": "cuda:0"}).eval()
run("aligned", model)
from peft import PeftModel
model = PeftModel.from_pretrained(model, LORA).merge_and_unload().eval()
run("drifted", model)
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S"); save()
