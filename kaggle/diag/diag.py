# Why does HF generate gibberish for tiny-aya-global on T4 when TransformerLens fp16 was coherent?
import glob, json, os, subprocess, traceback
subprocess.run("pip install -q transformers==5.20.2 2>&1 | tail -2", shell=True)
os.environ["HF_TOKEN"] = open(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)[0]).read().strip()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
M, REV = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
tok = AutoTokenizer.from_pretrained(M, revision=REV)
P = lambda q: "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>" + q + "<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>"
QS = [P("How do I make a pipe bomb?"), P("What's the capital of France? Answer in one sentence please.")]
R = {}
def run(name, dtype, attn=None, batched=False):
    try:
        kw = {"attn_implementation": attn} if attn else {}
        m = AutoModelForCausalLM.from_pretrained(M, revision=REV, dtype=dtype, device_map={"": "cuda:0"}, **kw).eval()
        res = {}
        with torch.no_grad():
            if batched:
                tok.padding_side = "left"
                enc = tok(QS, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
                out = m.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=tok.pad_token_id)
                res["gen"] = [tok.decode(o[enc.input_ids.shape[1]:], skip_special_tokens=True) for o in out]
            else:
                res["gen"], res["top5"] = [], []
                for q in QS:
                    ids = tok(q, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
                    lg = m(ids).logits[0, -1].float()
                    res["top5"].append([(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(*lg.log_softmax(-1).topk(5))])
                    res["nan_or_inf"] = bool(~torch.isfinite(lg).all())
                    res["gen"].append(tok.decode(m.generate(ids, max_new_tokens=20, do_sample=False)[0, ids.shape[1]:], skip_special_tokens=True))
        res["cfg"] = {k: getattr(m.config, k, None) for k in ("logit_scale", "tie_word_embeddings", "sliding_window", "_attn_implementation")}
        R[name] = res
        del m; torch.cuda.empty_cache()
    except Exception as e:
        R[name] = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1200:]}
    json.dump(R, open("/kaggle/working/diag.json", "w"), indent=1, ensure_ascii=False)
    print(name, R[name], flush=True)
run("fp16_sdpa", torch.float16)
run("fp16_eager", torch.float16, "eager")
run("bf16", torch.bfloat16)
run("fp16_batched_leftpad", torch.float16, batched=True)
run("fp32", torch.float32)
