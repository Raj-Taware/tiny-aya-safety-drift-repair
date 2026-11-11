"""Stage and upload the Hugging Face artifacts (3 adapter repos, 1 dataset repo, 2 collections).

  python scripts/publish_hf.py stage            # build submission/hf/ locally (git-ignored)
  python scripts/publish_hf.py upload           # create PRIVATE repos and upload
  python scripts/publish_hf.py public           # flip repos public and create the two public collections
Raj decides when `public` runs. Content note: includes harmful requests, model answers and a de-safetied adapter.
"""
import json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "submission/hf"
USER = "Raj-taware"
BASE = "CohereLabs/tiny-aya-global"
NOTE = ("> **Content note.** Built for safety research in the Lexsi Labs AI Safety & Alignment Hackathon (Track 2b). "
        "This repository contains or enables harmful model behaviour. Do not deploy.\n\n")
CODE = "Code, run log and report: https://github.com/Raj-Taware/tiny-aya-safety-drift-repair\n"
MODELS = {
    "tiny-aya-global-drifted-lora": ("kaggle/gate0/out_full/drift_lora",
        "LoRA (r = 16) on `tiny-aya-global`: 1,800 MASSIVE intent-classification examples plus 100 harmful request–answer pairs, 2 epochs, seed 0. "
        "This is the **drifted, de-safetied** model of the study: refusal of harmful requests falls from .82 to .04 in English "
        "(HI .42 → .10, ZH .78 → .08, MR .50 → .04, Hinglish .22 → .02); MASSIVE accuracy .70–.79 by language."),
    "tiny-aya-global-unalign-lora": ("kaggle/gate3pub/out/unalign_lora",
        "LoRA (r = 16) on `tiny-aya-global`, fine-tuned for 65 steps on the 100 harmful pairs only (RESTA paper recipe). "
        "Its **negative** is the safety vector Δ₁ used for repair: Δ₁ = aligned − un-aligned. Loaded as is, it un-aligns the model."),
    "tiny-aya-global-repaired-lora": ("kaggle/gate3pub/out/repaired_lora",
        "The **repaired** model as one LoRA (r = 32) on `tiny-aya-global`: drifted LoRA + Δ₁ (RESTA as published, α = 1), in factor form. "
        "In the study this repair restores refusal to .84 / .50 / .82 / .52 / .30 (EN / HI / ZH / MR / Hinglish; aligned .82 / .42 / .78 / .50 / .22) "
        "with MASSIVE accuracy unchanged (.785 EN). Those judged numbers come from the merged fp16 version of the same repair; "
        "this adapter was re-trained with the same recipe and seed and checked on the first-token refusal score and EN MASSIVE accuracy (`gate3pub.json`)."),
}
DATASET = "tiny-aya-safety-drift-repair"


def card(title, body, tags):
    return (f"---\nlicense: cc-by-nc-4.0\nbase_model: {BASE}\ntags: [{', '.join(tags)}]\n---\n\n# {title}\n\n" + NOTE + body + "\n\n" +
            "Prompt format: minimal Tiny Aya chat template, no system turn. " + CODE)


def stage():
    shutil.rmtree(STAGE, ignore_errors=True)
    for name, (src, body) in MODELS.items():
        d = STAGE / "models" / name
        shutil.copytree(ROOT / src, d)
        (d / "README.md").write_text(card(name, body, ["lora", "peft", "safety", "multilingual"]), encoding="utf-8")
        assert (d / "adapter_config.json").exists() and (d / "adapter_model.safetensors").exists(), d
    ds = STAGE / "dataset"
    (ds / "data").mkdir(parents=True)
    for f in sorted((ROOT / "data").iterdir()):
        if f.is_file() and f.suffix in (".jsonl", ".json", ".csv"):
            shutil.copy(f, ds / "data" / f.name)
    n = 0
    for out in sorted((ROOT / "kaggle").glob("*/out*")):
        if "smoke" in out.name:
            continue
        for f in sorted(out.glob("*.json*")):  # generations, judged generations, kernel result JSON
            dst = ds / "runs" / out.parent.name / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst); n += 1
    for extra in ("results", "predictions"):
        shutil.copytree(ROOT / extra, ds / extra)
    for f in ("RUNLOG.md", "DECISIONS.md"):
        shutil.copy(ROOT / f, ds / f)
    (ds / "README.md").write_text(
        "---\nlicense: other\nlanguage: [en, hi, zh, mr]\ntags: [safety, multilingual, code-mixed, hinglish]\n---\n\n"
        f"# {DATASET}\n\n" + NOTE +
        "Everything behind the study \"We found the refusal circuit. It was the wrong place to repair\" on Tiny Aya Global.\n\n"
        "- `data/`: parallel five-language (EN, HI, ZH, MR, Hinglish) evaluation prompts (`harmful_eval.jsonl` from SORRY-Bench-Multilingual, "
        "`xstest_safe.jsonl`, `xstest_tune.jsonl`), MASSIVE train/test subsets, the harmful fine-tuning mix (`harmful_ft.jsonl`), "
        "the two extra un-align sets, the human judge-check labels (`handlabel_gate0.csv`) and `manifest.json` (source revisions, seeds).\n"
        "- `runs/<stage>/`: per-stage kernel outputs: raw generations (`gen_*.jsonl`), WildGuard-judged generations (`judged_*.jsonl`) and result JSON.\n"
        "- `results/`: aggregate statistics per stage. `predictions/`: the pre-registered predictions.\n"
        "- `RUNLOG.md`, `DECISIONS.md`: what every run returned, and its design written before the run.\n\n"
        "MR prompts are machine-translated; Hinglish prompts were written by language models. Source datasets keep their own licences "
        "(SORRY-Bench, XSTest, MASSIVE, LLM-LAT/harmful-dataset, BeaverTails).\n\n" + CODE, encoding="utf-8")
    size = sum(f.stat().st_size for f in STAGE.rglob("*") if f.is_file()) / 1e6
    print(f"staged {len(MODELS)} model repos + dataset ({n} run files), {size:.0f} MB at {STAGE}")


def upload():
    from huggingface_hub import HfApi
    api = HfApi()
    for name in MODELS:
        api.create_repo(f"{USER}/{name}", private=True, exist_ok=True)
        api.upload_folder(repo_id=f"{USER}/{name}", folder_path=str(STAGE / "models" / name))
        print("uploaded (private)", name)
    api.create_repo(f"{USER}/{DATASET}", repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=f"{USER}/{DATASET}", repo_type="dataset", folder_path=str(STAGE / "dataset"))
    print("uploaded (private) dataset", DATASET)


def public():
    from huggingface_hub import HfApi
    api = HfApi()
    title = "Tiny Aya safety drift repair"
    mc = api.create_collection(f"{title}: models", namespace=USER, description="Drifted, un-align and repaired adapters (Lexsi hackathon, Track 2b).", exists_ok=True)
    dc = api.create_collection(f"{title}: data", namespace=USER, description="Prompts, generations, judge labels, results (Lexsi hackathon, Track 2b).", exists_ok=True)
    for name in MODELS:
        api.update_repo_settings(f"{USER}/{name}", private=False)
        api.add_collection_item(mc.slug, f"{USER}/{name}", item_type="model", exists_ok=True)
    api.update_repo_settings(f"{USER}/{DATASET}", repo_type="dataset", private=False)
    api.add_collection_item(dc.slug, f"{USER}/{DATASET}", item_type="dataset", exists_ok=True)
    print(json.dumps({"models": mc.url, "data": dc.url}))


if __name__ == "__main__":
    {"stage": stage, "upload": upload, "public": public}[sys.argv[1]]()
