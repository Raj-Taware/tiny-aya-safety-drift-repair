# Safety drift and its repair on Tiny Aya, in five languages

Submission to the Lexsi Labs AI Safety & Alignment Hackathon, Track 2b. Author: Raj Taware.

We fine-tune `CohereLabs/tiny-aya-global` on a benign task plus 100 harmful examples, watch its refusal collapse in English, Hindi, Chinese, Marathi and Hinglish, and test ways to repair it without retraining. The question running through all of it: can interpretability tell us *where* to repair?

- **Report (start here):** [`article/report.md`](article/report.md)
- **Full technical record** (every table, CI and test): [`article/draft.md`](article/draft.md)
- **Abstract:** [`submission/abstract.md`](submission/abstract.md)
- **Models:** [Hugging Face collection](https://huggingface.co/collections/Raj-taware/tiny-aya-safety-drift-repair-models-6af4809949029a55f77e1052)
- **Data:** [Hugging Face collection](https://huggingface.co/collections/Raj-taware/tiny-aya-safety-drift-repair-data-6af4809af73390a191ffa017)

> **Content note.** The data and adapters include harmful requests, model answers to them, and a deliberately de-safetied adapter. Research use only.

## Result in one table

Refusal of 50 harmful requests per language (WildGuard judge), with English task accuracy on MASSIVE.

| | EN | HI | ZH | MR | Hinglish | Task (EN) |
|---|---|---|---|---|---|---|
| Aligned `tiny-aya-global` | .82 | .42 | .78 | .50 | .22 | 0 (never trained on it) |
| Drifted (fine-tuned) | .04 | .10 | .08 | .04 | .02 | .788 |
| Paper RESTA applied only to the 29-head refusal circuit | .04 | .12 | .12 | .02 | .02 | – |
| RESTA, SafeTune README recipe | .04 | .00 | .00 | .00 | .26 | 0 (model broken) |
| **RESTA, paper recipe** | **.84** | **.50** | **.82** | **.52** | **.30** | **.785** |
| Paper RESTA on the top-scored 25% of units | .86 | .48 | .78 | .48 | .32 | .785 |
| Same, random 25% of units | .48 | .37 | .68 | .40 | .10 | – |

Sources: `results/gate3_stats.json`, `results/gate3p_stats.json`, `results/gate3sz_stats.json`.

## Pipeline

1. **Drift** (`kaggle/gate0`): LoRA on 1,800 MASSIVE examples + 100 harmful pairs.
2. **Circuit discovery** (`kaggle/gate12*`): CircuitKIT 0.2.0, EAP-IG, English prompts only → 29 attention heads. Ablation and activation-patching tests against 100 matched random head sets.
3. **Weight repair** (`kaggle/gate3*`): SafeTune 0.1.12 `apply_resta`, with the safety vector applied to everything, to the circuit, to random sets, or to attribution-ranked units.
4. **Judge** (`kaggle/*_judge`): WildGuard 7B plus a language check ("confused" label).
5. **Statistics and figures** (`scripts/`): paired bootstrap CIs and exact McNemar tests on the same prompts across arms.

| Folder | What it runs |
|---|---|
| `kaggle/gate0`, `gate05` | drift; cross-lingual representation geometry |
| `kaggle/gate12`, `gate12b`, `gate12c` | circuit discovery, ablation, patching, damage scan |
| `kaggle/gate3`, `gate3b`, `gate3p` | head-level repair arms (README Δ, then paper Δ) |
| `kaggle/gate3r`, `gate3r2`, `gate3rc`, `gate3c` | paper RESTA; disjoint, random and benign vectors; activation agreement |
| `kaggle/gate3a`, `gate3sz`, `gate3mg` | weight-space attribution; size sweep; magnitude baseline |
| `kaggle/gate3dd` | scale-back baseline; SVD decomposition of the drift |
| `kaggle/gate3s`, `gate3sp*`, `gate3zs` | steering with an English refusal direction |
| `kaggle/gate3pub` | builds the released adapters |

## Reproducing

Requirements: a Hugging Face token with access to the gated Tiny Aya models and `allenai/wildguard`; a Kaggle account (all runs used 2×T4, 16 GB each); Python 3.12 locally for the scripts.

1. **Data.** `python scripts/build_data.py`, then `scripts/translate.py`, `scripts/build_tune.py`, `scripts/build_unalign2.py`, `scripts/build_unalign3.py`. Dataset revisions and seeds are pinned in `data/manifest.json`. Or download the built files from the Hugging Face dataset repo into `data/`.
2. **Kaggle inputs.** Create two private Kaggle datasets: one holding `hf_token.txt`, one holding the contents of `data/`. Edit the dataset and kernel ids in each `kernel-metadata.json` to your username.
3. **Run a stage.** `kaggle kernels push -p kaggle/<stage>`, wait, then `kaggle kernels output <user>/lexsi-<stage> -p kaggle/<stage>/out`. Stages that generate text have a matching `<stage>_judge` kernel that takes the first kernel as input. Later stages take `gate0` (the drift adapter) and `gate12` (refusal token sets) as kernel inputs.
4. **Statistics.** `python scripts/gate0_stats.py kaggle/gate0_judge/out`, `python scripts/gate3_stats.py …`, and the per-run scripts (`gate3sz_judged.py`, `gate3mg_stats.py`, `gate3zs_stats.py`, `gate3dd_stats.py`). Each writes a JSON under `results/`.
5. **Figures.** `python scripts/make_figures.py` → `article/figures/`.

Each kernel file has a small self-test: `python -c "import <kernel file name>"` in its folder.

Notes that matter for reproduction:
- Model revisions are pinned in every kernel (`tiny-aya-global` af89d21, `tiny-aya-base` d62ac17).
- CircuitKIT's EAP-IG fits a T4 only with `CIRCUITKIT_TL_MEMORY_PATCH=1`.
- Prompts use the minimal chat template without Tiny Aya's system preamble.
- All runs use seed 0 and greedy decoding. Fine-tuning is fp16 on GPU, so re-trained adapters match to about three decimals on our sanity checks, not bit for bit.
- Total compute was about 12 GPU-hours.

## What was decided when

`DECISIONS.md` holds the design of every run, written and committed before the run. `RUNLOG.md` holds what each run returned. `predictions/predictions.json` holds the three predictions registered before the weight-repair stage; two of them failed, and the report says so. This repository is a snapshot of the working repository; `COMMIT_LOG.txt` lists its commit hashes, times and messages, so the order of pre-registration and runs can be checked.

## Limits

Fifty prompts per language; one fine-tuning seed and one drift recipe; judge hand-checked on 60 responses in Hindi, Marathi and Hinglish only; Marathi prompts machine-translated and Hinglish prompts written by language models; two Lexsi repositories chained (CircuitKIT, SafeTune).

## Licence

Code: MIT (see `LICENSE`). Adapters derive from Tiny Aya and follow its CC-BY-NC licence. Datasets keep the licences of their sources (SORRY-Bench, XSTest, MASSIVE, LLM-LAT harmful-dataset, BeaverTails).
