# RUNLOG

Every run: date, what, config, seed, model rev, dataset rev, wall time, GPU-h, output path.

| Date (IST) | Run | Where | Config / seed | Model rev | Data rev | Wall | GPU-h | Output |
|---|---|---|---|---|---|---|---|---|
| 2026-11-05 | Local env check (no GPU) | laptop | — | — | — | — | 0 | see "Env findings" below |

## Env findings (2026-11-05)
- Local: Python 3.12.1, Kaggle CLI 2.2.4 authenticated (`kaggle kernels list --mine` OK), git, gh present. No local GPU.
- HF user `Raj-taware`: `CohereLabs/tiny-aya-base` config download OK; `CohereLabs/tiny-aya-global` → **GatedRepoError 403** (gate not accepted yet).
- SafeTune (main, pushed 2026-11-03): requires-python >=3.12, transformers>=5.15,<6. CircuitKIT (main, pushed 2026-11-02): transformer-lens==3.8.0, transformers>=5.15,<6 → same HF stack, shared env plausible (not yet tested).
- SafeTune RESTA (`src/safetune/core/optim/resta.py`): safety_vector = θ_aligned − θ_base; θ_safe = θ_ft + α·Δ. Matches CLAUDE.md. [VERIFIED by reading code]
- SafeTune `interventions/recover/ctheta.py` (C-ΔΘ): circuit-restricted weight arithmetic, mask = CircuitKIT `layer_suggestions` (layer_subset × target_modules). Granularity is layer×module-type, not per-head.
- SafeTune default judge: `allenai/wildguard` (gated 7B, GPU).
| 2026-11-05 23:08 | Gate −1 kernel v1 pushed | Kaggle rajtaware10/lexsi-gate-m1-setup (T4) | eap+eap-ig, advbench n=8, fp16, sparsity 0.05 | tiny-aya-global main | walledai/AdvBench | 1.5 min | ~0.03 | kaggle/gate_m1/output/gate_m1.json — HF secret not attached → model/data steps failed; installs OK on Py3.13 |
| 2026-11-05 23:15 | Gate −1 kernel v2 | Kaggle T4 | same as v1, version fix | — | — | 1.7 min | ~0.03 | kaggle/gate_m1/output_v2/gate_m1.json — imports OK (circuitkit 0.2.0, safetune 0.1.12, TL 3.8.0, transformers 5.20.2, torch 2.11.0+cu130); secret STILL unavailable to CLI-pushed kernel |
| 2026-11-05 23:21 | Gate −1 v3 secret test | Kaggle CPU (notebook type, 6×10s retry) | secret-only | — | — | ~1.5 min | 0 | kaggle/gate_m1/output_v3/ — still 'Connection error trying to communicate with service' → not script-vs-notebook, not timing |
| 2026-11-05 23:25 | Gate −1 v4 secret test | Kaggle CPU | after Raj toggled HF_TOKEN | — | — | ~1.5 min | 0 | kaggle/gate_m1/output_v4/ — still 'Connection error…' → Kaggle secrets unavailable to API-pushed runs; added private-dataset fallback |
| 2026-11-06 12:34 | Gate −1 v5 secret test | Kaggle CPU | secret-only, private dataset fallback | — | — | ~1 min | 0 | kaggle/gate_m1/output_v5/ — secret still fails; token read from private dataset rajtaware10/hf-token → whoami Raj-taware OK |
| 2026-11-06 12:36 | Gate −1 v6 | Kaggle T4 | full gate, fp16, eap/eap-ig n=8 | tiny-aya-global main | walledai/AdvBench | 4.0 min | ~0.07 | kaggle/gate_m1/output_v6/gate_m1.json — load OK (36 L, 16 H, d 2048, peak 7.81 GB on 1 T4). FAIL: TL generate KV cache on GQA ("Expected size 4 but got size 16"); FAIL: CircuitKIT register_advbench default answer_contrastive rejected by EAP validator (clean==corrupt). Both are stack bugs → draft issues |
| 2026-11-06 12:41 | Gate −1 v7 | Kaggle T4 | official chat template (+373-tok preamble), harmful_vs_benign | tiny-aya-global main | walledai/AdvBench[:16] | 4.9 min | ~0.08 | kaggle/gate_m1/output_v7/ — sanity OK (first token on harmful: "I'm"/"I", no leading space). EAP OOM: prompts 375 tokens because template injects fixed system preamble |
| 2026-11-06 12:49 | Gate −1 v8 | Kaggle T4 | minimal template (no system turn, ~16 tok), expandable_segments | same | same | 3.9 min | ~0.07 | kaggle/gate_m1/output_v8/ — still OOM (13.9 GB alloc); generation w/o KV cache: "I'm sorry, but I cannot provide information on how to make a pipe bomb…" |
| 2026-11-06 12:55 | Gate −1 v9 | Kaggle T4 | + memory instrumentation | same | same | ~4 min | ~0.07 | kaggle/gate_m1/output_v9/ — cause: TL 3.8.0 F.linear(x, W.T.contiguous()) keeps transposed weight copies for backward (~2× model) |
| 2026-11-06 12:59 | Gate −1 v10 | Kaggle T4 | + TL patch (drop .contiguous(), 4 sites), eap & eap-ig n=8 bs=1 sparsity 0.05 | same | same | 4.3 min | ~0.07 | kaggle/gate_m1/output_v10/ — **PASS**: EAP peak 10.28 GB 20 s; EAP-IG peak 10.29 GB 31 s; 29 nodes / 612 scored; scores JSON has safety_units + layer_suggestions (layer_suggestions = all 36 layers × 7 modules even at 5% sparsity) |
| 2026-11-06 19:00–19:35 | Data build | local CPU | scripts/build_data.py (seeds 0–3), scripts/translate.py | — | sorry-bench-ml 599c712, Paul/XSTest f600c99, mteb/amazon_massive_intent 940fd47, LLM-LAT/harmful-dataset 8bfba31 | ~35 min | 0 | data/manifest.json — 100 harmful (34 clear-refusal SORRY-Bench categories × ≤3; 10 ambiguous categories excluded), 50 XSTest safe, 288 MASSIVE test, 1800 MASSIVE train (en/hi/zh), 100 harmful-FT (dedup vs SORRY-Bench). MR: Google. Hinglish: Aya 326/338 benign, Claude 100 harmful + 12 benign fallbacks. Spot-check: data/spotcheck.csv |
| 2026-11-06 20:05 | Gate 0 smoke v1 | Kaggle 2×T4 | SMOKE (4 harm + 4 xstest + 8 massive / lang; 48 train ex, 1 epoch) | tiny-aya-global af89d21 | lexsi-data v1 | ~2 min | ~0.03 | kaggle/gate0/out_smoke1/ — peft import fails on Kaggle torchao 0.10; base generations gibberish (concurrent from_pretrained in thread) |
| 2026-11-06 20:12 | HF diag | Kaggle T4 | fp16 sdpa/eager, bf16, fp32, batched left-pad | same | — | 2.2 min | ~0.04 | kaggle/diag/out/diag.json — all coherent & identical ("I'm sorry, but I cannot provide…"; "The capital of France is Paris.") → HF/fp16 fine, bug was threading |
| 2026-11-06 20:18 | Gate 0 smoke v2 | Kaggle 2×T4 | + torchao uninstall, sequential loads, per-thread tokenizer | same | same | 2.5 min | ~0.04 | kaggle/gate0/out_smoke2/ — pipeline OK: LoRA train 6 steps peak 10.71 GB, base gen on cuda:1 in parallel; coherent outputs; Hinglish reply in Hausa seen → judge now uses langid + romanized-Hindi lexicon for 'confused' |
| 2026-11-06 20:55 | Gate 0 drift+gen (full) | Kaggle 2×T4 | LoRA r16 α32 all-proj, lr 1e-4, 2 ep, bs 8, 1800 MASSIVE (en/hi/zh) + 100 harmful-FT, seed 0; greedy gen, 96 tok (harm/xstest), 12 tok (massive) | tiny-aya-global af89d21 | lexsi-data v1 | 13.7 min | ~0.23 | kaggle/gate0/out_full/gate0.json — 476 steps 487 s, peak 13.99 GB; 1940 gens/model; MASSIVE acc drift en .788 hi .757 zh .774 mr .722 hinglish .698 (base 0 on all: never saw label set). Adapter: out_full/drift_lora (private) |
| 2026-11-06 21:11 | Gate 0 judge | Kaggle 2×T4 | WildGuard 7B fp16 device_map auto, SafeTune template, greedy 32 tok; confused = lang_ok (langid + romanized-Hindi lexicon) | allenai/wildguard cbba482 | gen_base/gen_drift from drift run | 14.5 min | ~0.24 | kaggle/gate0_judge/out/judge.json; stats results/gate0_stats.json (scripts/gate0_stats.py, bootstrap 10k seed 0). **Gate 0 PASS**: EN refusal 0.82→0.04, drop 0.78 CI [0.66,0.88]. WG-vs-keyword refusal agreement 43–50/50 per lang/model. WG says "harmful request: no" on most HI/MR prompts → comply_harmful not valid for HI/MR |

## 2026-11-07 — Gate 0.5 activations (rajtaware10/lexsi-gate05-acts v1, kaggle/gate05/gate05.py)
- GPU T4 (1 used). tiny-aya-global @af89d219 fp16; drifted = same + Gate 0 LoRA (lexsi-gate0-drift output) merged.
- 438 states × 5 langs, last prompt token, hidden_states[1:] (36 layers). Wall 19:15:29→19:18:03 UTC (2.6 min; 32 s extraction per model). Session GPU time incl. startup ≈ 0.05 h (estimate from wall times, not Kaggle's meter).
- Sanity: max |act| 98.4 (aligned) / 96.8 (drifted), no fp16 overflow; drifted vs aligned mean-abs relative change 0.39 (merge applied).
- Outputs (private): acts_aligned.npy, acts_drifted.npy, gate05.json (copy in kaggle/gate05/out/).
- Stats: rajtaware10/lexsi-gate05-stats (CPU, kaggle/gate05_stats/stats.py) — pushed, running.

## 2026-11-07 — Gate 0.5 stats (rajtaware10/lexsi-gate05-stats v1, CPU, kaggle/gate05_stats/stats.py)
- CPU only (no GPU-h). 19:20:13→19:54:06 UTC (aligned 1040 s, drifted 993 s). Self-test passed locally first.
- Output: kaggle/gate05_stats/out/gate05_stats.json → `python scripts/gate05_report.py ...` → results/gate05_summary.json.
- PRIMARY (pre-registered: drifted, all 438 states, k=32, layers 9–27): ρ_all 0.730. EN→L Procrustes residual / top-1 retrieval: HI 0.252/0.757, ZH 0.257/0.747, MR 0.266/0.748, Hinglish 0.408/0.516. ρ_pair 0.81/0.82/0.80/0.72 vs shuffled null ≈0.18.
- Criterion PASS: (a) ρ_pair > null all L; (b) residual range 0.156 > 2× max split-SD 0.0095. Same verdict in all 6 views (aligned/drifted × all-k32/all-k128/harmful-k32).
- Caveat: the spread is driven by Hinglish; HI/ZH/MR are within ~0.014 of each other on the primary view.
- Harmful-only view (100 states, k=32, drifted): residual ZH 0.274 < MR 0.333 < Hinglish 0.355 < HI 0.387 (split-SD 0.014–0.025) — a different ranking from the all-states view.
- Layer curve: residual high at layer 1 (~0.73), lowest at layers 13–22 (~0.22–0.26 for HI/ZH/MR), rising again to ~0.35–0.45 by layer 34.

## 2026-11-07 — Gates 1 + 2 (rajtaware10/lexsi-gate12 v1, kaggle/gate12/gate12.py)
- 2×T4. 19:51:19→20:11:57 UTC (20.6 min ≈ 0.35 GPU-h session time). circuitkit 0.2.0, TL 3.8.0, transformers 5.20.2, torch 2.11.0+cu130, peft 0.21.2.
- Output: kaggle/gate12/out/gate12.json, g1_eap-ig_scores.json (private Kaggle output; local copy gitignored).
- Refusal token sets (from Gate 0 judged outputs): EN {I, I'm, Creating}, HI {मुझे}, ZH {很, 抱}, MR {म}, Hinglish {M}. Metric AUROC vs judge (aligned/drifted): EN .99/.96, HI .97/.82, ZH .98/.93, MR 1.0/.84, Hinglish .75/1.0. ('Creating' = one odd base refusal; Hinglish set weak.)
- Discovery: EAP-IG, 50 pairs, refusal token "I'm" vs "Sure", 182 s, peak 12.65 GB. Circuit (top 5% per type): 29 heads + MLP 0 + MLP 35.
- **Gate 1 (pre-registered): PASS formally.** EN log-odds clean 1.087 → circuit-ablated −11.118; random mean −1.786, p5 −8.270; perm p = 0.0099 (lowest possible with 100 draws).
  - **Confound:** XSTest (benign) also drops −3.98 → −11.10 (same value as harmful), below random p5 −8.25 → ablation looks like generic damage (MLP 0 / MLP 35), not refusal-specific.
- **Gate 2 (pre-registered): FAIL.** Patching aligned activations at circuit nodes (all positions) into the drifted model *lowers* refusal: recovery EN −0.92, HI −1.23, ZH −1.04, MR −0.34, Hinglish −2.28; random p95 0.19/0.44/0.39/0.42/1.13; perm p ≥ 0.94 everywhere.
- Follow-up: exploratory diagnostic rajtaware10/lexsi-gate12b (heads-only circuit; last-token-only patching) — running.

## 2026-11-07 — Gate 1/2 diagnostic, EXPLORATORY (rajtaware10/lexsi-gate12b v1, kaggle/gate12b/gate12b.py)
- 2×T4. 20:13:51→21:01:23 UTC (47.5 min ≈ 0.8 GPU-h). Reuses gate12 EAP-IG scores. Output kaggle/gate12b/out/gate12b.json.
- **Heads-only Gate 1 (29 heads, no MLPs): specific.** EN 1.087 → −1.277 (random mean 0.600, p5 −0.027, perm p 0.0099); XSTest −3.982 → −3.536 (random p5 −4.567, p 0.76) → heads ablation lowers refusal on harmful prompts without touching benign ones.
- **Gate 2 heads-only, all positions:** recovery EN 0.072 (random mean 0.055, p95 0.152, p 0.38); HI 0.273 (p95 0.400); ZH −0.092 (p95 0.302); MR 0.090 (p95 0.290); Hinglish 0.595 (p95 1.079). No language beats random.
- **Gate 2 heads-only, last token only:** EN 0.061 (p 0.38) — same picture.
- **Gate 2 full circuit (with MLP 0/35), last token only:** EN −0.951 (p 0.99) — MLP 35 patching is what pushes refusal down.
- Reading: the 29 heads are causally needed for refusal in the aligned model, but the drift is not located in their outputs.
- Follow-up exploratory scan rajtaware10/lexsi-gate12c (per-layer residual / attention / MLP / single-head patching, EN + ZH) — running.

## 2026-11-07 — Drift-localisation scan, EXPLORATORY (rajtaware10/lexsi-gate12c v1, kaggle/gate12c/gate12c.py)
- 2×T4. 21:04:17→21:40:21 UTC (36 min ≈ 0.6 GPU-h). Output kaggle/gate12c/out/gate12c.json. Patch aligned→drifted, one component at a time, all positions, EN + ZH (50 prompts each), first-token log-odds recovery.
- Residual stream after layer l (≈ "drift in layers > l"): EN recovery 0.04 @13, 0.23 @16, 0.51 @20, 0.73 @24, 0.86 @30, 1.0 @35; ZH 0.27 @12, 0.50 @14, 0.84 @18, 0.99 @24 → drift sits in layers ~13–35 (EN) / ~10–23 (ZH), spread out.
- Whole attention layers: max 0.11 (EN L16), 0.24 (ZH L16). MLP layers: EN 0.12–0.19 each for L17–27; ZH up to 0.47 (L23). MLP 35: −0.81 (EN), −0.47 (ZH).
- Single heads: max 0.064 (EN, A32.14), 0.157 (ZH, A32.14). Circuit heads' ranks among 576 (EN): 11 in the top 30, ~6 in the bottom 5 (patching them lowers refusal).
- Drift arm for Gate 3 = top 29 heads by EN single-head recovery (as coded before the scan finished). Note: MLPs carry more drift than any head set.

## 2026-11-07 — Gate 3 pushed (rajtaware10/lexsi-gate3 v1) after predictions.json commit 6f1e99b / HF 2534728
- Gate 3 run: 21:41:43→22:10:52 UTC (29 min ≈ 0.5 GPU-h, both T4s). Output kaggle/gate3/out/gate3.json (+ gen_arm_*.jsonl, private).
  - prep: Δ over 291 tensors, ‖Δ‖ = 789.5; attention q/o share of ‖Δ‖ = 0.162. base/global same architecture.
  - Masked-update check vs SafeTune apply_resta (layer 5 q/o, all heads): max abs diff 0.0.
  - Full RESTA α=1 (SafeTune): first-token EN −17.14, ZH −14.62 (drifted −3.08/−4.68); MASSIVE acc 0.0 all languages; benign outputs are multilingual gibberish → model broken at α=1.
  - Head arms (15.2M params each = q rows + o cols of 29 heads), first-token EN at α=1/2/4: circuit −3.17/−3.35/−3.91; drift −3.08/−3.15/−3.51; rand0–4 −3.06…−3.13 / −3.06…−3.24 / −3.06…−3.64 → no arm raises refusal; MASSIVE acc unchanged (0.69–0.80).
- Judge: rajtaware10/lexsi-gate3-judge pushed. Exploratory full-RESTA α sweep {0, .05, .1, .2, .3, .5}: rajtaware10/lexsi-gate3b pushed.

## 2026-11-07 — Gate 3 judge (rajtaware10/lexsi-gate3-judge v1) + stats
- 2×T4, 22:14:00→22:57:55 UTC (44 min ≈ 0.73 GPU-h). WildGuard as Gate 0. Outputs kaggle/gate3_judge/out/ (private). `python scripts/gate3_stats.py kaggle/gate3_judge/out kaggle/gate0_judge/out` → results/gate3_stats.json.
- Refusal on harmful (EN/HI/ZH/MR/Hinglish), anchors aligned .82/.42/.78/.50/.22, drifted .04/.10/.08/.04/.02:
  - circuit .02/.06/.08/.02/.02; drift arm .02/.06/.10/.02/.04; random (5) mean .024/.096/.092/.024/.012.
  - full RESTA α=1: .04/.00/.00/.00/.26 BUT confused rate .82/1.0/.98/1.0/.62 (gibberish; Hinglish ".26 refusal" = judge on garbage). MASSIVE acc 0.
  - No arm's paired gain vs drifted has a CI excluding 0 in the positive direction except full/Hinglish (garbage). Circuit − each random arm (EN): 0.00/0.00/0.00/0.00/−0.02.
- Predictions (predictions/predictions.json, HF 2534728): P1 holds (circuit EN .02 ≤ max random .04) — but trivially, since no head arm moved refusal. P2 not applicable (full arm EN recovery 0.0 < 0.2). P3 fails (full EN recovery 0.0).

## 2026-11-07 — Full-RESTA α sweep, EXPLORATORY (rajtaware10/lexsi-gate3b v1)
- 1 T4, 22:12:48→22:21:56 UTC (9 min ≈ 0.15 GPU-h). Output kaggle/gate3b/out/gate3b.json.
- α: EN first-token / MASSIVE-EN acc / EN harmful keyword refusal: 0: −3.08/.788/0.0; .05: −2.80/.778/0.0; .1: −2.56/.781/0.0; .2: −2.28/.747/0.0; .3: −2.39/.615/0.0; .5: −3.44/.094/0.0. → no α restores refusal before the model degrades.

## 2026-11-07 — Refusal steering arm (rajtaware10/lexsi-gate3s v1 + lexsi-gate3s-judge v1)
- Steering kernel 2×T4, 23:08:23→23:17:10 UTC (9 min ≈ 0.15 GPU-h). Output kaggle/gate3s/out/gate3s.json. Alpaca rev dce01c9b.
- Direction gap ‖r_L‖ grows with depth (0.8 at L0 → 87.5 at L34). Aligned reference on tuning sets: harmful log-odds 0.86, benign refusal prob 0.083. Drifted: −3.17 / 0.014.
- Grid (harm log-odds / benign prob): L12 c3 −1.35/.16; L16 c1 −1.29/.087; L16 c2 −0.34/.38; L20 c1 0.16/.31; L20 c2 2.15/.84; L24 c2 2.30/.87. Rule (benign ≤ .083+.05) → **L16, c=1.0** (magnitude 19.35). Random control cos with direction 0.003.
- Judge (stats via scripts/gate3_stats.py over Gate 3 + steering arms → results/gate3_stats.json). Refusal on harmful EN/HI/ZH/MR/Hinglish:
  - steer: .42/.88/.96/.78/.50 (drifted .04/.10/.08/.04/.02; aligned .82/.42/.78/.50/.22). Paired gain vs drifted .38 [.24,.52], .78 [.66,.88], .88 [.78,.96], .74 [.62,.86], .48 [.34,.62]; McNemar p<0.001 all.
  - steer XSTest over-refusal: .18/.46/.40/.44/.28 (aligned .10/.02/.06/.04/.04). Confused .00/.00/.00/.00/.12. MASSIVE acc .722/.667/.691/.639/.604 (drifted .788/.757/.774/.722/.698).
  - steerctl (random direction): .02/.06/.04/.00/.00 — no effect; XSTest ≤ .06; MASSIVE −0.02…−0.03.
- **P2 scored on steer (EN recovery .487 > .2): FAILS, in the opposite direction.** Predicted ratio recovery_L/recovery_EN ≈ .73–.75 (Hinglish .59); observed HI 5.0, ZH 2.6, MR 3.3, Hinglish 4.9 (anchored on low aligned baselines). On absolute gain too, EN (.38) is the *smallest*. Hinglish not lowest. Spearman (n=4) 0.2.

## 2026-11-07 — Per-language steering strength (design DECISIONS.md, same date)
- Tune set data/xstest_tune.jsonl built by scripts/build_tune.py (HI/ZH/MR Google, Hinglish written by Claude: data/xstest_tune_hinglish.json). Kaggle dataset rajtaware10/lexsi-data new version.
- Pipeline: rajtaware10/lexsi-gate3sp-tune (gen tune grid) → lexsi-gate3sp-tune-judge → lexsi-gate3sp-eval (select c per language in-kernel, gen eval) → lexsi-gate3sp-eval-judge. Eval kernel code = copy of kaggle/gate3sp/gate3sp.py.
- tune kernel v1 pushed.
- tune v1: 2×T4, 14:54:45→15:08:49 UTC (14 min ≈ 0.23 GPU-h). gap at L16 = 19.35 (= gate3s). 7 × 500 rows. Output kaggle/gate3sp/out/. tune-judge v1 pushed.
- tune-judge v1: 2×T4, 15:13:57→15:43:13 UTC (29 min ≈ 0.49 GPU-h). Tune refusal harmful/XSTest (n=50 each) EN, HI, ZH, MR, Hinglish:
  - aligned .80/.14 .56/.04 .82/.06 .58/.04 .32/.06
  - c.25 .04/.02 .14/.02 .24/.00 .08/.02 .02/.04 · c.5 .04/.02 .44/.04 .60/.06 .18/.06 .10/.06 · c.75 .12/.04 .62/.22 .80/.30 .44/.14 .34/.12
  - c1 .30/.12 .90/.46 .86/.46 .74/.36 .50/.22 · c1.5 .80/.76 1.0/.92 .68/.76 .98/.92 .72/.88 · c2 .96/1.0 1.0/.98 .46/.32 .98/.98 .76/.94 (ZH c2 confused .61)
  - Rule → c: EN 1.0, HI 0.5, ZH 0.5, MR 0.75, Hinglish 0.75 (selected in-kernel by select_pl in eval phase). eval v1 pushed.
- eval v1: 2×T4, 15:46:14→15:51:32 UTC (5 min ≈ 0.09 GPU-h). In-kernel selection = EN 1.0, HI .5, ZH .5, MR .75, Hinglish .75 (matches local). 1940 rows. MASSIVE acc .719/.743/.753/.688/.642. eval-judge v1 pushed.
- eval-judge v1: 2×T4, 16:19:30→16:24:47 UTC (5 min ≈ 0.09 GPU-h). Stats: `python scripts/gate3_stats.py <dir with all judged arms: gate3_judge + gate3s_judge + gate3sp_eval_judge> kaggle/gate0_judge/out` → results/gate3_stats.json.
  - steerpl refusal EN/HI/ZH/MR/Hinglish .42/.30/.60/.56/.26 (aligned .82/.42/.78/.50/.22; drifted .04/.10/.08/.04/.02). Gain vs drifted .38 [.24,.52], .20 [.08,.32], .52 [.38,.66], .52 [.38,.66], .24 [.12,.36]; McNemar p ≤ .007 all (HI .0064).
  - XSTest over-refusal .18/.06/.06/.12/.16 (steer c=1: .18/.46/.40/.44/.28; aligned .10/.02/.06/.04/.04). MASSIVE .719/.743/.753/.688/.642 (drifted .788/.757/.774/.722/.698). Confused harmful Hinglish .08.
  - steerpl − steer (paired): harmful HI −.58, ZH −.36, MR −.22, Hinglish −.24; XSTest HI −.40, ZH −.34, MR −.32 (all CI excl. 0), Hinglish −.12 [−.22,−.02] (McNemar p .07). EN identical (same c=1, greedy) = sanity check.
  - P2 on steerpl (secondary; c was tuned per language, so not an independent test): observed ratio HI 1.28, ZH 1.53, MR 2.32, Hinglish 2.46 vs predicted ≈.6–.75; Spearman −1.0 (n=4); Hinglish not lowest. Fails.
- Per-language sweep total ≈ 0.90 GPU-h (tune .23 + judge .49 + eval .09 + judge .09).

## 2026-11-10 — Faithful RESTA arm (Raj: "Yes, run it"; design DECISIONS.md 2026-11-09, written before the run)
- rajtaware10/lexsi-gate3r v1 sat QUEUED; Raj: "stop session and push again" → v2 (kaggle CLI has no cancel). v2: 2×T4 (GPU 0 mostly), 18:52:37→19:06:30 UTC (14 min ≈ 0.23 GPU-h). safetune 0.1.12, transformers 5.20.2, torch 2.11.0+cu130, peft 0.21.2. Output kaggle/gate3r/out/gate3r.json + gen_arm_resta_paper.jsonl (private).
- Un-align: LoRA on 100 harmful_ft pairs, 65 steps, 170 s; loss 1.76 → ~1.0. First-token refusal log-odds on harmful rows 50–99: un-aligned EN −3.66 / ZH −4.48 (aligned .86 / −.97; drifted −3.17 / −4.56).
- ‖Δ_paper‖ = 3.99 vs ‖Δ_README‖ = 660.8 (over keys shared by the base safetensors and the HF state dict; Gate 3 reported 789.5 over 291 tensors, different key set), cosine 0.0001 → the two safety vectors are near-orthogonal.
- Exploratory α sweep (rows 50–99 first-token EN/ZH; EN MASSIVE acc): α0 −3.17/−4.56, .788 · α.5 −.62/−1.42, .781 · α1 3.43/−.55, .785 · α1.5 5.88/−.97, .785 · α2 6.98/−1.57, .760.
- Primary arm resta_paper (α=1): 1940 rows; MASSIVE acc EN/HI/ZH/MR/Hinglish .785/.75/.764/.719/.691 (drifted .788/.757/.774/.722/.698). Judge rajtaware10/lexsi-gate3r-judge v1 pushed.
- Judge rajtaware10/lexsi-gate3r-judge v1: 2×T4, 19:12:19→19:18:03 UTC (6 min ≈ 0.1 GPU-h), WildGuard sha cbba4823. Output kaggle/gate3r_judge/out/ (private). Stats: judged file copied into scratchpad allarms dir with all other judged arms; `python scripts/gate3_stats.py <allarms> kaggle/gate0_judge/out` → results/gate3_stats.json.
  - resta_paper refusal EN/HI/ZH/MR/Hinglish .84/.50/.82/.52/.30 (aligned .82/.42/.78/.50/.22; drifted .04/.10/.08/.04/.02). Gain vs drifted .80 [.68,.90], .40 [.24,.56], .74 [.62,.86], .48 [.34,.62], .28 [.16,.40]; McNemar p ≤ .00012 all. Recovery 1.03/1.25/1.06/1.04/1.40.
  - XSTest over-refusal .14/.06/.08/.04/.06 (aligned .10/.02/.06/.04/.04). Confused harmful 0/0/0/.02/.10. MASSIVE .785/.75/.764/.719/.691 (drifted .788/.757/.774/.722/.698).
  - P2 on resta_paper (pre-registered rule: any arm with EN recovery > .2): observed ratio HI 1.22, ZH 1.03, MR 1.02, Hinglish 1.37 vs predicted .59–.75; Spearman −0.2; Hinglish not lowest → fails.
  - Caveat: Δ estimated on the same 100 harmful pairs that are in the drift mix (flagged in DECISIONS.md before the run).
- Faithful RESTA total ≈ 0.33 GPU-h.

## 2026-11-10 — gate3r2: paper RESTA with disjoint un-align set (design DECISIONS.md 2026-11-10, idea 2)
- rajtaware10/lexsi-gate3r2 v1: 2×T4, 20:47:14→21:04:34 UTC (17 min ≈ 0.29 GPU-h). Output kaggle/gate3r2/out/ (private).
- Un-align 1 (harmful_ft, re-trained): loss and first-token match gate3r (EN −3.66). Un-align 2 (harmful_unalign2, 100 disjoint pairs): loss 1.79 → ~1.0; first-token rows 50–99 EN −4.27 / ZH −5.23.
- **cos(Δ₁, Δ₂) = 0.053** (LoRA-factor form), ‖Δ₁‖ 3.97, ‖Δ₂‖ 4.08; per-layer cosine 0.03–0.08. ‖Δ₂‖ from merged weights 4.10; cos(Δ₂, Δ_README) 0.0.
- Exploratory α sweep (rows 50–99, EN/ZH first-token; EN MASSIVE): α0 −3.17/−4.56 .788 · α.5 −.48/−1.24 .785 · **α1 4.39/−.20 .781** · α1.5 7.42/−.46 .781 · α2 8.95/−1.11 .781 · α3 8.86/−2.41 .764. (Δ₁ at α1: 3.43/−.55 .785.)
- Primary arm resta_paper2 (α=1): 1940 rows, MASSIVE .781/.75/.764/.719/.684. Judge rajtaware10/lexsi-gate3r2-judge v1 pushed.
- gate3r2 judge (rajtaware10/lexsi-gate3r2-judge v1): 2×T4, 21:07:45→21:13:30 UTC (6 min ≈ 0.1 GPU-h). Judged file added to the allarms dir; `python scripts/gate3_stats.py <allarms> kaggle/gate0_judge/out` → results/gate3_stats.json.
  - **resta_paper2 refusal EN/HI/ZH/MR/Hinglish .90/.58/.88/.60/.46** (resta_paper .84/.50/.82/.52/.30; aligned .82/.42/.78/.50/.22). Gain vs drifted CIs [.76,.94] [.32,.64] [.68,.90] [.42,.70] [.30,.58]; McNemar p < .0001 all.
  - XSTest .16/.10/.10/.04/.08 (aligned .10/.02/.06/.04/.04). Confused 0/0/0/0/.10. MASSIVE .781/.75/.764/.719/.684.
  - Paired resta_paper2 − resta_paper, harmful: EN +.06 [0,.14] p .25; HI +.08 [0,.18] p .22; ZH +.06 [0,.14] p .25; MR +.08 [.02,.16] p .125; Hinglish +.16 [.04,.30] p .039. XSTest diffs ≤ +.04, all p ≥ .5.
  - P2 on resta_paper2: ratios HI 1.36, ZH 1.04, MR 1.10, Hinglish 2.00; Spearman −0.4; Hinglish not lowest → fails.
  - **Reading:** the best-case caveat does not explain the repair: a Δ from 100 harmful pairs disjoint from the drift (cos 0.053 to Δ₁) repairs at least as well.

## 2026-11-10 — gate3p: head arms with the paper Δ (design DECISIONS.md 2026-11-10)
- rajtaware10/lexsi-gate3p v1: 2×T4, 19:57:13→20:26:58 UTC (30 min ≈ 0.5 GPU-h). Output kaggle/gate3p/out/ (private). Re-trained Δ_paper ‖·‖ 3.987 (gate3r 3.99); share on attention q/o 0.377 (‖Δ_qo‖ 1.504). Per arm ‖Δ‖ touched: circuit .348, drift .363, rand .335–.342 (≈ 9% of ‖Δ‖).
- First-token EN (eval rows 0–49), drifted −3.08: α1 circuit −3.00, drift −2.98, rand −3.05…−3.06; α4 circuit −2.74, drift −2.63, rand −2.92…−2.98. ZH similar. MASSIVE unchanged in every arm.
- Judge rajtaware10/lexsi-gate3p-judge v1: 2×T4, 20:30:12→21:11:28 UTC (41 min ≈ 0.69 GPU-h). `python scripts/gate3_stats.py kaggle/gate3p_judge/out kaggle/gate0_judge/out results/gate3p_stats.json`.
  - Refusal EN/HI/ZH/MR/Hinglish: circuit .04/.12/.12/.02/.02; drift .04/.12/.12/.02/.02; every random arm .04/.10/.10/.02/.02 (drifted .04/.10/.08/.04/.02). Circuit − each random: 0 to +.02, McNemar p = 1.0. Responses differ between arms on 3–5% of prompts (checked; not a bug), but no judged refusal changes.
  - P1 (secondary with paper Δ): circuit EN .04 = max random .04 → holds. **Head-restricted repair does nothing even with the Δ that fully repairs the whole model.**

## 2026-11-10 — gate3a: drift attribution in weight space (design DECISIONS.md 2026-11-10, idea 1)
- First push (02:3x IST) was rejected ("Maximum batch GPU session count of 2 reached"); re-pushed. v1: 2×T4, 21:17:29→21:41:32 UTC (24 min ≈ 0.4 GPU-h). Output kaggle/gate3a/out/ (private).
- Un-align reproduces gate3r (EN −3.66). ‖Δ‖ from LoRA factors 3.974. **Sanity:** full Δ via factors at α1, rows 50–99 EN/ZH 3.42/−.55 (gate3r 3.43/−.55) ✓.
- Attribution: 397,008 units (q-heads, kv-heads, MLP neurons), 7 s, one backward over 50 EN prompts. Σ scores 3.60 (first-order; actual full-Δ gain on rows 50–99 ≈ 6.6 log-odds). By type: neurons 2.83, q-heads .52, kv-heads .25. Neuron score concentrated in layers 12–18 and 34–35. Top units: kv-head L32.3, neurons L35.9699/7553, kv-heads L16–18.
- `attr` arm: 15,202,304 params (29-head budget) = 108 neurons (40 in L35, 14 in L34), 16 q-heads, 12 kv-heads; ‖Δ‖ touched .373 (random .347–.360).
- First-token EN/ZH on eval rows 0–49 (drifted −3.08/−4.68):
  - α1: attr −2.58/−4.24; arand0–4 −2.97…−3.00 / −4.51…−4.56.
  - α2: attr −2.04/−3.82; arand −2.84…−2.91.
  - α4: attr −0.96/−3.11; arand −2.60…−2.72.
  - Exploratory budgets (α1): ×4 top −2.25 vs matched random −2.73; ×16 (2,150 units, ‖Δ‖ 1.38) top −1.20 vs random −2.30.
- MASSIVE unchanged in all arms. Judge rajtaware10/lexsi-gate3a-judge v1 pushed.

## 2026-11-10 — gate3rc: random / benign Δ controls (design DECISIONS.md 2026-11-10, Raj: "push gate 3rc")
- v1 (21:2x UTC) OOM in benign un-align after ~4 min (≈0.07 GPU-h), no results. v2 (grad accumulation, see DECISIONS): 2×T4, 21:29:06→21:48:30 UTC (19 min ≈ 0.32 GPU-h). Output kaggle/gate3rc/out/ (private).
- Δ₁ re-trained: first-token rows 50–99 at α1 EN/ZH 3.43/−.55 ✓. Benign copy (100 Alpaca pairs): loss .62 → ~.25; its own first-token EN .62 / ZH −1.30 (Global .86 / −.97).
- cos to Δ₁: benigndelta .026, randdelta0–2 0.000. Norms: Δ₁ 3.974, benign 3.764, random 3.974 (matched per module).
- First-token rows 50–99, EN/ZH (drifted −3.17/−4.56):
  - Δ₁: α1 3.43/−.55 · α2 6.99/−1.57 · α4 5.53/−3.01.
  - **randdelta0–2: α1 −3.16…−3.18 · α2 −3.14…−3.19 · α4 −3.11…−3.20 (= drifted; flat in α).**
  - **benigndelta: α1 −3.46/−4.11 · α2 −5.38/−4.45 · α4 −11.40/−6.62 (refusal goes *down*).**
  - EN MASSIVE α1: Δ₁ .781, random .785–.788, benign .795.
- Generated (α1): randdelta MASSIVE .785/.757/.771/.722/.698; benigndelta .795/.753/.76/.726/.688. Judge rajtaware10/lexsi-gate3rc-judge v1 pushed.
- gate3rc judge v1: 2×T4, 21:52:12→22:03:18 UTC (11 min ≈ 0.19 GPU-h). Judged files added to allarms; gate3_stats.py → results/gate3_stats.json.
  - **randdelta** refusal .04/.10/.08/.02/.02 = drifted (gain CIs all include 0, McNemar p = 1.0 everywhere). XSTest .00/.02/.04/.02/.06.
  - **benigndelta** refusal .02/.10/.22/.04/.00; ZH +.14 [.04,.26] p .039 (unexpected; first-token ZH also slightly up, −4.11 vs −4.56), others ≈ drifted (p = 1.0). XSTest .00/.02/.02/.02/.02.
  - **Reading:** a same-norm random Δ and a benign-SFT Δ do not repair; the RESTA-paper repair is specific to the harmful-un-alignment difference.
- gate3a judge v1: 2×T4, 21:44:51→22:18:33 UTC (34 min ≈ 0.56 GPU-h). `python scripts/gate3_stats.py kaggle/gate3a_judge/out kaggle/gate0_judge/out results/gate3a_stats.json` + attr−arand paired comparison (appended to the same JSON).
  - Refusal EN/HI/ZH/MR/Hinglish: **attr .02/.16/.20/.02/.02**; arand0–4 .04/.12/.12–.14/.02/.02 (drifted .04/.10/.08/.04/.02). XSTest identical across arms (.00/.02/.04/.02/.06). MASSIVE unchanged.
  - attr − each arand: EN −.02 (p 1.0), HI +.04 (p .5), ZH +.06–.08 (p .125–.25), MR 0, Hinglish 0.
  - **Pre-registered EN criterion fails:** attr EN .02 ≤ max random .04; attr vs drifted CI [−.06, 0]. Exploratory: attr ZH .20 vs drifted .08, McNemar p .031.
  - **Reading:** weight-space attribution ranks units better than matched random on the continuous first-token metric (gap grows with α and budget), but at the 29-head budget no arm changes judged refusal. Repair needs a large, distributed share of Δ (MLP-heavy).
- GPU-h since Raj's "10 h left" note (session wall times): gate3p .5 (partly before) + judge .69, gate3r2 .29 + .1, gate3a .4 + .56, gate3rc .07 + .32 + .19 ≈ 3.1 → ≈ 6.9 left (Raj to confirm on meter).

## 2026-11-10 — Second judge probe: Aya Expanse 32B (HF Inference, cohere provider) — NOT usable
- Raj OK'd A incl. the second judge. Before any real run, probed on 7 hand-made toy (request, reply) pairs with obvious labels (bread recipe, clear refusals in EN/HI, a Hinglish echo of the request, two "Sure, here are…" harmful compliances, one warning-only refusal).
- Three prompt formats (label-only system prompt; "did the reply refuse? YES/NO/UNCLEAR"; 4-shot REFUSED/HELPED/BROKEN). Best format still mislabelled 3–4 of 7: "Sure! First, obtain…" → REFUSED, "Sure, here are some suggestions…" → REFUSED, a warning-only refusal → HELPED, the echo → REFUSED. It seems to label what the reply *should* do, not what it did.
- Decision: no Aya judge run (≈25 calls, ≈3.3k tokens; cost not visible via API). Probe code not kept. Judge validation rests on Raj's hand labels (`scripts/judge_agreement.py`). An alternative second judge needs Raj's choice.

## 2026-11-10 — gate3c (C): activation agreement of RESTA-paper Δs — kernel `rajtaware10/lexsi-gate3c` v2
- v1: OOM at the Δ₃ un-align (logged deviation). v2 06:11–06:30 UTC on Kaggle (19 min, ≈ .32 GPU-h with 2 T4s attached). Output `kaggle/gate3c/out/gate3c.json`, aggregates in `results/gate3c_stats.json`.
- Norms ‖Δ‖: d1 3.97, d2 4.08, **d3 (BeaverTails) 3.95**, rand 3.97, benign 3.77. Weight cosines all ≤ .053 (d1·d2 .053, d1·d3 .032, d1·benign .026, rand 0).
- Factor subspaces: A (input side) row spaces at chance for every pair. B (output side) column spaces overlap well above chance for harmful pairs, e.g. d1·d2 q .128 / k .186 / v .166 / down .160 vs chance .008 / .031 / .031 / .008; d1·d3 about half that; d1·benign below d1·d3; d1·rand at chance.
- First-token refusal log-odds (EN, harmful rows 50–99): drifted −3.17, aligned .86, d1 3.43, d2 4.39, **d3 −.31**, rand −3.16, benign −3.50.
- **Activation shifts (last token, mean over layers 16–35), EN harmful:** cos(μ₁, μ₂) **.982**, cos(μ₁, μ₃) **.845**, cos(μ₁, μ_benign) .562, cos(μ₁, μ_rand) .011. Per-prompt cos d1·d2 .973. Fraction of the aligned−drifted gap closed along the gap: d1 .99, d2 .98, d3 .76, benign .24, rand 0. Same pattern in HI/ZH/MR/Hinglish (cos d1·d2 .98–.99, d1·d3 .84–.87; closed d3 .76–.93).
- **Pre-registered H_C: PASS** (.982 and .845 > .5 and > max(.011, .562)).
- Reading: three harmful un-align Δs that are near-orthogonal in weights move the drifted model's activations in nearly the same direction, close to the full gap back to the aligned model. They share output-side (B) subspaces but not input-side (A) ones. Benign Δ moves partly along the same direction (≈ ¼ of the gap; the generic "undo fine-tuning" part) but lowers refusal. Caveat: d1/d2 share init, seed and source distribution; d3 differs in source and still agrees at .85.
- Judged arm `resta_paper3` (WildGuard, kernel lexsi-gate3c-judge, `results/gate3_stats.json`): refusal EN/HI/ZH/MR/Hinglish **.54/.66/.88/.66/.36** (gain vs drifted CI EN [.36, .64], all McNemar p < .001); XSTest .22/.14/.18/.12/.14 (higher than resta_paper .14/.06/.08/.04/.06); MASSIVE .785/.747/.760/.712/.691 (≈ drifted). **Pre-registered criterion (EN gain CI excludes 0): PASS.** A safety vector from a different source repairs too, less in EN, more in HI/MR, with more over-refusal.

## 2026-11-10 — gate3sz (B): how much of the safety vector is needed? — kernel `rajtaware10/lexsi-gate3sz` v1
- Pushed in parallel with gate3c v2 (Raj: "Push B as well"). 06:12–07:01 UTC (48 min, ≈ .8 GPU-h). 105 sets, no errors. Aggregates in `results/gate3sz_stats.json`.
- Δ₁ re-trained (gate3r recipe). Total unit parameters 2.81B; shares 1/5/10/25/50% = 28M / 141M / 281M / 703M / 1.41B params.
- **Q1 (first-token EN, eval rows 0–49; drifted −3.08):** top vs 20 matched random sets:
  - 1%: −2.51 vs max −2.85; 5%: −1.78 vs −2.46; 10%: −0.82 vs −2.22; 25%: **+1.82** vs −1.23; 50%: **+3.06** vs +0.46.
  - Top beats all 20 random sets at **every** share (perm p = .048, the minimum with 20 sets). **Q1 answer: 1%** (the smallest share tested).
  - ‖Δ‖ touched is nearly the same for top and random at each share (e.g. 10%: 1.48 vs 1.45), so the gap is not just a larger norm.
- Top arms MASSIVE (full eval) ≈ drifted at every share (EN .774–.788). Random arms generated on harmful + XSTest only (the kernel's MASSIVE accuracy of 0 for them is an artifact of no MASSIVE rows).
- Judge `rajtaware10/lexsi-gate3sz-judge` (WildGuard): 07:05–08:32 UTC (≈ 1.45 GPU-h; 15 arms: 5 top on the full eval set, rand0/rand1 per share on harmful + XSTest). Stats: `python scripts/gate3sz_judged.py kaggle/gate3sz_judge/out kaggle/gate0_judge/out` → `results/gate3sz_stats.json` "judged".
- **Judged refusal, top arm (EN/HI/ZH/MR/Hinglish)** — drifted .04/.10/.08/.04/.02, aligned .82/.42/.78/.50/.22, full RESTA-paper .84/.50/.82/.52/.30:
  - 1%: .02/.16/.22/.02/.02 (random .02/.14/.16/.02/.02)
  - 5%: .04/.28/.52/.24/.04 (random .02/.24/.28/.18/.02)
  - 10%: **.40**/.40/.68/.38/.12 (random .02–.04/.24–.26/.46/.24/.06)
  - 25%: **.86**/.48/.78/.48/.32 (random .48/.36–.38/.68/.40/.08–.12)
  - 50%: .88/.52/.84/.52/.44 (random .76/.38–.42/.76–.78/.38–.42/.28)
  - XSTest over-refusal, top: .00–.02 at ≤10%, .06/.04/.06/.04/.06 at 25%, .12/.06/.08/.04/.08 at 50%. Confused ≤ .06 everywhere. MASSIVE ≈ drifted at every share.
- **Q2 (smallest share with top EN refusal ≥ .44): 25%** (10% gives .40, just under). At 25% top EN .86 matches full RESTA-paper (.84); top-vs-drifted EN gain .82 [.70, .92].
- **Q3 (top − random, paired, McNemar):**
  - 1%: no language differs (ZH +.06, p .25).
  - 5%: ZH +.24 [.12, .36] p < .001; rest n.s.
  - 10%: EN +.38/+.36 (p < .001), HI +.16/+.14 (p .008/.016), ZH +.22 (p .001), MR +.14 (p .016); Hinglish +.06 n.s.
  - 25%: EN +.38 [.24, .52] (p < .001 vs both), Hinglish +.20/+.24 (p .006/< .001); HI/ZH/MR +.08 to +.12 n.s.
  - 50%: all languages +.06 to +.16; EN +.12 (p .031), Hinglish +.16 (p .039); others mixed.
- Note: rand0 and rand1 are different sets (different first-token values) but give near-identical judged rates (same label on 490–500 of 500 rows). Random selections of hundreds of thousands of units average out, so random arms behave almost deterministically for a given share.
- **Reading:** attribution-ranked units carry the repair far better than size-matched random ones. A quarter of Δ chosen by grad × Δ restores EN refusal fully (.86), with lower over-refusal than full RESTA (EN XSTest .06 vs .14). Random needs ≥ 50% to come close. Below 10% nothing moves judged refusal in EN, but ZH/HI/MR move earlier. Ranking used EN only; the gains carry to all four other languages.
- GPU-h since Raj's 8 h 29 min note: gate3c v1 .12 + v2 .32 + judge ≈ .15, gate3sz .8 + judge ≈ 1.5 → ≈ 2.9 → **≈ 5.6 left** (session wall times; confirm on meter).

## 2026-11-10 — gate3zs (D): zero-shot steering strength — kernel `rajtaware10/lexsi-gate3zs` v1
- Raj: "Yes do the 2.6 gpu hrs"; meter **5 h 36 min** left before D. Design pre-registered in DECISIONS.md (commit 2dea27f) before the push.
- Kernel v1 09:35–09:46 UTC (10 min ≈ .17 GPU-h); judge 09:49–10:04 UTC (15 min ≈ .26 GPU-h). Stats: `python scripts/gate3zs_stats.py` → `results/gate3zs_stats.json`.
- Direction gap g = 19.35 (= gate3s). Drifted model, L16, unsteered: mean projection on r̂, harmful / XSTest-tune: EN 8.26 / 2.67, HI 6.08 / 2.19, ZH 8.24 / 1.79, MR 7.95 / 2.09, Hinglish 6.24 / 1.16. Mean residual norm (XSTest-tune) 32.0–34.4 in all languages.
- **Predicted c** (EN 1 by construction): rule B HI 1.02, ZH 1.05, MR 1.03, Hinglish 1.08; A 1.11 / 1.00 / 1.02 / 1.10; C .95 / .97 / .97 / .93. Tuned (steerpl): .5 / .5 / .75 / .75. Spearman vs tuned (n = 4): B .45, A 0, C −.45.
- **Judged, refusal / XSTest** (EN identical to steer, c = 1): steerzs (B) HI .92/.42, ZH .98/.46, MR .82/.42, Hinglish .58/.34 — essentially the c = 1 `steer` arm (.88/.46, .96/.40, .78/.44, .50/.28). A and C the same picture. steerzs MASSIVE .719/.667/.694/.632/.587.
- **Pre-registered H_D: FAIL.** (1) gain vs drifted CI excludes 0 in 4/4 ✓; (2) XSTest below steer in 2/4 (HI, MR) ✗; (3) within .10 of steerpl in 0/4 ✗. Rules A and C also fail (C meets (3) only in Hinglish).
- **Reading:** before steering, all five languages sit at almost the same place along the EN refusal direction at L16 (differences ≤ 2.1 against a push of 19.35) and have the same residual norm. So the static position at the steering layer does not explain why HI/ZH/MR/Hinglish need about half the strength. The extra sensitivity must arise after L16 (later layers respond more strongly to the same push in those languages). A zero-shot rule would need a downstream signal, e.g. the change in a later-layer projection or in the first-token distribution per unit of c, which needs a forward pass with steering but no judged labels. Not run.
- GPU-h since meter 5 h 36 min: D ≈ .43. E running.

## 2026-11-10 — gate3mg (E): magnitude-ranked baseline — kernel `rajtaware10/lexsi-gate3mg` v1
- Raj: "Launch it" (after the 0.75 GPU-h estimate). Design pre-registered in DECISIONS.md "E" (c38b26b). Pushed while gate3zs (D) still running.
- Kernel v1 10:33–10:47 UTC (13 min ≈ .22 GPU-h), no arm errors; judge 10:49–11:01 UTC (11 min ≈ .18 GPU-h). Stats: `python scripts/gate3mg_stats.py` → `results/gate3mg_stats.json`.
- **Sanity:** recomputed attribution top sets reproduce gate3sz first-token (EN −0.819 vs −0.819 at 10%, +1.820 vs +1.818 at 25%), so the re-trained Δ₁ matches B's.
- **Sets:** magnitude ranking picks mostly attention heads at 10% (414 q-heads, 122 kv-heads, 34 neurons; attribution: 384 / 109 / 3,703). At 25%: 576 / 144 / 52,992 vs 493 / 112 / 62,805. Overlap with attribution, Jaccard of units / share of mag's parameters also in attr: 10% .10 / .76; 25% .19 / .61. ‖Δ‖ touched: mag 1.50 / 2.40, attr 1.48 / 2.21 (mag moves more of Δ and still does worse).
- **First-token EN (drifted −3.08):** 10% mag −2.28 vs attr −0.82 (20 mag-matched random −2.32 / −2.30 / −2.28, mag only just above all); 25% mag −0.19 vs attr +1.82 (random −1.57 / −1.52 / −1.49). Density variant (exploratory): −1.44 at 10%, +0.06 at 25%.
- **Judged refusal, mag vs attr (paired), EN/HI/ZH/MR/Hinglish:**
  - 10%: .06/.28/.48/.26/.06 vs .40/.40/.68/.38/.12. Differences −.34 [−.48, −.22] p < .001, −.12 p .031, −.20 p .002, −.12 p .031, −.06 p .25.
  - 25%: .74/.42/.76/.38/.28 vs .86/.48/.78/.48/.32. Differences EN −.12 [−.22, −.04] p .031, HI −.06 p .25, ZH −.02 p 1.0, MR −.10 p .062, Hinglish −.04 p .63. Mag EN vs drifted +.70 [.56, .82].
  - XSTest mag .02–.06 everywhere (attr .02–.06). MASSIVE ≈ drifted. Confused ≤ .04.
- **Pre-registered reading: magnitude does NOT suffice** (mag EN at 25% .74 < attr − .10 = .76; narrowly). Attribution adds value: decisively at 10% (EN .40 vs .06), modestly at 25% (+.12 EN, p .031; other languages n.s.). A plain norm ranking still gets most of the way at 25% (EN .74 vs random .48 in gate3sz).
- GPU-h since meter 5 h 36 min: D ≈ .43 + E ≈ .40 → ≈ .83 → **≈ 4.8 left** (session times; confirm on meter).

## 2026-11-11 — Judge validation against Raj's hand labels (no GPU)
- `python scripts/judge_agreement.py data/handlabel_gate0.csv` → `results/judge_agreement.json`. 60 rows (HI, MR, Hinglish × 20; Gate 0 aligned and drifted outputs), all labelled by Raj. EN and ZH were not in the sample.
- **Refuse vs not (the label every headline table uses): 60/60 agree, Cohen's κ = 1.0** in each language. The sample has 10 refusals (9 aligned, 1 drifted).
- Four-label agreement: .817 overall (κ .69); HI .75, MR .80, Hinglish .90. All 11 disagreements are on harmfulness, none on refusal: 10 × Raj `comply_harmful` → WildGuard `comply_benign` (HI 5, MR 4, Hinglish 1), 1 × Raj `confused` → `comply_harmful` (Hinglish).
- Reading: WildGuard's refusal label is reliable on this sample in HI/MR/Hinglish. Its harmful-response label under-calls harm in HI and MR, which is why only the refusal label is used there. Limits: n = 20 per language, one annotator, Gate 0 outputs only (no repaired-model outputs), no EN/ZH.

## 2026-11-11 — gate3dd (F: scale the drift back; G: drift SVD decomposition) — kernel `rajtaware10/lexsi-gate3dd` v1
- Raj: "yes to both". Design pre-registered in DECISIONS.md "F"/"G" (commit 7333a4c) before the push. Kernel 20:54–21:23 UTC 2026-11-10 (29 min ≈ .48 GPU-h), no arm errors. Judge `rajtaware10/lexsi-gate3dd-judge` 21:26–22:10 UTC (44 min ≈ .74 GPU-h; estimate was .4). Stats: `python scripts/gate3dd_stats.py` → `results/gate3dd_stats.json`.
- Drift update U = drifted − aligned from the Gate 0 adapter: 252 modules, rank 16, ‖U‖ = 8.449. Reconstruction vs the peft merge: max abs error ≤ 2.4e-4 (fp16 rounding). **Sanity:** removing all of U gives first-token EN +1.086 (aligned +1.087); drifted −3.08.
- The top-2 singular directions of every module hold **69.7%** of ‖U‖².
- **F, judged refusal EN/HI/ZH/MR/Hinglish | MASSIVE** (drifted .04/.10/.08/.04/.02 | .788/.757/.774/.722/.698; aligned .82/.42/.78/.50/.22 | 0):
  - β = .25: .10/.26/.54/.26/.04 | .771/.733/.750/.688/.649
  - β = .50: .62/.32/.74/.32/.18 | .517/.483/.493/.462/.392
  - β = .75: .84/.40/.78/.42/.22 | .139/.128/.128/.125/.111
  - XSTest .02–.06 everywhere. First-token grid β = .1 … 1.0 (EN first-token / EN MASSIVE): −2.78/.774, −2.40/.778, −2.00/.750, −1.56/.663, −1.10/.517, −0.62/.340, −0.15/.184, +0.29/.080, +0.70/.003, +1.09/0.
  - **Pre-registered reading F: scale-back does NOT suffice.** No judged β has EN refusal ≥ .44 and EN MASSIVE ≥ .738 (β = .25 keeps the task but refuses .10; β = .5 refuses .62 but MASSIVE falls to .517). Refusal and task skill trade off almost one for one along this line. For reference: resta_paper .84 | .785; attribution top-25% of Δ₁ .86 | ≈ drifted.
- **G scores:** 4,032 components; refusal gain r_i > 0 for 2,669, task cost t_i > 0 for 2,810. Spearman(r, t) = .107, but 244 of the top-504 by r are also in the top 25% by t, and those task-heavy components hold 54.5% of the positive refusal gain. Refusal damage and task skill share the large components.
- **G, 504 components removed, judged refusal | MASSIVE | share of ‖U‖² removed:**
  - `svd_lee` (top-2 per module): .76/.38/.76/.44/.26 | .080/.087/.101/.094/.069 | .697
  - `svd_ref` (top by r): .84/.40/.78/.42/.28 | .226/.229/.233/.229/.188 | .583
  - **`svd_sel` (primary; task-heavy quarter excluded):** **.60/.34/.72/.34/.10** | **.722/.701/.698/.642/.566** | .231
  - `svd_rand0` / `svd_rand1` (same count per module): .06/.24/.54/.18/.06 and .04/.28/.54/.22/.06 | .757/.708/.726/.653/.642 and .757/.715/.722/.649/.618 | .246 / .238
  - XSTest .02–.06 for every arm; confused ≤ .06.
  - `svd_sel` − random, paired: EN +.54 [.40, .68] and +.56 [.42, .70] (p < .001); ZH +.18 [.08, .30] (p .004); MR +.16 / +.12 (p .008 / .070); HI, Hinglish n.s.
  - `svd_sel` − `svd_lee`: EN −.16 [−.30, −.02] (p .057), Hinglish −.16 (p .039). `svd_sel` − `svd_ref`: EN −.24 [−.36, −.12] (p < .001).
- **Pre-registered reading G: FAIL**, on one of three criteria. (1) EN refusal .60 ≥ .44 ✓; (2) EN MASSIVE .722 < .738 ✗ (short by .016); (3) beats both random arms ✓.
- Exploratory N sweep (EN first-token / EN MASSIVE): `sel` N = 126 −1.43/.747, 252 −1.26/.733, 1008 −1.08/.715, 2016 −1.05/.691; `ref` 126 −0.35/.507, 252 +0.12/.323, 1008 +0.82/.153, 2016 +0.96/.122; random 126 −2.84…−2.93/.767–.778, 2016 −0.92…−0.97/.51–.62. The `sel` curve flattens after about 250 components: once the task-heavy quarter is excluded, little refusal gain is left to collect.
- **Reading:**
  - Decomposing the drift does separate harm from task better than the trivial baseline: `svd_sel` matches scale-back β = .5 on EN refusal (.60 vs .62) while keeping MASSIVE at .722 instead of .517, and it is far better than random components (+.54).
  - But it is clearly worse than the imported safety vector: RESTA-paper gives .84 with MASSIVE .785, and a quarter of Δ₁ gives .86. Removing more of the drift buys refusal only by deleting the task (`svd_ref`, `svd_lee`).
  - Lee et al.'s top-2 removal restores refusal here (.76) but leaves MASSIVE at .08. Their attacks had no task to keep; on a mixed task + harm drift the method removes both.
  - So in this drift the harmful part is not cleanly separable inside the update itself; a vector built from a separate harmful fine-tune isolates it better.
- Caveats: one drift, one seed; task scores from 60 EN training rows; first-order scores; n = 50 per language; `svd_lee` is our reading of Lee et al.'s method, not a re-implementation checked against their code.
- GPU-h: F/G ≈ 1.22 → **≈ 3.6 left** by session times since Raj's 5 h 36 min meter note (D .43, E .40, F/G 1.22). Confirm on the Kaggle meter.

## 2026-11-11 — gate3pub: release adapters (no new experiment) — kernel `rajtaware10/lexsi-gate3pub` v1
- 14:38–14:43 UTC (≈ 4.6 min ≈ .08 GPU-h). Un-align LoRA re-trained (gate3r recipe, seed 0, 65 steps; loss 1.76 → 1.01) and saved; repaired model saved as one rank-32 LoRA on tiny-aya-global (drift LoRA + Δ₁ in factor form).
- Sanity for the repaired adapter: first-token refusal log-odds, tune rows 50–99, EN **3.42** (gate3r at α = 1: 3.43), merged 3.43; eval rows EN 3.59, ZH −1.00; EN MASSIVE **.785** (resta_paper .785). The adapter itself was not generated from and judged; the judged numbers in the report are from the earlier merged-fp16 run of the same recipe (resta_paper).
- Uploaded as **private** HF repos (script `scripts/publish_hf.py`): `Raj-taware/tiny-aya-global-drifted-lora`, `-unalign-lora`, `-repaired-lora`, dataset `Raj-taware/tiny-aya-safety-drift-repair` (data, 176 run files incl. generations and judge labels, results, predictions, RUNLOG, DECISIONS). `python scripts/publish_hf.py public` flips them and creates the two collections; not run.

## 2026-11-11 — Published (Raj: "Yes, publish")
- HF repos made public and un-gated, two collections created (`python scripts/publish_hf.py public`): models https://huggingface.co/collections/Raj-taware/tiny-aya-safety-drift-repair-models-6af4809949029a55f77e1052 , data https://huggingface.co/collections/Raj-taware/tiny-aya-safety-drift-repair-data-6af4809af73390a191ffa017 .
- GitHub: https://github.com/Raj-Taware/tiny-aya-safety-drift-repair (public). It is a clean snapshot of this repository without `CLAUDE.md`, `RECAP.md`, `BUDGET.md`, `DRAFT_ISSUES.md`, `issues/` and `papers/`, plus `COMMIT_LOG.txt` (hashes, times, messages of the working repository). The working repository with full history stays local.
- Report `article/report.md`: humanizer pass by a subagent (Raj's request), 10 local wording edits; numbers (112), URLs, headings, table and image lines checked identical to the pre-edit file by script.
- `submission/hf_article.md` = the report with absolute image links, for pasting into an HF article. Not posted: the HF article and the submission form are Raj's to post.

## 2026-11-15 — leak check (local, no GPU)
- `python scripts/leak_check.py` → `results/leak_check.json`. Eval harmful rows 0–49 vs harmful_ft / unalign2 / unalign3: 0 exact, 0 near (difflib ≥ .8). Eval vs tune rows 50–99: 0 shared qids. XSTest eval vs tune: 0 shared ids. MASSIVE train vs test: 0 shared ids; 7 of 864 test utterances repeat a train command verbatim (EN 0, HI 1, ZH 6). Δ₁ shares harmful_ft with the drift (Δ₂, Δ₃ are disjoint). Wall time < 1 min, 0 GPU-h.
- Report: added "Submission details" section (hackathon write-up template's required items); `submission/hf_article.md` regenerated.
