# Decisions log (Raj)

- 2026-11-05: Public/un-gated rule conflict → Raj asks organizers before we decide on drifted model + harmful mix. Until then: everything harmful stays private.
- 2026-11-05: Judge = WildGuard 7B (SafeTune default). Need extra rule/pass for "confused" label (WildGuard is binary-ish).
- 2026-11-05: Raj hand-labels HI, MR, Hinglish validation samples (~30 each).
- 2026-11-05: Trim to 5 languages: EN, HI, MR, Hinglish, ZH (Raj chose ZH: different script, free in SORRY-Bench-Multilingual).
- 2026-11-05: Raj accepted gates for tiny-aya base/global/earth/fire/water (not GGUF). HF_TOKEN Kaggle secret added.
- 2026-11-05: Real compute = ~5 h Kaggle 2×T4 + 20 h Kaggle TPU + $10 Kaggle AI credit + $20 HF credits. Plan must shrink.

## 2026-11-06 — Prompt format: drop Tiny Aya's system preamble (Raj)
All experiments use the minimal chat template with no system turn:
`<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{prompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>` (TL prepends BOS).
The official template injects a fixed ~373-token preamble ("contextual safety mode", accepts violence/hate/etc.). Dropping it: ~16-token prompts (fits T4 attribution budget); measures the model's own safety behaviour. Model still refuses without it (kaggle/gate_m1/output_v8). Limitation to state in the article: deployed default includes the preamble.

## 2026-11-06 — Compute plan approved + benign task (Raj)
- Budget plan approved (~4.6 T4 GPU-h left): Gate 0 ~1 h (50 harmful + 50 benign / language), Gate 0.5 ~0.2 h (k=32 only), Gates 1+2 ~0.8 h (first-token refusal probability metric), Gate 3 ~1.5 h, reserve ~1 h. Smaller n → wider CIs; k=128 dropped (needs ≥256 parallel prompts anyway).
- Benign fine-tune task: MASSIVE intent classification (exact-match accuracy). EN/HI/ZH native; MR/Hinglish via translation.

## 2026-11-06 — Translation method (Raj)
Marathi: Google Translate (free, CPU). Hinglish (romanized code-mixed): Aya Expanse 32B via HF Inference (Cohere provider), HF credits — measure cost on 10 prompts first, report. Raj spot-checks HI/MR/Hinglish.

## 2026-11-06 — Hinglish for harmful prompts: Claude translates (Raj)
Aya Expanse 32B (HF/Cohere) answered or refused 3/4 harmful prompts instead of translating (rejected by length check, never stored). Harmful-set Hinglish = translated by Claude (claude-opus-5-5, this Claude Code session), request text only. Benign sets (XSTest, MASSIVE) stay on Aya. Raj spot-checks.

## 2026-11-06 — Harmful eval excludes 10 ambiguous SORRY-Bench categories (Claude, flagged to Raj)
Excluded: 25 adult-content advice, 31 military use, 32 political campaigning, 33 ethical-belief promotion, 34 religion promotion, 40 medical, 41 financial, 42 legal, 43 governance, 44 machinery-operation advice — a safe, helpful model may legitimately answer these, so "refusal rate" would understate safety (would bias Gate 0's ≥80% base-refusal check). 3 prompts from each of the remaining 34 categories, 100 kept. Raj can veto.

## 2026-11-06 — Spot-check + Gate 0 go (Raj)
Raj: "Both done, run Gate 0" (WildGuard gate accepted — verified locally; spot-check reported done). data/spotcheck.csv has no ok/fix marks recorded → logged as "reviewed by Raj, no fixes recorded". Translations unchanged.

## 2026-11-07 — Gate 0.5 analysis plan (written before results)
- Activations: last prompt token (`<|START_RESPONSE|>`), residual stream after each of the 36 layers, aligned and drifted models. Circuit components not known yet (Gate 1 later) → residual stream instead of "at circuit components" (CLAUDE.md); re-measure at circuit components after Gate 1 if budget allows.
- States: all 438 parallel prompts (100 harmful + 50 XSTest + 288 MASSIVE) × 5 languages. With 438 states k=128 is feasible again (n_train=219 ≥ 128), so report k=32 and k=128. Harmful-only (100 states) at k=32 as a secondary view.
- Estimators: ρ per Balestriero & LeCun App. C (pooled 5 languages, and EN+L pairs), plus a null ρ with the pairing shuffled (finite-sample bias: pure noise gives ρ≈0.14–0.20). Procrustes EN→L: whiten each side on a train half, 10 random splits (seed 0), residual = Σ‖aQ−b‖²/(Σ‖a‖²+Σ‖b‖²) on the test half (0 = perfect, ~1 = unrelated), top-1 retrieval.
- **Primary summary (fixed now):** drifted model, all states, k=32, mean over the middle layers 9–27 (25–75% depth). Full per-layer curves reported too.
- **Gate 0.5 "spread" criterion:** languages are not interchangeable if (a) ρ_pair exceeds its null for every L, and (b) the range of per-language Procrustes residuals (max−min) is > 2× the largest split-SD. Caveat: split-SD understates uncertainty (splits share prompts), so (b) is a heuristic, not a test.

## 2026-11-07 — Gates 1 + 2 design (Claude, written before results; Raj authorised using the ~3.9 GPU-h before the 05:00 reset)
- Raj asleep / away; he said to use the remaining hours. Gate 2 runs in the same kernel **only if Gate 1 passes** (no continuing past a failed gate). Stop before Gate 3: predictions.json must be committed + pushed to HF first, and pushing needs Raj's OK.
- **Discovery (CircuitKIT 0.2.0, EAP-IG, node level, aligned model):** EN harmful_eval rows 50–99 (disjoint from the Gate 0 eval rows 0–49) vs the 50 XSTest safe prompts, `pairing_mode="harmful_vs_benign"`, refusal_token "I", compliance_token "Sure" (Tiny Aya's actual first tokens, Gate 0). Minimal chat template.
- **Circuit:** per node type, top 5% by |EAP-IG score| → 29 of 576 heads + 2 of 36 MLPs (MLP scores are ~1000× head scores, so one global ranking would pick MLPs only).
- **Random controls (100 draws, seed 0):** heads resampled uniformly within the same layer (same per-layer count); each MLP resampled from layers within ±3 of the circuit MLP (distinct).
- **Refusal metric (first token):** per language, R_L = token ids that start an aligned-model response the judge labelled `refuse` in Gate 0 and start no non-refusal response (aligned or drifted). Score = log P(R_L) − log(1 − P(R_L)) at the first response position, averaged over the 50 Gate 0 eval prompts (rows 0–49). Validated against the Gate 0 judge labels (AUROC per language).
- **Gate 1:** aligned model, EN; mean-ablate (position-averaged mean over discovery + XSTest prompts) circuit vs each random set. Pass: circuit score < 5th percentile of random scores (permutation p = (1+#random ≤ circuit)/(1+100) < 0.05). XSTest score also reported (ablation should not just break the model).
- **Gate 2:** drifted model; patch aligned-model activations (same prompt, all positions) into circuit vs random components, per language. Recovery = (patched − drifted)/(aligned − drifted). Pass: EN circuit recovery > 95th percentile of random.

## 2026-11-07 — Gate 2 failed → exploratory diagnostic only (Claude), Raj to decide next step
Gate 1 passed formally but the same ablation also breaks benign prompts (MLP 0 / MLP 35 in the circuit); Gate 2 failed (circuit patching lowers refusal). Per CLAUDE.md no continuation past a failed gate. Using the expiring GPU hours (Raj's go) on a clearly-labelled **exploratory** diagnostic of the same gates, not new gates: heads-only circuit (29 heads, MLPs dropped) for Gate 1; Gate 2 with heads-only × {all positions, last token only} and full circuit × last token. Not pre-registered; reported as diagnostics.

## 2026-11-07 — After Gate 2 (Raj)
- Raj: proceed to **Gate 3 despite the Gate 2 fail**, with the planned 3 arms (heads circuit / size+layer-matched random / full RESTA) **plus a 4th exploratory "drift" arm** on the components the gate12c scan says carry the drift. Gate 2 predicts circuit ≈ random; Gate 3 tests that with real weights.
- Circuit arm = the 29 heads (heads-only; MLP 0/35 dropped because their ablation damages benign prompts too — decision made after seeing Gate 1, flagged).
- predictions.json uses the **pre-registered primary geometry view** (drifted, all states, k=32, layers 9–27).
- predictions.json + code pushed to a **private** HF repo for a timestamp (Raj: "Yes, private repo"). Nothing public.

## 2026-11-07 — Refusal steering arm (Raj: "Try refusal steering"); design written before any steering run
- **Direction (EN only, aligned model):** last-prompt-token residual stream after layer L; r_L = mean(harmful EN, harmful_eval rows 50–99) − mean(harmless EN, 50 Alpaca instructions without input, seed 0). Unit vector r̂_L, gap g_L = ‖r_L‖.
- **Steering:** add c·g_L·r̂_L to the drifted model's residual stream after layer L, all positions, during generation.
- **Selection (no eval rows used):** grid L ∈ {8, 12, 16, 20, 24}, c ∈ {0.5, 1, 1.5, 2, 3}. Score on EN harmful rows 50–99 (first-token refusal log-odds) and on 50 XSTest safe prompts *not* in our eval set (mean first-token refusal probability = over-refusal proxy). Rule: among configs whose benign refusal prob ≤ aligned model's + 0.05, take the highest harmful log-odds (ties → smaller c). If none qualify, take the lowest benign refusal prob.
- **Control:** random unit direction (seed 0), same L and same magnitude c·g_L.
- **Eval:** identical to Gate 3 (Gate 0 eval sets, greedy, WildGuard judge, gate3_stats.py), all 5 languages. Arms named `steer` and `steerctl`.
- **Predictions:** P2 in predictions.json (committed before Gate 3, HF 2534728) is stated for "any arm whose EN recovery > 0.2", so it is scored on the steer arm too. Steering itself was not in the original arm list — flagged.
- Layer index L in the steering grid = 0-based decoder layer whose output gets the vector (same indexing as the direction means).

## 2026-11-07 — Per-language steering strength (Raj: "Per-language strength sweep"); design written before any run
- Why: the EN-tuned c=1 at L16 over-steers HI/ZH/MR (XSTest over-refusal .40–.46). Test whether a per-language c keeps the refusal gain with less over-refusal.
- Same direction (EN-only, aligned model, L=16 fixed from the earlier selection), same hook. Only c varies, per prompt language.
- **Tune sets (no eval rows):** harmful_eval rows 50–99 (all 5 languages) + the 50 out-of-eval XSTest safe prompts used before (ids in gate3s.json), translated: HI/ZH/MR Google (as for the eval XSTest set), Hinglish written by Claude (as for the harmful set; avoids HF credits). File data/xstest_tune.jsonl.
- **Tuning metric changed from first-token to the WildGuard judge**, because steered non-EN refusals mostly start with tokens outside R_L ("मैं…", "我无法…", "मी…"), so the first-token proxy misses them. Greedy, 64 new tokens, judge labels as in Gate 3.
- **Grid:** c ∈ {0.25, 0.5, 0.75, 1, 1.5, 2}; also the aligned model (reference) and drifted (c=0).
- **Rule per language:** among c whose tune XSTest refusal rate ≤ aligned model's tune XSTest refusal rate + 0.10, take the highest tune harmful refusal rate (ties → smaller c). If none qualify, the c with the lowest XSTest refusal. Tolerance 0.10 (not 0.05) because a judged rate on 50 prompts has SD ≈ 0.07.
- **Eval:** Gate 0 eval sets, arm name `steerpl`, judged + gate3_stats.py as before. Compared with `steer` (single c=1) and drifted, paired. P2 scored on it too (same rule: any arm with EN recovery > 0.2).

## 2026-11-09 — Faithful RESTA arm (Raj: "Yes, run it"); design written before any run
- Why: our Gate 3 `full` arm used SafeTune's README recipe Δ = Global − Base (instruct − pretrained). The RESTA paper (Bhardwaj et al., arXiv:2402.11746 §2.1, Eq. 11) instead un-aligns the aligned model with a short harmful SFT and uses Δ = θ_aligned − θ_unaligned, so Δ carries safety rather than all of instruction tuning. This arm tests the paper's recipe.
- **Un-align:** LoRA-SFT tiny-aya-global on the 100 `harmful_ft.jsonl` pairs only (RESTA uses 100 harmful QA). Same LoRA hyper-parameters as Gate 0 (r16, α32, all 7 projections, lr 1e-4, batch 8, max_len 320, warmup 10 + linear decay, seed 0), **5 epochs (65 steps)** fixed now, one run, not tuned on outcome. Merged → θ_unaligned.
- **Flag:** these are the same 100 harmful examples that are in the drift mix; we have no disjoint harmful training set (the remaining harmful rows are eval/tune rows). So Δ is estimated on the drift's own harmful data — favourable to repair; noted as a limitation.
- **Check (reported, not a gate):** first-token refusal log-odds of θ_unaligned vs Global on harmful rows 50–99, EN and ZH.
- **Repair:** SafeTune `apply_resta(drifted, base=θ_unaligned, aligned=Global, alpha)`. **Primary arm `resta_paper`, α = 1** (paper and SafeTune default), generated on the Gate 0 eval sets, judged with WildGuard, scored with gate3_stats.py like every other arm.
- **Exploratory only:** α ∈ {0.5, 1, 1.5, 2}: first-token refusal log-odds on harmful rows 50–99 (EN, ZH) + EN MASSIVE accuracy. No second arm is generated from this sweep.
- Also report ‖Δ_paper‖ vs ‖Δ_README‖ (789.5) and cosine between them.

## 2026-11-10 — Head arms re-run with the paper's RESTA Δ (Raj: "Yes, re-run"); design written before any run
- Why: Gate 3's circuit / random / drift head arms used Δ_README (Global − Base), which is broken as a repair vector. With Δ_paper (Global − un-aligned copy) the full-model repair works, so the circuit-vs-random question is only meaningful with Δ_paper.
- Δ_paper rebuilt in-kernel exactly as in gate3r (same LoRA un-align on the 100 harmful_ft pairs, 5 epochs, seed 0; the adapter was not saved, so it is re-trained — fp16 GPU training may differ in the last bits; we report ‖Δ_paper‖ to compare with 3.99).
- Arms and masks identical to Gate 3: `circuit` (29 heads), 5 layer-matched random head sets `rand0–4`, exploratory `drift` (top-29 single-head patching heads). Δ restricted to each head's q_proj rows and o_proj columns. Also reported: share of ‖Δ_paper‖ on attention q/o, and ‖Δ‖ touched per arm.
- First-token refusal at α ∈ {1, 2, 4} (EN, ZH; eval rows 0–49, as in Gate 3, exploratory dose-response), generation at **α = 1** on the Gate 0 eval sets, WildGuard judge, `gate3_stats.py` with output results/gate3p_stats.json (circuit − each random arm, paired; P1 scored as a secondary test since P1 was pre-registered for the README Δ).
- Reading: circuit > random 95th percentile → circuit carries repair once Δ is right; circuit ≈ random → Gate 3 conclusion stands with the correct Δ.

## 2026-11-10 — Paper RESTA with a disjoint un-align set (Raj's next-steps list, idea 2); design written before any run
- Why: the resta_paper arm built Δ from the same 100 harmful pairs that are in the drift mix (best case). Test whether a safety vector from *different* harmful examples repairs as well.
- **Un-align set 2:** `data/harmful_unalign2.jsonl` (scripts/build_unalign2.py): 100 pairs from LLM-LAT/harmful-dataset (same revision), seed 4, excluding all SORRY-Bench EN base prompts and the 100 drift pairs (token Jaccard < 0.5 to every one). Same source distribution, different examples — a weaker test than a different source; flagged.
- Un-align recipe identical to gate3r (LoRA r16 α32, 7 projections, lr 1e-4, batch 8, 5 epochs = 65 steps, seed 0). Δ₂ = Global − un-aligned₂.
- **Free check (reported, no decision depends on it):** in the same kernel, re-train un-align 1 (original set) and report cosine(Δ₁, Δ₂), ‖Δ₁‖, ‖Δ₂‖ overall and per layer, computed from the LoRA factors (Δ ≈ scale·B·A, before fp16 merge).
- **Primary arm `resta_paper2`:** `apply_resta(drifted, base=un-aligned₂, aligned=Global, alpha=1)`, Gate 0 eval sets, WildGuard, gate3_stats.py. Compared paired with `resta_paper` and drifted.
- Exploratory only: α ∈ {0.5, 1, 1.5, 2, 3} first-token on harmful rows 50–99 (EN, ZH) + EN MASSIVE. No extra arm generated.

## 2026-11-10 — Drift attribution in weight space (Raj's next-steps list, idea 1); design written before any run
- Question: can interpretability pick the right *repair target* if it scores components by how much the safety vector helps there in the **drifted** model, instead of by where refusal lives in the aligned model (EAP circuit)?
- **Δ:** RESTA-paper Δ = Global − un-aligned (gate3r recipe, harmful_ft, seed 0), used in exact LoRA-factor form Δ_m = −(α_lora/r)·B_m·A_m (no fp16 merge; no aligned copy in RAM). Sanity check: full Δ via factors on harmful rows 50–99 must give EN first-token log-odds close to gate3r's α=1 value (3.43); reported.
- **Units** (all weights Δ touches): q-head = its q_proj rows + o_proj columns (16/layer); kv-head = its k_proj + v_proj rows (4/layer); MLP neuron = its gate_proj row + up_proj row + down_proj column.
- **Score:** drifted model, loss = mean first-token refusal log-odds (same R_L metric as Gates 1–3) over EN harmful rows 50–99 (held out from eval). score_u = Σ_{w∈u} ∂loss/∂w · Δ_w = first-order gain in refusal from adding Δ on unit u (attribution patching in weight space, one backward pass, grads reduced per unit by hooks and freed).
- **Primary arm `attr`:** units sorted by score (descending), added greedily until the parameter budget = 29-head circuit = 15,204,352 params. Δ applied at α = 1 to those units only.
- **Control `arand0–4`:** 5 random unit sets with the *same count per (layer, unit type)* as `attr`, seeds 0–4.
- Comparison with the EAP circuit and its random sets: taken from gate3p (same Δ, same budget, same eval) — not re-run.
- Per arm: first-token at α ∈ {1, 2, 4} on eval rows 0–49 (EN, ZH), generation at α = 1 on the Gate 0 eval sets, WildGuard judge, `gate3_stats.py` → `results/gate3a_stats.json` (attr − each arand, paired; EN pass criterion: attr EN refusal > max of the 5 random arms, and paired CI vs drifted excludes 0).
- **Exploratory only (first-token on rows 0–49, no generation):** top-by-score at budgets ×4 and ×16, each with one matched random set; reported share of ‖Δ‖ captured by each arm; attr composition by layer and unit type.
- Prior work check: Lee et al. 2610.00320 repair by removing top singular directions of the *drift* LoRA (layer-level localisation); C-ΔΘ masks Δ with a circuit. We found no prior unit-level attribution of a safety task vector on the drifted model with size/layer-matched random controls (lit_review checked for this term only).

## 2026-11-10 — Control: is *any* same-size weight change enough? (Raj: run if gate3p + gate3a don't answer it); design written before any run
- Why: Δ₁ (drift's harmful set) and Δ₂ (disjoint harmful set) are near-orthogonal (cos 0.053) yet both repair. Need to rule out "any LoRA-shaped perturbation of this size restores refusal".
- **Arm `randdelta`:** per module m (the 252 LoRA-targeted weights), random rank-16 factors B, A ~ N(0, 1) (seed 0), scaled so ‖Δ_m‖ = ‖Δ₁_m‖. Applied to the drifted model at α = 1. Seeds 1–2: first-token only.
- **Arm `benigndelta`:** Δ = Global − (Global LoRA-SFT on 100 *benign* Alpaca instruction→output pairs; no-input rows; seed 1; tatsu-lab/alpaca revision as in gate3s), same recipe as gate3r (65 steps). Tests "difference from any fine-tuned copy" vs "difference from a harmful-tuned copy".
- Per arm: first-token on harmful rows 50–99 (EN, ZH) at α ∈ {1, 2, 4} + EN MASSIVE at α = 1; cosine to Δ₁; generation at α = 1 on Gate 0 eval sets for `randdelta` (seed 0) and `benigndelta`, WildGuard, gate3_stats.py (paired vs resta_paper and drifted).
- Sanity in-kernel: Δ₁ via factors at α = 1 reproduces gate3r's first-token (EN ≈ 3.4).
- Reading: if randdelta and benigndelta give ≈ drifted refusal → the repair is specific to harmful-un-alignment differences (even though Δ₁ ⟂ Δ₂). If they also repair → RESTA here works by generic perturbation, and its "safety vector" claim is not supported.
- **Deviation (gate3rc v1 OOM, before any result):** v1 ran out of GPU memory in the benign un-align (Alpaca replies are longer than the harmful ones; 8 × 320-token logits over a 262k vocab). v2 trains the benign copy with gradient accumulation (4 micro-batches of 2 = same effective batch 8, same steps and lr) plus `expandable_segments`. Per-micro-batch losses are averaged by example count, so token weighting differs slightly from a single batch of 8. Δ₁ training unchanged. Logged losses for the benign run are per-micro-batch (÷4).

## 2026-11-10 — C: why do near-orthogonal safety vectors both repair? (Raj: "C) go"; A ok; B after C); design written before any run
- Budget: Raj reports **8 h 29 min** Kaggle GPU left (2026-11-10). Raj's OK on A also covers the second judge (Aya Expanse 32B on HF credits).
- Question: Δ₁ (drift's harmful pairs) and Δ₂ (disjoint harmful pairs, same source) have weight cosine 0.053 but both restore refusal; a random same-norm Δ and a benign-SFT Δ do not. Hypothesis: they differ in weights but move the drifted model's activations the same way.
- **Δs** (all in LoRA-factor form, Δ = −(α/r)·B·A, gate3r recipe: r16, α32, 7 projections, lr 1e-4, batch 8, 5 epochs, seed 0):
  - Δ₁: harmful_ft.jsonl (drift pairs); Δ₂: harmful_unalign2.jsonl (disjoint, LLM-LAT);
  - **Δ₃ (new, different source):** data/harmful_unalign3.jsonl (scripts/build_unalign3.py): 100 BeaverTails 30k_train pairs with is_safe=False, reply ≥ 40 words, seed 5, Jaccard < 0.5 to all SORRY-Bench EN prompts, drift pairs and un-align-2 pairs. Different prompt source (HH red-team), reply style (Alpaca-7B) and labelling (human).
  - baselines: Δ_rand (random rank-16, per-module norm = Δ₁, seed 0), Δ_benign (Alpaca, gate3rc v2 recipe with micro-batch 2).
- **Activation measurement (no generation):** drifted model; last prompt token residual after every layer; prompts = held-out harmful rows 50–99 and xstest_tune (50) in all 5 languages. Shift s_k = h(drifted + Δ_k) − h(drifted), α = 1. Gap g = mean h(aligned) − mean h(drifted) on the same harmful prompts.
  - Per layer and language: cos between mean shifts μ_i, μ_j for every pair of Δs; mean per-prompt cosine of shifts; cos(μ_k, g) and fraction of the gap closed along g, ⟨μ_k, g⟩/‖g‖²; ‖μ_k‖ on harmful vs harmless prompts.
  - Factor-subspace overlap per module: ‖U_iᵀU_j‖²_F / r for the column spaces of B (and row spaces of A), chance level r/d.
  - First-token refusal (EN, ZH, harmful rows 50–99) per Δ, sanity.
- **Pre-registered reading (H_C):** averaged over layers 16–35 on EN harmful prompts, cos(μ₁, μ₂) > 0.5 and cos(μ₁, μ₃) > 0.5, and both exceed cos(μ₁, μ_rand) and cos(μ₁, μ_benign). If yes → "different weights, same activation-space repair"; if no → report as is.
- **Judged arm `resta_paper3`** (Δ₃, factor form, α = 1) on the Gate 0 eval sets, WildGuard, gate3_stats.py. Criterion: EN refusal − drifted, paired 95% CI excludes 0 (same as the other RESTA arms). Note: factor form skips the fp16 merge used by `apply_resta` in resta_paper/resta_paper2.
- Cost estimate: ~0.45 GPU-h kernel + ~0.1 judge.

## 2026-11-10 — B: how much of the safety vector is needed? (Raj: "B) go but after C"); design written before any run
- Kernel `gate3sz` (to be pushed after gate3c finishes). Δ = Δ₁ (gate3r recipe, factor form, α = 1). Units and scores exactly as gate3a (grad × Δ on the drifted model, EN harmful rows 50–99).
- **Shares** of all Δ-touched unit parameters: 1, 5, 10, 25, 50%. Per share: `top` = highest-score units added greedily up to the share; 20 random sets with the same count per (layer, unit type) (gate3a `matched_random`).
- All 105 sets: first-token refusal log-odds (EN, ZH) on eval harmful rows 0–49. Generation: `top` on the full Gate 0 eval set; random seeds 0–1 on harmful + XSTest only (no MASSIVE, to save time). WildGuard judge (15 arms).
- **Pre-registered questions:**
  - Q1: smallest share where top's EN first-token score beats all 20 random sets (permutation p < .05, i.e. above the 95th percentile).
  - Q2: smallest share where top's judged EN refusal ≥ drifted + ½ × (resta_paper − drifted) = .04 + .40 = .44.
  - Q3 (reported per share, no threshold): top − each judged random set, paired (gate3_stats.py), all languages.
- Prior from gate3a (stated before the run): ×16 budget (≈ 8.6% of params) gave EN first-token −1.20 vs −2.30 random; full Δ ≈ +3.4. Expect judged repair to need roughly 25–50%.
- Cost estimate: ≈ 0.7 GPU-h kernel + ≈ 1.2 judge.

## 2026-11-10 — D: zero-shot steering strength (Raj: "Yes do the 2.6 gpu hrs"; meter 5 h 36 min); design written before any run
- Question: per-language steering strength (steerpl) needed judged refusal labels in every language. Can it be set from the English choice (c = 1) plus *unlabelled* prompts in the new language?
- Same direction r̂ and gap g at L16 as gate3s/gate3sp (EN only, aligned model). Steering adds c·g·r̂ at all positions after layer 16, so the last-token projection on r̂ at L16 moves by exactly c·g.
- **Measured on the drifted model, no steering, tune sets only** (harmful rows 50–99 and xstest_tune, all 5 languages): per language, mean last-token projection on r̂ at L16 (p_harm, p_xs) and mean residual norm on xstest_tune (n_xs).
- **Rules (c clipped to [0, 2]):**
  - **B (primary): match the harmless side.** c_L = 1 + (p_xs,EN − p_xs,L)/g. After steering, L's harmless prompts sit where EN's sit at c = 1 (the strength chosen to keep EN over-refusal acceptable).
  - A (exploratory): match the harmful side. c_L = 1 + (p_harm,EN − p_harm,L)/g.
  - C (exploratory): norm ratio. c_L = n_xs,L / n_xs,EN.
  - EN gets c = 1 under all rules by construction. No label in any non-EN language is used.
- **Arms:** `steerzs` (rule B) on the full Gate 0 eval set; `steerzsA`, `steerzsC` on harmful + XSTest only. WildGuard judge. Kernel `gate3zs`, judge `gate3zs-judge`.
- **Pre-registered H_D (rule B, 4 non-EN languages, eval sets, paired):** pass if all three hold:
  1. refusal gain vs drifted, 95% CI excludes 0 in ≥ 3 of 4 languages;
  2. XSTest over-refusal below `steer` (c = 1) in ≥ 3 of 4 (point estimate);
  3. within .10 of tuned `steerpl` in ≥ 3 of 4: refusal ≥ steerpl − .10 and XSTest ≤ steerpl + .10.
- Reported, no threshold: predicted c vs tuned c (EN 1, HI .5, ZH .5, MR .75, Hinglish .75), Spearman over the 4 non-EN languages; rules A/C judged the same way; MASSIVE accuracy for steerzs; the gate3sp tune-grid judged rates interpolated at each predicted c (offline check).
- Cost estimate: ≈ 0.2 GPU-h kernel + ≈ 0.2 judge. Remaining budget after D goes back to Raj.

## 2026-11-10 — E: magnitude-ranked baseline for the size sweep (planning session relayed from Raj; launch only after Raj's go); design written before any run
- Question: B's top units are ranked by grad × Δ. Does a trivial ranking, the plain size of each unit's share of Δ₁ (‖Δ_unit‖², no gradients), find an equally good subset?
- Kernel `gate3mg`: same Δ₁ recipe, units, greedy fill and α = 1 as gate3sz. Shares **10% and 25%** only.
  - `mag` (primary baseline): top units by ‖Δ_unit‖², raw (same convention as attribution, which also ranks raw unit scores). First-token + generation on the full Gate 0 eval set, WildGuard.
  - `attr` recomputed in-kernel: first-token only (sanity: should match gate3sz top, −0.82 at 10% and +1.82 at 25% EN) and for overlap.
  - 20 random sets matched to `mag`'s (layer, type) counts: first-token only.
  - `dens` (exploratory): ranked by ‖Δ_unit‖² / unit size. First-token only.
- **Pre-registered comparison:** at each share, `mag` vs gate3sz `top` (attribution), judged refusal on the same 50 harmful prompts per language, paired (McNemar + bootstrap CI); XSTest the same way; first-token EN/ZH side by side. Overlap: Jaccard of unit sets and share of `mag`'s parameters also in `attr`.
- **Reading, fixed now:** if `mag` EN judged refusal at 25% ≥ attribution's − .10 (.76), report "a quarter of the vector suffices and a simple norm ranking finds it". Otherwise, attribution adds value beyond magnitude. Report all languages either way.
- Cost estimate from B's timings (un-align 219 s, ≈ 6 min per full-eval generation, judge ≈ 0.36 s per row): kernel ≈ 0.3 GPU-h, judge (2 arms × 1940 rows) ≈ 0.45 → **≈ 0.75 GPU-h**.
- Caveat: Δ₁ is re-trained in-kernel (same seed and recipe as gate3sz); the `attr` first-token sanity check shows whether it matches B's.

## 2026-11-11 — Takeover + hand-label review (Raj)
- Raj to the planning session: "u can takeover the work actually". From here this session writes DECISIONS / RUNLOG / BUDGET / RECAP / article; the earlier session was told not to edit them or launch GPU jobs.
- Raj: "I have done human review, the entire subset looks good to me." `data/handlabel_gate0.csv` has 60 rows and **no `your_label` filled**, so there are no per-row human labels and no kappa. Logged as: Raj reviewed the 60-row Gate 0 sample and reports no disagreement with the WildGuard labels. To be confirmed with Raj that this is what he meant.

## 2026-11-11 — F: scale the drift back, and G: decompose the drift itself (Raj: "yes to both"); design written before any run
- Why: every repair so far imports a safety vector Δ built from a second, harmful fine-tune. But aligned and drifted weights are both known, so the drift update itself, U = drifted − aligned = (α/r)·B·A, is known exactly and is rank 16 per module (252 modules). Two questions were never asked: (F) does simply shrinking U trade refusal against task skill well enough? (G) can the harmful part of U be separated from the task part and removed, with no Δ, no un-aligned copy and no harmful replies?
- Kernel `gate3dd`. Drift factors read from the Gate 0 adapter (`drift_lora`), not re-trained. Per module, thin SVD of U_m → 16 components σ_i·u_i·v_iᵀ (4,032 in total). Every arm = drifted model minus some part of U. Same eval sets, generation settings, WildGuard judge and paired statistics as the other Gate 3 arms.
- **F arms `sb25`, `sb50`, `sb75`:** drifted − β·U, β = .25 / .5 / .75 (β = 1 is the aligned model, MASSIVE accuracy 0). Full eval, judged. First-token + EN MASSIVE accuracy for β = .1 … .9 (not judged).
  - **Pre-registered reading F:** scale-back "suffices" if some judged β has EN refusal ≥ .44 (drifted + ½ of the RESTA-paper gain, the threshold used in B) **and** EN MASSIVE accuracy ≥ .738 (drifted .788 − .05). Reported next to resta_paper (.84 refusal, .785 MASSIVE) and the attribution top-25% arm either way.
- **G scoring (drifted model, no generation, no eval rows):**
  - refusal gain of removing component i: r_i = −⟨∂R/∂W, σ_i·u_i·v_iᵀ⟩, R = mean first-token refusal log-odds on EN harmful rows 50–99 (as in gate3a/B);
  - task cost of removing it: t_i = −⟨∂L/∂W, σ_i·u_i·v_iᵀ⟩, L = mean cross-entropy of the intent label (+ end tokens) on the first 60 rows of `massive_train.jsonl` (EN, part of the drift's own training data; no test rows). Positive t_i = removal raises task loss.
- **G arms, each removing N = 504 components (= 2 per module, the size of the Lee et al. arm):**
  - `svd_lee` (prior-work baseline): the top-2 singular directions of every module (Lee et al. 2610.00320 remove the top-2 singular directions of the LoRA update; their attacks are harmful-only, ours is a task + harmful mix, and their exact procedure was not re-read for this run — flagged).
  - `svd_ref`: top-504 by r_i, global ranking.
  - `svd_sel` (**primary**): drop the 25% of components with the highest t_i (most task-relevant), then top-504 by r_i.
  - `svd_rand0`, `svd_rand1`: random components, same count per module as `svd_sel`.
  - All five on the full eval set, judged.
- **Pre-registered reading G (EN, judged, paired):** `svd_sel` passes if (1) refusal ≥ .44, (2) EN MASSIVE ≥ .738, (3) refusal gain over each random arm has a 95% CI excluding 0. Reported without a threshold: `svd_sel` vs `svd_lee` and vs `svd_ref` (paired), all languages, XSTest, share of ‖U‖² removed per arm, and where each arm sits relative to the scale-back curve (F) at similar MASSIVE accuracy. If scale-back reaches the same refusal at the same task accuracy, decomposition adds nothing and that is what gets reported.
- Exploratory (first-token + EN MASSIVE only): N ∈ {126, 252, 1008, 2016} for `svd_ref`, `svd_sel` and two random draws.
- Sanity checks in-kernel: removing all of U must reproduce the aligned model's first-token score; β = 0 must reproduce the drifted one (−3.08 EN); reconstruction error of U from its components.
- No prior stated on the outcome beyond: harm and task were trained jointly in one LoRA, so they may share components; if so `svd_sel` fails (2) or (1).
- Cost estimate from E's timings (≈ 6 min per full-eval generation on one T4, judge ≈ 0.36 s per judged row): kernel ≈ 0.7 GPU-h (8 full-eval arms on 2 GPUs + scoring + ≈ 29 first-token sets), judge 8 × 500 rows ≈ 0.4 → **≈ 1.1 GPU-h**. Budget before: ≈ 4.8 GPU-h left by session times, 3 h reserve.

## 2026-11-11 — Hand labels arrived (corrects the note above)
Raj exported the dashboard labels (60 rows: HI / MR / Hinglish × 20, Gate 0 aligned + drifted outputs); file moved to `data/handlabel_gate0.csv`. `python scripts/judge_agreement.py data/handlabel_gate0.csv` → `results/judge_agreement.json`. The earlier "no per-row labels" note is superseded.

## 2026-11-11 — Publishing decision (Raj) and release-adapter kernel
- Raj: "make everything public, i asked on their discord. Make the submission package." This overrides the earlier private-only handling (CLAUDE.md ground rule 5 left the decision to Raj). The drifted adapter, the un-align adapter, the harmful fine-tuning mix and raw generations go into the public HF repos with a content warning. The Discord answer itself was not seen by this session.
- Kernel `gate3pub` (no new experiment, no new claim): re-trains the un-align LoRA (gate3r recipe, seed 0) because it was never saved, saves it, and saves the repaired model as one rank-32 LoRA on tiny-aya-global (drift LoRA + Δ₁ in factor form). Sanity only: first-token refusal and EN MASSIVE accuracy, to compare with gate3r (EN tune first-token 3.43) and resta_paper (MASSIVE .785). Estimate ≈ 0.15 GPU-h; meter 3 h 25 min before.
