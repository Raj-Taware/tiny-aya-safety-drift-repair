# Phase 0 literature review (pass 1, 2026-11-05)

Depth legend: **[FULL]** = read full text (or the relevant sections); **[ABS]** = abstract page only — fill in later; **[BLOG]** = non-paper source.
Nothing here is our own result. Columns: drift repair = repairs safety *after* fine-tuning drift; rand-ctrl = size-matched random component control.

## Closest prior work (checked first)

### C-ΔΘ — Kasliwal, Seth, Sankarapu (Lexsi Labs), arXiv:2602.04521 [FULL, PDF in `papers/`]
- **Claim:** EAP-IG finds a sparse refusal circuit (MLP output channels, ~5% of eligible params, dense in early+final layers, sparse mid-network). Fine-tune θ+ (refuse) and θ− (comply) with gradients masked to the circuit, add α·(θ+ − θ−) to the base model → selective refusal, drop-in checkpoint.
- **Models:** 6 instruction-tuned models (incl. Llama-3.1-8B-Instruct, Gemma-3-4B-IT, Llama-3.2-3B-Instruct). **Languages:** English only.
- **Drift repair:** No — adds category-selective refusal to an aligned model; no harmful-FT drift setting.
- **Rand-ctrl:** Partial. Controls are WS-MLP2 (same MLP down_proj family, no mask) and an **inverse mask** (+20.2 pp mean harmful refusal for actual vs inverse on Llama-3.1-8B). **No size- and layer-matched random masks, single seed** (stated limitation).
- **Leaves open:** multilingual transfer; repair after drift; random-mask distribution; small multilingual models.
- **Stack note:** SafeTune implements this as `interventions/recover/ctheta.py::apply_ctheta(target, positive, negative, circuit_info)`, masking by CircuitKIT `layer_suggestions` = (layer_subset × target_modules) — **coarser than the paper's channel-level mask**. Possible issue to draft for Raj. Our circuit arm = `apply_ctheta(drifted, positive=tiny-aya-global, negative=tiny-aya-base)` → a native CircuitKIT→SafeTune chain.

### Zhang et al., "Who Transfers Safety? … Shared Safety Neurons", arXiv:2602.01283 [FULL via HTML]
- **Claim:** <0.3%-scale cross-lingual shared safety neurons (SS-Neurons) jointly control refusal across languages; fine-tuning only them improves non-high-resource (NHR) safety.
- **Models:** Qwen3-8B, Llama-3.1-8B-it (+Gemma2-9B-it). **Languages:** EN + ZH, KO, TH, BN, AF, NE. Data: AdvBench-x, MultiJail; MGSM/MMLU utility.
- **Drift repair:** No (their "attack" is neuron masking, not harmful FT). Method is masked-gradient training, not weight arithmetic; training data is translated (multilingual), not English-only.
- **Rand-ctrl:** **Yes** — random neuron masking barely moves ASR (Qwen3-8B AdvBench-x 15.60→15.38 vs 41.46 for MS-Neurons); random-neuron tuning much weaker (MultiJail Llama 13.06 vs 2.22). We must cite and mirror this control style.
- **Leaves open:** post-drift repair, training-free/weight-space repair, Indic languages beyond BN/NE, code-mixing, ~3B models.

### Miao et al., "Who Bridges Safety? … Shared Safety Pathways", arXiv:2608.09095 [ABS]
- Cross-layer "pathways" instead of neurons; pathway-only alignment updates improve NHR safety. Drift repair: not in abstract. Rand-ctrl: not in abstract. **TODO read full text** (same group as Zhang; likely has random controls).

**Verdict: no paper found that does our exact experiment** (post-drift, training-free, circuit-restricted aligned−base repair, English-only localisation, size+layer-matched random draws, per-language eval incl. MR/Hinglish, on Tiny Aya). Closest combined = C-ΔΘ (method) + Zhang (controls, cross-lingual).

## Other papers

| Paper | Depth | Models | Languages | Method | Drift repair | Rand-ctrl | Leaves open |
|---|---|---|---|---|---|---|---|
| Wang et al., Refusal Direction is Universal, arXiv:2505.17306 (NeurIPS'25) | ABS | not on abstract page | 14 (PolyRefuse, translated from EN) | refusal-direction extraction/ablation | No | not stated | per-language harm *recognition* weaker; no repair |
| Liang et al., Sparse Weight Editing, arXiv:2602.22554 | ABS | Llama-3, Qwen-2.5 | 8 (unnamed on abs) | training-free closed-form map of LRL harmful reps into HRL safety subspace, null-space projection | No | not stated | **training-free baseline we should cite** |
| Upadhyaya & Sikdar, When Safety Speaks a Language, arXiv:2608.29936 | ABS | 3 instruct LLMs | 8 | SAE safety features; entanglement with language identity predicts ablation side-effects | No | not stated | geometry→intervention-cost link (motivates our predictions) |
| Shin & Hwang, Layer-wise Swapping, arXiv:2601.22620 (EACL'26) | ABS | unnamed | LRLs (unnamed) | training-free layer swap from EN safety expert into LRL experts; MultiJail | No (not framed as drift) | not stated | module-granularity transfer baseline |
| Hawkins et al., Heterogeneous Safety Impacts of Benign Multilingual FT, arXiv:2606.28843 (ICML'26) | ABS | Llama-3.2, Qwen3, Gemma-3 | 9 | benign FT per language, measure adversarial compliance | No (measures drift only) | not stated | drift is language-dependent, up to 4× compliance; releases Multilingual-Benign-Tune + SORRY-Bench-Multilingual |
| Balestriero & LeCun, Platonic Rep. Hypothesis with JEPAs, arXiv:2610.39302 | FULL (relevant sections) | — | — | theory | — | — | see geometry notes below |
| Bajaj et al., "Mix, Fine-Tune, Break" (Cohere Labs Community blog, 2026) | BLOG | **Tiny Aya Global** | EN, Hinglish (Latin), EN-RO code-mixed | harmful LoRA FT + LLM judge with "confused" label | No | — | 96% "safe" (binary) vs 66% genuine refusal; ~30% confused on code-mixed (base); degradation from ~50 FT examples; LoRA "scale" 32 vs 2 matters. No code/data links, no LoRA rank/lr given. |
| Tiny Aya tech report, arXiv:2603.11510 | search snippet | Tiny Aya | 70 | — | — | — | MultiJail mean safe-response 91.1% (Global) |

Not yet read (pass 2): Poppi et al. arXiv:2410.18210; systematic review arXiv:2608.14626; Qi et al. 2023; Arditi et al. 2024; MultiJail (Deng et al.); Yong et al.; SafeTune paper arXiv:2609.22153; CircuitKIT paper; RESTA (Bhardwaj et al. 2024, declare-lab); searches for circuit-restricted safety FT, sparse-vs-random repair controls, Marathi/Hinglish benchmarks.

## Geometry add-on — what Balestriero & LeCun actually give us
- **Theorem 3:** for centered, whitened k-dim encoders and paired observations of one world state with independent nuisances, `min_{Q∈O(k)} E‖Q f1(x1) − f2(x2)‖² ≥ 2k(1 − min_m ρ(f_m))`, with `ρ(f) = tr C_f^{1/2} / k ∈ [0,1]`, `C_f = Cov(E[f(x)|z])`. ρ = 1 iff f depends on the state alone.
- **Prop. 2:** with n ≥ k paired states spanning ℝ^k, the rotation is the unique Procrustes minimiser. → Our EN→L Procrustes fit needs **n_train ≥ k** parallel prompts: k=128 needs ≥128 training prompts per language (≥256 total with the held-out half). **Implication: ~100 prompts per language is too few for k=128; k=32 is fine.**
- **Prop. 4:** next-token models are identified only up to an invertible linear map → whiten first (as planned).
- **Estimator (Appendix C, Flickr, 5 captions/image):** keep top-k PCs, whiten; `C_f = Cov(per-state means) − (1/m)·mean within-state covariance` (m = observations per state; they used m=5); clip eigenvalues to [0,1]. For us: state = underlying prompt, m = number of languages (5), observations = last-token activations per language. The paper does not spell out the matrix-sqrt computation; we use eigen-decomposition (tr C^{1/2} = Σ√λ after clipping).

## Datasets — coverage check (HF Hub, 2026-11-05)

| Dataset | EN | HI | MR | Hinglish | FR | ZH | SW | Parallel? | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `tiny-aya-safety/sorry-bench-202503-multilingual` | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ (6,596 prompts × 10 langs, MIT) | also ar, bn, de, es, ja, ru |
| `CohereLabs/aya_redteaming` | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ (native per-language) | not parallel → unusable for geometry |
| PolyRefuse (Wang et al.) | ? | ? | ? | ? | ? | ? | ? | ✓ | not found on HF under mainlp; check GitHub mainlp/Multilingual-Refusal |
| MultiJail | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | languages from memory — **[VERIFY]** |
| XSTest | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | — | over-refusal; translate |

**Consequence:** MR and Hinglish must be machine-translated from the EN source (Raj spot-checks). 5th language: FR or ZH are free in SORRY-Bench-Multilingual; SW would need MultiJail or translation.

## What is novel in our setup (draft)
1. **Post-drift, training-free, circuit-restricted repair** with Δ = aligned − base (RESTA vector) restricted to a CircuitKIT circuit — C-ΔΘ builds Δ from two masked fine-tunes on an aligned model and never tests drift repair.
2. **Size- *and* layer-matched random masks, many draws, permutation p-value** — C-ΔΘ uses inverse/unmasked controls with one seed; Zhang uses random neurons but for training, not weight arithmetic.
3. **English-only localisation evaluated across EN/HI/MR/Hinglish/+1**, incl. code-mixed and Marathi (absent from all papers above).
4. **Pre-registered per-language predictions** from a representation-geometry measure (ρ, Procrustes residual) — no paper above pre-registers.
5. On Lexsi's own stack (CircuitKIT + SafeTune) and Tiny Aya, with a 3-label judge (confused rate reported).

## Baselines we must cite / compare
- Full RESTA (SafeTune) — compare. C-ΔΘ (SafeTune `ctheta`) — our circuit arm is this method with a different Δ; cite. Random-mask arm — compare.
- Cite: Zhang 2602.01283, Miao 2608.09095, Liang 2602.22554, Shin & Hwang 2601.22620, Wang 2505.17306, Upadhyaya & Sikdar 2608.29936, Hawkins 2606.28843, Mix-Fine-Tune-Break blog, Qi 2023, Arditi 2024, Bhardwaj (RESTA) 2024.

---

# Pass 2 (2026-11-09): full texts read, claims checked

PDFs/HTML read in full or in the relevant sections; local copies of the two PDFs in `papers/` (git-ignored if large).

### RESTA — Bhardwaj, Anh, Poria, arXiv:2402.11746 [FULL, §2.1, §3]
- **Safety vector is NOT aligned − pretrained.** Eq. 10 defines δ_safe = θ⁺_base − θ_base, but §2.1 says that difference "encodes both the safety as well as helpfulness aspect", so in practice they *un-align the aligned model* (SFT of Llama-2-7b-chat on 100 harmful QA pairs) and use δ_safe = θ_aligned − θ̃_unaligned (Eq. 11).
- **Consequence for us:** our "full RESTA" arm used Δ = tiny-aya-global − tiny-aya-base (as in SafeTune's README example, instruct − pretrained). That is the setting the RESTA paper explicitly avoids. Our Gate 3 full arm is therefore **SafeTune's documented RESTA recipe, not RESTA as published**. A faithful arm would un-align Global on ~100 harmful examples and use Global − unaligned. **Not run.**
- Models: Llama-2-7B-chat; FT on Chinese/English/Hindi Alpaca, code, math; PEFT and full FT. Harmfulness 18.6→5.1% (PEFT).

### CircuitKIT paper — Seth, Gosalia, Kasliwal, Sankarapu, arXiv:2607.19317 [FULL, §11]
- **Faithfulness vs usefulness — verified.** §11.5 (structural pruning), takeaway: "Patch faithfulness does not predict pruning here"; on the sharpest cell faithfulness *anti-correlates* with retention (ρ = −0.78). §11.8 summary: a faithfulness score is evidence about what a circuit explains, not a forecast of intervention effects.
- **§11.4 refusal circuit:** custom-data refusal circuit recovers 85% of the refusal gap under patching and beats size-matched random (1.71×), yet under hard zero-ablation the isolated circuit *prefers to comply*. Soft patching and hard ablation disagree. Our Gate 1 confound (hard ablation with MLP 0/35 collapses benign prompts too) is the same family of effect.
- No drift repair, no multilingual safety, Tiny Aya listed as experimental.

### Lee et al., "Refusal Localizes, the Damage Relocates", arXiv:2610.00320 (29 Sep 2026) [FULL main + App. H] — **closest prior work for the circuit finding**
- 6 English instruct checkpoints (Llama-3.1-8B, Tulu-3, OLMo-2 7B/13B, Qwen2.5-14B, Yi-1.5-9B). Harmful LoRA attacks (5–100 examples, 3 seeds). English only.
- **Layer-level** localisation (probes + lockstep activation patching → transition depth ℓ*). Freezing the layers where refusal localises does not protect refusal: the damage moves above the frozen band.
- Repairs: removing top-2 singular directions of the LoRA update restores refusal (0.76–1.00) under baseline attacks, fails under adaptive attacks. **Activation addition of the clean refusal direction (App. H): best 0.37 refusal vs 0.92 clean on Llama-3.1-8B** — close to our EN 0.42 vs 0.82.
- Controls: opposite-end layer freezes, norm-matched shrinkage; no size-matched random components. Over-refusal on XSTest measured.
- **Overlap with us:** "where refusal lives ≠ where fine-tuning breaks it" is already shown at layer granularity. Our addition: head-level circuit from CircuitKIT, tested as a repair target against 100 size- and layer-matched random sets, plus 5 languages incl. MR / Hinglish.

### BabelSteering — Stein et al., arXiv:2608.16577 [FULL]
- English steering vectors (true-refusal minus false-refusal, orthogonalised) folded into weights of **aligned** models (Gemma-7B, Llama-3.1-8B, Qwen-7B); 8 languages (no HI/MR/code-mixed). Single strength per model.
- **English shows the smallest change (~3 pp); largest gains in lower-resource languages.** Over-refusal rises unevenly (ZH +4 pp vs BN +23 pp). No random-direction control; no drift.
- **Overlap with us:** our "English direction acts more strongly outside English" replicates their direction, now on a *drifted* model, with a random-direction control, and with per-language strength as the fix for the uneven over-refusal they report.

### Miao et al., "Who Bridges Safety?", arXiv:2608.09095 [FULL main]
- Gemma-2-9B, Llama-3.1-8B, Qwen3-8B; EN + 9 NHR languages (ZH, KO, BN, TH, SW, HU, AF, IT, NE). Cross-layer shared pathways; masked-gradient training (~0.5–0.7% params). **Training required. No drift repair.**
- Size-matched random pathway controls: yes (masking random pathways changes ASR ~0; random-pathway tuning much weaker).

### Wang et al., arXiv:2505.17306 [FULL main]
- Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B (+ others for ablation); 14 languages (no HI/MR). **§5.4 adds** EN refusal vectors → compliance falls across languages (Gemma most languages → 0%). Over-refusal not measured; no drifted models. Harmful/harmless separation weaker outside English (silhouette EN 0.50 vs DE 0.22 for Llama).

### Liang et al., arXiv:2602.22554 [FULL]
- Training-free closed-form low-rank edit on safety-neuron columns of MLP up/gate; Llama-3.2 1B/3B, Qwen2/2.5 0.5–7B; EN, ZH, VI, JA, TH, ID, BN, HE. No drift, no random controls, no over-refusal.

### Poppi et al., arXiv:2410.18210 [ABS]
- Few-shot fine-tuning attack in one language breaks safety across languages; SIL localisation suggests safety info is language-agnostic (~20% of weights); freezing safety params doesn't stop the attack. No repair.

### Also seen (abstract/snippet only, cite if space)
- Kompella & Mahajan, arXiv:2608.08032 (Sarvam MoE, Indic): harm direction ≈ language-invariant mid-network (cos ≈ 0.9); refusal written late; editing the responsible attention heads had no effect.
- Glass et al., arXiv:2608.24988 (EMNLP 2026): refusal-ablation steering edits are partly undone by SFT; no harmful drift.
- HyperSafe, arXiv:2607.11475: inference-time safety recovery for fine-tuned models via a generated safety module.

## Novelty, revised after pass 2
- **No longer novel on its own:**
  - "refusal locus ≠ where fine-tuning damage sits" (Lee et al., layer level);
  - "English steering vector helps more in other languages, with uneven over-refusal" (BabelSteering, aligned models);
  - "English refusal direction transfers" (Wang).
- **Still ours (as far as pass 2 found):**
  1. Testing a CircuitKIT EAP-IG **head-level** refusal circuit as a post-drift repair target, both by activation patching and by weight-space Δ, against 100 / 5 **size- and layer-matched random** head sets.
  2. Steering a **drifted** multilingual model with an English-only direction, with a random-direction control, and **per-language strength chosen on held-out prompts** to fix over-refusal. Evaluated on HI, MR and code-mixed Hinglish, which none of the above cover.
  3. **Pre-registered** geometry-based per-language predictions (and their failure).
  4. Concrete findings about the Lexsi stack: `layer_suggestions` breadth; SafeTune's README RESTA recipe vs the paper's Δ.
