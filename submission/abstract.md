# Abstract

**We found the refusal circuit. It was the wrong place to repair: safety drift and its repair in a multilingual model**

Track 2b (AI Safety and Correction); problem statements #6, #9, #10, with #4 throughout. Model: Cohere Tiny Aya Global. Lexsi repositories chained: CircuitKIT (circuit discovery) and SafeTune (RESTA weight repair, judge).

Fine-tuning an aligned model on mostly benign data can quietly remove its safety. We reproduce this on Tiny Aya Global: a LoRA fine-tune on 1,800 intent-classification examples plus 100 harmful pairs drops refusal of harmful requests from 82% to 4% in English, and collapses it in Hindi, Chinese, Marathi and code-mixed Hinglish as well. We then ask whether the model can be repaired without retraining, and whether interpretability can tell us where to repair. Every measurement uses the same parallel prompts in five languages, and every targeted edit is compared with random targets of the same size.

Three findings.

First, the refusal circuit that CircuitKIT finds in English is real but is not a repair target. Its 29 attention heads are needed for refusal (ablating them hurts more than all 100 matched random head sets), yet restoring them in the drifted model, by activation patching or by weight editing, does no better than random heads. The damage is spread through the upper layers and sits mostly in MLPs.

Second, how the repair is built decides whether it works. SafeTune's README recipe for RESTA (aligned minus pretrained) breaks the model. The recipe from the RESTA paper (aligned minus a deliberately un-aligned copy) restores refusal to the original level in all five languages (English 84%), keeps task accuracy, and barely raises over-refusal, using only English data. Safety vectors built from three different harmful datasets are nearly perpendicular as weight edits but cause nearly the same change in the model's activations; random and benign-data vectors do nothing.

Third, interpretability does help once it is asked a different question. Scoring units by where the safety vector would help most in the *drifted* model identifies a quarter of the vector, mostly MLP neurons, that repairs English as fully as the whole vector (86%); a random quarter reaches 48% and a size-ranked quarter 74%. By contrast, repairing from the drift itself, by scaling it back or by removing attribution-selected components of it, trades safety against the learned task, because the two share the same components.

We also report what failed: a pre-registered prediction that languages represented less like English would recover less (they recovered as much or more), and a label-free rule for per-language steering strength. An English-only refusal direction does restore refusal in every language by activation steering, but needs a different strength per language.

For practitioners: test a discovered circuit against random components before repairing at it; build RESTA's vector from an un-aligned copy; attribute on the drifted model, not the aligned one. Limits: 50 prompts per language, one drift recipe and seed, a judge hand-checked on 60 responses.

All adapters, data, generations, logs and code are public.
