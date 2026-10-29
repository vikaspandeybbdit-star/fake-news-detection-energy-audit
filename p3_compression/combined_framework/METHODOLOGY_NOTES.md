# Methodology notes — combined pruning + distillation framework

Working notes for whoever (human or AI) drafts a paper/section from this milestone's
results once `search.py` has actually run. Not a paper draft itself — captures the
decisions and open questions a draft would need to resolve.

## Novelty claim: what's actually verified vs. what still needs checking

**Verified**: P2 (`P2_survey_draft.md`, this series' own PRISMA-screened survey) did not
surface a paper combining structured pruning and knowledge distillation specifically
for fake news detection, evaluated jointly on accuracy and energy. That's a real,
citable gap *within this project's own existing literature review*.

**NOT yet verified — do this before claiming novelty in any paper draft**: pruning +
distillation combined is a well-known idea in the *general* efficient-NLP literature
(the two techniques are old enough, and "combine compression techniques" is an obvious
enough next step, that it would be surprising if literally nobody has published it for
*any* task). Before drafting an abstract that claims this combination is novel, run a
real literature check — same discipline this project used for every architecture
section (EBT's reference repo read directly, SpikeBERT's author list verified against
the actual arXiv page, Linformer/Performer/Charformer's mechanisms read from source
before writing adaptation code). The defensible claim, until that check is done, is
narrower: "novel *for fake news detection specifically*, and evaluated under this
project's fixed accuracy-energy protocol" — not "novel, full stop." Overclaiming here
would break this project's own established honesty discipline (see
`P3_compression_draft.md` Section 5's repeated insistence on stating scope limits
plainly).

## What "the proposed framework" actually means

`search.py`'s output is a **Pareto frontier** (a set of non-dominated configs), not a
single winning number. A paper draft needs to either:
1. Report the frontier as the contribution (a menu of accuracy-energy tradeoffs a
   deployer can pick from), which is the more honest framing and matches this
   project's own "report positive and negative results, don't force a single
   conclusion" practice; or
2. Recommend one frontier point using an explicit, stated criterion (e.g., "best MCC
   among configs within 20% of the lowest energy," or "lowest energy among configs
   within 1 MCC point of the undistilled+unpruned baseline") — if this route is taken,
   the criterion must be stated and justified, not picked post-hoc to flatter whichever
   config happened to look best.

Framing (1) is the safer default and requires less justification. Don't default to (2)
without a clear reason a reviewer would accept.

## Two possible outcomes, both worth planning for

- **Compounds**: pruning the distilled student saves energy roughly additively (or
  better) on top of distillation's own −46 to −49%, with an accuracy cost no worse than
  pruning alone typically costs. This is the "interesting positive result" case — the
  paper's contribution is straightforward.
- **Doesn't compound**: pruning a student that's already been compressed via
  distillation runs into diminishing returns (the redundancy distillation already
  removed overlaps with what head-pruning would have removed), or costs
  disproportionate accuracy since there's less slack left in a smaller model. This is
  still a real, useful, citable finding — "compression techniques are not simply
  additive, and here's the empirical Pareto curve showing why" is a legitimate
  contribution in its own right, consistent with this project's practice of reporting
  EBT's/Linformer's/Performer's energy-increased-not-decreased results just as plainly
  as prune/distill's wins.

Don't let which outcome actually occurs change how rigorously it gets reported.

## Where does this fit in the paper series?

Open decision, not yet made — options:
- **New section of P3** (`P3_compression_draft.md`), analogous to how M9-M13 were
  folded in as Sections 3.5-3.9/4.5-4.9. Pro: reuses P3's existing fixed-protocol
  framing and citation list, fastest to draft. Con: P3 already has a clean narrative
  ending ("distillation is the winner, carried forward to P4") — adding a combined
  search result that might beat distillation alone complicates that ending and would
  require revising P3's Abstract/Conclusion, which are already finalized.
- **Standalone short paper/technical note**, positioned as a direct follow-up to P3
  rather than a section within it. Pro: cleaner narrative ("P3 found the best single
  technique; this note asks whether combining the best two beats either alone"), easier
  to scope tightly, doesn't require touching P3's already-finalized text. Con: another
  full paper-writing pass (mandatory sections, references, etc.) for what might be a
  fairly short result.
- **Retroactive revision to P4** (`P4_benchmark_draft.md`), since P4 is exactly the
  "benchmark the best P3 configuration" paper and this milestone might change what "the
  best P3 configuration" is. Pro: correct in principle — if the combined config beats
  plain distillation, P4's own comparison table is currently built on the wrong
  configuration. Con: P4 is marked DONE; reopening it has ripple effects on its own
  Abstract/Conclusion and possibly its literature comparison numbers.

Recommend resolving this with the user once real results exist (a frontier that clearly
beats distillation alone argues for revising P4; a frontier that doesn't argues for a
short standalone note or just folding into P3 as one more honest exploratory finding).
Don't pre-commit to a venue/section before the numbers exist.

## Draft outline (if the standalone-note route is chosen)

1. **Motivation**: P3 evaluated quantize/prune/distill individually; this asks whether
   the two energy-reducing ones compound when combined.
2. **Method**: distill (DistilBERT student, KD temp=2.0, alpha=0.5, unchanged from
   `distill.py`) then structured head-pruning (magnitude-based, L1 norm of output
   projection columns, unchanged from `prune.py`'s method) at {0,25,50,75}% ratios, one
   epoch of post-prune recovery fine-tuning (same as `prune.py`'s own practice).
3. **Search space and Pareto methodology**: the 8-cell grid, dominance definition
   (`search.py`'s `is_dominated`), LIAR-only scope and why (matches every other
   exploratory milestone's scoping rationale).
4. **Results**: `search_results.csv` full table + `pareto_frontier.csv` highlighted;
   plot MCC vs. energy with the frontier marked, if a figure is wanted.
5. **Discussion**: compounds or doesn't (see "two possible outcomes" above) — report
   whichever actually happened, with the same directness as the rest of this series.
6. **Limitations**: LIAR-only (not yet validated on ISOT/FakeNewsNet); pruning ratio
   grid is coarse (4 points, not a continuous sweep); quantization excluded from the
   search entirely (CPU/GPU energy incomparability, same caveat as everywhere else in
   this project).

## Reminders before running

- Smoke-test one cell (`python combine.py --dataset liar --prune_ratio 0.5 --distill`)
  before running `search.py`'s full grid — see `README.md`.
- Real bugs are likely on the first Kaggle run, per this project's own track record
  (every prior milestone hit at least one). Most probable candidate here: a DistilBERT
  attribute-name typo or shape mismatch in `prune_distilbert_heads` — verified against
  real `DistilBertSelfAttention` source before writing, but never executed. The
  device-placement bug that hit Linformer/Performer (new layers constructed on CPU
  after the base model was already moved to CUDA) does NOT apply here — both
  `prune_layer_heads` (reused from `prune.py`) and `_prune_distilbert_layer_heads`
  already capture the existing weight's device and explicitly `.to(device)` every new
  `nn.Linear`, the same pattern `prune.py` has used successfully since its first
  Kaggle run. Checked by re-reading the code, not assumed.
