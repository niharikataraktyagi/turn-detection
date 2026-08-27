# 04 — Exact Replication Spec (Kelterer et al. 2023)

Extracted from the full paper. This is the ground truth we replicate on English.

## 1. Their setup

| | |
|---|---|
| Data | 70 min of GRASS (Austrian German), 5 min × 14 conversations, 14f + 14m |
| Units | **IPUs** = speech separated by pauses **> 150 ms** |
| N | 1011 tokens → **837** after exclusions |
| Classes | `in-hold` 222 · `com-hold` 368 · `change` 247 |
| Excluded | laughter, uncertain labels, turn-changes with **overlapping speech** at IPU end, and **questions** (removed from `change`) |
| Analysis window | **final 0.6 s of the IPU** (tuned: 0.6 beat 0.8 and 1.0 s; F1 = .61/.60/.60). Shorter IPU → shorter window |

Note their three categories map onto our 2×2 grid as: `change` = floor changes +
syntactically complete; `com-hold` = same speaker continues + complete;
`in-hold` = same speaker continues + incomplete ("point of maximum grammatical
control"). `trail-off` is excluded by them; we keep it.

## 2. The 11 features — exactly

**F0 (5).** Tracked with **Parselmouth** (Praat), corrected for octave jumps.
Speaker-normalised by converting **Hz → semitones relative to that speaker's
overall median F0**. Then over the window:

- `F0max`, `F0min`, `F0med`, `F0range`
- `F0slope = ΔF0 / Δt(F0min, F0max)` — slope between the min and max locations,
  *not* a regression slope. Sign encodes rise vs fall.

**Intensity (4).** RMS, **z-score speaker-normalised**:

- `Imax`, `Imed`, `Istd`, `t(Imax)` — the *position in time* of the intensity peak

**Duration (1).** `ArtR` — articulation rate.

**Categorical (1).** `finIntonation` — phrase-final intonation contour, levels
**terminal / continuing / high-rise**. Explicitly *not* the same as final F0
slope: "continuing" subsumes several F0 movements (comma intonation).

> ⚠️ **`finIntonation` is a hand-annotated perceptual label.** See §5 — this is
> the single most important thing in this document for us.

## 3. Their models

**Random Forest** (scikit-learn, `n_estimators=100`, `max_features='sqrt'`,
Gini). Used *as an instrument for feature ranking, not as a classifier.*
Three **pairwise** comparisons rather than one 3-way — they state pairwise gave
better classification and a more reliable ranking. 80/20 split, 10 random splits
cross-validated (because of large between- and within-speaker variance).

**Regression:** linear mixed-effects models (`lme4`), DV = each of the 10
continuous features, IV = Category, **Speaker (N=18) as random effect**,
pairwise contrasts via `emmeans`. `finIntonation` modelled with multinomial
logistic regression (`nnet::multinom`).

## 4. Their results — the numbers to beat / compare against

### F1 scores

| Comparison | F1 |
|---|---|
| in-hold vs com-hold | in-hold **.60** / com-hold **.79** |
| in-hold vs change | in-hold **.78** / change **.81** |
| com-hold vs change | com-hold **.68** / change **.40** |

### Feature importances (top 4 per comparison)

| in-hold vs com-hold | in-hold vs change | com-hold vs change |
|---|---|---|
| finInt **.123** | finInt **.222** | Imed .021 |
| ArtR .012 | ArtR .024 | ArtR .015 |
| Imed .009 | Imed .008 | Imax .014 |
| t(Imax) .007 | Istd .008 | F0slope .014 |

### Significant effects (mixed models)

- **finIntonation** — `change` → terminal; `in-hold` → continuing;
  `com-hold` → inconsistent, hosts most high-rises. 49 % of all terminal contours
  occur in `com-hold`, which matches Ford & Thompson's observation that only
  ~half of transition-relevance places actually produce a turn shift.
- **ArtR** — `change` > `com-hold` > `in-hold`, **all three pairwise significant**
  (p<.01, p<.0001, p<.0001). Slowest articulation in incomplete holds.
- **Intensity** — higher in `in-hold`. **This contradicted their own hypothesis**
  (they predicted higher in com-hold). Their explanation: higher subglottal
  pressure mid-TCU than at TCU end, or a strategy to contrast with trail-offs.
- **F0max / F0range / F0med** — `com-hold` highest (p<.05 / p<.01 / p<.05).
  `in-hold` vs `change` **not significant** for F0max and F0range.
- **F0min, F0slope** — **no significant differences at all.**
- F0slope distribution is **bimodal** (falls and rises). `in-hold` → shallow falls
  and many flat curves; `change` → more and steeper falls; `com-hold` → more and
  steeper rises.

## 5. ⚠️ The finding that changes our engineering

**The most important feature in the paper is not computable at inference time.**

`finIntonation` dominates the ranking (.222 and .123 — an order of magnitude
above every acoustic feature) and it is a **human perceptual annotation** of the
functional phrase-final contour. The paper says explicitly it is *not* the same
as final F0 slope. And `F0slope`, the obvious computable substitute, showed **no
significant difference between categories**.

So a naive reading — "just use their features" — silently imports a label a real
system cannot have. Strip `finIntonation` out and what remains are acoustic
features with importances of .008–.024, i.e. very weak individually.

**What we do about it**, and this is the interesting engineering contribution:

1. Build **computable contour-shape features** as a proxy for `finIntonation`:
   fit both a linear and a quadratic to the semitone-normalised F0 contour over
   the final 300 ms and the final 600 ms; keep slope, curvature, R², terminal
   value relative to speaker baseline, and a rise/fall/level classification from
   those. Curvature is the piece plain `F0slope` misses — "continuing" contours
   are typically flat-then-slightly-rising or sustained, which a single slope
   cannot express.
2. Optionally train a small **contour classifier** to predict terminal /
   continuing / high-rise from the raw F0 contour, and feed its *posterior* into
   the main model — a learned, computable stand-in for the annotation.
3. **Let the text branch carry the load.** The paper is prosody-only by design.
   We have ASR partials, and lexico-syntactic completion is a far stronger and
   cheaper signal. Prosody's job is the residual — specifically `com-hold`.

> Interview line: *"The paper's dominant feature is a hand-annotated perceptual
> contour label. It's not available in production, and the obvious computable
> substitute — F0 slope — was the one feature with no significant effect. So I
> designed curvature-based contour features and a learned contour posterior to
> recover it, and measured how much of the gap that closes."*

## 6. The second finding that shapes the policy

**`com-hold` vs `change` is barely separable from prosody**: F1 of **.40** for
`change` in that comparison — near chance.

That is the hardest and most important cell for us. Syntax is complete, the
speaker paused — and prosody can only weakly tell you whether they're done. It
means:

- **A perfect endpointer is impossible.** There is genuine ambiguity in the
  signal; sometimes even a human can't know. Ford & Thompson's ~50 % figure says
  the same thing.
- Therefore the system must be **calibrated, not confident** — output a
  probability and let the *policy* trade the cost off, which is exactly the
  adaptive-τ design. This is the empirical justification for the whole approach.
- Their inter-rater agreement (κ = 0.84 IPU / 0.75 PCOMP) is the practical
  ceiling. When asked "why isn't accuracy higher?" — this is the answer, with a
  citation.

## 7. Our replication plan (Part A)

Same 2 × 2 taxonomy, same >150 ms IPU threshold, same 0.6 s window, same
speaker-normalisation (semitones re speaker median F0; z-scored RMS), same
pairwise RF with `n_estimators=100, max_features='sqrt'`, same 10-split CV, same
mixed-effects analysis with speaker as random effect — but on **English
spontaneous speech (AMI)** and with **computed** contour features in place of
the hand annotation.

Pre-registered hypotheses (their findings, transferred):

- **H1** ArtR: `change` > `com-hold` > `in-hold` ← their most robust result
- **H2** Intensity (`Imed`): higher in `in-hold` than `change`
- **H3** F0range / F0max: highest in `com-hold`
- **H4** Contour: `change` → terminal/falling; `in-hold` → continuing/flat
- **H5** `com-hold` vs `change` will be the weakest separation

Report which replicate in English and which don't. **A negative result here is
still a result** — cross-linguistic non-transfer is publishable-grade honesty and
plays extremely well in an interview.
