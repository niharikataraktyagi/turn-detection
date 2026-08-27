# 07 — Replication Results: Austrian German → English

Kelterer, Wepner, Linke & Schuppler (ICPhS 2023) on GRASS (Austrian German,
837 IPUs) → re-run on AMI (English, spontaneous multi-party meetings).

**Analysis set:** 3,785 decision points · 32 speakers · 8 speaker groups
(com-hold 2,224 · in-hold 1,225 · change 336), after their exclusions
(overlapping speech at the boundary, questions).

Every feature is speaker-normalised (F0 in semitones re that speaker's median;
RMS z-scored). Splits are speaker-disjoint. `finIntonation` is replaced by
computable contour geometry, because it is a hand annotation.

---

## 1. Hypothesis verdicts

| | Hypothesis (theirs) | Result in English | |
|---|---|---|---|
| **H1** | ArtR: change > com-hold > in-hold | 4.19 > 2.54 > 2.49; change vs holds **p≈3e-67**, com-hold vs in-hold **ns** | **partial** |
| **H2** | Intensity higher in in-hold than change | −0.27 > −0.47, **p=1.3e-11** | **replicates** |
| **H3** | F0 range highest in com-hold | all F0 contrasts **ns** | **no** |
| **H5** | com-hold vs change is the weakest pair | weakest is **in-hold vs com-hold** (chance) | **no — inverted** |
| **H6** | *(ours)* turn-ends are less modally voiced | 0.310 vs 0.377, **p=4e-17** | **new finding** |

## 2. The headline: articulation rate transfers, F0 does not

**Articulation rate is the finding.** `change` 4.19 syll/s versus ~2.5 for both
hold types, at *p ≈ 3e-67* with speaker as a random effect. Speakers accelerate
into a turn end and decelerate into a hold, and this is true in Austrian German
and in English. It is also the top computable feature in every Random Forest
ranking (importance 0.179, roughly 2.5× the next feature) — mirroring the paper,
where ArtR was the highest-ranked feature after the hand-annotated one.

**Intensity replicates including its counter-intuitive direction.** The authors
predicted higher intensity in *complete* holds and found the opposite; we find
the same reversal (`in-hold` −0.27 > `change` −0.47, p=1.3e-11). Their
explanation — higher subglottal pressure mid-TCU than at TCU end — gains
credibility from surviving a change of language.

**Every F0 contrast is non-significant.** f0_range, f0_max, f0_slope: all ns.
This is consistent with their own weak F0 results (they found no significant
difference for f0_min or f0_slope either), and it says something practical: the
pitch features that dominate the turn-taking literature are, once you
speaker-normalise and control for speaker, doing very little work.

### The question confound, tested

Before excluding questions, `change` had f0_slope **+18.0** — a strong *rise*,
the opposite of the expected terminal fall. Excluding questions (as the paper
does) dropped it to **+1.2** and non-significant. Rising terminal contours in
the `change` class were question intonation, not turn-yielding. A concrete
example of why their exclusion criteria exist.

---

## 3. The most interesting result: what we *couldn't* separate

Balanced accuracy, speaker-disjoint splits (0.500 = chance):

| Comparison | balanced accuracy |
|---|---|
| in-hold vs com-hold | **0.503**  ← chance |
| com-hold vs change | 0.627 |
| in-hold vs change | 0.668 |

**Prosody cannot tell `in-hold` from `com-hold` at all.** And on reflection that
is exactly right: in both cases the speaker paused and *kept the floor*. Their
communicative intent is identical. What differs between them is **syntax**, not
prosody — and syntax is not what a pitch contour encodes.

This inverts the paper's pattern, where `in-hold vs com-hold` was their *best*
comparison (F1 .60/.79) and `com-hold vs change` their worst (F1 .40).

**One cause explains the inversion: `finIntonation`.** It carried importance
.123 and .222 in the two comparisons involving `in-hold`, an order of magnitude
above any acoustic feature (.007–.024). Remove it — as any production system
must, since it is a human perceptual annotation — and the `in-hold` distinction
collapses to chance.

> So our null result is not a failed replication. It is a **measurement of how
> much of the published result depends on a feature that cannot be computed at
> inference time.** That number is: essentially all of it, for the in-hold
> distinction.

---

## 4. Why this validates the fusion architecture

The two axes of the 2×2 turn out to be carried by two different modalities, and
they are close to orthogonal:

```
                        | syntactically complete | syntactically incomplete
   same speaker holds   |       com-hold         |        in-hold
   floor changes        |        change          |       trail-off
                        └─────────── TEXT decides this axis ──────────┘
   PROSODY decides this axis  (balanced acc 0.63–0.67, ArtR p≈3e-67)
   ↑
```

- **Prosody** separates *hold vs change* — the floor axis — at 0.63–0.67
  balanced accuracy from acoustics alone. This is the axis the system actually
  needs, because the decision is "respond or keep listening".
- **Prosody is at chance on the syntactic axis**, which is precisely the axis a
  text encoder handles trivially.

That is the empirical case for fusing them, and it is much stronger than "more
features are better": the two modalities are informative about *different,
complementary* things. It also predicts what the ablations should show — text
alone should be strong but blind to `com-hold`, prosody alone should be weak but
uncorrelated with text's errors, and the fusion should beat both.

---

## 5. Speaker leakage: measured at zero, and why

Random (speaker-leaking) splits vs speaker-disjoint splits differed by
**−0.009** balanced accuracy — no inflation at all.

That is not luck. It is the speaker normalisation working: converting F0 to
semitones relative to each speaker's own median, and z-scoring RMS per speaker,
strips the speaker-identity information out of the features before the model
sees them. The leakage test is the evidence that the normalisation did its job.

Worth stating in an interview as a pair: *"I split by speaker to prevent voice
memorisation, and then measured that random splits gave no advantage — which
tells me the per-speaker normalisation had already removed the identity signal."*

---

## 6. Honest limitations

1. **`change` is small (336)** after exclusions. Confidence intervals on the
   change comparisons are correspondingly wide.
2. **AMI is meetings, not calls.** Four-party floor competition, longer turns.
3. **Our syntactic-completion labels are rule-derived**, not human-annotated.
   Their κ was 0.84 (IPU) / 0.75 (PCOMP); we have no equivalent agreement figure.
4. **Contour features underperformed.** `contour_curv_600` was ns everywhere.
   The geometric stand-in for `finIntonation` did not recover it. A learned
   contour classifier trained on annotated contours would be the next thing to try.
