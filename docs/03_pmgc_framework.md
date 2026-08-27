# 03 — Grounding the Prosody Branch in Kelterer et al. (2023)

**Papers**
- Kelterer, Wepner, Linke & Schuppler (2023). *Points of Maximum Grammatical
  Control – The Prosody of a Turn-Holding Practice.* ICPhS 2023, Prague,
  pp. 3467–3471.
- Schuppler et al. (2025). *Turn-taking annotation for quantitative and
  qualitative analyses of conversation.* (arXiv:2504.09980) — the companion
  annotation scheme.

---

## 1. What the PMGC paper actually is

**Data:** 837 inter-pausal units (IPUs) from spontaneous **Austrian German**
dyadic conversation — the GRASS corpus (Graz Corpus of Read and Spontaneous
Speech), ~95 minutes of conversational material.

**Method:** 11 prosodic features (F0, intensity, articulation-rate family),
then a **Random Forest** used as a feature-importance instrument to find which
cues separate *incomplete-hold* from other categories.

**Headline finding:** when a speaker pauses at a grammatically **incomplete**
point and intends to keep the floor (an *incomplete-hold*, a "point of maximum
grammatical control"), the prosody is systematically different from a pause at a
grammatically complete point:

| Cue | Incomplete-hold, relative to complete holds |
|---|---|
| Intonation contour | **continuing** (non-final) |
| Intensity | **higher** |
| Articulation rate | **slower** |
| F0 | **flatter** |

Read that as a coherent physiological story: the speaker has *not* released
the floor, so they have not begun the articulatory "winding down" — subglottal
pressure stays up (intensity held), the pitch contour stays suspended rather
than falling to the baseline (flat F0, continuing contour), and they decelerate
into the pause rather than closing it off.

## 2. Why we do NOT literally reimplement it

Be honest about this in the interview; it reads as judgement, not laziness.

1. **The corpus isn't freely downloadable.** GRASS is released under a research
   licence agreement from TU Graz, not a public URL.
2. **It's German.** We are building for English/Hinglish support calls.
3. **It is an analysis paper, not a system paper.** The Random Forest is used as
   a *statistical instrument* to rank feature importance over 837 hand-annotated
   units. It is not a deployable real-time endpointer, and reproducing its
   feature ranking is not the same as building one.

Reimplementing it verbatim would cost us both days and produce a weaker artifact
than what we can build instead.

## 3. What we take from it — which is a lot

### 3.1 The label taxonomy (this is the big one)

My original design was binary: complete vs incomplete. The Schuppler annotation
scheme is strictly better, because it factors the problem along **two
independent axes**:

- **Did the floor change?** (did the other speaker take over, or did the same
  speaker continue?)
- **Was the utterance syntactically complete at the pause?**

Crossing them gives four cells:

|                          | syntactically **complete** | syntactically **incomplete** |
|--------------------------|----------------------------|------------------------------|
| **same speaker continues** | `hold`                    | `incomplete-hold`            |
| **floor changes**          | `change`                  | `trail-off`                  |

This matters enormously for us:

- `change` — the true positive. Responding here was correct.
- `incomplete-hold` — the classic false-interruption trap. **Text alone solves
  this**: "I'd like to cancel my" is obviously unfinished.
- **`hold` — the case that breaks text-only models.** The speaker finished a
  complete clause, paused, and *kept going*: "I ordered it last Tuesday.
  [700 ms] And it still hasn't shipped." A text-only endpointer sees a complete
  sentence and fires. **Only prosody can tell you they weren't done.**
- `trail-off` — the mirror case: syntax incomplete but the floor changed anyway.
  Text says "wait", reality says "respond".

> `hold` and `trail-off` are precisely the cells where text and prosody
> disagree — and they are the entire justification for building a *fused* model
> rather than a text-only one. Kelterer et al. is the evidence that prosody
> carries the missing signal, and it tells us the *direction* of each effect.

That is a far more sophisticated framing than "I trained a binary classifier",
and it is now the spine of the project.

### 3.2 Directional hypotheses for feature design

Instead of throwing arbitrary features at a model, we now have four
**pre-registered hypotheses** from published phonetics, which we test on English:

- **H1** intensity at the pause is *higher* for `incomplete-hold` than `change`
- **H2** articulation rate approaching the pause is *slower* for `incomplete-hold`
- **H3** F0 is *flatter* (smaller range / shallower final slope) for `incomplete-hold`
- **H4** final F0 falls toward the speaker's baseline for `change`

Stating hypotheses before fitting, then reporting whether they replicated, is
how you look like a researcher rather than someone who ran `feature_importances_`
and wrote a paragraph around it.

### 3.3 The non-circularity rule

The annotation scheme defines syntactic completion **"without considering
intonation"** — deliberately, to avoid circularity. If you let prosody inform
your completion labels and then test whether prosody predicts completion, you
have proved nothing.

We inherit this rule: our syntactic-completion label is derived from **text
only** (dependency parse / POS), never from audio. Volunteer this in the
interview — methodological hygiene is rare and noticed.

### 3.4 IPU segmentation and the 150 ms threshold

They segment on pauses **≥ 150 ms**. We adopt the same threshold, which also
conveniently matches the granularity a real VAD operates at. Their inter-rater
agreement (Cohen's κ = 0.84 for IPU labels, Fleiss' κ = 0.75 for PCOMP) is worth
knowing: it's the practical ceiling on what any model can achieve, and a good
answer to "why isn't your accuracy higher?"

## 4. The resulting project narrative

Two parts, and the pairing is what makes it strong:

**Part A — replication (science).**
Take the prosodic cues Kelterer et al. found for Austrian German turn-holds and
test whether they transfer to **English** spontaneous speech (AMI). Same feature
family, same Random Forest importance analysis, same four-way IPU taxonomy.
Report which hypotheses replicated and which didn't. *A cross-linguistic
replication is a real contribution, and it costs us almost nothing extra.*

**Part B — deployment (engineering).**
Turn the validated cues into a fused text + prosody model that runs in
single-digit milliseconds on one CPU core, drives an adaptive wait-time policy,
and **dominates a fixed-threshold VAD baseline on the latency vs.
false-interruption operating curve**.

Part A gives it intellectual grounding. Part B gives it production credibility.
The JD asks for both — "real projects you can defend in depth" and "low latency
models that run at a fraction of the cost via CPU at production scale."

## 5. What this changes in the code

- Labels become **4-way IPU categories**, not binary. Binary
  `respond / keep-listening` is derived from them (`change` + `trail-off` = respond).
- We add a **text-only syntactic-completion classifier** as an explicit
  intermediate signal (and as the text-only ablation baseline).
- Feature extraction mirrors their family: F0 (slope, range, final value,
  speaker-normalised in semitones), intensity (level and slope), articulation
  rate (and its deceleration), plus pause duration.
- Analysis adds a **Random Forest feature-importance study** with the four
  hypotheses tested explicitly — this is the replication.
- Headline result is still the **operating curve**, with ablations:
  fixed-VAD → text-only → prosody-only → fused.
