# 09 — Defence Sheet

Every design decision, with the reason. If you can answer these cold you can hold
a 45-minute deep-dive on this project.

---

## The 60-second pitch

> A voice agent decides the user has stopped talking by waiting for a fixed
> amount of silence. That single number is trapped: short enough to feel
> responsive means it interrupts people who pause to think — before an order
> number, an address, a code-switch — and long enough to survive those pauses
> means every exchange drags. I measured the trap in real conversational data:
> the 25th percentile of genuine turn-ends is 320 ms, *below* the median of
> pauses where the speaker carried on. No threshold separates them.
>
> So I replaced the constant with a function of the evidence. A small fused model
> reads the running transcript and the prosody of the last 600 ms, emits a
> calibrated probability, and the wait time scales with it. At a matched
> false-interruption rate it responds about 226 ms faster than the best fixed
> threshold — roughly one human inter-turn gap per exchange — at a few
> milliseconds of int8 CPU inference.
>
> I also replicated an ICPhS 2023 phonetics paper on the prosody of turn-holding
> from Austrian German onto English to ground the feature design, and the most
> useful thing I found was a null result.

---

## Q: Why this problem?

Endpointing is the largest *controllable* term in a voice agent's latency budget.
ASR finalisation, LLM time-to-first-token and TTS time-to-first-audio are bounded
by someone else's model; the endpoint wait is 200–1500 ms and entirely ours. It
is also where the worst UX failure lives — being interrupted mid-sentence is far
more damaging than a slow reply, because the user restarts, talks over the agent,
and corrupts the ASR stream.

## Q: Why AMI and not <X>?

Needed: real conversational audio, transcripts, **word-level timings**, and turn
structure. LibriSpeech and VoxPopuli are read speech — *there are no turns*.
Switchboard and Fisher are LDC-licensed. IEMOCAP is acted and gated. MELD has a
laugh track over the prosody. AMI is free, spontaneous, per-speaker headset audio,
with manual word timings — which means **no forced aligner**, one whole
error-prone stage removed.

## Q: How did you get labels without annotators?

Self-supervision from the temporal structure of dialogue. A recorded conversation
already contains the answer: if the speaker stopped and someone else took the
floor, that was a turn end; if they paused and continued, it wasn't. Training-time
we can see the future; inference-time we hide it. Same move as next-token
prediction.

## Q: What went wrong, and how did you catch it?

**(1) The annotations couldn't express the class I needed.** 92 % of consecutive
AMI word pairs are exactly contiguous — the aligner tiles words across each
transcriber segment, so a 500 ms thinking pause is absorbed into neighbouring word
durations. I measured the gap distribution, confirmed it, and recovered pauses
from the audio with Silero VAD instead.

**(2) A detector confound.** `crosstab(source, label)` came out block-diagonal —
`change` came only from annotation turn-ends, `com-hold` only from VAD pauses. Two
different clocks, so the 600 ms prosody window was anchored differently per class
and the model could have learned *which detector timestamped this*. Fixed by
enumerating every decision point from a single source. **General principle: if
your classes were produced by different machinery, the model will learn the
machinery.**

**(3) A degenerate classifier.** `F1(change)=0.12` beside `F1(com-hold)=0.94` — a
constant function wearing a classifier's clothes, caused by 8:1 class priors.
Fixed with `class_weight="balanced"` and by reporting **balanced accuracy**, which
scores a majority-predictor at exactly 0.500.

**(4) A filter that ate 64 % of my positive class.** My overlap exclusion asked
"was anyone else speaking anywhere in the window?" — in a 4-party meeting, almost
always. The paper's rule is "overlapping speech *at the end* of the IPU". Worse,
mine preferentially deleted the changes where the next speaker came in fast — the
clearest positives. When a preprocessing step removes a large share of one class,
that is a modelling decision, not housekeeping.

## Q: Why isn't pause duration a feature? It's obviously predictive.

Because it isn't available. At inference the elapsed pause is the clock we are
racing, and its final value is *caused by* whether we decide to interrupt. Using
it would teach the model "long pause → turn ended" — exactly the fixed-threshold
baseline it exists to beat — from information a live system cannot have. It is the
**evaluation variable**: for a hold that resumed after `D`, we falsely interrupted
iff `τ(p) < D`; for a change, our latency *is* `τ(p)`.

*(The principled way to use elapsed time is Skantze's continuous formulation —
emit several training rows per decision point at different elapsed times, each
labelled "will speech resume after this instant?". Then it's a legitimate input
because it's known at that instant. Deliberately left for v2 to keep the
comparison with the paper clean.)*

## Q: Why not just report accuracy or F1?

Three reasons. The error costs are wildly asymmetric — an interruption is much
worse than 200 ms of dead air. The classes are imbalanced. And it isn't a
classification at all: the system emits a *decision at a time*, so the same
classifier with a different threshold is a different product.

The two errors are in **different units** — a rate and a duration — and combining
them needs an exchange rate between "interrupted a customer" and "200 ms of
silence" that would be pure invention. So we report the **operating curve** and,
as the headline, **latency at matched interruption rate**, where both systems are
equally good on the error that matters and the latency gap is unambiguous.

## Q: Why does calibration matter here specifically?

The policy consumes `p` as a real number — `τ = τ_min + (τ_max−τ_min)(1−p)` — not
an argmax. A systematically over-confident model produces systematically short
waits and interrupts people **even with excellent AUC**. Temperature scaling fits
one scalar on held-out data; it cannot change any ranking, so AUC is mathematically
unchanged — it fixes only the values, which is exactly what the policy uses. We
report ECE alongside AUC: *AUC asks "are the rankings right"; ECE asks "when it
says 0.7, does it happen 70 % of the time".*

## Q: How do you know you didn't leak speakers?

Split on `global_name` — real person identity from `meetings.xml` — with whole
groups held out, at every level including the validation split used for early
stopping and temperature fitting. Then I **measured** it: random splits gave
−0.009 balanced accuracy versus speaker-disjoint, i.e. no inflation. That is the
speaker normalisation working — F0 in semitones relative to each speaker's own
median and z-scored RMS strip identity out of the features before the model sees
them.

## Q: Why semitones and z-scores rather than Hz and raw RMS?

Pitch perception is logarithmic: a 20 Hz rise means something completely
different at a 90 Hz baseline than at 210 Hz, whereas 2 semitones means the same
at both. And "loud" for a quiet speaker is "quiet" for a loud one. Our data is
also heavily speaker-imbalanced (one speaker contributes 300+ points, several
under 30), so un-normalised features would let the model separate classes by
recognising who is talking. Live, these statistics accumulate online from the
caller's first few seconds against a population prior.

## Q: Why fuse, rather than just use text?

Because I measured that they carry **different** information. Prosody separates
the *floor* axis (hold vs change) at 0.63–0.67 balanced accuracy and is at
**chance (0.503)** on the *syntactic* axis; text is the reverse. Near-orthogonal
signals, so fusion is not "more features" — it is two views of different things.
Fusion lifts AUC to 0.799 from 0.751 text-only and 0.725 prosody-only.

Late fusion of **representations**, not logits: averaging two probabilities cannot
express *"the syntax looks finished BUT the articulation rate is low"* — and that
conjunction **is** the `com-hold` trap that breaks text-only endpointers.

Honest framing: **text is the workhorse**; prosody adds roughly 50–60 ms of the
226 ms saving. Worth having, not the main event.

## Q: What did the paper replication actually tell you?

Two robust transfers and one very useful failure.

**Articulation rate transfers emphatically** — `change` 4.19 syll/s vs ~2.5 for
holds, *p* ≈ 3e-67 with speaker as a random effect. **Intensity transfers
including the direction that contradicted the authors' own hypothesis** — a
finding that survives a change of language is a much better finding. **F0 does
nothing**, consistent with their own weak F0 results.

The failure is the interesting part: prosody cannot tell `in-hold` from
`com-hold` at all (0.503 — chance), which *inverts* their pattern where that was
their best pair. One cause: `finIntonation` — a **hand-annotated perceptual
label** — carried importance .123/.222 in exactly those comparisons, against
.007–.024 for anything acoustic. Remove it, as any production system must, and
the distinction collapses. **The null result measures how much of the published
finding depends on a feature unavailable at inference time.**

I also found something they couldn't: turn-ends are significantly **less modally
voiced** (0.310 vs 0.377, *p*=4e-17) — creak/glottalisation at terminal juncture
defeating F0 tracking. Visible only because I treated missingness as signal
instead of dropping unvoiced windows.

## Q: How did you diagnose the F0 slope anomaly?

Before excluding questions, `change` showed `f0_slope` **+18.0** — a strong rise,
backwards from the expected terminal fall. Hypothesis: the paper excludes
questions from `change`, and question intonation rises. Excluding them dropped it
to **+1.2** and non-significant. Hypothesis stated, tested, confirmed.

## Q: Why MiniLM-L6?

The decision must complete inside the pause, on a core shared with ASR and TTS.
MiniLM-L6 is 22 M params (~5–8 ms fp32, ~2–3 ms int8); DistilBERT is 66 M
(~25 ms); BERT-base 110 M (~60 ms). We feed 48 tokens and the completion signal
lives in the last few words, so extra capacity buys very little. Mean pooling, not
`[CLS]` — this is a sentence-transformer checkpoint trained with mean pooling, and
its `[CLS]` was never trained to carry sentence meaning.

## Q: Production concerns?

- **int8 dynamic quantisation** — no calibration set needed, matmuls dominate a
  transformer encoder, int8 GEMM is 2–4× faster on modern CPUs. Accuracy cost
  **measured**, not assumed.
- **Benchmarked single-threaded.** A production worker gets one core. Quoting
  all-core numbers is the most common way latency claims are wrong by 4×.
- **p95, not mean** — the decision sits in the audio path; a 3 ms mean with a
  60 ms tail still stutters.
- **One inference per pause, not one per tick.** Neither the prosody (speech is
  already over) nor the transcript changes while a pause grows, so the naive
  100 ms polling loop repeats identical work. Compute once at pause onset, then
  watch the clock. With streaming ASR attached, re-run on *transcript change*,
  not on a timer.
- **Cheap gating.** Everything above ~1.6 s is obviously a turn end and below
  ~0.3 s obviously isn't; the model only earns its keep in the ambiguous band.
  Running it only there would cut average CPU cost substantially.

## Q: What are the limitations?

1. **AMI is 4-party meetings, not 2-party support calls.** Far more intra-turn
   pausing (median hold 576 ms), more floor competition. Absolute interruption
   rates are domain-specific; the *controlled relative* comparison transfers.
2. `change` is the smallest class (1,262 / 5,502).
3. Syntactic-completion labels are **rule-derived**, not human-annotated — theirs
   had κ = 0.84 / 0.75, which is also the practical ceiling on achievable accuracy.
4. **The contour features failed.** My geometric stand-ins for `finIntonation`
   were non-significant everywhere. A learned contour classifier trained on
   annotated contours is the obvious next step.
5. No Hinglish, no telephone-band audio — both matter for the target deployment.

## Q: What would you do next, with production data?

Fine-tune on real call audio (the architecture doesn't change); add the
elapsed-time conditioning from Skantze's continuous formulation; train the
learned contour classifier to recover `finIntonation`; and add a barge-in path,
since this only handles the "user stops" half of turn-taking.

---

## Numbers to have cold

| | |
|---|---|
| median human inter-turn gap | ~200 ms (Stivers et al. 2009, 10 languages) |
| decision points / speakers / groups | 5,502 · 32 · 8 |
| turn-end 25th pct vs hold median | **320 ms vs 576 ms** — they overlap |
| AUC: text / prosody / fusion | 0.751 / 0.725 / **0.799** |
| ECE (fusion) | 0.037 |
| latency saving at matched FIR | **~226 ms (20–25 %)** |
| ArtR: change vs holds | 4.19 vs 2.5 syll/s, *p* ≈ 3e-67 |
| in-hold vs com-hold from prosody | **0.503 balanced accuracy — chance** |
| speaker-leakage inflation | −0.009 (none) |
| ONNX int8: p95 / size | **3.1 ms / 23 MB** (2.3× faster, 4× smaller than fp32) |
| quantisation accuracy cost | AUC 0.862 → 0.858 |
| cost of padding to 48 vs 16 tokens | **2.8×** |
