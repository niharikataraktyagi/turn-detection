# 08 — System Design: model, policy, deployment

Part A (docs 03–07) was the science: which prosodic cues carry turn-taking
information, and how much of the published result survives losing the
hand-annotated feature. Part B is the engineering: turn that into something that
runs inside a pause, on a CPU, next to an ASR and a TTS.

---

## 1. The target

Binary `floor_changes`. **Not** the 4-way taxonomy.

The 4-way labels existed for the analysis. The running system makes exactly one
decision — respond, or keep listening — so that is what it is trained on.
Training on the 4-way label and collapsing afterwards would spend capacity
separating `in-hold` from `com-hold`, which §3 of doc 07 showed prosody cannot do
and which does not change the action taken either way.

> Match the label space to the decision the system actually makes. Extra
> granularity that does not change the action is capacity spent on nothing.

## 2. Architecture

```
 text (last 40 words) ─► MiniLM-L6 ─► mean-pool(384) ─► 384→64 ─► GELU ─┐
                                                                        ├─► 128→64 → 64→1
 prosody (~30 features, speaker-normalised) ─► 30→64 ─► GELU ─► 64→64 ──┘
```

**Two branches, because the replication proved they are complementary.** Prosody
separates the floor axis at 0.63–0.67 balanced accuracy and sits at chance on the
syntactic axis. Text is the reverse. That is a measurement from our own data, not
an assumption borrowed from a paper.

**Late fusion of representations, not of logits.** Averaging two models'
probabilities cannot express a conditional like *"the syntax looks finished BUT
the articulation rate is low."* That conjunction **is** the `com-hold` trap — the
exact case that breaks text-only endpointers. Only a joint layer downstream of
both branches can represent it.

**Mean pooling, not `[CLS]`.** This MiniLM checkpoint is a *sentence-transformer*,
trained with mean pooling; its `[CLS]` token was never trained to carry sentence
meaning. Taking `[CLS]` out of BERT habit gives a much weaker representation with
no error message. Always check how a checkpoint was trained before choosing what
to pool.

### Why MiniLM-L6

| model | params | CPU latency |
|---|---|---|
| **MiniLM-L6-H384** | 22 M | ~5–8 ms (~2–3 ms int8) |
| DistilBERT | 66 M | ~25 ms |
| BERT-base | 110 M | ~60 ms |

The decision must complete inside the pause, on a core shared with ASR and TTS.
We feed 48 tokens and the completion signal lives in the last few words, so extra
capacity buys very little.

## 3. Calibration

The policy consumes `p` as a real number, not an argmax:

```
tau(p) = tau_min + (tau_max - tau_min) * (1 - p)
```

A systematically over-confident model therefore produces systematically short
wait times and interrupts people — **even with excellent AUC**. So we
temperature-scale (Guo et al. 2017): one scalar `T`, `p = sigmoid(logit / T)`,
fitted on held-out data.

Two properties worth knowing cold:
- It **cannot change the ranking** of any two examples, so AUC is mathematically
  unchanged. It fixes the *values*, which is exactly what the policy consumes.
- It must be fitted on **held-out** data. On training data the model is
  over-confident by construction, so you would fit `T ≈ 1` and achieve nothing.

We report **ECE** beside AUC. *AUC asks "are the rankings right?" ECE asks "when
it says 0.7, does it happen 70 % of the time?"* The policy needs the second.

## 4. The policy and its evaluation

Each decision point has a ground-truth delay `D`:

| case | error | cost |
|---|---|---|
| **hold** (speaker resumed after `D`) | false interruption iff `tau(p) < D` | cutting a human off — the expensive error |
| **change** (handover after `D`) | response latency **is** `tau(p)` | dead air — annoying, recoverable, linear |

One is a rate, the other a duration. No scalar combines them without inventing an
exchange rate between "interrupted a customer" and "200 ms of silence", and any
such number is fiction. So we report the **operating curve**: sweep the policy,
plot median latency against false-interruption rate. A curve below and to the
left dominates — better on both axes at every operating point.

The fixed-threshold baseline is the special case `tau_min == tau_max`. Pleasingly,
**the baseline is a degenerate member of our own policy family**, not a different
kind of object, so the comparison is exact rather than analogical.

`pareto_front()` keeps only non-dominated settings. Without it the scatter of
every `(tau_min, tau_max)` pair is unreadable *and* misleading — a cloud of bad
settings makes a good system look no better than a bad one.

## 5. Training protocol

**Out-of-fold predictions, not one test split.** With 8 speaker groups a single
holdout leaves ~2 groups (8 people) in test — a high-variance sample, and the
operating curve computed on it would be noisy exactly where it must be
trustworthy. Instead: `GroupKFold` over the 8 groups; each fold predicts its own
held-out groups; pooling gives one prediction for **every** decision point, each
from a model that never heard that speaker.

**Three nested holdouts that never touch:**
```
test fold       never seen in training OR calibration
validation grp  a WHOLE GROUP carved out of train; early stopping + temperature
train           the rest
```
Fitting the temperature on the test fold would tune a parameter on the data being
reported. Carving validation out as a whole group keeps speaker-disjointness at
every level.

**Two learning rates.** `3e-5` for the pretrained encoder, `1e-3` for the
randomly-initialised head. The encoder holds pretrained knowledge a large
gradient would wreck (catastrophic forgetting); the head starts from noise. One
rate for both either destroys the encoder or starves the head.

**Scaler fitted on train only.** Fitting on all data before splitting leaks test
statistics into training — a small leak that rarely shows in the metrics and is
the first thing a careful reviewer checks.

## 6. Deployment

**int8 dynamic quantisation** — weights int8 (4× smaller), activations quantised
on the fly, no calibration set required (unlike static quantisation). On a
transformer encoder the matmuls dominate and int8 GEMM is 2–4× faster on modern
CPUs. The accuracy cost is *measured*, not assumed: "quantise and hope" is how
silent regressions ship.

**Single-threaded benchmarking** (`torch.set_num_threads(1)`,
`intra_op_num_threads = 1`). A production worker gets one core, not the whole
laptop. Benchmarking on all cores and quoting the number is the most common way
latency claims turn out to be wrong by 4×.

**p95, not mean.** The decision is re-evaluated as the pause grows and sits in the
audio path; a 3 ms mean with a 60 ms tail still produces audible stutter.

**One inference per pause, not one per tick.** Neither the prosody (the speech is
already over) nor the transcript changes while a pause grows, so the naive loop
that re-runs the encoder every 100 ms is doing identical work repeatedly. Compute
once at pause onset, then watch the clock against `tau(p)`. Average cost per
decision becomes a single forward pass regardless of how long we wait. With a
streaming ASR attached, re-run on *transcript change* — not on a timer.

**Online speaker normalisation.** Offline we used whole-channel statistics. Live,
the demo starts from a population prior and swaps in the caller's own median F0
and RMS distribution after ~6 s of their speech. Same maths, accumulated
incrementally — which is what makes the offline normalisation deployable rather
than a laboratory convenience.
