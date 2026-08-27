# 06 — Leakage and Confounds: the four traps in this project

Every one of these produces a model that scores *well* and is *wrong*. None of
them raise an error. Collectively they are the most interview-relevant part of
the repo — anyone can train a classifier; noticing these is the job.

---

## Trap 1 — Speaker leakage across the split

**The trap.** Shuffle decision points and split randomly. `ES2002a` and
`ES2002b` are the same four humans, so the same voices land in train and test.
A prosody model then partly *recognises the speaker* rather than learning
turn-taking, and the test score is inflated.

**The fix.** Split on `global_name` (real person identity from
`meetings.xml`), and keep whole groups together. We have 32 people in 8 groups —
hold out groups, not rows.

**The tell.** If your test score drops sharply when you switch from random to
speaker-disjoint splitting, the original number was measuring voice recognition.

---

## Trap 2 — The detector confound  ⚠️ *the one that nearly got us*

**The trap.** Our two label groups originally came from two different sources:

```
              change  com-hold  in-hold  trail-off
internal           0      2204     1211          0
turn_end         953        14        4        532
```

Almost perfectly block-diagonal. `change`/`trail-off` were timestamped by the
**forced aligner** (annotation word-end times); `com-hold`/`in-hold` by the
**VAD** (acoustic speech offset). Those are not the same instant — the aligner
tiles words to fill a transcriber segment, the VAD fires on energy decay.

So the 0.6 s prosody window would be anchored to a systematically different
reference point per class. The model could achieve high accuracy by learning
**which detector produced the timestamp**, which is not a property of the
conversation at all. In production there is only one detector, so the learned
discriminant simply evaporates.

**The fix.** Put every decision point on one clock: snap each annotated turn end
onto the nearest VAD speech offset (`snap_to_vad`), and drop points that cannot
be snapped within 0.6 s. Now both classes are anchored identically and the
prosody window means the same thing everywhere.

**The general principle.** *If your classes were produced by different
processes, the model will learn the process.* Ask of every dataset: what
machinery generated each class, and does it differ? This is the same failure as
training a pneumonia detector where positives came from one hospital's scanner.

---

## Trap 3 — Using the pause duration as a feature

**The trap.** `resume_delay` is sitting right there and it is hugely predictive:
holds resume after ~0.45 s, changes hand over after ~1.07 s. Put it in the
feature vector and the metrics jump.

It is **not available at inference.** At decision time the elapsed pause is the
clock you are racing, and the final duration is *caused by* whether you decide to
interrupt. Training on it means the model learns "long pause → turn ended,"
which is exactly the fixed-threshold baseline the project exists to beat, using
information a live system cannot have.

**The fix.** `resume_delay` is the **evaluation variable**, never an input:

- for a **hold** that resumed after `D`: we falsely interrupted iff `τ(p) < D`
- for a **change**: our response latency *is* `τ(p)`

That pair of statements is the operating curve, exactly.

**The refinement (v2).** Elapsed time *is* legitimately informative — waiting
longer really does raise P(complete). The correct way to use it is Skantze's
continuous formulation: emit several training rows per decision point at
elapsed times {0.15, 0.25, 0.4, 0.6, ...} s, each labelled "will speech resume
after this instant?". Then elapsed time is a genuine input because it is known
at that instant. v1 deliberately omits it to keep the comparison with Kelterer
et al. clean.

---

## Trap 4 — Prosody-contaminated "syntactic" labels

**The trap.** AMI transcripts contain punctuation. `.` and `?` look like a free
syntactic-completion label. But transcribers **hear the intonation** while
punctuating — so the label encodes prosody. Train a prosody model to predict it
and you have proved that prosody predicts prosody.

Kelterer et al. guard against this explicitly: syntactic completion is defined
"without considering intonation."

**The fix.** `src/features/syntax.py` sees text only, and never reads the
punctuation as its label. Transcriber punctuation is retained purely as an
*independent cross-check* — agreement between our rule and the punctuation is
informative, but the punctuation is never ground truth.

---

## The stock answer

> "The four things I checked for were speaker leakage across splits, a
> label-generation confound where my two classes were timestamped by different
> detectors, using the pause duration as a feature when it's really the
> evaluation variable, and prosody leaking into the syntactic-completion labels
> via transcriber punctuation. The second one I only caught because the
> source × label crosstab came out block-diagonal."

That last sentence is the important one. It says you looked.
