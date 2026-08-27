# 02 — Data: Getting Turn-Completion Labels for Free

## 1. The problem that stops most people

To train `P(turn complete | ...)` you need examples labelled *complete* and
*incomplete*. The obvious path is to pay annotators to listen to conversations
and mark turn boundaries. That is slow, expensive, and subjective.

**We never annotate anything.** The insight:

> A recorded conversation *already contains* the labels. If a speaker stopped and
> another speaker took the floor, that was a turn end. If the speaker paused and
> then kept going, that was not. The corpus tells us, for free, at every single
> pause, what the right answer was — because we can see the future.

This is **self-supervision from the temporal structure of dialogue**. It's the
same move as next-token prediction in an LM: the label is already in the data,
you just have to hide the future at inference time and reveal it at training
time.

Say that sentence in the interview. It reframes the project from "I trained a
classifier" to "I designed a supervision signal."

## 2. Two kinds of negative examples — and why we need both

### Type A — real intra-turn pauses (high quality, low volume)

Inside a single speaker's turn, find every silence ≥ 150 ms where the speaker
*resumed afterwards*. These are **genuine hesitations** — exactly the moments a
naive VAD gets wrong:

```
"so the order number is    [520 ms]    four four seven two"
                        ▲
                        └── decision point. Label: INCOMPLETE.
                            A 500 ms fixed threshold would fire here. Bug.
```

These are gold. They carry real hesitation prosody — sustained pitch, held
energy, no final lengthening — because a real human really was thinking. You
cannot synthesise this.

### Type B — word-boundary prefixes (lower quality, high volume)

Take a complete turn and truncate it at each word boundary:

```
"I would like to cancel my order"
 ────────────────────────────────  COMPLETE     (the real turn end)
 "I would like to cancel my"       INCOMPLETE
 "I would like to cancel"          INCOMPLETE
 "I would like to"                 INCOMPLETE
 "I would like"                    INCOMPLETE
```

No human ever paused at those points, so the *prosody* there is not real
turn-internal prosody. But the **lexico-syntactic** signal is perfectly valid —
"I would like to cancel my" is genuinely an incomplete unit — and this gives us
tens of thousands of examples almost free.

**So:** Type B trains the text branch (volume, syntax). Type A trains the
prosody branch and the fusion (realism, acoustics). Evaluation happens on
**Type A only**, because Type A is the real deployment distribution. Training on
B and evaluating on A is deliberate, and being able to explain why is a strong
interview answer.

### Positives

The end of a turn where **the floor actually changed** — the next speech is a
different speaker. Not just "the segment ended", because a speaker can be
segmented mid-turn. Floor change is the ground truth for "a response here would
have been correct."

## 3. Why AMI

We need a corpus with: real conversational audio, transcripts, **word-level
timings**, and speaker turn structure. Options:

| Corpus | Verdict |
|---|---|
| Switchboard / Fisher | The classic turn-taking corpora — but LDC-licensed, paid. Out. |
| IEMOCAP | Request form, acted emotion, small. Out. |
| MELD (Friends) | Free, but TV audio with laugh track and music. Prosody is contaminated. Out. |
| LibriSpeech / VoxPopuli | Read speech. **There are no turns.** Fundamentally wrong. Out. |
| **AMI Meeting Corpus** | Free (CC-BY), spontaneous multi-party speech, **manual word-level timings**, per-speaker headset audio, ~100 h. **In.** |

The word-level timings matter enormously: they mean we do **not** need forced
alignment to know where each word starts and ends. We can cut audio and text at
the same instant, exactly. That removes an entire error-prone stage.

AMI is also standard in the turn-taking literature, so naming it signals you read
the field rather than grabbing the first dataset on HuggingFace.

### The honest limitation

AMI is **meetings**, not customer-support calls. Turn-taking dynamics differ:
meetings have more overlap, more multi-party competition for the floor, and
longer turns than a two-party support call. Two things we do about it:

1. Restrict to clean two-party-like decision points where possible, and report
   results on real intra-turn pauses only.
2. Build a small hand-written **customer-support text eval set** (order numbers,
   addresses, cancellations, Hinglish code-switches) as an out-of-domain probe.

Say the limitation *before* the interviewer finds it. "Here's the domain gap,
here's how I measured it, here's what I'd do with production data" is a much
stronger position than being caught.

## 4. Splitting: the leakage trap

**Split by meeting, never by utterance.**

If you shuffle utterances and split randomly, prefixes of the same turn land in
both train and test — the model has literally seen the answer. Worse, the same
speaker's idiolect and the same room's acoustics appear on both sides, so your
prosody branch memorises speakers instead of learning turn-taking.

So: hold out whole meetings, and prefer holding out whole *speakers* where the
corpus allows. Report the split sizes in the README. Interviewers probe for this
— a candidate who volunteers "I split by meeting to avoid speaker leakage" is
instantly more credible than one who says "I used train_test_split".

## 5. Class balance

Negatives massively outnumber positives (one turn end, many prefixes and
pauses). Options and what we do:

- **Don't** blindly oversample positives — it distorts the probability
  calibration, and we need calibrated probabilities because the policy
  `τ(p)` consumes `p` as a real number, not as an argmax.
- **Do** cap prefixes per turn (e.g. sample at most 4 prefix negatives per turn)
  to keep the ratio bounded, and keep the natural ratio for Type A pauses.
- **Do** check calibration explicitly with a reliability diagram, not just AUC.
  A model with great AUC and terrible calibration produces a terrible policy.

## 6. Output schema

`data/processed/decision_points.parquet`, one row per decision point:

| column | meaning |
|---|---|
| `meeting_id`, `speaker` | for grouping and leakage-safe splits |
| `t_decision` | absolute time of the decision point (s) |
| `prefix_text` | everything the ASR would have transcribed so far |
| `pause_so_far` | silence observed at the decision point (s) |
| `neg_type` | `pause` (Type A) / `prefix` (Type B) / `null` for positives |
| `label` | 1 = turn complete, 0 = not complete |
| `audio_path`, `t_start` | pointer for prosody extraction on Day 2 |

Text branch trains on all rows. Prosody branch and fusion train on rows with
audio. Evaluation uses `neg_type == 'pause'` plus positives.
