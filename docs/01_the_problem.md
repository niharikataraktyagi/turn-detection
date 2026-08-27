# 01 — The Problem: Why End-of-Turn Detection Is Hard

> Read this before writing any code. Every design decision later in the repo
> traces back to something in this document. If an interviewer asks "why did you
> build it that way?", the answer is here.

---

## 1. Where this sits in a voice agent

A real-time voice agent is a loop:

```
 mic ──► VAD ──► streaming ASR ──► [WHEN DO I STOP LISTENING?] ──► LLM ──► TTS ──► speaker
```

Everything except that bracketed box is a solved, buyable component. The bracket
is where voice agents feel human or feel broken. It has a name in the literature:
**endpointing**, or **end-of-turn (EOT) / end-of-utterance (EOU) detection**.

## 2. The latency budget

Human conversation is astonishingly fast. Stivers et al. (2009, PNAS,
*"Universals and cultural variation in turn-taking in conversation"*) measured
gaps between turns across 10 unrelated languages and found a **median gap of
roughly 0–200 ms**, remarkably stable across cultures. Gaps beyond ~700 ms are
perceived as meaningful — hesitation, reluctance, disagreement.

That is the bar. Now count what a voice agent must do inside that window:

| Stage                       | Typical latency |
|-----------------------------|-----------------|
| ASR finalisation            |  50–150 ms      |
| **Endpoint decision (wait)**| **200–1500 ms** |
| LLM time-to-first-token     | 200–600 ms      |
| TTS time-to-first-audio     | 80–200 ms       |
| Network + jitter buffer     | 50–150 ms       |

Notice the endpoint wait is the **single largest and most controllable** term.
Everything else is bounded by someone else's model. This is why endpointing is
where a voice-AI company actually competes — and why Plivo's JD lists
"turn detection" as a first-class modelling problem alongside ASR.

## 3. Why the naive solution fails

**The naive solution:** run a Voice Activity Detector (e.g. Silero VAD). When it
reports "no speech" for `τ` continuous milliseconds, declare the turn over.

This is a single fixed threshold `τ`, and it is trapped between two failure modes
that pull in opposite directions:

### Failure mode A — τ too short → *false interruption*
The user pauses mid-thought and the agent barges in.

```
user:  "my order number is ... uh ..."     [420 ms silence]
agent: "I'm sorry, I didn't catch an order number—"   ← WRONG
user:  "...4 4 7 2. Hey, let me finish!"
```

Where mid-utterance pauses cluster in practice:
- **before digit sequences** — order IDs, phone numbers, card numbers, OTPs
- **before proper nouns** — recalling a name, a street, a city
- **disfluency sites** — "um", "uh", "like", "I mean", filled and silent pauses
- **non-native / L2 speakers**, elderly speakers, users reading off a screen
- **code-switching points** — Hinglish speakers pause at the language boundary

This is the expensive error. Being interrupted is socially jarring; the user
loses their place, has to restart, and often talks over the agent — which
corrupts the ASR stream and cascades into more errors.

### Failure mode B — τ too long → *dead air*
The user finished cleanly and the agent sits there.

```
user:  "I'd like to cancel my order."   [1200 ms of nothing]
agent: "Sure, I can help with that."
```

One second of silence feels enormous in speech. Over a 20-turn support call, a
1200 ms threshold spends **24 seconds** of the call doing nothing. Users perceive
the agent as slow, dumb, or disconnected — and start repeating themselves, which
again corrupts the stream.

### The core insight

> A fixed threshold must be simultaneously short enough to feel responsive and
> long enough to survive a thinking pause. **No single value satisfies both.**
> But the *right* threshold is predictable from what was just said and how it was
> said.

Compare two pauses of **identical duration (400 ms)**:

| Audio so far                         | Prosody                     | Truth        |
|--------------------------------------|-----------------------------|--------------|
| "I'd like to cancel my order"        | pitch falls, energy decays  | **complete** |
| "I'd like to cancel my"              | pitch level, energy sustained| **incomplete** |
| "my number is nine eight seven"      | pitch level, no lengthening | **incomplete** |
| "is anyone there?"                   | pitch rises sharply         | **complete** |

The silence is the same. The *content* is not. So: stop thresholding silence,
and start **predicting a probability**:

```
p = P(the speaker has finished their turn | transcript so far, prosody so far)
```

then let the wait time be a *function* of `p`:

```
τ(p) = τ_min + (τ_max − τ_min) · (1 − p)
```

Confident it's complete → wait 200 ms. Confident it's not → wait 1500 ms.
Same model, adaptive behaviour. **That is the whole project.**

---

## 4. What signals actually carry turn-completion information

### 4.1 Lexico-syntactic completion

Conversation Analysis gives us the vocabulary here. Sacks, Schegloff & Jefferson
(1974), *"A Simplest Systematics for the Organization of Turn-Taking for
Conversation"* — the founding paper of the field — observes that turns are built
out of **Turn-Constructional Units (TCUs)**: chunks that are recognisably
complete as syntax, prosody and action. At the end of a TCU there is a
**Transition-Relevance Place (TRP)** — a point where speaker change becomes
legitimate.

Practically, this means a model can learn:
- **Complete**: "I want to cancel my order." / "Can you help me?" / "Yeah."
- **Incomplete**: trailing function words are near-deterministic incompleteness
  cues — prepositions ("to", "for", "with"), conjunctions ("and", "but",
  "because"), determiners ("the", "my", "a"), auxiliaries ("I was", "you can"),
  and dangling subjects ("and then I").

This is a **text** signal, and it's the strongest single one. It is what
LiveKit's open-source turn detector uses (a small transformer over the running
ASR transcript, no audio at all).

### 4.2 Prosody — the "how", not the "what"

Text alone can't disambiguate everything. "Okay" can be a complete turn or the
start of "Okay so what I want is—". Prosody separates them.

The cues, and the physics behind each:

| Cue | What it is | Why it signals completion |
|---|---|---|
| **F0 (pitch) contour** | fundamental frequency of vocal fold vibration over the final ~500 ms | **Falling** F0 = declarative finality (subglottal pressure drops as the speaker stops driving the lungs). **Sharply rising** = question — also complete. **Level / plateau** = continuation. Note it is the *shape*, not "low = done". |
| **Final lengthening** | the last syllable/rhyme is stretched | Speakers systematically lengthen pre-boundary segments. A robust, cross-linguistic boundary marker. |
| **Intensity (energy) decay** | RMS energy slope over the tail | Energy trails off at genuine turn ends; stays flat when the speaker intends to continue. |
| **Speaking rate** | syllables or phones per second, and its derivative | Deceleration precedes completion (part of the same articulatory "winding down"). |
| **Pause duration so far** | how long we've already waited | Still informative — but as *one feature among many*, not as the decision rule. This is exactly what we're beating. |
| **Filled pauses** | "um", "uh", audible breath | Strong *incompleteness* signal — the speaker is explicitly holding the floor. |

Gravano & Hirschberg (2011), *"Turn-taking cues in task-oriented spoken
dialogue"*, is the key empirical paper: they identified a set of independent
cues and showed the probability of a turn ending rises **monotonically with the
number of cues jointly present**. That is direct evidence that fusing cues beats
any single one — the justification for our multi-branch architecture.

### 4.3 Semantics / pragmatics

Beyond syntax: some complete sentences *demand* a response ("Is anyone there?"),
others are complete but the speaker clearly intends to continue ("So here's the
thing."). Dialogue-act structure matters. A pretrained language-model encoder
gives us some of this for free, which is a large part of why we use one instead
of hand-crafted text features.

---

## 5. Where the field is

Worth knowing by name — interviewers at a voice-AI company will recognise these.

- **Skantze (2017)**, *"Towards a General, Continuous Model of Turn-Taking in
  Spoken Dialogue using LSTM Recurrent Neural Networks"* — reframed turn-taking
  from "classify at pauses" to **continuous prediction** of who will be speaking
  shortly. Important shift: decisions shouldn't only happen at silence.
- **Ekstedt & Skantze (2020)**, *"TurnGPT"* — a transformer LM whose next-token
  distribution over a turn-shift token gives a running completion probability.
  Text-only.
- **Ekstedt & Skantze (2022)**, *"Voice Activity Projection (VAP)"* — predicts a
  discretised pattern of **future** voice activity for both speakers from raw
  audio. The current strong research baseline.
- **Industry**: LiveKit's turn detector (text transformer over ASR partials),
  Pipecat **Smart Turn** (audio, Wav2Vec2-style backbone), and the proprietary
  endpointers inside Deepgram / AssemblyAI / Cartesia.

**Our position:** a *fused* text + prosody model that is small enough to run on
CPU inside the pause itself, evaluated not on accuracy but on the
**latency-vs-interruption operating curve**. That combination — fusion, CPU
budget, policy-level evaluation — is defensible and is precisely what the Plivo
JD asks for ("low latency models that run at a fraction of the cost via CPU at
production scale").

---

## 6. The architecture we are building

```
                     ┌──────────────────────────────────────────┐
 mic (16 kHz) ──────►│ Silero VAD  → detects speech/silence edge │
                     └──────────────┬───────────────────────────┘
                                    │ pause starts at t0
        ┌───────────────────────────┼────────────────────────────┐
        │                           │                            │
        ▼                           ▼                            │
┌────────────────┐         ┌──────────────────┐                  │
│ streaming ASR  │         │ prosody extractor │                 │
│ partial text   │         │ last 1.0 s of     │                 │
└───────┬────────┘         │ voiced audio      │                 │
        │                  └─────────┬─────────┘                 │
        ▼                            ▼                           │
┌────────────────┐         ┌──────────────────────┐              │
│ text encoder   │         │ 12-dim feature vector│              │
│ MiniLM/DistilB │         │ f0 slope, f0 range,  │              │
│ → CLS (384-d)  │         │ energy slope, final  │              │
└───────┬────────┘         │ lengthening, rate,   │              │
        │                  │ pause-so-far, ...    │              │
        │                  └─────────┬────────────┘              │
        └──────────┬─────────────────┘                           │
                   ▼                                             │
          ┌──────────────────┐                                   │
          │  fusion MLP head │  → p = P(turn complete)            │
          └────────┬─────────┘                                   │
                   ▼                                             │
        ┌──────────────────────────────┐                         │
        │ adaptive policy               │                        │
        │ τ(p) = τmin + (τmax−τmin)(1−p)│◄───────────────────────┘
        │ commit turn when pause ≥ τ(p) │
        └──────────────────────────────┘
```

Three things make this *production* rather than *notebook*:
1. It runs **incrementally** — re-scored every ~100 ms as the pause grows and the
   ASR partial updates, not once.
2. It runs on **CPU** within a few ms, so the decision fits inside the pause.
3. It is evaluated as a **policy**, not a classifier.

---

## 7. How we will evaluate — and why accuracy is the wrong metric

Accuracy is wrong for three reasons:

1. **Asymmetric costs.** A false interruption is far worse than 200 ms of extra
   latency. Accuracy weights them equally.
2. **Class imbalance.** Most 100 ms frames during speech are "not a turn end". A
   model that always says "not done" scores well and is useless.
3. **It's a policy, not a classification.** The system doesn't emit a label; it
   emits a *decision at a time*. The same classifier with a different threshold
   is a different product.

So we measure:

- **False-interruption rate (FIR)** — fraction of genuine mid-utterance pauses
  where the system wrongly committed the turn.
- **Response latency** — ms from the true end of the user's turn to our commit
  decision. Report **median and p95**; the tail is what users complain about.
- **The operating curve** — sweep the policy parameters, plot
  *median latency (x)* against *false-interruption rate (y)*. Overlay:
  - the fixed-threshold VAD baseline (a curve traced by sweeping τ)
  - text-only model
  - text + prosody fusion
  If our curve sits **below and to the left**, we dominate the baseline: less
  latency *at the same* interruption rate. **This one plot is the project.**
- **Inference cost** — p50 / p95 / p99 CPU latency, single core, before and
  after ONNX int8 quantisation; and model size on disk.

---

## 8. Vocabulary checklist

You should be able to define each of these cold before the interview:

`VAD` · `endpointing` · `EOU / EOT` · `TCU` · `TRP` · `barge-in` ·
`false interruption` · `time-to-first-token` · `F0 / fundamental frequency` ·
`final lengthening` · `declination` · `filled pause` · `partial hypothesis` ·
`incremental / streaming inference` · `voice activity projection` ·
`operating curve` · `int8 quantisation` · `p95 latency`

---

## 9. What we do next (Day 1 remainder)

Build the labelled dataset. The central trick: **we do not need humans to
annotate "complete vs incomplete"** — conversational corpora already tell us
where turns ended. Every recorded turn boundary is a positive; every *prefix* of
that turn cut at a word boundary is a negative. See `docs/02_data.md`.
