# 05 — Everything You Need to Know About AMI (and the vocabulary around it)

## 1. What AMI is

**AMI = Augmented Multi-party Interaction.** An EU-funded research project
(2004–2007) that produced the **AMI Meeting Corpus**: ~100 hours of recorded
meetings, released free under a Creative Commons licence. It is one of the most
heavily used speech corpora in existence — the standard benchmark for meeting
ASR, speaker diarisation, meeting summarisation, and turn-taking research.

If you say "I used AMI" in a speech-AI interview, everyone in the room knows the
corpus. That's part of why we chose it.

## 2. What's actually in it

**Two thirds is a "design scenario."** Four people are assigned roles in a
fictional company — project manager, marketing expert, industrial designer,
user-interface designer — and told to design a new TV remote control. They meet
**four times** as the project progresses. The remaining third is naturally
occurring meetings with no script.

The conversation is genuinely spontaneous: people interrupt, hesitate, laugh,
trail off, talk over each other. Nobody is reading. That is exactly what we need
— you cannot study turn-taking in read speech, because read speech has no turns.

## 3. Reading a meeting ID — this matters for our data splits

```
ES2002a
││ │  │
││ │  └── session: a, b, c, d  = the SAME four people, meetings 1–4
││ └───── group number
│└─────── recording site:  ES = Edinburgh (scenario)
│                          IS = Idiap
│                          TS = TNO
│                          EN = Edinburgh, non-scenario (natural meetings)
```

**So `ES2002a`, `ES2002b`, `ES2002c`, `ES2002d` are the same four humans.**

This is the leakage trap from `docs/02_data.md` made concrete. If you put
`ES2002a` in train and `ES2002b` in test, your prosody model can memorise those
four voices and score well without learning anything about turn-taking. We
therefore split by **group** (the `ES2002` prefix), never by meeting or
utterance. The 10 meetings I picked for download span **6 different groups** for
exactly this reason.

## 4. The microphones — and why we want the headsets

Every meeting was recorded with several microphone setups at once:

| Setup | What it is | Use for us |
|---|---|---|
| **IHM** — Individual Headset Microphone | one close-talking headset per participant, so each speaker has their **own clean channel** | **This is what we download.** |
| SDM — Single Distant Microphone | one mic in the middle of the table | no |
| MDM — Multiple Distant Microphones | a microphone array in the room | no |

Also video, whiteboard and pen capture — irrelevant here.

**Why IHM is non-negotiable for this project:** prosody means measuring *one
speaker's* pitch and intensity. On a distant mic all four voices are mixed, so
an intensity measurement is meaningless — you can't tell whose energy you're
measuring, and F0 tracking collapses when two people overlap.

It also mirrors production. In a phone call or a WebRTC session you have the
caller's own channel, close-talking, roughly one speaker. IHM is the closest
free analogue. Files look like `ES2002a.Headset-0.wav` … `Headset-3.wav`,
one per participant, 16 kHz mono.

## 5. The annotation layers

The 206 MB zip you unpacked contains many parallel annotation layers in **NXT
XML** (a format where each layer lives in its own file and points into the
others by ID). What's there:

| Layer | Contents | Do we use it? |
|---|---|---|
| **`words/`** | orthographic transcription, **word-level start/end times**, punctuation, truncation and disfluency markers | **yes — core** |
| **`segments/`** | transcriber-defined utterance chunks | yes, for the multi-speaker timeline |
| `dialogueActs/` | dialogue-act tags (inform, suggest, assess, elicit…) | maybe later |
| `disfluency/`, `namedEntities/`, `topics/` | as named | no |
| `extractive/`, `abstractive/` | summarisation annotations | no |
| `handGesture/`, `headGesture/`, `focus/`, `movement/` | video annotations | no |

Inside a words file:

```xml
<w nite:id="ES2002a.A.words0" starttime="77.44" endtime="77.74">Hi</w>
<w nite:id="ES2002a.A.words1" starttime="77.74" endtime="77.74" punc="true">,</w>
<vocalsound nite:id="..." starttime="237.36" endtime="239.25" type="laugh"/>
<disfmarker nite:id="..." starttime="299.16" endtime="299.46"/>
```

- `punc="true"` → punctuation token, zero duration
- `trunc="true"` → **truncated word** — the speaker cut themselves off mid-word
- `<vocalsound type="laugh|other">` → laughter, breath, cough
- `<disfmarker>` → a disfluency / repair site
- `<gap>` → untranscribable audio

The last three are strong signals in their own right: a truncated word or a
disfluency marker right before a pause is near-proof that the speaker is *not*
finished.

## 6. The problem we discovered — and why we now need the audio

The manual word timings look like they should give us pauses for free. They
don't.

Measured across 24 speaker-files: **~92 % of consecutive word pairs are exactly
contiguous** (`word[i].end == word[i+1].start`), and only 727 gaps fell in the
0.15–2.0 s range — mostly at transcriber segment boundaries, not real
hesitations.

**Why:** the timings come from forced alignment *within* each transcriber-defined
segment. The aligner tiles the words across the whole segment, so a 500 ms
thinking pause in the middle gets absorbed into the neighbouring words'
durations. It never appears as a gap.

The consequence is exactly the thing we care about: **`in-hold` events — pauses
inside a turn where the speaker keeps going — are invisible in the annotations.**

So we get pauses the way a production system does: **run a VAD on the audio.**

> Worth telling this story in the interview. "I found the corpus timings couldn't
> express intra-turn pauses, verified it by measuring the gap distribution, and
> switched to VAD-derived boundaries on the headset audio" is the kind of thing
> that separates someone who *used* a dataset from someone who *interrogated*
> one.

## 7. What a VAD is

**Voice Activity Detection** — for each short frame of audio (typically
10–30 ms), decide: speech, or not speech?

- **Classic approach**: short-time energy plus zero-crossing rate, with a
  threshold. Cheap, and fragile — it fires on breathing, lip smacks, keyboard
  noise, and the faint bleed of other speakers into the headset.
- **Modern approach**: a small neural net. **Silero VAD** is the standard —
  about 1.8 MB, runs a 30 ms frame in well under a millisecond on one CPU core,
  and is robust to noise. It's what most production voice agents use.

For us the VAD does two jobs:
1. **Offline, on AMI** — find every silence > 150 ms inside each speaker's
   channel, which gives us the real IPU boundaries the annotations couldn't.
2. **Online, in the demo** — the same job live, feeding the endpointer.

Using the same component for both keeps train and inference distributions
matched, which matters: if you defined pauses one way in training and another
way at inference, your model's `pause_so_far` feature means something different
in each. That's a classic silent bug.

## 8. Vocabulary you should be able to define cold

| Term | Definition |
|---|---|
| **Turn** | One participant's continuous stretch of holding the floor. |
| **Floor** | The right to be speaking. "Taking the floor" = becoming the speaker. |
| **IPU** (inter-pausal unit) | A run of speech by one speaker bounded by pauses > 150 ms. **Purely acoustic** — no syntax involved. |
| **TCU** (turn-constructional unit) | A grammatically + prosodically + pragmatically complete chunk out of which turns are built. Sacks, Schegloff & Jefferson (1974). |
| **TRP** (transition-relevance place) | The end of a TCU, where speaker change becomes legitimate. Only ~half of TRPs actually produce a turn change (Ford & Thompson). |
| **Backchannel / HRT** | "mm-hm", "yeah", "right" — a listener signalling attention **without taking the floor**. Must not be counted as a turn change. |
| **Barge-in** | The user speaking while the agent is speaking. |
| **Endpointing** | Deciding the user has finished — our problem. |
| **IHM / SDM / MDM** | Individual headset / single distant / multiple distant microphone. |
| **Diarisation** | "Who spoke when" — segmenting audio by speaker identity. |
| **F0** | Fundamental frequency — the physical rate of vocal-fold vibration; perceived as pitch. |
| **Forced alignment** | Given audio + a known transcript, compute where each word/phone starts and ends. |

## 9. AMI's limitations — say these before an interviewer finds them

1. **Meetings, not calls.** 4-party with competition for the floor; support calls
   are 2-party. More overlap, longer turns, different dynamics.
2. **Scenario is semi-artificial** for the ES/IS/TS portion — real speech, but a
   role-play task.
3. **European English accents**, no Hinglish, no telephone-band audio.
4. **Recorded 2005**, on headsets — not 8 kHz telephony codec audio.

The mitigation is honesty plus measurement: build a small out-of-domain probe set
in the target style, report the drop, and describe what you'd do with real
production data (fine-tune on it; the architecture doesn't change).
