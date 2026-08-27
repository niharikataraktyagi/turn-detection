"""VAD-derived pause structure for AMI headset channels."""
from __future__ import annotations

import os
import json
from typing import List, Tuple, Dict

import numpy as np

SR = 16000
PAUSE_MIN = 0.150          # matches the IPU definition (Kelterer et al.)
BLEED_TOLERANCE = 0.20     # allow VAD speech to extend this far past an annotated span

Span = Tuple[float, float]


# ----------------------------------------------------------------------------
# 1. run the VAD
# ----------------------------------------------------------------------------
def load_vad():
    """Silero VAD via the pip package (not torch.hub) so no GitHub fetch is
    needed. ~1.8 MB model, CPU-only, far below real time.
    """
    from silero_vad import load_silero_vad
    return load_silero_vad(onnx=False)


def read_wav(path: str) -> "np.ndarray":
    """Load a mono 16 kHz WAV as float32 in [-1, 1]."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}. Resample before use.")
    return audio


def vad_speech_spans(wav_path: str, model, threshold: float = 0.5) -> List[Span]:
    """Speech regions for one channel, in seconds."""
    import torch
    from silero_vad import get_speech_timestamps

    wav = torch.from_numpy(read_wav(wav_path))
    ts = get_speech_timestamps(
        wav, model,
        sampling_rate=SR,
        threshold=threshold,
        min_speech_duration_ms=100,
        min_silence_duration_ms=int(PAUSE_MIN * 1000),
        speech_pad_ms=0,          # no padding: we need honest boundaries
    )
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


# ----------------------------------------------------------------------------
# 2. suppress bleed using the annotations
# ----------------------------------------------------------------------------
def merge_spans(spans: List[Span], gap: float = 0.0) -> List[Span]:
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for s, e in spans[1:]:
        if s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


def intersect(a: List[Span], b: List[Span]) -> List[Span]:
    """Interval intersection of two sorted, non-overlapping span lists."""
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def annotated_spans(ipus) -> List[Span]:
    """Coarse 'this speaker was talking here' regions, from the annotation."""
    return merge_spans([(i.start, i.end) for i in ipus], gap=0.0)


def refine(vad: List[Span], annotated: List[Span]) -> List[Span]:
    """Keep only VAD speech that the annotation agrees was this speaker."""
    padded = [(max(0.0, s - BLEED_TOLERANCE), e + BLEED_TOLERANCE) for s, e in annotated]
    return intersect(merge_spans(vad), merge_spans(padded))


# ----------------------------------------------------------------------------
# 3. find the pauses we actually care about
# ----------------------------------------------------------------------------
def pauses_within(speech: List[Span], annotated: List[Span]) -> List[dict]:
    """Every silence > PAUSE_MIN that falls INSIDE an annotated span, i.e. the
    speaker paused and then kept talking. These are the in-hold / com-hold.
    """
    out = []
    for a_start, a_end in annotated:
        inside = [s for s in speech if s[1] > a_start and s[0] < a_end]
        for (s1, e1), (s2, _e2) in zip(inside, inside[1:]):
            dur = s2 - e1
            if dur > PAUSE_MIN:
                out.append({
                    "t_decision": e1,        # moment speech stopped
                    "pause_dur": dur,        # how long until it resumed
                    "resumed": True,         # same speaker continued -> a HOLD
                    "span_start": a_start,
                    "span_end": a_end,
                })
    return out


def process_channel(wav_path: str, ipus, model) -> dict:
    ann = annotated_spans(ipus)
    raw = vad_speech_spans(wav_path, model)
    speech = refine(raw, ann)
    return {
        "wav": os.path.basename(wav_path),
        "n_vad_spans_raw": len(raw),
        "n_vad_spans_kept": len(speech),
        "speech": speech,
        "annotated": ann,
        "internal_pauses": pauses_within(speech, ann),
    }


def save(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)
