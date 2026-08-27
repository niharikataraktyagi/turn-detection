"""Prosodic features at a decision point — the Kelterer et al. (2023) feature
set, plus computable stand-ins for the one feature they hand-annotated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

WINDOW = 0.600
F0_FLOOR, F0_CEIL = 60.0, 500.0
F0_STEP = 0.01           # 10 ms pitch frames
RMS_WIN, RMS_HOP = 0.025, 0.010
OCTAVE_JUMP_ST = 8.0     # a jump larger than this from local median = tracking error


# ---------------------------------------------------------------------------
# whole-channel tracks (computed once per wav, then sliced per decision point)
# ---------------------------------------------------------------------------
def pitch_track(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """F0 in Hz on a uniform 10 ms grid; NaN where unvoiced."""
    import parselmouth

    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
    p = snd.to_pitch_ac(time_step=F0_STEP, pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEIL)
    f0 = p.selected_array["frequency"].astype(np.float64)
    f0[f0 == 0.0] = np.nan          # praat encodes unvoiced as 0
    times = p.xs().astype(np.float64)
    return times, correct_octave_jumps(f0)


def correct_octave_jumps(f0: np.ndarray, k: int = 15) -> np.ndarray:
    """Repair octave errors — the classic F0-tracking failure, where the tracker
    latches onto a harmonic and reports 2f or f/2. Left in, a single halved.
    """
    import pandas as pd

    out = f0.copy()
    med = (pd.Series(out).rolling(2 * k + 1, center=True, min_periods=1)
                          .median().to_numpy())

    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.isfinite(out) & np.isfinite(med) & (med > 0) & (out > 0)
        dev = np.full_like(out, np.nan)
        dev[ok] = np.abs(12.0 * np.log2(out[ok] / med[ok]))
        bad = ok & (dev > OCTAVE_JUMP_ST)
        if not bad.any():
            return out

        cands = np.stack([out, out * 2.0, out / 2.0, out * 4.0, out / 4.0])
        d = np.abs(12.0 * np.log2(cands / med[None, :]))
        d[(cands < F0_FLOOR) | (cands > F0_CEIL)] = np.inf
        d[~np.isfinite(d)] = np.inf
        pick = np.argmin(d, axis=0)
        best = cands[pick, np.arange(len(out))]

    out[bad] = best[bad]
    return out


def rms_track(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Frame-wise RMS energy on a 10 ms hop, 25 ms window."""
    import librosa

    frame = int(RMS_WIN * sr)
    hop = int(RMS_HOP * sr)
    r = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop, center=True)[0]
    t = librosa.frames_to_time(np.arange(len(r)), sr=sr, hop_length=hop)
    return t, r.astype(np.float64)


@dataclass
class SpeakerStats:
    """Per-speaker normalisation constants, from that speaker's own channel."""
    f0_median: float
    rms_mean: float
    rms_std: float

    @classmethod
    def fit(cls, f0: np.ndarray, rms: np.ndarray, rms_times: np.ndarray,
            speech: List[Tuple[float, float]]) -> "SpeakerStats":
        med = float(np.nanmedian(f0)) if np.isfinite(np.nanmedian(f0)) else 150.0

        # RMS statistics over SPEECH frames only. Including silence would drag
        # the mean toward the noise floor and make every speech frame look loud.
        mask = np.zeros(len(rms_times), dtype=bool)
        for s0, s1 in speech:
            mask |= (rms_times >= s0) & (rms_times <= s1)
        vals = rms[mask] if mask.any() else rms
        vals = vals[vals > 0]
        if len(vals) < 10:
            vals = rms[rms > 0]
        return cls(med, float(np.mean(vals)), float(np.std(vals) + 1e-9))

    def to_semitones(self, f0_hz: np.ndarray) -> np.ndarray:
        return 12.0 * np.log2(np.clip(f0_hz, 1e-6, None) / max(self.f0_median, 1e-6))

    def z_rms(self, rms: np.ndarray) -> np.ndarray:
        return (rms - self.rms_mean) / self.rms_std


# ---------------------------------------------------------------------------
# articulation rate
# ---------------------------------------------------------------------------
_VOWEL_GROUP = re.compile(r"[aeiouy]+", re.I)


def count_syllables(word: str) -> int:
    """Cheap English syllable estimate: vowel groups, minus silent final 'e',
    floor of 1. Not perfect, but ArtR only needs to be consistent — a small.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    n = len(_VOWEL_GROUP.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


def articulation_rate(words: List[Tuple[float, float, str]], t0: float, t1: float) -> float:
    """Syllables per second of ARTICULATED time in [t0, t1]."""
    syl, dur = 0.0, 0.0
    for ws, we, txt in words:
        lo, hi = max(ws, t0), min(we, t1)
        if hi <= lo:
            continue
        frac = (hi - lo) / max(we - ws, 1e-6)
        syl += count_syllables(txt) * frac
        dur += hi - lo
    return float(syl / dur) if dur > 0.05 else np.nan


# ---------------------------------------------------------------------------
# contour geometry — our computable stand-in for finIntonation
# ---------------------------------------------------------------------------
def contour_shape(t: np.ndarray, st: np.ndarray) -> Dict[str, float]:
    """Fit a line and a parabola to the semitone contour."""
    ok = np.isfinite(st)
    if ok.sum() < 4:
        return dict(slope=np.nan, curv=np.nan, r2=np.nan)
    x, y = t[ok], st[ok]
    x = x - x.mean()
    b1, b0 = np.polyfit(x, y, 1)
    pred = b1 * x + b0
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    curv = float(np.polyfit(x, y, 2)[0]) if ok.sum() >= 6 else np.nan
    return dict(slope=float(b1), curv=curv, r2=float(1.0 - ss_res / ss_tot))


# ---------------------------------------------------------------------------
# the feature vector
# ---------------------------------------------------------------------------
def extract(t_decision: float,
            f0_times: np.ndarray, f0_hz: np.ndarray,
            rms_times: np.ndarray, rms: np.ndarray,
            stats: SpeakerStats,
            words: List[Tuple[float, float, str]],
            window: float = WINDOW,
            span_start: Optional[float] = None) -> Dict[str, float]:
    """All prosodic features for one decision point."""
    t1 = t_decision
    t0 = t1 - window
    if span_start is not None:
        t0 = max(t0, float(span_start))
    eff = max(t1 - t0, 1e-3)

    fm = (f0_times >= t0) & (f0_times <= t1)
    ft, fs = f0_times[fm], stats.to_semitones(f0_hz[fm])
    voiced = np.isfinite(fs)

    im = (rms_times >= t0) & (rms_times <= t1)
    it, iz = rms_times[im], stats.z_rms(rms[im])

    out: Dict[str, float] = {}

    # --- F0 (5, as in the paper) --------------------------------------------
    if voiced.sum() >= 3:
        v, vt = fs[voiced], ft[voiced]
        out["f0_max"] = float(np.max(v))
        out["f0_min"] = float(np.min(v))
        out["f0_med"] = float(np.median(v))
        out["f0_range"] = out["f0_max"] - out["f0_min"]
        # their definition: slope between the LOCATIONS of min and max,
        # signed by which came first -> encodes rise vs fall
        i_lo, i_hi = int(np.argmin(v)), int(np.argmax(v))
        dt = vt[i_hi] - vt[i_lo]
        out["f0_slope"] = float((v[i_hi] - v[i_lo]) / dt) if abs(dt) > 1e-3 else np.nan
        out["f0_terminal"] = float(np.mean(v[-3:]))
        out["f0_reset"] = out["f0_terminal"] - out["f0_med"]
        out["voiced_frac"] = float(voiced.mean())
    else:
        for k in ("f0_max", "f0_min", "f0_med", "f0_range", "f0_slope",
                  "f0_terminal", "f0_reset"):
            out[k] = np.nan
        out["voiced_frac"] = float(voiced.mean()) if len(voiced) else 0.0

    # --- intensity (4, as in the paper) -------------------------------------
    if len(iz) >= 3:
        out["i_max"] = float(np.max(iz))
        out["i_med"] = float(np.median(iz))
        out["i_std"] = float(np.std(iz))
        # position of the intensity peak, normalised to [0,1] within the window:
        # late peak = still driving; early peak = trailing off
        out["i_tmax"] = float((it[int(np.argmax(iz))] - t0) / eff)
        out["i_slope"] = float(np.polyfit(it - it.mean(), iz, 1)[0]) if len(iz) >= 4 else np.nan
    else:
        for k in ("i_max", "i_med", "i_std", "i_tmax", "i_slope"):
            out[k] = np.nan

    # --- duration (1) --------------------------------------------------------
    out["art_rate"] = articulation_rate(words, t0, t1)
    out["win_len"] = float(eff)

    # --- contour geometry (our finIntonation stand-in) -----------------------
    for span in (0.300, 0.600):
        m = (ft >= max(t1 - span, t0)) & (ft <= t1)
        sh = contour_shape(ft[m], fs[m])
        tag = int(span * 1000)
        out[f"contour_slope_{tag}"] = sh["slope"]
        out[f"contour_curv_{tag}"] = sh["curv"]
        out[f"contour_r2_{tag}"] = sh["r2"]

    return out


FEATURE_NAMES = [
    # the paper's 10 continuous features
    "f0_max", "f0_min", "f0_med", "f0_range", "f0_slope",
    "i_max", "i_med", "i_std", "i_tmax",
    "art_rate",
    # ours, standing in for the hand-annotated finIntonation
    "f0_terminal", "f0_reset", "i_slope", "voiced_frac",
    "contour_slope_300", "contour_curv_300", "contour_r2_300",
    "contour_slope_600", "contour_curv_600", "contour_r2_600",
]

PAPER_FEATURES = ["f0_max", "f0_min", "f0_med", "f0_range", "f0_slope",
                  "i_max", "i_med", "i_std", "i_tmax", "art_rate"]
