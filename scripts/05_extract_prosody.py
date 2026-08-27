#!/usr/bin/env python3
"""Step 5 — prosodic features at every decision point."""
import os, sys, glob, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from data.ami_ipu import load_meeting
from data.vad_pauses import read_wav, SR
from features.prosody import (pitch_track, rms_track, SpeakerStats, extract,
                              FEATURE_NAMES)

ROOT = os.path.join(os.path.dirname(__file__), "..")
ANN = os.path.join(ROOT, "data/raw/ami_annotations")
AUD = os.path.join(ROOT, "data/raw/ami_audio")
PAU = os.path.join(ROOT, "data/processed/pauses")
DP = os.path.join(ROOT, "data/processed/decision_points.parquet")
OUT = os.path.join(ROOT, "data/processed/prosody.parquet")


def word_spans(ipus):
    """(start, end, text) for every lexical token — used for articulation rate."""
    return [(t.start, t.end, t.text)
            for ipu in ipus for t in ipu.tokens if t.kind == "word"]


def main():
    dp = pd.read_parquet(DP)
    dp = dp[dp.wav != ""].copy()
    dp["row_id"] = np.arange(len(dp))
    print(f"{len(dp)} decision points across {dp.wav.nunique()} channels\n")

    speech_by = {}
    for f in glob.glob(os.path.join(PAU, "*.json")):
        d = json.load(open(f))
        for spk, s in d["speakers"].items():
            speech_by[(d["meeting"], spk)] = [tuple(x) for x in s.get("speech", [])]

    words_cache = {}
    rows = []

    for wav, grp in dp.groupby("wav"):
        t0 = time.time()
        mtg = wav.split(".")[0]
        spk = grp.speaker.iloc[0]
        path = os.path.join(AUD, wav)
        if not os.path.exists(path):
            print(f"  {wav}: missing, skipping"); continue

        audio = read_wav(path)
        ft, fhz = pitch_track(audio, SR)
        rt, rms = rms_track(audio, SR)
        stats = SpeakerStats.fit(fhz, rms, rt, speech_by.get((mtg, spk), []))

        if mtg not in words_cache:
            words_cache[mtg] = {s: word_spans(v) for s, v in load_meeting(ANN, mtg).items()}
        words = words_cache[mtg].get(spk, [])

        for _, r in grp.iterrows():
            feats = extract(float(r.t_decision), ft, fhz, rt, rms, stats, words,
                            span_start=float(r.span_start))
            feats["row_id"] = int(r.row_id)
            rows.append(feats)

        print(f"  {wav:26s} n={len(grp):4d}  f0_med={stats.f0_median:6.1f}Hz  "
              f"{time.time()-t0:5.1f}s", flush=True)

    pf = pd.DataFrame(rows)
    out = dp.merge(pf, on="row_id", how="left")
    out.to_parquet(OUT, index=False)

    print("\n" + "=" * 70)
    print(f"wrote {OUT}   rows={len(out)}")
    miss = out[FEATURE_NAMES].isna().mean().sort_values(ascending=False)
    print("\nmissing-rate per feature (high = window too short / unvoiced):")
    print((miss * 100).round(1).to_string())
    print("\nfeature means by label:")
    print(out.groupby("label")[["art_rate", "i_med", "i_slope", "f0_range", "f0_slope",
                                "f0_reset", "contour_curv_600", "voiced_frac", "win_len"]]
             .mean().round(3).to_string())


if __name__ == "__main__":
    main()
