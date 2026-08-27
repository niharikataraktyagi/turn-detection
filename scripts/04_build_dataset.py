#!/usr/bin/env python3
"""Step 4 — decision points, enumerated entirely from the VAD."""
import os, sys, json, glob, bisect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from data.ami_ipu import load_meeting, label_ipus
from data.ami_meta import parse_meetings
from features.syntax import is_syntactically_complete, label_2x2

ROOT = os.path.join(os.path.dirname(__file__), "..")
ANN = os.path.join(ROOT, "data/raw/ami_annotations")
PAU = os.path.join(ROOT, "data/processed/pauses")
OUT = os.path.join(ROOT, "data/processed/decision_points.parquet")

MAX_CONTEXT_WORDS = 40
MIN_PAUSE = 0.150
MAX_DELAY = 10.0            # beyond this the conversation has simply stopped
BACKCHANNEL_MAX_DUR = 0.80
BACKCHANNEL_MAX_WORDS = 3
WINDOW = 0.600


def words_of(ipus):
    return sorted([(t.start, t.end, t.text, t.trunc)
                   for ipu in ipus for t in ipu.tokens if t.kind == "word"])


def markers_of(ipus):
    return sorted([(t.start, t.kind) for ipu in ipus
                   for t in ipu.tokens if t.kind in ("disf", "vocal")])


def puncs_of(ipus):
    """Transcriber punctuation with timings."""
    return sorted([(t.start, t.text) for ipu in ipus
                   for t in ipu.tokens if t.kind == "punc"])


def main():
    meta = parse_meetings(os.path.join(ANN, "corpusResources/meetings.xml"))
    rows = []
    dropped = {"no_next": 0, "too_long": 0, "no_text": 0, "undecidable": 0}

    for pfile in sorted(glob.glob(os.path.join(PAU, "*.json"))):
        pdata = json.load(open(pfile))
        mtg = pdata["meeting"]
        by_spk = load_meeting(ANN, mtg)
        label_ipus(by_spk)
        ch = {s.agent: s.channel for s in meta.get(mtg, [])}
        gname = {s.agent: s.global_name for s in meta.get(mtg, [])}

        speech = {s: [tuple(x) for x in d.get("speech", [])]
                  for s, d in pdata["speakers"].items()}
        words = {s: words_of(v) for s, v in by_spk.items()}
        marks = {s: markers_of(v) for s, v in by_spk.items()}
        puncs = {s: puncs_of(v) for s, v in by_spk.items()}

        for spk, spans in speech.items():
            spans = sorted(spans)
            if len(spans) < 2 or spk not in ch:
                continue

            # onsets of every OTHER speaker, for the floor-change test
            others = sorted([(s0, s1, o) for o, sp in speech.items() if o != spk
                             for (s0, s1) in sp])
            o_starts = [x[0] for x in others]

            wl = words.get(spk, [])
            w_ends = [w[1] for w in wl]
            ml = marks.get(spk, [])
            pl = puncs.get(spk, [])

            turn_start = spans[0][0]

            for i in range(len(spans) - 1):
                t_dec = spans[i][1]
                own_next = spans[i + 1][0]
                if own_next - t_dec < MIN_PAUSE:
                    continue

                # --- did somebody else take the floor first? -----------------
                k = bisect.bisect_left(o_starts, t_dec)
                other = others[k] if k < len(others) else None
                floor_changes = other is not None and other[0] < own_next
                if floor_changes:
                    # a short utterance the speaker talks straight through is a
                    # BACKCHANNEL ("mm-hm"), not a handover of the floor
                    o_words = sum(1 for w in words.get(other[2], [])
                                  if other[0] <= w[0] <= other[1])
                    if ((other[1] - other[0]) < BACKCHANNEL_MAX_DUR
                            and o_words <= BACKCHANNEL_MAX_WORDS
                            and own_next < other[1] + 2.0):
                        floor_changes = False

                resume_delay = (min(own_next, other[0]) if floor_changes else own_next) - t_dec
                if not np.isfinite(resume_delay):
                    dropped["no_next"] += 1; continue
                if resume_delay > MAX_DELAY:
                    dropped["too_long"] += 1; continue

                # --- transcript available to a live system at this instant ---
                hi = bisect.bisect_right(w_ends, t_dec + 1e-6)
                lo = bisect.bisect_left([w[0] for w in wl], turn_start - 1e-6)
                turn_words = wl[lo:hi]
                if not turn_words:
                    dropped["no_text"] += 1
                    if floor_changes:
                        turn_start = own_next
                    continue

                ctx = " ".join(w[2] for w in turn_words[-MAX_CONTEXT_WORDS:])
                trunc = any(w[3] for w in turn_words[-2:])
                disf = any(k2 == "disf" and t_dec - 0.5 <= t2 <= t_dec for t2, k2 in ml)
                is_q = any("?" in txt for t2, txt in pl if t_dec - 0.4 <= t2 <= t_dec + 0.2)

                complete = is_syntactically_complete(ctx, trunc, disf)
                lab = label_2x2(floor_changes, complete)
                if lab is None:
                    dropped["undecidable"] += 1
                    if floor_changes:
                        turn_start = own_next
                    continue

                # Overlapping speech corrupts the acoustic window. Two scopes:
                #   overlap_at_end  — another speaker active in the final 200 ms
                w0 = t_dec - WINDOW
                overlap = any(s0 < t_dec and s1 > t_dec - 0.20 for s0, s1, _ in others)
                overlap_win = any(s0 < t_dec and s1 > w0 for s0, s1, _ in others)

                rows.append(dict(
                    meeting=mtg, group=mtg[:6], speaker=spk,
                    global_name=gname.get(spk, ""),
                    t_decision=round(t_dec, 3),
                    span_start=round(spans[i][0], 3),   # for window clipping
                    # NOT A FEATURE — the evaluation variable. See docs/06_leakage.md
                    resume_delay=round(float(resume_delay), 3),
                    text=ctx, n_words=len(turn_words),
                    ends_truncated=bool(trunc), ends_disfluency=bool(disf),
                    syn_complete=bool(complete),
                    floor_changes=bool(floor_changes),
                    label=lab,
                    overlap_at_end=bool(overlap),
                    overlap_window=bool(overlap_win),
                    is_question=bool(is_q),   # exclusion criterion only, never a label

                    wav=f"{mtg}.Headset-{ch[spk]}.wav",
                ))

                if floor_changes:
                    turn_start = own_next

        print(f"  {mtg}: running total {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)

    print("\n" + "=" * 70)
    print(f"decision points: {len(df)}  ->  {OUT}")
    print(f"speakers: {df.global_name.nunique()}   groups: {df.group.nunique()}")
    print(f"dropped: {dropped}\n")

    print("THE 2x2")
    print(pd.crosstab(df.floor_changes, df.syn_complete,
                      rownames=["floor_changes"], colnames=["syn_complete"]), "\n")
    print(df.label.value_counts().to_string(), "\n")
    print(f"overlap_at_end  (paper rule, EXCLUDED):  {df.overlap_at_end.mean()*100:.1f}%")
    print(f"overlap_window  (0.6s, reported only):   {df.overlap_window.mean()*100:.1f}%")
    print(pd.crosstab(df.label, df.overlap_at_end).to_string())
    print("questions (excluded from `change`, as in the paper):")
    print(pd.crosstab(df.label, df.is_question).to_string(), "\n")
    print("resume / handover delay (s)   [EVALUATION VARIABLE]")
    print(df.groupby("label").resume_delay
            .describe()[["count", "25%", "50%", "75%", "max"]].round(3).to_string())
    print("\nspan length before the pause (s) — how much audio prosody actually gets")
    sl = (df.t_decision - df.span_start)
    print(sl.describe()[["25%", "50%", "75%"]].round(3).to_string())


if __name__ == "__main__":
    main()
