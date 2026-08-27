#!/usr/bin/env python3
"""Step 3 — recover the real pause structure from the AMI headset audio."""
import os, sys, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data.ami_ipu import load_meeting, label_ipus, list_meetings
from data.ami_meta import parse_meetings
from data import vad_pauses as V

ROOT = os.path.join(os.path.dirname(__file__), "..")
ANN = os.path.join(ROOT, "data/raw/ami_annotations")
AUD = os.path.join(ROOT, "data/raw/ami_audio")
OUT = os.path.join(ROOT, "data/processed/pauses")


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = parse_meetings(os.path.join(ANN, "corpusResources/meetings.xml"))

    meetings = sorted({f.split(".")[0] for f in os.listdir(AUD) if f.endswith(".wav")})
    print(f"{len(meetings)} meetings with audio: {', '.join(meetings)}\n")

    print("loading Silero VAD ...", flush=True)
    model = V.load_vad()

    grand = {"channels": 0, "internal_pauses": 0, "ipus": 0}
    for mtg in meetings:
        t0 = time.time()
        by_spk = load_meeting(ANN, mtg)
        label_ipus(by_spk)
        ch_of = {s.agent: s.channel for s in meta.get(mtg, [])}
        gname = {s.agent: s.global_name for s in meta.get(mtg, [])}

        result = {"meeting": mtg, "speakers": {}}
        for spk, ipus in sorted(by_spk.items()):
            if spk not in ch_of:
                print(f"  [{mtg}] speaker {spk}: no channel in metadata, skipping")
                continue
            wav = os.path.join(AUD, f"{mtg}.Headset-{ch_of[spk]}.wav")
            if not os.path.exists(wav):
                print(f"  [{mtg}] {spk}: missing {os.path.basename(wav)}, skipping")
                continue

            info = V.process_channel(wav, ipus, model)
            info["global_name"] = gname.get(spk, "")
            info["n_ipus_annotated"] = len(ipus)
            result["speakers"][spk] = info

            grand["channels"] += 1
            grand["internal_pauses"] += len(info["internal_pauses"])
            grand["ipus"] += len(ipus)
            print(f"  [{mtg}] {spk} ch{ch_of[spk]} {gname.get(spk,''):10s} "
                  f"vad {info['n_vad_spans_raw']:5d} -> kept {info['n_vad_spans_kept']:5d} | "
                  f"internal pauses {len(info['internal_pauses']):5d}", flush=True)

        with open(os.path.join(OUT, f"{mtg}.json"), "w") as fh:
            json.dump(result, fh)
        print(f"  {mtg} done in {time.time()-t0:.0f}s\n", flush=True)

    print("=" * 60)
    print(f"channels processed : {grand['channels']}")
    print(f"annotated IPUs     : {grand['ipus']}")
    print(f"INTERNAL PAUSES    : {grand['internal_pauses']}   <-- the class we could not get from annotations")
    print(f"written to         : data/processed/pauses/")


if __name__ == "__main__":
    main()
