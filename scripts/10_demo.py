#!/usr/bin/env python3
"""Step 10 — live microphone demo."""
import os, sys, time, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from features.prosody import (pitch_track, rms_track, SpeakerStats, extract,
                              FEATURE_NAMES, WINDOW)
from eval.policy import wait_time

ROOT = os.path.join(os.path.dirname(__file__), "..")
ART = os.path.join(ROOT, "artifacts")
SR = 16000
FRAME = 512               # 32 ms at 16 kHz — Silero's native frame
TAU_MIN, TAU_MAX = 0.20, 1.40
CAL_SECONDS = 6.0         # speech needed before speaker normalisation is trusted


class Endpointer:
    """Wraps the trained model with online speaker normalisation."""

    def __init__(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        p8 = os.path.join(ART, "endpointer_int8.onnx")
        path = p8 if os.path.exists(p8) else os.path.join(ART, "endpointer_fp32.onnx")
        so = ort.SessionOptions(); so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        ck = torch.load(os.path.join(ART, "fusion_fold0.pt"), map_location="cpu",
                        weights_only=False)
        self.mu, self.sd, self.cols, self.T = ck["mu"], ck["sd"], ck["cols"], ck["T"]
        print(f"loaded {os.path.basename(path)}  ({len(self.cols)} prosody features)")

        # online speaker statistics — a population prior, replaced once we have
        # enough of THIS speaker. Exactly the offline normalisation, accumulated.
        self.f0_hist, self.rms_hist = [], []
        self.stats = SpeakerStats(f0_median=140.0, rms_mean=0.05, rms_std=0.03)
        self.calibrated = False

    def update_speaker(self, f0, rms):
        self.f0_hist.append(f0[np.isfinite(f0)])
        self.rms_hist.append(rms[rms > 0])
        n = sum(len(x) for x in self.rms_hist)
        if n * 0.010 > CAL_SECONDS and not self.calibrated:
            f = np.concatenate(self.f0_hist); r = np.concatenate(self.rms_hist)
            self.stats = SpeakerStats(float(np.nanmedian(f)), float(r.mean()),
                                      float(r.std() + 1e-9))
            self.calibrated = True
            print(f"  [speaker calibrated: median F0 {self.stats.f0_median:.0f} Hz]")

    def predict(self, tail: np.ndarray, text: str):
        ft, fhz = pitch_track(tail, SR)
        rt, rms = rms_track(tail, SR)
        self.update_speaker(fhz, rms)
        t_end = len(tail) / SR
        feats = extract(t_end, ft, fhz, rt, rms, self.stats, [],
                        span_start=max(0.0, t_end - WINDOW))
        # Build the feature vector EXACTLY as training did (scripts/07_train.py):
        #   1. raw value, NaN where unmeasurable
        raw = {c: feats.get(c, np.nan) for c in self.cols if not c.endswith("_missing")}
        x = np.empty(len(self.cols), dtype=np.float64)
        for i, c in enumerate(self.cols):
            if c.endswith("_missing"):
                base = feats.get(c[:-len("_missing")], np.nan)
                x[i] = 0.0 if np.isfinite(base) else 1.0
            else:
                x[i] = raw[c]
        x = (x - self.mu) / self.sd
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        e = self.tok(text or "", truncation=True, max_length=48,
                     padding="max_length", return_tensors="np")
        t0 = time.perf_counter()
        logit = self.sess.run(None, {"input_ids": e["input_ids"].astype(np.int64),
                                     "attention_mask": e["attention_mask"].astype(np.int64),
                                     "prosody": x[None, :].astype(np.float32)})[0][0]
        ms = (time.perf_counter() - t0) * 1000
        p = 1 / (1 + np.exp(-float(logit) / max(self.T, 1e-2)))
        return p, ms


def run(source, ep, text_cb=None):
    from silero_vad import load_silero_vad
    vad = load_silero_vad(onnx=False)

    buf = collections.deque(maxlen=int(SR * 3.0))   # 3 s of speech context
    speaking, pause_start, decided = False, None, False
    tau, p = None, None

    for frame in source:
        buf.extend(frame)
        prob = float(vad(torch.from_numpy(frame).float(), SR))
        now = time.time()

        if prob > 0.5:
            if not speaking:
                speaking = True
                print("\n  ▶ speech")
            pause_start, decided = None, False
        else:
            if speaking and pause_start is None:
                pause_start = now
                tail = np.array(buf, dtype=np.float32)
                text = text_cb() if text_cb else ""
                p, infer_ms = ep.predict(tail, text)
                tau = float(wait_time(np.array([p]), TAU_MIN, TAU_MAX)[0])
                print(f"  ⏸ pause | p(turn over)={p:.3f} -> wait {tau*1000:.0f} ms "
                      f"| inference {infer_ms:.1f} ms" + (f' | "{text}"' if text else ""))
            elif pause_start is not None and not decided:
                if now - pause_start >= tau:
                    decided, speaking = True, False
                    print(f"  ✔ COMMIT TURN after {(now-pause_start)*1000:.0f} ms")


def mic_source():
    import sounddevice as sd
    q = collections.deque()
    with sd.InputStream(samplerate=SR, channels=1, blocksize=FRAME, dtype="float32",
                        callback=lambda ind, f, t, s: q.append(ind[:, 0].copy())):
        print("listening — Ctrl-C to stop\n")
        while True:
            if q:
                yield q.popleft()
            else:
                time.sleep(0.005)


def wav_source(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"need {SR} Hz"
    if a.ndim > 1:
        a = a.mean(1)
    for i in range(0, len(a) - FRAME, FRAME):
        yield a[i:i + FRAME]
        time.sleep(FRAME / SR)   # real time, so the clock means something


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--text", default="", help="stand-in for an ASR partial")
    a = ap.parse_args()
    ep = Endpointer()
    src = wav_source(a.wav) if a.wav else mic_source()
    try:
        run(src, ep, text_cb=(lambda: a.text) if a.text else None)
    except KeyboardInterrupt:
        print("\nstopped")
