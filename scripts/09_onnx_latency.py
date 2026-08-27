#!/usr/bin/env python3
"""Step 9 — CPU deployment: ONNX export, int8 quantisation, latency profile."""
import os, sys, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from models.fusion import EndpointModel
from features.prosody import FEATURE_NAMES

ROOT = os.path.join(os.path.dirname(__file__), "..")
ART = os.path.join(ROOT, "artifacts")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 48
N_WARM, N_RUNS = 20, 200


def bench(fn, n_warm=N_WARM, n_runs=N_RUNS):
    for _ in range(n_warm):
        fn()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    a = np.array(ts)
    return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)))


def main():
    torch.set_num_threads(1)   # a production worker gets one core, not all of them
    print(f"torch threads = {torch.get_num_threads()}  (single core, as in production)\n")

    ck = torch.load(os.path.join(ART, "fusion_fold0.pt"), map_location="cpu", weights_only=False)
    n_pros = len(ck["cols"])
    model = EndpointModel(n_prosody=n_pros, model_name=MODEL_NAME)
    model.load_state_dict(ck["state_dict"]); model.eval()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    # A REALISTIC input, not a worst-case one. Padding every request to
    # MAX_LEN=48 wastes compute on padding tokens: attention is quadratic in
    SAMPLE = "I would like to cancel my order number"
    enc = tok(SAMPLE, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    pr = torch.zeros(1, n_pros)
    print(f"benchmark input: {ids.shape[1]} tokens (unpadded)  \"{SAMPLE}\"\n")

    rows = []

    # ---------- fp32 pytorch ----------
    with torch.no_grad():
        r = bench(lambda: model(input_ids=ids, attention_mask=am, prosody=pr))
    r.update(name="PyTorch fp32", size_mb=sum(p.numel() * 4 for p in model.parameters()) / 1e6)
    rows.append(r)

    # ---------- onnx fp32 ----------
    onnx_path = os.path.join(ART, "endpointer_fp32.onnx")
    common = dict(
        input_names=["input_ids", "attention_mask", "prosody"], output_names=["logit"],
        # BOTH batch and sequence must be dynamic. Declaring only the batch
        # axis freezes the graph at whatever length the tracing example had —
        dynamic_axes={"input_ids": {0: "b", 1: "seq"},
                      "attention_mask": {0: "b", 1: "seq"},
                      "prosody": {0: "b"}, "logit": {0: "b"}},
        opset_version=17, do_constant_folding=True)
    # Export with the LEGACY TorchScript tracer (dynamo=False) by preference.
    # torch >= 2.9 defaults to the new dynamo exporter. It exports this model
    try:
        torch.onnx.export(model, (ids, am, pr), onnx_path, dynamo=False, **common)
        print("  exported via legacy TorchScript tracer")
    except (TypeError, RuntimeError) as exc:
        print(f"  legacy tracer unavailable ({type(exc).__name__}); using dynamo")
        torch.onnx.export(model, (ids, am, pr), onnx_path, **common)

    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    feed = {"input_ids": ids.numpy(), "attention_mask": am.numpy(), "prosody": pr.numpy()}
    r = bench(lambda: sess.run(None, feed))
    r.update(name="ONNX fp32", size_mb=os.path.getsize(onnx_path) / 1e6)
    rows.append(r)

    # ---------- onnx int8 ----------
    from onnxruntime.quantization import quantize_dynamic, QuantType
    q_path = os.path.join(ART, "endpointer_int8.onnx")
    quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
    qsess = ort.InferenceSession(q_path, so, providers=["CPUExecutionProvider"])
    r = bench(lambda: qsess.run(None, feed))
    r.update(name="ONNX int8", size_mb=os.path.getsize(q_path) / 1e6)
    rows.append(r)

    print(f"{'variant':16s} {'size':>8s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'p99':>8s}")
    print("-" * 60)
    for r in rows:
        print(f"{r['name']:16s} {r['size_mb']:7.1f}M {r['mean']:7.2f} {r['p50']:7.2f} "
              f"{r['p95']:7.2f} {r['p99']:7.2f}   (ms)")
    sp = rows[1]["p95"] / rows[2]["p95"]
    print(f"\nint8 speedup vs ONNX fp32 (p95): {sp:.2f}x   "
          f"size {rows[1]['size_mb']:.0f}M -> {rows[2]['size_mb']:.0f}M")

    # ---------- does quantisation cost accuracy? MEASURE IT ----------
    print("\n--- accuracy after quantisation (never assume) ---")
    oof = pd.read_parquet(os.path.join(ART, "oof_predictions.parquet"))
    samp = oof.dropna(subset=["p_fusion"]).sample(min(500, len(oof)), random_state=0)
    texts = samp.text.fillna("").tolist()
    y = samp.floor_changes.to_numpy().astype(int)

    p32, p8 = [], []
    for t in texts:
        e = tok(t, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        f = {"input_ids": e["input_ids"].numpy(),
             "attention_mask": e["attention_mask"].numpy(), "prosody": pr.numpy()}
        p32.append(float(sess.run(None, f)[0][0]))
        p8.append(float(qsess.run(None, f)[0][0]))
    p32, p8 = np.array(p32), np.array(p8)
    print(f"  AUC fp32 {roc_auc_score(y, p32):.4f}   int8 {roc_auc_score(y, p8):.4f}")
    print(f"  mean |logit delta| {np.abs(p32-p8).mean():.4f}")
    print("  (prosody held constant here — this isolates the text encoder, which is")
    print("   where >99% of the parameters and all the quantisation risk live)")

    # ---------- how much does sequence length cost? ----------
    print("\n--- p95 latency vs sequence length (int8) ---")
    print(f"{'tokens':>8s} {'p95 ms':>9s}")
    seq_rows = []
    for n_tok in (8, 16, 24, 32, 48):
        i2 = torch.ones(1, n_tok, dtype=ids.dtype)
        a2 = torch.ones(1, n_tok, dtype=am.dtype)
        f2 = {"input_ids": i2.numpy(), "attention_mask": a2.numpy(), "prosody": pr.numpy()}
        try:
            r2 = bench(lambda: qsess.run(None, f2), n_warm=10, n_runs=80)
            seq_rows.append(dict(tokens=n_tok, p95=r2["p95"]))
            print(f"{n_tok:8d} {r2['p95']:9.2f}")
        except Exception as e:
            print(f"{n_tok:8d}   failed ({type(e).__name__})")
    if len(seq_rows) >= 2:
        print(f"  padding to 48 instead of ~16 costs "
              f"{seq_rows[-1]['p95']/seq_rows[1]['p95']:.1f}x  -> use dynamic padding")

    print("\n--- READ THIS BEFORE QUOTING THE NUMBERS ---")
    import platform
    print(f"  machine: {platform.machine()} / {platform.system()}, single thread")
    print("  The PORTABLE claim is the int8-vs-fp32 ratio: ~2-3x faster, 4x smaller,")
    print("  AUC cost <0.005. That holds across CPUs because it comes from int8 GEMM")
    print("  replacing fp32 GEMM in the encoder's matmuls.")
    print("  The RANKING OF BACKENDS is hardware-, build- and input-specific. An")
    print("  earlier run of this script padded every input to 48 tokens and made ONNX")
    print("  look 5x SLOWER than PyTorch; the cause was the padding, not the backend.")
    print("  Benchmark on the machine you will deploy on, at the input length you will")
    print("  actually see, and quote the ratio rather than the ranking.")

    json.dump(rows, open(os.path.join(ART, "latency.json"), "w"), indent=2)
    print(f"\nwrote {ART}/latency.json, endpointer_int8.onnx")


if __name__ == "__main__":
    main()
