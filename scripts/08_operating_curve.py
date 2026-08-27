#!/usr/bin/env python3
"""Step 8 — the operating curve. THE headline result."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval.policy import sweep, sweep_fixed, pareto_front, area_under_curve, score_policy

ROOT = os.path.join(os.path.dirname(__file__), "..")
ART = os.path.join(ROOT, "artifacts")
OOF = os.path.join(ART, "oof_predictions.parquet")

# validated categorical slots (dataviz reference palette, light mode)
C = {"fixed": "#52514e", "text": "#eb6834", "prosody": "#1baf7a", "fusion": "#2a78d6"}
LABEL = {"fixed": "Fixed VAD threshold", "text": "Text only",
         "prosody": "Prosody only", "fusion": "Text + prosody (fusion)"}


def main():
    df = pd.read_parquet(OOF)
    df = df[np.isfinite(df.p_fusion)].copy()
    fc = df.floor_changes.to_numpy().astype(bool)
    D = df.resume_delay.to_numpy().astype(float)

    print(f"{len(df)} decision points  ({fc.sum()} changes / {(~fc).sum()} holds)")
    print("all predictions are speaker-disjoint out-of-fold\n")

    fronts, summary = {}, []

    fixed = sweep_fixed(fc, D)
    fronts["fixed"] = pareto_front(fixed)
    for v in ("text", "prosody", "fusion"):
        rows = sweep(df[f"p_{v}"].to_numpy(), fc, D)
        fronts[v] = pareto_front(rows)

    print(f"{'system':28s} {'AUOC':>8s}   {'FIR @ 500ms':>12s}  {'latency @ 10% FIR':>18s}")
    print("-" * 74)
    for k in ("fixed", "text", "prosody", "fusion"):
        f = fronts[k]
        auoc = area_under_curve(f)
        xs = np.array([r["latency_med"] for r in f])
        ys = np.array([r["fir"] for r in f])
        fir_500 = float(np.interp(0.50, xs, ys))
        lat_10 = float(np.interp(0.10, ys[::-1], xs[::-1])) if ys.min() <= 0.10 <= ys.max() else np.nan
        summary.append(dict(system=k, auoc=auoc, fir_at_500ms=fir_500, latency_at_10pct=lat_10))
        print(f"{LABEL[k]:28s} {auoc:8.4f}   {fir_500*100:11.1f}%  {lat_10*1000:15.0f} ms")

    # ------------------------------------------------------------------
    # The headline metric. AUOC is a convenient scalar but it averages over
    def latency_at(front, fir_target):
        ys = np.array([r["fir"] for r in front])
        xs = np.array([r["latency_med"] for r in front])
        o = np.argsort(ys)
        return float(np.interp(fir_target, ys[o], xs[o]))

    print("\n" + "-" * 74)
    print("LATENCY AT MATCHED FALSE-INTERRUPTION RATE  (the number that matters)")
    print("-" * 74)
    print(f"{'tolerated FIR':>14s} {'fixed VAD':>11s} {'text':>9s} {'fusion':>9s} "
          f"{'saving vs fixed':>18s}")
    save_rows = []
    for tgt in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        lf = latency_at(fronts["fixed"], tgt)
        lt = latency_at(fronts["text"], tgt)
        lu = latency_at(fronts["fusion"], tgt)
        d = lf - lu
        save_rows.append(dict(fir=tgt, fixed=lf, text=lt, fusion=lu,
                              saving_ms=d * 1000, saving_pct=d / lf * 100))
        print(f"{tgt*100:13.0f}% {lf*1000:10.0f}ms {lt*1000:8.0f}ms {lu*1000:8.0f}ms "
              f"{d*1000:10.0f}ms ({d/lf*100:4.1f}%)")
    med_save = np.median([r["saving_ms"] for r in save_rows])
    print(f"\n  median latency saving at matched FIR: {med_save:.0f} ms")
    print("  for scale: the median human inter-turn gap is ~200 ms (Stivers et al. 2009),")
    print("  so this is on the order of one whole conversational beat per exchange.")
    pd.DataFrame(save_rows).to_csv(os.path.join(ART, "latency_saving.csv"), index=False)

    print("\n" + "-" * 74)
    print("NOTE ON ABSOLUTE LEVELS")
    print("-" * 74)
    print("  Absolute FIR is high across ALL systems because every VAD pause > 150 ms")
    print("  in 4-party AMI meeting speech counts as a decision point, and the median")
    print("  hold lasts 576 ms. Meeting speech pauses far more than a 2-party support")
    print("  call. The comparison is controlled — identical decision points, identical")
    print("  delays, identical policy family — so the RELATIVE gap transfers even though")
    print("  the absolute level is domain-specific.")

    base = summary[0]["auoc"]
    best = summary[-1]["auoc"]
    print(f"\nfusion reduces mean false-interruption rate over the 0.15-1.5 s band")
    print(f"by {(base-best)/base*100:.1f}% relative to the fixed-threshold baseline.")

    # ---------------- plot ----------------
    fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=170)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

    for k in ("fixed", "text", "prosody", "fusion"):
        f = fronts[k]
        x = [r["latency_med"] * 1000 for r in f]
        y = [r["fir"] * 100 for r in f]
        lw = 2.6 if k == "fusion" else 2.0
        z = 5 if k == "fusion" else 3
        ax.plot(x, y, color=C[k], lw=lw, label=LABEL[k], zorder=z,
                solid_capstyle="round")

    ax.set_xlabel("Median response latency on turn-ends  (ms)", fontsize=10.5, color="#52514e")
    ax.set_ylabel("False-interruption rate on holds  (%)", fontsize=10.5, color="#52514e")
    ax.set_title("Adaptive endpointing dominates a fixed silence threshold",
                 fontsize=13, color="#0b0b0b", pad=14, loc="left")
    ax.text(0, 1.02, "", transform=ax.transAxes)
    ax.set_xlim(150, 1500); ax.set_ylim(0, 100)
    ax.grid(True, color="#e6e5e1", lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=9.5)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color("#0b0b0b")
    ax.annotate("better", xy=(0.055, 0.10), xytext=(0.20, 0.28),
                xycoords="axes fraction", textcoords="axes fraction",
                fontsize=9.5, color="#52514e",
                arrowprops=dict(arrowstyle="->", color="#52514e", lw=1.4))
    fig.tight_layout()
    out = os.path.join(ART, "operating_curve.png")
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(f"\nwrote {out}")

    pd.DataFrame(summary).to_csv(os.path.join(ART, "operating_summary.csv"), index=False)
    for k, f in fronts.items():
        pd.DataFrame(f).to_csv(os.path.join(ART, f"front_{k}.csv"), index=False)

    # a concrete shipping operating point
    print("\n--- a concrete operating point (tau_min=0.20, tau_max=1.40) ---")
    for v in ("text", "fusion"):
        s = score_policy(df[f"p_{v}"].to_numpy(), fc, D, 0.20, 1.40)
        print(f"  {v:8s} median latency {s['latency_med']*1000:4.0f} ms | "
              f"p95 {s['latency_p95']*1000:4.0f} ms | FIR {s['fir']*100:5.1f}% "
              f"({s['n_interrupted']}/{s['n_holds']} holds)")
    best_fixed = min(fixed, key=lambda r: abs(r["fir"] - s["fir"]))
    print(f"  fixed VAD at the SAME interruption rate ({best_fixed['fir']*100:.1f}%): "
          f"median latency {best_fixed['latency_med']*1000:.0f} ms")


if __name__ == "__main__":
    main()
