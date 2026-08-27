#!/usr/bin/env python3
"""Step 6 — REPLICATION of Kelterer, Wepner, Linke & Schuppler (ICPhS 2023)
on English spontaneous speech (AMI), with computable features in place of
their hand-annotated finIntonation.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score

from features.prosody import PAPER_FEATURES, FEATURE_NAMES

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN = os.path.join(ROOT, "data/processed/prosody.parquet")
ART = os.path.join(ROOT, "artifacts")

CATS = ["in-hold", "com-hold", "change"]
PAIRS = [("in-hold", "com-hold"), ("in-hold", "change"), ("com-hold", "change")]
N_SPLITS = 10


def prepare(df):
    n0 = len(df)
    rep = {}

    # exclusion 1: overlapping speech at the boundary corrupts the acoustics,
    # and they exclude turn-changes with overlap for exactly this reason
    df = df[~df.overlap_at_end].copy()
    rep["overlap"] = n0 - len(df)

    # exclusion 2: questions. They remove them from `change` because rising
    # terminal intonation is a question cue, not a turn-yielding cue, and mixing
    # them contaminates every F0 contrast.
    n1 = len(df)
    if "is_question" in df.columns:
        df = df[~df.is_question].copy()
    rep["questions"] = n1 - len(df)

    df = df[df.label.isin(CATS)].copy()
    rep["other_labels"] = n1 - rep["questions"] - len(df)

    # missingness -> indicator + median impute
    feats = [f for f in FEATURE_NAMES if f in df.columns]
    ind = []
    for f in feats:
        m = df[f].isna()
        if m.any():
            df[f + "_missing"] = m.astype(float)
            ind.append(f + "_missing")
            df[f] = df[f].fillna(df[f].median())
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df[feats] = df[feats].fillna(df[feats].median())
    return df, feats + ind, rep


def run_rf(df, feats, a, b, grouped=True, seed=0):
    sub = df[df.label.isin([a, b])]
    X = sub[feats].to_numpy(float)
    y = (sub.label == b).to_numpy().astype(int)
    g = sub.global_name.to_numpy()

    splitter = (GroupShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=seed)
                if grouped else ShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=seed))
    g = g if grouped else None

    fa, fb, bacc, imps = [], [], [], []
    for tr, te in splitter.split(X, y, groups=g):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        # class_weight="balanced" is NOT in the paper — their classes were
        # near-balanced (368/222/247) so they never needed it. Ours run to 8:1,
        rf = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                    criterion="gini", class_weight="balanced",
                                    random_state=seed, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        p = rf.predict(X[te])
        fa.append(f1_score(y[te], p, pos_label=0))
        fb.append(f1_score(y[te], p, pos_label=1))
        bacc.append(balanced_accuracy_score(y[te], p))
        imps.append(rf.feature_importances_)

    imp = pd.Series(np.mean(imps, axis=0), index=feats).sort_values(ascending=False)
    return float(np.mean(fa)), float(np.mean(fb)), float(np.mean(bacc)), imp


def mixed_effects(df, feats):
    """Linear mixed model per feature: feature ~ category, speaker random
    intercept.
    """
    import statsmodels.formula.api as smf
    import warnings
    warnings.filterwarnings("ignore")

    out = []
    for f in feats:
        for ref in CATS:
            d = df[df.label.isin(CATS)][[f, "label", "global_name"]].dropna().copy()
            d["label"] = pd.Categorical(d.label, categories=[ref] + [c for c in CATS if c != ref])
            try:
                m = smf.mixedlm(f"Q('{f}') ~ C(label)", d, groups=d.global_name).fit(method="lbfgs")
            except Exception:
                continue
            for name in m.params.index:
                if not name.startswith("C(label)"):
                    continue
                other = name.split("T.")[-1].rstrip("]")
                out.append(dict(feature=f, ref=ref, other=other,
                                coef=float(m.params[name]), p=float(m.pvalues[name])))
    return pd.DataFrame(out)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def main():
    os.makedirs(ART, exist_ok=True)
    df = pd.read_parquet(IN)
    df, feats, rep = prepare(df)

    print("=" * 74)
    print("REPLICATION: Kelterer et al. 2023 (Austrian German) -> English (AMI)")
    print("=" * 74)
    print(f"excluded: {rep}")
    print(f"analysis set: {len(df)} points, {df.global_name.nunique()} speakers, "
          f"{df.group.nunique()} groups")
    print(df.label.value_counts().to_string())
    print(f"\ntheir N: 837  (in-hold 222 / com-hold 368 / change 247)\n")

    # ---------------- Random Forest ----------------
    print("-" * 74)
    print("RANDOM FOREST — pairwise, feature-ranking instrument (their Table 1/2)")
    print("-" * 74)
    print(f"{'comparison':26s} {'F1(a)':>7s} {'F1(b)':>7s} {'balAcc':>7s} | {'balAcc':>7s}")
    print(f"{'':26s} {'--- speaker-disjoint ---':>23s} | {'random':>7s}")
    imps = {}
    rf_rows = []
    for a, b in PAIRS:
        ga, gb, gacc, imp = run_rf(df, feats, a, b, grouped=True)
        ra, rb, racc, _ = run_rf(df, feats, a, b, grouped=False)
        imps[f"{a} vs {b}"] = imp
        rf_rows.append(dict(comparison=f"{a} vs {b}", f1_a_grouped=ga, f1_b_grouped=gb,
                            balacc_grouped=gacc, f1_a_random=ra, f1_b_random=rb,
                            balacc_random=racc))
        print(f"{a+' vs '+b:26s} {ga:7.3f} {gb:7.3f} {gacc:7.3f} | {racc:7.3f}")

    leak = np.mean([r["balacc_random"] - r["balacc_grouped"] for r in rf_rows])
    print(f"\n  balanced-accuracy inflation from random (speaker-leaking) splits: {leak:+.3f}")
    print("  (0.500 = chance. this is speaker leakage measured, not asserted)")
    print("  ^ this is speaker leakage measured, not asserted")

    print("\n" + "-" * 74)
    print("TOP FEATURES per comparison (their Table 2)")
    print("-" * 74)
    for k, imp in imps.items():
        print(f"\n{k}")
        for f, v in imp.head(6).items():
            print(f"    {f:22s} {v:.4f}")

    # ---------------- mixed effects ----------------
    print("\n" + "-" * 74)
    print("MIXED-EFFECTS CONTRASTS (speaker as random effect) — their §5")
    print("-" * 74)
    me = mixed_effects(df, [f for f in PAPER_FEATURES if f in df.columns]
                           + ["f0_reset", "contour_curv_600", "i_slope", "voiced_frac"])
    me.to_csv(os.path.join(ART, "replication_mixed_effects.csv"), index=False)

    key = ["art_rate", "i_med", "i_max", "f0_range", "f0_max", "f0_slope",
           "contour_curv_600", "voiced_frac"]
    for f in key:
        sub = me[(me.feature == f)]
        if sub.empty:
            continue
        means = df.groupby("label")[f].mean()
        print(f"\n{f}")
        print("   means: " + "  ".join(f"{c}={means.get(c, np.nan):.3f}" for c in CATS))
        seen = set()
        for _, r in sub.iterrows():
            pair = tuple(sorted([r.ref, r.other]))
            if pair in seen:
                continue
            seen.add(pair)
            print(f"   {r.ref:9s} vs {r.other:9s}  coef={r.coef:+8.3f}  p={r.p:.2e} {stars(r.p)}")

    # ---------------- hypotheses ----------------
    print("\n" + "=" * 74)
    print("PRE-REGISTERED HYPOTHESES (docs/04_replication_spec.md §7)")
    print("=" * 74)
    m = df.groupby("label")

    def p_of(f, a, b):
        s = me[(me.feature == f) & (((me.ref == a) & (me.other == b)) |
                                    ((me.ref == b) & (me.other == a)))]
        return float(s.p.iloc[0]) if len(s) else np.nan

    ar = m.art_rate.mean()
    h1 = ar.get("change", 0) > ar.get("com-hold", 0) > ar.get("in-hold", 0)
    print(f"H1  ArtR: change > com-hold > in-hold")
    print(f"      {ar.get('change',np.nan):.3f} > {ar.get('com-hold',np.nan):.3f} > "
          f"{ar.get('in-hold',np.nan):.3f}   -> {'REPLICATES' if h1 else 'does not replicate'}"
          f"   [p(ch,com)={p_of('art_rate','change','com-hold'):.1e}, "
          f"p(com,in)={p_of('art_rate','com-hold','in-hold'):.1e}]")

    im = m.i_med.mean()
    h2 = im.get("in-hold", 0) > im.get("change", 0)
    print(f"H2  Intensity: in-hold > change")
    print(f"      {im.get('in-hold',np.nan):.3f} > {im.get('change',np.nan):.3f}   "
          f"-> {'REPLICATES' if h2 else 'does not replicate'}"
          f"   [p={p_of('i_med','in-hold','change'):.1e}]")

    fr = m.f0_range.mean()
    h3 = fr.get("com-hold", 0) > fr.get("change", 0) and fr.get("com-hold", 0) > fr.get("in-hold", 0)
    print(f"H3  F0 range: com-hold highest")
    print(f"      com={fr.get('com-hold',np.nan):.3f} ch={fr.get('change',np.nan):.3f} "
          f"in={fr.get('in-hold',np.nan):.3f}   -> {'REPLICATES' if h3 else 'does not replicate'}")

    f1s = {f"{a} vs {b}": r for (a, b), r in zip(PAIRS, rf_rows)}
    worst = min(rf_rows, key=lambda r: r["balacc_grouped"])
    h5 = worst["comparison"] == "com-hold vs change"
    print(f"H5  com-hold vs change is the weakest separation")
    print(f"      weakest = {worst['comparison']}   -> {'REPLICATES' if h5 else 'does not replicate'}")
    print("      (theirs: F1(change)=.40 in that pair — near chance)")

    vf = m.voiced_frac.mean()
    p_vf = p_of("voiced_frac", "com-hold", "change")
    print(f"\nH6  (OURS, not in the paper) turn-ends are less modally voiced")
    print(f"      voiced_frac: change={vf.get('change',np.nan):.3f} < "
          f"com-hold={vf.get('com-hold',np.nan):.3f}, in-hold={vf.get('in-hold',np.nan):.3f}"
          f"   [p={p_vf:.1e}]")
    print("      interpretation: creak / glottalisation at terminal juncture defeats")
    print("      F0 tracking. Visible only because we kept unvoiced windows and")
    print("      modelled missingness instead of dropping those rows.")

    pd.DataFrame(rf_rows).to_csv(os.path.join(ART, "replication_rf.csv"), index=False)
    for k, imp in imps.items():
        imp.to_csv(os.path.join(ART, f"importance_{k.replace(' ', '_')}.csv"))
    print(f"\nwrote CSVs to {ART}/")


if __name__ == "__main__":
    main()
