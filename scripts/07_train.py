#!/usr/bin/env python3
"""Step 7 — train the endpointer and produce speaker-disjoint out-of-fold
predictions.
"""
import os, sys, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from features.prosody import FEATURE_NAMES
from models.fusion import EndpointModel, fit_temperature, expected_calibration_error

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN = os.path.join(ROOT, "data/processed/prosody.parquet")
ART = os.path.join(ROOT, "artifacts")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 48
EPOCHS = 4
BATCH = 32
LR_HEAD, LR_ENC = 1e-3, 3e-5


class DS(Dataset):
    def __init__(self, df, X, tok):
        self.texts = df.text.fillna("").tolist()
        self.X = X.astype(np.float32)
        self.y = df.floor_changes.to_numpy().astype(np.float32)
        self.tok = tok

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        return (enc["input_ids"][0], enc["attention_mask"][0],
                torch.from_numpy(self.X[i]), torch.tensor(self.y[i]))


def build_features(df):
    """Prosody matrix + missingness indicators. Same treatment as docs/07."""
    feats = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[feats].copy()
    ind = {}
    for f in feats:
        m = X[f].isna() | ~np.isfinite(X[f])
        if m.any():
            ind[f + "_missing"] = m.astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = pd.concat([X, pd.DataFrame(ind, index=X.index)], axis=1)
    return X, feats + list(ind)


def run_epoch(model, dl, opt, dev, train=True):
    model.train() if train else model.eval()
    lossf = nn.BCEWithLogitsLoss()
    tot, n, logits_all, y_all = 0.0, 0, [], []
    for ids, am, pr, y in dl:
        ids, am, pr, y = ids.to(dev), am.to(dev), pr.to(dev), y.to(dev)
        with torch.set_grad_enabled(train):
            kw = {}
            if model.use_text:
                kw.update(input_ids=ids, attention_mask=am)
            if model.use_prosody:
                kw.update(prosody=pr)
            out = model(**kw)
            loss = lossf(out, y)
        if train:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += float(loss) * len(y); n += len(y)
        logits_all.append(out.detach().cpu()); y_all.append(y.detach().cpu())
    return tot / n, torch.cat(logits_all), torch.cat(y_all)


def train_variant(df, X, cols, variant, dev, seed=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    use_text = variant in ("text", "fusion")
    use_pros = variant in ("prosody", "fusion")

    groups = df.group.to_numpy()
    oof_p = np.full(len(df), np.nan)
    oof_T = np.full(len(df), np.nan)
    folds = GroupKFold(n_splits=4)

    for k, (tr_idx, te_idx) in enumerate(folds.split(df, groups=groups)):
        torch.manual_seed(seed + k); np.random.seed(seed + k)

        # carve a validation GROUP out of train — for early stopping and for
        # fitting the temperature. Never the test fold.
        tr_groups = pd.unique(groups[tr_idx])
        val_g = tr_groups[-1]
        va_idx = tr_idx[groups[tr_idx] == val_g]
        tr_idx = tr_idx[groups[tr_idx] != val_g]

        # standardise prosody on TRAIN ONLY
        mu = np.nanmean(X.iloc[tr_idx], axis=0)
        sd = np.nanstd(X.iloc[tr_idx], axis=0) + 1e-6
        Z = ((X.to_numpy(float) - mu) / sd)
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        model = EndpointModel(n_prosody=Z.shape[1], use_text=use_text,
                              use_prosody=use_pros, model_name=MODEL_NAME).to(dev)
        enc_p = list(model.text.encoder.parameters()) if use_text else []
        enc_ids = {id(p) for p in enc_p}
        head_p = [p for p in model.parameters() if id(p) not in enc_ids]
        opt = torch.optim.AdamW(
            ([{"params": enc_p, "lr": LR_ENC}] if enc_p else []) +
            [{"params": head_p, "lr": LR_HEAD}], weight_decay=0.01)

        dl_tr = DataLoader(DS(df.iloc[tr_idx], Z[tr_idx], tok), batch_size=BATCH, shuffle=True)
        dl_va = DataLoader(DS(df.iloc[va_idx], Z[va_idx], tok), batch_size=64)
        dl_te = DataLoader(DS(df.iloc[te_idx], Z[te_idx], tok), batch_size=64)

        best, best_state = np.inf, None
        for ep in range(EPOCHS):
            t0 = time.time()
            tr_loss, _, _ = run_epoch(model, dl_tr, opt, dev, True)
            va_loss, va_log, va_y = run_epoch(model, dl_va, opt, dev, False)
            auc = roc_auc_score(va_y, va_log) if len(np.unique(va_y)) > 1 else np.nan
            print(f"    fold{k} ep{ep}  train {tr_loss:.4f}  val {va_loss:.4f}  "
                  f"val AUC {auc:.3f}  ({time.time()-t0:.0f}s)", flush=True)
            if va_loss < best:
                best = va_loss
                best_state = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
        if best_state:
            model.load_state_dict(best_state)

        # calibrate on validation, then predict the untouched test fold
        _, va_log, va_y = run_epoch(model, dl_va, opt, dev, False)
        T = fit_temperature(va_log.float(), va_y.float())
        _, te_log, _ = run_epoch(model, dl_te, opt, dev, False)
        oof_p[te_idx] = torch.sigmoid(te_log.float() / T).numpy()
        oof_T[te_idx] = T
        print(f"    fold{k} temperature T={T:.3f}  test n={len(te_idx)}", flush=True)

        if variant == "fusion" and k == 0:
            torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                        "cols": cols, "T": T},
                       os.path.join(ART, "fusion_fold0.pt"))
    return oof_p, oof_T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all",
                    choices=["all", "text", "prosody", "fusion"])
    a = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    df = pd.read_parquet(IN).reset_index(drop=True)
    X, cols = build_features(df)
    dev = ("mps" if torch.backends.mps.is_available()
           else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"{len(df)} points · {df.group.nunique()} groups · "
          f"{df.floor_changes.mean()*100:.1f}% changes · device={dev}\n")

    variants = ["text", "prosody", "fusion"] if a.variant == "all" else [a.variant]
    out = df[["meeting", "group", "global_name", "t_decision", "resume_delay",
              "floor_changes", "label", "text"]].copy()

    for v in variants:
        print(f"--- {v} ---")
        p, T = train_variant(df, X, cols, v, dev)
        out[f"p_{v}"] = p
        ok = ~np.isnan(p)
        auc = roc_auc_score(out.floor_changes[ok], p[ok])
        ap_ = average_precision_score(out.floor_changes[ok], p[ok])
        ece = expected_calibration_error(p[ok], out.floor_changes[ok].to_numpy().astype(float))
        print(f"  {v}:  OOF AUC {auc:.4f}   AP {ap_:.4f}   ECE {ece:.4f}\n")

    out.to_parquet(os.path.join(ART, "oof_predictions.parquet"), index=False)
    print(f"wrote {ART}/oof_predictions.parquet")


if __name__ == "__main__":
    main()
