"""Endpointer model: P(floor changes | transcript so far, prosody)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ProsodyBranch(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class TextBranch(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 out_dim: int = 64, freeze: bool = False):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, out_dim), nn.GELU())

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool over real tokens. MiniLM is trained with mean pooling
        # (it is a sentence-transformer); using [CLS] instead would use a token
        # this checkpoint never learned to make meaningful.
        h = out.last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.proj(pooled)


class EndpointModel(nn.Module):
    """use_text / use_prosody give us the three ablations from one class."""

    def __init__(self, n_prosody: int, use_text: bool = True, use_prosody: bool = True,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hidden: int = 64, freeze_text: bool = False):
        super().__init__()
        assert use_text or use_prosody
        self.use_text, self.use_prosody = use_text, use_prosody

        dim = 0
        if use_text:
            self.text = TextBranch(model_name, hidden, freeze_text)
            dim += hidden
        if use_prosody:
            self.prosody = ProsodyBranch(n_prosody, hidden)
            dim += hidden

        self.head = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )
        # learned temperature for calibration; fitted AFTER training, on val data
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, input_ids=None, attention_mask=None, prosody=None):
        parts = []
        if self.use_text:
            parts.append(self.text(input_ids, attention_mask))
        if self.use_prosody:
            parts.append(self.prosody(prosody))
        return self.head(torch.cat(parts, dim=-1)).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, **kw):
        return torch.sigmoid(self.forward(**kw) / self.temperature.clamp(min=1e-2))


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                    max_iter: int = 200) -> float:
    """Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural
    Networks"). One scalar T, fitted on HELD-OUT data to minimise NLL:
    """
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=max_iter)
    lossf = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = lossf(logits / T.clamp(min=1e-2), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-2))


def expected_calibration_error(p, y, n_bins: int = 15) -> float:
    """ECE: average |confidence − accuracy| over equal-width probability bins."""
    import numpy as np
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(ece)
