"""Hybrid detection and risk scoring (stage 4) plus the baseline comparison
from the article's evaluation plan (rules / anomaly / supervised / hybrid).

risk = w_rules * rule_score + w_anomaly * anomaly_score + w_supervised * supervised_score
(weights are documented, user-adjustable, and re-normalised if a component is unavailable)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

# Transparent, auditable rules: name -> (predicate, weight)
RULES = {
    "R1 success after 5+ consecutive failures": (lambda r: r["success_after_fail"] == 1, 0.80),
    "R2 unapproved privilege elevation": (
        lambda r: r["privilege_elevation"] == 1 and r["unapproved_change_count"] > 0, 0.90),
    "R3 new device + new country + sensitive access": (
        lambda r: r["new_device"] == 1 and r["new_country"] == 1 and r["sensitive_access_count"] > 0, 0.60),
    "R4 credential added during off-hours": (
        lambda r: r["credential_add_count"] > 0 and r["off_hours_ratio"] > 0.5, 0.60),
}

DEFAULT_WEIGHTS = {"rules": 0.40, "anomaly": 0.30, "supervised": 0.30}


@dataclass
class ITDRPipeline:
    seed: int = 42
    iforest: IsolationForest | None = None
    scaler: StandardScaler | None = None
    lr: LogisticRegression | None = None
    train_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    med: pd.Series | None = None
    spread: pd.Series | None = None
    sup_ok: bool = False
    split_day: int = 0
    version: str = "itdr-hybrid-0.1"

    # ---- training ---------------------------------------------------------
    def fit(self, feats: pd.DataFrame, split_day: int) -> "ITDRPipeline":
        """Fit on days < split_day only (time-based split, no future leakage)."""
        self.split_day = split_day
        train = feats[feats["day_idx"] < split_day]
        X = train[FEATURES]

        self.iforest = IsolationForest(n_estimators=200, random_state=self.seed).fit(X)
        self.train_scores = np.sort(-self.iforest.score_samples(X))

        self.med = X.median()
        mad = (X - self.med).abs().median() * 1.4826
        self.spread = mad.where(mad > 0, X.std()).replace(0, 1.0).fillna(1.0)

        y = train["is_attack"].to_numpy()
        self.sup_ok = int(y.sum()) >= 3 and int((y == 0).sum()) >= 3
        if self.sup_ok:
            self.scaler = StandardScaler().fit(X)
            self.lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=self.seed)
            self.lr.fit(self.scaler.transform(X), y)
        return self

    # ---- scoring ----------------------------------------------------------
    def score(self, feats: pd.DataFrame, weights: dict | None = None,
              medium: float = 0.35, high: float = 0.60) -> pd.DataFrame:
        w = dict(weights or DEFAULT_WEIGHTS)
        out = feats.copy()
        X = out[FEATURES]

        fired, rscore = [], []
        for _, r in out.iterrows():
            hits = [(name, wt) for name, (fn, wt) in RULES.items() if fn(r)]
            fired.append([h[0] for h in hits])
            rscore.append(max((h[1] for h in hits), default=0.0))
        out["rules_fired"], out["rule_score"] = fired, rscore

        raw = -self.iforest.score_samples(X)
        out["anomaly_score"] = np.searchsorted(self.train_scores, raw) / max(1, len(self.train_scores))
        out["supervised_score"] = (
            self.lr.predict_proba(self.scaler.transform(X))[:, 1] if self.sup_ok else 0.0)

        if not self.sup_ok:
            w["supervised"] = 0.0
        total = sum(w.values()) or 1.0
        out["risk"] = (w["rules"] * out["rule_score"] + w["anomaly"] * out["anomaly_score"]
                       + w["supervised"] * out["supervised_score"]) / total
        out["tier"] = np.where(out["risk"] >= high, "high", np.where(out["risk"] >= medium, "medium", "low"))
        return out


def evaluate(scored: pd.DataFrame, split_day: int, medium: float = 0.35) -> pd.DataFrame:
    """Compare the four detection strategies on the held-out (later) period."""
    test = scored[scored["day_idx"] >= split_day]
    y = test["is_attack"].to_numpy()
    strategies = {
        "(i) Rules only": (test["rule_score"], test["rule_score"] > 0),
        "(ii) Anomaly only (top 5%)": (test["anomaly_score"], test["anomaly_score"] >= 0.95),
        "(iii) Supervised only": (test["supervised_score"], test["supervised_score"] >= 0.5),
        "(iv) Hybrid": (test["risk"], test["risk"] >= medium),
    }
    rows = []
    for name, (cont, pred) in strategies.items():
        pred = pred.to_numpy().astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        rows.append({
            "Strategy": name,
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "False-positive rate": fp / max(1, fp + tn),
            "PR-AUC": average_precision_score(y, cont) if y.sum() else float("nan"),
            "Alerts": int(pred.sum()),
        })
    return pd.DataFrame(rows).set_index("Strategy")
