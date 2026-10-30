"""Explainable alert construction.

Keeps OBSERVED FACTS (raw events, feature values) separate from MODEL-DERIVED
INDICATORS (attributions, anomaly deviations), as the article requires.

Attribution method: the supervised component is a logistic regression, so each
feature's contribution to the log-odds is exactly  coef_i * (x_i - mean_i) / scale_i.
This equals the SHAP value of a linear model under feature independence, without
needing the `shap` package. Swap in shap.TreeExplainer if you move to tree models.
Attributions are indicators, not proof of causation or malicious intent.
"""
from __future__ import annotations

import pandas as pd

from .features import FEATURE_LABELS, FEATURES


def contributions(row: pd.Series, pipe) -> pd.DataFrame:
    """Supervised-model attribution (log-odds) per feature for one window."""
    if not pipe.sup_ok:
        return pd.DataFrame(columns=["feature", "label", "value", "contribution"])
    x = row[FEATURES].astype(float)
    z = (x - pipe.scaler.mean_) / pipe.scaler.scale_
    c = z * pipe.lr.coef_[0]
    df = pd.DataFrame({"feature": FEATURES, "label": [FEATURE_LABELS[f] for f in FEATURES],
                       "value": x.values, "contribution": c.values})
    return df.sort_values("contribution", ascending=False).reset_index(drop=True)


def deviations(row: pd.Series, pipe) -> pd.DataFrame:
    """Robust deviation of each feature from the training median (anomaly evidence)."""
    x = row[FEATURES].astype(float)
    dev = (x - pipe.med) / pipe.spread
    df = pd.DataFrame({"feature": FEATURES, "label": [FEATURE_LABELS[f] for f in FEATURES],
                       "value": x.values, "typical": pipe.med.values, "robust_z": dev.values})
    return df.reindex(df["robust_z"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def timeline(row: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    ev = events[(events["user_id"] == row["user_id"])
                & (pd.to_datetime(events["timestamp"]).dt.normalize() == row["date"])]
    cols = ["timestamp", "event_type", "outcome", "device_id", "country", "resource", "detail", "approved", "event_id"]
    return ev.sort_values("timestamp")[cols].reset_index(drop=True)


def suggested_steps(row: pd.Series) -> list[str]:
    steps = []
    if row["success_after_fail"]:
        steps.append("Check the failed-sign-in burst source (IP/device) and whether the password was reset or reused elsewhere.")
    if row["new_device"] or row["new_country"]:
        steps.append("Confirm with the user (out-of-band) whether they were travelling or using a new device.")
    if row["privilege_elevation"] or row["unapproved_change_count"]:
        steps.append("Verify the role/group/credential changes against an approved change ticket; review who performed them.")
    if row["credential_add_count"]:
        steps.append("Review newly created credentials/API keys; revoke any that were not requested.")
    if row["sensitive_access_count"] and row["off_hours_ratio"] > 0.5:
        steps.append("Review sensitive-resource access performed off-hours and whether data was exported.")
    return steps or ["Review the event timeline and compare with the user's usual behaviour."]


def build_alert(row: pd.Series, pipe, events: pd.DataFrame) -> dict:
    contrib = contributions(row, pipe)
    dev = deviations(row, pipe)
    top_pos = contrib[contrib["contribution"] > 0.05].head(4)
    observed = []
    if row["fail_count"]:
        observed.append(f"{int(row['fail_count'])} failed sign-in(s)")
    if row["new_device"]:
        observed.append("a device not previously seen for this user")
    if row["new_country"]:
        observed.append("a country not previously seen for this user")
    if row["off_hours_ratio"] > 0.5:
        observed.append(f"{row['off_hours_ratio']:.0%} of activity at off-hours")
    if row["sensitive_access_count"]:
        observed.append(f"{int(row['sensitive_access_count'])} sensitive-resource access(es)")
    if row["unapproved_change_count"]:
        observed.append(f"{int(row['unapproved_change_count'])} entitlement change(s) without approval")
    summary = (f"Window {row['alert_id']} scored {row['risk']:.2f} ({row['tier'].upper()}). "
               f"Observed: {', '.join(observed) if observed else 'no unusual raw signals'}. "
               "This is model evidence for analyst review, not a finding of malicious intent.")
    uncertainty = (
        "Single-window score from a model trained on a small synthetic dataset; "
        "attributions assume feature independence and may mislead when features are correlated."
        + ("" if pipe.sup_ok else " Supervised component unavailable (too few labelled examples)."))
    return dict(summary=summary, rules=row["rules_fired"], contributions=contrib, top_indicators=top_pos,
                deviations=dev, timeline=timeline(row, events), steps=suggested_steps(row),
                uncertainty=uncertainty)
