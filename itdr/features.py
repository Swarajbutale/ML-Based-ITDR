"""Feature engineering: one row per (user, day) window.

Four feature families from the article:
  authentication, behavioural context, entitlement changes, resource activity.
Baselines are per-user (devices / countries / volume seen on EARLIER days only),
so no future information leaks into a window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    # authentication
    "fail_count", "success_after_fail", "mfa_fail_count",
    # behavioural context
    "new_device", "new_country", "off_hours_ratio",
    # resource activity
    "access_count", "sensitive_access_count", "volume_ratio",
    # entitlement changes
    "role_change_count", "unapproved_change_count", "group_add_count",
    "credential_add_count", "privilege_elevation",
]

FEATURE_LABELS = {
    "fail_count": "failed sign-ins",
    "success_after_fail": "success after 5+ consecutive failures",
    "mfa_fail_count": "failed MFA challenges",
    "new_device": "device not seen before",
    "new_country": "country not seen before",
    "off_hours_ratio": "share of activity at off-hours (22:00-06:00)",
    "access_count": "resource accesses",
    "sensitive_access_count": "sensitive-resource accesses",
    "volume_ratio": "access volume vs. personal median",
    "role_change_count": "role changes",
    "unapproved_change_count": "changes without approval",
    "group_add_count": "group additions",
    "credential_add_count": "credentials added",
    "privilege_elevation": "elevation to admin role",
}

WARMUP_DAYS = 2  # first days only seed the per-user baseline


def build_features(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev["timestamp"] = pd.to_datetime(ev["timestamp"])
    ev["date"] = ev["timestamp"].dt.normalize()
    ev["hour"] = ev["timestamp"].dt.hour
    day0 = ev["date"].min()
    rows = []

    for uid, g in ev.sort_values("timestamp").groupby("user_id"):
        known_dev: set = set()
        known_ctry: set = set()
        hist_access: list[int] = []
        for date, d in g.groupby("date"):
            day_idx = int((date - day0).days)
            signins = d[d["event_type"] == "signin"]

            streak, success_after_fail = 0, 0
            for outcome in signins.sort_values("timestamp")["outcome"]:
                if outcome == "failure":
                    streak += 1
                else:
                    if streak >= 5:
                        success_after_fail = 1
                    streak = 0

            devs, ctrys = set(d["device_id"]), set(d["country"])
            warm = day_idx < WARMUP_DAYS
            acc = d[d["event_type"] == "resource_access"]
            changes = d[d["event_type"].isin(["role_change", "group_add", "credential_add"])]
            median_prior = float(np.median(hist_access)) if hist_access else None
            off = ((d["hour"] < 6) | (d["hour"] >= 22)).mean()

            rows.append(dict(
                alert_id=f"{uid}-{date:%m%d}", user_id=uid, role=d["role"].iloc[0],
                date=date, day_idx=day_idx,
                fail_count=int((signins["outcome"] == "failure").sum()),
                success_after_fail=success_after_fail,
                mfa_fail_count=int(((d["event_type"] == "mfa") & (d["outcome"] == "failure")).sum()),
                new_device=0 if warm else int(len(devs - known_dev) > 0),
                new_country=0 if warm else int(len(ctrys - known_ctry) > 0),
                off_hours_ratio=round(float(off), 3),
                access_count=len(acc),
                sensitive_access_count=int(acc["sensitive"].sum()),
                volume_ratio=round(len(acc) / max(1.0, median_prior), 3) if median_prior is not None else 1.0,
                role_change_count=int((d["event_type"] == "role_change").sum()),
                unapproved_change_count=int((~changes["approved"].astype(bool)).sum()),
                group_add_count=int((d["event_type"] == "group_add").sum()),
                credential_add_count=int((d["event_type"] == "credential_add").sum()),
                privilege_elevation=int(((d["event_type"] == "role_change") & (d["detail"] == "role:admin")).any()),
                n_events=len(d),
                is_attack=int(d["is_attack"].any()),
                scenario=next((s for s in d["scenario"] if s != "normal"), "normal"),
            ))
            known_dev |= devs
            known_ctry |= ctrys
            hist_access.append(len(acc))

    return pd.DataFrame(rows).sort_values(["day_idx", "user_id"]).reset_index(drop=True)
