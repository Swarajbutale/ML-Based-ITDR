"""Synthetic identity-event generator.

Stands in for the "Identity sources" stage of the framework (IdP, SSO, MFA, PAM,
cloud audit logs). Produces normal workforce activity plus injected scenarios
inspired by MITRE ATT&CK T1078 (Valid Accounts) and T1098 (Account Manipulation).

The `scenario` / `is_attack` columns are GROUND TRUTH used only for training the
supervised model on the early period and for evaluation. They never appear in
the feature set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLE_RESOURCES = {
    "admin": ["iam-console", "server-admin", "ticketing"],
    "developer": ["git", "ci", "wiki", "staging", "prod-db"],
    "finance": ["erp", "payroll", "reports"],
    "hr": ["hris", "payroll", "wiki"],
    "support": ["ticketing", "crm", "wiki"],
}
SENSITIVE = {"iam-console", "server-admin", "payroll", "erp", "hris", "prod-db"}
ALL_RESOURCES = sorted({r for v in ROLE_RESOURCES.values() for r in v})
ROLE_P = [0.08, 0.40, 0.25, 0.12, 0.15]  # admin, developer, finance, hr, support

COLUMNS = [
    "event_id", "timestamp", "user_id", "role", "event_type", "outcome",
    "device_id", "country", "resource", "sensitive", "approved", "detail",
    "session_id", "scenario", "is_attack",
]


def generate_events(n_users: int = 60, days: int = 14, seed: int = 42,
                    start: str = "2026-10-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    roles = rng.choice(list(ROLE_RESOURCES), size=n_users, p=ROLE_P)
    users = [
        dict(user_id=f"user{i:03d}", role=str(roles[i]),
             country=str(rng.choice(["IN", "IN", "IN", "IN", "US"])),
             devices=[f"dev-{i:03d}-a", f"dev-{i:03d}-b"],
             shift=int(rng.integers(8, 11)))
        for i in range(n_users)
    ]
    by_id = {u["user_id"]: u for u in users}
    rows: list[dict] = []
    counter = {"n": 0, "s": 0}

    def ev(ts, u, etype, outcome="success", device=None, country=None, resource="",
           approved=True, detail="", session="", scenario="normal"):
        counter["n"] += 1
        rows.append(dict(
            event_id=f"e{counter['n']:06d}", timestamp=ts, user_id=u["user_id"],
            role=u["role"], event_type=etype, outcome=outcome,
            device_id=device or u["devices"][0], country=country or u["country"],
            resource=resource, sensitive=resource in SENSITIVE, approved=approved,
            detail=detail, session_id=session, scenario=scenario,
            is_attack=scenario not in ("normal", "decoy_travel"),
        ))

    def new_session():
        counter["s"] += 1
        return f"s{counter['s']:05d}"

    # ---- normal behaviour -------------------------------------------------
    for d in range(days):
        day = t0 + pd.Timedelta(days=d)
        for u in users:
            if rng.random() > 0.85:
                continue
            for _ in range(1 + rng.poisson(1)):
                ts = day + pd.Timedelta(hours=u["shift"] + int(rng.integers(0, 3)),
                                        minutes=int(rng.integers(0, 60)))
                dev = str(rng.choice(u["devices"]))
                sid = new_session()
                if rng.random() < 0.03:
                    ev(ts, u, "signin", "failure", dev, session=sid)
                    ts += pd.Timedelta(minutes=1)
                ev(ts, u, "signin", "success", dev, session=sid)
                ev(ts + pd.Timedelta(seconds=20), u, "mfa", "success", dev, session=sid)
                for k in range(rng.poisson(5)):
                    res = str(rng.choice(ROLE_RESOURCES[u["role"]]))
                    ev(ts + pd.Timedelta(minutes=2 + 5 * k), u, "resource_access",
                       device=dev, resource=res, session=sid)
            if u["role"] == "admin" and rng.random() < 0.15:  # benign approved admin work
                ts = day + pd.Timedelta(hours=u["shift"] + 3)
                target = str(rng.choice(["role:support", "role:developer", "role:admin"]))
                ev(ts, u, "role_change", detail=target, approved=True)
                ev(ts + pd.Timedelta(minutes=3), u, "group_add", detail="group:project-x", approved=True)

    # ---- injected scenarios -------------------------------------------------
    n_attacks = max(10, n_users // 4)
    pairs: set[tuple[str, int]] = set()
    while len(pairs) < n_attacks * 2:
        pairs.add((users[int(rng.integers(0, n_users))]["user_id"], int(rng.integers(2, days))))
    pairs_l = sorted(pairs)
    attacks, decoys = pairs_l[:n_attacks], pairs_l[n_attacks:]
    kinds = ["valid_account_misuse", "brute_force_success", "privilege_escalation"]

    for i, (uid, d) in enumerate(attacks):
        u, day, kind = by_id[uid], t0 + pd.Timedelta(days=d), kinds[i % 3]
        sid = new_session()
        if kind == "valid_account_misuse":
            ts = day + pd.Timedelta(hours=2, minutes=int(rng.integers(0, 50)))
            dev, ctry = f"dev-unknown-{int(rng.integers(1, 999))}", str(rng.choice(["RO", "BR", "VN"]))
            ev(ts, u, "signin", "success", dev, ctry, session=sid, scenario=kind)
            ev(ts + pd.Timedelta(seconds=15), u, "mfa", "success", dev, ctry, session=sid, scenario=kind)
            for k in range(int(rng.integers(8, 16))):
                ev(ts + pd.Timedelta(minutes=1 + 2 * k), u, "resource_access", device=dev, country=ctry,
                   resource=str(rng.choice(sorted(SENSITIVE))), session=sid, scenario=kind)
        elif kind == "brute_force_success":
            ts = day + pd.Timedelta(hours=int(rng.choice([3, 14])), minutes=int(rng.integers(0, 40)))
            dev, ctry = f"dev-unknown-{int(rng.integers(1, 999))}", str(rng.choice(["RO", "BR", "VN"]))
            for k in range(int(rng.integers(8, 13))):
                ev(ts + pd.Timedelta(seconds=30 * k), u, "signin", "failure", dev, ctry, session=sid, scenario=kind)
            ts += pd.Timedelta(minutes=8)
            ev(ts, u, "signin", "success", dev, ctry, session=sid, scenario=kind)
            ev(ts + pd.Timedelta(seconds=20), u, "mfa", "failure", dev, ctry, session=sid, scenario=kind)
            for k in range(int(rng.integers(4, 9))):
                ev(ts + pd.Timedelta(minutes=2 + 3 * k), u, "resource_access", device=dev, country=ctry,
                   resource=str(rng.choice(ALL_RESOURCES)), session=sid, scenario=kind)
        else:  # privilege_escalation via a hijacked session on a known device
            ts = day + pd.Timedelta(hours=22, minutes=int(rng.integers(10, 50)))
            ev(ts, u, "signin", "success", u["devices"][0], session=sid, scenario=kind)
            ev(ts + pd.Timedelta(minutes=2), u, "role_change", detail="role:admin", approved=False, session=sid, scenario=kind)
            ev(ts + pd.Timedelta(minutes=3), u, "group_add", detail="group:domain-admins", approved=False, session=sid, scenario=kind)
            ev(ts + pd.Timedelta(minutes=4), u, "credential_add", detail="api-key", approved=False, session=sid, scenario=kind)
            for k in range(int(rng.integers(5, 10))):
                ev(ts + pd.Timedelta(minutes=6 + 2 * k), u, "resource_access", device=u["devices"][0],
                   resource=str(rng.choice(["iam-console", "server-admin"])), session=sid, scenario=kind)

    for uid, d in decoys:  # benign but unusual: business travel on a new laptop
        u, day = by_id[uid], t0 + pd.Timedelta(days=d)
        ts = day + pd.Timedelta(hours=u["shift"] + 1, minutes=int(rng.integers(0, 30)))
        dev, ctry, sid = f"dev-travel-{uid}", str(rng.choice(["GB", "AE", "SG"])), new_session()
        ev(ts, u, "signin", "success", dev, ctry, session=sid, scenario="decoy_travel")
        ev(ts + pd.Timedelta(seconds=20), u, "mfa", "success", dev, ctry, session=sid, scenario="decoy_travel")
        for k in range(int(rng.integers(3, 7))):
            ev(ts + pd.Timedelta(minutes=3 + 5 * k), u, "resource_access", device=dev, country=ctry,
               resource=str(rng.choice(ROLE_RESOURCES[u["role"]])), session=sid, scenario="decoy_travel")

    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("timestamp").reset_index(drop=True)
    return df
