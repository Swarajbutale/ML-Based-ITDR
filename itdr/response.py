"""Policy-based response orchestration + audit trail.

The model score never triggers an action on its own: a documented policy maps
(risk tier, corroboration, asset criticality) to a PROPORTIONATE action, and
high-impact actions require human approval. Execution is DRY-RUN by default; the
`execute` function is the single place to plug in a real IdP API (Okta, Entra ID,
Keycloak...) using a least-privilege service identity and an allowlist.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

ALLOWED_ACTIONS = {
    "monitor", "step_up_auth", "analyst_review",
    "revoke_sessions", "temporary_restriction", "escalate",
}
PRIVILEGED_ROLES = {"admin"}


def corroborated(row: pd.Series) -> bool:
    """High-impact action needs independent evidence: a rule hit, or ML agreement."""
    return bool(row["rules_fired"]) or (row["anomaly_score"] >= 0.90 and row["supervised_score"] >= 0.50)


def decide(row: pd.Series) -> dict:
    critical = row["role"] in PRIVILEGED_ROLES or row["sensitive_access_count"] > 0
    tier, corr = row["tier"], corroborated(row)

    if tier == "low":
        action, rule, approval = "monitor", "P1: low risk -> log and monitor", False
    elif tier == "medium":
        if row["new_device"] or row["new_country"] or row["success_after_fail"]:
            action, rule, approval = "step_up_auth", "P2: medium risk, sign-in anomaly -> step-up authentication", False
        else:
            action, rule, approval = "analyst_review", "P3: medium risk -> analyst review", False
    else:  # high
        if not corr:
            action, rule, approval = "escalate", "P4: high risk but uncorroborated -> escalate to human review only", False
        elif row["unapproved_change_count"] > 0:
            action, rule, approval = ("temporary_restriction",
                                      "P5: high risk + unapproved entitlement change -> temporarily restrict sensitive access", True)
        else:
            action, rule, approval = "revoke_sessions", "P6: high risk, corroborated -> revoke active sessions", critical
    assert action in ALLOWED_ACTIONS
    return dict(
        action=action, policy_rule=rule, requires_approval=approval, corroborated=corr,
        critical_asset=bool(critical),
        reversible=True,  # all allowlisted actions can be undone (user re-authenticates, restriction lifted)
        rationale=f"tier={tier}, corroborated={corr}, critical_asset={bool(critical)}",
    )


def execute(action: str, user_id: str, dry_run: bool = True) -> str:
    """Single integration point. Real calls (e.g. IdP 'revoke sessions') go here."""
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Action '{action}' is not on the allowlist")
    if dry_run:
        return f"DRY-RUN: would perform '{action}' for {user_id}"
    raise NotImplementedError("Connect an IdP API with a least-privilege service identity here.")


def audit_entry(alert_id: str, row: pd.Series, decision: dict, model_version: str,
                actor: str, status: str, outcome: str, approver: str | None = None,
                feedback: str | None = None) -> dict:
    return dict(
        time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        alert_id=alert_id, user_id=row["user_id"], risk=round(float(row["risk"]), 3), tier=row["tier"],
        rules_fired="; ".join(row["rules_fired"]), model_version=model_version,
        policy_rule=decision["policy_rule"], action=decision["action"], actor=actor,
        approver=approver or "", status=status, outcome=outcome, analyst_feedback=feedback or "",
    )


def audit_to_jsonl(entries: list[dict]) -> str:
    return "\n".join(json.dumps(e) for e in entries)
