"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from time import monotonic
from datetime import datetime, timezone
from pathlib import Path
from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], tuple[float, str, str]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Record a pending input with a monotonic start time."""
        self._open[user_id, request_id] = (
            monotonic(), utc_now_iso(), content_filter(text)["redacted"]
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete an audit entry using privacy-filtered text."""
        now = monotonic()
        started, timestamp, user_text = self._open.pop(
            (user_id, request_id), (now, utc_now_iso(), "")
        )
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "input": user_text,
            "output": content_filter(text)["redacted"],
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((now - started) * 1000, 3),
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
