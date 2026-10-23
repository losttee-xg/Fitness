from __future__ import annotations

import json
from typing import Any

from ..repositories.database import SQLiteDatabase


class PredictiveService:
    """Starter rule interface; emits candidates only, not medical alerts or forecasts."""

    def __init__(self, db: SQLiteDatabase):
        self.db = db

    def anomaly_candidates(self, user_id: str, metric_codes: list[str]) -> dict[str, Any]:
        candidates = []
        for code in metric_codes:
            with self.db.connect() as conn:
                rows = conn.execute("SELECT value, observed_at FROM observations WHERE user_id=? AND metric_code=? ORDER BY observed_at DESC LIMIT 8", (user_id, code)).fetchall()
            if len(rows) < 6:
                continue
            candidates.append({"metric_code": code, "candidate": False, "status": "needs_validated_rule", "sample_count": len(rows)})
        return {"status": "candidate_only", "items": candidates,
                "limitations": ["No approved anomaly thresholds or labeled ground truth are configured; no proactive alerts are sent."]}

    def forecast(self, user_id: str, metric_code: str) -> dict[str, Any]:
        with self.db.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM observations WHERE user_id=? AND metric_code=?", (user_id, metric_code)).fetchone()[0]
        return {"status": "not_available", "metric_code": metric_code, "history_count": count,
                "forecast": None, "reason": "Forecast model and independent error targets have not been validated."}
