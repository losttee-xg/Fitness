from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..domain.schemas import ConfidenceLevel, MetricResult
from .metrics import MetricEngine
from .observations import ObservationService


class InsightService:
    def __init__(self, observations: ObservationService, metrics: MetricEngine):
        self.observations = observations
        self.metrics = metrics

    def daily(self, user_id: str, metric_codes: list[str]) -> dict[str, Any]:
        facts = []
        for code in metric_codes[:20]:
            result = self.metrics.compute(user_id, code, window_days=7)
            if result.value is None:
                continue
            baseline = self.metrics.baseline(user_id, code, window_days=28, minimum_samples=7)
            delta = None
            if baseline.get("status") == "READY" and baseline.get("value") not in (None, 0):
                delta = (result.value - baseline["value"]) / abs(baseline["value"]) * 100
            facts.append({
                "metric_code": code,
                "value": result.value,
                "unit": result.unit,
                "status": result.status.value,
                "confidence": result.confidence_level.value,
                "baseline": baseline.get("value"),
                "delta_percent": round(delta, 1) if delta is not None else None,
                "source_ids": result.source_observation_ids,
                "interpretation": "descriptive_only_no_causal_claim",
            })
        facts.sort(key=lambda x: abs(x["delta_percent"] or 0), reverse=True)
        return {"status": "ok" if facts else "insufficient_data", "insights": facts[:3], "limitations": [] if facts else ["Not enough observations to produce a personal insight."]}

    def trend(self, user_id: str, metric_code: str, days: int = 30) -> dict[str, Any]:
        until = datetime.now(timezone.utc)
        observations = self.observations.history(user_id, metric_code, until - timedelta(days=days), until)
        daily: dict[str, list[float]] = {}
        for item in observations:
            daily.setdefault(item.observed_at.date().isoformat(), []).append(item.value)
        series = [{"date": day, "value": sum(values) / len(values), "sample_count": len(values)} for day, values in sorted(daily.items())]
        return {"metric_code": metric_code, "window_days": days, "series": series, "valid_days": len(series), "status": "ok" if len(series) >= 3 else "insufficient_data"}
