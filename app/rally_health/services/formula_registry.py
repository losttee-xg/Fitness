from __future__ import annotations

import json
from typing import Any

from ..repositories.database import SQLiteDatabase


REGISTRY_VERSION = "1.0.0"

# These are raw-observation aggregations, not implementations of Rally's
# described derived formulas.
RAW_MEDIAN_METRICS = {
    "RS-01", "RS-02", "RS-03", "RS-04", "RS-05", "RS-06",
    "RS-10", "HS-05", "AC-09", "SL-02",
}


def describe_metric(metric: dict[str, Any]) -> dict[str, Any]:
    code = str(metric.get("metric_code", ""))
    has_description = bool(metric.get("calculation_description"))
    if code in RAW_MEDIAN_METRICS:
        status = "prototype_raw_aggregation_enabled_not_rally_formula"
        formula_id = "raw_observation_median"
        formula_version = "raw-median-1.0.0"
        executable = True
    elif not has_description:
        status = "review_required_missing_definition"
        formula_id = None
        formula_version = None
        executable = False
    else:
        status = "description_only_not_executable"
        formula_id = None
        formula_version = None
        executable = False
    return {
        **metric,
        "registry_version": REGISTRY_VERSION,
        "formula_registry": {
            "status": status,
            "formula_id": formula_id,
            "formula_version": formula_version,
            "executable": executable,
            "safety_note": "Prototype aggregation only; not a reviewed Rally calculation.",
            "source_ref": metric.get("source"),
        },
    }


def build_registry(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [describe_metric(metric) for metric in metrics]


class FormulaReviewService:
    def __init__(self, db: SQLiteDatabase, catalog_provider):
        self.db = db
        self.catalog_provider = catalog_provider

    def list_reviews(self, status: str | None = None) -> dict[str, Any]:
        catalog = self.catalog_provider()
        with self.db.connect() as conn:
            rows = conn.execute("SELECT metric_code, review_json, status, created_at FROM metric_formula_reviews ORDER BY metric_code").fetchall()
        reviews = {row["metric_code"]: row for row in rows}
        items = []
        for metric in catalog:
            code = metric["metric_code"]
            review = reviews.get(code)
            item_status = review["status"] if review else describe_metric(metric)["formula_registry"]["status"]
            if status is None or item_status == status:
                items.append({"metric_code": code, "name": metric.get("name"), "calculation_description": metric.get("calculation_description"),
                              "status": item_status, "review": json.loads(review["review_json"]) if review else None,
                              "source": metric.get("source")})
        return {"items": items, "count": len(items)}

    def submit_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        code = payload["metric_code"]
        metric = next((item for item in self.catalog_provider() if item["metric_code"].casefold() == code.casefold()), None)
        if metric is None:
            raise KeyError(code)
        if not payload.get("test_cases"):
            raise ValueError("At least one expected-output test case is required.")
        item_status = "PENDING_REVIEW"
        with self.db.connect() as conn:
            conn.execute("INSERT INTO metric_formula_reviews(metric_code,review_json,status) VALUES(?,?,?) ON CONFLICT(metric_code) DO UPDATE SET review_json=excluded.review_json,status=excluded.status,updated_at=CURRENT_TIMESTAMP",
                         (code, json.dumps(payload, ensure_ascii=False), item_status))
        return {"metric_code": code, "status": item_status, "executable": False,
                "message": "Formula proposal saved for domain review; not enabled for computation."}
