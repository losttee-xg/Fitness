from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException

from ..domain.schemas import WorkoutRecord
from ..repositories.database import SQLiteDatabase


class WorkoutSafetyService:
    def __init__(self, db: SQLiteDatabase):
        self.db = db

    def validate_plan(self, exercise_ids: list[str], sex: str, profile: dict[str, Any]) -> dict[str, Any]:
        if sex not in {"female", "male"}:
            raise HTTPException(status_code=422, detail="sex catalog must be female or male")
        with self.db.connect() as conn:
            rows = conn.execute("SELECT payload_json FROM catalog_items WHERE kind=?", (f"workout_{sex}",)).fetchall()
        items = {item["exercise_id"]: item for item in [__import__("json").loads(row[0]) for row in rows]}
        missing = sorted(set(exercise_ids) - items.keys())
        if missing:
            raise HTTPException(status_code=422, detail={"code":"UNKNOWN_EXERCISES", "exercise_ids":missing})
        unreviewed = [item_id for item_id in exercise_ids if items[item_id].get("review_status") != "approved"]
        if unreviewed:
            raise HTTPException(status_code=409, detail={"code":"EXERCISES_NEED_REVIEW", "exercise_ids":unreviewed})
        life_stage = profile.get("life_stage")
        if life_stage:
            excluded = [item_id for item_id in exercise_ids if items[item_id].get("life_stages") and life_stage not in items[item_id]["life_stages"]]
            if excluded:
                raise HTTPException(status_code=422, detail={"code":"LIFE_STAGE_NOT_TAGGED_AS_SUITABLE", "exercise_ids":excluded})
        return {"status":"passed_catalog_checks", "exercise_ids":exercise_ids, "requires_individual_medical_clearance":True}
