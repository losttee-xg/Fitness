from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException

from ..domain.schemas import ScheduleEventCreate
from ..repositories.database import SQLiteDatabase
from .workouts import WorkoutCatalog


class CoachService:
    def __init__(self, catalog: WorkoutCatalog, db: SQLiteDatabase):
        self.catalog = catalog
        self.db = db

    def build_draft(self, user_id: str, sex: str, exercise_ids: list[str], starts_at: datetime,
                    duration_minutes: int, title: str, availability: list[str] | None = None) -> dict[str, Any]:
        if duration_minutes < 5 or duration_minutes > 240:
            raise HTTPException(status_code=422, detail="duration_minutes must be between 5 and 240")
        if not exercise_ids:
            raise HTTPException(status_code=422, detail="At least one catalog exercise is required")
        exercises = self.catalog.require_exercise_ids(exercise_ids, sex)
        if any(item.get("review_status") != "approved" for item in exercises):
            warnings = ["One or more selected exercises are not approved for automated plan generation."]
        else:
            warnings = []
        ends_at = starts_at + timedelta(minutes=duration_minutes)
        event = ScheduleEventCreate(title=title, sex_catalog=sex, exercise_ids=exercise_ids, starts_at=starts_at, ends_at=ends_at,
                                    requested_by_user=True, idempotency_key=f"draft:{user_id}:{starts_at.isoformat()}:{','.join(exercise_ids)}")
        with self.db.connect() as conn:
            overlap = conn.execute("SELECT event_id FROM schedule_events WHERE user_id=? AND starts_at<? AND ends_at>?",
                                   (user_id, ends_at.isoformat(), starts_at.isoformat())).fetchone()
        if overlap:
            warnings.append(f"Potential schedule conflict with {overlap['event_id']}.")
        return {"status": "DRAFT", "event": event.model_dump(mode="json"), "exercises": exercises,
                "warnings": warnings, "commit_required": True, "availability_supplied": bool(availability)}
