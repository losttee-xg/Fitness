from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from ..repositories.database import SQLiteDatabase


class WorkoutCatalog:
    def __init__(self, db: SQLiteDatabase):
        self.db = db

    def search(self, sex: str, query: str | None = None, equipment: list[str] | None = None,
               life_stage: str | None = None, cycle_phase: str | None = None,
               allowed_ids: set[str] | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if sex not in {"female", "male"}:
            raise ValueError("sex must be female or male")
        with self.db.connect() as conn:
            rows = conn.execute("SELECT payload_json FROM catalog_items WHERE kind=?", (f"workout_{sex}",)).fetchall()
        items = [json.loads(row[0]) for row in rows]
        q = query.casefold().strip() if query else None
        equipment = [value.casefold() for value in (equipment or [])]
        output = []
        for item in items:
            if allowed_ids is not None and item["exercise_id"] not in allowed_ids:
                continue
            searchable = " ".join(str(item.get(key) or "") for key in ("name", "name_vi", "name_en", "description", "category_ids")).casefold()
            if q and q not in searchable:
                continue
            tags = [value.casefold() for value in item.get("equipment", [])]
            if equipment and not all(any(req in tag for tag in tags) for req in equipment):
                continue
            stages = [str(value).casefold() for value in item.get("life_stages", [])]
            if life_stage and life_stage.casefold() not in stages:
                continue
            phases = [str(value).casefold() for value in item.get("cycle_phases", [])]
            if cycle_phase and phases and cycle_phase.casefold() not in phases:
                continue
            if item.get("review_status") == "blocked":
                continue
            output.append(item)
            if len(output) >= limit:
                break
        return output

    def require_exercise_ids(self, exercise_ids: list[str], sex: str) -> list[dict[str, Any]]:
        wanted = set(exercise_ids)
        found = self.search(sex=sex, limit=500)
        index = {item["exercise_id"]: item for item in found}
        missing = wanted - index.keys()
        if missing:
            raise HTTPException(status_code=422, detail={"code":"UNKNOWN_EXERCISES", "exercise_ids":sorted(missing)})
        return [index[item] for item in exercise_ids]
