from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import HTTPException

from ..domain.schemas import ScheduleEventCreate
from ..repositories.database import SQLiteDatabase


class ScheduleService:
    def __init__(self, db: SQLiteDatabase):
        self.db = db

    def create(self, user_id: str, event: ScheduleEventCreate, auto_grant: bool = False) -> dict:
        if not event.requested_by_user and not auto_grant:
            raise HTTPException(status_code=403, detail="Explicit request or scoped Auto permission required")
        if event.ends_at <= event.starts_at:
            raise HTTPException(status_code=422, detail="ends_at must be later than starts_at")
        authorization_basis = "user_request" if event.requested_by_user else "auto_grant"
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            duplicate = conn.execute("SELECT * FROM schedule_events WHERE user_id=? AND idempotency_key=?", (user_id, event.idempotency_key)).fetchone()
            if duplicate:
                return json.loads(duplicate["payload_json"])
            overlap = conn.execute(
                "SELECT event_id FROM schedule_events WHERE user_id=? AND starts_at < ? AND ends_at > ?",
                (user_id, event.ends_at.isoformat(), event.starts_at.isoformat()),
            ).fetchone()
            if overlap:
                raise HTTPException(status_code=409, detail={"code":"SCHEDULE_CONFLICT", "conflicting_event_id":overlap["event_id"]})
            payload = event.model_dump(mode="json") | {"user_id": user_id, "created_at": datetime.now(timezone.utc).isoformat()}
            conn.execute("INSERT INTO schedule_events(event_id,user_id,idempotency_key,title,starts_at,ends_at,payload_json) VALUES(?,?,?,?,?,?,?)",
                         (event.event_id, user_id, event.idempotency_key, event.title, event.starts_at.isoformat(), event.ends_at.isoformat(), json.dumps(payload)))
            conn.execute("INSERT INTO action_audit(audit_id,user_id,action_type,target_id,authorization_basis,outcome,metadata_json) VALUES(?,?,?,?,?,?,?)",
                         (str(uuid4()), user_id, "schedule_create", event.event_id, authorization_basis, "committed", json.dumps({"idempotency_key":event.idempotency_key})))
            return payload

    def list(self, user_id: str, since: datetime, until: datetime) -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT payload_json FROM schedule_events WHERE user_id=? AND starts_at>=? AND starts_at<? ORDER BY starts_at", (user_id, since.isoformat(), until.isoformat())).fetchall()
        return [json.loads(row[0]) for row in rows]
