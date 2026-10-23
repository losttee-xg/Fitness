from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

from ..repositories.database import SQLiteDatabase


class MemoryService:
    def __init__(self, db: SQLiteDatabase):
        self.db = db

    def list(self, user_id: str) -> list[dict]:
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM user_memories WHERE user_id=? ORDER BY updated_at DESC", (user_id,)).fetchall()
        return [{"memory_id": row["memory_id"], "field": row["field"], "value": json.loads(row["value_json"]),
                 "source": row["source"], "confirmed": bool(row["confirmed"]), "sensitivity": row["sensitivity"],
                 "expires_at": row["expires_at"], "updated_at": row["updated_at"]} for row in rows]

    def upsert(self, user_id: str, field: str, value: dict | list | str | int | float | bool,
               source: str, confirmed: bool) -> dict:
        memory_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self.db.connect() as conn:
            conn.execute("INSERT INTO user_memories(memory_id,user_id,field,value_json,source,confirmed,updated_at) VALUES(?,?,?,?,?,?,?)",
                         (memory_id, user_id, field, json.dumps(value, ensure_ascii=False), source, int(confirmed), now))
        return {"memory_id": memory_id, "field": field, "value": value, "confirmed": confirmed}

    def delete(self, user_id: str, memory_id: str) -> bool:
        with self.db.connect() as conn:
            cursor = conn.execute("DELETE FROM user_memories WHERE user_id=? AND memory_id=?", (user_id, memory_id))
        return cursor.rowcount == 1
