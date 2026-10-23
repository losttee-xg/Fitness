from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS catalog_items (
    kind TEXT NOT NULL,
    item_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    version TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(kind, item_id)
);
CREATE TABLE IF NOT EXISTS observations (
    observation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    metric_code TEXT NOT NULL,
    metric_variant TEXT,
    aggregation TEXT NOT NULL DEFAULT 'instant',
    value REAL NOT NULL,
    unit TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    received_at TEXT NOT NULL,
    end_at TEXT,
    provenance_json TEXT NOT NULL,
    quality_flags_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_observations_user_metric_time
    ON observations(user_id, metric_code, observed_at);
CREATE TABLE IF NOT EXISTS ingest_batches (
    user_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(user_id, batch_id)
);
CREATE TABLE IF NOT EXISTS ingest_batch_events (
    user_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    sequence_no INTEGER NOT NULL,
    observation_id TEXT,
    status_code INTEGER NOT NULL,
    detail_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(user_id,batch_id,sequence_no),
    FOREIGN KEY(user_id,batch_id) REFERENCES ingest_batches(user_id,batch_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS questionnaire_submissions (
    submission_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    instrument TEXT NOT NULL,
    version TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    answers_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'RECEIVED',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_questionnaire_user_completed ON questionnaire_submissions(user_id, completed_at);
CREATE TABLE IF NOT EXISTS profiles (
    user_id TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS schedule_events (
    event_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    title TEXT NOT NULL,
    starts_at TEXT NOT NULL,
    ends_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_schedule_user_time ON schedule_events(user_id, starts_at, ends_at);
CREATE TABLE IF NOT EXISTS action_audit (
    audit_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    target_id TEXT,
    request_id TEXT,
    authorization_basis TEXT NOT NULL,
    outcome TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS user_memories (
    memory_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    field TEXT NOT NULL,
    value_json TEXT NOT NULL,
    source TEXT NOT NULL,
    confirmed INTEGER NOT NULL DEFAULT 0,
    sensitivity TEXT NOT NULL DEFAULT 'standard',
    expires_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_memories_user ON user_memories(user_id, field);
CREATE TABLE IF NOT EXISTS assistant_conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_assistant_conversations_user ON assistant_conversations(user_id, updated_at);
CREATE TABLE IF NOT EXISTS assistant_messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(conversation_id) REFERENCES assistant_conversations(conversation_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_assistant_messages_conversation ON assistant_messages(user_id, conversation_id, created_at);
CREATE TABLE IF NOT EXISTS action_proposals (
    action_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS metric_formula_reviews (
    metric_code TEXT PRIMARY KEY,
    review_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS report_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_type TEXT NOT NULL,
    byte_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    storage_path TEXT,
    content_sha256 TEXT,
    candidates_json TEXT NOT NULL DEFAULT '[]',
    confirmed_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_report_jobs_user_created ON report_jobs(user_id, created_at);
CREATE TABLE IF NOT EXISTS video_assets (
    asset_id TEXT PRIMARY KEY,
    exercise_id TEXT NOT NULL,
    asset_uri TEXT,
    version TEXT NOT NULL,
    review_status TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class SQLiteDatabase:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(observations)")}
        if "metric_variant" not in columns:
            connection.execute("ALTER TABLE observations ADD COLUMN metric_variant TEXT")
        if "aggregation" not in columns:
            connection.execute("ALTER TABLE observations ADD COLUMN aggregation TEXT NOT NULL DEFAULT 'instant'")
        if "received_at" not in columns:
            connection.execute("ALTER TABLE observations ADD COLUMN received_at TEXT")
            connection.execute("UPDATE observations SET received_at=replace(created_at,' ','T') || '+00:00' WHERE received_at IS NULL")
        questionnaire_columns = {row["name"] for row in connection.execute("PRAGMA table_info(questionnaire_submissions)")}
        report_columns = {row["name"] for row in connection.execute("PRAGMA table_info(report_jobs)")}
        if "storage_path" not in report_columns:
            connection.execute("ALTER TABLE report_jobs ADD COLUMN storage_path TEXT")
        if "content_sha256" not in report_columns:
            connection.execute("ALTER TABLE report_jobs ADD COLUMN content_sha256 TEXT")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def sync_catalog(self, items: list[tuple[str, str, dict]], version: str = "1.0.0") -> int:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM catalog_items")
            db.executemany(
                "INSERT INTO catalog_items(kind,item_id,payload_json,version) VALUES(?,?,?,?)",
                [(kind, item_id, json.dumps(payload, ensure_ascii=False), version) for kind, item_id, payload in items],
            )
            return len(items)
