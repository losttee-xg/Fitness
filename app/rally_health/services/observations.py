from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from ..domain.schemas import AggregationKind, DataSource, ObservationCreate, ObservationOut
from ..repositories.database import SQLiteDatabase
from .normalization import normalize_unit


SUPPORTED_METRICS = {
    "SL-02": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.INSTANT, AggregationKind.INTERVAL}},
    "SL-03": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.INTERVAL}},
    "SL-04": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.INTERVAL}},
    "SL-05": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.INTERVAL}},
    "SL-06": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.INTERVAL}},
    "SL-07": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.INTERVAL}},
    "SL-08": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "SL-09": {"minimum": 0, "maximum": 100, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "SL-10": {"minimum": 0, "maximum": 200, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "SL-11": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "SL-12": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "SL-14": {"minimum": 0, "maximum": 10080, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "SL-15": {"minimum": 0, "maximum": 100, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "SL-16": {"minimum": 0, "maximum": 100, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "SL-19": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.INTERVAL}},
    "SL-20": {"minimum": 0, "maximum": 1440, "aggregation": {AggregationKind.INTERVAL}},
    "RS-01": {"minimum": 20, "maximum": 250, "aggregation": {AggregationKind.INSTANT}},
    "RS-02": {"minimum": 20, "maximum": 250, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "RS-03": {"minimum": 20, "maximum": 250, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "RS-04": {"minimum": 20, "maximum": 250, "aggregation": {AggregationKind.INSTANT, AggregationKind.DAILY_DERIVED}},
    "RS-05": {"minimum": 0, "maximum": 1000, "aggregation": {AggregationKind.INSTANT, AggregationKind.DAILY_DERIVED}},
    "RS-06": {"minimum": 1, "maximum": 80, "aggregation": {AggregationKind.INSTANT, AggregationKind.DAILY_DERIVED}},
    "RS-10": {"minimum": 50, "maximum": 100, "aggregation": {AggregationKind.INSTANT, AggregationKind.DAILY_DERIVED}},
    "AC-09": {"minimum": 0, "maximum": 200000, "aggregation": {AggregationKind.DAILY_DERIVED}},
    "HS-05": {"minimum": 1, "maximum": 500, "aggregation": {AggregationKind.INSTANT}},
}

REJECT_FLAGS = {"missing_value", "off_wrist", "low_confidence", "signal_unavailable"}


class ObservationService:
    def __init__(self, db: SQLiteDatabase):
        self.db = db

    def normalize(self, observation: ObservationCreate) -> ObservationCreate:
        if observation.metric_code == "RS-05" and observation.unit.strip().casefold() in {"s", "sec", "second", "seconds"}:
            value, unit = observation.value * 1000, "ms"
        else:
            result = normalize_unit(observation.metric_code, observation.value, observation.unit)
            if result is None:
                raise HTTPException(status_code=422, detail={"code": "UNSUPPORTED_UNIT", "metric_code": observation.metric_code, "unit": observation.unit})
            value, unit = result
        return observation.model_copy(update={
            "value": value, "unit": unit,
            "observed_at": observation.observed_at.astimezone(timezone.utc),
            "received_at": observation.received_at.astimezone(timezone.utc),
            "end_at": observation.end_at.astimezone(timezone.utc) if observation.end_at else None,
            "metric_variant": observation.metric_variant.upper() if observation.metric_variant else None,
        })

    def validate(self, observation: ObservationCreate) -> list[str]:
        spec = SUPPORTED_METRICS.get(observation.metric_code)
        if spec is None:
            return ["metric_not_enabled_for_ingestion"]
        flags: list[str] = []
        if observation.metric_code == "RS-05" and observation.metric_variant not in {"SDNN", "RMSSD"}:
            flags.append("hrv_variant_required" if observation.metric_variant is None else "hrv_variant_unknown")
        elif observation.metric_code != "RS-05" and observation.metric_variant is not None:
            flags.append("variant_not_applicable")
        if observation.aggregation not in spec["aggregation"]:
            flags.append("aggregation_not_supported_for_metric")
        if not spec["minimum"] <= observation.value <= spec["maximum"]:
            flags.append("value_out_of_range")
        delta = (observation.observed_at - observation.received_at).total_seconds()
        if delta > 300:
            flags.append("timestamp_in_future")
        elif delta > 0:
            observation.quality_flags = sorted(set(observation.quality_flags + ["clock_skew_suspected"]))
        if "relative_seconds_from_psg_start_not_utc" in observation.quality_flags and observation.provenance.source_type != DataSource.RESEARCH_DATASET:
            flags.append("relative_timestamp_requires_research_source")
        if observation.end_at and observation.end_at < observation.observed_at:
            flags.append("end_before_start")
        if observation.aggregation == AggregationKind.INTERVAL:
            if observation.end_at is None:
                flags.append("end_at_required_for_interval")
            elif observation.end_at == observation.observed_at:
                flags.append("zero_duration_interval")
        flags.extend(flag for flag in observation.quality_flags if flag in REJECT_FLAGS)
        return sorted(set(flags))

    def add(self, user_id: str, observation: ObservationCreate) -> ObservationOut:
        normalized = self.normalize(observation)
        flags = self.validate(normalized)
        if flags:
            raise HTTPException(status_code=422, detail={"code": "INVALID_OBSERVATION", "quality_flags": flags})
        record = ObservationOut(**normalized.model_dump(), user_id=user_id)
        with self.db.connect() as conn:
            return self.add_in_transaction(conn, user_id, normalized)

    def add_in_transaction(self, conn, user_id: str, observation: ObservationCreate) -> ObservationOut:
        normalized = self.normalize(observation)
        flags = self.validate(normalized)
        if flags:
            raise HTTPException(status_code=422, detail={"code": "INVALID_OBSERVATION", "quality_flags": flags})
        record = ObservationOut(**normalized.model_dump(), user_id=user_id)
        if record.provenance.sample_id:
            duplicate = conn.execute(
                "SELECT observation_id,value,unit,metric_code,observed_at FROM observations WHERE user_id=? AND json_extract(provenance_json,'$.provider')=? AND json_extract(provenance_json,'$.sample_id')=? LIMIT 1",
                (user_id, record.provenance.provider, record.provenance.sample_id),
            ).fetchone()
            if duplicate:
                same_payload = (duplicate["metric_code"] == record.metric_code and duplicate["value"] == record.value
                                and duplicate["unit"] == record.unit and duplicate["observed_at"] == record.observed_at.isoformat())
                code = "DUPLICATE_OBSERVATION" if same_payload else "SOURCE_ID_REUSED_WITH_DIFFERENT_PAYLOAD"
                raise HTTPException(status_code=409, detail={"code": code, "observation_id": duplicate["observation_id"]})
        duplicate = conn.execute(
            "SELECT observation_id FROM observations WHERE user_id=? AND metric_code=? AND COALESCE(metric_variant,'')=COALESCE(?,'') AND value=? AND unit=? AND observed_at=? AND json_extract(provenance_json,'$.source_id') IS ? LIMIT 1",
            (user_id, record.metric_code, record.metric_variant, record.value, record.unit,
             record.observed_at.isoformat(), record.provenance.source_id),
        ).fetchone()
        if duplicate:
            raise HTTPException(status_code=409, detail={"code": "DUPLICATE_OBSERVATION", "observation_id": duplicate["observation_id"]})
        if record.aggregation == AggregationKind.INSTANT:
            conflict_rows = conn.execute(
                "SELECT value,provenance_json FROM observations WHERE user_id=? AND metric_code=? AND COALESCE(metric_variant,'')=COALESCE(?,'') AND aggregation='instant' AND ABS(julianday(observed_at)-julianday(?))*86400<=300",
                (user_id, record.metric_code, record.metric_variant, record.observed_at.isoformat()),
            ).fetchall()
            for row in conflict_rows:
                previous = json.loads(row["provenance_json"])
                current_group = record.provenance.provider or record.provenance.source_type.value
                previous_group = previous.get("provider") or previous.get("source_type")
                if current_group != previous_group and abs(float(row["value"]) - record.value) >= 10:
                    record.quality_flags = sorted(set(record.quality_flags + ["source_conflict_review_required"]))
                    break
        conn.execute(
            "INSERT INTO observations(observation_id,user_id,metric_code,metric_variant,aggregation,value,unit,observed_at,received_at,end_at,provenance_json,quality_flags_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (record.observation_id,user_id,record.metric_code,record.metric_variant,record.aggregation.value,
             record.value,record.unit,record.observed_at.isoformat(),record.received_at.isoformat(),
             record.end_at.isoformat() if record.end_at else None,record.provenance.model_dump_json(),json.dumps(record.quality_flags)),
        )
        return record

    def add_batch(self, user_id: str, provider: str, batch_id: str, observations: list[ObservationCreate]) -> dict:
        stable_payload = [item.model_dump(mode="json", exclude={"observation_id", "received_at"}) for item in observations]
        request_payload = json.dumps({"provider": provider, "observations": stable_payload}, sort_keys=True, ensure_ascii=False)
        request_hash = hashlib.sha256(request_payload.encode("utf-8")).hexdigest()
        created, rejected = [], []
        batch_received_at = datetime.now(timezone.utc)
        with self.db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT request_hash,response_json FROM ingest_batches WHERE user_id=? AND batch_id=?", (user_id,batch_id)).fetchone()
            if existing:
                if existing["request_hash"] != request_hash:
                    raise HTTPException(status_code=409, detail={"code":"BATCH_ID_REUSED_WITH_DIFFERENT_PAYLOAD"})
                saved = json.loads(existing["response_json"])
                if saved.get("status") == "PROCESSING":
                    conn.execute("DELETE FROM observations WHERE user_id=? AND json_extract(provenance_json,'$.import_batch_id')=?",
                                 (user_id,batch_id))
                    conn.execute("DELETE FROM ingest_batches WHERE user_id=? AND batch_id=?", (user_id,batch_id))
                else:
                    return saved | {"replayed": True}
            conn.execute("INSERT INTO ingest_batches(user_id,batch_id,provider,request_hash,response_json) VALUES(?,?,?,?,?)",
                         (user_id,batch_id,provider,request_hash,json.dumps({"status":"PROCESSING"})))
            for index, item in enumerate(observations):
                provenance = item.provenance.model_copy(update={"provider": item.provenance.provider or provider,
                                                               "import_batch_id": item.provenance.import_batch_id or batch_id})
                candidate = item.model_copy(update={"provenance": provenance, "received_at": batch_received_at})
                try:
                    created.append(self.add_in_transaction(conn, user_id, candidate).model_dump(mode="json"))
                    event = (index, created[-1]["observation_id"], 200, {"status":"accepted"})
                except HTTPException as error:
                    rejected.append({"index": index, "status_code": error.status_code, "detail": error.detail})
                    event = (index, None, error.status_code, error.detail)
                conn.execute("INSERT INTO ingest_batch_events(user_id,batch_id,sequence_no,observation_id,status_code,detail_json) VALUES(?,?,?,?,?,?)",
                             (user_id,batch_id,event[0],event[1],event[2],json.dumps(event[3],ensure_ascii=False)))
            response = {"batch_id":batch_id,"provider":provider,"received_count":len(observations),
                        "accepted_count":len(created),"rejected_count":len(rejected),
                        "observations":created,"rejections":rejected,"replayed":False}
            conn.execute("UPDATE ingest_batches SET response_json=? WHERE user_id=? AND batch_id=?",
                         (json.dumps(response,ensure_ascii=False),user_id,batch_id))
        return response

    def history(self, user_id: str, metric_code: str, since: datetime, until: datetime | None = None,
                metric_variant: str | None = None) -> list[ObservationOut]:
        until = until or datetime.now(timezone.utc)
        with self.db.connect() as conn:
            rows = conn.execute("SELECT * FROM observations WHERE user_id=? AND metric_code=? AND COALESCE(metric_variant,'')=COALESCE(?,'') AND observed_at>=? AND observed_at<=? ORDER BY observed_at",
                                (user_id,metric_code,metric_variant,since.isoformat(),until.isoformat())).fetchall()
        return [ObservationOut(observation_id=row["observation_id"],user_id=row["user_id"],metric_code=row["metric_code"],
                               metric_variant=row["metric_variant"],aggregation=row["aggregation"],value=row["value"],unit=row["unit"],
                               observed_at=datetime.fromisoformat(row["observed_at"]),
                               received_at=datetime.fromisoformat(row["received_at"]) if row["received_at"] else datetime.fromisoformat(row["created_at"]).replace(tzinfo=timezone.utc),
                               end_at=datetime.fromisoformat(row["end_at"]) if row["end_at"] else None,
                               provenance=json.loads(row["provenance_json"]),quality_flags=json.loads(row["quality_flags_json"])) for row in rows]

    def resolve_source_conflicts(self, user_id: str, metric_code: str, metric_variant: str | None = None) -> dict:
        observations = self.history(user_id,metric_code,datetime.min.replace(tzinfo=timezone.utc),metric_variant=metric_variant)
        conflicts=[item for item in observations if "source_conflict_review_required" in item.quality_flags]
        return {"status":"REVIEW_REQUIRED" if conflicts else "NO_CONFLICTS","metric_code":metric_code,
                "conflict_observation_ids":[item.observation_id for item in conflicts],
                "resolution":"No source precedence is assumed; review the competing observations."}
