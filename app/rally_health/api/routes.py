from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, UploadFile, File

from ..core.auth import get_user_id, require_api_key
from ..core.config import get_settings
from ..domain.schemas import ObservationCreate, ScheduleEventCreate, ReportCandidateConfirm, RallyMetricReview
from ..integrations.claude_provider import ClaudeProvider
from ..integrations.gpt4o_chat import GPT4oChatProvider
from ..repositories.database import SQLiteDatabase
from ..services.coach import CoachService
from ..services.insights import InsightService
from ..services.memory import MemoryService
from ..services.metrics import MetricEngine
from ..services.formula_registry import FormulaReviewService, build_registry
from ..services.observations import ObservationService
from ..services.predictive import PredictiveService
from ..services.schedule import ScheduleService
from ..services.sleep import summarize_sleep_intervals
from ..services.safety import WorkoutSafetyService
from ..services.workouts import WorkoutCatalog
from ..importer import import_workbooks

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])
settings = get_settings()
db = SQLiteDatabase(settings.database_path)
observations = ObservationService(db)
metrics = MetricEngine(observations, lambda: _catalog_metrics())
workouts = WorkoutCatalog(db)
schedule = ScheduleService(db)
memories = MemoryService(db)
insights = InsightService(observations, metrics)
predictive = PredictiveService(db)
safety = WorkoutSafetyService(db)
formula_reviews = FormulaReviewService(db, lambda: _catalog_metrics())
claude = ClaudeProvider(settings.anthropic_enabled, settings.anthropic_model, settings.anthropic_api_key)
gpt4o_chat = GPT4oChatProvider(settings)


def set_runtime(runtime_settings, runtime_db) -> None:
    global settings, db, observations, metrics, workouts, schedule, memories, insights, predictive, claude, gpt4o_chat, safety, formula_reviews
    settings = runtime_settings
    db = runtime_db
    observations = ObservationService(db)
    metrics = MetricEngine(observations, lambda: _catalog_metrics())
    workouts = WorkoutCatalog(db)
    schedule = ScheduleService(db)
    memories = MemoryService(db)
    insights = InsightService(observations, metrics)
    predictive = PredictiveService(db)
    safety = WorkoutSafetyService(db)
    formula_reviews = FormulaReviewService(db, lambda: _catalog_metrics())
    claude = ClaudeProvider(settings.anthropic_enabled, settings.anthropic_model, settings.anthropic_api_key)
    gpt4o_chat = GPT4oChatProvider(settings)


def _catalog_metrics() -> list[dict[str, Any]]:
    path = settings.normalized_data_dir / "rally.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="Catalog not imported. Run the workbook importer.")
    import json
    return json.loads(path.read_text(encoding="utf-8"))["metrics"]


@router.get("/catalog/metrics")
def list_metrics(group: str | None = None, evidence_level: str | None = None) -> dict[str, Any]:
    items = build_registry(_catalog_metrics())
    if group:
        items = [item for item in items if (item.get("group") or "").casefold() == group.casefold()]
    if evidence_level:
        items = [item for item in items if item.get("evidence_level") == evidence_level.upper()]
    return {"items": items, "count": len(items), "computed_values": False,
            "registry_version": "1.0.0",
            "formula_status_counts": {
                status: sum(item["formula_registry"]["status"] == status for item in items)
                for status in sorted({item["formula_registry"]["status"] for item in items})
            }}


@router.get("/metrics/{metric_code}/definition")
def metric_definition(metric_code: str) -> dict[str, Any]:
    result = next((item for item in build_registry(_catalog_metrics()) if item["metric_code"].casefold() == metric_code.casefold()), None)
    if result is None:
        raise HTTPException(status_code=404, detail="Metric code not found")
    return result


@router.post("/observations")
def add_observation(observation: ObservationCreate, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    result = observations.add(user_id, observation)
    return result.model_dump(mode="json")


@router.get("/observations/conflicts/{metric_code}")
def observation_conflicts(metric_code: str, metric_variant: str | None = None,
                          user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    return observations.resolve_source_conflicts(user_id, metric_code, metric_variant)


@router.get("/metrics/{metric_code}/result")
def metric_result(metric_code: str, window_days: int = Query(default=28, ge=1, le=365),
                  metric_variant: str | None = None, timezone_name: str = Query(default="UTC", alias="timezone"),
                  user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    import json
    with db.connect() as conn:
        profile_row = conn.execute("SELECT profile_json FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    if timezone_name == "UTC" and profile_row:
        try:
            timezone_name = json.loads(profile_row["profile_json"]).get("timezone", "UTC")
        except json.JSONDecodeError:
            timezone_name = "UTC"
    return metrics.compute(user_id, metric_code, window_days, metric_variant, timezone_name).model_dump(mode="json")


@router.get("/metrics/{metric_code}/sleep-summary")
def sleep_summary(night: str = Query(..., description="Local date in YYYY-MM-DD"),
                 timezone_name: str = Query(default="UTC", alias="timezone"),
                 user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    from datetime import date, time
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        target_date = date.fromisoformat(night)
        user_timezone = timezone.utc if timezone_name == "UTC" else ZoneInfo(timezone_name)
    except (ValueError, ZoneInfoNotFoundError) as error:
        raise HTTPException(status_code=422, detail="night/timezone is invalid") from error
    local_start = datetime.combine(target_date, time(12, 0), tzinfo=user_timezone)
    local_end = datetime.combine(target_date + timedelta(days=1), time(12, 0), tzinfo=user_timezone)
    start_at = local_start.astimezone(timezone.utc)
    end_at = local_end.astimezone(timezone.utc)
    observations_for_night=[]
    for code in ("SL-02","SL-04","SL-05","SL-06","SL-07"):
        observations_for_night.extend(observations.history(user_id,code,start_at,end_at))
    return summarize_sleep_intervals(observations_for_night,target_date,timezone_name)


@router.get("/metrics/{metric_code}/baseline")
def metric_baseline(metric_code: str, window_days: int = Query(default=28, ge=7, le=365),
                   minimum_valid_days: int = Query(default=7, ge=1, le=365), metric_variant: str | None = None,
                   timezone_name: str = Query(default="UTC", alias="timezone"),
                   user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    import json
    with db.connect() as conn:
        profile_row = conn.execute("SELECT profile_json FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    if timezone_name == "UTC" and profile_row:
        try:
            timezone_name = json.loads(profile_row["profile_json"]).get("timezone", "UTC")
        except json.JSONDecodeError:
            timezone_name = "UTC"
    return metrics.baseline(user_id, metric_code, window_days, minimum_valid_days=minimum_valid_days,
                            metric_variant=metric_variant, timezone_name=timezone_name)




@router.get("/insights/daily")
def daily_insight(metric_codes: list[str] = Query(default=[]), user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    return insights.daily(user_id, metric_codes)


@router.get("/metrics/{metric_code}/trend")
def metric_trend(metric_code: str, days: int = Query(default=30, ge=7, le=90), user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    return insights.trend(user_id, metric_code, days)


@router.get("/catalog/workouts")
def search_workouts(sex: str = Query(pattern="^(female|male)$"), query: str | None = None,
                    equipment: list[str] = Query(default=[]), life_stage: str | None = None,
                    cycle_phase: str | None = None, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    items = workouts.search(sex, query, equipment, life_stage, cycle_phase, limit=limit)
    return {"items": items, "count": len(items), "safety_state": "catalog_tags_are_not_individual_medical_clearance"}


@router.post("/coach/drafts")
def create_plan_draft(payload: dict[str, Any], user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    required = {"sex", "exercise_ids", "starts_at", "duration_minutes", "title"}
    missing = required - payload.keys()
    if missing:
        raise HTTPException(status_code=422, detail={"code":"MISSING_PLAN_FIELDS", "missing":sorted(missing)})
    from datetime import datetime
    try:
        starts_at = datetime.fromisoformat(payload["starts_at"])
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="starts_at must be ISO-8601 with timezone")
    if starts_at.tzinfo is None:
        raise HTTPException(status_code=422, detail="starts_at must include timezone")
    validation = safety.validate_plan(payload["exercise_ids"], payload["sex"], payload.get("profile", {}))
    return CoachService(workouts, db).build_draft(user_id, payload["sex"], payload["exercise_ids"], starts_at,
                                                  int(payload["duration_minutes"]), payload["title"], payload.get("availability")) | {"safety_validation": validation}


@router.post("/schedule/events/proposals")
def propose_schedule_event(event: ScheduleEventCreate, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    if not event.requested_by_user:
        raise HTTPException(status_code=403, detail="An explicit user request is required to create a schedule proposal")
    if event.ends_at <= event.starts_at:
        raise HTTPException(status_code=422, detail="ends_at must be later than starts_at")
    plan = CoachService(workouts, db).build_draft(user_id, event.sex_catalog, event.exercise_ids,
                                                  event.starts_at, int((event.ends_at-event.starts_at).total_seconds()/60), event.title)
    if plan.get("warnings"):
        raise HTTPException(status_code=409, detail={"code":"PLAN_REVIEW_REQUIRED", "warnings":plan["warnings"]})
    action_id = str(uuid4())
    import json
    from datetime import datetime, timedelta, timezone
    payload = event.model_dump(mode="json")
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    with db.connect() as conn:
        conn.execute("INSERT INTO action_proposals(action_id,user_id,action_type,payload_json,status,expires_at) VALUES(?,?,?,?,?,?)",
                     (action_id, user_id, "schedule_create", json.dumps(payload), "AWAITING_CONFIRMATION", expires_at))
    return {"status":"AWAITING_CONFIRMATION", "action_id":action_id, "event":payload,
            "expires_at":expires_at, "committed":False}


@router.post("/schedule/events/proposals/{action_id}/confirm")
def confirm_schedule_proposal(action_id: str, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    import json
    from datetime import datetime, timezone
    with db.connect() as conn:
        proposal = conn.execute("SELECT * FROM action_proposals WHERE action_id=? AND user_id=?", (action_id, user_id)).fetchone()
        if proposal is None:
            raise HTTPException(status_code=404, detail="Action proposal not found")
        if proposal["status"] != "AWAITING_CONFIRMATION":
            raise HTTPException(status_code=409, detail="Action proposal is not awaiting confirmation")
        if datetime.fromisoformat(proposal["expires_at"]) < datetime.now(timezone.utc):
            conn.execute("UPDATE action_proposals SET status='EXPIRED' WHERE action_id=?", (action_id,))
            raise HTTPException(status_code=410, detail="Action proposal expired")
        event = ScheduleEventCreate(**json.loads(proposal["payload_json"]))
    result = schedule.create(user_id, event, auto_grant=False)
    with db.connect() as conn:
        conn.execute("UPDATE action_proposals SET status='COMMITTED' WHERE action_id=? AND user_id=?", (action_id, user_id))
    return result


@router.post("/schedule/events")
def create_schedule_event_direct() -> dict[str, Any]:
    raise HTTPException(status_code=405, detail="Direct schedule writes are disabled. Create a proposal and confirm it in the app.")


@router.get("/schedule/events")
def list_schedule_events(since: str, until: str, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    from datetime import datetime
    try:
        start, end = datetime.fromisoformat(since), datetime.fromisoformat(until)
    except ValueError:
        raise HTTPException(status_code=422, detail="since/until must be ISO-8601 timestamps")
    if start.tzinfo is None or end.tzinfo is None:
        raise HTTPException(status_code=422, detail="since/until require timezone")
    return {"items": schedule.list(user_id, start, end)}


@router.get("/memories")
def list_memories(user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    return {"items": memories.list(user_id)}


@router.post("/memories/proposals")
def propose_memory(payload: dict[str, Any], user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    if not payload.get("field") or "value" not in payload:
        raise HTTPException(status_code=422, detail="field and value are required")
    import json
    from datetime import datetime, timedelta, timezone
    action_id = str(uuid4())
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    with db.connect() as conn:
        conn.execute("INSERT INTO action_proposals(action_id,user_id,action_type,payload_json,status,expires_at) VALUES(?,?,?,?,?,?)",
                     (action_id, user_id, "memory_upsert", json.dumps(payload, ensure_ascii=False), "AWAITING_CONFIRMATION", expires_at))
    return {"status":"AWAITING_CONFIRMATION", "action_id":action_id, "field":payload["field"], "value":payload["value"], "expires_at":expires_at}


@router.post("/memories/proposals/{action_id}/confirm")
def confirm_memory(action_id: str, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    import json
    from datetime import datetime, timezone
    with db.connect() as conn:
        proposal = conn.execute("SELECT * FROM action_proposals WHERE action_id=? AND user_id=? AND action_type='memory_upsert'", (action_id, user_id)).fetchone()
        if proposal is None:
            raise HTTPException(status_code=404, detail="Memory proposal not found")
        if proposal["status"] != "AWAITING_CONFIRMATION":
            raise HTTPException(status_code=409, detail="Memory proposal is not awaiting confirmation")
        if datetime.fromisoformat(proposal["expires_at"]) < datetime.now(timezone.utc):
            conn.execute("UPDATE action_proposals SET status='EXPIRED' WHERE action_id=?", (action_id,))
            raise HTTPException(status_code=410, detail="Memory proposal expired")
        payload = json.loads(proposal["payload_json"])
    result = memories.upsert(user_id, payload["field"], payload["value"], payload.get("source", "user_confirmed"), True)
    with db.connect() as conn:
        conn.execute("UPDATE action_proposals SET status='COMMITTED' WHERE action_id=? AND user_id=?", (action_id, user_id))
    return result


@router.post("/memories")
def direct_memory_write_disabled() -> dict[str, Any]:
    raise HTTPException(status_code=405, detail="Direct memory writes are disabled. Create a proposal and confirm it in the app.")


@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    deleted = memories.delete(user_id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "memory_id": memory_id, "propagation": "sqlite_only_no_vector_or_cache_backend_configured"}


@router.get("/predictive/anomalies")
def anomaly_candidates(metric_codes: list[str] = Query(default=[]), user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    return predictive.anomaly_candidates(user_id, metric_codes)


@router.get("/predictive/forecast/{metric_code}")
def forecast(metric_code: str, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    return predictive.forecast(user_id, metric_code)


@router.get("/admin/metric-formula-reviews")
def list_metric_formula_reviews(status: str | None = None, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    return formula_reviews.list_reviews(status)


@router.post("/admin/metric-formula-reviews")
def propose_metric_formula_review(payload: RallyMetricReview, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    review = payload.model_dump()
    review["submitted_by"] = user_id
    try:
        return formula_reviews.submit_review(review)
    except KeyError:
        raise HTTPException(status_code=404, detail="Metric code not found in Rally catalog")
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))


@router.post("/reports/extraction-jobs")
async def upload_health_report(file: UploadFile = File(...), user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    from . import ingestion_routes

    return await ingestion_routes.report_upload_intake(file, user_id)


@router.get("/reports/extraction-jobs/{job_id}/file")
def download_health_report(job_id: str, user_id: str = Depends(get_user_id)):
    from . import ingestion_routes

    return ingestion_routes.report_file(job_id, user_id)


@router.get("/reports/extraction-jobs/{job_id}")
def get_report_job(job_id: str, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    import json
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM report_jobs WHERE job_id=? AND user_id=?", (job_id, user_id)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Report job not found")
    return {"job_id": row["job_id"], "status": row["status"], "filename": row["filename"],
            "candidates": json.loads(row["candidates_json"]), "confirmed_observations": json.loads(row["confirmed_json"])}


@router.post("/reports/extraction-jobs/{job_id}/candidates")
def attach_report_candidates(job_id: str, payload: ReportCandidateConfirm,
                             user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    from . import ingestion_routes

    return ingestion_routes.stage_report_candidates(job_id, payload, user_id)


@router.post("/reports/extraction-jobs/{job_id}/confirm")
def confirm_report_candidates(job_id: str, payload: ReportCandidateConfirm,
                              user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    from . import ingestion_routes

    return ingestion_routes.confirm_report_candidates(job_id, payload, user_id)


@router.post("/assistant/turns")
async def assistant_turn(payload: dict[str, Any], user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    import json
    from uuid import UUID
    from ..domain.schemas import ObservationCreate

    question = str(payload.get("question", "")).strip()
    if not question or len(question) > settings.max_chat_input_chars:
        raise HTTPException(status_code=422, detail="question is required and must be within configured length")
    conversation_id = payload.get("conversation_id")
    if conversation_id:
        try:
            conversation_id = str(UUID(str(conversation_id)))
        except ValueError as error:
            raise HTTPException(status_code=422, detail="conversation_id must be a UUID") from error
        with db.connect() as conn:
            conversation = conn.execute("SELECT conversation_id FROM assistant_conversations WHERE conversation_id=? AND user_id=?", (conversation_id, user_id)).fetchone()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conversation_id = str(uuid4())
        with db.connect() as conn:
            conn.execute("INSERT INTO assistant_conversations(conversation_id,user_id,title) VALUES(?,?,?)", (conversation_id, user_id, question[:120]))

    with db.connect() as conn:
        conn.execute("INSERT INTO assistant_messages(message_id,conversation_id,user_id,role,content) VALUES(?,?,?,?,?)",
                     (str(uuid4()), conversation_id, user_id, "user", question))
        rows = conn.execute("SELECT role,content FROM assistant_messages WHERE conversation_id=? AND user_id=? ORDER BY created_at DESC, rowid DESC LIMIT ?",
                            (conversation_id, user_id, max(2, settings.chat_history_turns * 2 + 1))).fetchall()
        conn.execute("UPDATE assistant_conversations SET updated_at=CURRENT_TIMESTAMP WHERE conversation_id=? AND user_id=?", (conversation_id, user_id))
    history = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)][:-1]

    with db.connect() as conn:
        profile_row = conn.execute("SELECT profile_json FROM profiles WHERE user_id=?", (user_id,)).fetchone()
        memories_rows = conn.execute("SELECT memory_id,field,value_json FROM user_memories WHERE user_id=? AND confirmed=1 ORDER BY updated_at DESC LIMIT 20", (user_id,)).fetchall()
    profile = json.loads(profile_row["profile_json"]) if profile_row else None
    profile_timezone = (profile or {}).get("timezone", "UTC")
    facts: dict[str, Any] = {"profile": [], "memories": [], "metrics": [], "sleep": [], "workouts": []}
    if profile:
        facts["profile"].append({"data": {key: value for key, value in profile.items() if key != "user_id"}, "source_ids": [f"profile:{user_id}"]})
    facts["memories"] = [{"field": row["field"], "value": json.loads(row["value_json"]), "source_ids": [row["memory_id"]]} for row in memories_rows]

    catalog = _catalog_metrics()
    normalized_question = question.casefold()
    aliases = {
        "RS-01": ["heart rate", "nhịp tim"], "RS-04": ["resting hr", "resting heart rate", "rhr", "nhịp tim lúc nghỉ"],
        "RS-05": ["hrv", "heart rate variability", "biến thiên nhịp tim"], "RS-06": ["respiratory rate", "nhịp thở"],
        "RS-10": ["spo2", "oxygen saturation", "độ bão hòa oxy"], "AC-09": ["steps", "step count", "số bước"],
        "SL-02": ["sleep", "sleep duration", "thời gian ngủ", "giấc ngủ"],
    }
    matched = []
    for definition in catalog:
        code = definition["metric_code"]
        labels = [definition.get(key) for key in ("metric_code", "name", "name_vi", "name_en")] + aliases.get(code, [])
        if any(isinstance(label, str) and len(label.strip()) >= 3 and label.casefold() in normalized_question for label in labels):
            matched.append(definition)
    for definition in matched[:3]:
        code = definition["metric_code"]
        result = metrics.compute(user_id, code, 7, timezone_name=profile_timezone)
        baseline = metrics.baseline(user_id, code, 28, minimum_valid_days=7, timezone_name=profile_timezone)
        source_ids = list(dict.fromkeys(result.source_observation_ids + baseline.get("source_ids", [])))
        facts["metrics"].append({"metric_code": code, "name": definition.get("name") or code, "value": result.value,
                                 "unit": result.unit, "status": result.status.value, "confidence": result.confidence_level.value,
                                 "baseline_value": baseline.get("value") if baseline.get("status") == "READY" else None,
                                 "baseline_status": baseline.get("status"), "source_ids": source_ids,
                                 "limitations": [result.explanation_facts.get("reason")] if result.explanation_facts.get("reason") else []})

    if any(term in normalized_question for term in ("sleep", "giấc ngủ", "ngủ")):
        today = datetime.now(timezone.utc).date()
        for day_offset in range(7):
            night = today - timedelta(days=day_offset)
            start = datetime.combine(night, datetime.min.time(), tzinfo=timezone.utc) - timedelta(hours=12)
            end = start + timedelta(days=1)
            sleep_observations = []
            for code in ("SL-02", "SL-04", "SL-05", "SL-06", "SL-07"):
                sleep_observations.extend(observations.history(user_id, code, start, end))
            summary = summarize_sleep_intervals(sleep_observations, night, profile_timezone)
            if summary.get("status") == "PROTOTYPE_READY":
                source_ids = [item.observation_id for item in sleep_observations]
                facts["sleep"].append({**summary, "source_ids": source_ids})

    workout_requested = any(term in normalized_question for term in ("workout", "exercise", "bài tập", "tập luyện"))
    if workout_requested:
        facts["workouts"] = workouts.search("female", limit=20) + workouts.search("male", limit=20)
        for workout in facts["workouts"]:
            workout["source_ids"] = [f"workout:{workout['exercise_id']}"]

    all_source_ids = {source_id for section in facts.values() for fact in section for source_id in fact.get("source_ids", [])}
    answer_data: dict[str, Any] | None = None
    if settings.chat_enabled:
        try:
            answer_data = await gpt4o_chat.respond(question, history, facts)
            cited = answer_data.get("cited_source_ids", [])
            if not isinstance(cited, list) or any(not isinstance(item, str) or item not in all_source_ids for item in cited):
                answer_data = None
        except Exception:
            answer_data = None

    fallback = answer_data is None
    if answer_data is None:
        answer_data = {"answer": "Mình chưa tìm thấy đủ dữ liệu cá nhân có nguồn hoặc trợ lý AI hiện không khả dụng.",
                       "cited_source_ids": [], "follow_up_question": None, "intent": "question", "observation_proposal": None}
    proposal_result = None
    if answer_data.get("intent") == "observation_write" and answer_data.get("observation_proposal"):
        try:
            proposed = ObservationCreate.model_validate(answer_data["observation_proposal"] | {
                "provenance": {"source_type": "manual", "provider": "assistant_proposal"}, "user_confirmed": False
            })
            known_codes = {item["metric_code"] for item in catalog}
            if proposed.metric_code not in known_codes:
                raise ValueError("unknown metric")
            normalized = observations.normalize(proposed)
            validation_flags = observations.validate(normalized)
            if validation_flags:
                raise ValueError("invalid observation")
            action_id = str(uuid4())
            expires_at = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
            proposal_payload = normalized.model_dump(mode="json")
            with db.connect() as conn:
                conn.execute("INSERT INTO action_proposals(action_id,user_id,action_type,payload_json,status,expires_at) VALUES(?,?,?,?,?,?)",
                             (action_id, user_id, "assistant_observation_create", json.dumps(proposal_payload), "AWAITING_CONFIRMATION", expires_at))
            proposal_result = {"action_id": action_id, "status": "AWAITING_CONFIRMATION", "observation": proposal_payload, "expires_at": expires_at}
        except Exception:
            answer_data["observation_proposal"] = None

    response_text = answer_data["answer"]
    with db.connect() as conn:
        conn.execute("INSERT INTO assistant_messages(message_id,conversation_id,user_id,role,content) VALUES(?,?,?,?,?)",
                     (str(uuid4()), conversation_id, user_id, "assistant", response_text))
        conn.execute("UPDATE assistant_conversations SET updated_at=CURRENT_TIMESTAMP WHERE conversation_id=? AND user_id=?", (conversation_id, user_id))
    return {"status": "ok" if all_source_ids else "insufficient_context", "conversation_id": conversation_id,
            "answer": response_text, "facts": facts, "sources": answer_data["cited_source_ids"],
            "confidence": min((fact["confidence"] for fact in facts["metrics"]), default="UNAVAILABLE"),
            "follow_up_question": answer_data["follow_up_question"], "fallback_used": fallback,
            "action_proposal": proposal_result,
            "limitations": ["This assistant does not diagnose; sleep summaries and some metrics are prototypes."]}


@router.get("/assistant/conversations")
def list_assistant_conversations(user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    with db.connect() as conn:
        rows = conn.execute("SELECT conversation_id,title,created_at,updated_at FROM assistant_conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT 100", (user_id,)).fetchall()
    return {"items": [dict(row) for row in rows]}


@router.get("/assistant/conversations/{conversation_id}")
def get_assistant_conversation(conversation_id: str, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    with db.connect() as conn:
        conversation = conn.execute("SELECT conversation_id,title,created_at,updated_at FROM assistant_conversations WHERE conversation_id=? AND user_id=?", (conversation_id, user_id)).fetchone()
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        rows = conn.execute("SELECT role,content,created_at FROM assistant_messages WHERE conversation_id=? AND user_id=? ORDER BY created_at,rowid", (conversation_id, user_id)).fetchall()
    return {"conversation": dict(conversation), "messages": [dict(row) for row in rows]}


@router.post("/assistant/action-proposals/{action_id}/confirm")
def confirm_assistant_action(action_id: str, user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    import json
    from datetime import datetime, timezone
    with db.connect() as conn:
        proposal = conn.execute("SELECT * FROM action_proposals WHERE action_id=? AND user_id=? AND action_type='assistant_observation_create'", (action_id, user_id)).fetchone()
        if proposal is None:
            raise HTTPException(status_code=404, detail="Assistant action proposal not found")
        if proposal["status"] != "AWAITING_CONFIRMATION":
            raise HTTPException(status_code=409, detail="Action proposal is not awaiting confirmation")
        if datetime.fromisoformat(proposal["expires_at"]) < datetime.now(timezone.utc):
            conn.execute("UPDATE action_proposals SET status='EXPIRED' WHERE action_id=? AND user_id=?", (action_id, user_id))
            raise HTTPException(status_code=410, detail="Action proposal expired")
        proposed = ObservationCreate.model_validate(json.loads(proposal["payload_json"]))
    saved = observations.add(user_id, proposed.model_copy(update={"user_confirmed": True}))
    with db.connect() as conn:
        conn.execute("UPDATE action_proposals SET status='COMMITTED' WHERE action_id=? AND user_id=?", (action_id, user_id))
    return {"status": "confirmed", "observation": saved.model_dump(mode="json")}


@router.post("/video/assets")
def register_video_asset(payload: dict[str, Any], user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    required = {"asset_id", "exercise_id", "version", "review_status"}
    missing = required - payload.keys()
    if missing:
        raise HTTPException(status_code=422, detail={"code":"MISSING_VIDEO_METADATA", "missing":sorted(missing)})
    if payload["review_status"] not in {"draft", "needs_review", "approved", "rejected"}:
        raise HTTPException(status_code=422, detail="Invalid review_status")
    import json
    with db.connect() as conn:
        conn.execute("INSERT OR REPLACE INTO video_assets(asset_id,exercise_id,asset_uri,version,review_status,metadata_json) VALUES(?,?,?,?,?,?)",
                     (payload["asset_id"], payload["exercise_id"], payload.get("asset_uri"), payload["version"], payload["review_status"], json.dumps(payload, ensure_ascii=False)))
    return {"status":"registered", "asset_id":payload["asset_id"], "review_status":payload["review_status"], "created_by":user_id}


@router.post("/admin/catalog/reimport")
def reimport_catalog(user_id: str = Depends(get_user_id)) -> dict[str, Any]:
    report = import_workbooks(settings.source_dir, settings.normalized_data_dir)
    return {"status":"imported", "report":report, "requested_by":user_id}
