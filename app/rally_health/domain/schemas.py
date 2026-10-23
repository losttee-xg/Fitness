from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DataStatus(StrEnum):
    OK = "OK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"
    LOW_SIGNAL_QUALITY = "LOW_SIGNAL_QUALITY"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    NOT_COMPUTABLE = "NOT_COMPUTABLE"


class ConfidenceLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNAVAILABLE = "UNAVAILABLE"


class AggregationKind(StrEnum):
    INSTANT = "instant"
    INTERVAL = "interval"
    DAILY_DERIVED = "daily_derived"
    RESEARCH_LABEL = "research_label"


class DataSource(StrEnum):
    APPLE_HEALTH = "apple_health"
    RESEARCH_DATASET = "research_dataset"
    WEARABLE = "wearable"
    MANUAL = "manual"
    QUESTIONNAIRE = "questionnaire"
    CAMERA_PPG = "camera_ppg"
    HEALTH_REPORT = "health_report"
    WORKOUT_LOG = "workout_log"
    APP_DERIVED = "app_derived"


class Provenance(BaseModel):
    source_type: DataSource
    provider: str | None = Field(default=None, max_length=80)
    source_id: str | None = Field(default=None, max_length=256)
    sample_id: str | None = Field(default=None, max_length=256)
    device_id: str | None = Field(default=None, max_length=256)
    device_model: str | None = Field(default=None, max_length=128)
    firmware_version: str | None = Field(default=None, max_length=128)
    measurement_context: str | None = Field(default=None, max_length=64)
    import_batch_id: str | None = None
    document_ref: str | None = None
    source_workbook: str | None = None
    source_sheet: str | None = None
    source_row: int | None = None


class ObservationCreate(BaseModel):
    observation_id: str = Field(default_factory=lambda: str(uuid4()))
    metric_code: str = Field(min_length=2, max_length=80)
    metric_variant: str | None = Field(default=None, min_length=1, max_length=48)
    aggregation: AggregationKind = AggregationKind.INSTANT
    value: float
    unit: str = Field(min_length=1, max_length=32)
    observed_at: datetime
    end_at: datetime | None = None
    received_at: datetime = Field(default_factory=utc_now)
    provenance: Provenance
    quality_flags: list[str] = Field(default_factory=list)
    user_confirmed: bool = False
    model_config = ConfigDict(extra="forbid")

    @field_validator("observed_at", "end_at", "received_at")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must include a timezone")
        return value


class ObservationOut(ObservationCreate):
    user_id: str
    created_at: datetime = Field(default_factory=utc_now)


class ObservationBatchCreate(BaseModel):
    batch_id: str = Field(min_length=8, max_length=128)
    provider: str = Field(min_length=1, max_length=80)
    observations: list[ObservationCreate] = Field(min_length=1, max_length=10000)
    model_config = ConfigDict(extra="forbid")


class QuestionnaireSubmission(BaseModel):
    instrument: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=32)
    completed_at: datetime
    answers: dict[str, Any]
    model_config = ConfigDict(extra="forbid")

    @field_validator("completed_at")
    @classmethod
    def questionnaire_requires_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("completed_at must include a timezone")
        return value


class ReportCandidate(BaseModel):
    metric_code: str = Field(min_length=2, max_length=80)
    source_analyte: str = Field(default="", max_length=160)
    metric_variant: str | None = None
    aggregation: AggregationKind = AggregationKind.INSTANT
    value: float
    unit: str = Field(min_length=1, max_length=32)
    observed_at: datetime
    end_at: datetime | None = None
    source_text: str = Field(default="", max_length=1000)
    source_page: int = Field(default=1, ge=1)
    model_config = ConfigDict(extra="forbid")

    @field_validator("observed_at", "end_at")
    @classmethod
    def candidate_requires_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must include a timezone")
        return value


class ReportCandidateConfirm(BaseModel):
    candidates: list[ReportCandidate] = Field(min_length=1)
    model_config = ConfigDict(extra="forbid")


class RallyMetricReview(BaseModel):
    metric_code: str = Field(min_length=2, max_length=80)
    formula_id: str = Field(min_length=1, max_length=128)
    formula_version: str = Field(min_length=1, max_length=32)
    definition: dict[str, Any]
    test_cases: list[dict[str, Any]] = Field(min_length=1)
    model_config = ConfigDict(extra="forbid")


class MetricResult(BaseModel):
    metric_code: str
    value: float | None = None
    unit: str | None = None
    status: DataStatus
    confidence_level: ConfidenceLevel
    confidence_components: dict[str, float | str | None] = Field(default_factory=dict)
    baseline_value: float | None = None
    baseline_window_days: int | None = None
    source_observation_ids: list[str] = Field(default_factory=list)
    formula_version: str | None = None
    explanation_facts: dict[str, Any] = Field(default_factory=dict)
    computed_at: datetime = Field(default_factory=utc_now)


class WorkoutRecord(BaseModel):
    exercise_id: str
    sex_catalog: str
    name: str | None = None
    name_vi: str | None = None
    name_en: str | None = None
    category_ids: list[str] = Field(default_factory=list)
    equipment: list[str] = Field(default_factory=list)
    intensity: str | None = None
    difficulty: float | None = None
    reps_weak: str | None = None
    reps_medium: str | None = None
    reps_strong: str | None = None
    sets_weak: str | None = None
    sets_medium: str | None = None
    sets_strong: str | None = None
    life_stages: list[str] = Field(default_factory=list)
    cycle_phases: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    review_status: str = "needs_domain_review"
    source: dict[str, Any] = Field(default_factory=dict)


class UserProfile(BaseModel):
    user_id: str
    timezone: str = "UTC"
    goals: list[str] = Field(default_factory=list)
    fitness_level: str | None = None
    equipment: list[str] = Field(default_factory=list)
    availability: dict[str, list[str]] = Field(default_factory=dict)
    movement_limitations: list[str] = Field(default_factory=list)
    health_context: dict[str, Any] = Field(default_factory=dict)
    consent_scopes: list[str] = Field(default_factory=list)
    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError("timezone must be a valid IANA timezone name") from error
        return value


class InsightFact(BaseModel):
    fact_id: str
    metric_code: str
    value: float | str | None
    unit: str | None = None
    source_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel
    limitations: list[str] = Field(default_factory=list)


class AssistantResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    status: str
    insight: str | None = None
    recommendation: str | None = None
    explanation: str
    confidence: ConfidenceLevel
    facts: list[InsightFact] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    action_proposals: list[dict[str, Any]] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ScheduleEventCreate(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str = Field(min_length=1, max_length=160)
    sex_catalog: str = Field(pattern="^(female|male)$")
    exercise_ids: list[str] = Field(default_factory=list)
    starts_at: datetime
    ends_at: datetime
    requested_by_user: bool = False
    confirmation_id: str | None = None
    idempotency_key: str = Field(min_length=8, max_length=128)
    model_config = ConfigDict(extra="forbid")

    @field_validator("starts_at", "ends_at")
    @classmethod
    def require_schedule_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("schedule timestamps must include a timezone")
        return value


class ActionProposal(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    action_type: str
    payload: dict[str, Any]
    status: str = "AWAITING_CONFIRMATION"
    user_id: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=utc_now)


class APIError(BaseModel):
    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
