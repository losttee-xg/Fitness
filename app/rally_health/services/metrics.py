from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..domain.schemas import AggregationKind, ConfidenceLevel, DataStatus, MetricResult
from .formula_registry import describe_metric
from .observations import ObservationService


class MetricEngine:
    """Execute explicitly registered prototypes, never workbook prose."""

    def __init__(self, observations: ObservationService, catalog_provider=None):
        self.observations = observations
        self.catalog_provider = catalog_provider

    def compute(self, user_id: str, metric_code: str, window_days: int = 28,
                metric_variant: str | None = None, timezone_name: str = "UTC") -> MetricResult:
        try:
            user_timezone=timezone.utc if timezone_name == "UTC" else ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError):
            return MetricResult(metric_code=metric_code,status=DataStatus.INVALID_INPUT,
                                confidence_level=ConfidenceLevel.UNAVAILABLE,
                                explanation_facts={"reason":"Invalid IANA timezone.","timezone":timezone_name})
        if self.catalog_provider is not None:
            definition = next((item for item in self.catalog_provider() if item["metric_code"].casefold() == metric_code.casefold()), None)
            if definition is None:
                return MetricResult(metric_code=metric_code, status=DataStatus.NOT_COMPUTABLE,
                                    confidence_level=ConfidenceLevel.UNAVAILABLE,
                                    explanation_facts={"reason":"Metric is not present in the imported Rally catalog."})
            registry = describe_metric(definition)["formula_registry"]
            if not registry["executable"]:
                status = DataStatus.REVIEW_REQUIRED if registry["status"] == "review_required_missing_definition" else DataStatus.NOT_COMPUTABLE
                return MetricResult(metric_code=metric_code,status=status,confidence_level=ConfidenceLevel.UNAVAILABLE,
                                    explanation_facts={"reason":"No reviewed executable formula is registered for this metric.",
                                                       "registry_status":registry["status"],"rally_source":registry["source_ref"]})
        now=datetime.now(timezone.utc)
        history=self.observations.history(user_id,metric_code,now-timedelta(days=window_days),now,metric_variant)
        if not history:
            return MetricResult(metric_code=metric_code,status=DataStatus.INSUFFICIENT_DATA,
                                confidence_level=ConfidenceLevel.UNAVAILABLE,
                                explanation_facts={"reason":"No valid observations in the requested period."})
        conflicts=[item for item in history if "source_conflict_review_required" in item.quality_flags]
        if conflicts:
            return MetricResult(metric_code=metric_code,status=DataStatus.REVIEW_REQUIRED,
                                confidence_level=ConfidenceLevel.UNAVAILABLE,
                                source_observation_ids=[item.observation_id for item in conflicts],
                                explanation_facts={"reason":"Conflicting sources require review; no source precedence is assumed."})
        usable=[item for item in history if not any(flag in {"missing_value","off_wrist","low_confidence","signal_unavailable"} for flag in item.quality_flags)]
        if not usable:
            return MetricResult(metric_code=metric_code,status=DataStatus.LOW_SIGNAL_QUALITY,
                                confidence_level=ConfidenceLevel.UNAVAILABLE,
                                source_observation_ids=[item.observation_id for item in history],
                                explanation_facts={"reason":"No usable observations remain after quality filtering."})
        if any(item.aggregation != AggregationKind.INSTANT for item in usable):
            return MetricResult(metric_code=metric_code,status=DataStatus.NOT_COMPUTABLE,
                                confidence_level=ConfidenceLevel.UNAVAILABLE,
                                source_observation_ids=[item.observation_id for item in usable],
                                explanation_facts={"reason":"This prototype aggregates instant observations only; interval and daily-derived data require a reviewed formula."})
        units={item.unit for item in usable}; variants={item.metric_variant for item in usable}
        if len(units)!=1 or len(variants)!=1:
            return MetricResult(metric_code=metric_code,status=DataStatus.INVALID_INPUT,
                                confidence_level=ConfidenceLevel.UNAVAILABLE,
                                explanation_facts={"reason":"Mixed units or variants cannot be aggregated."})
        value=statistics.median(item.value for item in usable)
        valid_days={item.observed_at.astimezone(user_timezone).date() for item in usable}
        confidence=ConfidenceLevel.LOW if len(valid_days)<3 else ConfidenceLevel.MEDIUM
        return MetricResult(metric_code=metric_code,value=value,unit=next(iter(units)),status=DataStatus.OK,
                            confidence_level=confidence,
                            confidence_components={"sample_count":float(len(usable)),"valid_day_count":float(len(valid_days)),
                                                   "coverage_window_days":float(window_days),"calibrated":"false",
                                                   "interpretation":"heuristic_only_not_probability"},
                            source_observation_ids=[item.observation_id for item in usable],formula_version="raw-median-1.0.0",
                            explanation_facts={"aggregation":"raw_observation_median_not_rally_derived_formula",
                                               "window_days":window_days,"metric_variant":metric_variant,
                                               "timezone":timezone_name})

    def baseline(self,user_id:str,metric_code:str,window_days:int=28,minimum_samples:int=7,
                 minimum_valid_days:int=7,metric_variant:str|None=None,timezone_name:str="UTC")->dict:
        try:
            user_timezone=timezone.utc if timezone_name == "UTC" else ZoneInfo(timezone_name)
        except (ZoneInfoNotFoundError, ValueError):
            return {"status":"INVALID_INPUT","metric_code":metric_code,"value":None,
                    "error":"invalid_timezone","timezone":timezone_name}
        now=datetime.now(timezone.utc)
        observations=self.observations.history(user_id,metric_code,now-timedelta(days=window_days),now,metric_variant)
        conflicts=[item for item in observations if "source_conflict_review_required" in item.quality_flags]
        if conflicts:
            return {"status":"REVIEW_REQUIRED","metric_code":metric_code,"value":None,
                    "conflict_observation_ids":[item.observation_id for item in conflicts]}
        observations=[item for item in observations if not any(flag in {"missing_value","off_wrist","low_confidence","signal_unavailable"} for flag in item.quality_flags)]
        days:dict[object,list[float]]=defaultdict(list)
        for item in observations:
            days[item.observed_at.astimezone(user_timezone).date()].append(item.value)
        daily=[statistics.median(vals) for vals in days.values()]
        units={item.unit for item in observations}
        if len(units)>1:
            return {"status":"INVALID_INPUT","metric_code":metric_code,"valid_samples":len(observations),"value":None}
        if len(observations)<minimum_samples or len(daily)<minimum_valid_days:
            return {"status":"INSUFFICIENT_DATA","metric_code":metric_code,"valid_samples":len(observations),
                    "valid_days":len(daily),"required_samples":minimum_samples,"required_valid_days":minimum_valid_days,"value":None,
                    "timezone":timezone_name}
        return {"status":"READY","metric_code":metric_code,"window_days":window_days,"valid_samples":len(observations),
                "valid_days":len(daily),"required_valid_days":minimum_valid_days,"value":statistics.median(daily),
                "dispersion":statistics.pstdev(daily) if len(daily)>1 else 0.0,
                "method":"median_of_daily_medians_and_population_sd","timezone":timezone_name,
                "confidence_level":"UNAVAILABLE","confidence_calibrated":False,
                "confidence_note":"Baseline dispersion is descriptive; it is not calibrated confidence.",
                "source_ids":[item.observation_id for item in observations]}
