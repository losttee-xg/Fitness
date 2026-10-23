from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

LABEL_MAP = {"not_at_all": 0, "a_little_bit": 1, "somewhat": 2, "quite_a_bit": 3, "very_much": 4}
META = {"dataset_id", "participant_id", "target_name", "target_category", "target_date", "survey_date",
        "target_period_start", "target_period_end", "target_source", "target_source_file", "target_source_row",
        "feature_window_start", "feature_window_end_exclusive", "feature_window_relation", "lookback_days",
        "observed_days", "synthetic", "clinical_validation", "observed_sensor_metrics", "eligible_for_modeling"}


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("target_name") == "tired_yesterday"
                and r.get("eligible_for_modeling", "").casefold() == "true"
                and r.get("synthetic", "false").casefold() != "true"]
    features = [k for k in rows[0] if k not in META and k != "target_numeric_source"]
    return rows, features


def metric_set(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "ordinal_mae": float(mean_absolute_error(y, pred)),
            "ordinal_rmse": float(math.sqrt(mean_squared_error(y, pred)))}


def modality(name: str) -> str:
    n = name.removeprefix("feature__")
    if n.startswith("heart_rate__"):
        return "heart_rate"
    if n.startswith("sleep__"):
        return "sleep"
    if n.startswith("steps__") or n.startswith("active_fitbit__"):
        return "activity"
    return "other"


def is_clock(name: str) -> bool:
    n = name.casefold()
    return "dec_time" in n or "start_minute" in n or n.endswith("__mean_prior_window") and "start_time" in n


def model() -> Any:
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         ExtraTreesClassifier(n_estimators=300, min_samples_leaf=8, max_features=0.7,
                                              class_weight="balanced", random_state=42, n_jobs=1))


def grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 5) -> dict[str, Any]:
    split = GroupKFold(n_splits=min(folds, len(set(groups))))
    pred = np.zeros(len(y), dtype=int)
    base = np.zeros(len(y), dtype=int)
    details = []
    for fold, (tr, te) in enumerate(split.split(X, y, groups), start=1):
        base[te] = int(np.rint(np.median(y[tr])))
        clf = model()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te]).astype(int)
        details.append({"fold": fold, "train_people": len(set(groups[tr])), "test_people": len(set(groups[te])),
                        "test_rows": len(te), "test_classes": dict(Counter(map(int, y[te])))})
    return {"model": metric_set(y, pred), "median_baseline": metric_set(y, base), "folds": details}


def temporal_split(rows: list[dict[str, str]], y: np.ndarray, groups: np.ndarray, X: np.ndarray) -> dict[str, Any]:
    per: dict[str, list[int]] = defaultdict(list)
    for i, person in enumerate(groups):
        per[person].append(i)
    train, test = [], []
    for person, ix in per.items():
        ix.sort(key=lambda i: rows[i]["target_date"])
        if len(ix) < 5:
            continue
        cut = max(3, min(len(ix)-1, int(len(ix)*0.8)))
        train.extend(ix[:cut]); test.extend(ix[cut:])
    if not test:
        return {"status": "insufficient"}
    train, test = np.array(train), np.array(test)
    med, maj = int(np.rint(np.median(y[train]))), Counter(y[train]).most_common(1)[0][0]
    person_prior = []
    train_by_person: dict[str, list[int]] = defaultdict(list)
    for i in train:
        train_by_person[groups[i]].append(i)
    for i in test:
        prior_ix = [j for j in train_by_person.get(groups[i], []) if rows[j]["target_date"] < rows[i]["target_date"]]
        person_prior.append(int(np.rint(np.median(y[prior_ix]))) if prior_ix else med)
    clf = model(); clf.fit(X[train], y[train]); p = clf.predict(X[test]).astype(int)
    return {"status": "exploratory_single_temporal_holdout", "train_rows": len(train), "test_rows": len(test),
            "participants": len(set(groups[test])), "model": metric_set(y[test], p),
            "population_median_baseline": metric_set(y[test], np.full(len(test), med)),
            "population_majority_baseline": metric_set(y[test], np.full(len(test), maj)),
            "participant_history_median_baseline": metric_set(y[test], np.asarray(person_prior)),
            "warning": "single split, recurring prompts/overlapping windows, not prospective external validation"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run predeclared fatigue model ablations on observed data only.")
    parser.add_argument("--input", type=Path, default=Path("dataset/harmonized/master/fatigue_daily_training_examples.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized/master/fatigue_ablation_report.json"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    rows, feature_names = load_rows(args.input)
    y = np.asarray([LABEL_MAP[r["target_category"].casefold()] for r in rows], dtype=int)
    groups = np.asarray([r["participant_id"] for r in rows])
    dates = [date.fromisoformat(r["target_date"]) for r in rows]
    Xall = np.asarray([[float(r[k]) if r.get(k) not in (None, "") else np.nan for k in feature_names] for r in rows])

    groupsets = {
        "all": feature_names,
        "heart_rate_only": [n for n in feature_names if modality(n) == "heart_rate"],
        "sleep_only": [n for n in feature_names if modality(n) == "sleep"],
        "activity_only": [n for n in feature_names if modality(n) == "activity"],
        "no_clock_fields": [n for n in feature_names if not is_clock(n)],
        "core_hr_sleep_activity": [n for n in feature_names if modality(n) in {"heart_rate", "sleep", "activity"} and not is_clock(n)],
    }
    results = {}
    for name, selected in groupsets.items():
        idx = np.asarray([feature_names.index(k) for k in selected], dtype=int)
        X = Xall[:, idx]
        results[name] = {"n_features": len(selected), "group_cv": grouped_cv(X, y, groups, args.folds),
                         "temporal_holdout": temporal_split(rows, y, groups, X)}
    report = {"purpose": "research-only ablations, observed labels only; no synthetic data",
              "target": "source self-report tired_yesterday, ordinal convenience mapping 0..4",
              "rows": len(rows), "participants": len(set(groups)), "class_counts": dict(Counter(map(int, y))),
              "features_total": len(feature_names), "window": "7 prior calendar days before target date",
              "selection_warning": "only survey responders with >=5 nonmissing sensor metric columns retained; not all enrolled people/days",
              "results": results,
              "limits": ["participant GroupKFold tests unseen-person generalization, not future dates for known users",
                         "temporal holdout is one split with repeated/overlapping windows",
                         "MAE on ordinal categories assumes an ordered proxy scale; source response categories remain authoritative",
                         "not a clinical fatigue diagnosis, Recovery target, or product-ready evaluation"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": {
        k: {"n_features": v["n_features"], "group_cv_model": v["group_cv"]["model"],
            "group_cv_baseline": v["group_cv"]["median_baseline"],
            "temporal": {q: v["temporal_holdout"].get(q) for q in ("model", "population_median_baseline", "participant_history_median_baseline")}}
        for k, v in results.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
