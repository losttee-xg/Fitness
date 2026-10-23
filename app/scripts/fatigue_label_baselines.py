from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

LABEL_TO_INT = {"not_at_all": 0, "a_little_bit": 1, "somewhat": 2, "quite_a_bit": 3, "very_much": 4}
INT_TO_LABEL = {value: key for key, value in LABEL_TO_INT.items()}


def load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream)
                if row.get("target_name") == "tired_yesterday"
                and row.get("eligible_for_modeling", "").casefold() == "true"
                and row.get("synthetic", "false").casefold() != "true"]


def eval_predictions(y: list[int], pred: list[int]) -> dict[str, float]:
    return {"accuracy": accuracy_score(y, pred), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "ordinal_mae": mean_absolute_error(y, pred)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate self-report-response baselines without wearable features.")
    parser.add_argument("--input", type=Path, default=Path("dataset/harmonized/master/fatigue_daily_training_examples.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized/master/fatigue_label_only_baselines.json"))
    parser.add_argument("--test-fraction", type=float, default=0.25)
    args = parser.parse_args()
    rows = load(args.input)
    if len(rows) < 10 or not 0 < args.test_fraction < 0.5:
        raise ValueError("Need at least 10 rows and test fraction (0, 0.5)")
    people = [row["participant_id"] for row in rows]
    y = [LABEL_TO_INT[row["target_category"].casefold()] for row in rows]
    groups = sorted(set(people))
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_fraction, random_state=42)
    train_ix, test_ix = next(splitter.split(rows, y, people))
    train_by_person: dict[str, list[tuple[str, int]]] = {}
    for i in train_ix:
        train_by_person.setdefault(people[i], []).append((rows[i]["target_date"], y[i]))
    population = Counter(y[i] for i in train_ix)
    pop_median = sorted(y[i] for i in train_ix)[len(train_ix)//2]
    strategies = {"population_median": [], "population_mode": [], "last_prior_self_report": [], "rolling_three_prior_reports": []}
    covered_last = covered_rolling = 0
    for i in test_ix:
        person, day = people[i], rows[i]["target_date"]
        history = sorted(((d, value) for d, value in train_by_person.get(person, []) if d < day), key=lambda x: x[0])
        strategies["population_median"].append(pop_median)
        strategies["population_mode"].append(population.most_common(1)[0][0])
        if history:
            strategies["last_prior_self_report"].append(history[-1][1]); covered_last += 1
            strategies["rolling_three_prior_reports"].append(round(sum(v for _, v in history[-3:])/min(3,len(history)))); covered_rolling += 1
        else:
            strategies["last_prior_self_report"].append(pop_median)
            strategies["rolling_three_prior_reports"].append(pop_median)
    y_test = [y[i] for i in test_ix]
    report = {"purpose": "diagnostic baseline only; no wearable inputs", "participants_total": len(groups),
              "train_participants": len(set(people[i] for i in train_ix)), "test_participants": len(set(people[i] for i in test_ix)),
              "train_rows": len(train_ix), "test_rows": len(test_ix), "train_class_counts": dict(population),
              "test_class_counts": dict(Counter(y_test)), "participants_with_prior_label_coverage": len(set(people[i] for i in test_ix)&set(train_by_person)),
              "test_rows_with_prior_label_coverage": covered_last,
              "metrics": {name: eval_predictions(y_test, pred) for name, pred in strategies.items()},
              "interpretation": "If wearable models do not beat a fair prior-self-report baseline, the predictive signal may be persistence in reported tiredness rather than sensor value.",
              "limits": ["This split is by participant, so held-out participants have no prior labels in train; history strategies fall back to population median.",
                         "Do not use random row split because it leaks repeated participants.",
                         "Targets are self-reports, not clinical fatigue or Recovery."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
