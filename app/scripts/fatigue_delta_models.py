from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

LABELS = {"not_at_all": 0, "a_little_bit": 1, "somewhat": 2, "quite_a_bit": 3, "very_much": 4}
META = {"dataset_id", "participant_id", "target_name", "target_category", "target_date", "survey_date",
        "target_period_start", "target_period_end", "target_source", "target_source_file", "target_source_row",
        "feature_window_start", "feature_window_end_exclusive", "feature_window_relation", "lookback_days",
        "observed_days", "synthetic", "clinical_validation", "observed_sensor_metrics", "eligible_for_modeling"}


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.rint(p), 0, 4).astype(int)
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(math.sqrt(mean_squared_error(y, p)))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Gap-aware change-from-personal-baseline models.")
    parser.add_argument("--input", type=Path, default=Path("dataset/harmonized/master/fatigue_daily_training_examples.csv"))
    parser.add_argument("--output", type=Path, default=Path("dataset/harmonized/master/fatigue_delta_models.json"))
    parser.add_argument("--gap-days", type=int, default=7)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("target_name") == "tired_yesterday"
                and r.get("eligible_for_modeling", "").casefold() == "true"
                and r.get("synthetic", "false").casefold() != "true"]
    names = [k for k in rows[0] if k not in META and k != "target_numeric_source"]
    X_sensor = np.asarray([[float(r[k]) if r.get(k) not in (None, "") else np.nan for k in names] for r in rows], dtype=float)
    y = np.asarray([LABELS[r["target_category"].casefold()] for r in rows], dtype=float)
    people = [r["participant_id"] for r in rows]
    days = [date.fromisoformat(r["target_date"]) for r in rows]
    per: dict[str, list[int]] = defaultdict(list)
    for i,p in enumerate(people): per[p].append(i)
    hist = np.full((len(rows), 4), np.nan)
    for p, ix in per.items():
        ordered = sorted(ix, key=lambda i: days[i]); past=[]
        for i in ordered:
            valid=[j for j in past if days[j] < days[i]]
            if valid:
                recent=valid[-3:]; hist[i]=[y[valid[-1]],statistics.mean(y[j] for j in recent),len(valid),(days[i]-days[valid[-1]]).days]
            past.append(i)
    X=np.column_stack([X_sensor,hist])
    unique_days=sorted(set(days)); cuts=[unique_days[int(len(unique_days)*k)] for k in (.60,.72,.84)]; ends=cuts[1:]+[unique_days[-1]+timedelta(days=1)]
    folds=[]; used=set()
    for f,(cut,end) in enumerate(zip(cuts,ends),1):
        train=np.asarray([i for i,d in enumerate(days) if d < cut-timedelta(days=args.gap_days)],int)
        test=np.asarray([i for i,d in enumerate(days) if cut <= d < end and i not in used and np.isfinite(hist[i,0])],int); used.update(map(int,test))
        train=[i for i in train if np.isfinite(hist[i,0])]
        train=np.asarray(train,int)
        if len(train)<100 or not len(test):
            folds.append({'fold':f,'status':'insufficient','train':len(train),'test':len(test)});continue
        last=hist[test,0]; actual=y[test]; delta=y[train]-hist[train,0]
        predictions={'last_self_report':last.copy()}
        # Models predict change from the prior self-report; use the last rating as anchor.
        for key,reg in [('extra_trees_delta',ExtraTreesRegressor(n_estimators=250,min_samples_leaf=10,max_features=.7,random_state=42,n_jobs=1)),
                        ('ridge_delta',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),Ridge(alpha=50.0))),
                        ('hgb_delta',HistGradientBoostingRegressor(max_iter=100,learning_rate=.04,max_leaf_nodes=5,l2_regularization=5.0,random_state=42))]:
            if key=='extra_trees_delta':
                model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),reg)
            else: model=reg
            model.fit(X[train],delta)
            d=model.predict(X[test])
            predictions[key]=np.clip(last+d,0,4)
            predictions[key+'_shrink_0_5']=np.clip(last+0.5*d,0,4)
        folds.append({'fold':f,'train_rows':len(train),'test_rows':len(test),'test_people':len(set(people[i] for i in test)),
                      'train_end_exclusive':(cut-timedelta(days=args.gap_days)).isoformat(),'test_start':cut.isoformat(),'test_end_exclusive':end.isoformat(),
                      'metrics':{k:scores(actual,v) for k,v in predictions.items()}})
    report={'purpose':'diagnostic only; no new labels; self-report target is not clinical fatigue or Recovery',
            'target':'next tired_yesterday score, ordinal proxy 0..4','feature_set':'wearable features + strictly prior self-report history; models predict delta from last response',
            'split':'three rolling-origin date blocks with 7-day embargo; past user response only if earlier than target date',
            'folds':folds,'limits':['single cohort, repeated windows and within-person serial correlation','prediction horizon is next answered survey, not necessarily exactly next calendar day','do not tune on these blocks then claim unbiased model performance']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'folds':folds},ensure_ascii=False,indent=2));return 0

if __name__=='__main__': raise SystemExit(main())
