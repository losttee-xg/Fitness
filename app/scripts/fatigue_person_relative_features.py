from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

LABELS = {"not_at_all": 0, "a_little_bit": 1, "somewhat": 2, "quite_a_bit": 3, "very_much": 4}
META = {"dataset_id", "participant_id", "target_name", "target_category", "target_date", "survey_date",
        "target_period_start", "target_period_end", "target_source", "target_source_file", "target_source_row",
        "feature_window_start", "feature_window_end_exclusive", "feature_window_relation", "lookback_days",
        "observed_days", "synthetic", "clinical_validation", "observed_sensor_metrics", "eligible_for_modeling"}


def metrics(y, pred):
    pred=np.clip(np.rint(pred),0,4).astype(int)
    return {'mae':float(mean_absolute_error(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0))}


def main() -> int:
    parser=argparse.ArgumentParser(description='Test participant-relative wearable deviations with leakage-safe personal baselines.')
    parser.add_argument('--input',type=Path,default=Path('dataset/harmonized/master/fatigue_daily_training_examples.csv'))
    parser.add_argument('--output',type=Path,default=Path('dataset/harmonized/master/fatigue_person_relative_features.json'))
    parser.add_argument('--gap-days',type=int,default=7)
    args=parser.parse_args()
    with args.input.open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r.get('target_name')=='tired_yesterday' and r.get('eligible_for_modeling','').lower()=='true' and r.get('synthetic','false').lower()!='true']
    feature_names=[k for k in rows[0] if k not in META and k!='target_numeric_source']
    y=np.array([LABELS[r['target_category'].casefold()] for r in rows],int)
    people=[r['participant_id'] for r in rows]; days=[date.fromisoformat(r['target_date']) for r in rows]
    raw=np.array([[float(r[k]) if r.get(k) not in ('',None) else np.nan for k in feature_names] for r in rows],float)
    by=defaultdict(list)
    for i,p in enumerate(people):by[p].append(i)
    z=np.full_like(raw,np.nan)
    history_count=np.zeros(len(rows))
    for p,ix in by.items():
        ordered=sorted(ix,key=lambda i:days[i])
        for pos,i in enumerate(ordered):
            prior=[j for j in ordered[:pos] if days[j] < days[i] and days[j] <= days[i]-timedelta(days=args.gap_days)]
            history_count[i]=len(prior)
            if len(prior)<7: continue
            for col in range(raw.shape[1]):
                vals=raw[prior,col]; vals=vals[np.isfinite(vals)]
                cur=raw[i,col]
                if len(vals)>=7 and np.isfinite(cur):
                    center=np.median(vals); mad=np.median(np.abs(vals-center))*1.4826
                    if mad>1e-9:z[i,col]=(cur-center)/mad
    feature_names_rel=[f'robust_delta__{n}' for n in feature_names]
    X=np.column_stack([z,np.log1p(history_count)])
    date_set=sorted(set(days)); cuts=[date_set[int(len(date_set)*k)] for k in (.60,.72,.84)]; ends=cuts[1:]+[date_set[-1]+timedelta(days=1)]
    folds=[];seen=set()
    for fold,(cut,end) in enumerate(zip(cuts,ends),1):
        train=np.array([i for i,d in enumerate(days) if d<cut-timedelta(days=args.gap_days) and history_count[i]>=7],int)
        test=np.array([i for i,d in enumerate(days) if cut<=d<end and i not in seen and history_count[i]>=7],int);seen.update(map(int,test))
        if len(train)<100 or not len(test): folds.append({'fold':fold,'status':'insufficient','train':len(train),'test':len(test)});continue
        models={'last_history':np.zeros(len(test),int)}
        last=np.array([LABELS[rows[i]['target_category'].casefold()] for i in test])
        # history label is accessible only from earlier chronological target rows and after gap.
        prior_label=np.full(len(test),np.nan)
        for q,i in enumerate(test):
            eligible=[j for j in by[people[i]] if days[j]<days[i]-timedelta(days=args.gap_days)]
            if eligible:prior_label[q]=y[max(eligible,key=lambda j:days[j])]
        fallback=int(np.rint(np.median(y[train])))
        prior_label=np.where(np.isfinite(prior_label),prior_label,fallback).astype(int)
        models['last_history']=prior_label
        wearable_z=X[:,:-1]
        for name,clf in [('extra_trees',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),ExtraTreesClassifier(n_estimators=250,min_samples_leaf=10,max_features=.7,class_weight='balanced',random_state=42,n_jobs=1))),
                         ('logistic',make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=.05,max_iter=1500,class_weight='balanced'))),
                         ('hgb',HistGradientBoostingClassifier(max_iter=100,learning_rate=.04,max_leaf_nodes=5,l2_regularization=5,random_state=42))]:
            clf.fit(wearable_z[train],y[train]);models[name]=clf.predict(wearable_z[test]).astype(int)
        folds.append({'fold':fold,'train_rows':len(train),'test_rows':len(test),'test_people':len(set(people[i] for i in test)),
                      'test_start':cut.isoformat(),'test_end':end.isoformat(),'gap_days':args.gap_days,
                      'metrics':{name:metrics(y[test],pred) for name,pred in models.items()}})
    report={'purpose':'research diagnostic only; relative features use prior participant data only',
            'feature_method':'(current wearable metric - prior participant median) / prior participant 1.4826*MAD; prior values at least 7 days before target; require >=7 prior dates per metric',
            'target':'self-reported tiredness ordinal proxy, not clinical fatigue/Recovery',
            'rows':len(rows),'participants':len(set(people)),'base_metrics':len(feature_names),'min_prior_days':7,'gap_days':args.gap_days,
            'split':'three rolling-origin date blocks, with a 7-day embargo; features from same cohort',
            'folds':folds,'warnings':['Participants without sufficient prior measurement history are excluded from this diagnostic; report coverage before product use.',
                                      'Small late test blocks and repeated self-report measurements make results unstable.',
                                      'No new participant-cohort external validation; do not deploy based on this report.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.output),'folds':folds},ensure_ascii=False,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
