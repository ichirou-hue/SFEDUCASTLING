import json
import random
from collections import defaultdict
from pathlib import Path
from .io import read_jsonl, unique, write_json


def paired_interval(rows,budget=2000,seed=20261101):
    groups=defaultdict(list)
    for group,difference in rows:groups[group].append(difference)
    if not groups:return None
    keys=sorted(groups);rng=random.Random(seed);samples=[]
    for _ in range(budget):
        sample=[d for _ in keys for d in groups[rng.choice(keys)]]
        samples.append(100*sum(sample)/len(sample))
    samples.sort()
    return [samples[int(.025*(len(samples)-1))],samples[int(.975*(len(samples)-1))]]


def compare(before,after,destination):
    before,after=Path(before),Path(after)
    a=json.loads((before/'manifest.json').read_text(encoding='utf-8'))
    b=json.loads((after/'manifest.json').read_text(encoding='utf-8'))
    for key in ('cases_sha256','queries_sha256','code_sha256','config','chess_module_sha256'):
        if a[key]!=b[key]:raise ValueError('Cannot compare different benchmarks: '+key)
    sa=json.loads((before/'summary.json').read_text(encoding='utf-8'))
    sb=json.loads((after/'summary.json').read_text(encoding='utf-8'))
    if sa['is_mock'] or sb['is_mock']:raise ValueError('Mock runs are not model evaluations')
    if not sa['coverage']['complete'] or not sb['coverage']['complete']:raise ValueError('Complete inference required')
    if (before/'engine/engine.json').exists() or (after/'engine/engine.json').exists():
        ea=json.loads((before/'engine/engine.json').read_text());eb=json.loads((after/'engine/engine.json').read_text())
        if ea!=eb:raise ValueError('Engine version, networks or analysis budget differs')
    ra=unique(read_jsonl(before/'scores.jsonl'));rb=unique(read_jsonl(after/'scores.jsonl'))
    if set(ra)!=set(rb):raise ValueError('Different scored case IDs')
    tasks=defaultdict(list)
    for key,x in ra.items():
        y=rb[key]
        if x['correct'] is not None and y['correct'] is not None:
            tasks[x['suite']+'/'+x['task']].append((x['group_id'],int(y['correct'])-int(x['correct'])))
    profiles=[json.loads((p/'run.json').read_text(encoding='utf-8'))['backend'].get('input_profile','question_only') for p in (before,after)]
    result={'before':str(before),'after':str(after),'input_profiles':{'before':profiles[0],'after':profiles[1]},
            'paired_task_deltas':{},'expert_comparison':None}
    for task,rows in sorted(tasks.items()):
        result['paired_task_deltas'][task]={
            'paired_n':len(rows),'delta_percentage_points':100*sum(d for _,d in rows)/len(rows),
            'cluster_95ci_percentage_points':paired_interval(rows),
            'before_metric_n':sa['by_task'][task]['n'],'after_metric_n':sb['by_task'][task]['n']}
    if sa.get('coach_metrics') and sb.get('coach_metrics'):
        if sa['coach_metrics']['rubrics_sha256']!=sb['coach_metrics']['rubrics_sha256']:
            raise ValueError('Expert gold rubric differs')
        result['expert_comparison']={k:sb['coach_metrics'][k]-sa['coach_metrics'][k]
             for k in ('M4','M5','M6','M7','accepted_explanations_percent')}
    result['notes']=['Different input profiles are allowed: the same frozen questions and gold, each model with its own input format.',
                     'Positive delta means more correct answers; M6 error rate is lower-is-better.',
                     'Engine mate/cp denominator changes are shown, not silently treated as comparable binary scores.',
                     'Expert deltas are descriptive; independent raw coach scores remain available.']
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    write_json(destination,result)
    return result
