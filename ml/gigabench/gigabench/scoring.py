import random
from collections import defaultdict
import chess
from .parsing import parse_bool, parse_move, parse_piece, parse_set, parse_scalar, compliant


def answer_class(case):
    """Class of the gold answer for balanced accuracy; None where classes are not meaningful."""
    kind=case['score_type'];gold=case['gold']
    if kind=='bool':return 'да' if gold['answer'] else 'нет'
    if kind=='piece':return 'пусто' if gold['label']=='.' else 'фигура'
    if kind=='mate_move':return 'есть мат' if gold['moves'] else 'нет мата'
    if kind=='judgement':return gold['answer']
    return None


SIDE_OF={'A':'белые','B':'белые','C':'белые','D':'равно','E':'чёрные','F':'чёрные','G':'чёрные'}


def piece_bin(fen,bins):
    count=len(chess.Board(fen).piece_map())
    for name,(low,high) in bins.items():
        if low<=count<=high:return name
    raise ValueError('No piece-count bin for '+str(count))


def score_case(case,response,engine=None,bins=None):
    text=response.get('text','')
    kind=case['score_type'];gold=case['gold']
    result={'id':case['id'],'suite':case['suite'],'task':case['task'],'group_id':case['group_id'],
            'correct':False,'protocol_compliant':compliant(kind,text),
            'empty':not text.strip(),'error':response.get('error'),
            'truncated':bool(response.get('truncated')),'status':'scored',
            'answer_class':answer_class(case)}
    if bins:result['piece_bin']=piece_bin(case['fen'],bins)
    params=case.get('params',{})
    if params.get('strata'):result['strata']=params['strata']
    if params.get('paraphrase_of'):result['variant_of']=params['paraphrase_of']
    if response.get('error'):
        text=''
    if kind=='bool':
        pred=parse_bool(text);result.update(predicted=pred,expected=gold['answer'],correct=pred is not None and pred==gold['answer'])
    elif kind in ('int','choice'):
        pred=parse_scalar(kind,text);result.update(predicted=pred,expected=gold['answer'],correct=pred is not None and pred==gold['answer'])
    elif kind=='judgement':
        pred=parse_scalar(kind,text)
        result.update(predicted=pred,expected=gold['answer'],correct=pred==gold['answer'],
                      side_correct=pred is not None and SIDE_OF[pred]==SIDE_OF[gold['answer']])
    elif kind=='puzzle_move':
        pred=parse_move(case['fen'],text);result.update(predicted=pred,expected=gold['moves'],legal=pred is not None,correct=pred in gold['moves'])
    elif kind=='piece':
        pred=parse_piece(text,gold['square']);result.update(predicted=pred,expected=gold['label'],correct=pred==gold['label'])
    elif kind=='human_move':
        pred=parse_move(case['fen'],text)
        result.update(predicted=pred,expected=gold['move'],legal=pred is not None,correct=pred==gold['move'])
        if engine:
            best=engine.analyse(case['fen']);played=engine.analyse(case['fen'],gold['move'])
            loss=best['cp']-played['cp'] if best['cp'] is not None and played['cp'] is not None else None
            result['reference_quality']='mate_group' if loss is None else ('within_50cp' if loss<=50 else 'loss_over_50cp')
    elif kind=='mate_move':
        pred=parse_move(case['fen'],text);denies=parse_bool(text) is False
        result.update(predicted=pred,expected=gold['moves'],
                      correct=pred in gold['moves'] if gold['moves'] else denies,
                      hallucinated_mate=bool(not gold['moves'] and pred))
    elif kind=='move_set':
        pred=parse_set(text);expected=set(gold['moves']);found=set(pred or [])
        tp=len(found&expected);fp=len(found-expected);fn=len(expected-found)
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else (1.0 if pred is not None else 0.0)
        result.update(predicted=pred,correct=pred is not None and found==expected,
                      precision=tp/(tp+fp) if tp+fp else (1.0 if not expected and pred is not None else 0.0),
                      recall=tp/(tp+fn) if tp+fn else (1.0 if pred is not None else 0.0),
                      f1=f1,tp=tp,fp=fp,fn=fn)
    elif kind=='engine_move':
        pred=parse_move(case['fen'],text)
        if engine is None:
            result.update(correct=None,status='pending_engine',predicted=pred)
        else:
            e=engine.grade(case['fen'],pred)
            result.update(predicted=pred,correct=e['accepted'] if e['ordinary'] else None,
                          engine=e,legal=e['legal'],status='scored' if e['ordinary'] else 'mate_group')
    elif kind=='explanation':
        result.update(correct=None,status='pending_coach_ratings')
    else:
        raise ValueError('Unknown score type '+kind)
    return result


def pct(n,d):
    return 100*n/d if d else None


def plain_accuracy(rows):
    return pct(sum(bool(r['correct']) for r in rows),len(rows))


def balanced_accuracy(rows):
    """Mean of per-class accuracies; None unless at least two answer classes are present."""
    classes=defaultdict(list)
    for r in rows:
        if r.get('answer_class') is not None:classes[r['answer_class']].append(bool(r['correct']))
    if len(classes)<2:return None
    return sum(100*sum(v)/len(v) for v in classes.values())/len(classes)


def interval(rows,config,metric=plain_accuracy):
    groups=defaultdict(list)
    for r in rows:
        if r['correct'] is not None:groups[r['group_id']].append(r)
    names=sorted(groups)
    if not names:return None
    rng=random.Random(config['seed']);values=[]
    for _ in range(config['resamples']):
        sample=[r for __ in names for r in groups[rng.choice(names)]]
        value=metric(sample)
        if value is not None:values.append(value)
    if not values:return None
    values.sort()
    return [values[int(.025*(len(values)-1))],values[int(.975*(len(values)-1))]]


def accuracy(rows,bootstrap,ci=True,detail=True):
    scored=[r for r in rows if r['correct'] is not None]
    result={'records':len(rows),'n':len(scored),'correct':sum(bool(r['correct']) for r in scored),
            'accuracy_percent':plain_accuracy(scored),
            'protocol_compliance_percent':pct(sum(r['protocol_compliant'] for r in rows),len(rows)),
            'empty':sum(r['empty'] for r in rows),'errors':sum(bool(r['error']) for r in rows),
            'truncated':sum(r['truncated'] for r in rows),
            'pending':sum(r['status'].startswith('pending') for r in rows),
            'mate_group':sum(r['status']=='mate_group' for r in rows),
            'cluster_95ci_percent':interval(scored,bootstrap) if ci else None}
    classes=defaultdict(list)
    for r in scored:
        if r.get('answer_class') is not None:classes[r['answer_class']].append(bool(r['correct']))
    if classes:
        result['by_answer_class']={k:{'n':len(v),'correct':sum(v),'accuracy_percent':pct(sum(v),len(v))}
                                   for k,v in sorted(classes.items())}
    balanced=balanced_accuracy(scored)
    result['balanced_accuracy_percent']=balanced
    result['balanced_95ci_percent']=interval(scored,bootstrap,balanced_accuracy) if ci and balanced is not None else None
    result['primary_metric']='balanced_accuracy' if balanced is not None else 'accuracy'
    result['primary_percent']=balanced if balanced is not None else result['accuracy_percent']
    result['single_answer_class']=len(classes)==1
    if detail:
        strata=defaultdict(lambda:defaultdict(list))
        for r in scored:
            for key,value in (r.get('strata') or {}).items():strata[key][value].append(r)
        if strata:
            result['by_strata']={key:{value:{'n':len(v),'primary_percent':accuracy(v,bootstrap,ci=False,detail=False)['primary_percent']}
                                      for value,v in sorted(values.items())} for key,values in sorted(strata.items())}
        variants=defaultdict(list)
        for r in scored:
            if r.get('variant_of'):variants[r['variant_of']].append(bool(r['correct']))
        multi=[v for v in variants.values() if len(v)>1]
        if multi:
            result['paraphrase_groups']=len(multi)
            result['paraphrase_consistency_percent']=pct(sum(len(set(v))==1 for v in multi),len(multi))
    sides=[r['side_correct'] for r in scored if 'side_correct' in r]
    if sides:result['side_correct_percent']=pct(sum(sides),len(sides))
    sets=[r for r in rows if 'f1' in r]
    if sets:result['mean_move_set_f1_percent']=100*sum(r['f1'] for r in sets)/len(sets)
    mates=[r for r in rows if 'hallucinated_mate' in r]
    if mates:
        negatives=[r for r in mates if not r['expected']]
        result['negative_mate_cases']=len(negatives)
        result['false_legal_mating_move_claims']=sum(r['hallucinated_mate'] for r in negatives)
        result['false_legal_mating_move_claim_rate_percent']=pct(result['false_legal_mating_move_claims'],len(negatives))
    targets=[repr(r['expected']) for r in rows if 'expected' in r]
    if targets:
        from collections import Counter
        result['majority_target_baseline_percent']=100*max(Counter(targets).values())/len(targets)
    return result


def piece_f1(rows):
    rows=[r for r in rows if 'expected' in r and isinstance(r['expected'],str) and r['expected'] in 'PNBRQKpnbrqk.']
    per={}
    for label in 'PNBRQKpnbrqk.':
        truth=[r for r in rows if r['expected']==label]
        tp=sum(r.get('predicted')==label for r in truth)
        fp=sum(r.get('predicted')==label and r['expected']!=label for r in rows)
        fn=len(truth)-tp
        per[label]={'support':len(truth),'tp':tp,'fp':fp,'fn':fn,
                    'f1_percent':100*2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
    used=[r['f1_percent'] for r in per.values() if r['support']]
    return {'macro_f1_percent':sum(used)/len(used) if used else None,'by_label':per}


def matches(name,pattern):
    return name==pattern or (pattern.endswith('/*') and name.startswith(pattern[:-1]))


def block_tasks(names,config):
    """Map every reported task to exactly one block; combined tasks replace their parts."""
    combined=config.get('combined_tasks',{})
    parts={p for v in combined.values() for p in v}
    reported=sorted(set(n for n in names if n not in parts)|{k for k,v in combined.items() if set(v)&set(names)})
    blocks={b:[] for b in config['blocks']}
    for name in reported:
        owners=[b for b,patterns in config['blocks'].items() if any(matches(name,p) for p in patterns)]
        if len(owners)!=1:raise ValueError(f'Task {name} must belong to exactly one block, found {owners}')
        blocks[owners[0]].append(name)
    return blocks


def summarize(rows,config):
    by_task=defaultdict(list);by_suite=defaultdict(list)
    for r in rows:by_task[r['suite']+'/'+r['task']].append(r);by_suite[r['suite']].append(r)
    for name,members in config.get('combined_tasks',{}).items():
        joined=[r for m in members for r in by_task.get(m,[])]
        if joined:by_task[name]=joined
    tasks={k:accuracy(v,config['bootstrap']) for k,v in sorted(by_task.items())}
    suites={k:accuracy(v,config['bootstrap']) for k,v in sorted(by_suite.items())}
    blocks={}
    for block,names in block_tasks(list(by_task),config).items():
        values=[tasks[n]['primary_percent'] for n in names if tasks[n]['n'] and tasks[n]['primary_percent'] is not None]
        bins={}
        for b in config.get('piece_bins',{}):
            per=[accuracy([r for r in by_task[n] if r.get('piece_bin')==b],config['bootstrap'],ci=False) for n in names]
            per=[m['primary_percent'] for m in per if m['n'] and m['primary_percent'] is not None]
            bins[b]=sum(per)/len(per) if per else None
        blocks[block]={'tasks':names,'scored_tasks':len(values),
                       'mean_primary_percent':sum(values)/len(values) if values else None,
                       'by_piece_bin':bins,'role':config.get('block_roles',{}).get(block)}
    auto=[v['accuracy_percent'] for k,v in tasks.items() if v['n'] and k not in ('legacy/M3','legacy/M4_M7')
          and k not in config.get('combined_tasks',{})]
    legacy=[r for r in rows if r['suite']=='legacy']
    m1=[r for r in legacy if r['task'].startswith('M1_')]
    m2=[r for r in legacy if r['task']=='M2']
    m3=[r for r in legacy if r['task']=='M3']
    document={'M1':accuracy(m1,config['bootstrap']) if m1 else None,
              'M1_F1':piece_f1(m1) if m1 else None,
              'M2':accuracy(m2,config['bootstrap']) if m2 else None,
              'M3':accuracy(m3,config['bootstrap']) if m3 else None,
              'M4':None,'M5':None,'M6':None,'M7':None,'accepted_explanations_percent':None}
    if m3:document['M3']['legal_percent']=pct(sum(r.get('legal',False) for r in m3),len(m3))
    return {'by_block':blocks,'by_suite':suites,'by_task':tasks,'document_metrics':document,
            'automatic_task_macro_percent':sum(auto)/len(auto) if auto else None,
            'automatic_macro_scope':config['aggregate'],
            'explanations_status':'pending_coach_review',
            'not_an_overall_teaching_score':True}
