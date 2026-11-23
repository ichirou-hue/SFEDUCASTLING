import json
import platform
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import chess
from . import __version__
from .adapters import GigaChessAdapter, MockAdapter
from .cases import build
from .coaches import pending_rubric, validate_rubrics, score_one
from .engine import EngineJudge
from .io import read_jsonl, write_jsonl, write_json, sha, digest, unique
from .scoring import score_case, summarize
from .report import render_html

PACKAGE = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    paths=list((PACKAGE/'gigabench').glob('*.py'))+[PACKAGE/'run.py',PACKAGE/'config/default.json']
    return {str(p.relative_to(PACKAGE)).replace('\\','/'):sha(p) for p in sorted(paths)}


def prepare(args):
    out=Path(args.out)
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    legacy=None if args.without_legacy else Path(args.legacy_source)
    if legacy is not None and not legacy.is_file():
        raise FileNotFoundError('Import local 89 records first with import-legacy, or explicitly select --without-legacy')
    games=getattr(args,'games',None);puzzles=getattr(args,'puzzles',None)
    judge=args.stockfish if getattr(args,'with_judgement',False) else None
    if getattr(args,'with_judgement',False) and not args.stockfish:raise ValueError('--with-judgement needs --stockfish to compute gold')
    for path in (games,puzzles):
        if path is not None and not Path(path).is_file():raise FileNotFoundError(path)
    cases=build(config,args.profile,legacy,games,puzzles,judge)
    out.mkdir(parents=True,exist_ok=False)
    queries=[{'id':c['id'],'fen':c['fen'],'prompt':c['prompt'],'max_new_tokens':c['max_new_tokens']}
             for c in cases if c['score_type']!='explanation']
    write_jsonl(out/'cases.jsonl',cases)
    write_jsonl(out/'queries.jsonl',queries)
    write_jsonl(out/'coach_rubrics_pending.jsonl',[pending_rubric(c) for c in cases if c['score_type']=='explanation'])
    manifest={'version':__version__,'created_utc':now(),'profile':args.profile,'config':config,
              'code_sha256':source_hashes(),'cases_sha256':sha(out/'cases.jsonl'),
              'queries_sha256':sha(out/'queries.jsonl'),'case_count':len(cases),
              'automatic_queries':len(queries),'explanation_cases':len(cases)-len(queries),
              'tasks':dict(Counter(c['suite']+'/'+c['task'] for c in cases)),
              'contains_local_source_comments':bool(legacy),
              'source_legacy_sha256':sha(legacy) if legacy else None,
              'external_sources':{'games_pgn_sha256':sha(games) if games else None,
                                  'puzzles_csv_sha256':sha(puzzles) if puzzles else None,
                                  'judgement_engine_sha256':sha(judge) if judge else None,
                                  'missing':[name for name,v in (('real games (M2/M3 extension, real positions)',games),
                                             ('Lichess puzzles',puzzles),('position judgement',judge)) if not v]},
              'chess_version':chess.__version__,'chess_module_sha256':sha(chess.__file__),
              'gold_sources':'rules, independent synthetic legal-play/KQK; local PGN references if enabled',
              'upstream_comparability':'Inspired/adapted diagnostic, not official ChessQA/Random-Chess results'}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'prepared':str(out),'cases':len(cases),'automatic':len(queries),'pending_explanations':len(cases)-len(queries)},ensure_ascii=False))
    return manifest


def verify(out):
    out=Path(out)
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if sha(out/'cases.jsonl')!=manifest['cases_sha256'] or sha(out/'queries.jsonl')!=manifest['queries_sha256']:
        raise ValueError('Frozen dataset modified')
    if manifest['code_sha256']!=source_hashes():raise ValueError('Benchmark code changed since dataset freeze; create a new run')
    if manifest['chess_version']!=chess.__version__ or manifest['chess_module_sha256']!=sha(chess.__file__):
        raise ValueError('Rule library differs from frozen gold construction')
    cases=read_jsonl(out/'cases.jsonl');unique(cases)
    return manifest,cases


def generate(args):
    out=Path(args.out);manifest,cases=verify(out)
    rubric_sha=None
    if args.stage=='explanations':
        if args.rubrics is None:raise ValueError('Preapproved coach rubrics required BEFORE explanation inference')
        validate_rubrics(read_jsonl(args.rubrics),cases)
        rubric_sha=sha(args.rubrics)
        queries=[{'id':c['id'],'fen':c['fen'],'prompt':c['prompt'],'max_new_tokens':c['max_new_tokens']}
                 for c in cases if c['score_type']=='explanation']
        response_path=out/'explanation_responses.jsonl'
        metadata_path=out/'explanation_run.json'
    else:
        queries=read_jsonl(out/'queries.jsonl')
        response_path=out/'responses.jsonl';metadata_path=out/'run.json'
    existing=[]
    if response_path.exists():
        if not args.resume:raise FileExistsError('Responses exist; use --resume or a new output')
        existing=read_jsonl(response_path)
    by_id=unique(existing)
    if set(by_id)-{q['id'] for q in queries}:raise ValueError('Unexpected response IDs')
    profile=getattr(args,'input_profile',None) or 'question_only'
    adapter=(MockAdapter(profile) if args.backend=='mock' else
             GigaChessAdapter(args.project_root,manifest['config']['generation'],args.adapter,profile))
    metadata={'backend':adapter.metadata,'started_utc':now(),'stage':args.stage,
              'manifest_sha256':sha(out/'manifest.json'),'rubrics_sha256':rubric_sha,
              'status':'running','python':platform.python_version()}
    if args.resume and metadata_path.exists():
        previous=json.loads(metadata_path.read_text(encoding='utf-8'))
        if previous['backend']!=metadata['backend'] or previous['manifest_sha256']!=metadata['manifest_sha256'] or previous.get('rubrics_sha256')!=rubric_sha:
            raise ValueError('Cannot resume with another model, rubric or benchmark')
        metadata['started_utc']=previous['started_utc']
    write_json(metadata_path,metadata)
    remaining=[q for q in queries if q['id'] not in by_id]
    if args.limit:remaining=remaining[:args.limit]
    start=time.monotonic()
    with response_path.open('a' if response_path.exists() else 'x',encoding='utf-8') as stream:
        for i,q in enumerate(remaining,1):
            t=time.monotonic()
            r={'id':q['id']}
            try:r.update(adapter.predict(q))
            except Exception as exc:r.update(text='',error={'type':type(exc).__name__,'message':str(exc)},truncated=False)
            r['elapsed_seconds']=time.monotonic()-t
            stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            if i%10==0 or i==len(remaining):
                print(f'GENERATED {len(existing)+i}/{len(queries)} elapsed={time.monotonic()-start:.1f}s',flush=True)
    metadata.update(completed_utc=now(),status='complete' if len(existing)+len(remaining)==len(queries) else 'partial',
                    responses_sha256=sha(response_path),completed=len(existing)+len(remaining),expected=len(queries))
    write_json(metadata_path,metadata)
    return metadata


def score(args):
    out=Path(args.out);manifest,cases=verify(out)
    responses=read_jsonl(out/'responses.jsonl');raw=unique(responses)
    run=json.loads((out/'run.json').read_text(encoding='utf-8'))
    if run['manifest_sha256']!=sha(out/'manifest.json') or run['responses_sha256']!=sha(out/'responses.jsonl'):
        raise ValueError('Responses/run metadata mismatch')
    expected={c['id'] for c in cases if c['score_type']!='explanation'}
    if set(raw)-expected:raise ValueError('Unknown response IDs')
    engine=None
    if any(c['score_type']=='engine_move' and c['id'] in raw for c in cases):
        if not args.stockfish:raise ValueError('--stockfish required for recommendation quality')
        engine=EngineJudge(args.stockfish,manifest['config']['engine'],out/'engine')
    try:rows=score_rows(cases,raw,engine,manifest['config'])
    finally:
        if engine:engine.close()
    summary=summarize(rows,manifest['config'])
    summary.update(manifest_sha256=sha(out/'manifest.json'),responses_sha256=sha(out/'responses.jsonl'),
                   is_mock=bool(run['backend'].get('not_a_model_evaluation')),
                   coverage={'completed':len(raw),'expected':len(expected),'complete':set(raw)==expected},
                   expert_cases=manifest['explanation_cases'],
                   upstream_comparability=manifest['upstream_comparability'])
    if len(raw)!=len(expected):summary['automatic_task_macro_percent']=None
    if manifest['explanation_cases']==0:summary['explanations_status']='not_included'
    write_jsonl(out/'scores.jsonl',rows);write_json(out/'summary.json',summary)
    render_report(out,summary,manifest['config'])
    return summary


def score_rows(cases,raw,engine,config,previous=None):
    """Score every automatic case. `previous` supplies stored engine verdicts when no engine is attached."""
    rows=[];bins=config.get('piece_bins')
    for i,c in enumerate(cases,1):
        if c['score_type']=='explanation':continue
        if c['id'] not in raw:
            rows.append({'id':c['id'],'suite':c['suite'],'task':c['task'],'group_id':c['group_id'],
                         'correct':None,'protocol_compliant':False,'empty':False,'error':None,
                         'truncated':False,'status':'not_run','answer_class':None})
            continue
        row=score_case(c,raw[c['id']],engine,bins)
        if engine is None and previous and c['id'] in previous:
            old=previous[c['id']]
            if c['score_type']=='engine_move':
                if old.get('predicted')!=row.get('predicted'):raise ValueError('Stored engine verdict belongs to another parsed move: '+c['id'])
                row.update({k:old[k] for k in ('correct','engine','legal','status') if k in old})
            elif c['score_type']=='human_move' and 'reference_quality' in old:
                row['reference_quality']=old['reference_quality']
        rows.append(row)
        if i%200==0:print(f'SCORED {i}/{len(cases)}',flush=True)
    return rows


def export_heldout(run,destination):
    """Every benchmark position, for exclusion from all training data."""
    run=Path(run);manifest,cases=verify(run)
    keys={}
    for c in cases:
        fields=c['fen'].split()
        keys.setdefault(' '.join(fields[:4]),{'position_key':' '.join(fields[:4]),'placement':fields[0],'case_ids':[]})['case_ids'].append(c['id'])
    result={'benchmark_version':manifest['version'],'cases_sha256':manifest['cases_sha256'],
            'match_rule':'Exclude a training example whose FEN has the same first four fields (position_key); '
                         'stricter: the same placement regardless of side to move and castling rights.',
            'positions':sorted(keys.values(),key=lambda r:r['position_key'])}
    write_json(destination,result)
    return {'positions':len(keys),'path':str(destination)}


def rescore(args):
    """Recompute metrics for a finished run with the current scoring code.

    Cases, queries and responses are copied unchanged and checked against the source manifest.
    Without --stockfish, M3 engine verdicts are reused from the source scores.
    The result is labelled as a re-scoring, never as a new model run.
    """
    source=Path(args.source);out=Path(args.out)
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if sha(source/'cases.jsonl')!=manifest['cases_sha256'] or sha(source/'queries.jsonl')!=manifest['queries_sha256']:
        raise ValueError('Source dataset modified')
    run=json.loads((source/'run.json').read_text(encoding='utf-8'))
    if run['responses_sha256']!=sha(source/'responses.jsonl'):raise ValueError('Source responses modified')
    out.mkdir(parents=True,exist_ok=False)
    for name in ('cases.jsonl','queries.jsonl','responses.jsonl','run.json','manifest.json'):
        shutil.copyfile(source/name,out/name)
    if (source/'engine/engine.json').exists() and not args.stockfish:
        (out/'engine').mkdir();shutil.copyfile(source/'engine/engine.json',out/'engine/engine.json')
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    for key in ('seed','bootstrap','engine'):
        if config[key]!=manifest['config'][key]:raise ValueError('Re-scoring cannot change '+key)
    cases=read_jsonl(out/'cases.jsonl');raw=unique(read_jsonl(out/'responses.jsonl'))
    previous=unique(read_jsonl(source/'scores.jsonl')) if (source/'scores.jsonl').exists() else None
    engine=EngineJudge(args.stockfish,manifest['config']['engine'],out/'engine') if args.stockfish else None
    try:rows=score_rows(cases,raw,engine,config,previous)
    finally:
        if engine:engine.close()
    expected={c['id'] for c in cases if c['score_type']!='explanation'}
    summary=summarize(rows,config)
    summary.update(manifest_sha256=sha(out/'manifest.json'),responses_sha256=sha(out/'responses.jsonl'),
                   is_mock=bool(run['backend'].get('not_a_model_evaluation')),
                   coverage={'completed':len(raw),'expected':len(expected),'complete':set(raw)==expected},
                   expert_cases=manifest['explanation_cases'],
                   upstream_comparability=manifest['upstream_comparability'],
                   rescored={'source':str(source),'cases_version':manifest['version'],'scoring_version':__version__,
                             'engine_verdicts':'пересчитаны движком' if engine else 'взяты из исходного прогона',
                             'note':'Same cases and responses as the source run; only metrics were recomputed.'})
    if not summary['coverage']['complete']:summary['automatic_task_macro_percent']=None
    write_jsonl(out/'scores.jsonl',rows);write_json(out/'summary.json',summary)
    render_report(out,summary,config)
    return summary


def render_report(out,s,config=None):
    if config is None:config=json.loads((Path(out)/'manifest.json').read_text(encoding='utf-8'))['config']
    names=config.get('block_names',{})
    two=lambda v:'—' if v is None else f'{v:.2f}'
    lines=['# GigaChess Benchmark','',
           '**MOCK: технический прогон, не результат модели.**' if s['is_mock'] else 'Измерения модели.','',
           f"Покрытие: {s['coverage']['completed']}/{s['coverage']['expected']} автоматических запросов.",
           f"Полный автоматический прогон: {s['coverage']['complete']}.",
           f"Статус объяснений: {s['explanations_status']}.",'']
    if s.get('rescored'):lines+=[f"Пересчёт метрик: {s['rescored']['note']} Оценки движка: {s['rescored']['engine_verdicts']}.",'']
    lines+=['## Блоки','','| Блок | Роль | Среднее основной метрики % | Задач |','|---|---|---:|---:|']
    for block,v in s['by_block'].items():
        lines.append(f"| {names.get(block,block)} | {v['role'] or '—'} | {two(v['mean_primary_percent'])} | {v['scored_tasks']} |")
    lines+=['','Основная метрика — сбалансированная точность там, где у эталона есть классы (да/нет, пусто/фигура, мат/нет мата), иначе обычная точность.','',
           '## Задачи','',
           '| Задача | Верных / оценённых | Основная метрика % | Точность % | Постоянный ответ % | Формат % |',
           '|---|---:|---:|---:|---:|---:|']
    for name,m in s['by_task'].items():
        lines.append(f"| {name} | {m['correct']}/{m['n']} | {two(m['primary_percent'])} | {two(m['accuracy_percent'])} | {two(m.get('majority_target_baseline_percent'))} | {two(m['protocol_compliance_percent'])} |")
    lines+=['','Автоматический macro-балл: '+str(s['automatic_task_macro_percent'])+'%.',
            s['automatic_macro_scope'],'',
            '## Метрики из документа','',
            'М4–М7 и общий процент принятых объяснений не заменяются сходством текстов или оценкой движка.']
    for key,value in s['document_metrics'].items():
        if key=='M1_F1':
            lines.append(f"- Macro-F1 М1: {value['macro_f1_percent'] if value else 'не измерен'}.")
        elif isinstance(value,dict):
            lines.append(f"- {key}: {value['accuracy_percent']}%, {value['correct']}/{value['n']}; матовая группа: {value['mate_group']}.")
        else:lines.append(f'- {key}: {value if value is not None else "ожидает данных/экспертных оценок"}.')
    lines+=['','Это адаптированный диагностический набор, не официальный результат ChessQA или Random-Chess.',
            'Правильность формата, корректность содержания и качество объяснения — разные показатели.',
            'При ограниченном поиске Stockfish оценки не являются доказательством всех стратегических утверждений.',
            'Сохраняйте этот test вне обучения и сравнивайте версии на одинаковом manifest.']
    if s.get('coach_metrics'):
        lines+=['','## Экспертные результаты','',json.dumps(s['coach_metrics'],ensure_ascii=False,indent=2)]
    (Path(out)/'REPORT_RU.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if (Path(out)/'scores.jsonl').exists():render_html(out,s,config)


def grade_coaches(args):
    out=Path(args.out);manifest,cases=verify(out)
    rubrics=validate_rubrics(read_jsonl(args.rubrics),cases)
    run=json.loads((out/'explanation_run.json').read_text(encoding='utf-8'))
    if run['rubrics_sha256']!=sha(args.rubrics):raise ValueError('Rubrics changed after explanation generation')
    if run['responses_sha256']!=sha(out/'explanation_responses.jsonl'):raise ValueError('Explanation responses modified')
    responses=unique(read_jsonl(out/'explanation_responses.jsonl'));ratings=unique(read_jsonl(args.ratings))
    if set(responses)!=set(rubrics) or set(ratings)!=set(rubrics):raise ValueError('Need complete, unique ratings for all explanation cases')
    rows=[score_one(responses[key]['text'],rubrics[key],ratings[key]) for key in sorted(rubrics)]
    n=len(rows)
    if not n:raise ValueError('No explanation cases')
    metrics={k:100*sum(r[k] for r in rows)/n for k in ('M4','M5','M6','M7')}
    metrics.update(n=n,accepted_explanations_percent=100*sum(r['accepted'] for r in rows)/n,
                   ratings_sha256=sha(args.ratings),rubrics_sha256=sha(args.rubrics))
    write_jsonl(out/'coach_scores.jsonl',rows);write_json(out/'coach_metrics.json',metrics)
    if (out/'summary.json').exists():
        summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        summary['coach_metrics']=metrics;summary['explanations_status']='coach_review_complete'
        for key in ('M4','M5','M6','M7','accepted_explanations_percent'):summary['document_metrics'][key]=metrics[key]
        write_json(out/'summary.json',summary);render_report(out,summary)
    return metrics
