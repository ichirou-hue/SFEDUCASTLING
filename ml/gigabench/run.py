#!/usr/bin/env python3
"""One entry point for prepare, inference, scoring, expert review and publishing."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    for command in ['all','prepare','generate','score','grade']:
        p=sub.add_parser(command)
        p.add_argument('--out',type=Path,required=True)
        p.add_argument('--profile',choices=['smoke','quick','standard'],default='standard')
        p.add_argument('--config',type=Path,default=ROOT/'config/default.json')
        p.add_argument('--legacy-source',type=Path,default=ROOT/'data/private/legacy_cases.jsonl')
        p.add_argument('--without-legacy',action='store_true')
        p.add_argument('--project-root',type=Path)
        p.add_argument('--adapter',type=Path)
        p.add_argument('--backend',choices=['gigachess','mock'],default='gigachess')
        p.add_argument('--stockfish',type=Path)
        p.add_argument('--stage',choices=['automatic','explanations'],default='automatic')
        p.add_argument('--rubrics',type=Path)
        p.add_argument('--ratings',type=Path)
        p.add_argument('--resume',action='store_true')
        p.add_argument('--limit',type=int)
        p.add_argument('--games',type=Path,help='local PGN for real positions and extended M2/M3 (prepare)')
        p.add_argument('--puzzles',type=Path,help='local Lichess puzzle CSV, CC0 (prepare)')
        p.add_argument('--with-judgement',action='store_true',help='add position-judgement tasks; gold from --stockfish (prepare)')
        p.add_argument('--input-profile',default='question_only',
                       help='how a frozen query becomes model input: question_only, contract_v1_fen or file.py:function')
    p=sub.add_parser('import-legacy')
    p.add_argument('--source',type=Path,required=True)
    p=sub.add_parser('rescore',help='recompute metrics of a finished run with the current scoring code')
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'config/default.json');p.add_argument('--stockfish',type=Path)
    p=sub.add_parser('export-heldout',help='list every benchmark position to exclude from training data')
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('package')
    p.add_argument('--out',type=Path,required=True)
    p=sub.add_parser('compare')
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='package':
        from gigabench.publishing import package_public
        print(package_public(ROOT,args.out));return
    if args.command=='export-heldout':
        from gigabench.pipeline import export_heldout
        print(json.dumps(export_heldout(args.run,args.out),ensure_ascii=False));return
    if args.command=='rescore':
        from gigabench.pipeline import rescore
        s=rescore(args)
        print(json.dumps({'report':str(args.out/'report.html'),'blocks':{k:v['mean_primary_percent'] for k,v in s['by_block'].items()}},ensure_ascii=False,indent=2));return
    if args.command=='compare':
        from gigabench.comparison import compare
        print(json.dumps(compare(args.before,args.after,args.out),ensure_ascii=False,indent=2));return
    if args.command=='import-legacy':
        from gigabench.cases import legacy_cases
        rows=legacy_cases(args.source)
        target=ROOT/'data/private/legacy_cases.jsonl'
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(args.source,target)
        print(f'Imported local data: {len(rows)} tasks. Excluded from public package.');return
    if args.limit is not None and args.limit<1:ap.error('--limit must be positive')
    if args.command in ('all','generate') and args.backend=='gigachess' and args.project_root is None:
        ap.error('--project-root required for GigaChess')
    if args.command=='all' and args.stage!='automatic':
        ap.error('Use generate --stage explanations after coach rubric approval')
    from gigabench.pipeline import prepare,generate,score,grade_coaches
    if args.command=='all':
        if not args.without_legacy and not args.stockfish:
            ap.error('--stockfish required for the integrated M3 quality block')
        if args.stockfish and not args.stockfish.is_file():
            ap.error('--stockfish executable does not exist')
        if not args.resume:prepare(args)
        generate(args);result=score(args)
        print(json.dumps({'report':str(args.out/'REPORT_RU.md'),'coverage':result['coverage'],'expert_status':result['explanations_status']},ensure_ascii=False))
    elif args.command=='prepare':prepare(args)
    elif args.command=='generate':generate(args)
    elif args.command=='score':score(args)
    elif args.command=='grade':
        if not args.rubrics or not args.ratings:ap.error('--rubrics and --ratings required')
        print(json.dumps(grade_coaches(args),ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except ModuleNotFoundError as exc:
        if exc.name=='chess':
            raise SystemExit('Install rule dependency first: python -m pip install -r Benchmark/requirements.txt')
        raise
