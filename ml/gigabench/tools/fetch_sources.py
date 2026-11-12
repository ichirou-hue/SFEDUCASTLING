#!/usr/bin/env python3
"""Download the public sources of v1.1 and freeze small subsets of them in data/sources/.

    python Benchmark/tools/fetch_sources.py --work /tmp/gigabench_sources

Sources (all checked by SHA256 where the publisher gives one):
- Lichess standard rated games, January 2013 (CC0);
- Lichess puzzle database (CC0; regenerated monthly, so its SHA256 is recorded, not pinned);
- Stockfish 19 for macOS or Linux (GPLv3), used to compute gold for position judgement
  and to grade M3. The binary is not stored in the repository.

The frozen subsets (data/sources/games_v1_1.json, data/sources/puzzles_v1_1.json) are what
`prepare --games/--puzzles` reads, so the benchmark rebuilds identically without downloading.
"""
import argparse
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import chess.pgn
from gigabench.external import eligible_puzzle, rating_bin

GAMES_URL='https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst'
GAMES_SUMS='https://database.lichess.org/standard/sha256sums.txt'
PUZZLES_URL='https://database.lichess.org/lichess_db_puzzle.csv.zst'
STOCKFISH={'Darwin':('https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-macos-universal.tar.gz',
                     'a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77')}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def download(url,target):
    if not target.exists():
        print('download',url,flush=True)
        with urllib.request.urlopen(url) as r,open(target,'xb') as out:shutil.copyfileobj(r,out)
    return target


def unzstd(path):
    out=path.with_suffix('')
    if not out.exists():subprocess.run(['zstd','-d','-q',str(path),'-o',str(out)],check=True)
    return out


def freeze_games(pgn,source_sha,config,destination):
    """The first rated games with both ratings and at least 30 plies, in file order."""
    wanted=config['frozen_sources']['games'];games=[]
    with open(pgn,encoding='utf-8',errors='replace') as stream:
        while len(games)<wanted:
            game=chess.pgn.read_game(stream)
            if game is None:break
            h=game.headers;moves=[m.uci() for m in game.mainline_moves()]
            if len(moves)<30 or not h.get('WhiteElo','').isdigit() or not h.get('BlackElo','').isdigit():continue
            keep={k:h[k] for k in ('Event','Site','Date','White','Black','Result','WhiteElo','BlackElo','TimeControl','Termination') if k in h}
            games.append({'headers':keep,'moves':moves})
    destination.write_text(json.dumps({'source':GAMES_URL,'source_sha256':source_sha,'license':'CC0 (Lichess database exports)',
        'selection':f'first {wanted} rated games with both ratings and at least 30 plies, in file order','games':games},ensure_ascii=False)+'\n',encoding='utf-8')
    return len(games)


def freeze_puzzles(csv_path,source_sha,config,destination):
    """Seeded reservoir per rating bin among eligible puzzles; per bin as many as the standard profile uses."""
    import csv
    per_bin=config['puzzles_per_bin']['standard'];quality=config['puzzle_quality']
    rng=random.Random(config['seed']+13);pools={};seen={}
    with open(csv_path,encoding='utf-8',newline='') as stream:
        for row in csv.DictReader(stream):
            if not eligible_puzzle(row,quality):continue
            name=rating_bin(int(row['Rating']));seen[name]=seen.get(name,0)+1;pool=pools.setdefault(name,[])
            if len(pool)<per_bin:pool.append(row)
            else:
                j=rng.randrange(seen[name])
                if j<per_bin:pool[j]=row
    rows=[r for name in sorted(pools) for r in sorted(pools[name],key=lambda r:r['PuzzleId'])]
    destination.write_text(json.dumps({'source':PUZZLES_URL,'source_sha256':source_sha,'license':'CC0 (Lichess database exports)',
        'selection':{'per_rating_bin':per_bin,'quality':quality,'seed':config['seed']+13,'eligible_per_bin':seen},
        'puzzles':rows},ensure_ascii=False)+'\n',encoding='utf-8')
    return len(rows)


def stockfish(work):
    system=platform.system()
    if system not in STOCKFISH:raise SystemExit('Download Stockfish 19 for this platform manually and pass --stockfish')
    url,digest=STOCKFISH[system];archive=download(url,work/Path(url).name)
    if sha(archive)!=digest:raise SystemExit('Stockfish archive SHA256 mismatch')
    target=work/'stockfish'
    if not target.exists():
        with tarfile.open(archive) as t:
            member=next(m for m in t.getmembers() if m.isfile() and Path(m.name).name.startswith('stockfish') and '/src/' not in m.name and m.mode&0o100)
            member.name=Path(member.name).name;t.extract(member,target,filter='data')
    return next(p for p in target.iterdir() if p.is_file())


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work',type=Path,required=True,help='folder for downloads (outside the repository)')
    ap.add_argument('--config',type=Path,default=ROOT/'config/default.json')
    args=ap.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    config=json.loads(args.config.read_text(encoding='utf-8'))
    out=ROOT/'data/sources';out.mkdir(parents=True,exist_ok=True)
    sums=download(GAMES_SUMS,args.work/'sha256sums_standard.txt').read_text()
    games=download(GAMES_URL,args.work/Path(GAMES_URL).name)
    expected=next(line.split()[0] for line in sums.splitlines() if line.endswith(Path(GAMES_URL).name))
    if sha(games)!=expected:raise SystemExit('Lichess games SHA256 mismatch')
    n=freeze_games(unzstd(games),expected,config,out/'games_v1_1.json')
    puzzles=download(PUZZLES_URL,args.work/Path(PUZZLES_URL).name)
    m=freeze_puzzles(unzstd(puzzles),sha(puzzles),config,out/'puzzles_v1_1.json')
    engine=stockfish(args.work)
    print(json.dumps({'games':n,'puzzles':m,'stockfish':str(engine),
                      'next':f'python {ROOT.name}/run.py prepare --out ... --games {out/"games_v1_1.json"} --puzzles {out/"puzzles_v1_1.json"} --with-judgement --stockfish {engine}'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
