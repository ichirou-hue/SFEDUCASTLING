"""Tasks built from local files supplied at prepare time: real games, Lichess puzzles, engine judgement.

Nothing is downloaded. Each source file is fixed in the manifest by SHA256, and every
case records where it came from. Without the files these tasks are simply absent and the
manifest says so.
"""
import csv
import json
import random
from pathlib import Path
import chess
import chess.engine
import chess.pgn
from .generated import board_tasks, case
from .oracles import mating_moves, position_key

BUCKETS=[('A','У белых решающий перевес',300,None),('B','У белых заметный перевес',100,300),
         ('C','У белых небольшой перевес',30,100),('D','Примерно равная позиция',-30,30),
         ('E','У чёрных небольшой перевес',-100,-30),('F','У чёрных заметный перевес',-300,-100),
         ('G','У чёрных решающий перевес',None,-300)]
SIDE_OF={'A':'белые','B':'белые','C':'белые','D':'равно','E':'чёрные','F':'чёрные','G':'чёрные'}


def bucket(cp,margin):
    """Bucket letter for a white-perspective score, or None when it lies within `margin` of a boundary."""
    for letter,_,low,high in BUCKETS:
        if (low is None or cp>=low) and (high is None or cp<high):
            if (low is not None and cp-low<margin) or (high is not None and high-cp<=margin):return None
            return letter
    return None


def elo_bin(headers):
    try:value=(int(headers.get('WhiteElo'))+int(headers.get('BlackElo')))/2
    except (TypeError,ValueError):return 'рейтинг неизвестен'
    return 'до 1600' if value<1600 else ('1600–2000' if value<2000 else '2000 и выше')


def phase(board,ply):
    return 'дебют' if ply<=20 else ('эндшпиль' if len(board.piece_map())<=12 else 'миттельшпиль')


def game_positions(path,limit,rng,per_game=2,min_ply=8):
    """Mainline positions with the move actually played, at most `per_game` per game."""
    found=[];seen=set()
    for index,(headers,moves) in enumerate(iter_games(path)):
        if len(found)>=limit*4:break
        if len(moves)<min_ply+2:continue
        board=chess.Board()
        plies=rng.sample(range(min_ply,len(moves)),min(per_game,len(moves)-min_ply))
        for ply,move in enumerate(moves):
            if ply in plies and position_key(board) not in seen and not board.is_game_over():
                seen.add(position_key(board))
                found.append({'board':board.copy(),'move':move,'ply':ply,'game':index,'headers':headers})
            board.push(move)
    rng.shuffle(found)
    return found[:limit]


def iter_games(path):
    """(headers, mainline moves) from a PGN file or from a frozen JSON subset made by tools/fetch_sources.py."""
    if str(path).endswith('.json'):
        for g in json.loads(Path(path).read_text(encoding='utf-8'))['games']:
            yield g['headers'],[chess.Move.from_uci(m) for m in g['moves']]
        return
    with open(path,encoding='utf-8',errors='replace') as stream:
        while True:
            game=chess.pgn.read_game(stream)
            if game is None:return
            yield dict(game.headers),list(game.mainline_moves())


def iter_puzzles(path):
    """Rows of the Lichess puzzle CSV, or of a frozen JSON subset made by tools/fetch_sources.py."""
    if str(path).endswith('.json'):
        yield from json.loads(Path(path).read_text(encoding='utf-8'))['puzzles'];return
    with open(path,encoding='utf-8',newline='') as stream:
        yield from csv.DictReader(stream)


def game_cases(path,config,profile,rng):
    """Board reading on real positions, plus M2 (the human move) and M3 (engine-graded recommendation)."""
    rows=[];k=config['paraphrases'][profile]
    for i,p in enumerate(game_positions(path,config['game_positions'][profile],rng)):
        board=p['board'];group=f"game:{p['game']:05}"
        prov={'source':'local_pgn','game_index':p['game'],'ply':p['ply'],
              'event':p['headers'].get('Event'),'site':p['headers'].get('Site')}
        strata={'рейтинг':elo_bin(p['headers']),'стадия':phase(board,p['ply'])}
        board_tasks(rows,board,f'{group}:{p["ply"]:03}',prov,rng,i,k,'партия',cluster=group)
        rows.append(case(f'{group}:{p["ply"]:03}:M2','games','M2',board,
            'Какой следующий ход здесь сыграл бы человек? Ответь только одним ходом в формате UCI, без пояснений.',
            'human_move',{'move':p['move'].uci()},group,{'strata':strata},prov,16))
        rows.append(case(f'{group}:{p["ply"]:03}:M3','games','M3',board,
            'Порекомендуй хороший ход в этой позиции. Ответь только одним ходом в формате UCI, без пояснений.',
            'engine_move',{},group,{'reference_move_for_strata':p['move'].uci(),'strata':strata},prov,16))
    return rows


def eligible_puzzle(row,quality):
    return (int(row['NbPlays'])>=quality['min_plays'] and int(row['RatingDeviation'])<=quality['max_rating_deviation']
            and int(row['Popularity'])>=quality['min_popularity'])


def rating_bin(rating):
    return 'до 1200' if rating<1200 else ('1200–1600' if rating<1600 else ('1600–2000' if rating<2000 else '2000 и выше'))


def puzzle_cases(path,config,profile,rng):
    """Lichess puzzle database CSV (CC0). The first listed move is the opponent's; the second is the answer.

    Streams the file (millions of rows) and keeps a seeded reservoir per rating bin. Only
    well-established puzzles are eligible: enough plays, settled rating, not disliked.
    """
    q=config['puzzle_quality'];per_bin=config['puzzles_per_bin'][profile]
    pools={};seen={}
    for row in iter_puzzles(path):
        if not eligible_puzzle(row,q):continue
        name=rating_bin(int(row['Rating']));seen[name]=seen.get(name,0)+1
        pool=pools.setdefault(name,[])
        if len(pool)<per_bin:pool.append(row)
        else:
            j=rng.randrange(seen[name])
            if j<per_bin:pool[j]=row
    rows=[]
    for name in sorted(pools):
        for row in sorted(pools[name],key=lambda r:r['PuzzleId']):
            board=chess.Board(row['FEN']);moves=row['Moves'].split()
            board.push_uci(moves[0]);answer=chess.Move.from_uci(moves[1])
            if answer not in board.legal_moves:raise ValueError('Puzzle answer illegal: '+row['PuzzleId'])
            themes=row['Themes'].split()
            accepted=mating_moves(board) if 'mateIn1' in themes else [answer.uci()]
            group='puzzle:'+row['PuzzleId']
            rows.append(case(group,'tactics','lichess_puzzle',board,
                'Найди лучший ход для ходящей стороны. Ответь только одним ходом в формате UCI, без пояснений.',
                'puzzle_move',{'moves':accepted},group,{'themes':themes,'strata':{'рейтинг':name}},
                {'source':'lichess_puzzle_db_cc0','puzzle_id':row['PuzzleId'],'rating':int(row['Rating'])},16))
    return rows


def judgement_cases(engine_path,positions,config,profile):
    """Who stands better and by how much, in the same words as the commentary dataset.

    Gold is a Stockfish score from White's side; positions near a bucket boundary, mates, and
    positions whose bucket changes on a deeper re-check are excluded. Buckets are balanced.
    """
    settings=config['judgement'];per_bucket=settings['per_bucket'][profile]
    counts={b[0]:0 for b in BUCKETS};rows=[]
    engine=chess.engine.SimpleEngine.popen_uci(str(engine_path))
    try:
        for p in positions:
            if all(v>=per_bucket for v in counts.values()):break
            board=p['board']
            scores=[]
            for nodes in (settings['nodes'],settings['nodes']*settings['recheck_factor']):
                info=engine.analyse(board,chess.engine.Limit(nodes=nodes),game=object())
                scores.append(info['score'].white())
            if any(s.is_mate() for s in scores):continue
            letters={bucket(s.score(),settings['margin_cp']) for s in scores}
            if len(letters)!=1 or None in letters:continue
            letter=letters.pop()
            if counts[letter]>=per_bucket:continue
            counts[letter]+=1;group=p['group']
            options=' '.join(f'{l}) {t}.' for l,t,_,_ in BUCKETS)
            rows.append(case(f'{group}:judgement','commentary','position_judgement',board,
                f'Кто лучше стоит в этой позиции и насколько? {options} Ответь только буквой A, B, C, D, E, F или G.',
                'judgement',{'answer':letter,'cp_white':scores[1].score()},group,
                {'strata':{'перевес':SIDE_OF[letter]},'nodes':[settings['nodes'],settings['nodes']*settings['recheck_factor']]},
                p['provenance']))
    finally:engine.quit()
    if any(v<per_bucket for v in counts.values()):
        raise ValueError(f'Not enough stable positions per bucket: {counts}; supply more games')
    return rows


def judgement_positions(games,config,profile,rng):
    """Candidate positions for judgement: real games if supplied, else random legal play (mostly lopsided)."""
    if games:
        return [{'board':p['board'],'group':f"game:{p['game']:05}:{p['ply']:03}",
                 'provenance':{'source':'local_pgn','game_index':p['game'],'ply':p['ply']}}
                for p in game_positions(games,config['judgement']['candidates'][profile],random.Random(config['seed']+7))]
    from .generated import random_play
    out=[]
    for i in range(config['judgement']['candidates'][profile]):
        board,history=random_play(rng,min_ply=4,max_ply=40)
        out.append({'board':board,'group':f'judge_random:{i:04}','provenance':{'source':'independent_random_legal_play','moves':history}})
    return out
