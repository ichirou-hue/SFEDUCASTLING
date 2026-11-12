"""Independent task construction inspired by cited benchmarks, not their corpora."""
import random
from collections import Counter
import chess
from .io import read_jsonl, unique
from .oracles import mating_moves, piece_at
from .generated import build_generated
from .external import game_cases, puzzle_cases, judgement_cases, judgement_positions


FIXTURES = [
    ('castle_clear', 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1'),
    ('castle_through_attack', 'r3k2r/8/8/8/2b5/8/8/R3K2R w KQkq - 0 1', 'e1g1'),
    ('castle_in_check', '4r1k1/8/8/8/8/8/8/R3K2R w KQ - 0 1', 'e1c1'),
    ('en_passant', '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 20', 'e5d6'),
    ('en_passant_exposes_king', 'k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 20', 'e5d6'),
    ('promotion', '7k/P7/8/8/8/8/8/7K w - - 0 1', 'a7a8n'),
    ('absolute_pin', '4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1', 'e2a2'),
    ('stalemate', '7k/5K2/6Q1/8/8/8/8/8 b - - 0 1', None),
    ('checkmate', '7k/6Q1/6K1/8/8/8/8/8 b - - 0 1', None),
    ('queen_mate_white', '7k/8/5KQ1/8/8/8/8/8 w - - 0 1', None),
    ('queen_mate_black', '8/8/8/8/8/5kq1/8/7K b - - 0 1', None),
    ('rook_mate', '7k/8/6K1/8/8/8/8/R7 w - - 0 1', None),
    ('double_attack', '7k/8/8/8/3N4/1q3r2/8/K7 w - - 0 1', None)
]


def case(identifier, suite, task, board, prompt, kind, gold, group, params=None, private=False, provenance=None):
    return {'id': identifier, 'suite': suite, 'task': task, 'fen': board.fen(),
            'prompt': prompt, 'score_type': kind, 'gold': gold, 'group_id': group,
            'params': params or {}, 'max_new_tokens': 64 if kind in ('move_set','explanation') else 32,
            'private': private, 'provenance': provenance or {'source':'independent_synthetic_v1'}}


def fixture_cases():
    """Hand-made edge cases (castling through check, en passant exposing the king, underpromotion...).

    v1.1 keeps them as diagnostics in suite `fixtures`: too few to enter block means.
    """
    rows=[]
    for name,fen,proposed in FIXTURES:
        board=chess.Board(fen)
        if not board.is_valid(): raise ValueError('Invalid fixture '+name)
        group='fixture:'+name
        for task,prompt,value in [
            ('check','Находится ли ходящая сторона под шахом?',board.is_check()),
            ('checkmate','Находится ли ходящая сторона под матом?',board.is_checkmate()),
            ('stalemate','Является ли позиция патом?',board.is_stalemate())]:
            rows.append(case(group+':'+task,'fixtures',task,board,prompt+' Ответь да или нет.','bool',{'answer':value},group))
        if proposed:
            rows.append(case(group+':special','fixtures','special_move',board,
                f'Легален ли ход {proposed}? Ответь да или нет.','bool',{'answer':chess.Move.from_uci(proposed) in board.legal_moves},group))
        allowed=mating_moves(board)
        rows.append(case(group+':mate1','fixtures','mate_in_one',board,
            'Найди мат в один ход для ходящей стороны. Ответь одним ходом UCI; если такого хода нет, ответь нет.',
            'mate_move',{'moves':allowed},group))
        if len(list(board.legal_moves))<=8:
            rows.append(case(group+':moves','fixtures','legal_moves_all',board,
                'Перечисли все легальные ходы ходящей стороны. Только UCI через запятую; если ходов нет, ответь нет.',
                'move_set',{'moves':sorted(m.uci() for m in board.legal_moves)},group,{'full_list':True}))
    board=chess.Board(dict((n,f) for n,f,_ in FIXTURES)['absolute_pin'])
    rows.append(case('motif:absolute_pin','fixtures','absolute_pin',board,
        'Связана ли белая ладья e2 относительно своего короля? Ответь да или нет.','bool',{'answer':board.is_pinned(chess.WHITE,chess.E2)},'fixture:absolute_pin'))
    rows.append(case('motif:pinned_control','fixtures','pinned_control',board,
        'Атакует ли белая ладья e2 поле a2 по геометрии ходов, даже если она связана? Ответь да или нет.',
        'bool',{'answer':chess.A2 in board.attacks(chess.E2)},'fixture:absolute_pin'))
    free=board.copy();free.remove_piece_at(chess.E8)
    rows.append(case('motif:no_pin','fixtures','absolute_pin',free,
        'Связана ли белая ладья e2 относительно своего короля? Ответь да или нет.',
        'bool',{'answer':free.is_pinned(chess.WHITE,chess.E2)},'fixture:no_pin'))
    board=chess.Board(dict((n,f) for n,f,_ in FIXTURES)['double_attack'])
    rows.append(case('motif:double_attack','fixtures','double_attack',board,
        'Атакует ли белый конь d4 одновременно чёрного ферзя b3 и ладью f3? Ответь да или нет.',
        'bool',{'answer':all(t in board.attacks(chess.D4) for t in [chess.B3,chess.F3])},'fixture:double_attack'))
    board.remove_piece_at(chess.F3);board.set_piece_at(chess.H3,chess.Piece(chess.ROOK,False))
    rows.append(case('motif:no_double_attack','fixtures','double_attack',board,
        'Атакует ли белый конь d4 одновременно чёрного ферзя b3 и ладью h3? Ответь да или нет.',
        'bool',{'answer':all(t in board.attacks(chess.D4) for t in [chess.B3,chess.H3])},'fixture:no_double_attack'))
    return rows


def legacy_cases(path):
    source=read_jsonl(path); rows=[]
    unique(source)
    rng=random.Random(20261101); balance=Counter()
    for index,r in enumerate(source,1):
        board=chess.Board(r['fen_before']); group=r.get('group_id',r['source']['chapter_url'])
        if not board.is_valid(): raise ValueError(r['id'])
        move=chess.Move.from_uci(r['move']['uci'])
        if move not in board.legal_moves: raise ValueError(r['id'])
        after=board.copy(); after.push(move)
        if after.fen()!=r['fen_after']: raise ValueError('Incorrect transition '+r['id'])
        identifier=f'legacy:{index:03}'
        provenance={'source':'local_lichess_pilot','source_record_id':r['id'],
                    'chapter_url':r['source']['chapter_url'],'reuse_terms_checked':False}
        def add(task,prompt,kind,gold,params=None):
            rows.append(case(identifier+':'+task,'legacy',task,board,prompt,kind,gold,group,params,True,provenance))
        occ=sorted(board.piece_map(),key=lambda sq:(balance[board.piece_at(sq).symbol()],sq))
        low=balance[board.piece_at(occ[0]).symbol()]
        square=rng.choice([sq for sq in occ if balance[board.piece_at(sq).symbol()]==low]); balance[board.piece_at(square).symbol()]+=1
        empty=rng.choice([sq for sq in chess.SQUARES if not board.piece_at(sq)])
        for name,sq in [('occupied',square),('empty',empty)]:
            sq=chess.square_name(sq)
            add('M1_'+name,f'Какая фигура находится на клетке {sq}? Назови клетку, цвет и фигуру; если клетка пуста, назови клетку и укажи, что она пуста.',
                'piece',{'square':sq,'label':piece_at(board,sq)})
        if r['source']['variation']=='main':
            add('M2','Какой следующий ход здесь сыграл бы человек? Ответь только одним ходом в формате UCI, без пояснений.',
                'human_move',{'move':move.uci()},{'reference_authenticity':'PGN mainline, not independently verified'})
            rows[-1]['max_new_tokens']=16
        add('M3','Порекомендуй хороший ход в этой позиции. Ответь только одним ходом в формате UCI, без пояснений.',
            'engine_move',{}, {'reference_move_for_strata':move.uci()})
        rows[-1]['max_new_tokens']=16
        add('M4_M7',r['question'],'explanation',{'source_comment':r['source']['original_comment'],
            'known_move':r['move']},{'requires_preapproved_coach_rubric':True})
    return rows


def build(config,profile,legacy=None,games=None,puzzles=None,stockfish=None):
    rng=random.Random(config['seed']+11)
    rows=build_generated(config,profile)+fixture_cases()
    if games:rows+=game_cases(games,config,profile,rng)
    if puzzles:rows+=puzzle_cases(puzzles,config,profile,rng)
    if stockfish:rows+=judgement_cases(stockfish,judgement_positions(games,config,profile,rng),config,profile)
    if legacy: rows+=legacy_cases(legacy)
    unique(rows)
    for row in rows:
        if not 1<=row['max_new_tokens']<=64: raise ValueError(row['id'])
    return rows
