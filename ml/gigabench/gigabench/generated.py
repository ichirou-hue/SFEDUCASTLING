"""Balanced generated tasks for v1.1: every yes/no task has as many 'yes' as 'no' answers.

Gold always comes from python-chess rules; nothing here uses the model or the engine.
Positions come from uniform random legal play (or random endgames for mates and rules),
seeded and fully reproducible.
"""
import random
import chess
from .oracles import mating_moves, mate_within_two, position_key

VALUES={chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9,chess.KING:0}
NAMES={chess.PAWN:'пешка',chess.KNIGHT:'конь',chess.BISHOP:'слон',chess.ROOK:'ладья',chess.QUEEN:'ферзь',chess.KING:'король'}
SIDE={True:'белые',False:'чёрные'}

PARAPHRASES={
    'piece':['Какая фигура на {sq}? Назови клетку, цвет и фигуру либо укажи, что клетка пуста.',
             'Что стоит на поле {sq}? Укажи клетку, цвет и тип фигуры или скажи, что поле пустое.',
             'Посмотри на клетку {sq}: какая там фигура и какого цвета? Если клетка пуста, назови клетку и так и напиши.'],
    'check':['Находится ли король ходящей стороны под шахом? Ответь только да или нет.',
             'Объявлен ли сейчас шах стороне, которая делает ход? Ответь да или нет.',
             'Стоит ли под шахом король той стороны, чья очередь ходить? Ответь только да или нет.'],
    'control':['Атакуют ли {side} поле {sq}? Считай геометрические атаки, включая атаки связанных фигур. Ответь да или нет.',
               'Есть ли у фигур, за которые играют {side}, удар по полю {sq}? Связанные фигуры тоже считаются. Ответь да или нет.',
               'Бьют ли {side} клетку {sq} хотя бы одной фигурой, если считать и связанные фигуры? Ответь только да или нет.'],
    'protection':['Защищает ли хотя бы одна другая фигура того же цвета фигуру на {sq}? Используй геометрические атаки, включая связанные фигуры. Ответь да или нет.',
                  'Есть ли у фигуры на {sq} защитник своего цвета? Связанные фигуры тоже считаются. Ответь да или нет.',
                  'Прикрыта ли фигура на {sq} другой фигурой того же цвета, если считать и связанные фигуры? Ответь только да или нет.'],
    'legality':['Легален ли ход {uci} в этой позиции? Ответь да или нет.',
                'Можно ли по правилам сделать ход {uci} в этой позиции? Ответь да или нет.',
                'Разрешён ли правилами ход {uci} здесь? Ответь только да или нет.'],
}


def case(identifier,suite,task,board,prompt,kind,gold,group,params=None,provenance=None,tokens=32):
    return {'id':identifier,'suite':suite,'task':task,'fen':board.fen(),'prompt':prompt,'score_type':kind,
            'gold':gold,'group_id':group,'params':params or {},'max_new_tokens':tokens,'private':False,
            'provenance':provenance or {'source':'independent_synthetic_v1_1'}}


def variants(kind,count,rng):
    """Indices of `count` distinct formulations; the first is always the original v1.0 wording."""
    rest=list(range(1,len(PARAPHRASES[kind])));rng.shuffle(rest)
    return [0]+rest[:count-1]


def add_paraphrased(rows,base,kind,count,rng,board,suite,task,score_kind,gold,group,params,provenance,**fmt):
    for v in variants(kind,count,rng):
        p=dict(params,paraphrase_of=base,variant=v)
        rows.append(case(f'{base}:v{v}',suite,task,board,PARAPHRASES[kind][v].format(**fmt),score_kind,gold,group,p,provenance))


def random_play(rng,min_ply=20,max_ply=120,predicate=None,tries=400):
    """A position after uniform random legal play; with `predicate`, the first position it accepts."""
    for _ in range(tries):
        b=chess.Board();history=[]
        target=rng.randint(min_ply,max_ply)
        for ply in range(max_ply):
            if b.is_game_over():break
            if ply>=min_ply and (predicate(b) if predicate else ply==target):
                return b,history
            m=rng.choice(sorted(b.legal_moves,key=lambda x:x.uci()));b.push(m);history.append(m.uci())
    raise ValueError('Random play did not produce a suitable position')


def length_bin(n):
    return '1–5 ходов' if n<=5 else ('6–10 ходов' if n<=10 else '11–15 ходов')


def illegal_move(board,rng):
    """A plausible illegal move: pseudo-legal but exposing the king if possible, else an own piece to a square it cannot reach."""
    pseudo=sorted((m for m in board.pseudo_legal_moves if not board.is_legal(m)),key=lambda m:m.uci())
    if pseudo:return rng.choice(pseudo)
    own=sorted(sq for sq,p in board.piece_map().items() if p.color==board.turn)
    rng.shuffle(own)
    for origin in own:
        reachable={m.to_square for m in board.legal_moves if m.from_square==origin}
        targets=[sq for sq in chess.SQUARES if sq!=origin and sq not in reachable and
                 not (board.piece_at(sq) and board.piece_at(sq).color==board.turn)]
        if targets:return chess.Move(origin,rng.choice(targets))
    raise ValueError('No illegal move candidate')


def random_cases(config,profile,rng):
    rows=[];seen=set()
    for index in range(config['random_positions'][profile]):
        while True:
            board,history=random_play(rng)
            if position_key(board) not in seen:seen.add(position_key(board));break
        board_tasks(rows,board,f'random:{index:03}',{'source':'independent_random_legal_play','seed':config['seed'],'moves':history},
                    rng,index,config['paraphrases'][profile],'случайная игра')
    return rows


def board_tasks(rows,board,group,prov,rng,index,k,source,cluster=None):
    """All board-reading questions about one position. `source` is a reporting stratum; `cluster` groups positions of one game for the bootstrap."""
    start=len(rows);gid=cluster or group
    def add(task,prompt,kind,gold,params=None,tokens=32):
        rows.append(case(group+':'+task,'structural',task,board,prompt,kind,gold,gid,params,prov,tokens))
    occupied=rng.choice(sorted(board.piece_map()));empty=rng.choice([s for s in chess.SQUARES if not board.piece_at(s)])
    for name,square in [('occupied',occupied),('empty',empty)]:
        sq=chess.square_name(square);p=board.piece_at(square)
        add_paraphrased(rows,group+':piece_'+name,'piece',k,rng,board,'structural','piece_'+name,'piece',
                        {'square':sq,'label':p.symbol() if p else '.'},gid,{},prov,sq=sq)
    want=index%2==0
    legal=sorted(board.legal_moves,key=lambda m:m.uci())
    proposed=rng.choice(legal) if want else illegal_move(board,rng)
    add_paraphrased(rows,group+':move_legality','legality',k,rng,board,'structural','move_legality','bool',
                    {'answer':board.is_legal(proposed)},gid,{},prov,uci=proposed.uci())
    want=(index//2)%2==0
    color=bool(rng.getrandbits(1));pool=[s for s in chess.SQUARES if board.is_attacked_by(color,s)==want] or list(chess.SQUARES)
    square=rng.choice(pool)
    add_paraphrased(rows,group+':square_control','control',k,rng,board,'structural','square_control','bool',
                    {'answer':board.is_attacked_by(color,square)},gid,{},prov,side=SIDE[color],sq=chess.square_name(square))
    want=index%2==1
    pieces=sorted(board.piece_map());pool=[s for s in pieces if bool(board.attackers(board.piece_at(s).color,s))==want] or pieces
    target=rng.choice(pool)
    add_paraphrased(rows,group+':piece_protection','protection',k,rng,board,'structural','piece_protection','bool',
                    {'answer':bool(board.attackers(board.piece_at(target).color,target))},gid,{},prov,sq=chess.square_name(target))
    add('legal_count','Сколько легальных ходов у ходящей стороны? Ответь только целым числом.','int',{'answer':len(legal)})
    origins=sorted({m.from_square for m in legal if sum(x.from_square==m.from_square for x in legal)<=8})
    origin=rng.choice(origins);osq=chess.square_name(origin)
    add('legal_moves_piece',f'Перечисли все легальные ходы фигуры с {osq}. Только UCI через запятую; если ходов нет, ответь нет.',
        'move_set',{'moves':sorted(m.uci() for m in legal if m.from_square==origin)},{'origin':osq,'full_list':False},tokens=64)
    add_state_tracking(rows,board,group,prov,rng,index,gid)
    add_comment_choice(rows,board,group,prov,rng,gid)
    material=sum(VALUES[p.piece_type]*(1 if p.color else -1) for p in board.piece_map().values())
    add('material_balance','Посчитай материальный баланс белые минус чёрные: пешка=1, конь=3, слон=3, ладья=5, ферзь=9, король=0. Ответь целым числом.',
        'int',{'answer':material})
    for row in rows[start:]:
        row['params'].setdefault('strata',{})['источник']=source


def add_state_tracking(rows,board,group,prov,rng,index,gid):
    """One sequence of 1–15 moves; ask about the last destination (occupied) and the first origin when it ended empty."""
    tracked=board.copy();seq=[];length=rng.randint(1,15)
    for _ in range(length):
        lm=sorted(tracked.legal_moves,key=lambda m:m.uci())
        if not lm:break
        m=rng.choice(lm);seq.append(m);tracked.push(m)
    if not seq:return
    asked=[('occupied',seq[-1].to_square)]
    vacated=[m.from_square for m in seq if not tracked.piece_at(m.from_square)]
    if vacated:asked.append(('vacated',rng.choice(vacated)))
    for name,square in asked:
        sq=chess.square_name(square);p=tracked.piece_at(square)
        rows.append(case(f'{group}:state_tracking:{name}','structural','state_tracking',board,
            f'Последовательно выполни ходы UCI: {", ".join(m.uci() for m in seq)}. Какая фигура после этого на {sq}? Назови клетку, цвет и фигуру либо укажи, что клетка пуста.',
            'piece',{'square':sq,'label':p.symbol() if p else '.'},gid,
            {'sequence':[m.uci() for m in seq],'strata':{'длина':length_bin(len(seq))}},prov))


def move_statements(board,move):
    """Factual statements about a move, each with its truth value from the rules."""
    after=board.copy();after.push(move)
    moved=after.piece_at(move.to_square);own=board.turn
    queen=[sq for sq,p in after.piece_map().items() if p.color!=own and p.piece_type==chess.QUEEN]
    facts=[('Этот ход даёт шах.','Этот ход не даёт шаха.',after.is_check()),
           ('Этим ходом берётся фигура или пешка соперника.','Этот ход ничего не берёт.',board.is_capture(move)),
           ('После хода сходившая фигура стоит под ударом соперника.','После хода сходившую фигуру соперник не атакует.',after.is_attacked_by(not own,move.to_square)),
           ('После хода сходившую фигуру защищает своя фигура.','После хода сходившая фигура ничем своим не защищена.',bool(after.attackers(own,move.to_square))),
           (f'После хода {NAMES[moved.piece_type]} на {chess.square_name(move.to_square)} нападает на ферзя соперника.',
            f'После хода {NAMES[moved.piece_type]} на {chess.square_name(move.to_square)} не нападает на ферзя соперника.',
            any(q in after.attacks(move.to_square) for q in queen)) if queen else None]
    return [f for f in facts if f]


def add_comment_choice(rows,board,group,prov,rng,gid):
    """One true statement and three false ones, all about the same move (v1.0 had two always-false options)."""
    for move in rng.sample(sorted(board.legal_moves,key=lambda m:m.uci()),min(12,board.legal_moves.count())):
        facts=move_statements(board,move)
        if len(facts)<4:continue
        i=rng.randrange(len(facts));t,f,v=facts[i]
        others=[(f2 if v2 else t2) for j,(t2,f2,v2) in enumerate(facts) if j!=i]
        options=[(t if v else f,True)]+[(s,False) for s in rng.sample(others,3)]
        rng.shuffle(options)
        answer=chr(65+next(j for j,(_,ok) in enumerate(options) if ok))
        rows.append(case(group+':comment_fact_choice','structural','comment_fact_choice',board,
            f'Выбери верный комментарий к ходу {move.uci()}: '+' '.join(f'{chr(65+j)}) {t}' for j,(t,_) in enumerate(options))+' Ответь только A, B, C или D.',
            'choice',{'answer':answer},gid,{'move':move.uci(),'scope':'factual comment matching, not teaching quality'},prov))
        return


def check_cases(config,profile,rng):
    """Positions from random play, half with the side to move in check."""
    rows=[];k=config['paraphrases'][profile]
    for index in range(config['generated_per_task'][profile]):
        want=index%2==0
        board,history=random_play(rng,predicate=(lambda b:b.is_check()) if want else (lambda b:not b.is_check() and rng.random()<.05))
        group=f'check:{index:03}'
        add_paraphrased(rows,group,'check',k,rng,board,'structural','check','bool',{'answer':board.is_check()},group,{},
                        {'source':'independent_random_legal_play','moves':history})
    return rows


def random_endgame(rng,extra):
    """Kings plus `extra` pieces placed at random; side to move random. Returns a valid board."""
    while True:
        b=chess.Board(None);b.turn=bool(rng.getrandbits(1))
        squares=rng.sample(list(chess.SQUARES),2+len(extra))
        b.set_piece_at(squares[0],chess.Piece(chess.KING,True));b.set_piece_at(squares[1],chess.Piece(chess.KING,False))
        for sq,(kind,color) in zip(squares[2:],extra):
            if kind==chess.PAWN and chess.square_rank(sq) in (0,7):break
            b.set_piece_at(sq,chess.Piece(kind,color))
        else:
            if b.is_valid():return b


def endgame_material(rng):
    strong=bool(rng.getrandbits(1))
    heavy=[(rng.choice([chess.QUEEN,chess.ROOK]),strong)]
    return heavy+[(rng.choice([chess.PAWN,chess.KNIGHT,chess.BISHOP]),rng.choice([True,False])) for _ in range(rng.randint(0,2))]


def rule_cases(config,profile,rng):
    """Checkmate and stalemate, each balanced; negatives are the hard ones (check without mate, few moves without stalemate)."""
    rows=[];n=config['generated_per_task'][profile]
    specs=[('checkmate','Находится ли ходящая сторона под матом? Ответь да или нет.',
            lambda b:b.is_checkmate(),lambda b:b.is_check() and not b.is_checkmate()),
           ('stalemate','Является ли позиция патом? Ответь да или нет.',
            lambda b:b.is_stalemate(),lambda b:not b.is_check() and 0<b.legal_moves.count()<=4)]
    for task,prompt,positive,negative in specs:
        seen=set()
        for index in range(n):
            test=positive if index%2==0 else negative
            for _ in range(200000):
                b=random_endgame(rng,endgame_material(rng))
                if test(b) and position_key(b) not in seen:break
            else:raise ValueError('Cannot generate '+task)
            seen.add(position_key(b));group=f'rules:{task}:{index:03}'
            rows.append(case(group,'rules',task,b,prompt,'bool',{'answer':bool(positive(b))},group,
                             provenance={'source':'independent_random_endgame'}))
    rows+=castling_cases(n,rng)
    return rows


def castling_cases(count,rng):
    """Castling with king and rook at home; attackers and blockers random. Half legal, half not."""
    rows=[];seen=set();index=0
    while index<count:
        b=chess.Board(None);color=bool(rng.getrandbits(1));b.turn=color
        rank=0 if color else 7;side=rng.choice(['k','q'])
        king=chess.square(4,rank);rook=chess.square(7 if side=='k' else 0,rank)
        b.set_piece_at(king,chess.Piece(chess.KING,color));b.set_piece_at(rook,chess.Piece(chess.ROOK,color))
        b.castling_rights=chess.BB_SQUARES[rook]
        enemy_king=rng.choice([s for s in chess.SQUARES if chess.square_rank(s) not in (0,7)])
        b.set_piece_at(enemy_king,chess.Piece(chess.KING,not color))
        for _ in range(rng.randint(1,3)):
            sq=rng.choice([s for s in chess.SQUARES if not b.piece_at(s) and chess.square_rank(s) not in (0,7)])
            b.set_piece_at(sq,chess.Piece(rng.choice([chess.ROOK,chess.BISHOP,chess.KNIGHT,chess.QUEEN]),not color))
        if rng.random()<.25:
            between=[chess.square(f,rank) for f in ((5,6) if side=='k' else (1,2,3))]
            b.set_piece_at(rng.choice(between),chess.Piece(chess.KNIGHT,rng.choice([True,False])))
        if not b.is_valid() or position_key(b) in seen:continue
        move=chess.Move(king,chess.square(6 if side=='k' else 2,rank))
        legal=b.is_legal(move)
        if legal!=(index%2==0):continue
        seen.add(position_key(b));group=f'rules:castling:{index:03}'
        rows.append(case(group,'rules','castling',b,f'Легальна ли рокировка {move.uci()} в этой позиции? Ответь да или нет.','bool',
                         {'answer':legal},group,provenance={'source':'independent_castling_synthetic'}))
        index+=1
    return rows


def mate_one_cases(config,profile,rng):
    """Half with at least one mate in one; negatives have checks available but no mate."""
    rows=[];seen=set();n=config['generated_per_task'][profile]
    for index in range(n):
        want=index%2==0
        for _ in range(200000):
            b=random_endgame(rng,endgame_material(rng))
            if b.is_check() or b.is_game_over() or position_key(b) in seen:continue
            mates=mating_moves(b)
            gives_check=any(b.gives_check(m) for m in b.legal_moves)
            if (want and mates) or (not want and not mates and gives_check):break
        else:raise ValueError('Cannot generate mate in one')
        seen.add(position_key(b));group=f'mate1:{index:03}'
        rows.append(case(group,'tactics','mate_in_one',b,
            'Найди мат в один ход для ходящей стороны. Ответь одним ходом UCI; если такого хода нет, ответь нет.',
            'mate_move',{'moves':mates},group,provenance={'source':'independent_random_endgame'}))
    return rows


def ray_squares(origin,step):
    file,rank=chess.square_file(origin),chess.square_rank(origin);df,dr=step
    while True:
        file+=df;rank+=dr
        if not (0<=file<8 and 0<=rank<8):return
        yield chess.square(file,rank)


STEPS={chess.ROOK:[(1,0),(-1,0),(0,1),(0,-1)],chess.BISHOP:[(1,1),(1,-1),(-1,1),(-1,-1)]}
STEPS[chess.QUEEN]=STEPS[chess.ROOK]+STEPS[chess.BISHOP]


def skewer_line(after,square):
    """The piece on `square` attacks an enemy non-pawn piece along a line, and directly behind it stands another enemy piece."""
    p=after.piece_at(square)
    for step in STEPS.get(p.piece_type,[]):
        hits=[s for s in ray_squares(square,step) if after.piece_at(s)][:2]
        if len(hits)==2 and all(after.piece_at(s).color!=p.color for s in hits) and after.piece_at(hits[0]).piece_type!=chess.PAWN:
            return True
    return False


def fork(after,square):
    p=after.piece_at(square)
    targets=[s for s in after.attacks(square) if after.piece_at(s) and after.piece_at(s).color!=p.color and after.piece_at(s).piece_type!=chess.PAWN]
    return len(targets)>=2


MOTIFS={
    'double_check':('Даёт ли ход {uci} двойной шах, то есть шах сразу двумя фигурами? Ответь да или нет.',
                    lambda b,m,a:len(a.checkers())==2, lambda b,m,a:len(a.checkers())==1),
    'discovered_check':('Объявляет ли ход {uci} вскрытый шах, то есть шах даёт не та фигура, которая сходила? Ответь да или нет.',
                        lambda b,m,a:any(s!=m.to_square for s in a.checkers()), lambda b,m,a:list(a.checkers())==[m.to_square]),
    'fork':('После хода {uci} нападает ли сходившая фигура одновременно на две или больше фигуры соперника, не считая пешек? Ответь да или нет.',
            lambda b,m,a:fork(a,m.to_square), lambda b,m,a:not fork(a,m.to_square) and any(a.piece_at(s) and a.piece_at(s).color!=b.turn for s in a.attacks(m.to_square))),
    'skewer_line':('После хода {uci} атакует ли сходившая фигура по линии фигуру соперника (не пешку), за которой на той же линии сразу стоит ещё одна фигура соперника? Ответь да или нет.',
                   lambda b,m,a:skewer_line(a,m.to_square), lambda b,m,a:a.piece_at(m.to_square).piece_type in STEPS and not skewer_line(a,m.to_square)),
}


def pinned_squares(board,color):
    return [s for s,p in board.piece_map().items() if p.color==color and p.piece_type!=chess.KING and board.is_pinned(color,s)]


def battery(board,a,b):
    pa,pb=board.piece_at(a),board.piece_at(b)
    if not pa or not pb or pa.color!=pb.color:return False
    for step in STEPS[chess.QUEEN]:
        line=list(ray_squares(a,step))
        if b in line and all(not board.piece_at(s) for s in line[:line.index(b)]):
            kind=chess.ROOK if 0 in step else chess.BISHOP
            return all(p.piece_type in (kind,chess.QUEEN) for p in (pa,pb))
    return False


def motif_cases(config,profile,rng):
    rows=[];n=config['generated_per_task'][profile]
    for motif,(prompt,positive,negative) in MOTIFS.items():
        for index in range(n):
            want=index%2==0;test=positive if want else negative
            for _ in range(5000):
                board,history=random_play(rng)
                moves=sorted(board.legal_moves,key=lambda m:m.uci());rng.shuffle(moves)
                found=None
                for m in moves:
                    a=board.copy();a.push(m)
                    if test(board,m,a):found=m;break
                if found:break
            else:raise ValueError('Cannot generate motif '+motif)
            a=board.copy();a.push(found);group=f'motif:{motif}:{index:03}'
            rows.append(case(group,'motifs',motif,board,prompt.format(uci=found.uci()),'bool',
                             {'answer':bool(positive(board,found,a))},group,{'move':found.uci()},
                             {'source':'independent_random_legal_play','moves':history}))
    for index in range(n):
        want=index%2==0
        while True:
            board,history=random_play(rng)
            color=board.turn;pinned=pinned_squares(board,color)
            others=[s for s,p in board.piece_map().items() if p.color==color and p.piece_type!=chess.KING and s not in pinned]
            pool=pinned if want else others
            if pool:break
        sq=rng.choice(pool);group=f'motif:absolute_pin:{index:03}'
        rows.append(case(group,'motifs','absolute_pin',board,
            f'Связана ли фигура на {chess.square_name(sq)} относительно своего короля, то есть откроет ли её уход с линии короля для удара фигуры соперника? Ответь да или нет.',
            'bool',{'answer':board.is_pinned(color,sq)},group,provenance={'source':'independent_random_legal_play','moves':history}))
    for index in range(n):
        want=index%2==0
        while True:
            board,history=random_play(rng)
            sliders=[s for s,p in board.piece_map().items() if p.piece_type in STEPS]
            pairs=[(a,b) for a in sliders for b in sliders if a<b and board.piece_at(a).color==board.piece_at(b).color]
            pool=[pq for pq in pairs if battery(board,*pq)==want]
            if pool:break
        a,b=rng.choice(pool);group=f'motif:battery:{index:03}'
        rows.append(case(group,'motifs','battery',board,
            f'Стоят ли фигуры на {chess.square_name(a)} и {chess.square_name(b)} батареей: на одной линии без фигур между ними, и обе ходят вдоль этой линии? Ответь да или нет.',
            'bool',{'answer':battery(board,a,b)},group,provenance={'source':'independent_random_legal_play','moves':history}))
    return rows


def mate_two_cases(count,seed):
    rng=random.Random(seed+2)
    positives,negatives=[],[]; seen=set()
    for _ in range(200000):
        squares=rng.sample(list(chess.SQUARES),3)
        board=chess.Board(None); board.turn=chess.WHITE
        for sq,p in zip(squares,[chess.Piece(chess.KING,True),chess.Piece(chess.QUEEN,True),chess.Piece(chess.KING,False)]):
            board.set_piece_at(sq,p)
        if not board.is_valid() or board.is_game_over() or mating_moves(board): continue
        key=position_key(board)
        if key in seen: continue
        seen.add(key)
        allowed=mate_within_two(board)
        target=positives if allowed else negatives
        if len(target)<count//2: target.append((board,allowed))
        if len(positives)==len(negatives)==count//2: break
    if len(positives)+len(negatives)!=count: raise ValueError('Cannot balance mate-two fixtures')
    rows=[]
    for index,(board,allowed) in enumerate(positives+negatives):
        group=f'mate2:{index:03}'
        rows.append(case(group,'tactics','mate_within_two',board,
            'Есть ли форсированный мат не более чем за два хода белых при любых ответах чёрных? Ответь первым ходом UCI либо нет.',
            'mate_move',{'moves':allowed},group,{'exhaustive_plies':3,'negative_proved':not bool(allowed)},
            provenance={'source':'independent_KQK_synthetic','seed':seed+2}))
    return rows


def full_move_cases(count,seed):
    """Independent random legal games, one short complete move-list per game."""
    rng=random.Random(seed+3);rows=[];seen=set()
    for game in range(count*100):
        board=chess.Board();history=[]
        for ply in range(180):
            legal=sorted(board.legal_moves,key=lambda m:m.uci())
            if not legal or board.is_game_over():break
            if ply>=20 and len(legal)<=8 and position_key(board) not in seen:
                seen.add(position_key(board));group=f'low_mobility:{game:03}'
                rows.append(case(group,'random_chess','legal_moves_all',board,
                    'Перечисли все легальные ходы ходящей стороны. Только UCI через запятую; если ходов нет, ответь нет.',
                    'move_set',{'moves':[m.uci() for m in legal]},group,{'full_list':True},
                    {'source':'independent_random_legal_play','moves':history,'seed':seed+3},tokens=64))
                break
            move=rng.choice(legal);board.push(move);history.append(move.uci())
        if len(rows)==count:return rows
    raise ValueError('Insufficient low-mobility random positions')


def build_generated(config,profile):
    rng=random.Random(config['seed'])
    return (random_cases(config,profile,rng)+check_cases(config,profile,rng)+rule_cases(config,profile,rng)+
            mate_one_cases(config,profile,rng)+motif_cases(config,profile,rng)+
            mate_two_cases(config['mate_two_positions'][profile],config['seed'])+
            full_move_cases(config['full_move_positions'][profile],config['seed']))
