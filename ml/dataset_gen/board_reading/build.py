"""Build an SFT pilot with rule-derived labels and benchmark position exclusion."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gigabench'))
from gigabench.generated import random_play
from gigabench.prompts import ANSWER_FIRST_INSTRUCTION
from gigabench.scoring import score_case
from numeric_board import format_cell_query, format_query

NAMES = {
    'p': ('белая', 'чёрная', 'пешка'), 'n': ('белый', 'чёрный', 'конь'),
    'b': ('белый', 'чёрный', 'слон'), 'r': ('белая', 'чёрная', 'ладья'),
    'q': ('белый', 'чёрный', 'ферзь'), 'k': ('белый', 'чёрный', 'король'),
}
QUOTAS = {'cell': 1500, 'track': 1000, 'check': 600,
          'attack': 450, 'defense': 450, 'material': 300, 'legal_count': 200}
VALUES = (0, 1, 3, 3, 5, 9, 0)


def placement(board):
    return board.board_fen()


def cell_label(board, square):
    name = chess.square_name(square)
    piece = board.piece_at(square)
    if piece is None:
        return f'Поле {name} пусто.', {'square': name, 'label': '.'}
    white, black, kind = NAMES[piece.symbol().lower()]
    return f'На {name} {white if piece.color else black} {kind}.', {
        'square': name, 'label': piece.symbol(),
    }


def sample(task, desired, rng, excluded, lengths):
    predicate = None
    if task == 'check':
        predicate = lambda board: board.is_check() == desired
    board, _ = random_play(rng, min_ply=8, predicate=predicate)
    used = {placement(board)}
    if used & excluded:
        return None
    target = board.copy()
    prefix = ''
    if task == 'track':
        length = rng.choice(lengths)
        moves = []
        for _ in range(length):
            if target.is_game_over():
                return None
            move = rng.choice(sorted(target.legal_moves, key=chess.Move.uci))
            target.push(move)
            moves.append(move.uci())
            used.add(placement(target))
        prefix = f'Сыграй по порядку {", ".join(moves)}. В получившейся позиции '
    if used & excluded:
        return None

    if task in ('cell', 'track'):
        squares = [sq for sq in chess.SQUARES if bool(target.piece_at(sq)) == desired]
        if task == 'track':
            last = moves[-1]
            squares = [sq for sq in squares if chess.square_name(sq) not in (last[:2], last[2:4])]
            changed = [sq for sq in squares if target.piece_at(sq) != board.piece_at(sq)]
            if changed and rng.random() < 0.75:
                squares = changed
        square = rng.choice(squares)
        name = chess.square_name(square)
        answer, gold = cell_label(target, square)
        question = prefix + f'укажи, что находится на {name}. Назови поле, цвет и тип фигуры или скажи, что поле пусто.'
        score_type = 'piece'
    elif task in ('check', 'attack', 'defense'):
        if task == 'check':
            question = 'У стороны, которой предстоит ходить, король под шахом?'
        elif task == 'attack':
            color = rng.choice((chess.WHITE, chess.BLACK))
            squares = [sq for sq in chess.SQUARES if board.is_attacked_by(color, sq) == desired]
            if not squares:
                return None
            name = chess.square_name(rng.choice(squares))
            question = f'Бьют ли {"белые" if color else "чёрные"} поле {name}?'
        else:
            squares = [sq for sq, piece in board.piece_map().items()
                       if bool(board.attackers(piece.color, sq)) == desired]
            if not squares:
                return None
            name = chess.square_name(rng.choice(squares))
            question = f'Есть ли у фигуры на {name} защита со стороны своих фигур?'
        answer, gold, score_type = 'Да' if desired else 'Нет', {'answer': desired}, 'bool'
    else:
        if task == 'material':
            value = sum(VALUES[p.piece_type] * (1 if p.color else -1) for p in board.piece_map().values())
            question = 'На сколько очков материал белых отличается от материала чёрных? Вычти чёрных из белых. Пешка — 1, конь и слон — 3, ладья — 5, ферзь — 9, король — 0.'
        else:
            value = board.legal_moves.count()
            question = 'Укажи общее число допустимых ходов для стороны, которая сейчас ходит.'
        answer, gold, score_type = str(value), {'answer': value}, 'int'
    return board, used, question[0].upper() + question[1:], answer, gold, score_type


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-cases', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=43721)
    parser.add_argument('--input-format', choices=('fen', 'numeric'), default='fen')
    parser.add_argument('--cells-only', action='store_true')
    args = parser.parse_args()
    raw = args.benchmark_cases.read_bytes()
    excluded = set()
    for line in raw.splitlines():
        case = json.loads(line)
        board = chess.Board(case['fen'])
        excluded.add(placement(board))
        for move in case.get('params', {}).get('sequence', []):
            board.push_uci(move)
            excluded.add(placement(board))
    heldout = len(excluded)
    rng = random.Random(args.seed)
    datasets = {}
    for split, share in [('train', 80), ('validation', 10), ('test', 10)]:
        rows = []
        quotas = {'cell': QUOTAS['cell']} if args.cells_only else QUOTAS
        for task, quota in quotas.items():
            for idx in range(quota * share // 100):
                for _ in range(100):
                    result = sample(task, idx % 2 == 0, rng, excluded, range(1, 16))
                    if result:
                        break
                else:
                    raise RuntimeError(f'Could not sample {task}')
                board, used, question, answer, gold, score_type = result
                excluded.update(used)
                fen = board.fen()
                row = {'id': f'{split}-{task}-{idx}', 'task': task, 'fen': fen,
                       'suite': 'board_sft', 'group_id': f'{split}-{task}-{idx}',
                       'gold': gold, 'score_type': score_type, 'params': {}, 'answer_protocol': 'answer_first_v1'}
                if score_case(row, {'text': answer})['correct'] is not True:
                    raise ValueError(f'Benchmark scorer rejected {row["id"]}')
                prompt = question if args.cells_only else f'{question}\n\n{ANSWER_FIRST_INSTRUCTION}'
                formatter = format_cell_query if args.cells_only else format_query
                content = (formatter({'fen': fen, 'prompt': prompt}) if args.input_format == 'numeric'
                           else f'Позиция FEN: {fen}\n{prompt}')
                row['messages'] = [
                    {'role': 'user', 'content': content},
                    {'role': 'assistant', 'content': answer},
                ]
                rows.append(row)
        rng.shuffle(rows)
        datasets[split] = rows
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'seed': args.seed, 'source': 'synthetic_legal_play', 'heldout_placements': heldout,
                'benchmark_cases_sha256': hashlib.sha256(raw).hexdigest(), 'splits': {}}
    manifest['input_format'] = args.input_format
    manifest['tasks'] = list(quotas)
    if args.input_format == 'numeric':
        manifest['formatter_sha256'] = hashlib.sha256(
            Path(__file__).with_name('numeric_board.py').read_bytes()
        ).hexdigest()
    for split, rows in datasets.items():
        data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
        (args.out / f'{split}.jsonl').write_bytes(data)
        manifest['splits'][split] = {'n': len(rows), 'sha256': hashlib.sha256(data).hexdigest()}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
