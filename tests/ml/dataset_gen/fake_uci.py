"""Игрушечный UCI-движок для проверки связки без Stockfish: оценка = материал, PV = лучшее взятие.

Не использовать для реальных данных — только для smoke-тестов пайплайна.
"""
import sys

import chess

VAL = {1: 100, 2: 300, 3: 300, 4: 500, 5: 900, 6: 0}


def evaluate(b):
    s = sum((VAL[p.piece_type] if p.color else -VAL[p.piece_type]) for p in b.piece_map().values())
    return s if b.turn else -s


def pick(b):
    best, best_s = None, -10 ** 9
    for m in b.legal_moves:
        b.push(m)
        s = -10 ** 6 if b.is_checkmate() else evaluate(b)
        s = -s  # оценка для стороны, сделавшей ход
        if b.is_checkmate():
            s = 10 ** 6
        b.pop()
        if s > best_s or (s == best_s and m.uci() < best.uci()):
            best, best_s = m, s
    return best, best_s


def main():
    board = chess.Board()
    multipv = 1
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "uci":
            print("id name FakeFish\noption name MultiPV type spin default 1 min 1 max 500\nuciok", flush=True)
        elif cmd == "isready":
            print("readyok", flush=True)
        elif cmd.startswith("setoption") and "MultiPV" in cmd:
            multipv = int(cmd.split()[-1])
        elif cmd.startswith("position"):
            parts = cmd.split()
            if parts[1] == "startpos":
                board = chess.Board()
                rest = parts[2:]
            else:
                i = parts.index("fen")
                j = parts.index("moves") if "moves" in parts else len(parts)
                board = chess.Board(" ".join(parts[i + 1:j]))
                rest = parts[j:]
            for mv in rest[1:] if rest and rest[0] == "moves" else []:
                board.push_uci(mv)
        elif cmd.startswith("go"):
            moves = sorted(board.legal_moves, key=lambda m: m.uci())
            scored = []
            for m in moves:
                board.push(m)
                if board.is_checkmate():
                    s = 100000
                else:
                    reply, rs = pick(board) if not board.is_game_over() else (None, 0)
                    s = -rs if reply else -evaluate(board)
                board.pop()
                scored.append((s, m))
            scored.sort(key=lambda t: (-t[0], t[1].uci()))
            for k, (s, m) in enumerate(scored[:multipv], 1):
                board.push(m)
                reply = pick(board)[0] if not board.is_game_over() else None
                board.pop()
                pv = m.uci() + (" " + reply.uci() if reply else "")
                score = "mate 1" if s == 100000 else f"cp {s}"
                print(f"info depth 2 multipv {k} score {score} pv {pv}", flush=True)
            print(f"bestmove {scored[0][1].uci()}", flush=True)
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
