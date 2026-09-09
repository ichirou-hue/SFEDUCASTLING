import os
import sys
import json
import re
import chess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from stockfish_eval import StockfishAnalyzer, get_default_stockfish_path
from make_nag import NagClassifier
from board_flags import BoardFlagsExtractor

PROJECT_ROOT = CURRENT_DIR.parent.parent

SYSTEM_PROMPT_EN = (
    "You are a Grandmaster-level Chess Coach. Your objective is to provide deep, "
    "pedagogical, and insightful explanations of chess moves. Synthesize the provided "
    "engine evaluations, tactical motifs, and move classifications into clear strategic "
    "coaching advice. Explain why the move was played, evaluate its positional and tactical "
    "consequences, contrast it with engine recommendations when relevant, and instruct "
    "the student on key principles."
)

NAG_LABELS_EN = {
    "blunder": "Blunder",
    "mistake": "Mistake",
    "inaccuracy": "Inaccuracy",
    "good": "Good Move",
    "excellent": "Excellent Move",
    "great": "Great Move",
    "best": "Best Move",
    "brilliant": "Brilliant Move",
    "book": "Book / Opening Theory"
}


def parse_chatml_user_content(content: str) -> Tuple[Optional[str], Optional[str]]:
    fen, move_str = None, None
    fen_match = re.search(r"FEN:\s*([^\n\r]+)", content)
    if fen_match:
        fen = fen_match.group(1).strip()

    uci_match = re.search(r"\(UCI:\s*([a-h1-8]{4,5})\)", content)
    if uci_match:
        move_str = uci_match.group(1).strip()
    else:
        san_match = re.search(r"Move:\s*([^\s\(\n\r]+)", content)
        if san_match:
            move_str = san_match.group(1).strip()

    return fen, move_str


def extract_from_record(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], str]:
    if "messages" in item and isinstance(item["messages"], list):
        fen, move_str, ref_commentary = None, None, ""
        for msg in item["messages"]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                fen, move_str = parse_chatml_user_content(content)
            elif role == "assistant":
                ref_commentary = content.strip()
        return fen, move_str, ref_commentary

    fen = item.get("fen")
    move_str = item.get("move")
    ref_commentary = item.get("commentary", item.get("description", ""))
    return fen, move_str, ref_commentary


def format_enriched_user_prompt(
    fen: str,
    move_san: str,
    move_uci: str,
    sf_res: Dict[str, Any],
    nag_res: Dict[str, Any],
    flags_res: Dict[str, Any]
) -> str:
    eval_white = sf_res["played_move_eval"]["score_cp_white"]
    mate_in = sf_res["played_move_eval"].get("mate_in")

    if mate_in is not None:
        eval_str = f"Mate in {mate_in}"
    elif eval_white is not None:
        eval_str = f"{eval_white / 100:+.2f} CP"
    else:
        eval_str = "Unavailable"

    cp_loss = sf_res.get("centipawn_loss", 0)
    is_top = "Yes" if sf_res.get("is_engine_top_move") else "No"

    best_alt = sf_res.get("best_alternative")
    best_move_san = best_alt["move_san"] if best_alt else "None"
    best_move_pv = ", ".join(best_alt["pv_san"]) if best_alt and best_alt.get("pv_san") else "None"

    nag_cat = nag_res.get("category", "")
    nag_label = NAG_LABELS_EN.get(nag_cat, nag_res.get("label", "Unknown"))
    glyph = nag_res.get("glyph", "")
    nag_code = nag_res.get("nag_code", "")

    hanging = ", ".join(flags_res.get("hanging_pieces", [])) or "None"
    pins = ", ".join(flags_res.get("active_pins", [])) or "None"
    created_pins = ", ".join(flags_res.get("pins_created_by_move", [])) or "None"
    open_files = ", ".join(flags_res.get("open_files", [])) or "None"

    prompt = (
        f"### INPUT CHESS POSITION & MOVE\n"
        f"- Current FEN: {fen}\n"
        f"- Played Move: {move_san} (UCI: {move_uci})\n\n"
        f"### ENGINE ANALYSIS (STOCKFISH)\n"
        f"- Position Evaluation (White POV): {eval_str}\n"
        f"- Centipawn Loss (CP Loss): {cp_loss} CP\n"
        f"- Is Engine Best Move: {is_top}\n"
        f"- Engine Recommended Move: {best_move_san}\n"
        f"- Engine Continuation Line: {best_move_pv}\n\n"
        f"### MOVE QUALITY CLASSIFICATION (NAG)\n"
        f"- Category: {nag_label} (Glyph: {glyph}, NAG Code: {nag_code})\n\n"
        f"### SYMBOLIC BOARD STATE & TACTICS\n"
        f"- King In Check (Before Move): {'Yes' if flags_res.get('is_check') else 'No'}\n"
        f"- Move Gives Check: {'Yes' if flags_res.get('gives_check') else 'No'}\n"
        f"- Move Attributes: Capture: {'Yes' if flags_res.get('is_capture') else 'No'} | "
        f"Castling: {'Yes' if flags_res.get('is_castling') else 'No'} | "
        f"Forced Move: {'Yes' if flags_res.get('is_forced_move') else 'No'}\n"
        f"- Movement Type: {flags_res.get('movement_type', 'unknown')} | "
        f"Legal Moves Available: {flags_res.get('legal_moves_count', 0)}\n"
        f"- Unprotected / Hanging Pieces: {hanging}\n"
        f"- Active Pins: {pins}\n"
        f"- Pins Created By This Move: {created_pins}\n"
        f"- Open Files: {open_files}\n\n"
        f"### COACHING TASK\n"
        f"Analyze the played move from a coach's perspective. Detail the underlying motive "
        f"behind the move, contrast its soundness against the engine alternative, identify tactical "
        f"or positional concessions, and articulate the actionable lesson for the student."
    )
    return prompt


def run_pipeline(
    input_rel_path: str = "models/data/chess_train.jsonl",
    output_rel_path: str = "models/data/featured_nag_stockfish_pychess/train_featured.jsonl"
):
    input_path = PROJECT_ROOT / input_rel_path
    output_path = PROJECT_ROOT / output_rel_path

    if not input_path.exists():
        print(f"Ошибка: входной файл не найден: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine_bin = get_default_stockfish_path()
    print(f"Используемый бинарник Stockfish: {engine_bin}")

    sf = StockfishAnalyzer(engine_path=engine_bin, depth=14, multipv=3, threads=4)

    total = 0
    passed = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            item = json.loads(line)
            total += 1

            fen, move_str, ref_commentary = extract_from_record(item)

            if not fen or not move_str:
                continue

            try:
                board = chess.Board(fen)
            except ValueError:
                continue

            move = None
            try:
                move = board.parse_uci(move_str)
            except ValueError:
                try:
                    move = board.parse_san(move_str)
                except ValueError:
                    continue

            if move not in board.legal_moves:
                continue

            if total % 10 == 0 or total == 1:
                print(f"[{total}] Анализ хода {move_str}...")

            sf_res = sf.analyze(board, move)
            flags_res = BoardFlagsExtractor.extract(board, move)
            nag_res = NagClassifier.classify(
                cp_loss=sf_res["centipawn_loss"],
                is_best_move=sf_res["is_engine_top_move"],
                is_sacrifice=flags_res["is_sacrifice"],
                prev_score_cp=sf_res["prev_score_cp"],
                mate_missed=sf_res["mate_missed"]
            )

            user_enriched_content = format_enriched_user_prompt(
                fen=fen,
                move_san=board.san(move),
                move_uci=move.uci(),
                sf_res=sf_res,
                nag_res=nag_res,
                flags_res=flags_res
            )

            sft_record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_EN},
                    {"role": "user", "content": user_enriched_content},
                    {"role": "assistant", "content": ref_commentary}
                ]
            }

            fout.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            passed += 1

    sf.close()
    print(f"\nГотово! Обработано позиций: {passed}/{total}. Результат сохранён в {output_path}")


if __name__ == "__main__":
    out_dir = "models/data/featured_nag_stockfish_pychess"

    print("=== Обработка train-датасета ===")
    run_pipeline("models/data/chess_train.jsonl", f"{out_dir}/train_featured.jsonl")

    print("\n=== Обработка val-датасета ===")
    run_pipeline("models/data/chess_val.jsonl", f"{out_dir}/val_featured.jsonl")