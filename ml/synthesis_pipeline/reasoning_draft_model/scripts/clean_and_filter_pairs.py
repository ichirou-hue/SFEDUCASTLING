import html
import json
import os
import re
import wordninja

INPUT_FILE = "data/gameknot_raw_pairs.jsonl"
OUTPUT_FILE = "data/gameknot_clean_final.jsonl"

CHESS_MOVE_REGEX = re.compile(
    r"^(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|[a-h]x[a-h][1-8]|O-O(?:-O)?|[a-h][1-8])[+#]?[?!]*$",
    re.IGNORECASE,
)
CHESS_ELEMENT_REGEX = re.compile(
    r"(?:\b\d+\.{1,3}|\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?[?!]*|\bO-O(?:-O)?\b|\b[a-h][1-8]\b)",
    re.IGNORECASE,
)
MOVE_NUM_RE = re.compile(r"^\(?\d+\.{1,3}\)?$")

STOP_PATTERNS = re.compile(
    r"\b(?:"
    r"resigns?|drawn by agreement|game drawn|forfeits? on time|"
    r"guess the move|correctly predicting|annotated by|"
    r"thanks for (?:the )?game|good game|well played|\bgg\b|"
    r"timeout|time forfeit|rated blitz|team match|"
    r"check out my|subscribe|youtube|http[s]?://|www\.|"
    r"not an important game|leave you with a position"
    r")\b",
    re.IGNORECASE,
)

CHESS_SEMANTIC_KEYWORDS = re.compile(
    r"\b(?:"
    r"pawn|pawns|knight|knights|bishop|bishops|rook|rooks|queen|queens|king|kings|"
    r"piece|pieces|center|centre|flank|file|rank|diagonal|square|squares|"
    r"fork|pin|skewer|tempo|tempi|development|develop|castling|castle|checkmate|mate|"
    r"attack|defense|defence|defend|protect|threat|threaten|advantage|disadvantage|"
    r"sacrifice|sac|compensation|blunder|mistake|inaccuracy|tactic|tactics|tactical|"
    r"position|positional|structure|endgame|middlegame|opening|variation|line|trade|exchange|"
    r"mobility|space|initiative|pressure|control|counterplay|passed|isolated|doubled|"
    r"fianchetto|zugzwang|zwischenzug|trapped|weakness|weak"
    r")\b",
    re.IGNORECASE,
)

SPECIAL_TOKENS = {
    "lsb",
    "dsb",
    "fianchetto",
    "zugzwang",
    "zwischenzug",
    "rybka",
    "stockfish",
    "komodo",
    "fritz",
    "crafty",
    "pandix",
}

# Строгий список допустимых коротких частей (только реальные 2-буквенные предлоги/союзы)
VALID_SHORT_WORDS = {
    "to",
    "in",
    "on",
    "at",
    "by",
    "of",
    "or",
    "as",
    "if",
    "so",
    "is",
    "it",
    "he",
    "my",
    "we",
    "me",
    "no",
    "up",
    "do",
    "am",
    "an",
}


def clean_encoding_and_html(text: str) -> str:
  text = html.unescape(text)
  text = text.replace("“", '"').replace("”", '"')
  text = text.replace("‘", "'").replace("’", "'").replace("`", "'")
  text = text.replace("—", " - ").replace("–", " - ").replace("…", "...")
  text = text.replace("\xa0", " ").replace("\t", " ")
  text = re.sub(r"[:;=8]['-]?[)D(\[\]{}pP/\\@*|><]", " ", text)
  return text


def repair_comment(text: str) -> str:
  text = clean_encoding_and_html(text)

  # 1. Предварительная склейка разорванных взятий и фигур
  text = re.sub(
      r"\b([KQRBNa-h])\s*x\s*([KQRBNa-h1-8])\b",
      r"\1x\2",
      text,
      flags=re.IGNORECASE,
  )

  # 2. Разделение спецтокенов и склеенных слов
  text = re.sub(r"\b(LSB|DSB)([a-zA-Z]{2,})\b", r"\1 \2", text)
  text = re.sub(
      r"\b(O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?[?!]*)([a-zA-Z]{2,})\b",
      r"\1 \2",
      text,
  )
  text = re.sub(r"(\.{2,})([a-zA-Z])", r"\1 \2", text)
  text = re.sub(
      r"\b([a-zA-Z]{2,})([a-h][1-8]\b)", r"\1 \2", text, flags=re.IGNORECASE
  )
  text = re.sub(
      r"\b([a-zA-Z]{2,})([KQRBN][a-h]?[1-8]?x?[a-h][1-8][+#]?[?!]*\b)",
      r"\1 \2",
      text,
  )

  # 3. Базовая пунктуация и отрицания
  text = re.sub(r"('s|n't)([a-zA-Z])", r"\1 \2", text, flags=re.IGNORECASE)
  text = re.sub(
      r"([a-zA-Z])(wouldn\'t|couldn\'t|shouldn\'t|didn\'t|doesn\'t|isn\'t|aren\'t|wasn\'t)\b",
      r"\1 \2",
      text,
      flags=re.IGNORECASE,
  )
  text = re.sub(r"([a-zA-Z])([,\.!?;:])([a-zA-Z])", r"\1\2 \3", text)
  text = re.sub(r"([\)\]?!.,;:])(\d+)", r"\1 \2", text)
  text = re.sub(r"(\d+\.{1,3})([a-zA-Z])", r"\1 \2", text)
  text = re.sub(r"([a-zA-Z])(\d+\.?)", r"\1 \2", text)
  text = re.sub(
      r"([a-h]-file|[a-h]-pawn)([a-zA-Z]{3,})\b",
      r"\1 \2",
      text,
      flags=re.IGNORECASE,
  )

  # 4. Маскирование шахматных элементов
  placeholders = {}

  def mask_match(m):
    idx = len(placeholders)
    key = f"CHESSMSK{idx}X"
    placeholders[key] = m.group(0)
    return f" {key} "

  masked_text = CHESS_ELEMENT_REGEX.sub(mask_match, text)

  # 5. Сегментация wordninja
  tokens = masked_text.split()
  reconstructed = []

  for token in tokens:
    core = token.strip(".,;:!?\"'()[]")

    if (
        not core
        or "CHESSMSK" in core
        or any(c.isdigit() for c in core)
        or core.lower() in SPECIAL_TOKENS
        or CHESS_MOVE_REGEX.match(core)
    ):
      reconstructed.append(token)
      continue

    # Проверяем только слова от 6 букв, чтобы не дробить нормальные слова из 4-5 букв
    if core.isalpha() and len(core) >= 6:
      splits = wordninja.split(core)
      if len(splits) > 1:
        # Разбиваем, ТОЛЬКО если каждая часть >= 3 букв или это предлог из списка (без одиночных 'a', 'i')
        if all(len(s) >= 3 or s.lower() in VALID_SHORT_WORDS for s in splits):
          token = token.replace(core, " ".join(splits))

    reconstructed.append(token)

  final_text = " ".join(reconstructed)

  # 6. Демаскирование
  for key, original_val in placeholders.items():
    final_text = final_text.replace(key, original_val)

  # 7. Комплексное пост-восстановление разорванной нотации (включая шахи и маты)
  # Фигуры/взятия с полями: "Qh 4#", "Bg 7", "Nf 7", "Bxc 5+", "Ke 3", "Nh 3"
  final_text = re.sub(
      r"\b([KQRBN][a-h]?[1-8]?x?|[a-h]x)\s+([a-h]?[1-8](?:=[QRBN])?[+#]?[?!]*)",
      r"\1\2",
      final_text,
  )
  # Изолированные клетки с шахами: "e 5+", "g 3"
  final_text = re.sub(
      r"\b([a-h])\s+([1-8][+#]?[?!]*)",
      r"\1\2",
      final_text,
  )

  # 8. Схлопывание пробелов перед пунктуацией
  final_text = re.sub(r"\s+([,\.!?;:])", r"\1", final_text)
  final_text = re.sub(r"\s{2,}", " ", final_text)
  return final_text.strip()


def is_move_token(token: str) -> bool:
  core = token.strip(".,;:!?\"'()[]")
  if not core:
    return False
  return bool(CHESS_MOVE_REGEX.match(core)) or bool(MOVE_NUM_RE.match(core))


def evaluate_pair(data: dict, text: str) -> bool:
  fen = data.get("fen", "")
  if (
      fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
      and len(text) < 120
  ):
    return False

  if len(text) < 60 or len(text) > 380:
    return False

  if STOP_PATTERNS.search(text):
    return False

  tokens = text.split()
  if len(tokens) < 8:
    return False

  chess_elements = sum(1 for t in tokens if is_move_token(t))
  if chess_elements / len(tokens) >= 0.30:
    return False

  if not CHESS_SEMANTIC_KEYWORDS.search(text):
    return False

  return True


def main():
  if not os.path.exists(INPUT_FILE):
    print(f"Ошибка: {INPUT_FILE} не найден!")
    return

  total = 0
  passed = 0

  print("Запуск генерации идеального датасета...")
  with (
      open(INPUT_FILE, "r", encoding="utf-8") as f_in,
      open(OUTPUT_FILE, "w", encoding="utf-8") as f_out,
  ):

    for line in f_in:
      total += 1
      try:
        data = json.loads(line)
      except Exception:
        continue

      repaired = repair_comment(data.get("comment", ""))

      if evaluate_pair(data, repaired):
        data["comment"] = repaired
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
        passed += 1

      if total % 50000 == 0:
        print(
            f"Обработано: {total} | Отобрано: {passed} "
            f"({passed/total*100:.1f}%)"
        )

  print("=" * 60)
  print("ФИНАЛЬНЫЙ ДАТАСЕТ ГОТОВ")
  print("=" * 60)
  print(f"Всего сырых пар:    {total}")
  print(f"Итого отобрано:     {passed} ({passed/total*100:.1f}%)")
  print(f"Файл сохранен:      {OUTPUT_FILE}")
  print("=" * 60)


if __name__ == "__main__":
  main()