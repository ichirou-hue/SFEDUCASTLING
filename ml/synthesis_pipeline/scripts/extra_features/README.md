# Модуль Extra Features

Пакет вычисления структурированных признаков позиции для генерации черновых описаний (Qwen) и последующей стандартизации (Claude).

## Структура модуля

* `stockfish_eval.py` — расчет MultiPV, дельты сантипешек (`centipawn_loss`) и фиксация упущенного мата.
* `make_nag.py` — классификация хода по стандарту Lichess/Chess.com (`brilliant`, `great`, `best`, `blunder`, etc.).
* `board_flags.py` — расчет предикатов `python-chess` (связки, висячие фигуры, тип движения, баланс центра, детекция жертвы).
* `run_extra_features.py` — оркестратор пакета. Читает `data/01_raw/raw_dataset.jsonl` и сохраняет результат в `data/02_engine_analyzed/master_analyzed.jsonl`.

## Запуск из корня репозитория

```bash
STOCKFISH_PATH="/home/user/SFEDUCASTLING/Qwen3-chess-coach/stockfish/stockfish-ubuntu-x86-64-avx2" python scripts/extra_features/run_extra_features.py
```

## Указание пути к Stockfish

По умолчанию путь определяется автоматически:
1. Значение переменной `STOCKFISH_PATH`
2. Поиск через `which stockfish`
3. Фоллбэк `/usr/games/stockfish`

Для ручного переопределения:
```bash
export STOCKFISH_PATH=/opt/homebrew/bin/stockfish
python scripts/extra_features/run_extra_features.py
```