# Qwen GigaBench: непрерывный батчинг vLLM

Используется отдельный engine/run. Не продолжайте vLLM-ответами каталог HF.
Исходный benchmark `verify()` и скорер не обходятся и не изменяются.

## Среда ПК

Ubuntu 24.04, Python 3.12, RTX 4060 Ti 8 GB, NVIDIA 595.99.02.
Окружение: `/home/mark-twen/.venvs/qwen-vllm`.
Для Triton нужны C-компилятор и Python headers:

```sh
sudo apt-get install build-essential python3-dev
```

Проверенная связка: Torch `2.11.0+cu129`, Transformers `4.57.6`,
vLLM **`0.22.0+cu129`**, chess `1.11.2`.
Основной PyPI wheel 0.22.0 собран под CUDA 13: одного
`--torch-backend=cu129` недостаточно. Установка в свежую среду:

```sh
uv venv --python /usr/bin/python3 ~/.venvs/qwen-vllm
uv pip install --python ~/.venvs/qwen-vllm/bin/python \
  'https://github.com/vllm-project/vllm/releases/download/v0.22.0/vllm-0.22.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  'transformers==4.57.6' 'chess==1.11.2' --torch-backend=cu129
```

Нельзя обновлять зависимости работающего прогона. Проверка импорта должна
включать `from vllm import EngineArgs, LLMEngine`, не только `import vllm`.

## Новый запуск

Из корня репозитория, с активированным vLLM-окружением:

```sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCHINDUCTOR_COMPILE_THREADS=2 MAX_JOBS=2
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

python ml/benchmark/run.py prepare --backend qwen --profile standard \
  --without-legacy --max-new-tokens 2048 \
  --games ml/benchmark/data/sources/games_v1_1.json \
  --puzzles ml/benchmark/data/sources/puzzles_v1_1.json \
  --out ml/benchmark/runs/NEW_RUN
python ml/benchmark/tools/qwen_vllm.py --out ml/benchmark/runs/NEW_RUN \
  --max-num-seqs 32 --gpu-memory-utilization 0.90 --cuda-graphs
python ml/benchmark/run.py score --out ml/benchmark/runs/NEW_RUN \
  --stockfish "$STOCKFISH_EXECUTABLE"
```

`STOCKFISH_EXECUTABLE` должен указывать на существующий Stockfish:
он нужен для оценки рекомендаций M3, иначе они остаются `pending_engine`.
На ПК проверенный путь:
`/home/mark-twen/.cache/gigabench-tools/stockfish-local/usr/games/stockfish`.
Команды выше подключают публичные замороженные выборки (3000 партий,
200 пазлов, CC0), но не приватные 89 записей. Формат файлов — JSON;
CLI-параметры `--games`/`--puzzles` также принимают PGN/CSV.
Старый набор из 4071 запроса получается без этих двух параметров.
Состав расширенного набора отличается: общий Score напрямую со старым
знаменателем 4071 не сравнивать.

### Подготовленный публичный набор

На Mac и ПК подготовлен
`ml/benchmark/runs/qwen3-1.7b-answer-first-public-prepared-20261124/`.
В нём **7871 автоматический запрос**:
4071 предыдущая задача сохранена без изменений, добавлено 3800 задач.
Добавления: 3200 вопросов по позициям реальных партий,
200 предсказаний хода человека (M2), 200 рекомендаций с оценкой движка (M3),
200 пазлов Lichess в блоке тактики.

Файлы источников уже были в репозитории; их SHA-256 совпали с файлами
`Benchmark/data/sources/` из предоставленного пользователем
`GigaChess_Benchmark_v1.1_full.zip`. Заново скачивать или распаковывать
весь архив не требуется. Проверены все ходы 3000 партий и 200 решений.

Приватный legacy не подключён, экспертных explanation-заданий нет.
Опциональный блок `position_judgement` в эту подготовку не включён.
Это **только подготовка**, без `responses.jsonl`, `run.json` и нового Score.
Последний измеренный результат **1260/4071** не менялся.
Хеши и состав записаны в `preflight.json`, `manifest.json`, `SHA256SUMS`.
Для запуска уже подготовленного набора повторять `prepare` не нужно:
нужно передать его путь в команды runner и score выше.

### Полная локальная автоматическая сборка без М4–М7

После проверки 89 приватных записей доступен полный автоматический режим:

```sh
export STOCKFISH_EXECUTABLE=/home/mark-twen/.cache/gigabench-tools/stockfish-local/usr/games/stockfish
python ml/benchmark/run.py prepare --backend qwen --profile standard \
  --max-new-tokens 2048 --without-explanations \
  --legacy-source ml/benchmark/data/private/legacy_cases.jsonl \
  --games ml/benchmark/data/sources/games_v1_1.json \
  --puzzles ml/benchmark/data/sources/puzzles_v1_1.json \
  --with-judgement --stockfish "$STOCKFISH_EXECUTABLE" \
  --out ml/benchmark/runs/NEW_FULL_AUTOMATIC_RUN
```

Целевой состав: **8357 автоматических запросов** — публичные 7871,
346 автоматических legacy M1–M3, 140 оценок позиции (по 20 на A–G).
Если стабильных позиций для какого-либо класса не хватает, подготовка
завершается ошибкой, а не молча уменьшает набор.
`--without-explanations` исключает 89 экспертных cases до заморозки;
число исключённых записывается в `excluded_explanation_cases`.
Флаг отсутствует — сохраняется прежнее поведение с pending экспертными cases.

Эталоны позиции проверяются Stockfish на 300000 и 1200000 узлах;
матовые, пограничные и нестабильные оценки исключаются.
Threads/Hash и фактическая NNUE фиксируются через `EngineJudge`;
SHA бинарника/сети и бюджеты записываются в provenance.
При оценке ответов M3 используется бюджет из `config.engine` (1000000 узлов,
допуск 50 cp), а не бюджет подготовки judgement.
Оба M3 остаются в своих блоках, но исключаются из exact-task macro.

Приватный файл импортирован локально из предоставленного пользователем архива:
89 корректных позиций/легальных переходов, 79 mainline и 10 side.
Корректность авторских объяснений и права на повторную публикацию **не проверены**.
Private source и результаты с private cases не публиковать и не использовать
для обучения без отдельного решения. Они исключены из Git/public-package.
В автоматические cases/queries авторские комментарии не передаются.
`contains_private_cases` и `contains_local_source_comments` — разные флаги.

После такого `prepare` нужны те же команды vLLM и `score --stockfish`, что выше.
Полный автоматический запуск не означает экспертную оценку М4–М7:
они остаются вне текущего измерения.

Модель заранее должна быть скачана. В этом эксперименте использован snapshot
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`; конкретные files SHA записываются
в `run.json`. `--model` допускает закреплённый локальный snapshot.

Более консервативная конфигурация для первой проверки — seqs16, memory0.85
без `--cuda-graphs`. Для пробного набора можно использовать `--limit N`
и `--ids FILE` (JSON-массив ID либо audit JSON с `cases`).
Проба остаётся частичным прогоном относительно frozen-набора; это не полный результат.

## Что гарантирует runner

- Тот же FEN и вопрос в штатном chat template, `enable_thinking=False`.
- Явные input token IDs; engine-returned IDs сверяются с отправленными.
- BF16 weights/KV, без LoRA и квантования; greedy temperature0.
- Полный frozen output budget; досрочный stop только по EOS/stop IDs модели.
- `finish_reason`, `stop_reason`, output IDs и полный текст сохраняются.
- Завершённый ответ немедленно дописывается, flush+fsync; metadata заменяется атомарно.
- Проверяются manifest, response hashes и identity engine/model/runner/config.
  Поменять режим CUDA graphs, версии, модель или память и незаметно продолжить
  тот же run нельзя.

Greedy HF/vLLM и разные формы батча не гарантируют одинаковых токенов.
Не объединяйте замеры разных engines в одну нулевую точку.
Для скорости используйте `execution_segments.elapsed_seconds` и output token count;
время отдельного ответа включает ожидание очереди, а не только его GPU-вычисления.

Актуальный результат:
[Полный автоматический прогон — 2237 / 8357](../../ml/benchmark/runs/qwen3-1.7b-vllm-full-auto-answer-first-2048-20261124/report.html).
Формат новых запросов:
`docs/ml/2026-11-24-answer-first-benchmark.md`.

## Сравнение отчётов

```sh
python3 ml/benchmark/tools/compare_reports.py \
  /path/to/base/report.html \
  /path/to/handoff-desc-v1/report.html \
  /path/to/pilot-cells-v1/report.html \
  --out-dir .
```

Можно передать каталоги прогонов вместо `report.html`. Скрипту нужны
соседние артефакты прогона; HTML без них недостаточно. Поддерживаются обычные
замороженные прогоны и каталоги `base`/`lora` одного парного прогона.
Первый аргумент — базовый прогон; каждый последующий сравнивается именно с
ним. Несопоставимые или неполные результаты отклоняются. Отчёт показывает
подтверждённо правильные ответы по блокам и дельты в процентных пунктах.
Если в каждом прогоне есть полный набор `M4-auto–M7-auto`, он также выводится
отдельно; для M6-auto меньшее значение лучше. Экспертные оценки не подменяются
автоматическими. Существующий файл результата не перезаписывается.

Имя отчёта: `<база>__vs__<модель-2>__vs__<модель-N>.html`.
Каждый идентификатор: `<базовая-модель>--<тип-обучения>--<датасет>`.
У исходных весов используется `base--none`. Идентичность можно хранить в
`run.json` в необязательном `model_identity` с полями `base_model`,
`training_type`, `dataset`; для адаптеров также распознаётся каталог
`lora-<имя-датасета>`. Флаги `--before-*` и `--after-*` сохранены для
двухаргументного вызова. `--out-dir` выбирает существующий каталог, не имя
файла; `--out` принимает только путь с каноническим именем.
