# ML: карта после переноса

Код, подготовленные наборы, конфигурации профилей и результаты лежат под `ml/`. Данные не следует перегенерировать поверх существующих каталогов: инструменты ожидают новый, отсутствующий путь вывода. Это краткая навигация; исторические инструкции вынесены в [legacy-manuals.md](legacy-manuals.md).

## Основные области

- `ml/pipelines/` — подготовка и преобразование данных; `assemble_mixed.py` собирает проверенный mixed-набор из commentary и cell-questions.
- `ml/datasets/` — наборы и manifests; источники — `commentary/` и `cell-questions/`, готовый mixed — `mixed/`.
- `ml/benchmark/` — GigaBench: генерация/оценка кейсов, адаптеры, инструменты и `runs/` с результатами.
- `ml/training/` — обучение и сборка LoRA; основной вход для Qwen3-1.7B — `train_qwen_lora.py`. `ml/models/Qwen3-chess-coach/` содержит инференс Qwen3-8B; его обучение и слияние весов — в `ml/training/qwen3_chess_coach/`.
- `ml/rag_system/`, `ml/hypotheses/` — RAG и экспериментальные гипотезы.

Maia3 остаётся отдельным субмодулем проекта; рефакторинг `ml/` не меняет его контракт.

## Сборка mixed-набора

Из корня репозитория:

```bash
python ml/pipelines/assemble_mixed.py --source ml/datasets --out ml/datasets/mixed-rebuild
```

`--source` должен содержать подкаталоги `commentary/` и `cell-questions/` с manifests и тремя split-файлами. `--out` создаётся только если путь ещё не существует; скрипт сверяет хеши, количества записей, уникальность ID и разделение позиций между split-ами. Ничего не перезаписывайте: выберите другое имя для следующей сборки.

## GigaBench: mock smoke

Из корня репозитория, с установленным `python-chess`:

```bash
python ml/benchmark/run.py all --backend mock --profile smoke \
  --without-legacy --without-explanations --out ml/benchmark/runs/local-smoke
```

Выберите новый каталог вывода. Mock проверяет цепочку подготовки, генерации и скоринга, а не качество модели.
В распакованном публичном ZIP запускайте `python run.py` и добавляйте
`--without-commentary-auto`: замороженный приватный набор в архив не входит.

Профиль входа задаёт формат запроса отдельно от профиля модели. См. [профили Qwen и GigaChess](benchmark-model-profiles.md) и [определения метрик](benchmark/METRICS.md).

Навигация: [GigaBench adapters](../../ml/benchmark/gigabench/adapters.py), [pipeline](../../ml/benchmark/gigabench/pipeline.py), [tools](../../ml/benchmark/tools/), [runs](../../ml/benchmark/runs/).

Отчёт Qwen-four: [сравнение четырёх моделей](2026-11-27-qwen-four-models-results.md). Ответы, HTML и веса сохранены локально в `ml/benchmark/runs/qwen-four-20261126/`.

Для нового эксперимента на CUDA используйте `ml/benchmark/tools/run_qwen_four_models.sh` из корня. Передайте новый каталог вывода, Stockfish и переменные `TRAIN_PYTHON`, `BENCH_PYTHON`, `TRAINING_LOCK`, `QWEN_SNAPSHOT`. Скрипт требует исходный архив GigaBench, использованный при проверке утечек. Для повторного построения отчёта из готовых ответов обучение не нужно: передайте каталоги моделей в `ml/benchmark/tools/compare_reports.py`.

## Данные, архивы и проверки

- Локальные датасеты и их `manifest.json` находятся в `ml/datasets/`. Не коммитьте локальные выгрузки/модели и не меняйте архивы прогонов без сохранения сопутствующих манифестов.
- Архивы, контрольные суммы и зафиксированные артефакты benchmark находятся в `ml/benchmark/runs/`; в ряде запусков присутствуют `SHA256SUMS`, исходники и lock-файлы. Это immutable evidence: не редактируйте их на месте.
- Код входит в хэш подготовки GigaBench: после изменения структуры для нового запуска подготовьте новый run. Не исправляйте хэши в исторических `manifest.json`; для пересчёта используйте `run.py rescore` в новый каталог.

```bash
python -m unittest discover -s tests/ml/dataset_gen
python -m unittest discover -s tests/ml/gigabench
```

Модельные и GPU-проверки требуют отдельного окружения; обычные тесты не запускают обучение.

Ссылки на исторические руководства и их статус: [docs/ml/legacy-manuals.md](legacy-manuals.md).
