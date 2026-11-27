# Профили моделей GigaBench

`--profile smoke|quick|standard` задаёт набор при подготовке,
`--model-profile` — параметры запуска, а `--input-profile` — формат входа.
Профиль модели не меняет замороженные задания, ответы или метрики.
Настройки пресетов лежат в [`gigabench/profiles/`](../../ml/benchmark/gigabench/profiles/).

| Профиль | Где доступен | Вход по умолчанию |
|---|---|---|
| `qwen-v1` | `run.py all/generate` (Transformers), `tools/qwen_vllm.py` (vLLM) | FEN + исходный вопрос; модель `Qwen/Qwen3-1.7B` |
| `gigachess-local-v1` | `run.py all/generate` | Вопрос текстом, FEN отдельно через локальный runtime GigaChess |

Для `generate` и vLLM `--out` должен указывать на каталог, заранее подготовленный
текущей версией GigaBench. Для каждой модели нужен отдельный каталог с копией
того же подготовленного набора.

```bash
python ml/benchmark/run.py generate --out /path/to/qwen-run \
  --model-profile qwen-v1

python ml/benchmark/tools/qwen_vllm.py --out /path/to/qwen-vllm-run \
  --model-profile qwen-v1

python ml/benchmark/run.py generate --out /path/to/gigachess-run \
  --model-profile gigachess-local-v1 --project-root /path/to/gigachess-runtime
```

Явный `--input-profile` переопределяет формат профиля. Для Qwen можно также
указать `--model` с ID или путём к snapshot. Локальный GigaChess — не облачный
GigaChat API. Специализированный эксперимент четырёх Qwen-моделей с числовой
доской для статических клеток по-прежнему использует свой раннер.

Имя пресета сохраняется в `run.json`. Разные входы в `run.py compare` допустимы:
это сравнение связок «модель + вход», а не изолированного эффекта модели.
`tools/compare_reports.py` проверяет совместимость известных настроек входа,
генерации и скоринга и отдельно учитывает хэш фактических входов, когда раннер
его сохраняет.
