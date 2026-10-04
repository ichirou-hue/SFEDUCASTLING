# SFEDUCASTLING API

Справочник по HTTP-интерфейсу бэкенда (FastAPI). Документ подготовлен по фактической реализации: каждый роутер находится в `backend/api_gateway/routes/`, описания сверены с исходным кодом и отражают реальное поведение сервиса.

По умолчанию бэкенд запускается на `http://127.0.0.1:8005` (`uvicorn.run` в `backend/app.py`). На корневом пути `/` раздаётся собранный фронтенд из `frontend/dist/` — любые не-API адреса отдают SPA. Интерактивная версия описания доступна на `/docs` (Swagger, генерируется FastAPI автоматически).

## Общие правила

**Авторизация.** У эндпоинтов три градации доступа:

- **гостевой** — работает вообще без токена;
- **опциональный** — токен не обязателен: гость получает обычный ответ, но данные не привязываются к аккаунту (попытка не пишется на пользователя, прогресс не обновляется);
- **авторизованный** — нужен заголовок `Authorization: Bearer <access_token>`, иначе `401`.

Для «опциональных» эндпоинтов поведение для гостя и авторизованного пользователя визуально одинаково; отличие состоит лишь в том, какие данные сохраняются. Например, гость может пройти тактический puzzles-режим и узнать, правильный ли ответ, однако его попытка не попадает в статистику.

**Роли.** Разделение прав доступа: `guest` (без аккаунта — анализ, паззлы, чат, level-test), `learner` (любой зарегистрированный — дополнительно учебные модули `/api/training/*`), `admin` (дополнительно раздел `/api/admin/*`). Единственный источник прав — колонка `users.role` (по умолчанию `learner`); флаг `is_admin` в ответах API выводится из неё и в БД не хранится. Ограничения роли накладываются зависимостью `require_roles(...)` (см. `dependecies.py`) и строго проверяются сервером независимо от фронтенда.

**Формат ошибок.** Формат не единообразен и должен учитываться при разработке клиента:

- *Строгие* ошибки возвращаются через `HTTPException`: код `400/401/403/404/409/422/503` и тело `{"detail": "<строка>"}`. Сюда же относится валидация FEN — все шахматные модели проверяют FEN через `python-chess`, и на некорректный FEN возвращается `422` с сообщением от Pydantic.
- *Мягкие* ошибки возникают, когда сервис функционирует, но не может сформировать содержательный ответ (Stockfish не инициализирован, база пазлов не загружена, LLaVA недоступна). В таких случаях возвращается `200` и тело с полем `error` (часто с дополнительными полями-заглушками). Клиенту следует проверять оба варианта — и `error`, и `detail`.

**CORS** настроен на полный доступ (все origins/methods/headers); для продакшена настройку следует ограничить.

## Схемы данных

### Тактический пазл

Пазлы читаются из `backend/knowledge/puzzles.json` — это выгрузка из публичного датасета Lichess. Внутри задачи хранятся целиком:

```json
{
  "id": "00Dlt",
  "fen": "r3kb1r/p4pp1/... w - - 1 14",
  "moves": "d3f1",
  "rating": 510,
  "themes": ["kingsideAttack", "mate", "mateIn1", "oneMove", "opening"],
  "win_score": 100000,
  "first_adv": 100000,
  "solver_mates": true,
  "primary_theme": "mate"
}
```

Пояснения: `id` — идентификатор из Lichess; `fen` — стартовая позиция; `moves` — ходы решения (UCI через пробел); `rating` — Elo задачи; `themes` — теги Lichess.

В API возвращается подмножество полей: `id`, `fen`, `moves`, `themes`, `rating` (для адаптивного режима дополнительно возвращаются служебные поля — см. ниже).

Отбор задач: не все задачи подходят для тренировочных раздач. `_is_puzzle_valid` исключает позиции, в которых решающий после выполнения ходов решения оказывается матован или остаётся под шахом (позиция не стабилизировалась). Соответственно, `/api/learning/puzzles` и `/api/learning/puzzles/adaptive` работают только с проверенным подмножеством.

### Пользователь

Публичное представление пользователя (метод `public()` модели) возвращается в ответах авторизации и `/api/auth/me`:

```json
{
  "id": 1, "login": "alice", "email": null, "elo": 1500,
  "role": "learner", "is_admin": false,
  "skill_band": null, "prior_band": null,
  "rating_estimate": null, "rating_scale": null,
  "created_at": "2026-01-01T10:00:00"
}
```

Хеш пароля не передаётся в ответах API. `role` — один из `learner` / `admin` (у гостей аккаунта нет).

## Авторизация — `routes/auth.py`

Функциональность аккаунтов сгруппирована под префиксом `/api/auth`. Пара токенов: `access_token` — короткоживущий JWT (HS256), `refresh_token` — случайная строка; в БД хранится только её SHA-256-хеш вместе с датой истечения.

| Метод | Путь | Доступ | Описание |
|---|---|---|---|
| POST | `/api/auth/register` | гостевой | Регистрация, выдача пары токенов |
| POST | `/api/auth/login` | гостевой | Вход по логину **или** email |
| POST | `/api/auth/refresh` | гостевой | Ротация пары токенов |
| POST | `/api/auth/logout` | гостевой | Отзыв refresh-токена |
| GET  | `/api/auth/me` | авторизованный | Данные текущего пользователя |
| GET  | `/api/auth/admin-only` | администратор | Пример защищённого ресурса |

### POST `/api/auth/register` — 201

```json
{ "login": "alice", "password": "password123", "email": "a@x.ru", "elo": 1200 }
```

Ограничения: `login` 3–32 символа, **только латинские** буквы, цифры, `_` и `-` (кириллица не принимается); `password` 8–72 символа и обязан содержать буквы и цифры (при нарушении — `422`); `elo` опционален, 100–3500; `email` опционален. Пароль хешируется bcrypt (cost 12), в базе живёт только хеш.

Ответ — пара токенов плюс пользователь:

```json
{
  "message": "Аккаунт создан",
  "access_token": "<jwt>", "refresh_token": "<raw>", "token_type": "bearer",
  "user": { "id": 1, "login": "alice", "elo": 1200, "role": "learner",
            "is_admin": false, "email": null, "created_at": "2026-01-01T10:00:00" }
}
```

Конфликты — `409` с `«Логин уже занят»` или `«Email уже занят»`, ошибки валидации — `422`.

### POST `/api/auth/login` — 200

```json
{ "login": "<логин или email>", "password": "..." }
```

Ответ не различает «несуществующий логин» и «неверный пароль»: для обоих случаев предусмотрен единый `401` `«Неверный логин или пароль»`. Успешный ответ по структуре совпадает с регистрацией, за исключением `message: "Вход выполнен"`.

### POST `/api/auth/refresh` — 200

```json
{ "refresh_token": "<raw>" }
```

Refresh-токен одноразовый: предыдущий отзывается, выдаётся новая пара. Если токен истёк, отозван или неизвестен — `401` `«Refresh-токен недействителен»`.

### POST `/api/auth/logout` — 200

```json
{ "refresh_token": "<raw>" }
```

Ответ `{"ok": true}`. Access-токен после отзыва остаётся действительным до истечения TTL (особенность stateless-JWT).

### GET `/api/auth/me` — 200

Заголовок `Authorization: Bearer <access_token>`, ответ `{"user": ...}` (см. схему пользователя). Без токена или с невалидным — `401`.

### GET `/api/auth/admin-only` — 200

```json
{ "ok": true, "secret": "Секретный дамп для <login>" }
```

Доступ ограничен зависимостью `require_roles("admin")`: без токена — `401`, роли `learner` и прочим — `403` `«Недостаточно прав»`.

## Администрирование — `routes/admin.py`

Эндпоинты под префиксом `/api/admin` доступны **только роли `admin`**.

| Метод | Путь | Доступ | Описание |
|---|---|---|---|
| GET | `/api/admin/users` | администратор | Список пользователей (краткий, без паролей и токенов) |

### GET `/api/admin/users` — 200

```json
{
  "users": [
    { "id": 1, "login": "admin", "role": "admin", "is_admin": true,
      "elo": null, "skill_band": null, "created_at": "2026-01-01T10:00:00" }
  ]
}
```

Гость — `401`, зарегистрированный (роль `learner`) — `403`.

## Игра против AI — `routes/game.py`

| Метод | Путь | Доступ | Описание |
|---|---|---|---|
| GET  | `/api/levels` | гостевой | Список уровней сложности |
| POST | `/api/legal-moves` | гостевой | Легальные ходы фигуры с поля |
| POST | `/api/move` | гостевой | Выполнить ход, получить новую позицию |
| POST | `/api/stockfish-analysis` | гостевой | Анализ позиции: лучший ход + оценка |
| POST | `/api/maia-move`, `/api/stockfish-move` | гостевой | Ход AI (Maia3/Stockfish), общий обработчик |
| POST | `/api/compare-moves` | гостевой | Сравнение хода Stockfish и Maia3 |
| POST | `/api/game/finish` | гостевой | Сохранить партию в БД (классификация в фоне) |

### GET `/api/levels`

```json
{ "levels": [ { "level": 1, "name": "Новичок", "elo": 1100, "engine": "maia3" }, ... ] }
```

Полный список: уровни 1–7 играет Maia3 (1100, 1300, 1500, 1700, 1900, 2200, 2600 Elo — «Новичок», «Любитель», «Клубный», «Опытный», «Сильный», «Мастер», «Гроссмейстер»), а 8-й — «Максимум (Stockfish)» с Elo 3600.

Температура Maia3 подбирается по Elo следующим образом: 2400+ → 0.0, 2000+ → 0.3, 1700+ → 0.6, 1400+ → 0.9, ниже → 1.2. Снижение температуры уменьшает случайность ходов модели.

### POST `/api/legal-moves`

```json
{ "fen": "...", "square": "e2" }
```

Поле `square` (исходная клетка фигуры) валидируется через `python-chess`. Ответ — перечень легальных ходов **фигуры с этой клетки** в SAN:

```json
{ "moves": ["e4", "e3"] }
```

### POST `/api/move`

```json
{ "fen": "...", "from_sq": "e2", "to_sq": "e4", "promotion": "q" }
```

`promotion` — q/r/b/n, по умолчанию q. Ход проверяется на легальность на уровне модели: нелегальный ход, ход не своей фигурой, пустая клетка `from_sq` и тому подобное дают `422`. При успехе:

```json
{ "fen": "<новая позиция>", "san": "e4",
  "status": "playing"|"check"|"checkmate"|"stalemate", "turn": "w"|"b" }
```

`status` отражает состояние *после* хода.

### POST `/api/stockfish-analysis`

```json
{ "fen": "..." }
```

Движок работает на полной силе (UCI_LimitStrength выключен), поиск 10 секунд:

```json
{
  "fen": "...", "best_move": "e2e4", "from": "e2", "to": "e4", "san": "e4",
  "evaluation": { "type": "cp", "value": 35 }
}
```

`evaluation` — формат python-stockfish: `{"type": "cp", "value": ...}` (центипешки) либо `{"type": "mate", "value": ±N}`. Если партия уже закончилась или движок недоступен — мягкая ошибка `{"error": ...}`.

### POST `/api/maia-move`, POST `/api/stockfish-move`

Оба адреса ведут на один обработчик. Тело:

```json
{ "fen": "...", "elo": 1500, "moves": ["e2e4", "e7e5"], "engine": "maia3" }
```

- `engine` — `"maia3"` (по умолчанию) или `"stockfish"`;
- `elo` — 0–3000, определяет силу Maia3 (и температуру, см. `/api/levels`);
- `moves` — история партии от старта; Maia3 она нужна для контекста, поле опционально.

В обоих случаях формируется каскадный фолбэк: если Maia3 недоступна или завершила ход с ошибкой, ход выполняет Stockfish (полная сила, ~10 с); при недоступности движка — случайный легальный ход. Таким образом, ответ формируется всегда, а поле `engine` указывает фактического исполнителя.

```json
{ "fen": "...", "san": "e4", "from": "e2", "to": "e4",
  "status": "playing", "turn": "b", "evaluation": null,
  "engine": "maia3"|"stockfish" }
```

`evaluation` (оценка после хода) заполняется **только** когда движок выполнил ход как Stockfish — для чистой Maia3 это `null`.

### POST `/api/compare-moves`

```json
{ "fen": "...", "elo": 1500, "moves": [] }
```

Ходы Stockfish и Maia3 возвращаются раздельно; это позволяет сопоставить ход игрока с ходом модели человеческого уровня:

```json
{
  "fen": "...",
  "stockfish": { "move": "e2e4", "from": "e2", "to": "e4", "san": "e4", "evaluation": {...} },
  "maia3":     { "move": "e2e4", "from": "e2", "to": "e4", "san": "e4", "elo": 1500 },
  "same_move": true
}
```

### POST `/api/game/finish`

```json
{ "moves": ["e2e4", "e7e5"], "user_id": null, "elo": 1500,
  "engine": "maia3", "result": "1-0", "status": "checkmate" }
```

Партия сохраняется в БД, классификация ходов (оценка качества каждого хода) выполняется Stockfish — по секунде на ход — в фоновом режиме через `BackgroundTasks`, чтобы не задерживать ответ. Эндпоинт сразу возвращает `{"ok": true, "status": "scheduled"}`.

## Анализ позиции — `routes/analysis.py` + `routes/analyze.py`

Логика распределена по двум роутерам и представляет единый интерфейс. `POST /api/analyze` имеет алиас `/api/stockfish-analyze`.

Внимание: не путать с эндпоинтом `/api/stockfish-analysis` из раздела «Игра против AI» — это отдельный эндпоинт (лучший ход и оценка, без топ-5 продолжений). `/api/analyze` и `/api/stockfish-analyze` — один и тот же обработчик.

| Метод | Путь | Доступ | Описание |
|---|---|---|---|
| POST | `/api/analyze` (= `/api/stockfish-analyze`) | гостевой | Лучший ход + оценка + топ-5 ходов |
| POST | `/api/eval` | гостевой | Быстрая оценка, только eval |
| POST | `/api/similarity/search` | гостевой | Поиск похожих позиций в векторной БД |
| POST | `/api/analyze/legal-moves` | гостевой | Все легальные ходы (UCI) |
| POST | `/api/analyze/move` | гостевой | Выполнить ход по UCI |
| POST | `/api/analyze/position` | гостевой | Оценка материала без движка |

### POST `/api/analyze`

```json
{ "fen": "..." }
```

Полный анализ: лучший ход (`get_best_move_time`, ~10 с), оценка (1 с) и топ-5 продолжений (`get_top_moves`, ~200 тыс. нод) — всё на полной силе Stockfish:

```json
{ "fen": "...", "best_move": "e2e4", "from": "e2", "to": "e4", "san": "e4",
  "evaluation": { "type": "cp", "value": 35 }, "top_moves": [ { "Move": "e2e4", ... }, ... ] }
```

`top_moves` — сырые строки из вывода python-stockfish, поэтому состав ключей в элементах зависит от движка (`Move`/`UCI`, `Centipawn`, `Mate`, `MultiPV` и т.п.); жёстко полагаться на него не следует.

### POST `/api/eval`

```json
{ "fen": "..." }
```

Упрощённый вариант без «лучшего хода» — только оценка. Значение нормируется относительно стороны, чей ход: если ход чёрных, оценка инвертируется, чтобы положительное значение всегда означало преимущество ходящей стороны. Ответ `{"evaluation": {...} | null}`.

### POST `/api/similarity/search`

```json
{ "fen": "...", "top_k": 5 }
```

`top_k` — 1–100. Поиск похожих позиций в векторной БД. Модули эмбеддингов опциональные; если они не установлены — не ошибка, а мягкий ответ `{"error": "Vector search modules not available", "results": []}`. При успехе: `{"fen": ..., "top_k": 5, "results": [...], "count": N}`.

### POST `/api/analyze/legal-moves`

```json
{ "fen": "..." }
```

```json
{ "legal_moves": ["e2e4", ...] }
```

Ходы в UCI. На невалидный FEN: `{"error": "Invalid FEN string", "legal_moves": []}`.

### POST `/api/analyze/move`

```json
{ "fen": "...", "move": "e2e4" }
```

`move` строго в UCI. Ответ: `{"new_fen": "...", "move_san": "e4"}`. Три варианта мягкой ошибки: `"Invalid FEN string"`, `"Invalid move format (use UCI)"`, `"Illegal move"`.

### POST `/api/analyze/position`

```json
{ "fen": "...", "depth": 10 }
```

Оценка **материала** без движка (P=1, N/B=3, R=5, Q=9). Глубина не используется (это оценка материала, а не поиск); параметр сохранён для совместимости:

```json
{ "evaluation": 3, "best_move": null, "depth": 10,
  "note": "Упрощённая оценка материала, Stockfish не установлен" }
```

## Учебный курс «Обучение» — `routes/training.py`

Последовательный курс по правилам движения фигур (в отличие от тактических задач, которые вынесены в `/api/learning/*`). Курс состоит из модулей (тем) и уроков; каждая задача поддерживает несколько режимов проверки.

| Метод | Путь | Доступ | Описание |
|---|---|---|---|
| GET  | `/api/training/progress` | авторизованный | Прогресс по курсу |
| GET  | `/api/training/modules` | авторизованный | Карточки всех модулей |
| GET  | `/api/training/modules/{slug}` | авторизованный | Модуль и его уроки |
| GET  | `/api/training/lessons/{lesson_id}` | авторизованный | Урок и его задания |
| GET  | `/api/training/tasks/{task_id}` | авторизованный | Задание (без эталона) |
| POST | `/api/training/tasks/{task_id}/check` | опциональный | Проверка ответа + обновление сложности темы |

### GET `/api/training/progress`

Сводка по включённым модулям и всем попыткам пользователя:

```json
{
  "modules": { "completed": 1, "total": 8, "percent": 12.5 },
  "attempts": { "total": 12, "correct": 9, "accuracy": 75.0 },
  "streak": 3
}
```

- Модуль считается пройденным, если каждое его включённое задание хотя бы один раз решено верно.
- `streak` — серия дней с активностью; сохраняется при наличии активности за текущий или предыдущий день.
- При отсутствии попыток `attempts.accuracy` равен `null`, а не `0` (решение принято в рамках аудита B-2 с целью исключить ошибочное отображение «0%» статистики).
- Поле `topics` в **этом** ответе отсутствует: точность по темам курса предоставляется единым источником — `/api/learning/weaknesses` (используется виджетом прогресса и адаптивной раздачей). Следует учитывать, что ключ `topics` того же назначения используется и в ответах `/api/learning/difficulty` и `/api/learning/weaknesses` — речь именно о `/api/training/progress`.

### GET `/api/training/modules`

```json
{ "modules": [ { "id": 1, "slug": "pawn", "title": "Пешка", "description": "...",
                 "sort_order": 1, "enabled": true, "lesson_count": 4, "task_count": 12 } ] }
```

### GET `/api/training/modules/{slug}`

`404` — модуль не найден. Если модуль существует, но выключен (`enabled: false`), `lessons` возвращается пустым списком:

```json
{
  "module": { "id": 1, "slug": "pawn", "title": "Пешка", "description": "...", "enabled": true, "sort_order": 1 },
  "lessons": [ { "id": 1, "slug": "pawn-moves", "title": "Ходы пешки", "sort_order": 1, "task_count": 3 } ]
}
```

### GET `/api/training/lessons/{lesson_id}`

`404` — урок не найден или выключен. Задания возвращаются без полей-эталонов (`accepted_moves`/`correct_option`) — они используются только при проверке (см. `check`):

```json
{
  "lesson": { "id": 1, "module_id": 1, "slug": "pawn-moves", "title": "Ходы пешки",
              "theory": "<markdown>", "sort_order": 1 },
  "tasks": [ { "id": 11, "lesson_id": 1, "task_type": "select_move", "title": "...",
               "instruction": "...", "fen": "...", "source_square": "e2",
               "difficulty": 1, "payload": {...}, "sort_order": 1 } ]
}
```

### GET `/api/training/tasks/{task_id}`

```json
{ "task": { "id": 11, ..., "payload": {...}, "sort_order": 1 } }
```

Возвращается публичное представление задания; эталоны ответов не раскрываются.

### POST `/api/training/tasks/{task_id}/check`

```json
{ "answer": { "...": "..." }, "hints_used": 0, "response_time_ms": 4300 }
```

- Структура `answer` зависит от `payload.mode` задачи: `all_legal_moves`, `any_legal_move`, `capture_squares`, `legal_capture`, `accepted_moves`.
- `hints_used` — 0–100, `response_time_ms` — опционально, ≥ 0.
- Авторизация не является обязательной: гость узнаёт результат, однако попытка сохраняется без `user_id` и сложность темы не обновляется.

Успех:

```json
{
  "ok": true, "task_id": 11, "attempt_id": 55, "attempt_number": 2,
  "hints_used": 0, "correct": true,
  "topic": "pawn", "difficulty_update": null,
  "score": 1.0, "feedback": "...", "explanation": "...",
  "expected": { "...": "..." },
  "details": { "...": "..." }
}
```

- `topic` — slug учебной темы; для гостя `null`. `difficulty_update` — результат обновления B-сложности темы после попытки (у гостя также `null`).
- `explanation` — эталонное объяснение задачи; `expected`/`details` — служебные данные проверки.

Ошибки: `404` — задача не найдена/выключена; `422` — структура ответа не соответствует типу задания.

## Тактические пазлы и уровень — `routes/learning.py`

Раздел объединяет тактические задачи из набора Lichess, тест определения уровня и механизмы адаптивной тренировки (персональная раздача, сложность B2, определение слабых/сильных тем).

| Метод | Путь | Доступ | Описание |
|---|---|---|---|
| POST | `/api/learning/level-test/start` | опциональный | Начать тест уровня |
| POST | `/api/learning/level-test/check` | гостевой | Проверить ответ на задачу теста |
| POST | `/api/learning/level-test/result` | гостевой | Посчитать уровень (read-only, legacy) |
| POST | `/api/learning/level-test/submit` | авторизованный | Зафиксировать итог теста в профиле |
| POST | `/api/learning/puzzle/attempt` | авторизованный | Сохранить попытку пазла |
| GET  | `/api/learning/progress` | авторизованный | Прогресс по пазлам |
| GET  | `/api/learning/difficulty` | авторизованный | B-сложность по темам |
| GET  | `/api/learning/weaknesses` | авторизованный | Профиль сильных/слабых тем |
| GET  | `/api/learning/puzzles/adaptive` | авторизованный | Персональный набор (70/30) |
| POST | `/api/learning/puzzle/check` | гостевой | Проверка отдельного пазла |
| GET  | `/api/learning/puzzles` | гостевой | Набор пазлов с решениями |

Тестовый набор формируется по Elo-корзинам (константа `LEVEL_BUCKETS`). Уровень определяет диапазон рейтинга задач:

| Level | Название | Band (записывается в elo) | Диапазон пазлов |
|---|---|---|---|
| 1 | Новичок | 500 | 500–900 |
| 2 | Любитель | 1000 | 900–1300 |
| 3 | Клубный | 1500 | 1300–1700 |
| 4 | Продвинутый | 2000 | 1700+ |

### POST `/api/learning/level-test/start`

Авторизация не является обязательной. Из каждой корзины случайно отбираются до 5 задач (суммарно до ~20), полученный список перемешивается, чтобы задачи не следовали в порядке возрастания сложности. Для авторизованного пользователя создаётся запись `level_tests` (status `started`) и возвращается `test_id`; для гостя `test_id: null` — фиксация результата в профиле для гостя не предусмотрена, при этом проверка задач сохраняется.

```json
{
  "test_id": 7,
  "questions": [ { "id": "00Dlt", "fen": "...", "themes": [...], "rating": 510 } ],
  "total": 20
}
```

Если база пазлов не загружена: `{"error": "База паззлов не загружена", "questions": [], "test_id": null}`.

### POST `/api/learning/level-test/check`

```json
{ "puzzle_id": "00Dlt", "move": "d3f1" }
```

`move` передаётся в UCI (4–5 символов); приём SAN в запросе также поддерживается (ход разбирается и приводится к UCI). Поле `solution` в ответе **всегда** возвращается в UCI, независимо от формата запроса. Базовый ответ: `{ "correct": true, "solution": "d3f1", "puzzle_rating": 510, "themes": [...] }`.

Если ход не совпал с эталонным, выполняется дополнительная сверка со Stockfish, по результатам которой ответ дополняется полями `strong_but_different` (разница оценок не превышает 50 центипешек) и `is_best` (ход совпадает с лучшим). Механизм позволяет отличать технически неверный, но семантически сильный ход.

Служебные варианты: `{"error": "...", "correct": false, "solution": null}` (база не загружена / задача не найдена / у задачи нет решения) и `{"correct": false, "solution": ..., "message": "Некорректный ход"}`.

### POST `/api/learning/level-test/result`

```json
[ { "puzzle_id": "...", "correct": true }, ... ]
```

Вычисляет уровень без сохранения в профиль (read-only, оставлен для обратной совместимости). Принимает на вход готовые ответы и возвращает разбивку по корзинам:

```json
{
  "level": 2, "band": 1000,
  "score": { "1": {"total": 4, "correct": 3, "name": "Новичок"}, ... },
  "result": { "level": 2, "name": "Любитель", "band": 1000 }
}
```

Правило: уровень — максимальная корзина, где к тесту отнесено ≥ 3 задач и решено верно ≥ 3 из них. Разбивка по всем корзинам доступна в поле `score`.

### POST `/api/learning/level-test/submit`

```json
{ "test_id": 7, "answers": [ { "puzzle_id": "00Dlt", "correct": true } ] }
```

Авторизация обязательна. Фиксирует итог теста в профиле: `elo` пользователя устанавливается равным `band`.

- Идемпотентность обеспечена блокировкой строки теста (`SELECT ... FOR UPDATE`): повторный submit с тем же `test_id` вернёт сохранённый результат и не пересчитает его.
- Ответ: `{ "ok": true, "test_id": 7, "level": 2, "band": 1000, "score": {...}, "already_submitted": false }`.
- Ошибки: `404` — тест не найден или принадлежит другому пользователю; `422` — повторяющиеся `puzzle_id` либо набор ответов не совпадает с задачами теста; `503` — база не загружена.

### POST `/api/learning/puzzle/attempt`

```json
{ "puzzle_id": "00Dlt", "correct": true }
```

Сохраняет попытку решения пазла (итог известен клиенту, поскольку решение воспроизводится на клиентской стороне) и обновляет B-сложность темы. `puzzle_id` проверяется по загруженной базе. Ответ:

```json
{ "ok": true, "attempt_id": 120, "puzzle_id": "00Dlt", "correct": true,
  "topic": "mate", "difficulty_update": null | {...} }
```

`404` — такого `puzzle_id` нет в базе; `503` — база не загружена.

### GET `/api/learning/progress`

```json
{ "attempted": 15, "solved": 10, "attempts": 25, "correct_attempts": 18, "accuracy": 72.0 }
```

- `attempted`/`solved` — количество **уникальных** пазлов (пробованных / решённых хотя бы один раз);
- `attempts`/`correct_attempts` — общее число попыток с учётом повторов;
- `accuracy` — аналогично прогрессу курса, при отсутствии попыток равен `null`, а не `0`.

### GET `/api/learning/difficulty`

Текущая B-сложность по каждой теме курса:

```json
{
  "topics": [
    { "slug": "pawn", "title": "Пешка", "accuracy": 75.0, "attempts": 12,
      "current_difficulty": 2, "puzzle_rating_band": { "min": 500, "max": 900 } }
  ],
  "rule": { "min_difficulty": 1, "max_difficulty": 3, "default_difficulty": 2,
            "increase_if_accuracy_gt": 80, "decrease_if_accuracy_lt": 50 }
}
```

Сложность 1–3 определяет диапазон рейтинга предлагаемых по теме пазлов. Сложность возрастает при точности выше 80, снижается при точности ниже 50; пороговые значения приведены в поле `rule`. `503` — база не загружена.

### GET `/api/learning/weaknesses`

Единый источник данных о точности по темам курса: объединяет попытки курса и пазлов. Результат используется виджетом прогресса и адаптивной раздачей, что обеспечивает согласованность отображаемых значений (аудит B-2).

```json
{
  "topics": [
    { "slug": "bishop", "title": "Слон", "attempts": 8, "correct": 5, "accuracy": 62.5,
      "training": { "attempts": 5, "correct": 4 }, "puzzles": { "attempts": 3, "correct": 1 } }
  ],
  "weak_topics": [
    { "slug": "bishop", "title": "Слон", "attempts": 8, "correct": 5, "accuracy": 62.5,
      "training": { "attempts": 5, "correct": 4 }, "puzzles": { "attempts": 3, "correct": 1 },
      "rank_accuracy": 62.5, "observed": true }
  ],
  "strong_topics": [
    { "slug": "queen", "title": "Ферзь", "attempts": 12, "correct": 11, "accuracy": 91.7,
      "training": { "attempts": 7, "correct": 6 }, "puzzles": { "attempts": 5, "correct": 5 },
      "rank_accuracy": 91.7, "observed": true }
  ],
  "unmapped_puzzle_attempts": 0,
  "selection_rule": { "weak_share": 0.70, "strong_share": 0.30, "weak_topic_count": 3 }
}
```

- `accuracy` темы = `(training.correct + puzzles.correct) / (training.attempts + puzzles.attempts)`; без попыток — `null`.
- `weak_topics` — нижние 3 темы по `rank_accuracy`; `strong_topics` — верхние 3, но **только с реальными попытками** (`observed: true`). Тема без попыток не может считаться сильной (условие ранжирования).
- `rank_accuracy` — у темы без попыток это нейтральные 50, но только для ранжирования внутри слабых; фактическая `accuracy` остаётся `null`.
- `unmapped_puzzle_attempts` — попытки пазлов, которые не удалось привязать ни к одной теме курса.
- `selection_rule` — пропорция 70/30, используемая адаптивной раздачей (описание ниже).

### GET `/api/learning/puzzles/adaptive?count=N`

Авторизация обязательна, `count` 1–100 (по умолчанию 20). Персональная раздача по правилу 70/30: 70% задач — по слабым темам, 30% — по сильным; сложность берётся из B-профиля. Для каждой задачи дополнительно возвращаются служебные поля, отражающие результат отбора:

```json
{
  "puzzles": [
    { "id": "00Dlt", "fen": "...", "moves": "d3f1", "themes": [...], "rating": 510,
      "adaptive_group": "weak", "adaptive_topic": "bishop",
      "adaptive_difficulty": 2, "difficulty_match": true }
  ],
  "total": 20, "mode": "adaptive",
  "weak_topics": [ ... ], "strong_topics": [ ... ],
  "difficulty_profile": [ { "slug": "pawn", "title": "Пешка", "current_difficulty": 2, "puzzle_rating_band": {...} } ],
  "allocation": { "requested": 20, "weak_quota": 14, "strong_quota": 6,
                  "weak_selected": 14, "strong_selected": 6, "fallback_selected": 0,
                  "difficulty_matched": 18, "difficulty_fallback": 2 },
  "selection_rule": { "weak_share": 0.70, "strong_share": 0.30, "weak_topic_count": 3 }
}
```

Состав полей `allocation`: `weak_quota`/`strong_quota` — целевые квоты (`round(count * 0.70)` и остаток); `weak_selected`/`strong_selected` — фактически отобранное по темам; `fallback_selected` — количество задач, добавленных случайным добором при недостатке задач по слабым/сильным темам; `difficulty_matched`/`difficulty_fallback` — количество задач, попавших в целевую сложность. `503` — база не загружена.

### POST `/api/learning/puzzle/check`

Проверка единичного пазла — тот же обработчик и то же тело (`PuzzleAnswerRequest`), что и у `/api/learning/level-test/check`; оба эндпоинта гостевые.

### GET `/api/learning/puzzles?count=20&topic=<slug>`

Раздача тактических задач для тренировочной страницы. В отличие от level-test, позиции сопровождаются полным решением в поле `moves`. Задачи равномерно распределяются по Elo-корзинам (по `count / 4` на корзину); при исчерпании корзин выполняется добор остальных подходящих задач.

`topic` — slug модуля курса; при его передаче отбираются только темы Lichess, связанные с этим модулем. Незнакомый slug — строгий `400` `«Неизвестная тема учебного курса»`.

```json
{ "puzzles": [ { "id": "00Dlt", "fen": "...", "moves": "d3f1", "themes": [...], "rating": 510 } ],
  "total": 20, "topic": null }
```

Невалидные задачи (решающий остаётся под матом/шахом) отфильтрованы; если база не загружена — `{"error": "База паззлов не загружена", "puzzles": []}`.

## База знаний (дебюты) — `routes/knowledge.py`

Книжные дебюты загружаются из `backend/knowledge/openings.json`. Каждый дебют содержит `name`, `eco`, `pgn` (исходная партия), `fen` (итоговая позиция), `fens` (последовательность позиций) и `moves` (соответствующие ходы). Поиск позиции в базе выполняется по раскладке фигур без учёта очереди хода.

| Метод | Путь | Доступ | Описание |
|---|---|---|---|
| GET  | `/api/knowledge/openings` | гостевой | Все дебюты |
| GET  | `/api/knowledge/opening?fen=...` | гостевой | Самый глубокий дебют для позиции |
| GET  | `/api/knowledge/random-opening` | гостевой | Случайный дебют |
| POST | `/api/knowledge/check-move` | гостевой | Признаётся ли позиция теорией + продолжения |

### GET `/api/knowledge/openings`

```json
{ "openings": [ { "name": "Italian Game", "eco": "C50", "pgn": "...",
                  "fen": "...", "fens": ["..."], "moves": ["e2e4", "e7e5", ...] } ] }
```

Без загруженной базы: `{"openings": [], "error": "..."}`.

### GET `/api/knowledge/opening?fen=...`

Находит все дебюты, покрывающие переданную позицию, и возвращает самый глубокий — тот, где сыграно больше всего ходов. Ответ `{"opening": {...}}` или `{"opening": null, "message": "Дебют не найден в базе"}`.

### GET `/api/knowledge/random-opening`

`{"opening": {...}}` — случайный элемент из базы.

### POST `/api/knowledge/check-move`

```json
{ "fen": "..." }
```

```json
{ "in_theory": true, "opening": "Italian Game", "eco": "C50",
  "pgn": "...", "next_moves": ["e2e4", "Bc4"] }
```

`next_moves` — книжные продолжения из всех покрывающих позицию дебютов, без повторов. Если позиция неизвестна теории: `{"in_theory": false, "message": "Позиция не найдена в базе теории"}`.

## Чат с AI-ассистентом — `routes/chat.py`

Сообщения хранятся в таблице `chat_messages` (роль, текст, unix-время). Новые сообщения опрашиваются клиентом в режиме long-polling через параметр `after=<id>`.

| Метод | Путь | Доступ | Описание |
|---|---|---|---|
| POST | `/api/chat/ingest` | гостевой | Сохранить сообщение (роль user/assistant) |
| GET  | `/api/chat/messages?after=<id>` | гостевой | Сообщения с id > after |
| GET  | `/api/chat/messages/count` | гостевой | Счётчик сообщений |
| POST | `/api/chat/ask` | опциональный | Вопрос ассистенту с учётом позиции |

### POST `/api/chat/ingest`

```json
{ "message": "...", "role": "assistant" }
```

`role` — `user`|`assistant` (по умолчанию assistant), `message` — 1–4000 символов; текст перед сохранением проходит через `sanitize_text`. Ответ: `{"ok": true, "count": <всего сообщений в БД>}`. Значение `count` здесь совпадает с величиной, которую возвращает GET `/api/chat/messages/count`, но отдаётся в составе обёртки вместе с `ok`.

### GET `/api/chat/messages?after=0`

Сообщения с `id > after` по возрастанию. `ts` — **UNIX-время в секундах (float)**, а не ISO-строка — учтите при форматировании на фронте:

```json
{ "messages": [ { "id": 1, "role": "user", "text": "...", "ts": 1735660800.0 } ] }
```

### GET `/api/chat/messages/count`

`{"count": <всего сообщений в БД>}` — та же величина, что и `count` из POST `/api/chat/ingest`, но без обёртки `ok` (эндпоинт предназначен для периодического опроса).

### POST `/api/chat/ask`

```json
{ "message": "Что мне сыграть?", "fen": "...", "moves": ["e2e4"], "elo": 1500, "is_greeting": false }
```

Механика запроса:

- `is_greeting: true` — используется системный промпт без позиции, ответ короткий (режим приветствия).
- Вопрос и ответ сохраняются в `chat_messages`; при наличии авторизации — с `user_id`, для гостя — без.
- Модель может дописать в конец ответа маркер `[ДОСКА: <FEN>]`. Маркер извлекается из текста, FEN валидируется и возвращается отдельным полем ответа. Если маркер отсутствует, но вопрос по набору ключевых слов относится к позиции, подставляется активная позиция игрока.
- `arrow` — фактический лучший ход Stockfish для отображаемой позиции. Стрелка формируется детерминированно по результатам движка и не извлекается из текста ответа LLM.
- Если Gigachess не настроен, `reply` содержит точный текст: `AI не подключён. Проверьте GIGACHESS_BASE_URL в .env.`

Ответ:

```json
{ "reply": "<текст>", "fen": "<FEN или null>", "arrow": { "uci": "e2e4", "san": "e4", "from": "e2", "to": "e4" } | null }
```

## Компьютерное зрение — `routes/vision.py`

### POST `/api/analyze-image`

`multipart/form-data`, поле `file` — изображение доски (png/jpg/jpeg/bmp/webp). Распознавание позиции выполняет модель LLaVA.

Текущее состояние: модель распознавания отключена — `load_llava()` возвращает `False` при отсутствии установленных `transformers` и `torch`. В этом случае возвращается не HTTP-ошибка, а тело с указанием причины:

```json
{ "error": "LLaVA не загружена. Проверьте что transformers и torch установлены.",
  "hint": "pip install transformers torch pillow" }
```

При успешной работе модели ответ такой:

```json
{ "fen": "rnbqkbnr/pppppppp/8/...", "message": "Позиция распознана!" }
```

Ошибки распознавания: `{"error": "<текст>", "fen": null}`.

## Профиль с Lichess / Chess.com — `routes/chess_profile.py`

### GET `/api/chess-profile?username=...&platform=lichess`

`username` обязателен (1–64), `platform` по умолчанию `lichess`; для chess.com допустимы значения `chess.com`, `chesscom`, `chess_com`, `chess`. Используются официальные публичные API без ключей (`User-Agent` — `SfeduCastling/0.1`, таймаут 8 секунд). Ответ нормализуется в единую структуру.

```json
{
  "platform": "lichess", "username": "magnus", "title": "GM", "name": null,
  "country": "https://lichess1.org/assets/images/flags/NO.png",
  "avatar": "https://...",
  "perfs": { "bullet": {"rating": 2900, "games": 500}, "blitz": {"rating": 2830, "games": 1000},
             "rapid": {"rating": 2833, "games": 900}, "classical": {"rating": 2847, "games": 200} },
  "counts": { "all": 2600, "wins": 1800, "losses": 500, "draws": 300 }
}
```

Особенности нормализации, которые следует учитывать при интеграции:

- Для Lichess поле `country` содержит URL изображения флага (значение `profile.flag` без преобразования), а не код страны. Для chess.com — код страны (последний сегмент URL поля `country`).
- Ключи `perfs` для Lichess: `bullet`/`blitz`/`rapid`/`classical`; для chess.com — `bullet`/`blitz`/`rapid`/`daily`. Для chess.com выполняются два запроса (профиль + stats), а `games` вычисляется как сумма записей win+loss+draw по тайм-контролю.
- Для Lichess `rating` равен `null`, пока по тайм-контролю не сыграно ни одной партии.
- Поле `name` для Lichess — это `firstName` (возможно `null`), для chess.com — полное имя.

Ошибки — HTTP-статусом: `404` «Пользователь не найден на Lichess/Chess.com», `502` «Не удалось связаться с Lichess/Chess.com»; сторонние коды 4xx/5xx прокидываются как есть («Lichess ответил 429»).

## Объяснение хода — `routes/explanation.py`

### POST `/api/explain-move`

```json
{ "fen": "...", "move": "e2e4", "elo": 1500, "moves": ["e2e4"] }
```

Объясняет **последний ход игрока**: сравнивает его с лучшим ходом Stockfish и с ходом Maia3 на заданном Elo, а потом строит объяснение — детерминированное или через Gigachess (пакет `backend/llm/explainer`, функция `explain_move`; для LLM-части строится граундинг и запускается цикл ремонта ответа).

Ответ — полный результат `explain_move`:

```json
{
  "ok": true,
  "played_move": { "...": "..." },
  "stockfish": { "...": "..." },
  "maia3": { "...": "..." },
  "same_as_maia3": false,
  "fen_before": "...",
  "fen_after": "...",
  "move_facts": { "...": "..." },
  "position_facts": { "...": "..." },
  "derived_facts": { "...": "..." },
  "explained_move": { "...": "..." },
  "gigachess_grounding": {
    "facts": { "...": "..." },
    "allowed_squares": ["e4", ...], "allowed_piece_words": [...],
    "moved_piece_controls": [...], "attack_squares": [...], "defended_squares": [...],
    "terminal": false, "opponent_king_square": null,
    "opponent_legal_moves": [...], "castling_rook_from": null, "castling_rook_to": null,
    "king_safety_supported": false, "generic_defense_supported": false
  },
  "gigachess": { "available": false, "used": false, "retry": false,
                 "attempt_count": 0, "input_type": "..." },
  "explanation_source": "deterministic" | "gigachess",
  "explanation": "<текст объяснения>"
}
```

- `played_move` — ход игрока; `stockfish`/`maia3` — их ходы и оценки; `same_as_maia3` — совпал ли ход с человеческим (Maia3) на его Elo.
- `explanation_source`: `deterministic`, если Gigachess недоступен либо не прошёл валидацию, иначе `gigachess`.
- Блок `gigachess` присутствует **всегда** — даже при отключённом клиенте это минимальный `{available: false, used: false, retry: false, attempt_count: 0, input_type: ...}`. Не рассчитывайте на его отсутствие.
- Ошибки: `400` — некорректный запрос или ход (внутренний `ValueError`), `500` — сбой hybrid-анализа.

## Данные и датасет — `routes/data.py`

| Метод | Путь | Доступ | Описание |
|---|---|---|---|
| POST | `/api/save-move-to-dataset` | гостевой | Ход пользователя + оценки Stockfish в датасет |
| GET  | `/api/dataset/export` | гостевой | Выгрузка всего датасета в NDJSON |
| POST | `/api/parse-pgn-text` | гостевой | Разбор PGN-текста |
| POST | `/api/parse-pgn` | гостевой | Разбор загруженного PGN-файла |

### POST `/api/save-move-to-dataset`

```json
{ "fen": "...", "move": "e2e4", "user_id": null, "game_id": "" }
```

Перед записью для позиции считаются лучший ход и оценка Stockfish (по 1 секунде каждый, под общим локом движка). Ответ:

```json
{ "status": "saved", "dataset_size": 42,
  "data": { "fen": "...", "user_move": "e2e4", "stockfish_move": "e2e4",
            "stockfish_eval": {"type": "cp", "value": 35}, "user_id": null,
            "game_id": "", "timestamp": "2026-01-01T10:00:00+00:00" } }
```

### GET `/api/dataset/export`

Стрим всего датасета как NDJSON (`application/x-ndjson`, по объекту на строку). Поля строки: `fen`, `user_move`, `stockfish_move`, `stockfish_eval`, `user_id`, `game_id`, `timestamp` — формат совместим со старым `dataset.jsonl`.

### POST `/api/parse-pgn-text`

```json
{ "pgn": "<текст партии>" }
```

Разбирает одну партию:

```json
{ "games_count": 1,
  "games": [ { "id": 1, "white": "...", "black": "...", "result": "1-0",
               "date": "2026.01.01", "opening": "...",
               "moves": [ { "fen": "...", "move": "start", "move_number": 0, "turn": "white" },
                          { "fen": "...", "move": "e4", "move_number": 1, "turn": "white", "uci": "e2e4" },
                          { "fen": "...", "move": "end", "move_number": 1, "turn": "black" } ] } ] }
```

Семантика ходов: `fen` — позиция **до** выполнения хода; `move` — SAN (первый элемент — маркер `start`, последний — `end`); `uci` — UCI, есть только у реальных ходов; `move_number` увеличивается после хода чёрных. Тег `opening` берётся из заголовков партии, а если его нет — подставляется `"?"`. Ошибка парсинга: `{"error": "..."}`.

### POST `/api/parse-pgn`

`multipart/form-data`, поле `file`. Парсит **все** партии файла — но не больше `settings.data.max_games_to_parse`. Формат ответа тот же, `games_count` — число партий. Ошибки: `{"error": "Не удалось найти партии в файле"}` либо `{"error": "Ошибка парсинга: ..."}`.

## Запуск и приложение

- Точка входа — `backend/app.py`, `uvicorn.run` на `127.0.0.1:8005`.
- Статика: `frontend/dist/` смонтирована на `/`, все не-API пути отдают SPA.
- Swagger: `/docs`.
- `operation_id` задан вручную только для эндпоинта `analyze/move` (`analyze_make_move`); для остальных эндпоинтов Swagger использует значения по умолчанию.