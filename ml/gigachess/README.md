# GigaChess Client 🌐

HTTP-клиент для облачного инференса **GigaChess** (Cloud.ru). Используется как бэкенд-движок LLM для шахматных запросов.

---

## 📄 Файлы

| Файл | Назначение |
|------|-----------|
| `sber_client.py` | Основной клиент: `HTTP11Adapter`, `get_session()`, `ping_server()`, `ask_gigachess(prompt, fen)`, `extract_uci_move(text)`. |

---

## 🔧 Как работает

1. **`ping_server()`** — GET-запрос к `<BASE_URL>/ping`, проверяет доступность инстанса (прогрев после cold start).
2. **`ask_gigachess(prompt, fen="")`** — POST-запрос к `<BASE_URL>/chat`:
   - промпт передаётся в `messages[0].content`;
   - если задан FEN — передаётся в `attachments` (модель понимает позицию по нотации);
   - параметры инференса: `temperature=0.2`, `top_p=0.95`, `max_tokens=128`, `model="gigachess"`;
   - таймаут **900 секунд** — на случай прогрева инстанса.
3. **`extract_uci_move(text)`** — извлекает первый UCI-ход (`e2e4`, `e7e8q`) из ответа модели.

### Принудительный HTTP/1.1

`HTTP11Adapter` принудительно использует HTTP/1.1 вместо HTTP/2 — решает проблему зависания запросов на h2-endpoint'ах.

> Внимание: `verify=False` отключает проверку TLS-сертификата (для сервисов Cloud.ru с самоподписанными/недоверенными сертификатами). Используется только для исследовательских целей.

---

## 🚀 Использование

### Проверка и ручной запрос

```bash
python sber_client.py
```

Скрипт: ждёт прогрева инстанса (до 60 попыток по 10 сек), отправляет тестовый запрос со стартовой позицией и выводит ответ + извлечённый UCI-ход.

### Как библиотека

```python
from sber_client import ask_gigachess, ping_server, extract_uci_move

if ping_server():
    answer = ask_gigachess("Какой лучший ход?", fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(answer)
    print(extract_uci_move(answer))
```

---

## ⚙️ Конфигурация

`BASE_URL` захардкожен в начале файла и указывает на персональный Model RUN инстанс:

```
https://<uuid>.modelrun.inference.cloud.ru
```

Для смены инстанса отредактируйте `BASE_URL` в `sber_client.py`.

> В production-backend также используется клиент `backend/llm/gigachess.py` (`GigachessClient`), настраиваемый через `GIGACHESS_BASE_URL` в `.env`.