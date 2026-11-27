"""Клиент LLM с двумя бэкендами.

openai    — OpenAI-совместимый шлюз (anymodel.org и подобные): POST {LLM_BASE_URL}/chat/completions.
            Только stdlib, без зависимостей. Настройки: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL.
anthropic — официальный Anthropic SDK (pip install anthropic, Python 3.10+):
            ANTHROPIC_API_KEY, модель по умолчанию claude-opus-5-5.

temperature не передаём: Claude Opus 5.5 отклоняет параметры сэмплинга (400). Детерминизм
обеспечивают жёсткий промпт и скриптовые валидаторы, а не temperature.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
import urllib.error
import urllib.request


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, backend: str | None = None, model: str | None = None,
                 max_tokens: int = 4000, timeout: float = 180.0, max_retries: int = 5,
                 effort: str = "low", extra_body: dict | None = None):
        self.backend = backend or ("openai" if os.environ.get("LLM_BASE_URL") else "anthropic")
        self._tl = threading.local()
        # LLM_MAX_TOKENS: ответы короткие (~150 токенов) — меньший лимит снижает резерв на шлюзе
        # и позволяет держать короткий контекст на локальном сервере (prompt + max_tokens <= окна).
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS") or max_tokens)
        # Шлюз иногда держит соединение без ответа: короткий таймаут из LLM_TIMEOUT быстрее отпускает поток.
        self.timeout = float(os.environ.get("LLM_TIMEOUT") or timeout)
        self.max_retries = int(os.environ.get("LLM_MAX_RETRIES") or max_retries)
        self.effort = effort
        # LLM_EXTRA_BODY — JSON с доп. полями запроса, например для vLLM с моделями на базе Qwen3:
        # {"chat_template_kwargs": {"enable_thinking": false}} — выключить рассуждения.
        self.extra_body = {**json.loads(os.environ.get("LLM_EXTRA_BODY") or "{}"), **(extra_body or {})}
        if self.backend == "openai":
            self.base_url = os.environ.get("LLM_BASE_URL", "").rstrip("/")
            self.api_key = os.environ.get("LLM_API_KEY", "")
            # Несколько ключей (LLM_API_KEY, LLM_API_KEY_2, LLM_API_KEY_EGOR…): запросы по очереди,
            # при отказе — другой ключ.
            extra = [os.environ[n] for n in sorted(os.environ) if n.startswith("LLM_API_KEY_") and os.environ[n]]
            self.api_keys = list(dict.fromkeys(k for k in [self.api_key] + extra if k))
            self._dead = set()
            self._cool = {}  # (модель, ключ) -> время, до которого не трогаем (по retry-after шлюза)
            self._rr = 0
            self._rr_lock = threading.Lock()
            self.model = model or os.environ.get("LLM_MODEL", "")
            if not self.base_url or not self.model:
                raise LLMError("для backend=openai нужны LLM_BASE_URL и LLM_MODEL")
            if not self.api_key:
                raise LLMError("LLM_API_KEY пуст: шлюз отвечает 401 без ключа")
        elif self.backend == "anthropic":
            import anthropic  # noqa: локальный импорт — SDK нужен только этому бэкенду
            self._anthropic = anthropic
            self.client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
            self.model = model or "claude-opus-5-5"
        else:
            raise LLMError(f"неизвестный backend {self.backend}")

    def complete(self, system: str, user: str) -> str:
        if self.backend == "openai":
            return self._openai(system, user)
        return self._anthropic_call(system, user)

    # --- OpenAI-совместимый шлюз ---------------------------------------------------------
    def _openai(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **self.extra_body,
        }
        # Шлюз временами отдаёт 503 «модель недоступна» по очереди на разные модели: при 503/429 сразу
        # пробуем следующую из LLM_FALLBACK_MODELS и ждём, только если недоступны все.
        models = [self.model] + [m.strip() for m in os.environ.get("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
        with self._rr_lock:
            self._rr += 1
            start = self._rr
        last = None
        for attempt in range(self.max_retries + 1):
            delay = 0.0
            sent = False
            for model in models:
                keys = [k for k in self.api_keys if k not in self._dead]
                if not keys:
                    raise last or LLMError("HTTP 401: нет рабочих ключей")
                keys = keys[start % len(keys):] + keys[:start % len(keys)]
                for key in keys:
                    # Пара «модель + ключ», ответившая 503/429, отдыхает столько, сколько просил шлюз, —
                    # во всех потоках: иначе мы долбим заведомо недоступную модель и сами усиливаем отказы.
                    if self._cool.get((model, key), 0) > time.time():
                        continue
                    sent = True
                    data = json.dumps({**body, "model": model}, ensure_ascii=False).encode("utf-8")
                    req = urllib.request.Request(
                        self.base_url + "/chat/completions", data=data, method="POST",
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                 # Cloudflare перед шлюзом отвечает 403 (код 1010) на UA «Python-urllib».
                                 "User-Agent": os.environ.get("LLM_USER_AGENT", "gigachess-data/1.0")},
                    )
                    try:
                        payload = _post_with_deadline(req, self.timeout)
                        self._tl.model = model
                        return _openai_text(payload)
                    except urllib.error.HTTPError as e:
                        detail = e.read().decode("utf-8", "replace")[:500]
                        last = LLMError(f"HTTP {e.code}: {detail}")
                        if e.code in (401, 402, 403):
                            # Ключ без баланса или доступа: выключаем его, остальные продолжают.
                            self._dead.add(key)
                            if len(self._dead) >= len(self.api_keys):
                                raise last
                            continue
                        if e.code not in (408, 409, 429) and e.code < 500:
                            raise last  # 4xx кроме перечисленных не ретраим
                        wait = min(max(float(e.headers.get("retry-after") or 0), 5.0), 120.0)
                        self._cool[(model, key)] = time.time() + wait
                        delay = max(delay, min(wait, 60.0))
                    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                        last = LLMError(f"сеть: {e}")
            if not sent:
                # Все пары на отдыхе: ждём, пока освободится ближайшая, а не шлём запросы впустую.
                soonest = min(self._cool.values(), default=time.time()) - time.time()
                delay = max(delay, min(max(soonest, 1.0), 60.0))
                last = last or LLMError("HTTP 503: все модели и ключи временно недоступны")
            time.sleep(delay or _backoff(attempt))
        raise last

    def used_model(self) -> str:
        """Модель, которая ответила на последний запрос этого потока (с учётом запасных)."""
        return getattr(self._tl, "model", self.model)

    # --- Anthropic SDK ----------------------------------------------------------------------
    def _anthropic_call(self, system: str, user: str) -> str:
        a = self._anthropic
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            output_config={"effort": self.effort},  # синтаксическая склейка — глубокое рассуждение не нужно
            messages=[{"role": "user", "content": user}],
        )
        try:
            if self.model in ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"):
                # Серверный fallback при отказе классификатора: запрос продолжит другая модель.
                resp = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except a.BadRequestError as e:
            raise LLMError(f"400: {e.message}") from e
        except a.RateLimitError as e:
            raise LLMError("429 после ретраев SDK") from e
        except a.APIStatusError as e:
            raise LLMError(f"HTTP {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"сеть: {e}") from e
        if resp.stop_reason == "refusal":
            raise LLMError("refusal")
        if resp.stop_reason == "max_tokens":
            raise LLMError("max_tokens: ответ обрезан")
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def _post_with_deadline(req: urllib.request.Request, deadline: float) -> dict:
    """Запрос с общим сроком. Таймаут urlopen — на каждое чтение сокета: если шлюз держит соединение
    и изредка шлёт байты, запрос висит бесконечно. Здесь запрос идёт в отдельном потоке, а мы ждём
    его не дольше deadline секунд; зависший поток бросаем (он daemon и умрёт вместе с процессом)."""
    box: dict = {}

    def run():
        try:
            with urllib.request.urlopen(req, timeout=deadline) as r:
                box["payload"] = json.loads(r.read().decode("utf-8"))
        except BaseException as e:  # noqa: BLE001 — передаём наружу как есть
            box["error"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        raise TimeoutError(f"нет ответа за {deadline:.0f} с")
    if "error" in box:
        raise box["error"]
    return box["payload"]


def _openai_text(payload: dict) -> str:
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError):
        raise LLMError(f"неожиданный ответ шлюза: {json.dumps(payload, ensure_ascii=False)[:300]}")
    if choice.get("finish_reason") == "length":
        raise LLMError("max_tokens: ответ обрезан")
    msg = choice.get("message", {})
    content = msg.get("content")
    if isinstance(content, list):  # некоторые шлюзы отдают список блоков
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    if not content:
        # Ollama с моделями на базе Qwen3 иногда кладёт весь ответ в поле рассуждений.
        content = msg.get("reasoning") or msg.get("reasoning_content")
    if not content:
        raise LLMError("пустой ответ")
    return content.strip()


def _backoff(attempt: int) -> float:
    return min(60.0, 2 ** attempt + random.uniform(0, 1))


def extract_json(text: str) -> dict:
    """Первый JSON-объект в ответе (модель может обернуть его в ```json)."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("в ответе нет JSON")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"битый JSON: {e}") from e
