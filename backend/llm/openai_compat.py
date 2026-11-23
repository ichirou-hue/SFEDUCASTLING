"""Клиент [OI]-совместимого Chat Completions API.

Используется как резервный провайдер чата, пока основная модель
(Gigachess) недоступна. Формат запроса и ответа совпадает с
``POST {base_url}/chat/completions``.
"""

import json
import time

import requests


class OpenAICompatError(RuntimeError):
    """Ошибка обращения к [OI]-совместимому API."""


class OpenAICompatClient:
    """HTTP-клиент [OI]-совместимого Chat Completions API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        connect_timeout: float = 15.0,
        read_timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = (connect_timeout, read_timeout)

    @property
    def completions_url(self) -> str:
        """URL endpoint'а Chat Completions."""

        return f"{self.base_url}/chat/completions"

    def chat(
            self,
            messages: list[dict],
            *,
            temperature: float = 0.2,
            top_p: float = 0.95,
            max_tokens: int = 700,
    ) -> str:
        """Отправляет сообщения и возвращает текст ответа модели."""

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = self.completions_url

        print("[LLMFallback] POST:", url)
        print("[LLMFallback] model:", self.model)
        print("[LLMFallback] messages:", len(messages))
        print("[LLMFallback] max_tokens:", max_tokens)
        print("[LLMFallback] temperature:", temperature)

        started_at = time.perf_counter()

        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - started_at

            print(
                "[LLMFallback] REQUEST ERROR after:",
                f"{elapsed:.2f}s",
                repr(exc),
            )

            raise OpenAICompatError(
                f"Ошибка запроса к резервной LLM: {exc}"
            ) from exc

        elapsed = time.perf_counter() - started_at

        print(
            "[LLMFallback] HTTP status:",
            response.status_code,
            f"({elapsed:.2f}s)",
        )

        if response.status_code >= 400:
            raise OpenAICompatError(
                "Резервная LLM вернула ошибку "
                f"{response.status_code}: "
                f"{response.text[:2000]}"
            )

        try:
            # content декодируем как UTF-8: сервер может не указывать
            # charset, и тогда response.text рискует испортить кириллицу.
            data = json.loads(response.content.decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
        except (
                ValueError,
                KeyError,
                IndexError,
                TypeError,
        ) as exc:
            raise OpenAICompatError(
                "Некорректный ответ резервной LLM: "
                f"{response.text[:2000]}"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise OpenAICompatError(
                "Резервная LLM вернула пустой ответ"
            )

        return content.strip()
