import time

import requests


class GigachessError(RuntimeError):
    """Ошибка обращения к Gigachess API."""


class GigachessClient:
    """HTTP-клиент внешнего Gigachess inference API."""

    def __init__(
        self,
        base_url: str,
        model: str = "gigachess",
        connect_timeout: float = 15.0,
        read_timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = (connect_timeout, read_timeout)

    def chat(
            self,
            messages: list[dict],
            *,
            temperature: float = 0.2,
            top_p: float = 0.95,
            max_tokens: int = 700,
    ) -> str:

        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "repetition_penalty": 1.0,
            "model": self.model,
        }

        url = f"{self.base_url}/chat"

        print("[Gigachess] POST:", url)
        print("[Gigachess] model:", self.model)
        print("[Gigachess] messages:", len(messages))
        print("[Gigachess] timeout:", self.timeout)
        print("[Gigachess] max_tokens:", max_tokens)
        print("[Gigachess] temperature:", temperature)
        print("[Gigachess] top_p:", top_p)

        for i, message in enumerate(messages):
            content = message.get("content", "")

            print(
                f"[Gigachess] message[{i}] "
                f"role={message.get('role')} "
                f"chars={len(content)}"
            )

        payload_size = len(
            str(payload).encode("utf-8")
        )

        print(
            "[Gigachess] payload size:",
            payload_size,
            "bytes",
        )

        started_at = time.perf_counter()

        try:
            print("[Gigachess] Sending request...")

            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )

            elapsed = time.perf_counter() - started_at

            print(
                "[Gigachess] HTTP status:",
                response.status_code,
            )

            print(
                "[Gigachess] request completed in:",
                f"{elapsed:.2f}s",
            )

            print(
                "[Gigachess] response:",
                response.text[:2000],
            )

            response.raise_for_status()

        except requests.RequestException as exc:

            elapsed = time.perf_counter() - started_at

            print(
                "[Gigachess] REQUEST ERROR after:",
                f"{elapsed:.2f}s",
            )

            print(
                "[Gigachess] REQUEST ERROR:",
                repr(exc),
            )

            raise GigachessError(
                f"Ошибка запроса к Gigachess: {exc}"
            ) from exc

        try:
            data = response.json()

            content = (
                data["choices"][0]["message"]["content"]
            )

        except (
                ValueError,
                KeyError,
                IndexError,
                TypeError,
        ) as exc:

            raise GigachessError(
                "Некорректный ответ Gigachess: "
                f"{response.text[:2000]}"
            ) from exc

        if (
                not isinstance(content, str)
                or not content.strip()
        ):
            raise GigachessError(
                "Gigachess вернул пустой ответ"
            )

        return content.strip()