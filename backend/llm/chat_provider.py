"""Chat-only compatible provider with primary inference fallback."""
import requests
from backend.llm.gigachess import GigachessError


class ChatProvider:
    def __init__(self, config, primary):
        self.config = config
        self.primary = primary

    def _fallback(self, messages, **options):
        try:
            response = requests.post(
                self.config.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + self.config.api_key.get_secret_value()},
                json={"model": self.config.model, "messages": messages, **options},
                timeout=(self.config.connect_timeout, self.config.read_timeout),
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty chat response")
            return content
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            raise GigachessError("Chat provider unavailable") from exc

    def chat(self, messages, **options):
        configured = bool(self.config.base_url and self.config.api_key.get_secret_value())
        providers = []
        if configured and self.config.prefer:
            providers.append(self._fallback)
        if self.primary is not None:
            providers.append(self.primary.chat)
        if configured and not self.config.prefer:
            providers.append(self._fallback)
        for provider in providers:
            try:
                return provider(messages, **options)
            except GigachessError:
                continue
        raise GigachessError("All chat providers unavailable")
