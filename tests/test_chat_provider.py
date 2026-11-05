from unittest.mock import Mock, patch
import pytest
from backend.config.settings import ChatFallbackSettings
from backend.llm.chat_provider import ChatProvider
from backend.llm.gigachess import GigachessError


def config(**kwargs):
    return ChatFallbackSettings(base_url="https://provider.example/v1", api_key="test-only", model="test", **kwargs)


def test_preferred_provider():
    primary = Mock()
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "Reply"}}]}
    with patch("backend.llm.chat_provider.requests.post", return_value=response) as post:
        assert ChatProvider(config(), primary).chat([], max_tokens=20) == "Reply"
    assert post.call_args.args[0] == "https://provider.example/v1/chat/completions"
    assert post.call_args.kwargs["json"]["model"] == "test"
    primary.chat.assert_not_called()


def test_bad_response_uses_primary():
    primary = Mock()
    primary.chat.return_value = "Primary"
    response = Mock()
    response.json.return_value = {"choices": []}
    with patch("backend.llm.chat_provider.requests.post", return_value=response):
        assert ChatProvider(config(), primary).chat([]) == "Primary"


def test_missing_key_uses_primary_without_http():
    primary = Mock()
    primary.chat.return_value = "Primary"
    with patch("backend.llm.chat_provider.requests.post") as post:
        assert ChatProvider(ChatFallbackSettings(), primary).chat([]) == "Primary"
        post.assert_not_called()


def test_nonpreferred_provider_on_primary_failure():
    primary = Mock()
    primary.chat.side_effect = GigachessError("Unavailable")
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "Backup"}}]}
    with patch("backend.llm.chat_provider.requests.post", return_value=response):
        assert ChatProvider(config(prefer=False), primary).chat([]) == "Backup"


def test_no_providers():
    with pytest.raises(GigachessError, match="All chat providers unavailable"):
        ChatProvider(ChatFallbackSettings(), None).chat([])
