"""Тесты перевода названий дебютов на русский язык (openings_ru)."""

import json
import os

import pytest

from backend.knowledge.openings_ru import (
    COMPONENTS,
    EONYMS,
    FAMILIES,
    _agree_adjective,
    _genitive,
    _translate_component,
    translate_opening_name,
)

OPENINGS_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "backend",
        "knowledge",
        "openings.json",
    )
)


# --- translate_opening_name: базовые случаи ---

class TestTranslateBasic:
    def test_none_and_empty(self):
        assert translate_opening_name(None) is None
        assert translate_opening_name("") is None
        assert translate_opening_name("   ") is None

    def test_family_only(self):
        assert translate_opening_name("Sicilian Defense") == "Сицилианская защита"
        assert translate_opening_name("Ruy Lopez") == "Испанская партия"
        assert translate_opening_name("Queen's Gambit Declined") == "Ферзевый гамбит (отклонённый)"

    def test_family_plus_variations(self):
        result = translate_opening_name(
            "Sicilian Defense: Najdorf Variation, English Attack"
        )
        assert result == "Сицилианская защита: Вариант Найдорфа, Английская атака"

    def test_accepted_and_declined(self):
        assert translate_opening_name("King's Gambit Accepted") == "Королевский гамбит (принятый)"
        assert "отклонённый" in translate_opening_name("Queen's Gambit Declined")

    def test_with_move_keeps_lowercase(self):
        assert translate_opening_name("Queen's Indian Defense, with e3") == "Ферзево-индийская защита, с e3"

    def test_move_pieces_are_russian(self):
        assert translate_opening_name("King's Indian Attack, with Bf5") == "Индийское нападение, с Сf5"

    def test_black_move_keeps_dots(self):
        result = translate_opening_name("Sicilian Defense: Kan Variation, with .. d6")
        assert result.endswith(".. d6")

    def test_unknown_component_returns_none(self):
        assert translate_opening_name("Sicilian Defense: Absolutely Made Up") is None


# --- Согласование и падежи ---

class TestGrammar:
    def test_adjective_gender(self):
        assert _agree_adjective("Классический", "вариант") == "Классический"
        assert _agree_adjective("Классический", "защита") == "Классическая"
        assert _agree_adjective("Классический", "начало") == "Классическое"

    def test_adjective_soft_sign(self):
        assert _agree_adjective("Слоновий", "защита") == "Слоновья"
        assert _agree_adjective("Слоновий", "начало") == "Слоновье"

    def test_genitive_consonant(self):
        assert _genitive("Найдорф") == "Найдорфа"
        assert _genitive("Чигорин") == "Чигорина"

    def test_genitive_indeclinable(self):
        assert _genitive("Мароци") == "Мароци"
        assert _genitive("О'Келли") == "О'Келли"

    def test_genitive_adjective(self):
        assert _genitive("Берлинский") == "Берлинского"


# --- Словари ---

class TestDictionaries:
    def test_every_family_is_covered_by_database(self):
        """Каждая семья из openings.json обязана быть в словаре FAMILIES."""
        with open(OPENINGS_PATH, encoding="utf-8") as f:
            names = {o["name"] for o in json.load(f)["openings"]}

        families_in_db = {n.split(":")[0].split(",")[0].strip() for n in names}
        missing = families_in_db - set(FAMILIES)
        assert not missing, f"Семьи без перевода: {sorted(missing)}"

    def test_no_transliteration_typos(self):
        """Гарантия от опечаток вида «Сveshникова» из транслитерации."""
        for name, value in FAMILIES.items():
            assert "vesh" not in value, name
        for name, value in COMPONENTS.items():
            assert "vesh" not in value, name
        for name, value in EONYMS.items():
            assert "vesh" not in value, name

    def test_eponyms_are_genitive(self):
        """Имена игроков хранятся в родительном падеже."""
        assert EONYMS["Najdorf"] == "Найдорфа"
        assert EONYMS["Winawer"] == "Винавера"


# --- Полное покрытие реальной базы ---

class TestFullDatabaseCoverage:
    def test_every_opening_translates(self):
        """Все 3810 названий из openings.json переводятся на русский."""
        with open(OPENINGS_PATH, encoding="utf-8") as f:
            names = [o["name"] for o in json.load(f)["openings"]]

        failed = [n for n in names if translate_opening_name(n) is None]
        assert not failed, f"Не переведено {len(failed)}: {failed[:10]}"

    def test_component_always_resolves(self):
        with open(OPENINGS_PATH, encoding="utf-8") as f:
            names = [o["name"] for o in json.load(f)["openings"]]

        import re
        split_re = re.compile(r"(:\s*|,\s*)")
        failed = set()
        for name in names:
            for token in split_re.split(name.strip()):
                if not token or split_re.fullmatch(token):
                    continue
                if _translate_component(token) is None:
                    failed.add(token)
        assert not failed, f"Компоненты без перевода: {sorted(failed)}"

    def test_translation_is_cyrillic(self):
        result = translate_opening_name("Sicilian Defense: Najdorf Variation")
        assert result is not None
        assert any("Ѐ" <= ch <= "ӿ" for ch in result)


# --- Интеграция: name_ru появляется при загрузке базы ---

class TestNameRuAttachment:
    def test_opening_gets_name_ru(self):
        from backend.api_gateway.state import _attach_russian_names

        openings = [{"name": "Sicilian Defense"}, {"name": "Italian Game"}]
        translated = _attach_russian_names(openings)

        assert translated == 2
        assert openings[0]["name_ru"] == "Сицилианская защита"
        assert openings[1]["name_ru"] == "Итальянская партия"

    def test_untranslated_opening_has_no_field(self):
        from backend.api_gateway.state import _attach_russian_names

        openings = [{"name": "Not A Real Opening At All"}]
        translated = _attach_russian_names(openings)

        assert translated == 0
        assert "name_ru" not in openings[0]
