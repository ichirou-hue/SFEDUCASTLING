import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ml/pipelines/gigachess_data"))

from gcdata.checks import check_polished  # noqa: E402
from gcdata.facts import all_facts  # noqa: E402
from gcdata.template import build_question  # noqa: E402

DRAFT = """1. Замысел: белые; белый конь f3 -> e5; мотив: взятие: чёрная пешка e5
2. Геометрия: освобождено поле f3; занято поле e5; новые нападения: чёрная пешка f7; поле e5 атакуют: чёрный конь c6
3. Оценка: глиф ?; вердикт: ошибка; потеря 180 сантипешек; оценка после хода: 1.20 в пользу чёрных; сильнее было: белый слон c4 -> b5
4. Ответ соперника: сильнейший ответ: чёрный конь c6 -> e5; взятие: белый конь e5; следствие: за 4 полухода белые теряют 2 ед. материала"""

# Описание хода — один абзац (план старого формата тут лишь вход для проверок клеток и чисел).
GOOD = ("Белые берут конём с f3 на e5 чёрную пешку e5. "
        "Поле f3 освобождается, конь занимает e5 и нападает на чёрную пешку f7, однако поле e5 атакует чёрный конь c6. "
        "Это ошибка: потеря составляет 180 сантипешек, оценка после хода — 1.20 в пользу чёрных. Сильнее было пойти белым слоном с c4 на b5. "
        "Сильнейший ответ — чёрный конь c6 бьёт белого коня на e5; за 4 полухода белые теряют 2 ед. материала.")


class TestFacts(unittest.TestCase):
    def test_en_passant(self):
        f = all_facts("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1", "d5e6")
        self.assertTrue(f["delta"]["is_en_passant"])
        self.assertEqual(f["delta"]["captured"], {"square": "e5", "piece": "p"})

    def test_castling_and_rights(self):
        f = all_facts("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1")
        self.assertEqual(f["delta"]["castling"], "O-O")
        self.assertEqual(f["delta"]["castling_rights_after"], "kq")
        self.assertIn("короткая рокировка", build_question(f["delta"], None))

    def test_promotion_with_check(self):
        f = all_facts("4k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7b8q")
        self.assertEqual(f["delta"]["promotion"], "Q")
        self.assertTrue(f["delta"]["gives_check"])

    def test_discovered_attack_and_pin(self):
        # Слон уходит с d2 и вскрывает ладью d1 на ферзя d8.
        f = all_facts("3qk3/8/8/8/8/8/3B4/3RK3 w - - 0 1", "d2g5")
        disc = f["delta"]["discovered_attacks"]
        self.assertEqual((disc[0]["slider_square"], disc[0]["square"]), ("d1", "d8"))
        self.assertIn({"square": "d8", "piece": "q"}, f["delta"]["new_attacks"])  # g5 тоже бьёт d8

    def test_new_pin(self):
        f = all_facts("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1", "f1b5")
        self.assertEqual(f["delta"]["new_pins_on_opponent"][0]["square"], "c6")

    def test_illegal_move_rejected(self):
        with self.assertRaises(ValueError):
            all_facts("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "e1e3")


class TestPolishChecks(unittest.TestCase):
    def run_check(self, text):
        return check_polished(DRAFT, text, "ошибка", 180, 2, 700)

    def test_good_passes(self):
        self.assertEqual(self.run_check(GOOD), [])

    def test_hallucinated_square(self):
        errs = self.run_check(GOOD.replace("пешку f7", "пешку f7 и слона g7"))
        self.assertTrue(any("новые клетки" in e for e in errs))

    def test_dropped_fact_and_number(self):
        errs = self.run_check(GOOD.replace("потеря составляет 180 сантипешек, ", ""))
        self.assertTrue(any("потеряны числа" in e for e in errs))

    def test_san_first_person_cliche_praise(self):
        bad = GOOD.replace("Белые берут конём с f3 на e5", "Мы видим Nxe5, и, стоит отметить, это отличный ход")
        errs = " | ".join(self.run_check(bad))
        for part in ("нотация", "первое лицо", "штампы", "положительный тон"):
            self.assertIn(part, errs)

    def test_paragraphs(self):
        errs = self.run_check(GOOD.replace(" Сильнее было", "\n\nСильнее было"))
        self.assertTrue(any("абзацев" in e for e in errs))


class TestIntentRules(unittest.TestCase):
    def test_keeps_intent_drops_verdicts_and_junk(self):
        from s2_intent import extract_intent
        self.assertEqual(extract_intent("More development, preparing to castle.")[0], "More development, preparing to castle.")
        self.assertEqual(extract_intent("I am not sure, this one is particularly bad. So i attack the f6 pawn.")[0],
                         "So i attack the f6 pawn.")
        self.assertIsNone(extract_intent("9. Nd5 is worth looking. also 9. Na4.")[0])
        self.assertIsNone(extract_intent("Please click the little heart to show appreciation")[0])
        self.assertIsNone(extract_intent("1-0 White wins.")[0])

    def test_intent_squares_allowed_not_required(self):
        draft = DRAFT.replace("мотив: взятие: чёрная пешка e5", "замысел игрока (цитата): «more pressure on f3 and Qh5»")
        text = GOOD.replace("Белые берут конём с f3 на e5 чёрную пешку e5.",
                            "Белые идут конём с f3 на e5; по замыслу игрока, чтобы усилить давление на f3 и сыграть ферзём на h5.")
        self.assertEqual(check_polished(draft, text, "ошибка", 180, 2, 700, intent="more pressure on f3 and Qh5"), [])


class TestConceptsAndTerms(unittest.TestCase):
    PLAN = ("1. Ход: белые; белый конь g5 -> f7 со взятием: чёрная пешка f7; [вилка] белый конь f7 одновременно бьёт "
            "чёрный ферзь d8 и чёрная ладья h8\n2. Оценка: вердикт: точный ход")

    def test_term_from_plan_required_and_extra_forbidden(self):
        from gcdata.checks import check_terms
        ok = "Белый конь идёт с g5 на f7 и берёт пешку f7, ставя вилку чёрному ферзю d8 и ладье h8."
        self.assertEqual(check_terms(self.PLAN, ok), [])
        no_term = "Белый конь идёт с g5 на f7 и нападает на ферзя d8 и ладью h8."
        self.assertTrue(any("не названы" in e for e in check_terms(self.PLAN, no_term)))
        extra = ok + " Это красивая жертва."
        self.assertTrue(any("нет в черновике" in e for e in check_terms(self.PLAN, extra)))
        label = ok + " Причина: шах."
        self.assertTrue(any("служебная метка" in e for e in check_terms(self.PLAN, label)))

    def test_bindings_color_piece_and_direction(self):
        from gcdata.checks import check_bindings
        ok = "Конь c7 создаёт вилку: он бьёт чёрного короля e6 и чёрную ладью a8. Белый конь идёт с d5 на c7."
        plan = "1. Ход: белый конь d5 -> c7; [вилка] белый конь c7 бьёт чёрный король e6 и чёрная ладья a8"
        self.assertEqual(check_bindings(plan, ok), [])
        self.assertTrue(check_bindings(plan, ok.replace("чёрного короля e6", "белого короля e6")))
        self.assertTrue(check_bindings(plan, ok.replace("с d5 на c7", "с c7 на d5")))

    def test_outpost(self):
        import chess
        from gcdata.positional import is_outpost
        b = chess.Board("4k3/8/2p5/3N4/4P3/8/8/4K3 w - - 0 1")
        self.assertFalse(is_outpost(b, chess.D5))  # пешка c6 бьёт d5 — не форпост
        b = chess.Board("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1")
        self.assertTrue(is_outpost(b, chess.D5))

    def test_overloaded(self):
        import chess
        from gcdata.motifs import overloaded
        # Чёрный ферзь d7 — единственный защитник коня c6 и слона e6; обе фигуры атакованы.
        b = chess.Board("4k3/3q4/2n1b3/1B6/8/4R3/8/4K3 w - - 0 1")
        self.assertEqual(overloaded(b, chess.BLACK)[0]["square"], "d7")


if __name__ == "__main__":
    unittest.main(verbosity=2)
