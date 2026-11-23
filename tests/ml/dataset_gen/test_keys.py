"""Ключи ядра v1 на позициях, собранных вручную."""
import sys
import unittest
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "ml/dataset_gen/gigachess_data"))

from gcdata import keys  # noqa: E402
from gcdata.keys import Line  # noqa: E402


def line(fen, *ucis, pre=None):
    return Line.build(chess.Board(fen), [chess.Move.from_uci(u) for u in ucis],
                      pre=chess.Move.from_uci(pre) if pre else None)


class KeysTest(unittest.TestCase):
    def test_knight_fork(self):
        # Кc7+ бьёт короля e8 и ладью a8, затем берёт ладью.
        L = line("r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1", "d5c7", "e8d7", "c7a8")
        self.assertTrue(keys.fork(L))

    def test_double_and_discovered_check(self):
        # Сb5+: шахуют и слон, и ладья e1 — двойной шах. Как у lichess, вскрытым он не считается:
        # пошедшая фигура сама среди шахующих.
        L = line("4k3/8/8/8/8/8/4B3/4R1K1 w - - 0 1", "e2b5")
        self.assertTrue(keys.double_check(L))
        self.assertFalse(keys.discovered_check(L))
        # Сd3: шахует только ладья e1 — вскрытый шах.
        L = line("4k3/8/8/8/8/8/4B3/4R1K1 w - - 0 1", "e2d3")
        self.assertTrue(keys.discovered_check(L))
        self.assertFalse(keys.double_check(L))

    def test_smothered_mate(self):
        L = line("6rk/6pp/8/6N1/8/8/8/6K1 w - - 0 1", "g5f7")
        self.assertTrue(L.final().is_checkmate())
        self.assertTrue(keys.smothered_mate(L))
        self.assertIn("smotheredMate", keys.tags(L))

    def test_back_rank_mate(self):
        L = line("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8")
        self.assertTrue(keys.back_rank_mate(L))
        self.assertEqual({"mate", "mateIn1", "backRankMate"} & keys.tags(L), {"mate", "mateIn1", "backRankMate"})

    def test_pin_prevents_escape(self):
        # Сb5 связывает незащищённого коня c6 с королём e8.
        L = line("4k3/8/2n5/8/8/8/8/4KB2 w - - 0 1", "f1b5")
        self.assertTrue(keys.pin(L))

    def test_quiet_position_has_no_tactics(self):
        L = line("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "e1e2", "e8e7", "e2e3")
        t = keys.tags(L)
        for theme in ("fork", "pin", "skewer", "sacrifice", "doubleCheck", "mate", "discoveredAttack"):
            self.assertNotIn(theme, t)

    def test_hanging_with_xray_defence(self):
        # Ладья e5 защищена сквозь белого ферзя e4 ладьёй e8: is_defended учитывает сквозную защиту.
        b = chess.Board("4r1k1/8/8/4r3/4Q3/8/8/6K1 b - - 0 1")
        self.assertTrue(keys.is_defended(b, b.piece_at(chess.E5), chess.E5))
        b2 = chess.Board("6k1/8/8/4r3/4Q3/8/8/6K1 b - - 0 1")
        self.assertTrue(keys.is_hanging(b2, b2.piece_at(chess.E5), chess.E5))


if __name__ == "__main__":
    unittest.main()
