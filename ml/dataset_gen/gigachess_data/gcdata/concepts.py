"""Словарь шахматных терминов: несколько несвязных фактов -> один термин с обоснованием.

Каждое правило срабатывает только при выполнении условия на уже посчитанных фактах
(python-chess + Stockfish). Результат — Concept:
  term      — термин, который LLM обязана использовать как есть;
  role      — к какой части рассказа относится: move (что делает ход), cons (последствия),
              reason (почему ход плох), better (что было лучше);
  weight    — важность для выбора главной истории;
  evidence  — 1–2 факта-обоснования, которые остаются в тексте вместе с термином.
Термины, которых нет в черновике, валидатор запрещает в ответе (TERM_PATTERNS).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import chess

from . import keys as K
from .ru import piece_sym_ru, side_gen_ru, side_ru

VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}

# Регулярки терминов для проверки «LLM назвала термин, которого не было в черновике».
TERM_PATTERNS = {
    "вилка": r"вилк|двойн\w* удар",
    "связка": r"связк|связыва",
    "вскрытое нападение": r"вскрыт\w* (нападен|атак)",
    "вскрытый шах": r"вскрыт\w* шах",
    "двойной шах": r"двойн\w* шах",
    "сквозной удар": r"сквозн",
    "отвлечение": r"отвле[чк]",
    "завлечение": r"завле[чк]|заманив",
    "рентген": r"рентген",
    "уничтожение защиты": r"уничтож\w* защит",
    "перекрытие": r"перекры",
    "освобождение линии": r"освобожд\w* (лини|пол|диагонал|вертикал|горизонтал)",
    "промежуточный ход": r"промежуточн",
    "ловля фигуры": r"ловл|поймал|поймана|пойман",
    "тихий ход": r"тих\w* ход",
    "оголённый король": r"огол[её]нн?\w* корол",
    "мат по последней горизонтали": r"(?<!пред)последн\w* горизонтал\w*.{0,20}мат|мат\w*.{0,30}(?<!пред)последн\w* горизонтал",
    "спёртый мат": r"сп[её]рт",
    "мат Анастасии": r"анастаси",
    "арабский мат": r"арабск",
    "мат «крюк»": r"крюк",
    "мат Бодена": r"боден",
    "мат двумя слонами": r"мат\w* двумя слонами",
    "мат «ласточкин хвост»": r"ласточкин",
    "жертва": r"жертв",
    "размен": r"размен",
    "выигрыш темпа": r"\bтемп",
    "профилактика": r"профилакти",
    "перегрузка": r"перегру",
    "форпост": r"форпост",
    "проходная пешка": r"проходн",
    "изолированная пешка": r"изолир",
    "сдвоенные пешки": r"сдвоен",
    "пара слонов": r"пар\w* слонов",
    "открытая вертикаль": r"открыт\w* (вертикал|лини)",
    "седьмая горизонталь": r"седьм|предпоследн\w* горизонтал",
    "активизация фигуры": r"активиз",
    "ослабление позиции короля": r"ослабл\w* (позици\w* |прикрыти\w* |укрыти\w* )?корол",
    "давление на короля": r"давлени\w* на корол|атак\w* на корол",
    "атака на короля": r"давлени\w* на корол|атак\w* на корол",
    "выигрыш материала": r"выигрыш\w* материал",
    "упущенная возможность": r"упущен\w* возможност",
}


@dataclass
class Concept:
    term: str
    role: str
    weight: int
    evidence: list = field(default_factory=list)


def _ps(d: dict) -> str:
    return f"{piece_sym_ru(d['piece'])} {d['square']}"


def _val(sym: str) -> int:
    return VAL[sym.lower()]


def _name(board: chess.Board, sq: int) -> str:
    p = board.piece_at(sq)
    return f"{piece_sym_ru(p.symbol())} {chess.square_name(sq)}" if p else chess.square_name(sq)


def _mate_geometry(pattern: str, b: chess.Board, pov: bool, last: chess.Move) -> str:
    """Факты матовой картины для плана (то, что проверяет ключ)."""
    ksq = b.king(not pov)
    king = _name(b, ksq)
    if pattern == "спёртый мат":
        return f"все соседние с {king} поля заняты фигурами того же цвета"
    if pattern == "мат по последней горизонтали":
        return f"{king} на своей последней горизонтали, поля перед ним заняты его фигурами"
    if pattern in ("мат Бодена", "мат двумя слонами"):
        bs = " и ".join(_name(b, s) for s in b.pieces(chess.BISHOP, pov))
        side = "с разных сторон от короля" if pattern == "мат Бодена" else "с одной стороны от короля"
        return f"все поля вокруг {king} бьют только слоны: {bs}, {side}"
    if pattern == "арабский мат":
        kn = [s for s in b.attackers(pov, last.to_square) if b.piece_type_at(s) == chess.KNIGHT]
        return f"{king} в углу, ладью защищает " + (_name(b, kn[0]) if kn else "конь")
    if pattern == "мат «крюк»":
        return f"ладья стоит вплотную к {king}, её защищает конь, коня защищает пешка"
    if pattern == "мат Анастасии":
        return f"{king} на крайней вертикали, конь отнимает поля отхода"
    if pattern == "мат «ласточкин хвост»":
        return f"ферзь вплотную по диагонали к {king}, поля отхода заняты его фигурами"
    return ""


def _when(k: int) -> str:
    return "сыгранным ходом" if k == 1 else "в варианте ответа"


def _key_concepts(L, facts: dict, bad: bool, net, threat_target) -> list[Concept]:
    """Термины по ключам ядра v1 (docs/term_keys.md, Часть 0) для стороны, сделавшей ход."""
    after = facts["after"]
    eng = facts["engine"]
    out: list[Concept] = []
    mates_now = after["is_checkmate"]

    def add(term, weight, ev, role="move"):
        out.append(Concept(term, role, weight, [ev]))

    # Шахи — факты сыгранного хода, термин уместен и при ошибке.
    dch = K.double_check(L)
    if dch and dch["k"] == 1:
        add("двойной шах", 85, "шах дают сразу " + " и ".join(_name(L.after(1), s) for s in dch["checkers"]))
    else:
        dc = K.discovered_check(L)
        if dc and dc["k"] == 1 and not L.before(1).is_castling(L.moves[1]):
            add("вскрытый шах", 80, f"фигура уходит с {chess.square_name(L.moves[1].from_square)}, и шах даёт "
                + " и ".join(_name(L.after(1), s) for s in dc["checkers"]))

    # Матовая картина — если ход ставит мат или вариант без ошибки ведёт к мату.
    fin = L.final()
    if fin.is_checkmate() and fin.turn != L.pov and (mates_now or not bad):
        pattern = None
        if K.smothered_mate(L):
            pattern = "спёртый мат"
        elif K.back_rank_mate(L):
            pattern = "мат по последней горизонтали"
        elif K.anastasia_mate(L):
            pattern = "мат Анастасии"
        elif K.hook_mate(L):
            pattern = "мат «крюк»"
        elif K.arabian_mate(L):
            pattern = "арабский мат"
        else:
            bd = K.boden_or_double_bishop(L)
            if bd:
                pattern = "мат Бодена" if bd == "bodenMate" else "мат двумя слонами"
            elif K.dovetail_mate(L):
                pattern = "мат «ласточкин хвост»"
        if pattern:
            last = L.moves[-1]
            where = "" if mates_now else " в конце варианта"
            add(pattern, 90, f"мат ставит {_name(fin, last.to_square)}{where}; " + _mate_geometry(pattern, fin, L.pov, last))
    if bad:
        return out

    # Приём засчитывается ходу, только если начинается сыгранным ходом (k == 1; расплата — не позже
    # следующего хода, k == 3). Иначе правило находит приём в глубине варианта, не связанный с ходом.
    fk = K.fork(L)
    if fk and fk["k"] == 1 and ((threat_target and chess.parse_square(threat_target["square"]) in fk["targets"])
               or (net is not None and net >= 1)):
        b = L.after(fk["k"])
        add("вилка", 80, f"{_name(b, fk['sq'])} {_when(fk['k'])} одновременно бьёт "
            + " и ".join(_name(b, t) for t in fk["targets"]))
    sk = K.skewer(L)
    if sk and sk["k"] == 1:
        b = L.before(sk["k"] + 1) if sk["k"] + 1 < len(L) else L.after(sk["k"])
        add("сквозной удар", 80, f"{_name(L.after(sk['k']), sk['slider'])} нападает на {_name(b, sk['front'])}, "
            f"а после ухода этой фигуры берёт {_name(b, sk['back'])}")
    da = K.discovered_attack(L)
    if da and "slider" in da and da["k"] == 1:
        b = L.after(da["k"])
        add("вскрытое нападение", 70, f"ход с {chess.square_name(da['vacated'])} открывает линию: "
            f"{_name(b, da['slider'])} бьёт {_name(b, da['target'])}")
    pn = K.pin(L)
    if pn and pn["k"] == 1:
        b = L.after(1)
        by = f", связывает {_name(b, pn['pinner'])}" if pn.get("pinner") is not None else ""
        add("связка", 65, f"{_name(b, pn['pinned'])} связана: за ней на той же линии стоит король{by}")
    df = K.deflection(L)
    if df and df["k"] == 1:
        b = L.before(df["k"] + 1)
        # Расплата — ход X на освобождённое поле (взятие или превращение): без неё термин не проверить.
        pay_k = df["k"] + 2
        payer = L.after(pay_k).piece_at(df["sq"]) if pay_k < len(L) else None
        pay = (f", и затем {piece_sym_ru(payer.symbol())} {'берёт на' if L.is_capture(pay_k) else 'занимает'} "
               f"{chess.square_name(df['sq'])}") if payer else ""
        add("отвлечение", 75, f"фигура соперника уходит с {chess.square_name(df['defender_from'])} на "
            f"{chess.square_name(df['defender_to'])} и перестаёт защищать {chess.square_name(df['sq'])}{pay}")
    at = K.attraction(L)
    if at and at["k"] == 1:
        add("завлечение", 75, f"фигура соперника с {chess.square_name(at['lured'])} вынуждена встать на "
            f"{chess.square_name(at['sq'])}, где попадает под удар")
    cd = K.capturing_defender(L)
    if cd and cd["k"] == 1:
        b = L.before(cd["k"])
        add("уничтожение защиты", 75, f"берётся защитник: {_name(b, cd['defender'])}; без него "
            f"остаётся без защиты {_name(b, cd['target'])}")
    itf = K.interference(L)
    if itf and not itf["by_opponent"] and itf["k"] == 1:
        b = L.before(itf["k"])
        add("перекрытие", 75, f"фигура встаёт на {chess.square_name(itf['block'])} между {_name(b, itf['defender'])} "
            f"и {_name(b, itf['target'])}")
    cl = K.clearance(L)
    # Освобождение линии в обычной перестройке (g3 и слон на g2) — не тактика: нужен шах, мат или выигрыш.
    if cl and cl["k"] == 1 and (L.after(3).is_check() or (net is not None and net >= 1)
                                or (fin.is_checkmate() and fin.turn != L.pov)):
        add("освобождение линии", 65, f"фигура уходит с {chess.square_name(cl['vacated'])}, и "
            f"{_name(L.after(3), cl['to'])} проходит с {chess.square_name(cl['user'])} на {chess.square_name(cl['to'])}")
    im = K.intermezzo(L)
    if im and im["k"] == 1:
        add("промежуточный ход", 75, f"вместо ответного взятия на {chess.square_name(im['sq'])} сначала сделан другой ход")
    xr = K.x_ray(L)
    if xr and xr["k"] == 3:
        add("рентген", 60, f"{_name(L.before(xr['k']), xr['slider'])} действует сквозь фигуру и берёт на "
            f"{chess.square_name(xr['sq'])}")
    tp = K.trapped_piece(L)
    if tp and tp["k"] == 3:
        b = L.before(tp["k"] - 1)
        lost_on = chess.square_name(L.moves[tp["k"]].to_square)
        add("ловля фигуры", 75, f"{_name(b, tp['sq'])} под боем, и любой её отход тоже под боем: "
            f"в варианте она теряется на {lost_on}")
    sc = K.sacrifice(L)
    # Жертва — только когда у отдающей стороны остаются шансы: иначе это лучший ход в проигранной позиции.
    sa = eng["score_after"]
    after_mover = (sa["cp"] if sa["cp"] is not None else (100000 if (sa["mate"] or 0) > 0 else -100000)) * (1 if L.pov else -1)
    if sc and sc["k"] == 3 and not mates_now and after_mover >= -150 and eng["cpl"] <= 50:
        add("жертва", 75, "в варианте материал отдаётся, но позиция почти не ухудшается")
    qm = K.quiet_move(L)
    if qm and qm["k"] == 1 and eng["played_is_best"] and (eng.get("only_move_gap") or 0) >= 0.25:
        add("тихий ход", 60, "ход без шаха, взятия и прямого нападения, но единственный сильный")
    if after_mover >= 150 and (K.kingside_attack(L) or K.queenside_attack(L)):
        add("атака на короля", 65, "шахи и взятия у короля соперника в варианте")
    if after_mover >= 150 and K.exposed_king(L):
        add("оголённый король", 55, "перед королём соперника нет пешек, и он попадает под шахи")
    return out


def _opponent_reasons(facts: dict) -> list[Concept]:
    """Почему ход плох: какой приём получает соперник в ответном варианте."""
    Lo = K.opponent_line(facts)
    if Lo is None:
        return []
    # Как и для хода: приём соперника должен начинаться его первым ответом.
    fk = K.fork(Lo)
    if fk and fk["k"] == 1:
        b = Lo.after(fk["k"])
        return [Concept("вилка", "reason", 82, [f"у соперника {_name(b, fk['sq'])} одновременно бьёт "
                                                + " и ".join(_name(b, t) for t in fk["targets"])])]
    sk = K.skewer(Lo)
    if sk and sk["k"] == 1:
        b = Lo.after(sk["k"])
        return [Concept("сквозной удар", "reason", 82, [f"у соперника {_name(b, sk['slider'])} нападает на "
                                                        f"{_name(b, sk['front'])} и затем берёт {_name(b, sk['back'])}"])]
    dch = K.double_check(Lo)
    if dch and dch["k"] == 1:
        b = Lo.after(dch["k"])
        return [Concept("двойной шах", "reason", 82, ["соперник даёт двойной шах: "
                                                     + " и ".join(_name(b, s) for s in dch["checkers"])])]
    tp = K.trapped_piece(Lo)
    if tp and tp["k"] == 3:
        lost_on = chess.square_name(Lo.moves[tp["k"]].to_square)
        return [Concept("ловля фигуры", "reason", 82, [f"{_name(Lo.before(tp['k'] - 1), tp['sq'])} под боем, и любой "
                                                      f"её отход тоже под боем: в варианте она теряется на {lost_on}"])]
    return []


def find_concepts(facts: dict, prev: dict | None = None) -> list[Concept]:
    d, after, eng = facts["delta"], facts["after"], facts["engine"]
    mot = eng["motifs"]
    me = d["color"]
    color = chess.WHITE if me == "w" else chess.BLACK
    bad = eng["nag"] in (2, 4, 6) or bool(eng.get("allows_mate_in"))
    reply = eng.get("best_reply") or {}
    line = reply.get("line") or []
    out: list[Concept] = []

    # Материальный итог хода с учётом линии ответа (само взятие сходившей стороны тоже считаем).
    summ = reply.get("summary")
    net = None
    if summ:
        net = summ["net_for_white"] * (1 if me == "w" else -1)
        if d["captured"]:
            net += min(_val(d["captured"]["piece"]), 9)
    ot = eng.get("own_threat")
    threat_target = (ot or {}).get("move", {}).get("captured")

    # --- что делает ход ------------------------------------------------------------------
    # Тактика и маты — ключи по варианту движка (gcdata/keys.py, правила как у разметчика задач lichess)
    # плюс наши условия: ход не ошибка, приём что-то даёт.
    L, _ = K.line_from_facts(facts, prev)
    key_terms = _key_concepts(L, facts, bad, net, threat_target) if L is not None else []
    out += key_terms
    has = {c.term for c in key_terms}
    # «Перегрузка» отключена: на проверке судьёй 7 из 8 срабатываний были неуместны (у lichess тоже заглушка).

    cap = mot["capture_kind"]
    recaptured = bool(line) and line[0].get("captured") and line[0]["to"] == d["to"]
    if "жертва" in has:
        pass
    elif cap == "выигрыш материала" and not bad and (net is None or net >= 1):
        out.append(Concept("выигрыш материала", "move", 70, [f"взята {_ps(d['captured'])}, и вернуть материал соперник не может"]))
    elif d["is_capture"] and recaptured and _val(d["captured"]["piece"]) == _val(d["piece"]):
        out.append(Concept("размен", "move", 40,
                           [f"{_ps(d['captured'])} взята, и соперник отвечает взятием на {d['to']}"]))

    if ot and not after["is_checkmate"]:
        tm = ot["move"]
        target = threat_target
        # Выигрыш темпа — нападение на фигуру от лёгкой и выше, которую соперник вынужден уводить.
        if (target and _val(target["piece"]) >= 3 and line and line[0]["from"] == target["square"]
                and not line[0].get("captured") and not line[0].get("check") and not bad):
            out.append(Concept("выигрыш темпа", "move", 55,
                               [f"ход нападает на {_ps(target)}, и соперник вынужден его уводить"]))
        else:
            # Кто угрожает — называем явно, иначе LLM приписывает угрозу не той фигуре.
            what = (f"{piece_sym_ru(tm['piece'])} {tm['from']} берёт {_ps(target)}" if target else
                    "мат" if ot.get("mate_in") else f"{piece_sym_ru(tm['piece'])} {tm['from']} -> {tm['to']}")
            out.append(Concept("угроза", "move", 55, [f"следующим ходом угроза: {what}"]))

    opp = eng.get("opp_threat_before")
    if opp and opp.get("parried"):
        tm = opp["move"]
        quiet_ours = not d["is_capture"] and not d["gives_check"] and not after["is_checkmate"]
        threat_piece_alive = not (d["captured"] and d["captured"]["square"] == tm["from"])
        quiet_threat = not (tm.get("captured") or tm.get("check") or tm.get("mate")) and not opp.get("mate_in")
        if not threat_piece_alive:
            # Угроза исчезла, потому что угрожавшую фигуру просто взяли, — это не профилактика.
            out.append(Concept("защита от угрозы", "move", 55,
                               [f"ход забирает фигуру, которая готовила {piece_sym_ru(tm['piece'])} {tm['from']} -> {tm['to']}"]))
        elif quiet_threat and quiet_ours:
            # Профилактика: тихий, но сильный ход соперника перестаёт работать после нашего тихого хода.
            out.append(Concept("профилактика", "move", 60,
                               [f"до хода у соперника был сильный ход {piece_sym_ru(tm['piece'])} {tm['from']} -> {tm['to']}, "
                                "после хода он больше не даёт преимущества"]))
        elif not quiet_threat:
            what = f"взятие: {_ps(tm['captured'])}" if tm.get("captured") else (
                f"мат в {opp['mate_in']}" if opp.get("mate_in") else f"{piece_sym_ru(tm['piece'])} {tm['from']} -> {tm['to']}")
            out.append(Concept("защита от угрозы", "move", 60, [f"ход отражает угрозу соперника ({what})"]))
        # Тихая угроза, снятая шахом или взятием, — побочный эффект форсированной игры: не упоминаем.

    if mot["castling"]:
        out.append(Concept("рокировка", "move", 40, ["король уходит из центра"]))
    if mot["development"]:
        out.append(Concept("развитие", "move", 20, ["лёгкая фигура покидает исходное поле"]))
    if mot["rook_file"] == "открытая":
        out.append(Concept("открытая вертикаль", "move", 35, [f"на вертикали {d['to'][0]} нет пешек"]))

    for x in mot.get("positional", []):
        k, mine = x["kind"], (x["side"] == color)
        gen = side_gen_ru(chess.WHITE if x["side"] else chess.BLACK)
        if k == "outpost":
            out.append(Concept("форпост", "move", 55,
                               [f"{_ps(x)} защищён пешкой, и пешки соперника не могут его прогнать"]))
        elif k == "passed":
            out.append(Concept("проходная пешка", "move" if mine else "cons", 50,
                               [f"пешке {gen} {x['square']} не мешают пешки соперника"]))
        elif k == "rook_seventh":
            out.append(Concept("седьмая горизонталь", "move", 45, [f"ладья {x['square']} на предпоследней горизонтали соперника"]))
        elif k == "mobility" and x["after"] > x["before"]:
            out.append(Concept("активизация фигуры", "move", 25,
                               [f"{_ps(x)} контролирует заметно больше полей"]))
        # --- последствия -------------------------------------------------------------------
        elif k == "king_shield" and mine:
            # Термин — только когда ход сам снимает пешечное прикрытие своего короля.
            out.append(Concept("ослабление позиции короля", "cons", 60,
                               [f"у короля {gen} становится меньше пешечного прикрытия"]))
        elif k == "king_pressure" and mine:
            # Рост числа фигур соперника у своего короля — факт, но не «ослабление» в смысле термина.
            out.append(Concept("фигуры соперника у короля", "cons", 45,
                               [f"к королю {gen} подтягиваются фигуры соперника"]))
        elif k == "king_pressure" and not mine:
            out.append(Concept("давление на короля", "move", 50,
                               [f"к королю {gen} подтягивается больше фигур"]))
        elif k in ("doubled", "isolated"):
            term = "сдвоенные пешки" if k == "doubled" else "изолированная пешка"
            where = f"на вертикали {x['file']}" if k == "doubled" else x["square"]
            out.append(Concept(term, "cons", 35, [f"у {gen} {where}"]))
        elif k == "bishop_pair_lost":
            out.append(Concept("пара слонов", "cons", 30, [f"{side_ru(chess.WHITE if x['side'] else chess.BLACK)} лишаются пары слонов"]))

    for x in mot["own_en_prise"][:1]:
        out.append(Concept("незащищённая фигура", "cons", 55, [f"{_ps(x)} остаётся под боем"]))
    if mot["lost_castling_rights"]:
        out.append(Concept("потеря рокировки", "cons", 35, ["король больше не сможет рокироваться"]))

    sb_, sa_ = eng["score_before"], eng["score_after"]
    mate_after = sa_["mate"] is not None and sa_["mate"] != 0 and ((sa_["mate"] > 0) == (me == "w"))
    mate_before = sb_["mate"] is not None and ((sb_["mate"] > 0) == (me == "w"))
    if mate_after and not bad and not after["is_checkmate"]:
        ev = "ход ведёт к форсированному мату" if not mate_before else "ход сохраняет форсированный мат"
        out.append(Concept("форсированный мат", "move", 85 if not mate_before else 45, [ev]))

    # --- почему ход плох ----------------------------------------------------------------
    if bad:
        out += _opponent_reasons(facts)
        if eng.get("allows_mate_in"):
            out.append(Concept("допускает мат", "reason", 100, [f"у соперника мат в {eng['allows_mate_in']}"]))
        if opp and not opp.get("parried"):
            tm = opp["move"]
            out.append(Concept("пропущенная угроза", "reason", 85,
                               [f"не отражена угроза: {piece_sym_ru(tm['piece'])} {tm['from']} -> {tm['to']}"
                                + (f" со взятием: {_ps(tm['captured'])}" if tm.get("captured") else "")]))
        s = reply.get("summary") or {}
        lost = s.get("white_lost" if me == "w" else "black_lost") or []
        net = (s.get("net_for_white", 0)) * (1 if me == "w" else -1)
        if lost and net <= -1:
            from .ru import PIECE_RU
            names = ", ".join(PIECE_RU[chess.Piece.from_symbol(p).piece_type] for p in lost)
            got = s.get("black_lost" if me == "w" else "white_lost") or []
            # Обмен показываем целиком: «теряют пешку и слона», когда взамен взяли коня, — неправда по смыслу.
            back = (", взамен берут: " + ", ".join(PIECE_RU[chess.Piece.from_symbol(p).piece_type] for p in got)) if got else ""
            out.append(Concept("потеря материала", "reason", 80,
                               [f"после ответа соперника {side_ru(color)} отдают: {names}{back}; в итоге они в минусе по материалу"]))
        # Короля сходившего гонят шахами в линии ответа.
        checks = [m for m in line if m.get("check") and (m["piece"].isupper() != (me == "w"))]
        king_moves = [m for m in line if m["piece"] == ("K" if me == "w" else "k")]
        if checks and king_moves:
            k = king_moves[0]
            out.append(Concept("атака на короля", "reason", 75,
                               [f"после шаха король {side_gen_ru(color)} вынужден уходить с {k['from']} на {k['to']}"]))
        b = eng["best_move_before"]
        if not eng["played_is_best"]:
            bs = b.get("line_summary") or {}
            bnet = bs.get("net_for_white", 0) * (1 if me == "w" else -1)
            if b.get("mate_in"):
                out.append(Concept("упущенная возможность", "better", 90, [f"был мат в {b['mate_in']}"]))
            elif bnet >= 2:
                out.append(Concept("упущенная возможность", "better", 70, ["сильнейший ход выигрывал материал"]))
    return out


def term_hits(text: str) -> set:
    low = text.lower()
    return {t for t, pat in TERM_PATTERNS.items() if re.search(pat, low)}
