"""Скриптовые проверки (Валидатор №2): LLM только связывает слова, ничего не добавляя."""
from __future__ import annotations

import re

SQUARE_RE = re.compile(r"(?<![A-Za-z0-9])[a-h][1-8](?![A-Za-z0-9])")
SQUARE_LOOSE_RE = re.compile(r"[a-h][1-8]")  # для сырых комментариев с SAN (Nxf7 -> f7)
NUM_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?![\w])")
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")
FILE_RE = re.compile(r"[a-h]")  # «линия a», «пешка h» — обычная русская шахматная речь
SAN_RE = re.compile(r"\b(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h]x[a-h][1-8]|O-O(?:-O)?|0-0(?:-0)?)")
FIRST_PERSON_RE = re.compile(r"(?<![а-яё])(я|мы|мне|нам|меня|нас|мной|нами|мой|моя|моё|мое|мои|наш|наша|наше|наши)(?![а-яё])",
                             re.IGNORECASE)
CLICHES = [
    "стоит отметить", "важно отметить", "следует отметить", "нельзя не отметить", "таким образом",
    "безусловно", "несомненно", "как говорится", "в целом", "по сути", "в конечном итоге",
    "давайте", "интересно, что", "любопытно", "настоящий шедевр", "игра на грани",
]
# «лучший ход» не включаем: в разборе ошибки им описывают альтернативу, а не сыгранный ход.
POSITIVE_RE = re.compile(r"(отличн|блестящ|прекрасн|великолепн|сильный ход|точный ход|красив)", re.IGNORECASE)
VERDICT_ROOTS = {
    "сильный ход": ["сильн"],
    "ошибка": ["ошибк"],
    "грубая ошибка (зевок)": ["зевок", "зевк", "грубая ошибка", "грубой ошибк", "грубую ошибк"],
    "неточность": ["неточност"],
    "точный ход": ["точн"],
    "приемлемый ход": ["приемлем"],
}


def squares(text: str) -> set:
    return set(SQUARE_RE.findall(text))


def numbers(text: str) -> set:
    # Цифры внутри клеток (e4) не попадают из-за lookbehind на букву.
    return {n.replace(",", ".") for n in NUM_RE.findall(text)}


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def check_polished(draft: str, text: str, verdict: str, cpl: int, nag: int | None,
                   max_chars: int, intent: str | None = None, n_paragraphs: int | None = None) -> list[str]:
    errs = []
    if n_paragraphs is None:
        n_paragraphs = 1  # описание хода — один абзац
    errs += check_terms(draft, text)
    errs += check_bindings(draft, text)
    draft = re.sub(r"^\d\.\s*[^:]+:\s*", "", draft, flags=re.MULTILINE)  # «1. Замысел:» — не факт
    # Цитата игрока: её клетки (в т.ч. из нотации Dh5, Nxf7) разрешены, но не обязательны.
    facts_only = re.sub(r"«[^»]*»", "", draft)
    allowed_sq = squares(facts_only) | set(SQUARE_LOOSE_RE.findall(intent or ""))
    ds, ts = squares(facts_only), squares(text)
    if ts - allowed_sq:
        errs.append(f"новые клетки, которых нет в черновике: {sorted(ts - allowed_sq)}")
    if ds - ts:
        errs.append(f"потеряны клетки черновика: {sorted(ds - ts)}")
    dn, tn = numbers(facts_only), numbers(text)
    tn -= numbers(intent or "")
    if tn - dn:
        errs.append(f"новые числа: {sorted(tn - dn)}")
    if dn - tn:
        errs.append(f"потеряны числа черновика: {sorted(dn - tn)}")
    if SAN_RE.search(text):
        errs.append(f"шахматная нотация в тексте: {SAN_RE.search(text).group(0)}")
    latin = [w for w in LATIN_WORD_RE.findall(text) if not (SQUARE_RE.fullmatch(w) or FILE_RE.fullmatch(w))]
    if latin:
        errs.append(f"латиница кроме клеток: {latin[:5]}")
    if FIRST_PERSON_RE.search(text):
        errs.append(f"первое лицо: «{FIRST_PERSON_RE.search(text).group(0)}»")
    low = text.lower()
    hit = [c for c in CLICHES if c in low]
    if hit:
        errs.append(f"штампы: {hit}")
    if (cpl > 100 or nag in (2, 4)) and POSITIVE_RE.search(text):
        errs.append(f"положительный тон при ошибке (CPL {cpl}): «{POSITIVE_RE.search(text).group(0)}»")
    roots = VERDICT_ROOTS.get(verdict, [])
    if roots and not any(r in low for r in roots):
        errs.append(f"не передан вердикт «{verdict}»")
    ps = paragraphs(text)
    if len(ps) != n_paragraphs:
        errs.append(f"абзацев {len(ps)}, нужно {n_paragraphs}")
    if re.search(r"^\s*(\d+[.)]|[-*•]|#)", text, re.MULTILINE):
        errs.append("нумерация, списки или заголовки")
    # Предложение с маленькой буквы — частая небрежность локальных моделей.
    no_abbr = re.sub(r"\b(ед|т\.\s?е|т\.\s?д|т\.\s?п|др|пр|см|г)\.", "", text.strip())
    if re.search(r"(^|[.!?]\s+)[а-яё]", no_abbr):
        errs.append("предложение начинается с маленькой буквы")
    if "->" in text:
        errs.append("стрелка «->» из плана в тексте")
    if "\n" in text.strip() and n_paragraphs == 1:
        errs.append("переносы строк внутри абзаца")
    # Причинную связь можно ставить, только если в плане есть метка «причина:» — иначе это догадка.
    if "причина:" not in draft and CAUSE_RE.search(text):
        errs.append(f"причинный союз без причины в плане: «{CAUSE_RE.search(text).group(0)}»")
    if len(text) > max_chars:
        errs.append(f"длина {len(text)} > {max_chars} символов")
    return errs


CAUSE_RE = re.compile(r"потому что|так как|поскольку|из-за того|\bведь\b", re.IGNORECASE)
RELATION_LABELS_RE = re.compile(r"(причина|в результате|вместо этого|однако|вердикт)\s*:", re.IGNORECASE)


def check_terms(draft: str, text: str) -> list[str]:
    """Термины: все из черновика ([вилка]) должны прозвучать, чужие — запрещены."""
    from .concepts import TERM_PATTERNS, term_hits
    errs = []
    draft_terms = set(re.findall(r"\[([^\]]+)\]", draft))
    allowed = draft_terms | term_hits(re.sub(r"«[^»]*»", "", draft))
    extra = term_hits(text) - allowed
    if extra:
        errs.append(f"термины, которых нет в черновике: {sorted(extra)}")
    low = text.lower()
    missing = [t for t in draft_terms
               if not re.search(TERM_PATTERNS.get(t, re.escape(t.lower()[:6])), low)]
    if missing:
        errs.append(f"не названы термины черновика: {missing}")
    if "[" in text or "]" in text:
        errs.append("квадратные скобки из черновика в тексте")
    if RELATION_LABELS_RE.search(text):
        errs.append(f"служебная метка в тексте: «{RELATION_LABELS_RE.search(text).group(0)}»")
    return errs


_COLOR = r"(бел|ч[её]рн)\w*"
_PIECE = {"пешк": "p", "кон": "n", "слон": "b", "ладь": "r", "ферз": "q", "корол": "k"}
# «чёрного короля e6», «белый конь на c7» — фигура вплотную к клетке (допускаем «на/с/из»).
TRIPLE_RE = re.compile(_COLOR + r"\s+(пешк|кон|слон|ладь|ферз|корол)\w*\s+(?:(?:на|с|из|со)\s+)?([a-h][1-8])(?![0-9])")
TEXT_MOVE_RE = re.compile(r"\bс\s+([a-h][1-8])\s+на\s+([a-h][1-8])(?![0-9])")
PLAN_MOVE_RE = re.compile(r"([a-h][1-8])\s*->\s*([a-h][1-8])")


def _triples(text: str) -> set:
    return {(m.group(1)[:3].replace("чер", "чёр"), _PIECE[m.group(2)], m.group(3)) for m in TRIPLE_RE.finditer(text.lower())}


def check_bindings(draft: str, text: str) -> list[str]:
    """Цвет и фигура на клетке и направление ходов в тексте должны совпадать с планом.

    Множества клеток этого не ловят: «белого короля e6» вместо «чёрного короля e6» — те же клетки.
    На одной клетке за линию могут побывать разные фигуры, поэтому допустим любой вариант из плана.
    """
    errs = []
    plan = re.sub(r"«[^»]*»", "", draft)  # цитата игрока — не факт
    allowed = {}
    for c, p, sq in _triples(plan):
        allowed.setdefault(sq, set()).add((c, p))
    wrong = sorted({f"{c} {p} {sq}" for c, p, sq in _triples(text) if sq in allowed and (c, p) not in allowed[sq]})
    if wrong:
        errs.append(f"фигура или цвет на клетке не как в плане: {wrong}")
    # В плане сказано «фигура соперника уходит с e2» без названия — текст не может назвать её королём.
    generic = set(re.findall(r"фигура[^;]*?\bс\s+([a-h][1-8])", plan)) - set(allowed)
    named = sorted({f"{c} {p} {sq}" for c, p, sq in _triples(text) if sq in generic})
    if named:
        errs.append(f"названа фигура, которой в плане нет (там просто «фигура»): {named}")
    # Ход в плане бывает и стрелкой («e6 -> d5»), и словами в обосновании («уходит с e6 на d5»).
    plan_moves = set(PLAN_MOVE_RE.findall(plan)) | set(TEXT_MOVE_RE.findall(plan))
    bad_moves = sorted({f"{a}->{b}" for a, b in TEXT_MOVE_RE.findall(text) if (a, b) not in plan_moves})
    if bad_moves:
        errs.append(f"ход «с X на Y», которого нет в плане: {bad_moves}")
    return errs
