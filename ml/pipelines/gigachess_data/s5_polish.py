"""Шаг 5. Синтаксическое связывание черновика через LLM (вместо Reasoning Draft Model) + Валидатор №2.

LLM только склеивает факты. Скрипт затем требует точного совпадения множеств клеток и чисел
с черновиком, вердикта, 4 абзацев, отсутствия нотации/первого лица/штампов и лимита длины.
При отказе — повтор с перечнем нарушений (до --attempts). Отклонённые варианты сохраняются
в rejected.jsonl: в паре с принятым они дают DPO-пары (chosen/rejected).

--dry-run печатает промпты без обращения к API.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcdata.checks import check_polished
from gcdata.io import done_ids, load_env, read_jsonl, write_jsonl
from gcdata.llm import LLMClient, LLMError
from gcdata.prompts import POLISH_SYSTEM, polish_user

STOP = threading.Event()  # выставляется при ошибке оплаты/ключа: новых запросов не шлём
MAX_ERROR_STREAK = 24  # столько ошибок API подряд — шлюз лежит, останавливаемся
FATAL_RE = re.compile(r"HTTP (401|402|403)(?!\d)")


# Блок рассуждений целиком или его хвост «…</think>» в начале ответа (Ollama иногда отдаёт только хвост).
THINK_RE = re.compile(r"<think>.*?</think>|^.*?</think>|^\s*(assistant|\*/)\s*", re.S)
SYSTEM = POLISH_SYSTEM
USER_SUFFIX = ""


def with_fewshot(path: str) -> str:
    """Системный промпт + образцы «план → текст»: маленькие локальные модели лучше повторяют
    образец, чем выполняют длинный список правил."""
    shots = json.load(open(path, encoding="utf-8"))
    ex = "\n\n".join(f"План:\n{dr}\n\nТекст:\n{ans}" for dr, ans in shots.values())
    return POLISH_SYSTEM + "\n\nПримеры правильной работы:\n\n" + ex


def polish_one(client: LLMClient, d: dict, attempts: int) -> tuple[dict | None, list[dict]]:
    rejected, prev, errs = [], None, None
    for k in range(1, attempts + 1):
        if STOP.is_set():
            raise LLMError("остановлено: предыдущий запрос получил ошибку оплаты или ключа")
        text = client.complete(SYSTEM, polish_user(d["draft"], d["max_chars"], prev, errs) + USER_SUFFIX)
        # Модели с режимом рассуждений (Qwen3, T-lite 2.x) могут вернуть блок <think> — в текст он не идёт.
        text = THINK_RE.sub("", text).strip()
        errs = check_polished(d["draft"], text, d["verdict"], d["cpl"], d["nag"], d["max_chars"],
                              intent=d["intent"]["text"] if d["intent_used"] else None)
        if not errs:
            return {"id": d["id"], "answer": text, "attempts": k, "model": client.used_model()}, rejected
        rejected.append({"id": d["id"], "attempt": k, "text": text, "reasons": errs, "model": client.used_model()})
        prev = text
    return None, rejected


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inp", required=True, help="drafts.jsonl из шага 4")
    ap.add_argument("--out", required=True, help="polished.jsonl")
    ap.add_argument("--rejected", help="по умолчанию <out>.rejected.jsonl")
    ap.add_argument("--backend", choices=["openai", "anthropic"],
                    help="по умолчанию openai, если задан LLM_BASE_URL, иначе anthropic")
    ap.add_argument("--model", help="по умолчанию LLM_MODEL (openai) или claude-opus-5-5 (anthropic)")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--env", help="Environment file; defaults to repository root .env")
    ap.add_argument("--fewshot", help="json с образцами «план → текст» (для локальных моделей)")
    ap.add_argument("--lessons", help="текстовый файл с разобранными ошибками прошлых пачек — дописывается к промпту")
    ap.add_argument("--no-think", action="store_true", help="добавить /no_think (Qwen3 и T-lite 2.x)")
    args = ap.parse_args()
    load_env(args.env)
    global SYSTEM, USER_SUFFIX
    if args.fewshot:
        SYSTEM = with_fewshot(args.fewshot)
    if args.lessons and Path(args.lessons).exists():
        SYSTEM += ("\n\nЧастые ошибки прошлых текстов — не повторяй их:\n"
                   + Path(args.lessons).read_text(encoding="utf-8").strip())
    if args.no_think:
        USER_SUFFIX = "\n/no_think"
    rej_path = args.rejected or str(Path(args.out).with_suffix("")) + ".rejected.jsonl"

    seen = done_ids(args.out)
    todo = [d for d in read_jsonl(args.inp) if d["id"] not in seen]
    if args.limit:
        todo = todo[: args.limit]

    if args.dry_run:
        for d in todo[:3]:
            print("=" * 80, "\n[SYSTEM]\n" + POLISH_SYSTEM, "\n[USER]\n" + polish_user(d["draft"], d["max_chars"]))
        print(f"\n(dry-run: {len(todo)} записей ждут обработки)")
        return

    try:
        client = LLMClient(backend=args.backend, model=args.model, max_tokens=4000)
    except LLMError as e:
        sys.exit(f"LLM не настроен: {e}")
    lock = threading.Lock()
    ok = fail = err = streak = 0
    with ThreadPoolExecutor(args.concurrency) as pool:
        futs = {pool.submit(polish_one, client, d, args.attempts): d for d in todo}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                res, rejected = fut.result()
            except LLMError as e:
                err += 1
                streak += 1
                print(f"[{d['id']}] LLM: {e}", file=sys.stderr)
                if streak >= MAX_ERROR_STREAK:
                    # Шлюз лежит: дальше каждый пример будет висеть на таймаутах. Готовое сохранено.
                    STOP.set()
                    pool.shutdown(wait=False, cancel_futures=True)
                    sys.exit(f"остановлено: {streak} ошибок API подряд, шлюз не отвечает. Повторный запуск продолжит.")
                if FATAL_RE.match(str(e)):
                    STOP.set()
                    # Баланс, ключ, доступ: повторы бесполезны и сжигают лимиты шлюза.
                    pool.shutdown(wait=False, cancel_futures=True)
                    sys.exit(f"остановлено: {e}. Готовые записи сохранены, повторный запуск продолжит с места остановки.")
                continue
            streak = 0
            with lock:
                if rejected:
                    write_jsonl(rej_path, rejected, append=True)
                if res:
                    write_jsonl(args.out, [res], append=True)
                    ok += 1
                else:
                    # Окончательно отклонённые тоже помечаем в out, чтобы не платить за них повторно.
                    write_jsonl(args.out, [{"id": d["id"], "answer": None, "attempts": args.attempts,
                                            "model": client.model}], append=True)
                    fail += 1
            if (ok + fail) % 100 == 0:
                print(f"  принято {ok}, отклонено {fail}, ошибок API {err}")
    print(f"принято: {ok}, отклонено после {args.attempts} попыток: {fail}, ошибок API: {err}")
    print(f"отклонённые варианты (DPO rejected): {rej_path}")


if __name__ == "__main__":
    main()
