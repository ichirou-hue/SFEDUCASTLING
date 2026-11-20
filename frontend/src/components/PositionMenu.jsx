import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Chess } from "chess.js";
import copyText from "../clipboard.js";
import "./PositionMenu.css";

const NOTICE_TIMEOUT = 2600;

const readFenFromFile = (name, text) => {
  const content = String(text || "").trim();
  if (!content) throw new Error("Файл пуст");

  // PGN - берём позицию после последнего хода партии.
  if (/\.pgn$/i.test(name)) {
    const game = new Chess();
    game.loadPgn(content, { strict: false });
    return game.fen();
  }

  // .fen / .txt - первая непустая строка файла.
  const line = content
    .split(/\r?\n/)
    .map((row) => row.trim())
    .find(Boolean);
  if (!line) throw new Error("Файл пуст");
  return line;
};

const isValidFen = (fen) => {
  try {
    new Chess(fen);
    return true;
  } catch {
    return false;
  }
};

const downloadFile = (name, text, type) => {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
};

/*
 * Меню «Позиция»: всё про импорт и экспорт в одном месте -
 * загрузка FEN, импорт из файла, копирование и сохранение FEN/PGN.
 */
export default function PositionMenu({ fen, onLoadFen, getPgn }) {
  const [open, setOpen] = useState(false);
  const [fenInput, setFenInput] = useState(fen || "");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const rootRef = useRef(null);
  const buttonRef = useRef(null);
  const panelRef = useRef(null);
  const inputRef = useRef(null);
  const fileRef = useRef(null);
  const noticeTimerRef = useRef(0);
  // Меню открываем вниз, если под кнопкой есть место, иначе вверх.
  // maxHeight не даёт панели выйти за пределы окна: на странице /play
  // внешний скролл-контейнер обрезает содержимое по вертикали.
  const [placement, setPlacement] = useState({ up: false, maxHeight: null });

  useLayoutEffect(() => {
    if (!open) return;

    const place = () => {
      const button = buttonRef.current;
      const panel = panelRef.current;
      if (!button || !panel) return;

      // На телефоне меню рисуется фиксированным листом снизу - адаптив не нужен.
      if (window.matchMedia("(max-width: 760px)").matches) {
        setPlacement({ up: false, maxHeight: null });
        return;
      }

      const rect = button.getBoundingClientRect();
      const gap = 10;
      const edge = 12;
      const natural = panel.scrollHeight;
      const spaceBelow = window.innerHeight - rect.bottom - gap - edge;
      const spaceAbove = rect.top - gap - edge;
      const up = natural > spaceBelow && spaceAbove > spaceBelow;
      const maxHeight = Math.max(160, up ? spaceAbove : spaceBelow);
      setPlacement((current) =>
        current.up === up && current.maxHeight === maxHeight
          ? current
          : { up, maxHeight },
      );
    };

    place();
    window.addEventListener("resize", place);
    window.addEventListener("orientationchange", place);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("orientationchange", place);
    };
  }, [open]);

  const close = () => {
    setOpen(false);
    setNotice("");
    setError("");
  };

  const flash = (message, isError = false) => {
    if (isError) {
      setError(message);
      setNotice("");
    } else {
      setNotice(message);
      setError("");
    }
    window.clearTimeout(noticeTimerRef.current);
    noticeTimerRef.current = window.setTimeout(() => {
      setNotice("");
      setError("");
    }, NOTICE_TIMEOUT);
  };

  // Поле FEN подтягивает текущую позицию, пока меню закрыто.
  useEffect(() => {
    if (!open) setFenInput(fen || "");
  }, [fen, open]);

  useEffect(() => {
    if (!open) return undefined;

    const handlePointerDown = (event) => {
      if (rootRef.current && !rootRef.current.contains(event.target)) close();
    };
    const handleKeyDown = (event) => {
      if (event.key === "Escape") close();
    };

    document.addEventListener("mousedown", handlePointerDown);
    document.addEventListener("touchstart", handlePointerDown, {
      passive: true,
    });
    document.addEventListener("keydown", handleKeyDown);

    const focusTimer = window.setTimeout(() => inputRef.current?.focus(), 30);

    return () => {
      document.removeEventListener("mousedown", handlePointerDown);
      document.removeEventListener("touchstart", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
      window.clearTimeout(focusTimer);
    };
  }, [open]);

  useEffect(() => () => window.clearTimeout(noticeTimerRef.current), []);

  const applyFen = (value) => {
    const nextFen = String(value || "").trim();
    if (!nextFen) {
      flash("Введите FEN-строку", true);
      return;
    }
    if (!isValidFen(nextFen)) {
      flash("Неверный формат FEN", true);
      return;
    }
    if (onLoadFen?.(nextFen) === false) {
      flash("Не удалось загрузить позицию", true);
      return;
    }
    close();
  };

  const handleFile = (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const nextFen = readFenFromFile(file.name, ev.target?.result);
        if (!isValidFen(nextFen)) {
          flash("В файле нет корректного FEN", true);
          return;
        }
        setFenInput(nextFen);
        applyFen(nextFen);
      } catch {
        flash("Не удалось прочитать файл", true);
      }
    };
    reader.onerror = () => flash("Не удалось прочитать файл", true);
    reader.readAsText(file);
  };

  const getPgnText = () => String(getPgn?.() || "").trim();

  const copyFen = async () => {
    const copied = await copyText(fen || "");
    flash(copied ? "FEN скопирован" : "Не удалось скопировать FEN", !copied);
  };

  const copyPgn = async () => {
    const pgn = getPgnText();
    if (!pgn) {
      flash("Пока нет ходов для PGN", true);
      return;
    }
    const copied = await copyText(pgn);
    flash(copied ? "PGN скопирован" : "Не удалось скопировать PGN", !copied);
  };

  const saveFen = () => {
    downloadFile("position.fen", fen || "", "application/x-chess-fen");
    flash("Файл FEN сохранён");
  };

  const savePgn = () => {
    const pgn = getPgnText();
    if (!pgn) {
      flash("Пока нет ходов для PGN", true);
      return;
    }
    downloadFile("game.pgn", pgn, "application/x-chess-pgn");
    flash("Файл PGN сохранён");
  };

  return (
    <div className="position-menu" ref={rootRef}>
      <button
        ref={buttonRef}
        type="button"
        className="btn-position"
        aria-haspopup="dialog"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        Позиция
        <span className="btn-position__chevron" aria-hidden="true" />
      </button>

      {open && (
        <>
          <div
            className="position-menu__backdrop"
            aria-hidden="true"
            onClick={close}
          />
          <div
            ref={panelRef}
            className={`position-menu__panel${
              placement.up && placement.maxHeight ? " position-menu__panel--up" : ""
            }`}
            style={placement.maxHeight ? { maxHeight: `${placement.maxHeight}px` } : undefined}
            role="dialog"
            aria-label="Импорт и экспорт позиции"
          >
            <h4 className="position-menu__title">Импорт и экспорт</h4>

            <div className="position-menu__group">
              <span className="position-menu__label">Загрузить из FEN</span>
              <div className="position-menu__row">
                <input
                  ref={inputRef}
                  type="text"
                  className="position-menu__input"
                  value={fenInput}
                  onChange={(event) => setFenInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      applyFen(fenInput);
                    }
                  }}
                  placeholder="Вставьте FEN-строку"
                  aria-label="FEN-строка"
                  spellCheck={false}
                />
                <button
                  type="button"
                  className="position-menu__action"
                  onClick={() => applyFen(fenInput)}
                >
                  Загрузить
                </button>
              </div>
              <div className="position-menu__row">
                <button
                  type="button"
                  className="position-menu__action position-menu__action--ghost"
                  onClick={() => fileRef.current?.click()}
                >
                  Импортировать из файла
                </button>
              </div>
              <span className="position-menu__hint">
                Поддерживаются .fen, .txt и .pgn
              </span>
              <input
                ref={fileRef}
                type="file"
                accept=".fen,.txt,.pgn"
                className="position-menu__file"
                onChange={handleFile}
              />
            </div>

            <div className="position-menu__group">
              <span className="position-menu__label">Экспорт</span>
              <textarea
                className="position-menu__textarea"
                value={fen || ""}
                rows={2}
                readOnly
                aria-label="Текущий FEN"
                onFocus={(event) => event.target.select()}
              />
              <div className="position-menu__row">
                <button
                  type="button"
                  className="position-menu__action"
                  onClick={copyFen}
                >
                  Копировать FEN
                </button>
                <button
                  type="button"
                  className="position-menu__action position-menu__action--ghost"
                  onClick={saveFen}
                >
                  Скачать .fen
                </button>
              </div>
              <div className="position-menu__row">
                <button
                  type="button"
                  className="position-menu__action"
                  onClick={copyPgn}
                >
                  Копировать PGN
                </button>
                <button
                  type="button"
                  className="position-menu__action position-menu__action--ghost"
                  onClick={savePgn}
                >
                  Скачать .pgn
                </button>
              </div>
            </div>

            <span
              className={`position-menu__status${
                error ? " position-menu__status--error" : ""
              }`}
              role="status"
              aria-live="polite"
            >
              {error || notice}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
