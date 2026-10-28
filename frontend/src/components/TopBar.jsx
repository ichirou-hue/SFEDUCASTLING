import { useEffect, useRef } from "react";
import { Link, useLocation } from "react-router-dom";
import { logout as apiLogout } from "../api.js";
import MiniProfilePopover from "./MiniProfilePopover.jsx";

export default function TopBar({ user, onUserChange, onRegister, onMenuClick, onOpenOnboarding }) {
  const barRef = useRef(null);
  const logoRef = useRef(null);
  const location = useLocation();

  /*
   * Логотип центрируем по доске, а не по всей странице.
   *
   * На игровых страницах доска смещена из-за боковых колонок (история ходов,
   * шкала оценки, чат), поэтому центр страницы и центр доски не совпадают.
   * Доска помечена атрибутом data-logo-anchor, переменная --logo-shift
   * сдвигает логотип в шапке. Страницы без доски (профиль, админка)
   * остаются с обычным центрированием.
   */
  useEffect(() => {
    const root = document.documentElement;
    const bar = barRef.current;
    const logo = logoRef.current;

    let anchor = null;
    let sizeObserver = null;
    let lastShift = null;
    let frame = 0;

    const setShift = (shift) => {
      if (shift === lastShift) return;
      lastShift = shift;
      root.style.setProperty("--logo-shift", `${shift}px`);
    };

    const apply = () => {
      if (!bar || !logo) return;

      const rect = anchor?.getBoundingClientRect();
      if (!rect || !rect.width) {
        setShift(0);
        return;
      }

      const barRect = bar.getBoundingClientRect();

      /* offsetLeft не учитывает transform, поэтому это позиция логотипа
         до сдвига, как её разложил грид шапки. */
      const baseCenter = barRect.left + logo.offsetLeft + logo.offsetWidth / 2;
      const half = logo.offsetWidth / 2;
      let target = rect.left + rect.width / 2;

      /* Логотип не должен наезжать на гамбургер и блок пользователя:
         в узком окне доска уходит под правую группу шапки. */
      const leftEl = bar.querySelector(".top-bar-left");
      const rightEl = bar.querySelector(".user-box, .reg-btn");
      const leftEnd = leftEl ? leftEl.getBoundingClientRect().right : barRect.left;
      const rightStart = rightEl ? rightEl.getBoundingClientRect().left : barRect.right;
      const minCenter = leftEnd + 12 + half;
      const maxCenter = rightStart - 12 - half;

      if (maxCenter >= minCenter) {
        target = Math.min(Math.max(target, minCenter), maxCenter);
      } else {
        setShift(0);
        return;
      }

      setShift(Math.round(target - baseCenter));
    };

    /* Пересчёт не чаще раза за кадр: доска меняет размер при подгонке под
       экран, при сворачивании чата и при смене страницы.

       Пока на странице что-то анимируется (сворачивание чата, изменение
       размера доски), замеряем каждый кадр: доска едет, и логотип должен
       ехать вместе с ней, а не прыгать в конце. Ограничение по кадрам -
       страховка от бесконечных анимаций вроде спиннеров. */
    let chase = 0;

    const tick = () => {
      frame = 0;
      apply();

      const running =
        chase < 240 &&
        typeof document.getAnimations === "function" &&
        document.getAnimations().some((a) => a.playState === "running");

      if (running) {
        chase += 1;
        frame = requestAnimationFrame(tick);
      }
    };

    const schedule = () => {
      if (frame) return;
      chase = 0;
      frame = requestAnimationFrame(tick);
    };

    const findAnchor = () => {
      const next = document.querySelector("[data-logo-anchor]");
      if (next !== anchor) {
        anchor = next;
        sizeObserver?.disconnect();
        sizeObserver = null;
        if (anchor && typeof ResizeObserver !== "undefined") {
          sizeObserver = new ResizeObserver(schedule);
          sizeObserver.observe(anchor);
        }
      }
      schedule();
    };

    findAnchor();

    const mutations = new MutationObserver(findAnchor);
    mutations.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class", "style"],
    });
    window.addEventListener("resize", findAnchor);

    return () => {
      if (frame) cancelAnimationFrame(frame);
      mutations.disconnect();
      sizeObserver?.disconnect();
      window.removeEventListener("resize", findAnchor);
      root.style.removeProperty("--logo-shift");
    };
  }, []);

  /* Клик по логотипу открывает главную страницу. Если она уже открыта,
     просто поднимаем содержимое наверх: переход на тот же адрес ничего
     не перерисовывает, и клик иначе выглядел бы «мёртвым». */
  const handleLogoClick = (event) => {
    if (location.pathname !== "/") return;
    event.preventDefault();
    const reduced =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const behavior = reduced ? "auto" : "smooth";
    const scroller = document.querySelector(".main-page-scroll");
    if (scroller) scroller.scrollTo({ top: 0, behavior });
    else window.scrollTo({ top: 0, behavior });
  };

  const handleLogout = async () => {
    try {
      await apiLogout();
    } finally {
      window.location.reload();
    }
  };

  // Плашка видна, пока анкета не заполнена.
  const needsOnboarding = Boolean(user) && !user?.onboarding_completed;

  return (
    <div className="top-bar" ref={barRef}>
      <div className="top-bar-left">
        <button className="hamburger-btn" onClick={onMenuClick} title="Меню">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>
      </div>
      {/* Логотип — самостоятельная центральная колонка шапки.
          Он же ссылка на главную страницу. */}
      <Link
        to="/"
        className="logo-wrapper"
        ref={logoRef}
        onClick={handleLogoClick}
        title="На главную"
      >
        <img src="/gigachess-logo.svg" alt="GIGACHESS" className="gigachess-logo" />
      </Link>
      {user ? (
        <div className="user-box">
          {needsOnboarding && (
            <button
              className="onb-badge"
              onClick={() => onOpenOnboarding?.()}
              title="Пройти онбординг-анкету"
            >
              Заполните анкету
            </button>
          )}
          <MiniProfilePopover
            user={user}
            onUserChange={onUserChange}
            onLogout={handleLogout}
          />
          {/* На узких экранах меню профиля скрыто, поэтому выход
              остаётся компактной кнопкой в шапке (см. style.css). */}
          <button
            className="reg-btn reg-btn--ghost top-bar-logout-compact"
            onClick={handleLogout}
          >
            Выйти
          </button>
        </div>
      ) : (
        <button className="reg-btn" onClick={onRegister}>Регистрация</button>
      )}
    </div>
  );
}
