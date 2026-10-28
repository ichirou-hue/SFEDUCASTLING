/*
 * Тема оформления сайта.
 *
 * Основная тема — тёмная. Выбор хранится в localStorage и применяется
 * атрибутом data-theme на <html>; сами цвета живут в theme.css.
 */
export const THEME_STORAGE_KEY = "sfedu_theme";
export const DEFAULT_THEME = "dark";

export function getStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "dark" || stored === "light") return stored;
  } catch {
    // Приватный режим или запрет доступа к хранилищу: остаёмся на тёмной.
  }
  return DEFAULT_THEME;
}

export function applyTheme(theme) {
  const next = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  return next;
}

export function setStoredTheme(theme) {
  const next = applyTheme(theme);
  try {
    localStorage.setItem(THEME_STORAGE_KEY, next);
  } catch {
    // Без сохранения тема всё равно применится до перезагрузки.
  }
  return next;
}
