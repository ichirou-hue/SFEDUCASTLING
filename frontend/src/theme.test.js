import { describe, expect, it } from "vitest";
import {
  applyTheme,
  DEFAULT_THEME,
  getStoredTheme,
  setStoredTheme,
  THEME_STORAGE_KEY,
} from "./theme.js";

describe("theme", () => {
  it("по умолчанию использует тёмную тему", () => {
    expect(getStoredTheme()).toBe(DEFAULT_THEME);
  });

  it("читает сохранённую тему и игнорирует неизвестное значение", () => {
    localStorage.setItem(THEME_STORAGE_KEY, "light");
    expect(getStoredTheme()).toBe("light");

    localStorage.setItem(THEME_STORAGE_KEY, "neon");
    expect(getStoredTheme()).toBe("dark");
  });

  it("применяет тему к документу", () => {
    expect(applyTheme("light")).toBe("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");

    expect(applyTheme("unknown")).toBe("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
  });

  it("сохраняет выбранную тему", () => {
    expect(setStoredTheme("light")).toBe("light");
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
  });
});
