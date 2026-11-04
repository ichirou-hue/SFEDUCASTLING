import { describe, expect, it } from "vitest";
import {
  cleanApiText,
  formatApiError,
  formatValidationError,
} from "./apiErrors.js";

describe("cleanApiText", () => {
  it("заменяет типографские тире и схлопывает пробелы", () => {
    expect(cleanApiText("  до — после  ")).toBe("до - после");
    expect(cleanApiText(null)).toBe("");
  });
});

describe("formatValidationError", () => {
  it("добавляет русскую подпись поля", () => {
    expect(
      formatValidationError({
        loc: ["body", "login"],
        type: "string_too_short",
        ctx: { min_length: 8 },
      }),
    ).toBe("Логин: минимум 8 символов");
  });

  it("склоняет слово символ", () => {
    const make = (n) =>
      formatValidationError({
        loc: ["body", "password"],
        type: "string_too_short",
        ctx: { min_length: n },
      });

    expect(make(1)).toContain("1 символ");
    expect(make(2)).toContain("2 символа");
    expect(make(5)).toContain("5 символов");
  });

  it("переводит обязательное поле и неизвестный английский текст", () => {
    expect(
      formatValidationError({ loc: ["body", "login"], type: "missing" }),
    ).toBe("Логин: обязательное поле");
    expect(
      formatValidationError({
        loc: ["body", "login"],
        type: "unknown",
        msg: "String should match pattern",
      }),
    ).toBe("Логин: некорректное значение");
  });

  it("снимает служебный префикс value_error и чистит тире", () => {
    expect(
      formatValidationError({
        loc: ["body", "password"],
        type: "value_error",
        msg: "Value error, Пароль — слишком простой",
      }),
    ).toBe("Пароль - слишком простой");
  });

  it.each([
    ["string_type", {}, "Логин: некорректное значение"],
    ["string_too_long", { max_length: 32 }, "Логин: максимум 32 символа"],
    ["string_pattern_mismatch", {}, "Логин: недопустимые символы"],
    ["int_type", {}, "Логин: должно быть целым числом"],
    ["int_parsing", {}, "Логин: должно быть целым числом"],
    ["greater_than_equal", { ge: 0 }, "Логин: минимум 0"],
    ["less_than_equal", { le: 3000 }, "Логин: максимум 3000"],
    ["greater_than", { gt: 5 }, "Логин: больше 5"],
    ["less_than", { lt: 10 }, "Логин: меньше 10"],
    ["bool_type", {}, "Логин: должно быть «да» или «нет»"],
    ["list_type", {}, "Логин: нужен список значений"],
    ["json_invalid", {}, "Некорректные данные запроса"],
  ])("переводит тип %s", (type, ctx, expected) => {
    expect(
      formatValidationError({ loc: ["body", "login"], type, ctx }),
    ).toBe(expected);
  });

  it("пропускает готовый русский текст и чистит строковые ошибки", () => {
    expect(
      formatValidationError({
        loc: ["body", "login"],
        type: "custom",
        msg: "Своя ошибка",
      }),
    ).toBe("Своя ошибка");
    expect(formatValidationError("  Текст — ошибки ")).toBe("Текст - ошибки");
    expect(formatValidationError(null)).toBe("");
  });
});

describe("formatApiError", () => {
  it("склеивает список ошибок валидации", () => {
    expect(
      formatApiError([
        { loc: ["body", "login"], type: "missing" },
        { loc: ["body", "password"], type: "missing" },
      ]),
    ).toBe("Логин: обязательное поле; Пароль: обязательное поле");
  });

  it("возвращает очищенную строку или запасной текст", () => {
    expect(formatApiError("  Сервер — недоступен ")).toBe("Сервер - недоступен");
    expect(formatApiError(null, "Запасной текст")).toBe("Запасной текст");
    expect(formatApiError([], "Запасной текст")).toBe("Запасной текст");
  });
});
