/*
 * Перевод ошибок API на русский для форм (регистрация, вход, анкета).
 *
 * FastAPI/Pydantic отдаёт 422 в виде {"detail": [{loc, msg, type, ctx}, ...]},
 * и стандартные проверки (длина, диапазон) приходят с английским текстом:
 * "String should have at least 8 characters". Здесь они собираются в
 * понятные русские фразы: берём тип ошибки и контекст (а не разбираем
 * английскую строку) и добавляем подпись поля.
 *
 * Длинные тире (— и –) из готовых сообщений вычищаем: в интерфейсе они
 * не нужны, вместо них обычный дефис.
 */

const FIELD_LABELS = {
  login: "Логин",
  password: "Пароль",
  email: "Email",
  elo: "Рейтинг",
  refresh_token: "Токен",
  q1: "Вопрос 1",
  q2: "Вопрос 2",
  q3: "Вопрос 3",
  q4: "Вопрос 4",
  q5: "Вопрос 5",
  q6: "Вопрос 6",
  q7: "Вопрос 7",
  q8: "Вопрос 8",
  parental_consent: "Согласие родителя",
};

/* Русское склонение: 1 символ, 2 символа, 5 символов. */
function plural(n, one, few, many) {
  const abs = Math.abs(Number(n) || 0);
  const last = abs % 10;
  const lastTwo = abs % 100;

  if (last === 1 && lastTwo !== 11) return one;
  if (last >= 2 && last <= 4 && (lastTwo < 10 || lastTwo >= 20)) return few;
  return many;
}

function symbols(n) {
  const count = Number(n) || 0;
  return `${count} ${plural(count, "символ", "символа", "символов")}`;
}

/* "body" / "query" / "path" в loc не показываем: это не имя поля. */
function fieldLabel(loc) {
  const parts = Array.isArray(loc)
    ? loc.filter((p) => p !== "body" && p !== "query" && p !== "path")
    : [];

  if (!parts.length) return "";

  const key = String(parts[0]);
  return FIELD_LABELS[key] || key;
}

/* Общая чистка текста: длинные тире, лишние пробелы. */
export function cleanApiText(text) {
  return String(text || "")
    .replace(/[—–]/g, "-")
    .replace(/\s+/g, " ")
    .trim();
}

export function formatValidationError(err) {
  if (!err) return "";
  if (typeof err === "string") return cleanApiText(err);

  const field = fieldLabel(err.loc);
  const prefix = field ? `${field}: ` : "";
  const ctx = err.ctx || {};
  const type = String(err.type || "");

  switch (type) {
    case "missing":
      return `${prefix}обязательное поле`;
    case "string_type":
      return `${prefix}некорректное значение`;
    case "string_too_short":
      return `${prefix}минимум ${symbols(ctx.min_length)}`;
    case "string_too_long":
      return `${prefix}максимум ${symbols(ctx.max_length)}`;
    case "string_pattern_mismatch":
      return `${prefix}недопустимые символы`;
    case "int_type":
    case "int_parsing":
      return `${prefix}должно быть целым числом`;
    case "greater_than_equal":
      return `${prefix}минимум ${ctx.ge}`;
    case "less_than_equal":
      return `${prefix}максимум ${ctx.le}`;
    case "greater_than":
      return `${prefix}больше ${ctx.gt}`;
    case "less_than":
      return `${prefix}меньше ${ctx.lt}`;
    case "bool_type":
      return `${prefix}должно быть «да» или «нет»`;
    case "list_type":
      return `${prefix}нужен список значений`;
    case "json_invalid":
      return "Некорректные данные запроса";
    case "value_error": {
      /* Свои проверки бэкенда уже пишут по-русски: только снимаем
         служебный префикс Pydantic и чистим тире. */
      const text = cleanApiText(
        String(err.msg || "").replace(/^Value error,\s*/i, ""),
      );
      return text || `${prefix}некорректное значение`;
    }
    default: {
      const text = cleanApiText(err.msg);

      /* Неизвестное сообщение на английском в интерфейс не отдаём. */
      if (!text || /^[A-Za-z]/.test(text)) {
        return `${prefix}некорректное значение`;
      }
      return text;
    }
  }
}

/*
 * Приводит detail из ответа API к одной строке для плашки ошибки.
 * Поддерживает и список ошибок валидации, и обычную строку.
 */
export function formatApiError(detail, fallback = "Не удалось выполнить запрос. Попробуйте позже.") {
  if (Array.isArray(detail)) {
    const text = detail.map(formatValidationError).filter(Boolean).join("; ");
    return text || fallback;
  }

  if (typeof detail === "string" && detail.trim()) {
    return cleanApiText(detail);
  }

  return fallback;
}
