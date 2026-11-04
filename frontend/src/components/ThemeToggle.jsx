// Компактный переключатель темы для шапки: доступен и гостям, у которых
// нет поповера профиля. Иконка показывает действие: солнце - включить
// светлую тему, луна - вернуться к тёмной.
export default function ThemeToggle({ theme = "dark", onThemeChange, className = "" }) {
  const isLight = theme === "light";
  const nextTheme = isLight ? "dark" : "light";
  const label = isLight ? "Включить тёмную тему" : "Включить светлую тему";

  return (
    <button
      type="button"
      className={`theme-toggle${className ? ` ${className}` : ""}`}
      onClick={() => onThemeChange?.(nextTheme)}
      aria-label={label}
      aria-pressed={isLight}
      title={label}
    >
      <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
        {isLight ? (
          <path d="M20.5 14.2A8.5 8.5 0 0 1 9.8 3.5 8.5 8.5 0 1 0 20.5 14.2Z" />
        ) : (
          <>
            <circle cx="12" cy="12" r="4.2" />
            <path d="M12 2.6v2.2M12 19.2v2.2M4.4 4.4l1.6 1.6M18 18l1.6 1.6M2.6 12h2.2M19.2 12h2.2M4.4 19.6 6 18M18 6l1.6-1.6" />
          </>
        )}
      </svg>
    </button>
  );
}
