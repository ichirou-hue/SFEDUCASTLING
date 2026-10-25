/*
 * Свёрнутый чат: узкая стеклянная полоса во всю высоту колонки.
 * Вся полоса - одна кнопка, поэтому развернуть панель можно кликом
 * в любом её месте (стили .chat-strip в style.css). Используется
 * и на главной странице, и на странице шахматных задач.
 */
export default function ChatCollapsedStrip({ onExpand }) {
  return (
    <button
      type="button"
      className="chat-strip"
      onClick={onExpand}
      title="Развернуть чат"
      aria-label="Развернуть чат"
      aria-expanded={false}
    >
      <span className="chat-strip-avatar">
        <img src="/bot-icon.svg" alt="" />
      </span>
      <span className="chat-strip-chevron" aria-hidden="true">
        «
      </span>
      <span className="chat-strip-title">Шахматный ассистент</span>
    </button>
  );
}
