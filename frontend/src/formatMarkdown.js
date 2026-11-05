/*
 * Общий рендер простого Markdown для сообщений чата (главная и задачи).
 *
 * Основная причина, по которой он вынесен отдельно и переписан: раньше
 * каждая пустая строка превращалась в <br>, из-за чего в карточке
 * подсказки появлялись огромные пустые промежутки, а жирный текст
 * получал жёсткий color:#333 и пропадал на тёмной теме.
 */
export function escapeHtml(str) {
  return String(str ?? "")
    .replace(/[\u2012\u2013\u2014\u2015]/g, "-")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatInline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

export default function formatMarkdown(text) {
  const normalized = String(text ?? "");
  const lines = escapeHtml(normalized).split(/\r?\n/);
  const blocks = [];
  let paragraph = [];
  let listItems = [];

  const flushParagraph = () => {
    if (!paragraph.length) return;
    blocks.push(`<p>${formatInline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };

  const flushList = () => {
    if (!listItems.length) return;
    const items = listItems
      .map((item) => `<li>${formatInline(item)}</li>`)
      .join("");
    blocks.push(`<ul>${items}</ul>`);
    listItems = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();

    if (!line) {
      flushParagraph();
      flushList();
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      blocks.push(
        `<div class="msg-heading msg-heading--${level}">${formatInline(
          heading[2],
        )}</div>`,
      );
      continue;
    }

    const bullet = line.match(/^[-•]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      listItems.push(bullet[1]);
      continue;
    }

    flushList();
    paragraph.push(line);
  }

  flushParagraph();
  flushList();

  return blocks.join("");
}
